"""Diagnostic: does the online "hyb+topo" disconnected-jump rate (5.9%-8.7pp
family of numbers recorded across the 16x16 -> 32x32 ckpts) concentrate on
steps where the PREVIOUS step was already wrong, or is it roughly uniform
regardless of prior correctness?

Motivation: proposed fix for the online jump rate is scheduled sampling /
DAgger-style training (feed the model's own prior choice instead of the
pos_idx ground truth as previous-state conditioning some fraction of the
time), on the theory that training-time teacher-forcing on ground truth
never shows the model a "recovering from a wrong turn" state, so once
inference makes one error, the next choice is drawn from a distribution the
model never trained on -> disconnected jump. That's a training/inference
CONDITIONING mismatch story.

The competing story: jumps are driven by genuinely ambiguous/sparse GPS
geometry (a fork, a gap) independent of whether the PREVIOUS pick happened
to match the nearest-candidate pseudo-label. If jump rate is roughly the
same whether the prior step was a "hit" or a "miss", the conditioning-
mismatch story is not supported and scheduled sampling would not be
expected to move the metric -- look elsewhere (the already-tried topo-hard
hard-connectivity mask, e.g., which trades -1.5pp match@1 for -0.57pp fewer
jumps and was rejected as not worth it, per eval_online_topo.txt).

Reuses eval_online_topo.py's exact "hyb+topo" math (WM logp + NK emission +
NK route-distance transition), single variant only, against the PROMOTED
P2 32x32 "final" checkpoint (the actual deployed matcher.py model), same
held-out 500-traj split as eval_research2.py. CPU by default -- read-only,
no training.

Run (from .worktrees/research2):  python diag_jump_vs_priorerror.py [--max-traj 100]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
HMM = BASE / ".worktrees" / "HMM_baseline" / "hmm_baseline"
sys.path.insert(0, str(HMM))

import pyarrow.parquet  # noqa: F401,E402  must load before torch on Windows
import baseline_hmm as hb  # noqa: E402

HERE = Path(__file__).resolve().parent
DP = BASE / ".worktrees" / "data-preprocess"
sys.path = [str(HERE), str(DP), *[p for p in sys.path if p not in {str(HERE), str(DP)}]]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_research2 as ev  # noqa: E402

BETA = 20.0


def nk_log_emit(dperp, valid):
    logit = np.where(valid, -(dperp.astype(np.float64) ** 2) / (2 * hb.SIGMA ** 2), -np.inf)
    return hb.log_softmax_rows(logit)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--ckpt", default="final",
                    help="key into ev._checkpoint_table(), or a direct .pt path, "
                         "default = promoted P2 32x32")
    args = ap.parse_args()

    from pathlib import Path as _Path
    from models.world_model2 import straight_through_sample

    ckpt_path = _Path(args.ckpt) if str(args.ckpt).endswith(".pt") else ev._checkpoint_table()[args.ckpt]
    print(f"[diag] jump-vs-prior-error  city=porto  ckpt={ckpt_path}  device={ev.DEVICE}  beta={BETA}")
    road_z = ev._road_z()
    stage1 = ev._stage1(road_z.size(1))
    rssm, heads = ev._load_wm(ckpt_path, road_z.size(1))
    rd = hb.RouteDist(hb.build_road_digraph("porto"))

    loader = ev._loader()
    ds, collate = loader.dataset, loader.collate_fn
    n = len(ds) if args.max_traj is None else min(args.max_traj, len(ds))

    # 2x2 table: jump x prior-hit, plus overall hit-rate for a sanity cross-check.
    jump_given_prev_hit = []
    jump_given_prev_miss = []
    jumps_all = []   # unconditional -- directly comparable to the historical jump% numbers
    hits_all = []
    t0 = time.time()

    for i in range(n):
        b_cpu = collate([ds[i]])
        b = {k: v.to(ev.DEVICE) for k, v in b_cpu.items()}
        seg = b_cpu["cand_segment_id"][0].numpy()
        dperp = b_cpu["cand_d_perp_m"][0].numpy()
        valid = (b_cpu["cand_mask"][0].numpy() > 0.5) & (seg >= 0)
        lat = b_cpu["lat"][0].numpy()
        lon = b_cpu["lon"][0].numpy()
        dt = b_cpu["dt"][0].numpy()
        L = seg.shape[0]
        pos = dt[dt > 0]
        med = float(np.median(pos)) if pos.size else 0.0

        z1, _, _, _, _ = stage1.forward(b, road_z, apply_mask=False)
        cpad = (b["cand_mask"] < 0.5) | (b["cand_segment_id"] < 0)
        le_nk = nk_log_emit(dperp, valid)
        d1 = np.where(valid.any(1), np.where(valid, dperp, np.inf).min(1), np.inf)

        h, _ = rssm.initial(1, ev.DEVICE)
        zq = None
        prev = None       # topo anchor (None after gap/no-candidate)
        drive = None
        prev_hit = None   # was the step that set `prev` a tolerant-hit?

        for t in range(L):
            gap = t == 0 or (med > 0 and dt[t] > 3 * med)
            d_gc = hb.haversine_m(lat[t - 1], lon[t - 1], lat[t], lon[t]) if t > 0 else 0.0
            vmask = valid[t]
            cmask = ~cpad[:, t]

            if t > 0:
                h = rssm.step(h, zq, road_z[drive.clamp(min=0)])
            zq = straight_through_sample(rssm.posterior(h, z1[:, t])).view(1, -1)
            logits = heads.road(h, zq, road_z, b["cand_segment_id"][:, t],
                                b["cand_heading_deg"][:, t], b["cand_oneway"][:, t],
                                b["cand_highway_id"][:, t], cmask)
            lp = torch.log_softmax(logits.masked_fill(~cmask, -1e4), dim=-1)
            score = lp[0].double().cpu().numpy().copy()
            score[~vmask] = -np.inf

            use_topo = prev is not None and not gap
            score = score + le_nk[t]
            if use_topo:
                T = hb.log_transition(np.array([prev]), np.array([True]),
                                      seg[t], vmask, d_gc, rd, BETA)[0]
                score = score + T

            drive_slot = int(logits.argmax(-1).item())
            if vmask.any():
                slot = int(np.argmax(score))
                hit = bool(dperp[t, slot] <= d1[t] + 5.0)
                hits_all.append(hit)
                if use_topo:
                    cut = 2.0 * d_gc + 500.0
                    jump = rd.dist(prev, int(seg[t, slot]), cut) is None
                    jumps_all.append(jump)
                    if prev_hit is True:
                        jump_given_prev_hit.append(jump)
                    elif prev_hit is False:
                        jump_given_prev_miss.append(jump)
                prev = int(seg[t, slot])
                prev_hit = hit
                drive_slot = slot
            else:
                prev = None
                prev_hit = None
            drive = b["cand_segment_id"][:, t].gather(
                -1, torch.tensor([[drive_slot]], device=ev.DEVICE)).squeeze(-1)

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            nhit = np.mean(jump_given_prev_hit) if jump_given_prev_hit else float("nan")
            nmiss = np.mean(jump_given_prev_miss) if jump_given_prev_miss else float("nan")
            print(f"  [{i + 1}/{n}]  {el:.0f}s  jump|hit={nhit:.4f} ({len(jump_given_prev_hit)})"
                  f"  jump|miss={nmiss:.4f} ({len(jump_given_prev_miss)})")

    n_hit, n_miss = len(jump_given_prev_hit), len(jump_given_prev_miss)
    p_hit = float(np.mean(jump_given_prev_hit)) if n_hit else float("nan")
    p_miss = float(np.mean(jump_given_prev_miss)) if n_miss else float("nan")
    overall_hit_rate = float(np.mean(hits_all))
    overall_jump_rate = float(np.mean(jumps_all)) if jumps_all else float("nan")

    print()
    print(f"RESULTS  N={n} trajs  ({len(hits_all)} scored fixes)")
    print(f"  match@1 (overall tolerant-hit rate) = {overall_hit_rate:.4f}")
    print(f"  jump%   (overall, unconditional)    = {100.0 * overall_jump_rate:.2f}%   "
          f"n={len(jumps_all)}  <- directly comparable to the historical hyb+topo jump% numbers")
    print(f"  P(jump | prev step was a hit)  = {p_hit:.4f}   n={n_hit}")
    print(f"  P(jump | prev step was a miss) = {p_miss:.4f}   n={n_miss}")
    if n_hit and n_miss:
        ratio = p_miss / p_hit if p_hit > 0 else float("inf")
        # pooled two-proportion z-test
        k_hit = p_hit * n_hit
        k_miss = p_miss * n_miss
        p_pool = (k_hit + k_miss) / (n_hit + n_miss)
        se = (p_pool * (1 - p_pool) * (1 / n_hit + 1 / n_miss)) ** 0.5
        z = (p_miss - p_hit) / se if se > 0 else float("nan")
        print(f"  ratio (miss/hit) = {ratio:.2f}x   two-proportion z = {z:.2f}")
        print()
        supported = ratio >= 1.5 and z >= 3.0
        print("Verdict: " + (
            "SUPPORTED -- jump rate is substantially and significantly higher right after "
            "a miss, consistent with the training/inference conditioning-mismatch story. "
            "Scheduled sampling is a reasonable next experiment (with its own written bar)."
            if supported else
            "NOT SUPPORTED -- jump rate after a miss is not meaningfully/significantly higher "
            "than after a hit. Jumps look driven by local geometry/topology at the CURRENT "
            "step, not by carrying a wrong previous state. Scheduled sampling would not be "
            "expected to move this metric much -- do not build it on this basis alone."
        ))
    else:
        print("  not enough samples in one bucket to compare")


if __name__ == "__main__":
    main()
