"""OOD offline Viterbi eval for the research2 decoder-light WM on T-Drive.

Mirrors eval_offline_viterbi.py, but evaluates the Porto-trained Run-2-XL WM
zero-shot on the T-Drive/Beijing graph and candidate cache.

Usage, from src/evals:
  python eval_ood_tdrive_offline_viterbi.py
  python eval_ood_tdrive_offline_viterbi.py --max-traj 25
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path[:0] = [str(SRC), str(SRC / "hmm_baseline")]
BASE = Path(os.environ.get("AE_REPO_ROOT", SRC.parent))

import pyarrow.parquet  # noqa: F401,E402  must load before torch on Windows
import baseline_hmm as hb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_research2 as ev  # noqa: E402
import eval_ood_tdrive as ood  # noqa: E402

BETA = 20.0


@torch.no_grad()
def wm_log_emit(rssm, heads, stage1, road_z, b):
    """Closed-loop WM road-head emissions for one trajectory batch with B=1."""
    from models.world_model2 import straight_through_sample

    valid = b["seq_mask"] > 0.5
    batch, length = valid.shape
    z1, _, _, _, _ = stage1.forward(b, road_z, apply_mask=False)
    cpad_all = (b["cand_mask"] < 0.5) | (b["cand_segment_id"] < 0)

    out = np.full((length, b["cand_segment_id"].shape[-1]), -np.inf)
    h, _ = rssm.initial(batch, ev.DEVICE)
    prev_seg = None
    for t in range(length):
        if t > 0:
            h = rssm.step(h, z_q, road_z[prev_seg.clamp(min=0)])
        z_q = straight_through_sample(rssm.posterior(h, z1[:, t])).view(batch, -1)
        cmask = ~cpad_all[:, t]
        logits = heads.road(h, z_q, road_z, b["cand_segment_id"][:, t],
                            b["cand_heading_deg"][:, t], b["cand_oneway"][:, t],
                            b["cand_highway_id"][:, t], cmask)
        slot = logits.argmax(-1)
        prev_seg = b["cand_segment_id"][:, t].gather(-1, slot.unsqueeze(-1)).squeeze(-1)
        lp = torch.log_softmax(logits.masked_fill(~cmask, -1e4), dim=-1)
        out[t] = lp[0].double().cpu().numpy()
    return out


def nk_log_emit(dperp, valid):
    logit = np.where(valid, -(dperp.astype(np.float64) ** 2) / (2 * hb.SIGMA ** 2), -np.inf)
    return hb.log_softmax_rows(logit)


def main():
    ap = argparse.ArgumentParser(description="Run-2-XL WM OOD offline Viterbi eval on T-Drive")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--max-traj", type=int, default=None)
    args = ap.parse_args()

    ckpts = ev._checkpoint_table()
    path = ckpts[args.ckpt]
    print(f"[eval] OOD offline Viterbi over decoder-light WM  train=porto  eval=tdrive  "
          f"ckpt={args.ckpt}  device={ev.DEVICE}  beta={BETA}")

    road_z = ood._road_z_tdrive()
    stage1 = ev._stage1(road_z.size(1))
    rssm, heads = ev._load_wm(path, road_z.size(1))

    loader = ood._loader_tdrive(args.max_traj)
    ds, collate = loader.dataset, loader.collate_fn

    rd = hb.RouteDist(hb.build_road_digraph("tdrive"))
    variants = ["wm-argmax", "wm-viterbi", "hybrid", "nk-viterbi"]
    tol = {v: [] for v in variants}
    n = len(ds) if args.max_traj is None else min(args.max_traj, len(ds))
    t0 = time.time()

    for i in range(n):
        b_cpu = collate([ds[i]])
        b = {k: v.to(ev.DEVICE) for k, v in b_cpu.items()}

        seg = b_cpu["cand_segment_id"][0].numpy()
        dperp = b_cpu["cand_d_perp_m"][0].numpy()
        cm = b_cpu["cand_mask"][0].numpy()
        lat = b_cpu["lat"][0].numpy()
        lon = b_cpu["lon"][0].numpy()
        dt = b_cpu["dt"][0].numpy()
        valid = (cm > 0.5) & (seg >= 0)
        valid_fix = valid.any(1)

        le_wm = wm_log_emit(rssm, heads, stage1, road_z, b)
        le_wm[~valid] = -np.inf
        le_nk = nk_log_emit(dperp, valid)
        emits = {"wm-viterbi": le_wm, "hybrid": le_wm + le_nk, "nk-viterbi": le_nk}

        for t in np.where(valid_fix)[0]:
            s = int(np.argmax(le_wm[t]))
            best = float(dperp[t][valid[t]].min())
            tol["wm-argmax"].append(float(dperp[t][s]) <= best + 5.0)

        for piece in hb.split_pieces(valid_fix, dt):
            idx = np.asarray(piece)
            coords = np.stack([lat[idx], lon[idx]], axis=1)
            for v in ("wm-viterbi", "hybrid", "nk-viterbi"):
                path_slots = hb.viterbi(emits[v][idx], seg[idx], valid[idx], coords, rd, BETA)
                for pos, t in enumerate(idx):
                    s = path_slots[pos]
                    best = float(dperp[t][valid[t]].min())
                    tol[v].append(float(dperp[t][s]) <= best + 5.0)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{n}]  {elapsed:.0f}s  " +
                  "  ".join(f"{v}={np.mean(tol[v]):.4f}" for v in variants))

    print()
    print(f"RESULTS  train=porto  eval=tdrive  ckpt={args.ckpt}  N={n} trajs  "
          f"beta={BETA}  tolerant Hit@1")
    print("-" * 72)
    for v in variants:
        print(f"  {v:12s}  {np.mean(tol[v]):.4f}   ({len(tol[v])} fixes)")
    print()
    print("OOD anchor: T-Drive NK-HMM offline Viterbi in project summary = 0.7955")


if __name__ == "__main__":
    main()
