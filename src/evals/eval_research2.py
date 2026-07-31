"""research2 eval: decoder-light WM + actor-critic vs the existing baselines.

Same protocol as the Run-3 harness (eval_stage2.py) so numbers are directly
comparable with the HMM Track-B results and the Run-3 RSSM table:
  - identical split: Porto held-out trajs [200000, 200500) (HMM baseline split),
    same candidate cache (porto_holdout_n500_r50_k10.npz), perm_seed=0
  - identical conventions: tolerant Hit@1 (pred d_perp <= d1 + 5m), T_WARM=8,
    Change-5 self-driven open-loop imagination over REAL candidate sets

Metrics:
  match_hit1   ONLINE map matching: at every fix, posterior z from real obs,
               decoder (actor if --actor-ckpt else the WM road head) scores the
               real candidate set, argmax, tolerant Hit@1. Transition conditioned
               on the decoder's OWN previous choice (deployment-honest -- no
               pseudo-label conditioning). NOTE: HMM Viterbi (0.8447) is an
               OFFLINE smoother with future context; this is an online filter --
               comparable but not identical problem difficulty.
  hit@{1,5,15} prior-rollout road tolerant Hit@1 at horizon N (path prediction;
               compare HMM beta=20 transition-only + Run-3 RSSM gate-C4 numbers).
  ol/cv@{1,4,8,16}  open-loop GPS error (km) vs constant-velocity. GPS decode is
               the z-ONLY light head -- cl_gps is REDEFINED vs Runs 1-3 (flagged
               design decision, methodology-comparison Sec.5 Rank-1 caveat 2).
  kl, ent      posterior-collapse monitors (the redesign's primary claim).

Usage (from src/evals):
  python eval_research2.py                      # all stage2r ckpts, road-head decode
  python eval_research2.py --actor-ckpt ckpt/stage3_porto.pt   # actor decode
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
BASE = Path(os.environ.get("AE_REPO_ROOT", SRC.parent))

import pyarrow.parquet  # noqa: E402  must precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

DATA_ROOT = Path(os.environ.get("AE_DATA_ROOT", BASE / "data"))
CKPT_ROOT = Path(os.environ.get("AE_CKPT_ROOT", BASE / "ckpt"))
PROC_ROOT = DATA_ROOT / "processed"
OSM_ROOT = DATA_ROOT / "osm"
CKPT_DIR = CKPT_ROOT
BASE_CKPT_DIR = CKPT_ROOT
S0_CKPT = BASE_CKPT_DIR / "stage0_porto.pt"
S1_CKPT = BASE_CKPT_DIR / "stage1_porto.pt"
EVAL_CACHE = BASE_CKPT_DIR / "cache_test" / "porto_holdout_n500_r50_k10.npz"
PARQUET = PROC_ROOT / "porto" / "part-000.parquet"

SKIP_TRAJS = 200_000   # training used trajs [0, 200000) -- same held-out slice as HMM
N_TRAJS = 500
BATCH_SIZE = 16
T_WARM = 8
GPS_HORIZONS = [1, 4, 8, 16]
ROAD_HORIZONS = [1, 5, 15]
ALL_HORIZONS = sorted(set(GPS_HORIZONS) | set(ROAD_HORIZONS))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _checkpoint_table() -> dict[str, Path]:
    step_ckpts = sorted(CKPT_DIR.glob("stage2r_porto_step*.pt"),
                        key=lambda p: int(p.stem.rsplit("step", 1)[1]))
    ckpts = {p.stem.replace("stage2r_porto_", ""): p for p in step_ckpts}
    final = CKPT_DIR / "stage2r_porto.pt"
    if final.is_file():
        ckpts["final"] = final
    return ckpts


def _road_z():
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT
    data = load_pyg(str(OSM_ROOT), "porto").to(DEVICE)
    enc = RoadGAT(num_cont=data.x.size(1),
                  num_highway=max(64, int(data.highway_id.max()) + 1)).to(DEVICE)
    enc.load_state_dict(torch.load(str(S0_CKPT), map_location=DEVICE, weights_only=True))
    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.highway_id, data.edge_index)
    return z.detach()


def _stage1(road_dim):
    from models.gps_encoder import Stage1Encoder
    m = Stage1Encoder(road_dim=road_dim).to(DEVICE)
    m.load_state_dict(torch.load(str(S1_CKPT), map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def _loader():
    import pandas as pd
    from dataset.candidates import CandidateIndex
    from dataset.trajectories import TrajectoryGraphDataset, collate_fn
    from dataset.config import SequenceConfig, RetrievalConfig

    tbl = pyarrow.parquet.read_table(str(PARQUET), columns=["traj_id"])
    uniq = pd.Series(tbl["traj_id"].to_pandas().unique())
    held = set(uniq.iloc[SKIP_TRAJS: SKIP_TRAJS + N_TRAJS].tolist())
    del tbl
    tbl_full = pyarrow.parquet.read_table(str(PARQUET))
    df = tbl_full.to_pandas(); del tbl_full
    df = df[df["traj_id"].isin(held)].reset_index(drop=True)
    ci = CandidateIndex.from_city(str(OSM_ROOT), "porto")
    ds = TrajectoryGraphDataset(df, {"porto": ci},
                                SequenceConfig(), RetrievalConfig(),
                                cache_path=str(EVAL_CACHE), perm_seed=0)  # Change 7A
    print(f"[data]  {len(ds)} trajs  {len(df):,} fixes")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn, num_workers=0)


def _load_wm(path, road_dim):
    from training.stage3 import load_world_model
    return load_world_model(str(path), road_dim, DEVICE)


def _load_actor(path, road_dim):
    from models.world_model2 import RoadScorer
    sd = torch.load(str(path), map_location=DEVICE, weights_only=True)
    actor = RoadScorer(h_dim=512, road_dim=road_dim).to(DEVICE)
    actor.load_state_dict(sd["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def eval_checkpoint(rssm, heads, decoder, stage1, road_z, loader):
    """decoder: callable (h, z, road_z, seg, head, oneway, hw, mask) -> logits[B,K]
    (either the frozen WM road head -- 'No RL' ablation -- or the Stage-3 actor)."""
    from models.world_model2 import straight_through_sample, kl_categorical, _mixed_probs

    cl_gps, kl_vals, ent_vals, match_hits = [], [], [], []
    ol_errs = {h: [] for h in GPS_HORIZONS}
    cv_errs = {h: [] for h in GPS_HORIZONS}
    road_hit = {h: [] for h in ROAD_HORIZONS}

    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        valid = b["seq_mask"] > 0.5
        B, L = valid.shape
        if L < T_WARM + max(ALL_HORIZONS):
            continue

        z1, _, coords, _, _ = stage1.forward(b, road_z, apply_mask=False)

        cpad_all = (b["cand_mask"] < 0.5) | (b["cand_segment_id"] < 0)
        dp_all = b["cand_d_perp_m"].masked_fill(cpad_all, float("inf"))

        # -- closed-loop ONLINE matching: self-driven conditioning --
        h, _ = rssm.initial(B, DEVICE)
        prev_seg = None
        for t in range(L):
            if t > 0:
                h = rssm.step(h, z_q, road_z[prev_seg.clamp(min=0)])
            post = rssm.posterior(h, z1[:, t])
            prior = rssm.prior(h)
            z_q = straight_through_sample(post).view(B, -1)

            kl = kl_categorical(post, prior)
            kl_vals.extend(kl[valid[:, t]].cpu().tolist())
            pp = _mixed_probs(prior)
            ent = -(pp * pp.log()).sum(-1).sum(-1)
            ent_vals.extend(ent[valid[:, t]].cpu().tolist())

            gps_hat, _ = heads(h, z_q)
            err = ((gps_hat - coords[:, t]) ** 2).sum(-1).sqrt()
            cl_gps.extend(err[valid[:, t]].cpu().tolist())

            cmask = ~cpad_all[:, t]
            logits = decoder(h, z_q, road_z, b["cand_segment_id"][:, t],
                             b["cand_heading_deg"][:, t], b["cand_oneway"][:, t],
                             b["cand_highway_id"][:, t], cmask)
            slot = logits.argmax(-1)
            prev_seg = b["cand_segment_id"][:, t].gather(-1, slot.unsqueeze(-1)).squeeze(-1)

            hc = valid[:, t] & cmask.any(-1)
            if hc.any():
                d1 = dp_all[:, t].min(-1).values
                pred_dp = dp_all[:, t].gather(-1, slot.unsqueeze(-1)).squeeze(-1)
                match_hits.extend((pred_dp <= d1 + 5.0)[hc].cpu().tolist())

        # -- open-loop: warm-up (self-driven), then actor-driven prior rollout --
        h, _ = rssm.initial(B, DEVICE)
        prev_seg = None
        for t in range(T_WARM):
            if t > 0:
                h = rssm.step(h, z_q, road_z[prev_seg.clamp(min=0)])
            z_q = straight_through_sample(rssm.posterior(h, z1[:, t])).view(B, -1)
            cmask = ~cpad_all[:, t]
            logits = decoder(h, z_q, road_z, b["cand_segment_id"][:, t],
                             b["cand_heading_deg"][:, t], b["cand_oneway"][:, t],
                             b["cand_highway_id"][:, t], cmask)
            slot = logits.argmax(-1)
            prev_seg = b["cand_segment_id"][:, t].gather(-1, slot.unsqueeze(-1)).squeeze(-1)
        h = rssm.step(h, z_q, road_z[prev_seg.clamp(min=0)])  # into t = T_WARM

        vel = coords[:, T_WARM] - coords[:, T_WARM - 1]
        h_ol = h.clone()

        for horizon in ALL_HORIZONS:
            h_tmp = h_ol.clone()
            pred_slot = h_eval = z_eval = None
            for step_idx in range(horizon):
                cur_t = T_WARM + step_idx
                z_tmp = straight_through_sample(rssm.prior(h_tmp)).view(B, -1)
                cmask = ~cpad_all[:, cur_t]
                logits = decoder(h_tmp, z_tmp, road_z, b["cand_segment_id"][:, cur_t],
                                 b["cand_heading_deg"][:, cur_t], b["cand_oneway"][:, cur_t],
                                 b["cand_highway_id"][:, cur_t], cmask)
                pred_slot = logits.argmax(-1)
                pred_seg = b["cand_segment_id"][:, cur_t].gather(-1, pred_slot.unsqueeze(-1)).squeeze(-1)
                if step_idx == horizon - 1:
                    h_eval, z_eval = h_tmp, z_tmp
                    break
                h_tmp = rssm.step(h_tmp, z_tmp, road_z[pred_seg.clamp(min=0)])

            t_pred = T_WARM + horizon - 1
            if t_pred >= L:
                continue
            mask = valid[:, t_pred]
            if not mask.any():
                continue

            if horizon in GPS_HORIZONS:
                gps_hat, _ = heads(h_eval, z_eval)
                target = coords[:, t_pred]
                ol_errs[horizon].extend(((gps_hat - target) ** 2).sum(-1).sqrt()[mask].cpu().tolist())
                cv_pred = coords[:, T_WARM - 1] + vel * horizon
                cv_errs[horizon].extend(((cv_pred - target) ** 2).sum(-1).sqrt()[mask].cpu().tolist())

            if horizon in ROAD_HORIZONS:
                hc = mask & (~cpad_all[:, t_pred]).any(-1)
                if hc.any():
                    dp_t = dp_all[:, t_pred]
                    d1 = dp_t.min(-1).values
                    pred_dp = dp_t.gather(-1, pred_slot.unsqueeze(-1)).squeeze(-1)
                    road_hit[horizon].extend((pred_dp <= d1 + 5.0)[hc].cpu().tolist())

    def m(lst): return float(np.mean(lst)) if lst else float("nan")
    return {
        "match_hit1": m(match_hits), "cl_gps": m(cl_gps),
        "kl": m(kl_vals), "ent": m(ent_vals),
        **{f"ol{h}": m(ol_errs[h]) for h in GPS_HORIZONS},
        **{f"cv{h}": m(cv_errs[h]) for h in GPS_HORIZONS},
        **{f"hit{h}": m(road_hit[h]) for h in ROAD_HORIZONS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-ckpt", default=None,
                    help="stage3 checkpoint; if omitted, decode with the WM road head (No-RL ablation)")
    ap.add_argument("--only", default=None,
                    help="evaluate only these checkpoint labels, comma-separated (e.g. final or step17000,step18000)")
    args = ap.parse_args()

    ckpts = _checkpoint_table()
    if args.only:
        allow = set(args.only.split(","))
        ckpts = {k: v for k, v in ckpts.items() if k in allow}
    if not ckpts:
        print(f"no stage2r checkpoints found in {CKPT_DIR}"); return

    mode = "actor" if args.actor_ckpt else "road-head (No-RL ablation)"
    print(f"[eval] research2 decoder-light WM  device={DEVICE}  N={N_TRAJS}  "
          f"T_warm={T_WARM}  decode={mode}\n")
    road_z = _road_z()
    stage1 = _stage1(road_z.size(1))
    loader = _loader()
    actor = _load_actor(args.actor_ckpt, road_z.size(1)) if args.actor_ckpt else None

    hdr = (f"{'ckpt':12s}  {'match@1':>7s}  {'cl_gps':>7s}  {'kl':>5s}  {'ent':>5s}"
           f"  {'ol@1':>6s}  {'ol@4':>6s}  {'ol@8':>6s}  {'ol@16':>6s}"
           f"  {'cv@1':>6s}  {'cv@4':>6s}  {'cv@8':>6s}  {'cv@16':>6s}"
           f"  {'hit@1':>6s}  {'hit@5':>6s}  {'hit@15':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for label, path in ckpts.items():
        rssm, heads = _load_wm(path, road_z.size(1))
        decoder = actor if actor is not None else heads.road
        r = eval_checkpoint(rssm, heads, decoder, stage1, road_z, loader)
        beat = {h: "<cv" if r[f"ol{h}"] < r[f"cv{h}"] else ">cv" for h in GPS_HORIZONS}
        print(f"{label:12s}  {r['match_hit1']:>7.4f}  {r['cl_gps']:>7.4f}"
              f"  {r['kl']:>5.3f}  {r['ent']:>5.2f}"
              f"  {r['ol1']:>6.4f}  {r['ol4']:>6.4f}  {r['ol8']:>6.4f}  {r['ol16']:>6.4f}"
              f"  {r['cv1']:>6.4f}  {r['cv4']:>6.4f}  {r['cv8']:>6.4f}  {r['cv16']:>6.4f}"
              f"  {r['hit1']:>6.3f}  {r['hit5']:>6.3f}  {r['hit15']:>6.3f}"
              f"  {beat[1]}/{beat[4]}/{beat[8]}/{beat[16]}")

    print()
    print("KEY / comparison anchors:")
    print("  match@1   ONLINE matching tolerant Hit@1 (self-driven closed loop).")
    print("            HMM Track-B anchor: 0.8447 (OFFLINE Viterbi -- easier problem).")
    print("  hit@N     prior-rollout road tolerant Hit@1 (prediction). Run-3 RSSM best")
    print("            hit@1 0.648; HMM beta=20 transition-only loses to RSSM by 5.2-7.9pp")
    print("            at every horizon (hmm_results.md gate C4).")
    print("  cl_gps    z-ONLY GPS decode (km) -- REDEFINED vs Runs 1-3 (decoder-light),")
    print("            not directly comparable to the old cl_gps<=0.15km gate.")
    print("  kl/ent    collapse monitors: kl~0 = collapsed (the failure this redesign")
    print("            exists to fix); 0.4-0.9 healthy (16x16 latent).")


if __name__ == "__main__":
    main()
