"""OOD eval for the research2 decoder-light WM on T-Drive/Beijing.

This is intentionally a thin wrapper around eval_research2.py: it reuses the
same metric code, checkpoints, and decoder path, but swaps the road graph,
candidate cache, and trajectories from Porto to T-Drive.

Usage, from .worktrees/research2:
  python eval_ood_tdrive.py --only final
  python eval_ood_tdrive.py --only final --max-traj 25
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet  # noqa: F401  must load before torch on Windows
import torch
from torch.utils.data import DataLoader

import eval_research2 as ev

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
PROC_ROOT = BASE / "data" / "processed"
OSM_ROOT = BASE / "data" / "osm"
BASE_CKPT_DIR = BASE / ".worktrees" / "kaggle" / "ckpt"
S0_CKPT = BASE_CKPT_DIR / "stage0_porto.pt"
EVAL_CACHE = BASE_CKPT_DIR / "cache_test" / "tdrive_n500_r50_k10.npz"


def _road_z_tdrive():
    """Apply the Porto-trained road encoder to the Beijing road graph zero-shot."""
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT

    data = load_pyg(str(OSM_ROOT), "tdrive").to(ev.DEVICE)
    enc = RoadGAT(num_cont=data.x.size(1),
                  num_highway=max(64, int(data.highway_id.max()) + 1)).to(ev.DEVICE)
    enc.load_state_dict(torch.load(str(S0_CKPT), map_location=ev.DEVICE, weights_only=True))
    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.highway_id, data.edge_index)
    print(f"[road_z]  {tuple(z.shape)}  city=tdrive  zero-shot Stage-0=porto")
    return z.detach()


def _loader_tdrive(max_traj: int | None):
    from dataset.trajectories import load_source_df, TrajectoryGraphDataset, collate_fn
    from dataset.candidates import CandidateIndex
    from dataset.config import SequenceConfig, RetrievalConfig

    n = 500 if max_traj is None else max_traj
    df = load_source_df(str(PROC_ROOT), "tdrive", limit_trajs=n)
    ci = CandidateIndex.from_city(str(OSM_ROOT), "tdrive")
    ds = TrajectoryGraphDataset(df, {"tdrive": ci},
                                SequenceConfig(), RetrievalConfig(),
                                cache_path=str(EVAL_CACHE), perm_seed=0)
    print(f"[data]  {len(ds)} T-Drive trajs  {len(df):,} fixes  max_traj={max_traj or 500}")
    return DataLoader(ds, batch_size=ev.BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn, num_workers=0)


def main():
    ap = argparse.ArgumentParser(description="Run-2-XL decoder-light WM OOD eval on T-Drive")
    ap.add_argument("--only", default="final", help="checkpoint label from eval_research2 table")
    ap.add_argument("--max-traj", type=int, default=None, help="optional smoke subset")
    args = ap.parse_args()

    ckpts = ev._checkpoint_table()
    if args.only:
        ckpts = {k: v for k, v in ckpts.items() if k == args.only}
    if not ckpts:
        print(f"no matching stage2r checkpoints found for --only={args.only!r}")
        return

    print(f"[eval] research2 decoder-light WM OOD  train=porto  eval=tdrive  "
          f"device={ev.DEVICE}  T_warm={ev.T_WARM}  decode=road-head\n")
    road_z = _road_z_tdrive()
    stage1 = ev._stage1(road_z.size(1))
    loader = _loader_tdrive(args.max_traj)

    hdr = (f"{'ckpt':12s}  {'match@1':>7s}  {'cl_gps':>7s}  {'kl':>5s}  {'ent':>5s}"
           f"  {'ol@1':>6s}  {'ol@4':>6s}  {'ol@8':>6s}  {'ol@16':>6s}"
           f"  {'cv@1':>6s}  {'cv@4':>6s}  {'cv@8':>6s}  {'cv@16':>6s}"
           f"  {'hit@1':>6s}  {'hit@5':>6s}  {'hit@15':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for label, path in ckpts.items():
        rssm, heads = ev._load_wm(path, road_z.size(1))
        r = ev.eval_checkpoint(rssm, heads, heads.road, stage1, road_z, loader)
        beat = {h: "<cv" if r[f"ol{h}"] < r[f"cv{h}"] else ">cv" for h in ev.GPS_HORIZONS}
        print(f"{label:12s}  {r['match_hit1']:>7.4f}  {r['cl_gps']:>7.4f}"
              f"  {r['kl']:>5.3f}  {r['ent']:>5.2f}"
              f"  {r['ol1']:>6.4f}  {r['ol4']:>6.4f}  {r['ol8']:>6.4f}  {r['ol16']:>6.4f}"
              f"  {r['cv1']:>6.4f}  {r['cv4']:>6.4f}  {r['cv8']:>6.4f}  {r['cv16']:>6.4f}"
              f"  {r['hit1']:>6.3f}  {r['hit5']:>6.3f}  {r['hit15']:>6.3f}"
              f"  {beat[1]}/{beat[4]}/{beat[8]}/{beat[16]}")

    print()
    print("OOD anchors:")
    print("  T-Drive NK-HMM offline Viterbi anchor recorded in project summary: 0.7955.")
    print("  This script reports the online WM road-head filter and prior rollouts zero-shot.")


if __name__ == "__main__":
    main()
