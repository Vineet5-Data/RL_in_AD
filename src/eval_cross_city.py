"""General cross-city generalization eval: any stage2r/stage2 checkpoint against any
prepped city, with a BUILT-IN training-membership guard.

Why this exists: eval_ood_tdrive.py hardcodes city=tdrive and silently assumes the
checkpoint never trained on it -- true for single-city Porto ckpts, false for any
train_multi ckpt that included tdrive. Evaluating the 2026-07-26 multi-city ckpt
against eval_ood_tdrive.py produced a T-Drive match@1 uplift that looked like "OOD
gap closed" but wasn't: T-Drive was one of the 5 training cities (18.3% weight), so
it isn't OOD for that checkpoint at all (project_summary.md, session-1 eval entry).
This script reads the checkpoint's own `meta["cities"]` (written by train_multi,
stage2r.py:512) and REFUSES to call a result "OOD" if the target city was in the
training mix -- prints a loud warning and the training weight instead of a clean
verdict, so this mistake can't recur silently.

True cross-city generalization currently has no genuinely held-out target among
this project's 5 prepped cities (porto/tdrive/geolife_car/cabspotting/romataxi are
ALL in the current multi-city mix) -- this script is the harness for when one
becomes available (a 6th city, or a future leave-one-city-out training run).

Usage (from .worktrees/research2):
  python eval_cross_city.py --ckpt <path> --city tdrive [--source tdrive] [--max-traj 500]
  python eval_cross_city.py --ckpt ckpt/stage2r_porto_p2_32x32.pt --city cabspotting
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet  # noqa: F401  must load before torch on Windows
import torch

import eval_research2 as ev

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
PROC_ROOT = BASE / "data" / "processed"
OSM_ROOT = BASE / "data" / "osm"
BASE_CKPT_DIR = BASE / ".worktrees" / "kaggle" / "ckpt"
S0_CKPT = BASE_CKPT_DIR / "stage0_porto.pt"
CACHE_DIR = BASE_CKPT_DIR / "cache_test"


def _road_z_city(city: str):
    """Frozen Porto-trained Stage-0 RoadGAT applied to an arbitrary city's road
    graph -- the encoder itself is city-agnostic (GAT over whatever graph you pass),
    only the graph and the resulting embeddings are city-specific. Always zero-shot
    for the road encoder regardless of whether the WM (rssm/heads) trained on this
    city -- matches the convention `eval_ood_tdrive._road_z_tdrive` already used."""
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT

    data = load_pyg(str(OSM_ROOT), city).to(ev.DEVICE)
    enc = RoadGAT(num_cont=data.x.size(1),
                  num_highway=max(64, int(data.highway_id.max()) + 1)).to(ev.DEVICE)
    enc.load_state_dict(torch.load(str(S0_CKPT), map_location=ev.DEVICE, weights_only=True))
    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.highway_id, data.edge_index)
    print(f"[road_z]  {tuple(z.shape)}  city={city}  zero-shot Stage-0=porto")
    return z.detach()


def _loader_city(city: str, source: str, max_traj: int | None):
    from dataset.trajectories import load_source_df, TrajectoryGraphDataset, collate_fn
    from dataset.candidates import CandidateIndex
    from dataset.config import SequenceConfig, RetrievalConfig
    from torch.utils.data import DataLoader

    n = 500 if max_traj is None else max_traj
    df = load_source_df(str(PROC_ROOT), source, limit_trajs=n)
    ci = CandidateIndex.from_city(str(OSM_ROOT), city)
    retr = RetrievalConfig()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{source}_n{n}_r{retr.radius_m:g}_k{retr.k}.npz"
    ds = TrajectoryGraphDataset(df, {source: ci}, SequenceConfig(), retr,
                                cache_path=str(cache_path), perm_seed=0)
    print(f"[data]  {len(ds)} {source} trajs  {len(df):,} fixes  max_traj={max_traj or 500}")
    return DataLoader(ds, batch_size=ev.BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, num_workers=0)


def _training_membership(ckpt_path: Path, city: str, source: str):
    """Read the checkpoint's own meta -- ground truth for what it trained on,
    not an assumption. Returns (is_member, weight_or_None, all_cities_list)."""
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    meta = sd.get("meta", {})
    cities = meta.get("cities")
    if not cities:
        return None, None, None  # single-city ckpt: no "cities" meta at all
    key = f"{city}:{source}"
    is_member = key in cities
    weight = None
    if is_member:
        weights = meta.get("city_weights")
        if weights:
            weight = weights[cities.index(key)]
    return is_member, weight, cities


def main():
    ap = argparse.ArgumentParser(description="Cross-city generalization eval with a training-membership guard")
    ap.add_argument("--ckpt", required=True, help="path to a stage2r checkpoint (.pt)")
    ap.add_argument("--city", required=True, help="target city's OSM road graph (e.g. tdrive, cabspotting, romataxi)")
    ap.add_argument("--source", default=None, help="GPS source under --city (default: same as --city)")
    ap.add_argument("--max-traj", type=int, default=None, help="optional smoke subset (default 500)")
    args = ap.parse_args()
    ckpt_path = Path(args.ckpt)
    source = args.source or args.city

    is_member, weight, cities = _training_membership(ckpt_path, args.city, source)
    print(f"[eval] cross-city generalization  ckpt={ckpt_path}  target={args.city}:{source}")
    if is_member is None:
        print("[guard] single-city checkpoint (no `cities` meta) -- treating as a genuine zero-shot OOD test.\n")
        ood_valid = True
    elif is_member:
        print(f"[guard] *** NOT A VALID OOD TEST *** -- {args.city}:{source} IS in this checkpoint's training "
              f"mix (weight={weight:.3f} of steps). Trained cities: {cities}. "
              f"Any result below reflects in-domain-but-underweighted performance, not generalization to an "
              f"unseen city -- do not report it as closing an OOD gap.\n")
        ood_valid = False
    else:
        print(f"[guard] confirmed: {args.city}:{source} is NOT in this checkpoint's training mix "
              f"(trained on: {cities}). Genuine zero-shot cross-city test.\n")
        ood_valid = True

    road_z = _road_z_city(args.city)
    stage1 = ev._stage1(road_z.size(1))
    loader = _loader_city(args.city, source, args.max_traj)
    rssm, heads = ev._load_wm(ckpt_path, road_z.size(1))
    r = ev.eval_checkpoint(rssm, heads, heads.road, stage1, road_z, loader)

    print(f"match@1 {r['match_hit1']:.4f}  cl_gps {r['cl_gps']:.4f}  kl {r['kl']:.3f}  ent {r['ent']:.2f}  "
          f"hit@1 {r['hit1']:.3f}  hit@5 {r['hit5']:.3f}  hit@15 {r['hit15']:.3f}")
    print(f"\nOOD-VALID: {ood_valid}")


if __name__ == "__main__":
    main()
