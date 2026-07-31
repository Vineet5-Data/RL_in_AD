"""Build candidate cache for T-Drive trajectories (500 held-out slice).

Runs AFTER build_tdrive_osm.py. Finds road candidates within 50m for each
GPS fix and saves to ckpt/cache_test/tdrive_n500_r50_k10.npz.

Usage:
  python build_tdrive_cache.py
"""

import os, sys, time
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
DP   = BASE / ".worktrees" / "data-preprocess"
sys.path.insert(0, str(DP))
sys.path.insert(0, str(BASE / ".worktrees" / "kaggle"))

import pyarrow.parquet  # noqa

PROC_ROOT  = BASE / "data" / "processed"
OSM_ROOT   = BASE / "data" / "osm"
CKPT_DIR   = BASE / ".worktrees" / "kaggle" / "ckpt"
CACHE_OUT  = CKPT_DIR / "cache_test" / "tdrive_n500_r50_k10.npz"
N_TRAJS    = 500

from dataset.trajectories import load_source_df, TrajectoryGraphDataset, collate_fn
from dataset.candidates   import CandidateIndex
from dataset.config       import SequenceConfig, RetrievalConfig

t0 = time.time()
df = load_source_df(str(PROC_ROOT), "tdrive", limit_trajs=N_TRAJS)
ci = CandidateIndex.from_city(str(OSM_ROOT), "tdrive")
print(f"[cache]  {N_TRAJS} trajs  {len(df)} fixes  building candidates...")

ds = TrajectoryGraphDataset(
    df, {"tdrive": ci},
    SequenceConfig(), RetrievalConfig(),
    cache_path=str(CACHE_OUT),
)
dt = time.time() - t0
print(f"[cache]  done  {len(ds)} trajectories  elapsed={dt:.1f}s")
print(f"Saved: {CACHE_OUT}")
