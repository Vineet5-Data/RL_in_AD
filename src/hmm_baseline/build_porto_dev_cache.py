"""Build candidate cache for Porto dev slice (trajs 0-49, TRAINING split -- never
the held-out 200000:200500 range). Used for the beta sweep (classical_baseline_spec.md
Compute section: "sweep on a 50-trajectory dev slice carved from the TRAINING split").

Usage:
  cd ~/Desktop/AlphaEvolve_research/.worktrees/kaggle
  python build_porto_dev_cache.py
"""

import os, sys, time
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
DP   = BASE / ".worktrees" / "data-preprocess"
sys.path.insert(0, str(DP))
sys.path.insert(0, str(BASE / ".worktrees" / "kaggle"))

import pyarrow.parquet  # noqa: must precede torch
import pandas as pd

PROC_ROOT  = BASE / "data" / "processed"
OSM_ROOT   = BASE / "data" / "osm"
CKPT_DIR   = BASE / ".worktrees" / "kaggle" / "ckpt"
CACHE_OUT  = CKPT_DIR / "cache_test" / "porto_dev_n50_r50_k10.npz"
PARQUET    = PROC_ROOT / "porto" / "part-000.parquet"

SKIP_TRAJS  = 0
N_TRAJS     = 50

t0 = time.time()

tbl  = pyarrow.parquet.read_table(str(PARQUET), columns=["traj_id"])
uniq = pd.Series(tbl["traj_id"].to_pandas().unique())  # insertion order preserved
dev_ids = set(uniq.iloc[SKIP_TRAJS : SKIP_TRAJS + N_TRAJS].tolist())
print(f"[dev] traj range [{SKIP_TRAJS}, {SKIP_TRAJS+N_TRAJS})  IDs sample: {list(dev_ids)[:2]}")

tbl_full = pyarrow.parquet.read_table(str(PARQUET))
df_full  = tbl_full.to_pandas()
df       = df_full[df_full["traj_id"].isin(dev_ids)].reset_index(drop=True)
print(f"[dev] {len(df):,} fixes across {df['traj_id'].nunique()} trajectories")
del df_full, tbl_full, tbl

from dataset.candidates   import CandidateIndex
from dataset.trajectories import TrajectoryGraphDataset, collate_fn
from dataset.config       import SequenceConfig, RetrievalConfig

ci = CandidateIndex.from_city(str(OSM_ROOT), "porto")
ds = TrajectoryGraphDataset(
    df, {"porto": ci},
    SequenceConfig(), RetrievalConfig(),
    cache_path=str(CACHE_OUT),
)
print(f"[dev] cache built  {len(ds)} trajectories  elapsed={time.time()-t0:.1f}s")
print(f"Saved: {CACHE_OUT}")
