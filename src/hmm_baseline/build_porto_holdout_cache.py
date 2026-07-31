"""Build candidate cache for held-out Porto slice (trajs starting at 200000).

Training used the first 200k trajectories. Slice starts at index 200000 to
get a true held-out set for Porto, size given by --n-trajs.

load_source_df has no skip param, so we: read traj_id col to get ordered
unique IDs, slice [200000:200000+n], filter parquet to those IDs.

Usage (from src/hmm_baseline):
  python build_porto_holdout_cache.py --n-trajs 500
"""

import os, sys, time, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
BASE = Path(os.environ.get("AE_REPO_ROOT", SRC.parent))

import pyarrow.parquet  # noqa: must precede torch
import pyarrow as pa
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n-trajs", type=int, default=500)
args = ap.parse_args()

DATA_ROOT  = Path(os.environ.get("AE_DATA_ROOT", BASE / "data"))
CKPT_ROOT  = Path(os.environ.get("AE_CKPT_ROOT", BASE / "ckpt"))
PROC_ROOT  = DATA_ROOT / "processed"
OSM_ROOT   = DATA_ROOT / "osm"
CKPT_DIR   = CKPT_ROOT
CACHE_OUT  = CKPT_DIR / "cache_test" / f"porto_holdout_n{args.n_trajs}_r50_k10.npz"
PARQUET    = PROC_ROOT / "porto" / "part-000.parquet"

SKIP_TRAJS  = 200_000
N_TRAJS     = args.n_trajs

t0 = time.time()

# ── get ordered traj_ids for held-out slice ───────────────────────────────────
tbl  = pyarrow.parquet.read_table(str(PARQUET), columns=["traj_id"])
uniq = pd.Series(tbl["traj_id"].to_pandas().unique())  # insertion order preserved
held_ids = set(uniq.iloc[SKIP_TRAJS : SKIP_TRAJS + N_TRAJS].tolist())
print(f"[holdout] traj range [{SKIP_TRAJS}, {SKIP_TRAJS+N_TRAJS})  IDs sample: {list(held_ids)[:2]}")

# ── read full parquet filtered to held-out IDs ───────────────────────────────
# ponytail: load full traj_id col (~20MB) once to build the filter set, then
# stream-filter parquet — avoids 1.85GB full load
tbl_full = pyarrow.parquet.read_table(str(PARQUET))
df_full  = tbl_full.to_pandas()
df       = df_full[df_full["traj_id"].isin(held_ids)].reset_index(drop=True)
print(f"[holdout] {len(df):,} fixes across {df['traj_id'].nunique()} trajectories")
del df_full, tbl_full, tbl

# ── build candidate index and cache ──────────────────────────────────────────
from dataset.candidates   import CandidateIndex
from dataset.trajectories import TrajectoryGraphDataset
from dataset.config       import SequenceConfig, RetrievalConfig

ci = CandidateIndex.from_city(str(OSM_ROOT), "porto")
ds = TrajectoryGraphDataset(
    df, {"porto": ci},
    SequenceConfig(), RetrievalConfig(),
    cache_path=str(CACHE_OUT),
)
print(f"[holdout] cache built  {len(ds)} trajectories  elapsed={time.time()-t0:.1f}s")
print(f"Saved: {CACHE_OUT}")
