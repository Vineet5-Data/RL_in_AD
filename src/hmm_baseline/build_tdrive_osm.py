"""Build Beijing/T-Drive road graph.

v2 bbox: expanded from 5th-95th percentile to recover the 20.7% no-candidate
gap caused by GPS fixes near clipped edges. ~52 km x 55 km.
Saves to data/osm/tdrive/ (overwrites prior build).

Usage (from src/hmm_baseline):
  python build_tdrive_osm.py
"""

import os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
BASE = Path(os.environ.get("AE_REPO_ROOT", SRC.parent))

import pyarrow.parquet  # noqa: must precede torch
from roadgraph.build  import build_city
from roadgraph.config import GraphConfig
from roadgraph.io     import load_vocab, save_city, save_vocab

OSM_OUT = Path(os.environ.get("AE_DATA_ROOT", BASE / "data")) / "osm"
# 5th–95th percentile of T-Drive lat/lon (covers ~90% of trajectories)
# lon_min, lat_min, lon_max, lat_max
TDRIVE_BBOX = (116.05, 39.70, 116.70, 40.20)  # expanded ~52 km x 55 km

cfg   = GraphConfig(network_type="drive")
vocab = load_vocab(OSM_OUT)

t0 = time.time()
print(f"[tdrive] bbox={TDRIVE_BBOX}  tiles=3  downloading...")
result = build_city(TDRIVE_BBOX, cfg, vocab, tiles=3)
meta   = save_city(OSM_OUT, "tdrive", result)
save_vocab(OSM_OUT, vocab)
dt = time.time() - t0

print(f"[tdrive] segments={meta['n_segments']:,}  nodes={meta['n_nodes']:,}  "
      f"turn_edges={meta['n_turn_edges']:,}  elapsed={dt:.1f}s")
print(f"Saved to: {OSM_OUT / 'tdrive'}")
