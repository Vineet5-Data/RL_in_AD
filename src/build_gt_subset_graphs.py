"""Build OUR OSM graphs for selected Kubicka GT tracks (gt_benchmark_spec.md
Phase 1 step 2). One small bbox per track, shared feature vocab so the
Porto-trained Stage-0 road encoder applies zero-shot (same recipe as tdrive).

Usage, from .worktrees/research2:
  python build_gt_subset_graphs.py 00000002 00000015 ...
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
DP = BASE / ".worktrees" / "data-preprocess"
sys.path.insert(0, str(DP))

import pyarrow.parquet  # noqa: F401,E402
import numpy as np  # noqa: E402

from roadgraph.build import build_city  # noqa: E402
from roadgraph.config import GraphConfig  # noqa: E402
from roadgraph.io import load_vocab, save_city, save_vocab  # noqa: E402

DATA = BASE / "data" / "gt_benchmarks" / "kubicka"
OSM_OUT = BASE / "data" / "osm"
MARGIN = 0.02  # deg (~2 km) around the track extent


def main():
    tids = sys.argv[1:]
    assert tids, "pass track ids, e.g. 00000002"
    cfg = GraphConfig(network_type="drive")
    vocab = load_vocab(OSM_OUT)
    for tid in tids:
        city = f"kub{tid}"
        if (OSM_OUT / city / "segments.parquet").exists():
            print(f"[{city}] exists, skip")
            continue
        from eval_gt_kubicka import load_track  # lon/lat order handled there
        track, _, _, _ = load_track(tid)
        lat, lon = track[:, 0], track[:, 1]
        bbox = (float(lon.min() - MARGIN), float(lat.min() - MARGIN),
                float(lon.max() + MARGIN), float(lat.max() + MARGIN))
        t0 = time.time()
        print(f"[{city}] bbox={tuple(round(b, 3) for b in bbox)}  downloading...")
        result = build_city(bbox, cfg, vocab, tiles=1)
        meta = save_city(OSM_OUT, city, result)
        save_vocab(OSM_OUT, vocab)
        print(f"[{city}] segments={meta['n_segments']:,}  "
              f"nodes={meta['n_nodes']:,}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
