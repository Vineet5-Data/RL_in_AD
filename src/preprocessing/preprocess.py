"""Driver: parse -> clean -> segment -> write training-ready Parquet + manifest.

Usage (from the worktree root):
    python -m preprocessing.preprocess --sources geolife tdrive porto
    python -m preprocessing.preprocess --sources porto --limit 2000   # quick smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .clean import clean_trajectory
from .parsers import PARSERS
from .schema import ARROW_SCHEMA, CleanConfig

DEFAULT_DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)
FLUSH_POINTS = 250_000  # rows buffered before a Parquet row group is written


class _Buffer:
    """Accumulates cleaned segments and flushes them as Parquet row groups."""

    def __init__(self, writer: pq.ParquetWriter, source: str):
        self.writer = writer
        self.source = source
        self.cols: dict[str, list] = {k: [] for k in (
            "traj_id", "seq", "lat", "lon", "t", "dt", "dist_m", "speed_mps", "bearing_deg"
        )}
        self.n = 0

    def add(self, traj_id: str, seg: dict) -> None:
        m = seg["seq"].shape[0]
        self.cols["traj_id"].append(np.full(m, traj_id, dtype=object))
        for k in ("seq", "lat", "lon", "t", "dt", "dist_m", "speed_mps", "bearing_deg"):
            self.cols[k].append(seg[k])
        self.n += m
        if self.n >= FLUSH_POINTS:
            self.flush()

    def flush(self) -> None:
        if self.n == 0:
            return
        traj_id = np.concatenate(self.cols["traj_id"])
        arrays = {k: np.concatenate(self.cols[k]) for k in self.cols if k != "traj_id"}
        table = pa.table(
            {
                "traj_id": pa.array(traj_id, type=pa.string()),
                "source": pa.array([self.source] * self.n, type=pa.string()),
                "seq": arrays["seq"],
                "lat": arrays["lat"],
                "lon": arrays["lon"],
                "t": arrays["t"],
                "dt": arrays["dt"],
                "dist_m": arrays["dist_m"],
                "speed_mps": arrays["speed_mps"],
                "bearing_deg": arrays["bearing_deg"],
            },
            schema=ARROW_SCHEMA,
        )
        self.writer.write_table(table)
        self.cols = {k: [] for k in self.cols}
        self.n = 0


def process_source(source: str, data_root: str, out_dir: str, cfg: CleanConfig,
                   limit: int = 0) -> dict:
    parser = PARSERS[source]
    counter: Counter = Counter()
    src_out = os.path.join(out_dir, source)
    os.makedirs(src_out, exist_ok=True)
    part = os.path.join(src_out, "part-000.parquet")

    t0 = time.time()
    writer = pq.ParquetWriter(part, ARROW_SCHEMA, compression="zstd")
    buf = _Buffer(writer, source)
    try:
        for n_traj, (raw_id, lats, lons, ts) in enumerate(parser(data_root), start=1):
            for seg_i, seg in enumerate(
                clean_trajectory(lats, lons, ts, source, cfg, counter)
            ):
                buf.add(f"{source}_{raw_id}_{seg_i:03d}", seg)
            if limit and n_traj >= limit:
                break
        buf.flush()
    finally:
        writer.close()

    stats = dict(counter)
    stats["elapsed_s"] = round(time.time() - t0, 2)
    stats["output"] = os.path.relpath(part, out_dir)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=None, help="default: <data-root>/processed")
    ap.add_argument("--sources", nargs="+", default=["geolife", "tdrive", "porto"],
                    choices=list(PARSERS))
    ap.add_argument("--limit", type=int, default=0,
                    help="max raw trajectories per source (0 = all)")
    ap.add_argument("--max-speed-mps", type=float, default=55.0)
    ap.add_argument("--gap-split-s", type=float, default=180.0)
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--no-bbox", action="store_true")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.data_root, "processed")
    os.makedirs(out_dir, exist_ok=True)
    cfg = CleanConfig(
        max_speed_mps=args.max_speed_mps,
        gap_split_s=args.gap_split_s,
        min_points=args.min_points,
        apply_bbox=not args.no_bbox,
    )

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": os.path.abspath(args.data_root),
        "config": vars(cfg).copy(),
        "sources": {},
    }
    manifest["config"].pop("bbox", None)

    for src in args.sources:
        print(f"[{src}] processing ...", flush=True)
        stats = process_source(src, args.data_root, out_dir, cfg, limit=args.limit)
        manifest["sources"][src] = stats
        print(f"[{src}] " + "  ".join(f"{k}={v}" for k, v in stats.items()), flush=True)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest -> {os.path.join(out_dir, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
