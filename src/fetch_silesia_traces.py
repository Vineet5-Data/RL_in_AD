"""Fetch OSM public GPS traces for the Silesia region and write the processed
training parquet (gt_benchmark_spec.md Phase 2, data step).

Region: lon 18.60-19.05, lat 49.45-50.32 (holds Kubicka GT tracks 27/33/60).
API: /api/0.6/trackpoints paged GPX, 0.1x0.1 deg tiles (well under the 0.25
deg^2 limit). Only timestamped points (public/identifiable traces) are kept —
anonymous points have no order and cannot form training sequences.

Each <trkseg> becomes one raw trace; the existing preprocessing pipeline
(clean_trajectory) then dedupes/splits/filters and emits the schema parquet at
data/processed/silesia/part-000.parquet, ready for stage2r --source silesia.

Usage:  python fetch_silesia_traces.py [--max-pages-per-tile 300]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
DP = BASE / ".worktrees" / "data-preprocess"
sys.path.insert(0, str(DP))

BBOX = (18.60, 49.45, 19.05, 50.32)   # lon_min, lat_min, lon_max, lat_max
TILE = 0.1
API = "https://api.openstreetmap.org/api/0.6/trackpoints"
RAW = BASE / "data" / "gt_benchmarks" / "silesia_traces_raw.jsonl.gz"
OUT_DIR = BASE / "data" / "processed" / "silesia"
NS = {"g": "http://www.topografix.com/GPX/1/0"}


def tiles():
    lon0, lat0, lon1, lat1 = BBOX
    lon = lon0
    while lon < lon1:
        lat = lat0
        while lat < lat1:
            yield (round(lon, 2), round(lat, 2),
                   round(min(lon + TILE, lon1), 2), round(min(lat + TILE, lat1), 2))
            lat += TILE
        lon += TILE


def fetch(max_pages: int):
    n_seg = n_pt = 0
    with gzip.open(RAW, "wt", encoding="utf-8") as fh:
        for ti, (a, b, c, d) in enumerate(tiles()):
            for page in range(max_pages):
                url = f"{API}?bbox={a},{b},{c},{d}&page={page}"
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(url, timeout=120) as r:
                            xml = r.read()
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        time.sleep(5.0 * (attempt + 1))
                root = ET.fromstring(xml)
                got = 0
                for seg in root.iter("{http://www.topografix.com/GPX/1/0}trkseg"):
                    pts = []
                    for pt in seg.findall("g:trkpt", NS):
                        t = pt.find("g:time", NS)
                        if t is None:
                            continue
                        ts = datetime.fromisoformat(t.text.replace("Z", "+00:00"))
                        pts.append((float(pt.get("lat")), float(pt.get("lon")),
                                    int(ts.replace(tzinfo=timezone.utc).timestamp())))
                    got += len(pts)
                    if len(pts) >= 20:
                        fh.write(json.dumps(pts) + "\n")
                        n_seg += 1
                        n_pt += len(pts)
                n_total = sum(1 for _ in root.iter("{http://www.topografix.com/GPX/1/0}trkpt"))
                if n_total < 5000:   # last page of this tile
                    break
                time.sleep(0.25)
            print(f"[tile {ti + 1}/45] ({a},{b})  pages={page + 1}  "
                  f"kept_segs={n_seg}  kept_pts={n_pt:,}", flush=True)
    print(f"[fetch] done  segs={n_seg}  timestamped_pts={n_pt:,}  -> {RAW}")


def preprocess():
    from collections import Counter

    import numpy as np
    import pandas as pd
    from preprocessing.clean import clean_trajectory
    from preprocessing.schema import CleanConfig

    # sanity box = fetch bbox + margin (lat_min, lat_max, lon_min, lon_max)
    cfg = CleanConfig(bbox={"silesia": (49.40, 50.40, 18.55, 19.10)})
    counter = Counter()
    frames, traj_id = [], 0
    with gzip.open(RAW, "rt", encoding="utf-8") as fh:
        for line in fh:
            pts = json.loads(line)
            pts.sort(key=lambda p: p[2])
            lat = [p[0] for p in pts]
            lon = [p[1] for p in pts]
            t = [p[2] for p in pts]
            for segd in clean_trajectory(lat, lon, t, "silesia", cfg, counter):
                n = len(segd["lat"])
                if n < 20:
                    continue
                frames.append(pd.DataFrame({
                    "traj_id": np.full(n, traj_id, dtype=np.int64),
                    "source": "silesia",
                    **{k: segd[k] for k in ("seq", "lat", "lon", "t", "dt",
                                             "dist_m", "speed_mps", "bearing_deg")},
                }))
                traj_id += 1
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["lat", "lon", "t"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "part-000.parquet", index=False)
    print(f"[preprocess] trajs={traj_id:,}  fixes={len(df):,}  -> {OUT_DIR / 'part-000.parquet'}")
    print(f"[D1 bar] fixes>=300k: {'PASS' if len(df) >= 300_000 else 'FAIL'}   "
          f"trajs>=3k: {'PASS' if traj_id >= 3_000 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages-per-tile", type=int, default=300)
    ap.add_argument("--skip-fetch", action="store_true", help="re-run preprocessing only")
    args = ap.parse_args()
    if not args.skip_fetch:
        fetch(args.max_pages_per_tile)
    preprocess()


if __name__ == "__main__":
    main()
