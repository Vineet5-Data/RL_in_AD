"""CLI: build OSM road graphs for cities and write them under data/osm/."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from roadgraph.build import build_city
from roadgraph.config import CITY_BBOX, GraphConfig, TEST_TILES
from roadgraph.io import load_vocab, save_city, save_vocab

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data" / "osm"
AUTO_TILES = {"beijing": 6, "porto": 1}  # Beijing metro is too big for one Overpass query


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build OSM road graphs (P1b).")
    ap.add_argument("--cities", nargs="+", default=["beijing", "porto"],
                    choices=sorted(CITY_BBOX))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--test-tile", action="store_true",
                    help="use a small central tile per city (fast validation)")
    ap.add_argument("--tiles", type=int, default=0,
                    help="grid subdivisions for the Overpass download (0 = per-city auto)")
    ap.add_argument("--network-type", default="drive")
    args = ap.parse_args(argv)

    cfg = GraphConfig(network_type=args.network_type)
    args.out.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab(args.out)

    summary = []
    for city in args.cities:
        bbox = TEST_TILES[city] if args.test_tile else CITY_BBOX[city]
        tiles = 1 if args.test_tile else (args.tiles or AUTO_TILES.get(city, 1))
        t0 = time.time()
        print(f"[{city}] bbox={bbox} tiles={tiles} downloading...", flush=True)
        result = build_city(bbox, cfg, vocab, tiles=tiles)
        meta = save_city(args.out, city, result)
        dt = time.time() - t0
        summary.append((city, meta["n_segments"], meta["n_nodes"], meta["n_turn_edges"], dt))
        print(f"[{city}] segments={meta['n_segments']:,} nodes={meta['n_nodes']:,} "
              f"turn_edges={meta['n_turn_edges']:,} elapsed={dt:.1f}s", flush=True)

    save_vocab(args.out, vocab)
    print("\n=== summary ===")
    for c, s, n, e, dt in summary:
        print(f"{c:10s} segments={s:>8,} nodes={n:>8,} turn_edges={e:>9,} {dt:6.1f}s")
    print(f"highway vocab size = {len(vocab)} -> {args.out / 'highway_vocab.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
