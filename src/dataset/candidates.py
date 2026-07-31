"""Spatial candidate-road retrieval + turn adjacency (methodology 2.2 / 2.5-C4).

CandidateIndex projects road segments to a metric CRS and answers, for any GPS
fix, "which segments pass within d_perp <= radius, and what are their road
attributes?" -- the C_t set the cross-modal encoder attends over and the P3
reward scores against.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.config import CITY_UTM_CRS


class CandidateIndex:
    """STRtree over projected segment geometries with attached road attributes."""

    def __init__(self, gdf_metric, attrs: dict):
        from shapely import STRtree, length

        self._geoms = gdf_metric.geometry.values
        self._tree = STRtree(self._geoms)
        self._lengths = length(self._geoms)  # precomputed once, reused by both query paths
        self.segment_id = attrs["segment_id"]
        self.heading = attrs["heading_deg"]
        self.speed_kph = attrs["speed_kph"]
        self.oneway = attrs["oneway"]
        self.highway_id = attrs["highway_id"]

        from pyproj import Transformer
        self._to_metric = Transformer.from_crs("EPSG:4326", gdf_metric.crs, always_xy=True)

    @classmethod
    def from_city(cls, osm_root, city: str) -> "CandidateIndex":
        import geopandas as gpd

        gdf = gpd.read_parquet(Path(osm_root) / city / "segments.parquet")
        try:
            metric_crs = gdf.estimate_utm_crs()
        except Exception:
            metric_crs = CITY_UTM_CRS[city]
        gm = gdf.to_crs(metric_crs)
        attrs = {c: gdf[c].to_numpy() for c in
                 ("segment_id", "heading_deg", "speed_kph", "oneway", "highway_id")}
        return cls(gm, attrs)

    def query(self, lat: float, lon: float, radius_m: float = 50.0, k: int = 10) -> dict:
        """Return up to k nearest segments within radius_m of (lat, lon).

        Keys: segment_id, d_perp_m, frac (position along segment in [0,1]),
        heading_deg, speed_kph, oneway, highway_id. Empty arrays if none in range.
        """
        from shapely.geometry import Point

        x, y = self._to_metric.transform(lon, lat)
        p = Point(x, y)
        idx = np.asarray(self._tree.query(p, predicate="dwithin", distance=radius_m))
        if idx.size == 0:
            return self._empty()

        d = np.array([self._geoms[i].distance(p) for i in idx])
        order = np.argsort(d)[:k]
        sel = idx[order]
        frac = np.array([
            (self._geoms[i].project(p) / self._geoms[i].length) if self._geoms[i].length > 0 else 0.0
            for i in sel
        ])
        return {
            "segment_id": self.segment_id[sel],
            "d_perp_m": d[order],
            "frac": frac,
            "heading_deg": self.heading[sel],
            "speed_kph": self.speed_kph[sel],
            "oneway": self.oneway[sel],
            "highway_id": self.highway_id[sel],
        }

    def query_batch(self, lats: np.ndarray, lons: np.ndarray, radius_m: float = 50.0,
                     k: int = 10) -> dict:
        """Vectorized `query` for many fixes at once (single bulk STRtree call).

        Same fields as `query` plus `mask`, all as [N, K] padded arrays -- this is
        the whole-dataset precompute path `TrajectoryGraphDataset` caches once
        instead of re-querying per fix, per epoch.
        """
        from shapely import points, distance, line_locate_point

        n = len(lats)
        out = {name: np.full((n, k), fill, dtype=dt) for name, fill, dt in (
            ("segment_id", -1, np.int64), ("d_perp_m", 0.0, np.float32),
            ("frac", 0.0, np.float32), ("heading_deg", 0.0, np.float32),
            ("speed_kph", 0.0, np.float32), ("oneway", 0, np.int64),
            ("highway_id", 0, np.int64))}
        out["mask"] = np.zeros((n, k), dtype=np.float32)
        if n == 0:
            return out

        x, y = self._to_metric.transform(np.asarray(lons), np.asarray(lats))
        pts = points(x, y)
        input_idx, tree_idx = self._tree.query(pts, predicate="dwithin", distance=radius_m)
        if input_idx.size == 0:
            return out

        geoms = self._geoms[tree_idx]
        d = distance(geoms, pts[input_idx])
        lengths = self._lengths[tree_idx]
        proj = line_locate_point(geoms, pts[input_idx])
        frac = np.divide(proj, lengths, out=np.zeros_like(proj), where=lengths > 0)

        # top-k nearest per fix: numpy lexsort (stable) instead of a pandas
        # sort_values+groupby.cumcount over the flattened match array -- dwithin can
        # return dozens of matches/point in dense road networks, and the pandas path
        # measured ~45min per 1M-point chunk on Kaggle CPU at 200k-traj scale.
        order = np.lexsort((d, input_idx))  # primary key = last arg = input_idx
        idx_s, tree_s, d_s, frac_s = input_idx[order], tree_idx[order], d[order], frac[order]
        new_group = np.concatenate(([True], idx_s[1:] != idx_s[:-1]))
        group_start = np.flatnonzero(new_group)
        rank = np.arange(idx_s.size) - group_start[np.cumsum(new_group) - 1]
        keep = rank < k
        ii, rr, tt = idx_s[keep], rank[keep], tree_s[keep]

        out["segment_id"][ii, rr] = self.segment_id[tt]
        out["d_perp_m"][ii, rr] = d_s[keep]
        out["frac"][ii, rr] = frac_s[keep]
        out["heading_deg"][ii, rr] = self.heading[tt]
        out["speed_kph"][ii, rr] = self.speed_kph[tt]
        out["oneway"][ii, rr] = self.oneway[tt]
        out["highway_id"][ii, rr] = self.highway_id[tt]
        out["mask"][ii, rr] = 1.0
        return out

    @staticmethod
    def _empty() -> dict:
        return {
            "segment_id": np.empty(0, np.int64), "d_perp_m": np.empty(0),
            "frac": np.empty(0), "heading_deg": np.empty(0), "speed_kph": np.empty(0),
            "oneway": np.empty(0, np.int64), "highway_id": np.empty(0, np.int64),
        }


def build_neighbor_map(osm_root, city: str) -> dict[int, set]:
    """segment_id -> set of downstream segment_ids (for the topological-continuity reward)."""
    edges = pd.read_parquet(Path(osm_root) / city / "line_graph_edges.parquet")
    nb: dict[int, set] = defaultdict(set)
    for s, d in zip(edges["src"].to_numpy(), edges["dst"].to_numpy()):
        nb[int(s)].add(int(d))
    return nb
