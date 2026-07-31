"""Download OSM drivable roads for a bbox and assemble segment features (P1b)."""

from __future__ import annotations

import math
from typing import Optional

import networkx as nx
import osmnx as ox

from roadgraph.config import (
    GraphConfig, HWY_LANES, HWY_SPEEDS_KPH, USEFUL_TAGS_WAY,
)
from roadgraph.features import (
    curvature, parse_flag, parse_lanes, parse_oneway, primary_highway,
)
from roadgraph.linegraph import build_turn_edges


def _num(val, default: float) -> float:
    """Coerce a possibly-None / NaN / inf OSM numeric to a finite float."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if (math.isnan(f) or math.isinf(f)) else f


def _configure_osmnx() -> None:
    ox.settings.useful_tags_way = sorted(
        set(ox.settings.useful_tags_way) | set(USEFUL_TAGS_WAY)
    )


def download_graph(bbox: tuple, cfg: GraphConfig) -> nx.MultiDiGraph:
    """Raw simplified drive graph for one bbox (lon_min, lat_min, lon_max, lat_max)."""
    _configure_osmnx()
    return ox.graph_from_bbox(
        bbox=bbox,
        network_type=cfg.network_type,
        simplify=cfg.simplify,
        truncate_by_edge=cfg.truncate_by_edge,
    )


def download_graph_tiled(bbox: tuple, cfg: GraphConfig, tiles: int) -> nx.MultiDiGraph:
    """Split bbox into a tiles x tiles grid, download each, compose into one graph.

    Keeps each Overpass request small enough to avoid timeouts on metro-scale boxes.
    Duplicate (u, v, key) edges from overlapping tiles are merged by nx.compose.
    """
    if tiles <= 1:
        return download_graph(bbox, cfg)
    left, bottom, right, top = bbox
    dx = (right - left) / tiles
    dy = (top - bottom) / tiles
    graphs = []
    for i in range(tiles):
        for j in range(tiles):
            sub = (left + i * dx, bottom + j * dy, left + (i + 1) * dx, bottom + (j + 1) * dy)
            try:
                graphs.append(download_graph(sub, cfg))
            except ox._errors.InsufficientResponseError:
                continue  # empty tile (e.g. over water / desert) -> skip
    if not graphs:
        raise RuntimeError("all tiles were empty")
    return nx.compose_all(graphs)


def enrich_graph(G: nx.MultiDiGraph, cfg: GraphConfig) -> nx.MultiDiGraph:
    """Add length / bearing / imputed speed edge attributes (graph must be lat-lon)."""
    if not all("length" in d for _, _, d in G.edges(data=True)):
        G = ox.distance.add_edge_lengths(G)
    G = ox.bearing.add_edge_bearings(G)
    G = ox.routing.add_edge_speeds(G, hwy_speeds=HWY_SPEEDS_KPH, fallback=cfg.speed_fallback_kph)
    return G


def graph_to_segments(
    G: nx.MultiDiGraph,
    cfg: GraphConfig,
    highway_to_id: dict[str, int],
) -> tuple[list[dict], "object"]:
    """Convert an enriched graph to per-segment feature records + nodes GeoDataFrame.

    `highway_to_id` is mutated in place: unseen highway classes get fresh ids so the
    embedding vocabulary stays consistent across cities.
    """
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    n = len(edges_gdf)

    coords = {osmid: (float(r.y), float(r.x)) for osmid, r in nodes_gdf.iterrows()}
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())

    def col(name, default=None):
        return edges_gdf[name].tolist() if name in edges_gdf.columns else [default] * n

    idx = edges_gdf.index
    us = idx.get_level_values("u").tolist()
    vs = idx.get_level_values("v").tolist()
    keys = idx.get_level_values("key").tolist()
    hw_raw, ow_raw, ln_raw = col("highway"), col("oneway"), col("lanes")
    spd, brg, length = col("speed_kph"), col("bearing"), col("length")
    bridge_raw, tunnel_raw, geom = col("bridge"), col("tunnel"), col("geometry")

    if "unknown" not in highway_to_id:
        highway_to_id["unknown"] = 0

    records: list[dict] = []
    for s, (u, v, key) in enumerate(zip(us, vs, keys)):
        hw = primary_highway(hw_raw[s])
        hid = highway_to_id.setdefault(hw, len(highway_to_id))
        la1, lo1 = coords.get(u, (0.0, 0.0))
        la2, lo2 = coords.get(v, (0.0, 0.0))
        records.append(
            {
                "segment_id": s,
                "u": int(u), "v": int(v), "key": int(key),
                "length_m": _num(length[s], 0.0),
                "heading_deg": _num(brg[s], 0.0),
                "curvature": curvature(_num(length[s], 0.0), la1, lo1, la2, lo2),
                "oneway": parse_oneway(ow_raw[s]),
                "lanes": parse_lanes(ln_raw[s], hw, cfg.default_lanes, HWY_LANES),
                "speed_kph": _num(spd[s], cfg.speed_fallback_kph),
                "highway": hw,
                "highway_id": hid,
                "bridge": parse_flag(bridge_raw[s]),
                "tunnel": parse_flag(tunnel_raw[s]),
                "u_indeg": int(indeg.get(u, 0)), "u_outdeg": int(outdeg.get(u, 0)),
                "v_indeg": int(indeg.get(v, 0)), "v_outdeg": int(outdeg.get(v, 0)),
                "geometry": geom[s],
            }
        )
    return records, nodes_gdf


def build_city(
    bbox: tuple,
    cfg: GraphConfig,
    highway_to_id: dict[str, int],
    tiles: int = 1,
) -> dict:
    """Full pipeline for one bbox -> segments, turn edges, nodes, meta."""
    G = download_graph_tiled(bbox, cfg, tiles)
    G = enrich_graph(G, cfg)
    segments, nodes_gdf = graph_to_segments(G, cfg, highway_to_id)
    turn_edges = build_turn_edges(segments, exclude_uturn=cfg.exclude_uturn)
    meta = {
        "bbox": list(bbox),
        "crs": "EPSG:4326",
        "network_type": cfg.network_type,
        "tiles": tiles,
        "n_segments": len(segments),
        "n_nodes": int(len(nodes_gdf)),
        "n_turn_edges": len(turn_edges),
        "osmnx_version": ox.__version__,
    }
    return {"segments": segments, "turn_edges": turn_edges, "nodes_gdf": nodes_gdf, "meta": meta}
