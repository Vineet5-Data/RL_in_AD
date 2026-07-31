"""Persist / load road graphs: Parquet segments + turn edges + shared vocab + PyG."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

VOCAB_FILE = "highway_vocab.json"


def load_vocab(out_root: Path) -> dict[str, int]:
    f = Path(out_root) / VOCAB_FILE
    if f.exists():
        return json.loads(f.read_text())
    return {"unknown": 0}


def save_vocab(out_root: Path, vocab: dict[str, int]) -> None:
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / VOCAB_FILE).write_text(json.dumps(vocab, indent=2, sort_keys=True))


def _feature_stats(df: pd.DataFrame) -> dict:
    cols = ["length_m", "curvature", "speed_kph", "lanes"]
    return {c: {"mean": float(df[c].mean()), "std": float(df[c].std() or 1.0)} for c in cols}


def save_city(out_root: Path, city: str, result: dict) -> dict:
    """Write segments.parquet, line_graph_edges.parquet, nodes.parquet, meta.json."""
    city_dir = Path(out_root) / city
    city_dir.mkdir(parents=True, exist_ok=True)

    seg_gdf = gpd.GeoDataFrame(result["segments"], geometry="geometry", crs="EPSG:4326")
    seg_gdf.to_parquet(city_dir / "segments.parquet")

    edges = pd.DataFrame(result["turn_edges"], columns=["src", "dst"])
    edges.to_parquet(city_dir / "line_graph_edges.parquet", index=False)

    nodes = result["nodes_gdf"].copy()
    nodes = nodes.reset_index().rename(columns={"osmid": "node_id"})
    keep = [c for c in ("node_id", "x", "y", "geometry") if c in nodes.columns]
    gpd.GeoDataFrame(nodes[keep], geometry="geometry", crs="EPSG:4326").to_parquet(
        city_dir / "nodes.parquet"
    )

    meta = dict(result["meta"])
    meta["feature_stats"] = _feature_stats(seg_gdf)
    (city_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_pyg(out_root: Path, city: str, normalize: bool = True):
    """Build a torch_geometric Data object (segment nodes + turn edges) for a city.

    x          : (N, 13) continuous features
    highway_id : (N,) long, for an embedding lookup (P1b road_type_embed)
    edge_index : (2, E) turn connectivity
    """
    import numpy as np
    import torch
    from torch_geometric.data import Data

    city_dir = Path(out_root) / city
    cols = [
        "length_m", "heading_deg", "curvature", "oneway", "lanes", "speed_kph",
        "bridge", "tunnel", "u_indeg", "u_outdeg", "v_indeg", "v_outdeg", "highway_id",
    ]
    df = pd.read_parquet(city_dir / "segments.parquet", columns=cols)
    edges = pd.read_parquet(city_dir / "line_graph_edges.parquet")

    rad = np.radians(df["heading_deg"].to_numpy())
    feats = {
        "log_length": np.log1p(df["length_m"].to_numpy()),
        "head_sin": np.sin(rad),
        "head_cos": np.cos(rad),
        "curvature": df["curvature"].to_numpy(),
        "oneway": df["oneway"].to_numpy(float),
        "lanes": df["lanes"].to_numpy(float),
        "speed_kph": df["speed_kph"].to_numpy(float),
        "bridge": df["bridge"].to_numpy(float),
        "tunnel": df["tunnel"].to_numpy(float),
        "u_indeg": df["u_indeg"].to_numpy(float),
        "u_outdeg": df["u_outdeg"].to_numpy(float),
        "v_indeg": df["v_indeg"].to_numpy(float),
        "v_outdeg": df["v_outdeg"].to_numpy(float),
    }
    x = np.column_stack(list(feats.values())).astype("float32")
    if normalize:
        # z-score the unbounded columns; leave sin/cos and binary flags untouched.
        # By name, not position -- a reordered/added feats key can't silently
        # z-score (or skip z-scoring) the wrong column.
        pass_through = {"head_sin", "head_cos", "oneway", "bridge", "tunnel"}
        zcols = [i for i, k in enumerate(feats) if k not in pass_through]
        mu = x[:, zcols].mean(0)
        sd = x[:, zcols].std(0)
        sd[sd == 0] = 1.0
        x[:, zcols] = (x[:, zcols] - mu) / sd

    edge_index = torch.tensor(edges[["src", "dst"]].to_numpy().T, dtype=torch.long)
    return Data(
        x=torch.from_numpy(x),
        highway_id=torch.tensor(df["highway_id"].to_numpy(), dtype=torch.long),
        edge_index=edge_index,
        num_nodes=len(df),
    )
