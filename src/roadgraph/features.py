"""Pure, network-free helpers to turn raw OSM edge tags into numeric features.

Each function is total: OSM tags are messy (missing, string, or list-valued), so
every parser coerces to a clean scalar with a documented fallback.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _first(val):
    """OSM tags are sometimes lists (a merged way); take the first element."""
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return val


def primary_highway(val) -> str:
    """Single highway class string; 'unknown' if absent."""
    v = _first(val)
    return str(v) if v is not None else "unknown"


def parse_oneway(val) -> int:
    """1 if the segment is one-way, else 0. Handles bool / 'yes' / list."""
    if isinstance(val, (list, tuple)):
        return int(any(parse_oneway(x) for x in val))
    if isinstance(val, bool):
        return int(val)
    return int(str(val).strip().lower() in {"yes", "true", "1", "-1", "reversible"})


def parse_lanes(val, highway: str, default_lanes: int, hwy_lanes: dict) -> int:
    """Integer lane count; impute by highway class then global default if missing."""
    v = _first(val)
    if v is not None:
        try:
            return max(1, int(round(float(v))))
        except (ValueError, TypeError):
            pass
    return int(hwy_lanes.get(highway, default_lanes))


def parse_flag(val) -> int:
    """Binary OSM flag (bridge/tunnel): 1 unless absent or explicitly 'no'."""
    v = _first(val)
    if v is None:
        return 0
    return int(str(v).strip().lower() not in {"no", "false", "0", ""})


def curvature(length_m: float, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Sinuosity = path length / straight-line endpoint distance, clamped to [1, 5].

    1.0 == perfectly straight. Loops (u == v) and zero-length straights return 1.0.
    """
    straight = haversine_m(lat1, lon1, lat2, lon2)
    if straight < 1.0 or length_m <= 0:
        return 1.0
    return float(min(max(length_m / straight, 1.0), 5.0))
