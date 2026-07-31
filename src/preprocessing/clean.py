"""Source-agnostic cleaning + trip segmentation for raw GPS trajectories."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from .geo import EARTH_RADIUS_M, haversine_m, initial_bearing_deg
from .schema import CleanConfig


def _haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = lat2r - lat1r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def _finalize_segment(lat: np.ndarray, lon: np.ndarray, t: np.ndarray) -> dict:
    """Compute derived per-point fields for one contiguous cleaned segment."""
    m = lat.shape[0]
    dt = np.zeros(m, dtype=np.float64)
    dist = np.zeros(m, dtype=np.float64)
    bearing = np.zeros(m, dtype=np.float64)
    if m > 1:
        dt[1:] = (t[1:] - t[:-1]).astype(np.float64)
        dist[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        bearing[1:] = initial_bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
    speed = np.zeros(m, dtype=np.float64)
    moving = dt > 0
    speed[moving] = dist[moving] / dt[moving]
    return {
        "seq": np.arange(m, dtype=np.int32),
        "lat": lat,
        "lon": lon,
        "t": t,
        "dt": dt.astype(np.float32),
        "dist_m": dist.astype(np.float32),
        "speed_mps": speed.astype(np.float32),
        "bearing_deg": bearing.astype(np.float32),
    }


def clean_trajectory(
    lats: Sequence[float],
    lons: Sequence[float],
    ts: Sequence[int],
    source: str,
    cfg: CleanConfig,
    counter: Counter,
) -> list[dict]:
    """Clean one raw trajectory into zero or more training-ready segments.

    Pipeline: drop invalid/out-of-box fixes -> sort by time -> single sequential
    pass dropping non-increasing timestamps and >max_speed jumps -> split on time
    gaps -> keep only segments with >= cfg.min_points. Each returned dict holds
    numpy arrays for the canonical per-point columns (minus traj_id/source).
    """
    counter["traj_in"] += 1
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    t = np.asarray(ts, dtype=np.int64)
    n0 = lat.shape[0]
    counter["points_in"] += n0
    if n0 == 0:
        return []

    valid = (
        np.isfinite(lat) & np.isfinite(lon)
        & (lat >= -90.0) & (lat <= 90.0)
        & (lon >= -180.0) & (lon <= 180.0)
        & ~((lat == 0.0) & (lon == 0.0))
    )
    box = cfg.box_for(source)
    if box is not None:
        la0, la1, lo0, lo1 = box
        valid &= (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    counter["dropped_invalid"] += int(n0 - valid.sum())
    lat, lon, t = lat[valid], lon[valid], t[valid]
    if lat.shape[0] == 0:
        return []

    order = np.argsort(t, kind="stable")
    lat, lon, t = lat[order], lon[order], t[order]

    # Sequential pass: re-references each candidate against the last KEPT point so
    # that dropping a jump/duplicate does not corrupt the next speed estimate.
    keep = []
    last = -1
    for i in range(t.shape[0]):
        if last < 0:
            keep.append(i)
            last = i
            continue
        gap = t[i] - t[last]
        if gap <= 0:
            counter["dropped_dup"] += 1
            continue
        speed = _haversine_scalar(lat[last], lon[last], lat[i], lon[i]) / gap
        if speed > cfg.max_speed_mps:
            counter["dropped_speed"] += 1
            continue
        keep.append(i)
        last = i

    if not keep:
        return []
    idx = np.asarray(keep, dtype=np.int64)
    lat, lon, t = lat[idx], lon[idx], t[idx]

    # Split into sub-trajectories wherever the time gap exceeds the threshold.
    gaps = np.diff(t)
    split_points = np.nonzero(gaps > cfg.gap_split_s)[0] + 1
    chunks = np.split(np.arange(t.shape[0]), split_points)

    segments = []
    for ch in chunks:
        if ch.shape[0] < cfg.min_points:
            counter["segments_too_short"] += 1
            continue
        seg = _finalize_segment(lat[ch], lon[ch], t[ch])
        counter["segments_out"] += 1
        counter["points_out"] += ch.shape[0]
        segments.append(seg)
    return segments
