"""Canonical trajectory schema + cleaning configuration shared by all sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pyarrow as pa

# One row per GPS fix. This is the training-ready contract the P1a encoder and
# P3 reward functions consume (they need lat/lon/time + derived speed/heading).
CANONICAL_COLUMNS = [
    "traj_id",      # globally unique: "{source}_{raw_id}_{seg:03d}"
    "source",       # geolife | porto | tdrive
    "seq",          # 0-based point index within the cleaned (sub)trajectory
    "lat",          # WGS84 latitude  (float64)
    "lon",          # WGS84 longitude (float64)
    "t",            # unix seconds (int64)
    "dt",           # seconds since previous kept point (0 at seq 0)
    "dist_m",       # haversine metres from previous kept point (0 at seq 0)
    "speed_mps",    # dist_m / dt   (0 at seq 0)
    "bearing_deg",  # initial bearing prev->this in [0,360) (0 at seq 0)
]

ARROW_SCHEMA = pa.schema(
    [
        ("traj_id", pa.string()),
        ("source", pa.string()),
        ("seq", pa.int32()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("t", pa.int64()),
        ("dt", pa.float32()),
        ("dist_m", pa.float32()),
        ("speed_mps", pa.float32()),
        ("bearing_deg", pa.float32()),
    ]
)

# Geographic sanity boxes (lat_min, lat_max, lon_min, lon_max) to drop fixes that
# teleport out of the city. Generous padding around the documented coverage.
BBOX = {
    "geolife": (39.4, 41.2, 115.4, 117.6),   # Beijing
    "geolife_car": (39.4, 41.2, 115.4, 117.6),  # Beijing (same as geolife)
    "tdrive": (39.4, 41.2, 115.4, 117.6),    # Beijing
    "porto": (40.9, 41.4, -8.8, -8.4),       # Porto, Portugal
    "cabspotting": (37.2, 38.05, -122.6, -121.95),  # San Francisco Bay Area (CRAWDAD epfl/mobility)
    "romataxi": (41.45, 42.35, 11.7, 13.05),        # Rome, Italy (CRAWDAD roma/taxi)
    "hannover": (52.1, 52.7, 9.4, 10.2),             # Hannover, Germany (held-out eval city)
}


@dataclass
class CleanConfig:
    """Tunable cleaning thresholds (methodology Open Question #5)."""

    max_speed_mps: float = 55.0   # ~198 km/h: drop fixes implying faster motion
    gap_split_s: float = 180.0    # split into a new sub-trajectory on time gaps > this
    min_points: int = 5           # discard (sub)trajectories shorter than this
    apply_bbox: bool = True       # drop out-of-city fixes
    bbox: Optional[dict] = field(default=None)

    def box_for(self, source: str):
        table = self.bbox if self.bbox is not None else BBOX
        return table.get(source) if self.apply_bbox else None
