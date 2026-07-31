"""Configuration + constants for road-graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Full extents (lon_min, lat_min, lon_max, lat_max) == osmnx 2.x bbox order
# (left, bottom, right, top). Matched to the cleaned-GPS extent per city.
CITY_BBOX = {
    # Beijing covers both geolife and tdrive (~200x190 km metro region).
    "beijing": (115.40, 39.40, 117.60, 41.20),
    # Porto (~32x55 km).
    "porto": (-8.78, 40.90, -8.40, 41.40),
    # San Francisco Bay Area, matched to the cleaned Cabspotting GPS extent.
    "cabspotting": (-122.6, 37.2, -121.95, 38.05),
    # Rome, matched to the cleaned roma/taxi GPS extent.
    "romataxi": (11.7, 41.45, 13.05, 42.35),
    # Hannover, Germany -- held-out cross-city generalization eval target, not trained on.
    "hannover": (9.4, 52.1, 10.2, 52.7),
}

# Small central tiles for fast end-to-end validation without a metro-scale download.
TEST_TILES = {
    "beijing": (116.387, 39.884, 116.427, 39.924),   # ~Tiananmen +/- 0.02 deg
    "porto": (-8.631, 41.130, -8.591, 41.170),        # central Porto +/- 0.02 deg
}

# Cross-city priors (km/h) used to impute speed where OSM has no maxspeed tag.
# Tagged maxspeed always takes precedence; these only fill gaps.
HWY_SPEEDS_KPH = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 60, "primary_link": 40, "secondary": 50, "secondary_link": 40,
    "tertiary": 40, "tertiary_link": 30, "residential": 30, "living_street": 15,
    "unclassified": 40, "service": 20, "road": 40,
}

# Lane-count priors used only when OSM omits the lanes tag.
HWY_LANES = {
    "motorway": 3, "trunk": 2, "primary": 2, "secondary": 2, "tertiary": 2,
    "residential": 1, "living_street": 1, "unclassified": 1, "service": 1,
}

# Tags pulled from OSM ways onto edges (superset of osmnx defaults we rely on).
USEFUL_TAGS_WAY = [
    "highway", "oneway", "lanes", "maxspeed", "bridge", "tunnel",
    "junction", "ref", "name", "access", "service",
]

# Canonical per-segment schema written to segments.parquet.
SEGMENT_COLUMNS = [
    "segment_id", "u", "v", "key",
    "length_m", "heading_deg", "curvature",
    "oneway", "lanes", "speed_kph",
    "highway", "highway_id", "bridge", "tunnel",
    "u_indeg", "u_outdeg", "v_indeg", "v_outdeg",
    "geometry",
]


@dataclass
class GraphConfig:
    """Tunable knobs for OSM road-graph extraction."""

    network_type: str = "drive"     # vehicle GPS -> drivable roads only
    simplify: bool = True           # merge interstitial nodes -> segments = ways between junctions
    speed_fallback_kph: float = 40.0
    default_lanes: int = 1
    exclude_uturn: bool = True      # drop e->reverse(e) turn edges in the line graph
    truncate_by_edge: bool = True   # keep edges that cross the bbox boundary
    bbox: Optional[tuple] = field(default=None)  # explicit override (lon_min,lat_min,lon_max,lat_max)
