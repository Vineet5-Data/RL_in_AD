"""Configuration for candidate retrieval, reward weighting, and sequence batching."""

from __future__ import annotations

from dataclasses import dataclass

# Which OSM city graph each GPS source is matched against.
CITY_OF_SOURCE = {"geolife": "beijing", "tdrive": "beijing", "porto": "porto"}

# Fallback metric CRS per city if geopandas UTM estimation is unavailable.
CITY_UTM_CRS = {"beijing": "EPSG:32650", "porto": "EPSG:32629"}


@dataclass
class RetrievalConfig:
    """Candidate road retrieval (methodology 2.2): C_t = {e : d_perp(o_t,e) <= 50m}."""

    radius_m: float = 50.0   # d_perp cutoff
    k: int = 10              # max candidates per fix (padded)


@dataclass
class RewardConfig:
    """P3 physical-constraint reward weights (methodology 2.5)."""

    sigma_gps_m: float = 10.0
    oneway_penalty: float = 5.0   # M
    # lambda order: [dist, head, oneway, speed, topo, smooth]
    weights: tuple = (1.0, 0.5, 1.0, 0.3, 2.0, 0.2)


@dataclass
class SequenceConfig:
    """Trajectory sequence batching (methodology Stage 1 / Stage 2)."""

    max_len: int = 128       # Stage-1 transformer cap; Stage-2 RSSM uses 64
    min_len: int = 5
    mask_ratio: float = 0.15  # masked-GPS-modeling fraction
    pad_seg_id: int = -1
