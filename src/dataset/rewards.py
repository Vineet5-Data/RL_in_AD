"""P3 label-free physical-constraint rewards (methodology 2.5).

All functions are pure and unit-testable. Inputs use SI where noted:
GPS speed in m/s, road speed limit in km/h (converted internally).
"""

from __future__ import annotations

import math
from typing import Iterable

from dataset.config import RewardConfig

KMH_TO_MS = 1.0 / 3.6


def angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angle between two compass bearings, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def reward_dist(d_perp_m: float, sigma_gps_m: float = 10.0) -> float:
    """Constraint 1: closer perpendicular distance is better."""
    return -float(d_perp_m) / sigma_gps_m


def reward_heading(theta_gps: float, theta_road: float) -> float:
    """Constraint 2a: cosine alignment of GPS course and road bearing, in [-1, 1]."""
    return math.cos(math.radians(theta_gps - theta_road))


def reward_oneway(oneway: int, theta_gps: float, theta_road: float,
                  penalty: float = 5.0) -> float:
    """Constraint 2b: large penalty for driving a one-way road against its direction."""
    if oneway and angular_diff_deg(theta_gps, theta_road) > 90.0:
        return -float(penalty)
    return 0.0


def reward_speed(v_gps_ms: float, v_limit_kph: float) -> float:
    """Constraint 3: quadratic penalty for exceeding the segment speed limit."""
    v_limit_ms = v_limit_kph * KMH_TO_MS
    if v_limit_ms <= 0:
        return 0.0
    over = max(0.0, (v_gps_ms - v_limit_ms) / v_limit_ms)
    return -(over ** 2)


def reward_topo(seg_id: int, prev_seg_id, neighbors: Iterable[int]) -> float:
    """Constraint 4: 1 if the chosen segment is graph-adjacent to the previous one."""
    if prev_seg_id is None:
        return 1.0  # no constraint on the first step
    return 1.0 if seg_id in set(neighbors) else 0.0


def reward_smooth(dtheta_gps: float, dtheta_road: float) -> float:
    """Constraint 5: penalize divergence between GPS and road turn angles."""
    return -angular_diff_deg(dtheta_gps, dtheta_road)


def combined_reward(components: dict, cfg: RewardConfig = RewardConfig()) -> float:
    """Weighted sum in lambda order [dist, head, oneway, speed, topo, smooth]."""
    order = ("dist", "head", "oneway", "speed", "topo", "smooth")
    return float(sum(w * components[k] for w, k in zip(cfg.weights, order)))
