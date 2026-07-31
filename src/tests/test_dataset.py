"""Network-free tests for the training-data layer (rewards, temporal, collate, groups)."""

import numpy as np
import pandas as pd
import pytest
import torch

from dataset.candidates import build_neighbor_map
from dataset.config import RewardConfig
from dataset.rewards import (
    angular_diff_deg, combined_reward, reward_dist, reward_heading,
    reward_oneway, reward_smooth, reward_speed, reward_topo,
)
from dataset.temporal import day_of_week, hour_of_day, time_of_day_frac
from dataset.trajectories import _contiguous_groups, collate_fn


# ---- rewards ----
def test_angular_diff_wraps():
    assert angular_diff_deg(350, 10) == pytest.approx(20.0)
    assert angular_diff_deg(10, 190) == pytest.approx(180.0)


def test_reward_dist_and_heading():
    assert reward_dist(0.0) == 0.0
    assert reward_dist(20.0, sigma_gps_m=10.0) == -2.0
    assert reward_heading(90, 90) == pytest.approx(1.0)
    assert reward_heading(0, 180) == pytest.approx(-1.0)


def test_reward_oneway():
    assert reward_oneway(1, 0, 180, penalty=5.0) == -5.0   # against one-way
    assert reward_oneway(1, 0, 10, penalty=5.0) == 0.0      # with one-way
    assert reward_oneway(0, 0, 180, penalty=5.0) == 0.0     # two-way: never penalized


def test_reward_speed():
    assert reward_speed(10.0, 50.0) == 0.0                  # 10 m/s < 50 km/h (~13.9)
    assert reward_speed(20.0, 36.0) < 0.0                   # 20 m/s > 36 km/h (10 m/s)


def test_reward_topo():
    assert reward_topo(5, None, []) == 1.0                  # first step unconstrained
    assert reward_topo(5, 2, [5, 7]) == 1.0
    assert reward_topo(9, 2, [5, 7]) == 0.0


def test_reward_smooth_and_combined():
    assert reward_smooth(10, 10) == 0.0
    assert reward_smooth(10, 40) == pytest.approx(-30.0)
    comp = {"dist": -1.0, "head": 1.0, "oneway": 0.0, "speed": 0.0, "topo": 1.0, "smooth": 0.0}
    # weights [1.0, 0.5, 1.0, 0.3, 2.0, 0.2] -> -1 + 0.5 + 2.0 = 1.5
    assert combined_reward(comp, RewardConfig()) == pytest.approx(1.5)


# ---- temporal ----
def test_temporal_epoch():
    t = np.array([0, 13 * 3600])
    assert hour_of_day(t).tolist() == [0, 13]
    assert day_of_week(np.array([0]))[0] == 4          # 1970-01-01 was Thursday
    assert time_of_day_frac(np.array([43200]))[0] == pytest.approx(0.5)


# ---- grouping ----
def test_contiguous_groups():
    arr = np.array(["a", "a", "b", "b", "b", "c"])
    assert _contiguous_groups(arr) == [(0, 2), (2, 5), (5, 6)]
    assert _contiguous_groups(np.array([])) == []


# ---- adjacency ----
def test_build_neighbor_map(tmp_path):
    p = tmp_path / "porto"; p.mkdir()
    pd.DataFrame({"src": [0, 0, 1], "dst": [1, 2, 3]}).to_parquet(p / "line_graph_edges.parquet")
    nb = build_neighbor_map(tmp_path, "porto")
    assert nb[0] == {1, 2} and nb[1] == {3}


# ---- collate ----
def _item(L, K=3):
    g = {k: torch.arange(L, dtype=torch.float32) for k in
         ("lat", "lon", "dt", "speed_mps", "bearing_deg")}
    g.update({k: torch.zeros(L, dtype=torch.long) for k in ("hour", "dow")})
    for k in ("cand_d_perp_m", "cand_heading_deg", "cand_speed_kph", "cand_mask"):
        g[k] = torch.ones(L, K, dtype=torch.float32)
    for k in ("cand_segment_id", "cand_oneway", "cand_highway_id"):
        g[k] = torch.zeros(L, K, dtype=torch.long)
    g["length"] = L
    return g


def test_collate_pads_and_masks():
    batch = [_item(3), _item(5)]
    out = collate_fn(batch)
    assert out["lat"].shape == (2, 5)
    assert out["cand_mask"].shape == (2, 5, 3)
    assert out["seq_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert out["seq_mask"][1].tolist() == [1, 1, 1, 1, 1]
    assert out["lat"][0, 3:].sum() == 0.0                  # padded region is zero
