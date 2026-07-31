"""Network-free unit tests for road-graph feature parsing + turn connectivity."""

import math

import pytest

from roadgraph.config import HWY_LANES
from roadgraph.features import (
    curvature, haversine_m, parse_flag, parse_lanes, parse_oneway, primary_highway,
)
from roadgraph.linegraph import build_turn_edges


@pytest.mark.parametrize("val,exp", [
    (True, 1), (False, 0), ("yes", 1), ("no", 0), ("-1", 1),
    (["no", "yes"], 1), (None, 0),
])
def test_parse_oneway(val, exp):
    assert parse_oneway(val) == exp


def test_parse_lanes_value_list_and_impute():
    assert parse_lanes("3", "primary", 1, HWY_LANES) == 3
    assert parse_lanes(["2", "3"], "primary", 1, HWY_LANES) == 2       # first element
    assert parse_lanes("2.0", "primary", 1, HWY_LANES) == 2
    assert parse_lanes(None, "motorway", 1, HWY_LANES) == HWY_LANES["motorway"]
    assert parse_lanes("garbage", "unknown", 1, HWY_LANES) == 1        # default fallback


@pytest.mark.parametrize("val,exp", [
    (None, 0), ("yes", 1), ("no", 0), ("viaduct", 1), (["yes"], 1), ("", 0),
])
def test_parse_flag(val, exp):
    assert parse_flag(val) == exp


def test_primary_highway():
    assert primary_highway("residential") == "residential"
    assert primary_highway(["primary", "secondary"]) == "primary"
    assert primary_highway(None) == "unknown"


def test_haversine_one_degree_lat():
    assert haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(111_195.0, rel=1e-3)


def test_curvature_bounds():
    d = haversine_m(39.90, 116.40, 39.91, 116.40)  # ~1112 m straight
    assert curvature(d, 39.90, 116.40, 39.91, 116.40) == pytest.approx(1.0, abs=1e-3)
    assert curvature(2 * d, 39.90, 116.40, 39.91, 116.40) == pytest.approx(2.0, abs=1e-2)
    assert curvature(0.0, 39.90, 116.40, 39.90, 116.40) == 1.0   # zero-length loop
    assert curvature(100 * d, 39.90, 116.40, 39.91, 116.40) == 5.0  # clamp


# Turn graph: s0=(1,2), s1=(2,3), s2=(2,1)=reverse(s0), s3=(3,2)=reverse(s1)
_SEGS = [
    {"segment_id": 0, "u": 1, "v": 2},
    {"segment_id": 1, "u": 2, "v": 3},
    {"segment_id": 2, "u": 2, "v": 1},
    {"segment_id": 3, "u": 3, "v": 2},
]


def test_turn_edges_exclude_uturn():
    edges = set(build_turn_edges(_SEGS, exclude_uturn=True))
    assert edges == {(0, 1), (3, 2)}


def test_turn_edges_with_uturn():
    edges = build_turn_edges(_SEGS, exclude_uturn=False)
    assert len(edges) == 6
    assert all(s != d for s, d in edges)  # never a self-loop
