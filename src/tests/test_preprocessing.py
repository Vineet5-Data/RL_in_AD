"""Unit tests for the trajectory cleaning core (no dataset files required)."""

from collections import Counter

import numpy as np
import pytest

from preprocessing.clean import clean_trajectory
from preprocessing.geo import haversine_m, initial_bearing_deg
from preprocessing.schema import CleanConfig

BEIJING_LON = 116.40
BASE_LAT = 39.900


def _cfg(**kw):
    base = dict(max_speed_mps=55.0, gap_split_s=180.0, min_points=3, apply_bbox=True)
    base.update(kw)
    return CleanConfig(**base)


def test_haversine_one_degree_latitude():
    d = float(haversine_m(np.array([0.0]), np.array([0.0]),
                          np.array([1.0]), np.array([0.0]))[0])
    assert d == pytest.approx(111_195.0, rel=1e-3)


def test_bearing_due_north():
    b = float(initial_bearing_deg(np.array([39.9]), np.array([116.4]),
                                  np.array([39.91]), np.array([116.4]))[0])
    assert b == pytest.approx(0.0, abs=1e-6)


def test_happy_path_single_segment():
    """A clean, evenly-sampled track yields one segment with correct derived fields."""
    lats = [BASE_LAT + 0.001 * i for i in range(5)]
    lons = [BEIJING_LON] * 5
    ts = [60 * i for i in range(5)]
    counter = Counter()
    segs = clean_trajectory(lats, lons, ts, "geolife", _cfg(), counter)

    assert len(segs) == 1
    seg = segs[0]
    assert seg["seq"].tolist() == [0, 1, 2, 3, 4]
    assert seg["dt"][0] == 0.0 and seg["dist_m"][0] == 0.0
    np.testing.assert_allclose(seg["dt"][1:], 60.0)
    # ~111 m per 0.001 deg lat over 60 s => ~1.85 m/s
    assert seg["speed_mps"][1:].mean() == pytest.approx(1.85, abs=0.1)
    assert counter["points_out"] == 5 and counter["segments_out"] == 1


def test_velocity_cap_drops_teleport_and_rereferences():
    """A >200 km/h jump is dropped; the next point is measured from the last kept fix."""
    lats = [BASE_LAT, BASE_LAT + 0.001, BASE_LAT + 1.0, BASE_LAT + 0.002, BASE_LAT + 0.003]
    lons = [BEIJING_LON] * 5
    ts = [0, 60, 120, 180, 240]
    counter = Counter()
    segs = clean_trajectory(lats, lons, ts, "geolife", _cfg(), counter)

    assert counter["dropped_speed"] == 1
    assert len(segs) == 1
    assert segs[0]["seq"].shape[0] == 4
    assert float(segs[0]["speed_mps"].max()) < 55.0


def test_duplicate_timestamp_dropped():
    lats = [BASE_LAT + 0.001 * i for i in range(5)]
    lons = [BEIJING_LON] * 5
    ts = [0, 60, 60, 120, 180]  # repeated timestamp at index 2
    counter = Counter()
    segs = clean_trajectory(lats, lons, ts, "geolife", _cfg(), counter)

    assert counter["dropped_dup"] == 1
    assert sum(s["seq"].shape[0] for s in segs) == 4


def test_gap_split_into_two_segments():
    lats = [BASE_LAT + 0.001 * i for i in range(6)]
    lons = [BEIJING_LON] * 6
    ts = [0, 60, 120, 1000, 1060, 1120]  # 880 s gap after the 3rd point
    counter = Counter()
    segs = clean_trajectory(lats, lons, ts, "geolife", _cfg(), counter)

    assert len(segs) == 2
    assert [s["seq"].shape[0] for s in segs] == [3, 3]
    assert counter["segments_out"] == 2


def test_too_short_dropped():
    counter = Counter()
    segs = clean_trajectory(
        [BASE_LAT, BASE_LAT + 0.001], [BEIJING_LON, BEIJING_LON], [0, 60],
        "geolife", _cfg(), counter,
    )
    assert segs == []
    assert counter["segments_too_short"] == 1


def test_invalid_and_out_of_bbox_dropped():
    lats = [BASE_LAT, 0.0, 999.0, BASE_LAT + 0.001, 39.9, BASE_LAT + 0.002, BASE_LAT + 0.003]
    lons = [BEIJING_LON, 0.0, 116.4, BEIJING_LON, -8.6, BEIJING_LON, BEIJING_LON]
    ts = [0, 60, 120, 180, 240, 300, 360]
    counter = Counter()
    segs = clean_trajectory(lats, lons, ts, "geolife", _cfg(), counter)

    # (0,0), lat=999, and the Porto-region fix (lon -8.6) are all removed
    assert counter["dropped_invalid"] == 3
    assert sum(s["seq"].shape[0] for s in segs) == 4
