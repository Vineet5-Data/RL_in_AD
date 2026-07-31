"""Per-source readers. Each yields (raw_id, lats, lons, ts) for one raw trajectory.

All timestamps are emitted as unix seconds. Naive local clock times (GeoLife,
T-Drive) are interpreted as UTC consistently across the dataset: only relative
time deltas drive the cleaning and the P3 physics rewards, so a constant timezone
offset is irrelevant. Porto carries an explicit unix start time.
"""

from __future__ import annotations

import calendar
import csv
import glob
import json
import os
import sys
from typing import Iterator, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

Trajectory = Tuple[str, list, list, list]


def _ymdhms_to_unix(s: str) -> int:
    """Parse 'YYYY-MM-DD HH:MM:SS' to unix seconds (UTC), fast path without strptime."""
    return calendar.timegm(
        (int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]), int(s[17:19]), 0, 0, 0)
    )


# --------------------------------------------------------------------------- #
# GeoLife: data_root/Geolife Trajectories 1.3/.../Data/<user>/Trajectory/*.plt
# .plt = 6 header lines, then: lat,lon,0,alt_feet,days_since_1899,YYYY-MM-DD,HH:MM:SS
# --------------------------------------------------------------------------- #
def iter_geolife(data_root: str) -> Iterator[Trajectory]:
    matches = glob.glob(os.path.join(data_root, "**", "Data"), recursive=True)
    if not matches:
        return
    data_dir = matches[0]
    for user in sorted(os.listdir(data_dir)):
        traj_dir = os.path.join(data_dir, user, "Trajectory")
        if not os.path.isdir(traj_dir):
            continue
        for plt in sorted(glob.glob(os.path.join(traj_dir, "*.plt"))):
            lats, lons, ts = [], [], []
            with open(plt, "r", encoding="utf-8", errors="ignore") as fh:
                for _ in range(6):  # skip header
                    fh.readline()
                for line in fh:
                    f = line.rstrip("\n").split(",")
                    if len(f) < 7:
                        continue
                    try:
                        lats.append(float(f[0]))
                        lons.append(float(f[1]))
                        ts.append(_ymdhms_to_unix(f[5] + " " + f[6]))
                    except (ValueError, IndexError):
                        continue
            if lats:
                yield (f"{user}_{os.path.splitext(os.path.basename(plt))[0]}", lats, lons, ts)


# --------------------------------------------------------------------------- #
# Porto / PKDD-15: one large CSV. POLYLINE is a JSON list of [lon, lat] pairs,
# sampled every 15 s from TIMESTAMP (unix). MISSING_DATA flags broken streams.
# The macOS unzip nests the real file as train.csv/train.csv.
# --------------------------------------------------------------------------- #
def _resolve_porto_csv(data_root: str) -> str:
    base = os.path.join(data_root, "pkdd-15-predict-taxi-service-trajectory-i")
    for cand in (
        os.path.join(base, "train.csv", "train.csv"),
        os.path.join(base, "train.csv"),
    ):
        if os.path.isfile(cand):
            return cand
    hits = [p for p in glob.glob(os.path.join(base, "**", "train.csv"), recursive=True) if os.path.isfile(p)]
    if not hits:
        raise FileNotFoundError(f"Porto train.csv not found under {base}")
    return hits[0]


def iter_porto(data_root: str, interval_s: int = 15) -> Iterator[Trajectory]:
    path = _resolve_porto_csv(data_root)
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return
        col = {name: i for i, name in enumerate(header)}
        i_id, i_ts = col["TRIP_ID"], col["TIMESTAMP"]
        i_missing, i_poly = col["MISSING_DATA"], col["POLYLINE"]
        for row in reader:
            if len(row) <= i_poly or row[i_missing] == "True":
                continue
            try:
                pts = json.loads(row[i_poly])
            except json.JSONDecodeError:
                continue
            if not pts:
                continue
            start = int(row[i_ts])
            lats = [p[1] for p in pts]
            lons = [p[0] for p in pts]
            ts = [start + interval_s * k for k in range(len(pts))]
            yield (row[i_id], lats, lons, ts)


# --------------------------------------------------------------------------- #
# T-Drive: release/taxi_log_2008_by_id/<id>.txt
# each line: taxi_id,YYYY-MM-DD HH:MM:SS,lon,lat
# --------------------------------------------------------------------------- #
def iter_tdrive(data_root: str) -> Iterator[Trajectory]:
    matches = glob.glob(os.path.join(data_root, "**", "taxi_log_2008_by_id"), recursive=True)
    if not matches:
        return
    log_dir = matches[0]
    for txt in sorted(glob.glob(os.path.join(log_dir, "*.txt")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else 0):
        lats, lons, ts = [], [], []
        with open(txt, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                f = line.rstrip("\n").split(",")
                if len(f) < 4:
                    continue
                try:
                    ts.append(_ymdhms_to_unix(f[1]))
                    lons.append(float(f[2]))
                    lats.append(float(f[3]))
                except (ValueError, IndexError):
                    continue
        if lats:
            yield (os.path.splitext(os.path.basename(txt))[0], lats, lons, ts)


# --------------------------------------------------------------------------- #
# GeoLife car+taxi filter: same raw .plt files, restricted to the 69/182 users
# who have a labels.txt (ground-truth transportation-mode windows) and only
# the "car"/"taxi" -tagged time ranges. Ground-truth mode beats the Silesia
# iteration-2 recipe's speed-only heuristic (that let walker/rail speeds
# through and poisoned the retrain -- see project_summary.md Phase 2).
# labels.txt: header row, then "Start Time\tEnd Time\tMode" (YYYY/MM/DD HH:MM:SS).
# --------------------------------------------------------------------------- #
CAR_MODES = {"car", "taxi"}


def _read_geolife_car_windows(user_dir: str) -> list:
    path = os.path.join(user_dir, "labels.txt")
    windows = []
    if not os.path.isfile(path):
        return windows
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        next(fh, None)  # header
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3 or f[2].strip().lower() not in CAR_MODES:
                continue
            try:
                t0 = _ymdhms_to_unix(f[0].replace("/", "-"))
                t1 = _ymdhms_to_unix(f[1].replace("/", "-"))
            except (ValueError, IndexError):
                continue
            windows.append((t0, t1))
    windows.sort()
    return windows


def iter_geolife_car(data_root: str) -> Iterator[Trajectory]:
    matches = glob.glob(os.path.join(data_root, "**", "Data"), recursive=True)
    if not matches:
        return
    data_dir = matches[0]
    for user in sorted(os.listdir(data_dir)):
        user_dir = os.path.join(data_dir, user)
        windows = _read_geolife_car_windows(user_dir)
        if not windows:
            continue
        traj_dir = os.path.join(user_dir, "Trajectory")
        if not os.path.isdir(traj_dir):
            continue
        for plt in sorted(glob.glob(os.path.join(traj_dir, "*.plt"))):
            pts = []
            with open(plt, "r", encoding="utf-8", errors="ignore") as fh:
                for _ in range(6):
                    fh.readline()
                for line in fh:
                    f = line.rstrip("\n").split(",")
                    if len(f) < 7:
                        continue
                    try:
                        pts.append((float(f[0]), float(f[1]),
                                    _ymdhms_to_unix(f[5] + " " + f[6])))
                    except (ValueError, IndexError):
                        continue
            if not pts:
                continue
            run_lat: list = []
            run_lon: list = []
            run_t: list = []
            run_i = 0
            wi = 0
            for lat, lon, t in pts:
                while wi < len(windows) and t > windows[wi][1]:
                    wi += 1
                in_window = wi < len(windows) and windows[wi][0] <= t <= windows[wi][1]
                if in_window:
                    run_lat.append(lat)
                    run_lon.append(lon)
                    run_t.append(t)
                elif run_t:
                    yield (f"{user}_{os.path.splitext(os.path.basename(plt))[0]}_{run_i:02d}",
                           run_lat, run_lon, run_t)
                    run_i += 1
                    run_lat, run_lon, run_t = [], [], []
            if run_t:
                yield (f"{user}_{os.path.splitext(os.path.basename(plt))[0]}_{run_i:02d}",
                       run_lat, run_lon, run_t)


# --------------------------------------------------------------------------- #
# San Francisco Cabspotting (CRAWDAD epfl/mobility):
# data_root/cabspotting/new_<cabid>.txt, each line: "lat lon occupied unix_ts",
# taxi-only by construction, files are newest-first -- sort ascending by t.
# --------------------------------------------------------------------------- #
def iter_cabspotting(data_root: str) -> Iterator[Trajectory]:
    base = os.path.join(data_root, "cabspotting")
    if not os.path.isdir(base):
        return
    for path in sorted(glob.glob(os.path.join(base, "new_*.txt"))):
        cab_id = os.path.splitext(os.path.basename(path))[0][len("new_"):]
        lats, lons, ts = [], [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                f = line.split()
                if len(f) < 4:
                    continue
                try:
                    lats.append(float(f[0]))
                    lons.append(float(f[1]))
                    ts.append(int(f[3]))
                except (ValueError, IndexError):
                    continue
        if not ts:
            continue
        order = sorted(range(len(ts)), key=ts.__getitem__)
        yield (cab_id, [lats[i] for i in order], [lons[i] for i in order],
               [ts[i] for i in order])


# --------------------------------------------------------------------------- #
# Rome taxi (CRAWDAD roma/taxi): data_root/roma_taxi/taxi_february.txt, one
# huge file, all taxis interleaved by time: "id;TIMESTAMP+TZ;POINT(lat lon)".
# Timezone suffix dropped (project convention: naive local clock, only
# relative deltas matter -- see module docstring). Taxi-only by construction.
# --------------------------------------------------------------------------- #
def iter_romataxi(data_root: str) -> Iterator[Trajectory]:
    import numpy as np
    import pandas as pd

    path = os.path.join(data_root, "roma_taxi", "taxi_february.txt")
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path, sep=";", header=None, names=["id", "ts", "point"],
                      dtype={"id": np.int64, "ts": str, "point": str})
    coords = df["point"].str.extract(r"POINT\(([-\d.]+) ([-\d.]+)\)").astype(np.float64)
    df["lat"], df["lon"] = coords[0], coords[1]
    # fractional-second digit count varies row to row -- strip the "+HH" tz
    # suffix by splitting rather than a fixed-width slice, then let pandas
    # infer the variable-precision ISO8601 fraction.
    ts_naive = df["ts"].str.split("+").str[0]
    dt = pd.to_datetime(ts_naive, format="ISO8601")
    df["t"] = dt.astype(np.int64) // 10 ** 9
    df.sort_values(["id", "t"], inplace=True, kind="mergesort")
    for tid, g in df.groupby("id", sort=False):
        yield (str(tid), g["lat"].tolist(), g["lon"].tolist(), g["t"].tolist())


# --------------------------------------------------------------------------- #
# Hannover single-user collection (Uni Hannover data repository):
# data_root/hannover_gps/hannover_table.csv, columns
# gid,trip_id,unixtime,north_utm,east_utm (UTM zone 32N). Already trip-segmented
# via trip_id; one driver/one car, car-only by construction, no mode filtering
# needed. Deliberately NOT added to any training mix -- reserved as the one
# genuinely held-out city for cross-city generalization eval (project_summary.md,
# "no held-out city left" gap) since porto/tdrive/geolife_car/cabspotting/romataxi
# are all inside the current multi-city training mix.
# --------------------------------------------------------------------------- #
def iter_hannover(data_root: str) -> Iterator[Trajectory]:
    import pandas as pd
    from pyproj import Transformer

    path = os.path.join(data_root, "hannover_gps", "hannover_table.csv")
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    df.sort_values(["trip_id", "unixtime"], inplace=True, kind="mergesort")
    to_wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(df["east_utm"].to_numpy(), df["north_utm"].to_numpy())
    df["lat"], df["lon"] = lat, lon
    for tid, g in df.groupby("trip_id", sort=False):
        yield (str(tid), g["lat"].tolist(), g["lon"].tolist(), g["unixtime"].tolist())


PARSERS = {"geolife": iter_geolife, "geolife_car": iter_geolife_car,
           "porto": iter_porto, "tdrive": iter_tdrive,
           "cabspotting": iter_cabspotting, "romataxi": iter_romataxi,
           "hannover": iter_hannover}
