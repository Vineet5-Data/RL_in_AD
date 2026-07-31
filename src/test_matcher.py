"""Smoke test for matcher.py — bars pre-registered 2026-07-11 before first run:

  porto   match_offline (hybrid) >= 0.83 tolerant Hit@1, 25 held-out trajs
          match_online  (WM)     >= 0.70
          predict(horizon=15): 15 segments, consecutive pairs graph-connected
  tdrive  match_offline (auto -> nk) >= 0.75, 10 trajs

Run from src/:  python test_matcher.py
"""

import numpy as np

from matcher import Matcher, ev  # noqa: E402  (matcher sets up sys.path + import order; ev used below)

import pyarrow.parquet as pq  # noqa: E402
import pandas as pd  # noqa: E402


def tol_hit(m, lat, lon, pred):
    """Tolerant Hit@1: chosen segment's d_perp <= (min candidate d_perp) + 5 m."""
    q = m.ci.query_batch(np.asarray(lat), np.asarray(lon), 50.0, 10)
    seg, dp, msk = q["segment_id"], q["d_perp_m"], q["mask"]
    ok = []
    for i in range(len(lat)):
        v = (msk[i] > 0) & (seg[i] >= 0)
        if not v.any():
            continue
        d1 = float(dp[i][v].min())
        j = np.where((seg[i] == pred[i]) & v)[0]
        ok.append(bool(len(j)) and float(dp[i][j[0]]) <= d1 + 5.0)
    return float(np.mean(ok))


def trajs_porto(n=25):
    tbl = pq.read_table(str(ev.PARQUET), columns=["traj_id"])
    uniq = pd.Series(tbl["traj_id"].to_pandas().unique())
    held = set(uniq.iloc[ev.SKIP_TRAJS: ev.SKIP_TRAJS + n].tolist())
    df = pq.read_table(str(ev.PARQUET)).to_pandas()
    df = df[df["traj_id"].isin(held)]
    for _, g in df.groupby("traj_id", sort=False):
        yield g["lat"].to_numpy(), g["lon"].to_numpy(), g["t"].to_numpy(np.int64)


def trajs_tdrive(n=10):
    from dataset.trajectories import load_source_df
    df = load_source_df(str(ev.PROC_ROOT), "tdrive", limit_trajs=n)
    for _, g in df.groupby("traj_id", sort=False):
        yield g["lat"].to_numpy(), g["lon"].to_numpy(), g["t"].to_numpy(np.int64)


def main():
    print("[1/3] porto (trained city: hybrid offline / WM online / predict)")
    m = Matcher("porto")
    assert m.trained
    off, on = [], []
    first = None
    for lat, lon, t in trajs_porto():
        first = first or (lat, lon, t)
        off.append(tol_hit(m, lat, lon, m.match_offline(lat, lon, t)))
        on.append(tol_hit(m, lat, lon, m.match_online(lat, lon, t)))
    off_m, on_m = float(np.mean(off)), float(np.mean(on))
    print(f"  offline hybrid {off_m:.4f} (bar 0.83)   online WM {on_m:.4f} (bar 0.70)")
    assert off_m >= 0.83, f"offline {off_m} < 0.83"
    assert on_m >= 0.70, f"online {on_m} < 0.70"

    lat, lon, t = first
    path = m.predict(lat, lon, t, horizon=15)
    assert len(path) == 15
    connected = all(int(b) in (m.nbr.get(int(a), set()) | {int(a)})
                    for a, b in zip(path[:-1], path[1:]))
    print(f"  predict: {len(path)} segs, connected={connected}")
    assert connected

    print("[2/3] tdrive (unseen city: auto NK fallback)")
    mt = Matcher("tdrive")
    assert not mt.trained
    offt = [tol_hit(mt, lat, lon, mt.match_offline(lat, lon, t))
            for lat, lon, t in trajs_tdrive()]
    offt_m = float(np.mean(offt))
    print(f"  offline nk {offt_m:.4f} (bar 0.75)")
    assert offt_m >= 0.75, f"tdrive offline {offt_m} < 0.75"

    print("[3/3] OK — all bars met")


if __name__ == "__main__":
    main()
