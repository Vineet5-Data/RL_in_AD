"""P0b: measure the pseudo-label entropy floor (run2_open_checks_resolution.md L2).

Road CE trains against the soft target softmax(-d_perp^2 / 2*sigma^2) (sigma=10m,
_soft_road_ce in training/stage2.py). CE against a soft target is bounded below by
H(target): min CE = H(soft_tgt). So mean H over Porto candidate sets IS the floor
the road head can reach. Compare to the observed plateau road CE ~1.36 nats:
  H ~= 1.36  -> model sits AT the floor: signal-bound, capacity/schedule can't help.
  H <<  1.36 -> headroom: the plateau is model/schedule, not the labels.

Candidates via the porto retrieval index (radius 50m, k 10 = RetrievalConfig),
mirroring eval_w1_silesia. Local inference only.

Usage, from src/evals:  python eval_p0b_floor.py [--n 3000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from matcher import Matcher, BASE  # noqa: E402
from dataset.trajectories import load_source_df  # noqa: E402

SIGMA = 10.0  # must match _soft_road_ce


def soft_target_entropy(d_perp):
    """H(softmax(-d^2/2sigma^2)) in nats for one candidate set."""
    d = np.asarray(d_perp, np.float64)
    logits = -(d ** 2) / (2 * SIGMA ** 2)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="fixes to sample")
    ap.add_argument("--source", default="porto")
    args = ap.parse_args()

    m = Matcher(args.source)
    df = load_source_df(str(BASE / "data" / "processed"), args.source, None)
    lat = df["lat"].to_numpy(np.float64)
    lon = df["lon"].to_numpy(np.float64)

    # even stride across the file so we span many trajectories, not one region
    idx = np.linspace(0, len(lat) - 1, min(args.n, len(lat))).astype(np.int64)
    Hs, k_hist = [], []
    for i in idx:
        q = m.ci.query(lat[i], lon[i], radius_m=50.0, k=10)
        dp = np.asarray(q["d_perp_m"], np.float64)
        if len(dp) < 2:          # 0/1 candidate -> H=0 trivially, not informative
            continue
        Hs.append(soft_target_entropy(dp))
        k_hist.append(len(dp))

    Hs = np.asarray(Hs)
    print(f"[P0b] source={args.source}  sampled={len(idx)}  usable(>=2 cand)={len(Hs)}  "
          f"mean_cand={np.mean(k_hist):.1f}")
    print(f"[P0b] soft-target entropy floor H:  mean {Hs.mean():.4f}  "
          f"median {np.median(Hs):.4f}  p10 {np.percentile(Hs,10):.4f}  "
          f"p90 {np.percentile(Hs,90):.4f}  nats")
    obs = 1.36
    print(f"[P0b] observed plateau road CE ~= {obs}")
    gap = obs - Hs.mean()
    print(f"[P0b] CE - floor = {gap:.4f} nats  -> "
          + ("AT floor (signal-bound; capacity/schedule can't help)" if gap < 0.20
             else "HEADROOM below floor (plateau is model/schedule, not labels)"))


if __name__ == "__main__":
    main()
