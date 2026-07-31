"""W1 held-out proxy check (gt_benchmark_spec.md Phase 2).

Retrained silesia WM vs zero-shot Porto WM, ONLINE proxy tolerant Hit@1 on
held-out silesia trajs (train used first 7,000 traj ids; this scores the rest).
mode="base" isolates the WM road head (no NK terms); hyb+topo reported too.

Conventions mirror eval_research2: candidates = ci.query(radius 50 m, k 10),
tolerant hit = chosen d_perp <= best d_perp + 5 m, first 8 fixes warmup-skipped.

Bar (pre-registered): retrained base > zero-shot base on same split.

Usage, from src/evals:  python eval_w1_silesia.py [--max-traj 100]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from matcher import Matcher, BASE  # noqa: E402
from dataset.trajectories import load_source_df  # noqa: E402

WARM = 8
MAX_FIX = 200        # ponytail: cap per-traj fixes, enough for a health check


def _maybe_load_unfrozen_encoder(m, ckpt_path):
    """Matcher._road_z always loads the FROZEN Porto stage0 encoder -- an
    --unfreeze-encoder training run bakes updated weights into ckpt['road_enc']
    but the stock Matcher path silently ignores them. Override m.road_z in
    place when present, so L3 (encoder-confound isolation) actually tests the
    unfrozen encoder instead of re-testing the frozen one under a new name."""
    import torch
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if "road_enc" not in sd:
        return False
    import eval_research2 as ev
    from models.road_encoder import RoadGAT
    from roadgraph.io import load_pyg
    data = load_pyg(str(ev.OSM_ROOT), m.city).to(ev.DEVICE)
    enc = RoadGAT(num_cont=data.x.size(1),
                  num_highway=max(64, int(data.highway_id.max()) + 1)).to(ev.DEVICE)
    enc.load_state_dict(sd["road_enc"])
    enc.eval()
    with torch.no_grad():
        m.road_z = enc(data.x, data.highway_id, data.edge_index).detach()
    return True


def proxy_hits(m, lat, lon, pred):
    hits = []
    for i in range(WARM, len(lat)):
        if pred[i] < 0:
            continue
        q = m.ci.query(lat[i], lon[i], radius_m=50.0, k=10)
        dp = np.asarray(q["d_perp_m"], np.float64)
        sid = np.asarray(q["segment_id"], np.int64)
        if len(sid) == 0:
            continue
        sel = dp[sid == pred[i]]
        pred_dp = sel.min() if len(sel) else np.inf
        hits.append(pred_dp <= dp.min() + 5.0)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-traj", type=int, default=100)
    ap.add_argument("--source", default="silesia")
    ap.add_argument("--train-trajs", type=int, default=7000,
                    help="must match the training launch --limit-trajs")
    ap.add_argument("--ckpt", default="ckpt/stage2r_silesia.pt",
                    help="retrained ckpt path to compare against zero-shot")
    args = ap.parse_args()

    df = load_source_df(str(BASE / "data" / "processed"), args.source, None)
    uniq = np.asarray(df["traj_id"].unique())
    held = set(uniq[args.train_trajs:])
    print(f"[w1] source={args.source} trajs total={len(uniq)}  held-out={len(held)}  "
          f"scoring first {args.max_traj} held-out with >= {WARM + 2} fixes  ckpt={args.ckpt}")

    trajs = []
    for tid, g in df[df["traj_id"].isin(held)].groupby("traj_id", sort=False):
        if len(g) >= WARM + 2:
            g = g.iloc[:MAX_FIX]
            trajs.append((g["lat"].to_numpy(), g["lon"].to_numpy(),
                          g["t"].to_numpy(np.int64)))
        if len(trajs) >= args.max_traj:
            break

    ckpts = {"zero-shot": "final", "retrained": args.ckpt}
    modes = ("base", "hyb+topo")
    acc = {}
    for name, ck in ckpts.items():
        m = Matcher("silesia", ckpt=ck)
        if name == "retrained" and _maybe_load_unfrozen_encoder(m, ck):
            print(f"  [{name}] road_enc found in ckpt -- using UNFROZEN encoder weights")
        for mode in modes:
            t0, hits = time.time(), []
            for lat, lon, t in trajs:
                pred = m.match_online(lat, lon, t, mode=mode)
                hits.extend(proxy_hits(m, lat, lon, pred))
            acc[(name, mode)] = np.mean(hits)
            print(f"  {name:9s}  {mode:8s}  match@1 {np.mean(hits):.4f}  "
                  f"(n={len(hits)})  [{time.time()-t0:.0f}s]")

    print()
    zs, rt = acc[("zero-shot", "base")], acc[("retrained", "base")]
    print(f"W1 proxy: retrained base {rt:.4f} vs zero-shot base {zs:.4f}  "
          f"-> {'PASS' if rt > zs else 'FAIL'}")


if __name__ == "__main__":
    main()
