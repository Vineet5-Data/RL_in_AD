"""Offline NK-HMM Viterbi relabeling of the road-CE training targets (IL round-3 H1).

WHY. Training is behavior cloning: the road head is fit to a per-point target
softmax(-d_perp^2/2sigma^2) over the K cached candidates. That target is
TOPOLOGY-BLIND -- it is the nearest road to this one fix, computed with no
reference to where the vehicle was a second earlier or goes a second later.
Scheduled sampling (already shipped) makes the student visit its own states but
still scores them against that same blind target, which is only half of DAgger
and is a provably inconsistent objective (Huszar, arXiv:1511.05101). The other
half -- relabel the visited states with an EXPERT's action -- is what this
script supplies: a full-trajectory Newson-Krumm HMM Viterbi decode, which is
topology-consistent by construction and scores 0.8447 tolerant Hit@1 offline
against the student's 0.8122 online.

WHY NK-ONLY AND NOT THE STRONGER HYBRID EXPERT. The hybrid Viterbi (0.8655) is
better but its emission is the student's own road head, so relabeling with it
would feed the student its own beliefs -- a self-confirmation loop, and it
breaks the exogenous-expert assumption DAgger's O(T*eps) bound rests on (Ross &
Bagnell, AISTATS 2011). NK emission is pure cached geometry + graph topology and
never touches the checkpoint, so this arm runs first. The hybrid arm is gated on
this one showing a lift (project_summary.md, IL round-3 ranked experiments).

OUTPUT. One int64 .npy per (city, source), length = len(df), holding the chosen
SEGMENT ID per fix and -1 where the expert produced no label. Segment id, not
candidate slot, because dataset Change 7A permutes the K slots on every
__getitem__ call -- a slot index would be stale the moment it was written.
Consumed via `--viterbi-dir` + `--viterbi-alpha` in stage2r (see
stage2.road_soft_target).

CPU-only and GPU-free by construction (no checkpoint is loaded), so it belongs
in a CPU Kaggle kernel like the candidate-cache precompute, not a metered GPU
session. Resumable: work is sharded, finished shards are skipped on rerun.

Usage:
  python -m training.relabel_viterbi --processed-root ... --osm-root ... \
      --city porto --source porto --limit-trajs 1658437 \
      --cache-dir ckpt/cache/porto__porto --out-dir viterbi_labels --workers 4
  python -m training.relabel_viterbi --self-check --processed-root ... --osm-root ...
"""

from __future__ import annotations

import pyarrow.parquet  # noqa: F401  must precede torch (Windows segfault guard)

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from hmm_baseline.baseline_hmm import (LOGFLOOR, SIGMA, RouteDist, build_road_digraph,
                                            log_softmax_rows, split_pieces, viterbi)
except ImportError:  # flat layout (Kaggle CODE_ROOT on sys.path)
    from baseline_hmm import (LOGFLOOR, SIGMA, RouteDist, build_road_digraph,  # type: ignore
                               log_softmax_rows, split_pieces, viterbi)

# baseline_hmm hard-codes OSM_ROOT to the local checkout; build_road_digraph reads
# it off the module, so point it at whatever root this run was given.
try:
    import hmm_baseline.baseline_hmm as _bh
except ImportError:  # flat layout
    import baseline_hmm as _bh  # type: ignore


_W: dict = {}   # per-worker state (fork-inherited arrays + that worker's Dijkstra memo)


def _load_arrays(processed_root, osm_root, city, source, limit_trajs, cache_path):
    """df-row-aligned candidate arrays, exactly as the trainer sees them.

    Reads `_cand` straight off the dataset instead of going through
    __getitem__, which would apply Change 7A's per-call slot permutation --
    labels must be written against the UNPERMUTED cache rows."""
    from dataset.trajectories import load_source_df, TrajectoryGraphDataset
    from dataset.config import SequenceConfig, RetrievalConfig

    df = load_source_df(processed_root, source, limit_trajs)
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"candidate cache missing: {cache_path}\nRelabeling reuses the trainer's own "
            f"cache; it must NOT silently fall back to a from-scratch STRtree precompute "
            f"(that is the documented OOM failure mode, project_summary.md).")
    # indices={} is safe: the cache exists, so _precompute_candidates is never called.
    ds = TrajectoryGraphDataset(df, {}, SequenceConfig(), RetrievalConfig(),
                                cache_path=str(cache_path))
    return df, ds


def _init_worker(shared: dict):
    _W.update(shared)
    _W["rd"] = RouteDist(_W["G"])


def _relabel_chunk(task):
    """Decode one contiguous block of trajectories. Returns (rows, segs) so the
    caller owns all file writes -- workers stay side-effect free."""
    chunk_id, groups = task
    seg_all = _W["seg"]; dp_all = _W["dperp"]; cm_all = _W["mask"]
    lat_all = _W["lat"]; lon_all = _W["lon"]; dt_all = _W["dt"]
    rd, beta, sigma = _W["rd"], _W["beta"], _W["sigma"]

    rows_out, segs_out = [], []
    for (s, e) in groups:
        # ponytail: bound RouteDist's per-source Dijkstra memo by dropping it whole
        # when it gets big. It is pure cache (correctness unaffected), but it grows
        # with geographic coverage and each forked worker keeps its own copy -- the
        # calibration's 300-trajectory sample never reached the size a full-city pass
        # will. Blunt clear, not an LRU: swap in an LRU only if profiling shows the
        # rebuild cost actually matters.
        if len(rd.ss) > 20_000:
            rd.ss.clear()
        seg = np.asarray(seg_all[s:e]); dp = np.asarray(dp_all[s:e], dtype=np.float64)
        cm = np.asarray(cm_all[s:e])
        valid = (cm > 0.5) & (seg >= 0)
        with np.errstate(over="ignore"):
            logit = np.where(valid, -(dp ** 2) / (2 * sigma ** 2), -np.inf)
        log_emit = log_softmax_rows(logit)
        log_emit[~valid] = -np.inf
        valid_fix = valid.any(1)
        if not valid_fix.any():
            continue
        lat = lat_all[s:e]; lon = lon_all[s:e]; dt = dt_all[s:e]
        for piece in split_pieces(valid_fix, dt):
            idx = np.asarray(piece)
            coords = np.stack([lat[idx], lon[idx]], axis=1)
            path = viterbi(log_emit[idx], seg[idx], valid[idx], coords, rd, beta)
            for pos, t in enumerate(idx):
                rows_out.append(s + int(t))
                segs_out.append(int(seg[t, path[pos]]))
    return chunk_id, np.asarray(rows_out, dtype=np.int64), np.asarray(segs_out, dtype=np.int64)


def relabel(processed_root, osm_root, city, source, limit_trajs, cache_path, out_path,
            beta=20.0, sigma=SIGMA, workers=0, shard_dir=None, shard_size=2000,
            max_groups=None, max_hours=None):
    _bh.OSM_ROOT = Path(osm_root)
    t0 = time.time()
    df, ds = _load_arrays(processed_root, osm_root, city, source, limit_trajs, cache_path)
    groups = ds.groups if max_groups is None else ds.groups[:max_groups]
    n = len(df)
    print(f"[relabel] {city}:{source}  {len(groups)} trajs  {n:,} fixes  beta={beta} sigma={sigma}",
          flush=True)

    G = build_road_digraph(city)
    print(f"[relabel] road digraph: {G.number_of_nodes()} segs, {G.number_of_edges()} edges "
          f"({time.time() - t0:.0f}s)", flush=True)

    shared = {"seg": ds._cand["segment_id"], "dperp": ds._cand["d_perp_m"],
              "mask": ds._cand["mask"], "lat": df["lat"].to_numpy(),
              "lon": df["lon"].to_numpy(), "dt": df["dt"].to_numpy(),
              "G": G, "beta": beta, "sigma": sigma}

    tasks = [(i, groups[b:b + shard_size])
             for i, b in enumerate(range(0, len(groups), shard_size))]
    shard_dir = Path(shard_dir) if shard_dir else Path(out_path).with_suffix(".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (shard_dir / f"shard_{t[0]:05d}.npz").exists()]
    print(f"[relabel] {len(tasks)} shards, {len(todo)} to do "
          f"({len(tasks) - len(todo)} already on disk)", flush=True)

    def _save(chunk_id, rows, segs):
        p = shard_dir / f"shard_{chunk_id:05d}.npz"
        tmp = p.with_suffix(".npz.tmp")
        # write through a handle: np.savez APPENDS '.npz' to any path not already
        # ending in it, so passing `tmp` directly would produce shard.npz.tmp.npz
        # and leave the rename below pointing at a file that was never created.
        with open(tmp, "wb") as fh:
            np.savez(fh, rows=rows, segs=segs)
        tmp.replace(p)          # atomic: a half-written shard never looks done

    stopped_early = False
    if workers and workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(shared,)) as pool:
            for k, (cid, rows, segs) in enumerate(pool.imap_unordered(_relabel_chunk, todo), 1):
                _save(cid, rows, segs)
                print(f"[relabel] shard {k}/{len(todo)} (id {cid}) {len(rows)} fixes "
                      f"{time.time() - t0:.0f}s", flush=True)
                if max_hours and (time.time() - t0) / 3600.0 > max_hours:
                    print(f"[relabel] --max-hours {max_hours} hit, stopping "
                          f"(rerun to resume from shards)", flush=True)
                    pool.terminate(); stopped_early = True
                    break
    else:
        _init_worker(shared)
        for k, task in enumerate(todo, 1):
            cid, rows, segs = _relabel_chunk(task)
            _save(cid, rows, segs)
            print(f"[relabel] shard {k}/{len(todo)} (id {cid}) {len(rows)} fixes "
                  f"{time.time() - t0:.0f}s", flush=True)
            if max_hours and (time.time() - t0) / 3600.0 > max_hours:
                print(f"[relabel] --max-hours {max_hours} hit, stopping "
                      f"(rerun to resume from shards)", flush=True)
                stopped_early = True
                break

    done = sorted(shard_dir.glob("shard_*.npz"))
    if len(done) < len(tasks):
        print(f"[relabel] INCOMPLETE: {len(done)}/{len(tasks)} shards -- not writing "
              f"{out_path} (a short label file would be rejected by the loader anyway)",
              flush=True)
        return None
    out = np.full(n, -1, dtype=np.int64)
    for p in done:
        z = np.load(p)
        out[z["rows"]] = z["segs"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), out)
    _report(out, ds, n, out_path, time.time() - t0, stopped_early)
    return out


def _report(out, ds, n, out_path, secs, stopped_early):
    """The go/no-go number is `differs`: if the expert agrees with the geometric
    target on ~every fix, relabeling is a no-op and the training arm is pointless
    -- better to learn that here than after a GPU session."""
    labeled = out >= 0
    seg = np.asarray(ds._cand["segment_id"][:n])
    dp = np.asarray(ds._cand["d_perp_m"][:n], dtype=np.float64)
    cm = np.asarray(ds._cand["mask"][:n]) > 0.5
    valid = cm & (seg >= 0)
    nearest = seg[np.arange(n), np.where(valid, dp, np.inf).argmin(1)]
    inrange = labeled & valid.any(1)
    differs = float((out[inrange] != nearest[inrange]).mean()) if inrange.any() else 0.0
    # every emitted label must be one of that fix's own candidates -- catches any
    # slot/row misalignment, the one bug class that would silently poison training
    hit = (seg == out[:, None]) & valid
    assert bool(hit[labeled].any(1).all()), "label not among its own fix's candidates"
    print(f"[relabel] wrote {out_path}  labeled {labeled.mean():.1%} of {n:,} fixes  "
          f"differs-from-nearest {differs:.2%}  ({secs / 60:.1f} min"
          f"{', EARLY STOP' if stopped_early else ''})", flush=True)
    if differs < 0.005:
        print("[relabel] WARNING: expert almost never disagrees with the geometric "
              "target -- relabeling would be a near-no-op, do NOT spend a training "
              "session on this arm without investigating first.", flush=True)


def _self_check():
    """Alignment check on synthetic data: a 3-segment path where the geometric
    nearest label is deliberately wrong at one fix (a parallel road is closer)
    and only topology can recover it."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_weighted_edges_from([(0, 1, 50.0), (1, 2, 50.0)])   # true route 0->1->2
    T = 3
    seg = np.array([[0, 9], [1, 9], [2, 9]], dtype=np.int64)   # 9 = decoy parallel road
    dperp = np.array([[3.0, 20.0], [12.0, 4.0], [3.0, 20.0]])  # fix 1 is nearer the decoy
    mask = np.ones((T, 2), dtype=np.float32)
    lat = np.array([41.1500, 41.1504, 41.1508]); lon = np.full(T, -8.61)
    _W.clear()
    _init_worker({"seg": seg, "dperp": dperp, "mask": mask, "lat": lat, "lon": lon,
                  "dt": np.array([0.0, 1.0, 1.0]), "G": G, "beta": 20.0, "sigma": SIGMA})
    _, rows, segs = _relabel_chunk((0, [(0, T)]))
    assert rows.tolist() == [0, 1, 2], rows
    nearest = seg[np.arange(T), dperp.argmin(1)]
    assert nearest.tolist() == [0, 9, 2], nearest        # geometry alone gets fix 1 wrong
    assert segs.tolist() == [0, 1, 2], segs              # topology recovers it
    print("[selfcheck] OK: Viterbi overrode the geometric label at the decoy fix "
          f"(nearest={nearest.tolist()} -> viterbi={segs.tolist()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root"); ap.add_argument("--osm-root")
    ap.add_argument("--city"); ap.add_argument("--source")
    ap.add_argument("--limit-trajs", type=int, default=None)
    ap.add_argument("--cache-path", help="the trainer's own candidate cache .npz for this city:source")
    ap.add_argument("--out", help="output .npy of per-fix expert segment ids")
    ap.add_argument("--shard-dir", default=None)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--beta", type=float, default=20.0, help="NK transition beta (20 = tuned best)")
    ap.add_argument("--sigma", type=float, default=SIGMA, help="NK emission sigma (m)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-groups", type=int, default=None, help="cap trajectories (calibration runs)")
    ap.add_argument("--max-hours", type=float, default=None, help="stop early; rerun resumes")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check(); return
    for req in ("processed_root", "osm_root", "city", "source", "cache_path", "out"):
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required")
    relabel(a.processed_root, a.osm_root, a.city, a.source, a.limit_trajs, a.cache_path,
            a.out, beta=a.beta, sigma=a.sigma, workers=a.workers, shard_dir=a.shard_dir,
            shard_size=a.shard_size, max_groups=a.max_groups, max_hours=a.max_hours)


if __name__ == "__main__":
    main()
