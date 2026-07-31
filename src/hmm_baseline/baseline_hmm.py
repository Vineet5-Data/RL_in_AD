"""Newson-Krumm HMM Viterbi map-matching baseline (Track B, classical_baseline_spec.md B1/B1-P).

The HMM state space per fix is exactly the cached K<=10 candidates (mask-aware); no
new candidate search. Fix lat/lon / dt / speed come from the same parquet-backed
loader eval_stage2.py uses; the road digraph for route distances is read straight
from segments.parquet + line_graph_edges.parquet (NOT roadgraph/io.py::load_pyg,
which z-scores features and drops raw lengths -- see spec B0).

Emission variants (--emission):
  nk          Gaussian on cached d_perp, sigma=10m: p ~ exp(-d_perp^2/2sigma^2).
  s1-visible  Stage-1 forward, geo_mask all False; softmax over candidate logits.
  s1-masked   Stage-1 forward, geo_mask all True (d_perp zeroed) -> context-only.

Transition (Newson & Krumm 2009): p ~ (1/beta) exp(-|d_route - d_gc| / beta), with
d_gc = haversine between consecutive fixes, d_route = shortest path over the segment
digraph (Dijkstra, cutoff = 2*d_gc + 500m), unreachable -> probability floor 1e-12.

Decode: log-space Viterbi over <=10 states/fix (--predict swaps this for forward
filtering to t=T_warm then transition-only rollout -- the classical counterpart of
the RSSM prior rollout in eval_stage2.py).

Usage (from src/hmm_baseline):
  python baseline_hmm.py --emission nk --max-traj 5            # matching smoke test
  python baseline_hmm.py --emission nk --predict --max-traj 5  # prediction smoke test
"""

import os, sys, math, time, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
BASE = Path(os.environ.get("AE_REPO_ROOT", SRC.parent))

import pyarrow.parquet  # noqa: must precede torch
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F

DATA_ROOT = Path(os.environ.get("AE_DATA_ROOT", BASE / "data"))
CKPT_ROOT = Path(os.environ.get("AE_CKPT_ROOT", BASE / "ckpt"))
PROC_ROOT = DATA_ROOT / "processed"
OSM_ROOT  = DATA_ROOT / "osm"
CKPT_DIR  = CKPT_ROOT
S0_CKPT   = CKPT_DIR / "stage0_porto.pt"
S1_CKPT   = CKPT_DIR / "stage1_porto.pt"

SPLIT_CFG = {
    ("porto", "holdout"):  {"parquet": PROC_ROOT / "porto" / "part-000.parquet",
                             "cache":   CKPT_DIR / "cache_test" / "porto_holdout_n500_r50_k10.npz",
                             "skip": 200_000, "n": 500},
    ("porto", "dev"):      {"parquet": PROC_ROOT / "porto" / "part-000.parquet",
                             "cache":   CKPT_DIR / "cache_test" / "porto_dev_n50_r50_k10.npz",
                             "skip": 0, "n": 50},
    ("tdrive", "holdout"): {"parquet": PROC_ROOT / "tdrive" / "part-000.parquet",
                             "cache":   CKPT_DIR / "cache_test" / "tdrive_n500_r50_k10.npz",
                             "skip": 0, "n": 500},
    ("porto", "holdout30k"): {"parquet": PROC_ROOT / "porto" / "part-000.parquet",
                               "cache":   CKPT_DIR / "cache_test" / "porto_holdout_n30000_r50_k10.npz",
                               "skip": 200_000, "n": 30_000},
}

T_WARM       = 8
SIGMA        = 10.0            # GPS-noise scale for the NK Gaussian emission
FLOOR        = 1e-12
LOGFLOOR     = math.log(FLOOR)
PRED_HORIZONS = [1, 5, 15]
DEVICE       = "cpu"          # spec: CPU-only task


# -- road digraph & memoised route distance ------------------------------------

def build_road_digraph(city):
    """Weighted segment digraph: edge src->dst costs dst's length (cost to ENTER dst)."""
    seg = pd.read_parquet(OSM_ROOT / city / "segments.parquet", columns=["segment_id", "length_m"])
    length = dict(zip(seg["segment_id"].astype(int).tolist(), seg["length_m"].astype(float).tolist()))
    lg = pd.read_parquet(OSM_ROOT / city / "line_graph_edges.parquet", columns=["src", "dst"])
    src = lg["src"].to_numpy(); dst = lg["dst"].to_numpy()
    # ponytail: node-length weighting -- d_route ~= sum of lengths of segments entered
    # after seg_a. Coarser than true metric route distance (ignores partial-segment
    # offsets at each endpoint) but standard for an NK baseline; upgrade to exact
    # projected offsets only if the beta sweep proves it matters.
    w = np.fromiter((length.get(int(d), 0.0) for d in dst), dtype=float, count=len(dst))
    G = nx.DiGraph()
    G.add_weighted_edges_from(zip(src.tolist(), dst.tolist(), w.tolist()))
    return G


class RouteDist:
    """d_route(a,b) via single-source Dijkstra, memoised per source (superset of the
    per-pair memo the spec asks for). Reachable distance is cutoff-independent, so a
    cached source is reused whenever its stored cutoff >= the requested one."""

    def __init__(self, G):
        self.G = G
        self.ss = {}   # seg_a -> (dist_dict, cutoff_used)

    def dist(self, a, b, cutoff):
        if a == b:
            return 0.0
        if a not in self.G:
            return None
        cache = self.ss.get(a)
        if cache is None or cache[1] < cutoff:
            dd = nx.single_source_dijkstra_path_length(self.G, a, cutoff=cutoff, weight="weight")
            self.ss[a] = (dd, cutoff)
            cache = self.ss[a]
        return cache[0].get(b)   # None -> unreachable within cutoff


from preprocessing.geo import haversine_m  # noqa: E402  scalar args work fine (numpy ufuncs)


# -- log-space helpers (all handle -inf) ---------------------------------------

def log_softmax_rows(logit):
    """All-(-inf) rows (zero valid candidates for a fix) are forced to -inf output
    without going through log(0); those fixes are excluded downstream via valid_fix
    anyway, but this avoids the spurious divide-by-zero/invalid-subtract warning."""
    m = np.max(logit, axis=1, keepdims=True)
    finite_row = np.isfinite(m)
    m_safe = np.where(finite_row, m, 0.0)
    e = np.exp(logit - m_safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (logit - m_safe) - np.log(e.sum(1, keepdims=True))
    return np.where(finite_row, out, -np.inf)


def logsumexp0(M):
    """logsumexp over axis 0 of [I,J] -> [J], -inf columns stay -inf."""
    m = np.max(M, axis=0)
    finite = np.isfinite(m)
    out = np.full(M.shape[1], -np.inf)
    if finite.any():
        Mf = M[:, finite] - m[finite][None, :]
        out[finite] = m[finite] + np.log(np.exp(Mf).sum(0))
    return out


def norm_log(v):
    m = np.max(v)
    if not np.isfinite(m):
        return v
    return v - (m + math.log(np.exp(v - m).sum()))


# -- transition matrix for one fix step ----------------------------------------

def log_transition(seg_prev, valid_prev, seg_cur, valid_cur, d_gc, rd, beta):
    """[Kprev, Kcur] log NK transition. Invalid cur cols / invalid prev rows -> -inf;
    reachable-but-off-route or unreachable candidate pairs -> log(FLOOR)."""
    Kp, Kc = len(seg_prev), len(seg_cur)
    T = np.full((Kp, Kc), -np.inf)
    cutoff = 2.0 * d_gc + 500.0
    logbeta = math.log(beta)
    for i in range(Kp):
        if not valid_prev[i]:
            continue
        a = int(seg_prev[i])
        for j in range(Kc):
            if not valid_cur[j]:
                continue
            dr = rd.dist(a, int(seg_cur[j]), cutoff)
            if dr is None:
                T[i, j] = LOGFLOOR
            else:
                T[i, j] = max(-logbeta - abs(dr - d_gc) / beta, LOGFLOOR)
    return T


# -- Viterbi over one contiguous piece -----------------------------------------

def viterbi(log_emit, seg, valid, coords, rd, beta):
    """Returns the chosen candidate slot index per fix in the piece."""
    Tn, K = seg.shape
    delta = log_emit[0].copy()
    back = np.zeros((Tn, K), dtype=int)
    for t in range(1, Tn):
        d_gc = haversine_m(coords[t - 1, 0], coords[t - 1, 1], coords[t, 0], coords[t, 1])
        trans = log_transition(seg[t - 1], valid[t - 1], seg[t], valid[t], d_gc, rd, beta)
        scores = delta[:, None] + trans                 # [Kprev, Kcur]
        best_prev = np.argmax(scores, axis=0)           # [Kcur]
        delta = log_emit[t] + scores[best_prev, np.arange(K)]
        back[t] = best_prev
    last = int(np.argmax(delta))
    path = [last]
    for t in range(Tn - 1, 0, -1):
        last = int(back[t, last])
        path.append(last)
    path.reverse()
    return path


# -- emissions -----------------------------------------------------------------

def compute_log_emit(emission, batch, stage1, road_z):
    seg = batch["cand_segment_id"][0].numpy()
    if emission == "nk":
        dperp = batch["cand_d_perp_m"][0].numpy().astype(np.float64)
        cm = batch["cand_mask"][0].numpy()
        valid = (cm > 0.5) & (seg >= 0)
        logit = np.where(valid, -(dperp ** 2) / (2 * SIGMA ** 2), -np.inf)
        return log_softmax_rows(logit)
    # learned emissions: replicate the Stage-1 contrastive logits (compute_losses)
    L = batch["seq_mask"].shape[1]
    gm = torch.zeros(1, L, dtype=torch.bool) if emission == "s1-visible" else torch.ones(1, L, dtype=torch.bool)
    with torch.no_grad():
        z, cand, _, _, cand_pad = stage1.forward(batch, road_z, apply_mask=False, geo_mask=gm)
        zp = F.normalize(stage1.z_proj(z), dim=-1)
        cp = F.normalize(stage1.c_proj(cand), dim=-1)
        logits = (zp.unsqueeze(2) * cp).sum(-1) / stage1.temp
        logits = logits.masked_fill(cand_pad, -1e4)
        le = F.log_softmax(logits, dim=-1)[0].numpy().astype(np.float64)
    return le


# -- gap / validity splitting --------------------------------------------------

def split_pieces(valid_fix, dt):
    """Contiguous runs of fixes that (a) have >=1 candidate and (b) are not preceded
    by a dt > 3x median gap (spec B1 gap handling)."""
    L = len(valid_fix)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    pieces, cur = [], []
    for t in range(L):
        if not valid_fix[t]:
            if cur:
                pieces.append(cur); cur = []
            continue
        if cur and med > 0 and dt[t] > 3 * med:
            pieces.append(cur); cur = [t]
        else:
            cur.append(t)
    if cur:
        pieces.append(cur)
    return pieces


# -- model / data loading ------------------------------------------------------

def load_road_z(city):
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT
    data = load_pyg(str(OSM_ROOT), city)
    enc = RoadGAT(num_cont=data.x.size(1), num_highway=max(64, int(data.highway_id.max()) + 1))
    enc.load_state_dict(torch.load(str(S0_CKPT), map_location="cpu", weights_only=True))
    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.highway_id, data.edge_index)
    return z.detach()


def load_stage1(road_dim):
    from models.gps_encoder import Stage1Encoder
    m = Stage1Encoder(road_dim=road_dim)
    m.load_state_dict(torch.load(str(S1_CKPT), map_location="cpu", weights_only=True))
    m.eval()
    return m


def load_dataset(city, split, max_traj):
    cfg = SPLIT_CFG[(city, split)]
    tbl = pyarrow.parquet.read_table(str(cfg["parquet"]), columns=["traj_id"])
    uniq = pd.Series(tbl["traj_id"].to_pandas().unique())
    held = set(uniq.iloc[cfg["skip"]:cfg["skip"] + cfg["n"]].tolist())
    del tbl
    df = pyarrow.parquet.read_table(str(cfg["parquet"])).to_pandas()
    df = df[df["traj_id"].isin(held)].reset_index(drop=True)

    z = np.load(cfg["cache"])
    n_rows = int(z["n_rows"]); z.close()
    if n_rows != len(df):
        raise RuntimeError(f"cache stale: n_rows={n_rows} vs df={len(df)}; rebuild via build_porto_holdout_cache.py")

    from dataset.trajectories import TrajectoryGraphDataset, collate_fn
    from dataset.config import SequenceConfig, RetrievalConfig
    # indices={} is safe: the cache matches (n_rows,k) so _precompute_candidates
    # (the only consumer of `indices`) is never called -- avoids loading geopandas.
    ds = TrajectoryGraphDataset(df, {}, SequenceConfig(), RetrievalConfig(),
                                cache_path=str(cfg["cache"]), perm_seed=0)
    if max_traj is not None:
        ds.groups = ds.groups[:max_traj]
    print(f"[data]  {len(ds)} trajs  {len(df):,} fixes  (city={city})")
    return ds, collate_fn


# -- drivers -------------------------------------------------------------------

def run_matching(ds, collate_fn, emission, stage1, road_z, rd, beta, floor=False):
    """floor=True: per-fix argmax on the emission alone, no transitions/Viterbi --
    the best variant (iii) could do with zero help from the transition model
    (classical_baseline_spec.md next-step: quantifies transition-model lift)."""
    tol, hard, mhe = [], [], []
    for i in range(len(ds)):
        batch = collate_fn([ds[i]])
        lat = batch["lat"][0].numpy(); lon = batch["lon"][0].numpy()
        dt = batch["dt"][0].numpy()
        seg = batch["cand_segment_id"][0].numpy()
        dperp = batch["cand_d_perp_m"][0].numpy()
        cm = batch["cand_mask"][0].numpy()
        valid = (cm > 0.5) & (seg >= 0)

        log_emit = compute_log_emit(emission, batch, stage1, road_z)
        log_emit[~valid] = -np.inf
        valid_fix = valid.any(1)

        if floor:
            for t in np.where(valid_fix)[0]:
                s = int(np.argmax(log_emit[t]))
                dp = dperp[t]; vm = valid[t]
                best = float(dp[vm].min())
                chosen = float(dp[s])
                tol.append(chosen <= best + 5.0)
                hard.append(chosen <= best + 1e-6)
                mhe.append(chosen)
            continue

        for piece in split_pieces(valid_fix, dt):
            idx = np.asarray(piece)
            coords = np.stack([lat[idx], lon[idx]], axis=1)
            path = viterbi(log_emit[idx], seg[idx], valid[idx], coords, rd, beta)
            for pos, t in enumerate(idx):
                s = path[pos]
                dp = dperp[t]; vm = valid[t]
                best = float(dp[vm].min())
                chosen = float(dp[s])
                tol.append(chosen <= best + 5.0)
                hard.append(chosen <= best + 1e-6)
                mhe.append(chosen)
    return tol, hard, mhe


def run_predict(ds, collate_fn, emission, stage1, road_z, rd, beta):
    max_h = max(PRED_HORIZONS)
    need = T_WARM + max_h
    hits = {h: [] for h in PRED_HORIZONS}
    used = 0
    for i in range(len(ds)):
        batch = collate_fn([ds[i]])
        L = batch["seq_mask"].shape[1]
        if L < need:
            continue
        lat = batch["lat"][0].numpy(); lon = batch["lon"][0].numpy()
        dt = batch["dt"][0].numpy(); spd = batch["speed_mps"][0].numpy()
        seg = batch["cand_segment_id"][0].numpy()
        dperp = batch["cand_d_perp_m"][0].numpy()
        cm = batch["cand_mask"][0].numpy()
        valid = (cm > 0.5) & (seg >= 0)
        if not valid[:need].any(1).all():
            continue

        log_emit = compute_log_emit(emission, batch, stage1, road_z)
        log_emit[~valid] = -np.inf

        # forward filtering over fixes 0 .. T_WARM-1
        alpha = norm_log(log_emit[0].copy())
        for t in range(1, T_WARM):
            d_gc = haversine_m(lat[t - 1], lon[t - 1], lat[t], lon[t])
            trans = log_transition(seg[t - 1], valid[t - 1], seg[t], valid[t], d_gc, rd, beta)
            alpha = norm_log(logsumexp0(alpha[:, None] + trans) + log_emit[t])

        # transition-only rollout; future d_gc unavailable -> constant-velocity estimate
        # from the warm boundary: expected step displacement = speed_{T_warm} * dt_{T_warm}.
        # ponytail: single boundary speed*dt held constant over the horizon -- no future
        # fix is peeked at; upgrade to a per-step speed model only if it changes C4.
        d_gc_exp = float(max(spd[T_WARM], 0.0) * max(dt[T_WARM], 0.0))
        prev = T_WARM - 1
        cur_log = alpha
        for step in range(1, max_h + 1):
            cur_t = T_WARM - 1 + step
            trans = log_transition(seg[prev], valid[prev], seg[cur_t], valid[cur_t], d_gc_exp, rd, beta)
            cur_log = norm_log(logsumexp0(cur_log[:, None] + trans))
            prev = cur_t
            if step in PRED_HORIZONS:
                s = int(np.argmax(cur_log))
                dp = dperp[cur_t]; vm = valid[cur_t]
                best = float(dp[vm].min())
                hits[step].append(float(dp[s]) <= best + 5.0)
        used += 1
    return hits, used


def _mean(lst):
    return float(np.mean(lst)) if lst else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Newson-Krumm HMM Viterbi map-matching baseline")
    ap.add_argument("--emission", choices=["nk", "s1-visible", "s1-masked"], default="nk")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--city", choices=["porto", "tdrive"], default="porto")
    ap.add_argument("--split", choices=["holdout", "dev", "holdout30k"], default="holdout",
                     help="holdout = eval slice (spec-fixed per city); dev = 50-traj Porto "
                          "training-split slice for beta sweeps (porto only)")
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--floor", action="store_true",
                     help="matching only: per-fix argmax on the emission, no Viterbi/transitions")
    args = ap.parse_args()

    if (args.city, args.split) not in SPLIT_CFG:
        raise NotImplementedError(f"no split_cfg for city={args.city} split={args.split}")

    t0 = time.time()
    print(f"[hmm]  emission={args.emission}  beta={args.beta}  predict={args.predict}  "
          f"city={args.city}  split={args.split}  device={DEVICE}")
    ds, collate_fn = load_dataset(args.city, args.split, args.max_traj)

    stage1 = road_z = None
    if args.emission != "nk":
        road_z = load_road_z(args.city)
        stage1 = load_stage1(road_z.size(1))

    rd = RouteDist(build_road_digraph(args.city))

    if args.predict:
        hits, used = run_predict(ds, collate_fn, args.emission, stage1, road_z, rd, args.beta)
        print(f"\n[predict]  emission={args.emission}  beta={args.beta}  trajs_used={used}")
        for h in PRED_HORIZONS:
            print(f"  tolerant Hit@1 @ H={h:<2d} : {_mean(hits[h]):.4f}  (n={len(hits[h])})")
    else:
        tol, hard, mhe = run_matching(ds, collate_fn, args.emission, stage1, road_z, rd, args.beta,
                                       floor=args.floor)
        print(f"\n[match]  emission={args.emission}  beta={args.beta}  floor={args.floor}  fixes={len(tol)}")
        print(f"  tolerant Hit@1 : {_mean(tol):.4f}")
        print(f"  hard Hit@1     : {_mean(hard):.4f}")
        print(f"  MHE (m)        : {_mean(mhe):.4f}")

    print(f"[hmm]  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
