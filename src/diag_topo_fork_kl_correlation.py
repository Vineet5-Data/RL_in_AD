"""Diagnostic: does per-timestep active-latent-group fraction (kl_act) track
real GRAPH-TOPOLOGY fork ambiguity (does the vehicle face an actual choice of
next road), as opposed to diag_fork_kl_correlation.py's GPS-candidate-crowding
proxy (which came back ~uncorrelated, r=-0.046 -- flagged there as possibly
the wrong operationalization of "multimodal").

Fork here = forward branching factor at the resolved (pseudo-label) current
road: total topological neighbors of that road segment, MINUS the segment we
arrived from (so a plain pass-through road scores 0-1, a real fork/junction
with 2+ legal next segments scores >=2). "Arrived from" membership uses the
same SuccessorTable.contains() the reward head already relies on (training/
rewards.py) -- no new topology logic invented, reuses the project's existing
adjacency test.

  - positive correlation  -> active capacity tracks real topological choice
    points; low overall kl_act reflects that most of a route has no real
    branching, not a family-mismatch collapse.
  - flat/no correlation   -> capacity isn't being spent on real junctions
    either, i.e. the null result from the GPS-crowding version wasn't just a
    bad proxy -- supports genuine under-utilization regardless of how
    "fork" is defined.

Same held-out Porto split as eval_research2.py / diag_fork_kl_correlation.py
(trajs [200000, 200500)). CPU by default -- keep off the GPU while the
Silesia encoder-unfrozen job trains concurrently.

Run (from .worktrees/research2):
  python diag_topo_fork_kl_correlation.py --ckpt ckpt/stage2r_porto_run2xl_16x16_final.pt
"""
import argparse
import os
import sys
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
HERE = Path(__file__).resolve().parent
DP = BASE / ".worktrees" / "data-preprocess"
sys.path = [str(HERE), str(DP), *[p for p in sys.path if p not in {str(HERE), str(DP)}]]

import pyarrow.parquet  # noqa: E402  must precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

PROC_ROOT = BASE / "data" / "processed"
OSM_ROOT = BASE / "data" / "osm"
BASE_CKPT_DIR = BASE / ".worktrees" / "kaggle" / "ckpt"
S0_CKPT = BASE_CKPT_DIR / "stage0_porto.pt"
S1_CKPT = BASE_CKPT_DIR / "stage1_porto.pt"
EVAL_CACHE = BASE_CKPT_DIR / "cache_test" / "porto_holdout_n500_r50_k10.npz"
PARQUET = PROC_ROOT / "porto" / "part-000.parquet"

SKIP_TRAJS = 200_000  # same held-out slice as eval_research2.py / HMM baseline
N_TRAJS = 500
BATCH_SIZE = 16


def _road_graph(device):
    from roadgraph.io import load_pyg
    from training.rewards import SuccessorTable
    data = load_pyg(str(OSM_ROOT), "porto").to(device)
    num_nodes = int(data.num_nodes)
    out_degree = torch.bincount(data.edge_index[0].long(), minlength=num_nodes).to(device)
    succ = SuccessorTable(data.edge_index, num_nodes, device)
    return data, out_degree, succ, num_nodes


def _road_z(data, device):
    from models.road_encoder import RoadGAT
    enc = RoadGAT(num_cont=data.x.size(1),
                  num_highway=max(64, int(data.highway_id.max()) + 1)).to(device)
    enc.load_state_dict(torch.load(str(S0_CKPT), map_location=device, weights_only=True))
    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.highway_id, data.edge_index)
    return z.detach()


def _stage1(road_dim, device):
    from models.gps_encoder import Stage1Encoder
    m = Stage1Encoder(road_dim=road_dim).to(device)
    m.load_state_dict(torch.load(str(S1_CKPT), map_location=device, weights_only=True))
    m.eval()
    return m


def _loader():
    import pandas as pd
    from dataset.candidates import CandidateIndex
    from dataset.trajectories import TrajectoryGraphDataset, collate_fn
    from dataset.config import SequenceConfig, RetrievalConfig

    tbl = pyarrow.parquet.read_table(str(PARQUET), columns=["traj_id"])
    uniq = pd.Series(tbl["traj_id"].to_pandas().unique())
    held = set(uniq.iloc[SKIP_TRAJS: SKIP_TRAJS + N_TRAJS].tolist())
    del tbl
    tbl_full = pyarrow.parquet.read_table(str(PARQUET))
    df = tbl_full.to_pandas(); del tbl_full
    df = df[df["traj_id"].isin(held)].reset_index(drop=True)
    ci = CandidateIndex.from_city(str(OSM_ROOT), "porto")
    ds = TrajectoryGraphDataset(df, {"porto": ci},
                                SequenceConfig(), RetrievalConfig(),
                                cache_path=str(EVAL_CACHE), perm_seed=0)
    print(f"[data]  {len(ds)} trajs  {len(df):,} fixes")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, num_workers=0)


@torch.no_grad()
def collect(rssm, heads, stage1, road_z, out_degree, succ, num_nodes, loader, device):
    from models.world_model2 import kl_categorical_pergroup, straight_through_sample
    from training.stage2r import extract_features_r

    active_frac_all, branch_all = [], []
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        feats = extract_features_r(stage1, road_z, b)
        z1, road_embed, valid = feats["z1"], feats["road_embed"], feats["valid"]
        has_cand, cand_seg, pos_idx = feats["has_cand"], feats["cand_segment_id"], feats["pos_idx"]
        B, L = valid.shape
        h, z = rssm.initial(B, device)
        prev_seg = None
        for t in range(L):
            if t > 0:
                h = rssm.step(h, z, road_embed[:, t - 1])
            prior_logits = rssm.prior(h)
            post_logits = rssm.posterior(h, z1[:, t])
            z = straight_through_sample(post_logits).reshape(B, -1)

            resolved_seg = cand_seg[:, t].gather(-1, pos_idx[:, t].unsqueeze(-1)).squeeze(-1)
            resolved_seg = resolved_seg.long().clamp(min=0, max=num_nodes - 1)

            mask = valid[:, t] & has_cand[:, t]
            if mask.any():
                kl_dyn_pg = kl_categorical_pergroup(post_logits, prior_logits)  # [B, groups]
                active_frac = (kl_dyn_pg > 1.0).float().mean(-1)               # [B]

                raw_deg = out_degree[resolved_seg].float()
                if prev_seg is not None:
                    came_from = succ.contains(resolved_seg, prev_seg).float()
                else:
                    came_from = torch.zeros(B, device=device)
                forward_branch = (raw_deg - came_from).clamp(min=0)

                active_frac_all.append(active_frac[mask].cpu().numpy())
                branch_all.append(forward_branch[mask].cpu().numpy())

            prev_seg = resolved_seg

    return np.concatenate(active_frac_all), np.concatenate(branch_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "stage2r_porto_run2xl_16x16_final.pt"))
    ap.add_argument("--device", default="cpu",
                    help="default cpu -- Silesia encoder-unfrozen job trains on GPU concurrently")
    a = ap.parse_args()
    device = a.device

    data, out_degree, succ, num_nodes = _road_graph(device)
    road_z = _road_z(data, device)
    stage1 = _stage1(road_z.size(1), device)
    loader = _loader()

    from training.stage3 import load_world_model
    rssm, heads = load_world_model(a.ckpt, road_z.size(1), device)
    print(f"[wm] {a.ckpt}  groups={rssm.groups} classes={rssm.classes}")
    print(f"[graph] {num_nodes} segments  mean out_degree {out_degree.float().mean():.2f}")

    active_frac, branch = collect(rssm, heads, stage1, road_z, out_degree, succ, num_nodes,
                                   loader, device)
    n = len(branch)
    r = float(np.corrcoef(active_frac, branch)[0, 1])

    uniq, counts = np.unique(np.minimum(branch, 4).astype(int), return_counts=True)
    hist = ", ".join(f"{('4+' if u == 4 else u)}:{c}" for u, c in zip(uniq, counts))

    fork, nonfork = branch >= 2, branch <= 1
    af_fork, af_nonfork = active_frac[fork], active_frac[nonfork]

    print(f"\nn={n} samples   forward_branch histogram (capped at 4+): {hist}")
    print(f"fork (>=2 forward options): {int(fork.sum())}   nonfork (<=1): {int(nonfork.sum())}")
    print(f"overall kl_act (active_frac mean): {active_frac.mean():.4f}")
    print(f"Pearson r(active_frac, forward_branch): {r:.4f}")
    if fork.sum() > 0 and nonfork.sum() > 0:
        print(f"active_frac | fork:    {af_fork.mean():.4f} +/- {af_fork.std():.4f}")
        print(f"active_frac | nonfork: {af_nonfork.mean():.4f} +/- {af_nonfork.std():.4f}")
        diff = af_fork.mean() - af_nonfork.mean()
        pooled_std = float(np.sqrt((af_fork.var() + af_nonfork.var()) / 2))
        cohens_d = diff / pooled_std if pooled_std > 0 else float("nan")
        print(f"diff: {diff:+.4f}   cohen's d: {cohens_d:.3f}")
    else:
        print("not enough samples in one bucket for fork/nonfork comparison")

    print("\nVerdict: " + (
        "positive correlation -- active latent capacity DOES track real "
        "topological junctions; low overall kl_act reflects that most of a "
        "route has no real branching, not a family-mismatch collapse."
        if r > 0.15 else
        "weak/no correlation -- active-group usage is NOT tracking real "
        "topological choice points either. Combined with the GPS-crowding "
        "result, this is not a proxy artifact: the model isn't spending "
        "latent capacity on genuine junctions specifically."
    ))


if __name__ == "__main__":
    main()
