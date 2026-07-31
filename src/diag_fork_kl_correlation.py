"""Diagnostic: does per-timestep active-latent-group fraction (kl_act) track
real road-fork ambiguity, or is low kl_act just "few forks in Porto"?

User hypothesis (2026-07-26): posterior collapse (low kl_act) happens because
GPS/road data is genuinely multimodal (parallel roads, forks -- a "multi-
discrete Gaussian" target) and the model can't/won't spend latent capacity to
represent it. Test: correlate per-(sample,timestep) active-group fraction
(same 1-nat free-bits definition training uses for kl_act) against per-
(sample,timestep) soft road-target entropy (_road_target_entropy -- the
entropy of the Gaussian-softmax-over-candidates target used for road CE; high
entropy = two+ candidates comparably likely = a real fork).

  - positive correlation  -> active capacity DOES track real ambiguity; low
    overall kl_act reflects genuine target structure (few forks), not a
    family-mismatch collapse.
  - flat/no correlation   -> extra latent capacity isn't being spent to
    resolve real ambiguity, which supports the multimodal-mismatch collapse
    story.

Uses the same held-out Porto split as eval_research2.py (trajs
[200000, 200500)) so this measures generalization behavior, not memorized
training-set behavior.

Run (from .worktrees/research2). CPU by default -- the Silesia encoder-
unfrozen job is training on GPU concurrently, keep this off the GPU:
  python diag_fork_kl_correlation.py --ckpt ckpt/stage2r_porto_run2xl_16x16_final.pt
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
FORK_THR = 0.6931     # ln(2) -- entropy of a clean 2-way tie
NONFORK_THR = 0.10    # near-single-candidate target


def _road_z(device):
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT
    data = load_pyg(str(OSM_ROOT), "porto").to(device)
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
def collect(rssm, heads, stage1, road_z, loader, device):
    from models.world_model2 import kl_categorical_pergroup, straight_through_sample
    from training.stage2r import extract_features_r, _road_target_entropy

    active_frac_all, tgt_ent_all = [], []
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        feats = extract_features_r(stage1, road_z, b)
        z1, road_embed, valid = feats["z1"], feats["road_embed"], feats["valid"]
        has_cand, cand_mask, cand_dperp = feats["has_cand"], feats["cand_mask"], feats["cand_d_perp_m"]
        B, L = valid.shape
        h, z = rssm.initial(B, device)
        for t in range(L):
            if t > 0:
                h = rssm.step(h, z, road_embed[:, t - 1])
            prior_logits = rssm.prior(h)
            post_logits = rssm.posterior(h, z1[:, t])
            z = straight_through_sample(post_logits).reshape(B, -1)

            mask = valid[:, t] & has_cand[:, t]
            if not mask.any():
                continue
            kl_dyn_pg = kl_categorical_pergroup(post_logits, prior_logits)   # [B, groups]
            active_frac = (kl_dyn_pg > 1.0).float().mean(-1)                # [B]
            tgt_ent = _road_target_entropy(cand_dperp[:, t], cand_mask[:, t])  # [B]

            active_frac_all.append(active_frac[mask].cpu().numpy())
            tgt_ent_all.append(tgt_ent[mask].cpu().numpy())

    return np.concatenate(active_frac_all), np.concatenate(tgt_ent_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "stage2r_porto_run2xl_16x16_final.pt"))
    ap.add_argument("--device", default="cpu",
                    help="default cpu -- Silesia encoder-unfrozen job trains on GPU concurrently")
    a = ap.parse_args()
    device = a.device

    road_z = _road_z(device)
    stage1 = _stage1(road_z.size(1), device)
    loader = _loader()

    from training.stage3 import load_world_model
    rssm, heads = load_world_model(a.ckpt, road_z.size(1), device)
    print(f"[wm] {a.ckpt}  groups={rssm.groups} classes={rssm.classes}")

    active_frac, tgt_ent = collect(rssm, heads, stage1, road_z, loader, device)
    n = len(tgt_ent)
    r = float(np.corrcoef(active_frac, tgt_ent)[0, 1])

    fork, nonfork = tgt_ent >= FORK_THR, tgt_ent < NONFORK_THR
    af_fork, af_nonfork = active_frac[fork], active_frac[nonfork]

    print(f"\nn={n} samples  (fork tgt_ent>={FORK_THR:.3f}: {int(fork.sum())}   "
          f"nonfork tgt_ent<{NONFORK_THR:.2f}: {int(nonfork.sum())})")
    print(f"overall kl_act (active_frac mean): {active_frac.mean():.4f}")
    print(f"Pearson r(active_frac, tgt_ent): {r:.4f}")
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
        "positive correlation -- active latent capacity DOES track real fork "
        "ambiguity, kl_act's low value reflects genuine target structure (few "
        "forks), not a family-mismatch collapse."
        if r > 0.15 else
        "weak/no correlation -- active-group usage is NOT tracking real fork "
        "ambiguity. Low kl_act is not explained by 'few forks'; if forks exist "
        "but the latent doesn't activate for them, that supports the "
        "multimodal-mismatch collapse hypothesis."
    ))


if __name__ == "__main__":
    main()
