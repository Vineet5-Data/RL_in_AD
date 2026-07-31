"""Run-3 pre-flight (stage2_correction.md line 504): z1 cross-trajectory variance check.

Load one real batch of different trajectories, run frozen Stage-1
(apply_mask=False), confirm z1 varies meaningfully ACROSS trajectories
(pairwise distances), not just over time within one. Near-constant -> STOP,
Stage-1 audit instead of Run 3.

Run: cd .worktrees/Stage2_kaggle && python preflight_z1_variance.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / ".worktrees" / "kaggle"))
sys.path.insert(0, str(BASE / ".worktrees" / "data-preprocess"))

import pyarrow.parquet  # noqa: F401  must precede torch
import torch

from training.stage1 import build_loader, _road_embeddings

device = "cpu"
osm_root = str(BASE / "data" / "osm")
processed_root = str(BASE / "data" / "processed")
stage0_ckpt = str(BASE / ".worktrees" / "kaggle" / "ckpt" / "stage0_porto.pt")
stage1_ckpt = str(BASE / ".worktrees" / "kaggle" / "ckpt" / "stage1_porto.pt")

road_z = _road_embeddings(osm_root, "porto", stage0_ckpt, device)

from models.gps_encoder import Stage1Encoder
enc = Stage1Encoder(road_dim=road_z.shape[-1]).to(device)
enc.load_state_dict(torch.load(stage1_ckpt, map_location=device))
enc.eval()

_, loader = build_loader(processed_root, osm_root, "porto", "porto",
                          limit_trajs=200_000, batch=16, workers=0,
                          cache_dir=str(BASE / ".worktrees" / "kaggle" / "ckpt" / "cache_test"))
batch = next(iter(loader))

with torch.no_grad():
    z, cand, coords, mgm_mask, cand_pad = enc.forward(batch, road_z, apply_mask=False)
    valid = batch["seq_mask"] > 0.5
    # per-trajectory summary: mean over valid timesteps
    z_traj = (z * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)  # [B, d]

    b = z_traj.shape[0]
    dists = torch.cdist(z_traj, z_traj)  # [B, B]
    off_diag = dists[~torch.eye(b, dtype=torch.bool)]
    within_std = (z * valid.unsqueeze(-1)).std(dim=1).mean()  # avg over-time std within traj

    print(f"batch size: {b}")
    print(f"cross-traj pairwise dist: mean={off_diag.mean():.4f} std={off_diag.std():.4f} "
          f"min={off_diag.min():.4f} max={off_diag.max():.4f}")
    print(f"within-traj over-time std (avg): {within_std:.4f}")
    ratio = off_diag.mean() / within_std.clamp(min=1e-8)
    print(f"cross/within ratio: {ratio:.2f}")
    if off_diag.mean() < 1e-3:
        print("RESULT: FAIL — z1 near-constant across trajectories. STOP Run 3, audit Stage-1.")
    else:
        print("RESULT: PASS — z1 varies meaningfully across trajectories. Proceed with Run 3.")
