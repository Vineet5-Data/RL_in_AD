"""Local-CPU reward-weight tuning for the Stage-3 actor.

Root-cause context (2026-07-25 diagnosis): imagined rollouts in stage3.py
score every candidate through the FROZEN heads.reward head (training/rewards.py
docstring: "stage3 never computed directly -- rewards there come from the
trained reward head"). heads.reward was regressed once, in stage2r, against
candidate_rewards() computed with whatever RewardConfig.weights existed at
that time. So simply editing dataset/config.py's weights tuple and re-running
stage3 against the frozen WM checkpoint -- the fix the project record proposed
-- is a NO-OP: stage3 never calls candidate_rewards() itself, and the actor
never sees the new weighting. The reward head must be refit first.

Pipeline per grid config (dist,head,oneway,speed,topo,smooth):
  1. refit heads.reward (deepcopy, unfreeze, short MSE regression against
     candidate_rewards(new_weights) over real teacher-forced sequences) --
     everything else (RSSM, road/gps/speed heads) stays frozen throughout.
  2. short stage3 actor-critic (imagine_and_losses, reused verbatim) against
     that refit reward head.
  3. small held-out match@1 proxy eval (eval_research2.eval_checkpoint reused
     verbatim) to rank configs fast.
Winner gets a full 500-traj confirm eval. CPU-only, short budgets -- this is
a proxy sweep for ranking, not a substitute for a full-budget run.

Run (from src/evals): python tune_reward_weights.py
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
BASE = Path(os.environ.get("AE_REPO_ROOT", SRC.parent))

import pyarrow.parquet  # noqa: E402  must precede torch
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

DEVICE = "cpu"  # user decision 2026-07-25: local CPU tuning, not Kaggle GPU

from training.stage3 import load_world_model, posterior_states, imagine_and_losses  # noqa: E402
from training.stage2r import extract_features_r, _successor_table  # noqa: E402
from training.stage2 import _stage1_encoder, LengthCapped  # noqa: E402
from training.rewards import candidate_rewards  # noqa: E402
from models.world_model2 import RoadScorer, Critic, symlog  # noqa: E402
from dataset.config import RewardConfig  # noqa: E402
from dataset.trajectories import load_source_df, TrajectoryGraphDataset, collate_fn  # noqa: E402
from dataset.candidates import CandidateIndex  # noqa: E402
from dataset.config import SequenceConfig, RetrievalConfig  # noqa: E402

import eval_research2 as ev  # noqa: E402  reuse _road_z / _stage1 / eval_checkpoint
ev.DEVICE = DEVICE  # force CPU regardless of local CUDA availability (user decision 2026-07-25)

CKPT_ROOT = Path(os.environ.get("AE_CKPT_ROOT", BASE / "ckpt"))
DATA_ROOT = Path(os.environ.get("AE_DATA_ROOT", BASE / "data"))
CKPT_DIR = CKPT_ROOT
WM_CKPT = CKPT_DIR / "stage2r_porto.pt"
PROC_ROOT = DATA_ROOT / "processed"
OSM_ROOT = DATA_ROOT / "osm"
CACHE_DIR = CKPT_ROOT / "cache_test"
TRAIN_CACHE = CACHE_DIR / "porto_n20000_r50_k10.npz"   # exact-size, already precomputed

TRAIN_TRAJS = 20_000
PROXY_EVAL_TRAJS = 80     # held-out [200000, 200080) -- fast ranking pass
FINAL_EVAL_TRAJS = 500    # full held-out confirm, winner only (reuses ev._loader)
BATCH = 16
SEQ_LEN = 64
REFIT_STEPS = 250
ACTOR_STEPS = 250
HORIZON = 15

# lambda order: [dist, head, oneway, speed, topo, smooth]  (dataset/config.py:29)
# baseline is the shipped config; the rest cut topo/head (over-weighted per the
# 2026-07-25 diagnosis: r_topo alone = 20m of distance-equivalent budget against
# a 50m candidate window) either in isolation (attribution) or combined, with an
# optional relative dist boost.
GRID = {
    "baseline":        (1.0, 0.5, 1.0, 0.3, 2.0, 0.2),
    "topo_only":       (1.0, 0.5, 1.0, 0.3, 0.5, 0.2),
    "head_only":       (1.0, 0.2, 1.0, 0.3, 2.0, 0.2),
    "moderate":        (1.0, 0.3, 1.0, 0.3, 1.0, 0.2),
    "aggressive":      (1.0, 0.2, 1.0, 0.3, 0.5, 0.2),
    "dist_boost_mod":  (1.5, 0.3, 1.0, 0.3, 1.0, 0.2),
    "dist_boost_agg":  (2.0, 0.2, 1.0, 0.3, 0.5, 0.2),
}


def train_loader():
    df = load_source_df(str(PROC_ROOT), "porto", TRAIN_TRAJS)
    ci = CandidateIndex.from_city(str(OSM_ROOT), "porto")
    ds = TrajectoryGraphDataset(df, {"porto": ci}, SequenceConfig(),
                                 RetrievalConfig(), cache_path=str(TRAIN_CACHE))
    ds = LengthCapped(ds, SEQ_LEN)
    return DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, num_workers=0)


def proxy_eval_loader():
    tbl = pyarrow.parquet.read_table(str(PROC_ROOT / "porto" / "part-000.parquet"),
                                      columns=["traj_id"])
    import pandas as pd
    uniq = pd.Series(tbl["traj_id"].to_pandas().unique())
    held = set(uniq.iloc[200_000: 200_000 + PROXY_EVAL_TRAJS].tolist())
    del tbl
    tbl_full = pyarrow.parquet.read_table(str(PROC_ROOT / "porto" / "part-000.parquet"))
    df = tbl_full.to_pandas(); del tbl_full
    df = df[df["traj_id"].isin(held)].reset_index(drop=True)
    ci = CandidateIndex.from_city(str(OSM_ROOT), "porto")
    cache = CACHE_DIR / f"porto_holdout_n{PROXY_EVAL_TRAJS}_r50_k10.npz"
    ds = TrajectoryGraphDataset(df, {"porto": ci}, SequenceConfig(), RetrievalConfig(),
                                 cache_path=str(cache), perm_seed=0)
    return DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn, num_workers=0)


def refit_reward_head(rssm, heads, reward_cfg, succ, road_z, stage1, loader, steps):
    """Deepcopy+refit heads.reward only; RSSM and every other head stay frozen."""
    heads_cfg = copy.deepcopy(heads)
    for p in heads_cfg.reward.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(heads_cfg.reward.parameters(), lr=3e-4)
    it = iter(loader)
    for step in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        b = {k: v for k, v in b.items()}
        feats = extract_features_r(stage1, road_z, b)
        with torch.no_grad():
            H_all, Z_all = posterior_states(rssm, feats, road_z)
        valid, has_cand = feats["valid"], feats["has_cand"]
        cand_seg, cand_head = feats["cand_segment_id"], feats["cand_heading_deg"]
        cand_oneway, cand_hw = feats["cand_oneway"], feats["cand_highway_id"]
        cand_mask, cand_dperp, cand_kph = (feats["cand_mask"], feats["cand_d_perp_m"],
                                            feats["cand_speed_kph"])
        bearing, speed_mps, pos_idx = feats["bearing_deg"], feats["speed_mps"], feats["pos_idx"]
        L = valid.shape[1]
        prev_head_deg = None
        total_loss, n = torch.zeros(()), 0
        for t in range(L):
            hc = has_cand[:, t]
            if hc.any():
                prev = {}
                if t > 0:
                    prev = dict(prev_seg=cand_seg[:, t - 1].gather(
                                    -1, pos_idx[:, t - 1].unsqueeze(-1)).squeeze(-1),
                                prev_road_heading_deg=prev_head_deg,
                                prev_gps_bearing_deg=bearing[:, t - 1])
                with torch.no_grad():
                    r_tgt = candidate_rewards(reward_cfg, succ, cand_dperp[:, t], cand_head[:, t],
                                               cand_kph[:, t], cand_oneway[:, t], cand_seg[:, t],
                                               cand_mask[:, t], bearing[:, t], speed_mps[:, t], **prev)
                r_hat = heads_cfg.reward(H_all[:, t], Z_all[:, t], road_z, cand_seg[:, t],
                                          cand_head[:, t], cand_oneway[:, t], cand_hw[:, t],
                                          cand_mask[:, t])
                rmask = cand_mask[:, t] & hc.unsqueeze(-1) & valid[:, t].unsqueeze(-1)
                if rmask.any():
                    total_loss = total_loss + F.mse_loss(r_hat[rmask], symlog(r_tgt)[rmask],
                                                           reduction="sum")
                    n += int(rmask.sum())
            prev_head_deg = cand_head[:, t].gather(-1, pos_idx[:, t].unsqueeze(-1)).squeeze(-1)
        if n > 0:
            opt.zero_grad()
            (total_loss / n).backward()
            opt.step()
    heads_cfg.eval()
    for p in heads_cfg.reward.parameters():
        p.requires_grad_(False)
    return heads_cfg


def train_actor(rssm, heads_cfg, road_z, stage1, loader, steps):
    actor = RoadScorer(h_dim=rssm.h_dim, road_dim=road_z.size(1))
    actor.load_state_dict(heads_cfg.road.state_dict())
    critic = Critic(h_dim=rssm.h_dim)
    critic_tgt = copy.deepcopy(critic)
    for p in critic_tgt.parameters():
        p.requires_grad_(False)
    opt_a = torch.optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=1e-4)
    it = iter(loader)
    for step in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        feats = extract_features_r(stage1, road_z, b)
        out_l = imagine_and_losses(rssm, heads_cfg, actor, critic, critic_tgt, feats, road_z,
                                    horizon=HORIZON, gamma=0.99, lam=0.95, eta_entropy=1e-3)
        opt_a.zero_grad(); opt_c.zero_grad()
        (out_l["actor"] + out_l["critic"]).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
        opt_a.step(); opt_c.step()
        with torch.no_grad():
            for pt, po in zip(critic_tgt.parameters(), critic.parameters()):
                pt.lerp_(po, 0.02)
    actor.eval()
    return actor


def main():
    print(f"[tune] device={DEVICE} grid={list(GRID)} refit_steps={REFIT_STEPS} "
          f"actor_steps={ACTOR_STEPS} train_trajs={TRAIN_TRAJS} proxy_eval_trajs={PROXY_EVAL_TRAJS}")
    road_z = ev._road_z()
    stage1 = ev._stage1(road_z.size(1))
    rssm, heads = load_world_model(str(WM_CKPT), road_z.size(1), DEVICE)
    succ = _successor_table(str(OSM_ROOT), "porto", DEVICE)
    tloader = train_loader()
    ploader = proxy_eval_loader()

    results = {}
    for name, w in GRID.items():
        print(f"\n[tune] === {name} {w} ===", flush=True)
        reward_cfg = RewardConfig(weights=w)
        heads_cfg = refit_reward_head(rssm, heads, reward_cfg, succ, road_z, stage1, tloader,
                                       REFIT_STEPS)
        actor = train_actor(rssm, heads_cfg, road_z, stage1, tloader, ACTOR_STEPS)
        r = ev.eval_checkpoint(rssm, heads_cfg, actor, stage1, road_z, ploader)
        results[name] = r
        print(f"[tune] {name}: match@1={r['match_hit1']:.4f} hit@1={r['hit1']:.4f} "
              f"kl={r['kl']:.3f}", flush=True)
        torch.save({"actor": actor.state_dict(), "reward_head": heads_cfg.reward.state_dict(),
                    "weights": w}, CKPT_DIR / f"tune_{name}.pt")

    print("\n[tune] === ranked by match@1 (proxy, n={}) ===".format(PROXY_EVAL_TRAJS))
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["match_hit1"]):
        print(f"  {name:16s} match@1={r['match_hit1']:.4f}  hit@1={r['hit1']:.4f}  "
              f"weights={GRID[name]}")

    winner = max(results, key=lambda k: results[k]["match_hit1"])
    print(f"\n[tune] winner: {winner} {GRID[winner]} -- running full {FINAL_EVAL_TRAJS}-traj "
          f"confirm eval", flush=True)
    ev.N_TRAJS = FINAL_EVAL_TRAJS
    final_loader = ev._loader()
    ckpt = torch.load(CKPT_DIR / f"tune_{winner}.pt", map_location=DEVICE, weights_only=True)
    heads_final = copy.deepcopy(heads)
    heads_final.reward.load_state_dict(ckpt["reward_head"])
    actor_final = RoadScorer(h_dim=rssm.h_dim, road_dim=road_z.size(1))
    actor_final.load_state_dict(ckpt["actor"])
    actor_final.eval()
    r_final = ev.eval_checkpoint(rssm, heads_final, actor_final, stage1, road_z, final_loader)
    road_head_baseline = ev.eval_checkpoint(rssm, heads, heads.road, stage1, road_z, final_loader)
    print(f"[tune] FINAL {winner}: match@1={r_final['match_hit1']:.4f}  "
          f"(road-head baseline={road_head_baseline['match_hit1']:.4f}, "
          f"stage3-old-actor was 0.593)")


if __name__ == "__main__":
    main()
