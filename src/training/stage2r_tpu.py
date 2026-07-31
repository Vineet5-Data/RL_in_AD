"""TPU/XLA-compatible port of research2/training/stage2r.py's wm_losses + train.

Why a separate file, not an in-place edit of stage2r.py: the original loss
loop uses Python-level `if tensor.any():` branching and boolean-mask indexing
(`x[mask]`) throughout. Both are fatal for torch_xla/TPU:
  - `if tensor.any():` forces a synchronous device->host transfer every
    timestep (XLA is lazy/async; a Python `if` on a tensor value blocks and
    materializes it) -- the classic "step-boundary sync" perf killer.
  - `x[mask]` produces a tensor whose shape depends on how many elements are
    True -- a genuinely dynamic shape. XLA compiles a fixed graph per unique
    shape it sees; dynamic shapes cause full recompilation on (close to)
    every step, which is far slower than not using the accelerator at all.

Fix: replace every "select valid rows, reduce" with "compute for all rows,
multiply by a 0/1 mask, reduce" (shape-preserving) and accumulate counts as
tensors, converting to Python numbers exactly once at the end of the whole
training step (one host sync per step, not one per masked-branch per
timestep -- ~4x fewer syncs than the original's 4 `.any()` gates x L
timesteps).

This module is import-safe without torch_xla installed (falls back to
cuda/cpu) so `_smoke_equivalence()` can validate correctness on a normal
machine before ever touching real TPU quota -- run it first.

Local dev import path: this file lives in the TPU_kaggle worktree but reads
research2 (models/, training/) and data-preprocess (dataset/, roadgraph/,
preprocessing/) from their sibling worktrees, same convention as every eval
script in research2 already uses. On Kaggle the flat-layout except branch
kicks in (this file gets staged alongside models/training/dataset/roadgraph
into one directory, same as upload_datasets.sh does for stage2r.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_BASE = _HERE.parents[3]  # .../AlphaEvolve_research (training/ -> TPU_kaggle/ -> .worktrees/ -> here)
_RESEARCH2 = _BASE / ".worktrees" / "research2"
_DATA_PREPROCESS = _BASE / ".worktrees" / "data-preprocess"
for _p in (_RESEARCH2, _DATA_PREPROCESS):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pyarrow.parquet  # noqa: F401  must load before torch (Windows segfault guard)
import torch
import torch.nn.functional as F

try:
    from models.world_model2 import (RSSM2, HeadsLight, straight_through_sample,
                                      kl_categorical_pergroup, symlog)
    from models.world_model import _mixed_probs
    from training.stage2 import (LengthCapped, _stage1_encoder, _soft_road_ce,
                                  extract_frozen_features)
    from training.stage1 import _road_embeddings, build_loader
    from training.rewards import SuccessorTable, candidate_rewards
    from training.stage2r import (extract_features_r, _road_target_entropy,
                                   _successor_table, _synthetic_feats, wm_losses,
                                   _save_ckpt)
except ImportError:  # flat layout (Kaggle CODE_ROOT on sys.path)
    from world_model2 import (RSSM2, HeadsLight, straight_through_sample,  # type: ignore
                               kl_categorical_pergroup, symlog)
    from world_model import _mixed_probs  # type: ignore
    from stage2 import (LengthCapped, _stage1_encoder, _soft_road_ce,  # type: ignore
                         extract_frozen_features)
    from stage1 import _road_embeddings, build_loader  # type: ignore
    from rewards import SuccessorTable, candidate_rewards  # type: ignore
    from stage2r import (extract_features_r, _road_target_entropy,  # type: ignore
                          _successor_table, _synthetic_feats, wm_losses, _save_ckpt)

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr  # xrt_world_size/get_ordinal removed torch_xla>=2.7 -- use xr.*
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def _is_master() -> bool:
    return (not HAS_XLA) or xr.global_ordinal() == 0


def _save_ckpt_tpu(path, rssm, heads, opt, sched, step, norm):
    """xm.save (not plain torch.save) on XLA tensors: it rendezvous-syncs all
    replicas and transfers to CPU correctly before the master writes. Plain
    torch.save on a live XLA-resident state_dict can silently write stale or
    incomplete data since XLA execution is lazy/async."""
    ckpt = {"rssm": rssm.state_dict(), "heads": heads.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
            "meta": {"norm": norm, "stoch_groups": rssm.groups, "stoch_classes": rssm.classes}}
    if HAS_XLA:
        xm.save(ckpt, path, master_only=True)
    else:
        torch.save(ckpt, path)


def wm_losses_tpu(rssm: "RSSM2", heads: "HeadsLight", feats: dict, road_z: torch.Tensor,
                   succ: "SuccessorTable", reward_cfg, beta: float = 1.0,
                   lambda_prior_road: float = 0.5, lambda_reward: float = 1.0,
                   road_keep_prob: float = 0.6) -> dict:
    """Same math as stage2r.wm_losses, rewritten mask-multiply (no boolean
    indexing, no per-timestep .any()/int() host syncs) for XLA compilability.
    Verify against the original with _smoke_equivalence() before trusting this."""
    z1, coords, pos_idx = feats["z1"], feats["coords"], feats["pos_idx"]
    has_cand, road_embed, valid = feats["has_cand"], feats["road_embed"], feats["valid"]
    speed_mps, bearing = feats["speed_mps"], feats["bearing_deg"]
    cand_seg, cand_head = feats["cand_segment_id"], feats["cand_heading_deg"]
    cand_oneway, cand_hw = feats["cand_oneway"], feats["cand_highway_id"]
    cand_mask, cand_dperp = feats["cand_mask"], feats["cand_d_perp_m"]
    cand_kph = feats["cand_speed_kph"]
    device = z1.device
    B, L = valid.shape

    h, z = rssm.initial(B, device)
    zgps = speed_l = road_l = prior_road_l = reward_l = kl_loss = torch.zeros((), device=device)
    kl_dyn_raw_sum = kl_rep_raw_sum = cl_gps_z_sum = torch.zeros((), device=device)
    tgt_ent_sum = post_ent_sum = prior_ent_sum = kl_act_sum = torch.zeros((), device=device)
    # counts as TENSORS throughout the loop -- .item() happens once, after the
    # loop, not inside any per-timestep branch (that was the original's 4
    # separate host syncs x L; this is 1 per training step total).
    n_valid_t = n_road_t = n_rew_t = torch.zeros((), device=device)
    kl_groups = 1
    log_speed = torch.log1p(speed_mps.clamp(min=0))
    prev_head_deg = None

    for t in range(L):
        if t > 0:
            re_t = road_embed[:, t - 1]
            if rssm.training:  # static Python bool, not data-dependent -- fine under XLA
                keep = (torch.rand(B, device=device) < road_keep_prob).unsqueeze(-1)
                re_t = torch.where(keep, re_t, rssm.road_mask_token)
            h = rssm.step(h, z, re_t)
        prior_logits = rssm.prior(h)
        post_logits = rssm.posterior(h, z1[:, t])
        z = straight_through_sample(post_logits).reshape(B, -1)
        gps_hat, speed_hat = heads(h, z)

        road_logits = heads.road(h, z, road_z, cand_seg[:, t], cand_head[:, t],
                                  cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])
        r_hat = heads.reward(h, z, road_z, cand_seg[:, t], cand_head[:, t],
                              cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])

        vt = valid[:, t]
        vt_f = vt.to(gps_hat.dtype)

        # --- always compute for the full batch, then mask-multiply (no [vt] indexing) ---
        zgps = zgps + (F.huber_loss(gps_hat, coords[:, t], reduction="none").sum(-1) * vt_f).sum()
        speed_diff = F.mse_loss(speed_hat, log_speed[:, t].unsqueeze(-1) if speed_hat.dim() > log_speed[:, t].dim() else log_speed[:, t], reduction="none")
        speed_l = speed_l + (speed_diff.reshape(B, -1).sum(-1) * vt_f).sum()

        kl_dyn_pg = kl_categorical_pergroup(post_logits.detach(), prior_logits)
        kl_rep_pg = kl_categorical_pergroup(post_logits, prior_logits.detach())
        kl_dyn_raw_sum = kl_dyn_raw_sum + (kl_dyn_pg * vt_f.unsqueeze(-1)).sum()
        kl_rep_raw_sum = kl_rep_raw_sum + (kl_rep_pg * vt_f.unsqueeze(-1)).sum()
        kl_t = 0.5 * kl_dyn_pg.clamp(min=1.0).sum(-1) + 0.1 * kl_rep_pg.clamp(min=1.0).sum(-1)
        kl_loss = kl_loss + (kl_t * vt_f).sum()

        with torch.no_grad():
            cl_gps_z_sum = cl_gps_z_sum + ((gps_hat - coords[:, t]).norm(dim=-1) * vt_f).sum()
            kl_groups = kl_dyn_pg.size(-1)
            kl_act_sum = kl_act_sum + ((kl_dyn_pg > 1.0).to(kl_dyn_pg.dtype) * vt_f.unsqueeze(-1)).sum()
            post_p, prior_p = _mixed_probs(post_logits), _mixed_probs(prior_logits)
            post_ent_row = -(post_p * post_p.log()).sum(dim=tuple(range(1, post_p.dim())))
            prior_ent_row = -(prior_p * prior_p.log()).sum(dim=tuple(range(1, prior_p.dim())))
            post_ent_sum = post_ent_sum + (post_ent_row * vt_f).sum()
            prior_ent_sum = prior_ent_sum + (prior_ent_row * vt_f).sum()

        z_prior = straight_through_sample(prior_logits).reshape(B, -1)
        road_logits_prior = heads.road(h, z_prior, road_z, cand_seg[:, t], cand_head[:, t],
                                        cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])
        hc = has_cand[:, t]
        hc_f = hc.to(gps_hat.dtype)
        hc_kl_f = vt_f * hc_f
        ce_prior = _soft_road_ce(road_logits_prior, cand_dperp[:, t], cand_mask[:, t])
        prior_road_l = prior_road_l + (ce_prior * hc_kl_f).sum()
        n_valid_t = n_valid_t + vt_f.sum()

        ce_post = _soft_road_ce(road_logits, cand_dperp[:, t], cand_mask[:, t])
        road_l = road_l + (ce_post * hc_f).sum()
        n_road_t = n_road_t + hc_f.sum()
        with torch.no_grad():
            tgt_ent_sum = tgt_ent_sum + (_road_target_entropy(cand_dperp[:, t], cand_mask[:, t]) * hc_f).sum()

        if t > 0:
            prev = dict(prev_seg=feats["cand_segment_id"][:, t - 1].gather(
                            -1, pos_idx[:, t - 1].unsqueeze(-1)).squeeze(-1),
                        prev_road_heading_deg=prev_head_deg,
                        prev_gps_bearing_deg=bearing[:, t - 1])
        else:
            prev = {}
        with torch.no_grad():
            r_tgt = candidate_rewards(reward_cfg, succ, cand_dperp[:, t], cand_head[:, t],
                                       cand_kph[:, t], cand_oneway[:, t], cand_seg[:, t],
                                       cand_mask[:, t], bearing[:, t], speed_mps[:, t], **prev)
        rmask = cand_mask[:, t] & hc.unsqueeze(-1) & vt.unsqueeze(-1)
        rmask_f = rmask.to(gps_hat.dtype)
        # symlog targets: DreamerV3 (keeps reward gradient from dominating the shared trunk)
        reward_diff = F.mse_loss(r_hat, symlog(r_tgt), reduction="none")
        reward_l = reward_l + (reward_diff * rmask_f).sum()
        n_rew_t = n_rew_t + rmask_f.sum()

        prev_head_deg = cand_head[:, t].gather(-1, pos_idx[:, t].unsqueeze(-1)).squeeze(-1)

    # single host sync per training step (was up to 4*L in the original)
    n_valid = float(n_valid_t.clamp(min=1.0).detach().cpu())
    n_road = float(n_road_t.clamp(min=1.0).detach().cpu())
    n_rew = float(n_rew_t.clamp(min=1.0).detach().cpu())
    zgps, speed_l, kl_loss = zgps / n_valid, speed_l / n_valid, kl_loss / n_valid
    kl_dyn_raw, kl_rep_raw = kl_dyn_raw_sum / n_valid, kl_rep_raw_sum / n_valid
    cl_gps_z = cl_gps_z_sum / n_valid
    road_l, prior_road_l = road_l / n_road, prior_road_l / n_road
    reward_l = reward_l / n_rew
    tgt_ent = tgt_ent_sum / n_road
    post_ent, prior_ent = post_ent_sum / n_valid, prior_ent_sum / n_valid
    kl_act = kl_act_sum / (n_valid * kl_groups)

    total = (zgps + speed_l + road_l + lambda_prior_road * prior_road_l
             + lambda_reward * reward_l + beta * kl_loss)
    return {"total": total, "zgps": zgps.detach(), "speed": speed_l.detach(),
            "road": road_l.detach(), "prior_road": prior_road_l.detach(),
            "reward": reward_l.detach(), "kl": kl_loss.detach(),
            "kl_dyn_raw": kl_dyn_raw.detach(), "kl_rep_raw": kl_rep_raw.detach(),
            "cl_gps_z": cl_gps_z.detach(),
            "tgt_ent": tgt_ent, "road_ex": (road_l - tgt_ent).detach(),
            "prior_road_ex": (prior_road_l - tgt_ent).detach(),
            "post_ent": post_ent, "prior_ent": prior_ent, "kl_act": kl_act}


def _smoke_equivalence(seed: int = 0, atol: float = 1e-4, rtol: float = 1e-3) -> None:
    """Correctness gate: wm_losses_tpu must match the original wm_losses on
    identical inputs/model state. Run this BEFORE spending any real TPU
    quota -- CPU-only, no torch_xla required."""
    from dataset.config import RewardConfig

    torch.manual_seed(seed)
    feats, road_z, succ = _synthetic_feats()
    reward_cfg = RewardConfig()

    torch.manual_seed(seed)
    rssm_a = RSSM2(obs_dim=256, road_dim=256, norm="batch")
    heads_a = HeadsLight(h_dim=rssm_a.h_dim, road_dim=256)
    torch.manual_seed(seed)
    rssm_b = RSSM2(obs_dim=256, road_dim=256, norm="batch")
    heads_b = HeadsLight(h_dim=rssm_b.h_dim, road_dim=256)
    # belt-and-suspenders: force identical weights regardless of RNG-call-count
    # drift between the two module constructions.
    rssm_b.load_state_dict(rssm_a.state_dict())
    heads_b.load_state_dict(heads_a.state_dict())
    rssm_a.eval(); heads_a.eval(); rssm_b.eval(); heads_b.eval()  # kill road_keep_prob RNG divergence

    # straight_through_sample is a plain function (not an nn.Module), so .eval()
    # does NOT make it deterministic -- it samples from the global RNG regardless
    # of train/eval mode. Re-seed immediately before EACH call so both loss fns
    # draw the identical z sequence; without this the two calls run back-to-back
    # against a shared, already-advanced RNG stream and silently diverge.
    torch.manual_seed(seed)
    out_orig = wm_losses(rssm_a, heads_a, feats, road_z, succ, reward_cfg)
    torch.manual_seed(seed)
    out_tpu = wm_losses_tpu(rssm_b, heads_b, feats, road_z, succ, reward_cfg)

    mismatches = []
    for k in out_orig:
        a, b = float(out_orig[k]), float(out_tpu[k])
        if abs(a - b) > atol + rtol * abs(a):
            mismatches.append((k, a, b))
    print("[equivalence] " + "  ".join(f"{k}: orig={float(out_orig[k]):.6f} tpu={float(out_tpu[k]):.6f}"
                                        for k in out_orig))
    if mismatches:
        raise AssertionError(f"wm_losses_tpu diverges from wm_losses: {mismatches}")
    print("[equivalence] PASS -- wm_losses_tpu matches the original within tolerance")


def _smoke_train() -> None:
    """Standalone gradient-descent convergence check (mirrors stage2r._smoke,
    run against wm_losses_tpu alone -- catches autograd issues the forward-only
    equivalence test above can't see)."""
    from dataset.config import RewardConfig

    feats, road_z, succ = _synthetic_feats()
    reward_cfg = RewardConfig()
    rssm = RSSM2(obs_dim=256, road_dim=256, norm="batch")
    heads = HeadsLight(h_dim=rssm.h_dim, road_dim=256)
    opt = torch.optim.AdamW(list(rssm.parameters()) + list(heads.parameters()), lr=3e-3)

    first = last = None
    for i in range(150):
        opt.zero_grad()
        out_l = wm_losses_tpu(rssm, heads, feats, road_z, succ, reward_cfg)
        out_l["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(list(rssm.parameters()) + list(heads.parameters()), 100.0)
        if not torch.isfinite(gn):
            raise AssertionError(f"non-finite grad norm at step {i}: {gn}")
        opt.step()
        if first is None:
            first = float(out_l["total"])
        last = float(out_l["total"])
    print(f"[smoke_train] total {first:.3f} -> {last:.3f}  reward {float(out_l['reward']):.3f} "
          f"prior_road {float(out_l['prior_road']):.3f} kl_act {float(out_l['kl_act']):.3f}")
    kl_floor = 9.6
    assert last - kl_floor < 0.6 * (first - kl_floor), \
        f"above-floor loss did not drop enough: {first:.3f} -> {last:.3f}"
    assert float(out_l["reward"]) < 2.0, f"reward head did not learn: {out_l['reward']:.3f}"
    assert float(out_l["prior_road"]) < 2.0, f"prior_road did not learn: {out_l['prior_road']:.3f}"
    print("[smoke_train] PASS")


def train(processed_root, osm_root, stage0_ckpt, stage1_ckpt, city="porto", source="porto",
          limit_trajs=200_000, batch=16, steps=40_000, curriculum_step=10_000, seq_len=64,
          curriculum_len=16, lr=1e-4, warmup_steps=1000, beta=1.0,
          lambda_prior_road=0.5, lambda_reward=1.0, road_keep_prob=0.6, norm="batch",
          grad_clip=100.0, weight_decay=1e-4, out_dir="ckpt",
          ckpt_every=1000, log_every=50, seed=0, workers=0, resume=None,
          max_hours=None, stoch_groups=16, stoch_classes=16):
    device = xm.xla_device() if HAS_XLA else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    from dataset.config import RewardConfig
    reward_cfg = RewardConfig()

    road_z = _road_embeddings(osm_root, city, stage0_ckpt, device)
    stage1 = _stage1_encoder(road_z.size(1), stage1_ckpt, device)
    succ = _successor_table(osm_root, city, device)
    base_ds, _ = build_loader(processed_root, osm_root, city, source, limit_trajs, batch,
                               workers=workers, cache_dir=str(out / "cache"))

    from dataset.trajectories import collate_fn
    from torch.utils.data import DataLoader

    world_size = xr.world_size() if HAS_XLA else 1
    rank = xr.global_ordinal() if HAS_XLA else 0

    def loader_at(L):
        ds = LengthCapped(base_ds, L)
        sampler = None
        if world_size > 1:
            # each of the 8 TPU cores must train on a DISTINCT shard -- plain
            # shuffle=True would have all replicas redundantly re-train the
            # same full dataset instead of actually using all 8 cores' worth
            # of throughput.
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        loader = DataLoader(ds, batch_size=batch, shuffle=(sampler is None), sampler=sampler,
                             collate_fn=collate_fn, num_workers=workers, drop_last=True)
        if HAS_XLA:
            import torch_xla.distributed.parallel_loader as pl
            loader = pl.MpDeviceLoader(loader, device)
        return iter(loader)

    rssm = RSSM2(obs_dim=road_z.size(1), road_dim=road_z.size(1),
                 groups=stoch_groups, classes=stoch_classes, norm=norm).to(device)
    heads = HeadsLight(h_dim=rssm.h_dim, stoch_dim=stoch_groups * stoch_classes,
                       road_dim=road_z.size(1)).to(device)
    params = list(rssm.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                                                total_iters=max(1, warmup_steps))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warmup_steps))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                   milestones=[warmup_steps])

    start_step = 0
    if resume is not None and Path(resume).is_file():
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        rssm.load_state_dict(ckpt["rssm"]); heads.load_state_dict(ckpt["heads"])
        opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        if _is_master():
            print(f"[stage2r_tpu] resumed from {resume} at step {start_step}")

    loss_kw = dict(beta=beta, lambda_prior_road=lambda_prior_road,
                   lambda_reward=lambda_reward, road_keep_prob=road_keep_prob)
    if _is_master():
        print(f"[stage2r_tpu] device={device} xla={HAS_XLA} world_size={world_size} steps={steps} "
              f"curriculum={curriculum_len}->{seq_len}@{curriculum_step} batch={batch}/core "
              f"(global={batch * world_size}) lr={lr:.1e} warmup={warmup_steps} norm={norm} "
              f"lambda_reward={lambda_reward}")
    it = loader_at(seq_len if start_step >= curriculum_step else curriculum_len)
    rssm.train(); heads.train()
    t0 = time.time()
    for step in range(start_step, steps):
        if step == curriculum_step:
            it = loader_at(seq_len)
            if _is_master():
                print(f"[stage2r_tpu] step {step}: curriculum L -> {seq_len}")
        try:
            b = next(it)
        except StopIteration:
            it = loader_at(curriculum_len if step < curriculum_step else seq_len)
            b = next(it)
        if not HAS_XLA:  # MpDeviceLoader already places tensors on-device
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

        opt.zero_grad()
        feats = extract_features_r(stage1, road_z, b)
        out_l = wm_losses_tpu(rssm, heads, feats, road_z, succ, reward_cfg, **loss_kw)
        out_l["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if HAS_XLA:
            xm.optimizer_step(opt)  # all-reduce + step, replicated-safe
        else:
            opt.step()
        sched.step()
        if HAS_XLA:
            xm.mark_step()  # execute the traced graph; without this, XLA queues ops indefinitely

        if step % log_every == 0 and _is_master():
            # .item()/float() here is a deliberate, infrequent (every log_every
            # steps, not every step) host sync purely for the printed line --
            # never gate control flow on it. Master-only: 8 replicas printing
            # the same line would just be the duplicate-log problem again,
            # 8x over instead of 2x.
            print(f"step {step:06d}  total {float(out_l['total']):.4f}  zgps {out_l['zgps']:.4f}  "
                  f"speed {out_l['speed']:.4f}  road {out_l['road']:.4f}  "
                  f"prior_road {out_l['prior_road']:.4f}  reward {out_l['reward']:.4f}  "
                  f"kl {out_l['kl']:.4f}  kl_dyn_raw {out_l['kl_dyn_raw']:.4f}  "
                  f"kl_rep_raw {out_l['kl_rep_raw']:.4f}  cl_gps_z_km {out_l['cl_gps_z']:.4f}  "
                  f"tgt_ent {out_l['tgt_ent']:.4f}  road_ex {out_l['road_ex']:.4f}  "
                  f"prior_ex {out_l['prior_road_ex']:.4f}  kl_act {out_l['kl_act']:.3f}  "
                  f"post_ent {out_l['post_ent']:.2f}  prior_ent {out_l['prior_ent']:.2f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            _save_ckpt_tpu(out / f"stage2r_{city}_step{step}.pt", rssm, heads, opt, sched, step, norm)
        if max_hours is not None and (time.time() - t0) > max_hours * 3600:
            _save_ckpt_tpu(out / f"stage2r_{city}.pt", rssm, heads, opt, sched, step, norm)
            if _is_master():
                print(f"[stage2r_tpu] max-hours {max_hours} reached at step {step}; "
                      f"saved -> {out / f'stage2r_{city}.pt'} (chain next session with --resume)", flush=True)
            return

    _save_ckpt_tpu(out / f"stage2r_{city}.pt", rssm, heads, opt, sched, steps - 1, norm)
    if _is_master():
        print(f"[stage2r_tpu] saved -> {out / f'stage2r_{city}.pt'}")


def _mp_fn(index, cfg):
    """Entry point for xmp.spawn: one call per TPU core (index=0..N-1). `cfg`
    is the exact kwargs dict train() takes; index is unused directly -- XLA's
    ordinal/world-size are read via xr.global_ordinal()/xr.world_size() from
    inside train() itself, set implicitly by the spawn machinery."""
    train(**cfg)


def main():
    p = argparse.ArgumentParser(description="TPU/XLA-compatible decoder-light WM pretraining")
    p.add_argument("--processed-root", default="data/processed")
    p.add_argument("--osm-root", default="data/osm")
    p.add_argument("--stage0-ckpt", default="ckpt/stage0_porto.pt")
    p.add_argument("--stage1-ckpt", default="ckpt/stage1_porto.pt")
    p.add_argument("--city", default="porto")
    p.add_argument("--source", default="porto")
    p.add_argument("--limit-trajs", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--curriculum-step", type=int, default=10_000)
    p.add_argument("--curriculum-len", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lambda-prior-road", type=float, default=0.5)
    p.add_argument("--lambda-reward", type=float, default=1.0)
    p.add_argument("--road-keep-prob", type=float, default=0.6)
    p.add_argument("--norm", default="batch", choices=["batch", "layer", "none"])
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--out-dir", default="ckpt")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--stoch-groups", type=int, default=16)
    p.add_argument("--stoch-classes", type=int, default=16)
    p.add_argument("--tpu-cores", type=int, default=1,
                   help="whether to fan out across all TPU cores via xmp.spawn (>1 triggers it). "
                        "torch_xla>=2.1 auto-spawns across every visible core (nprocs is no longer "
                        "settable to an explicit count -- this flag is a switch, not an exact count). "
                        "Ignored (in-process) when torch_xla isn't available.")
    p.add_argument("--smoke", action="store_true",
                    help="equivalence + convergence checks vs the original wm_losses, CPU-safe, no XLA needed")
    a = p.parse_args()
    if a.smoke:
        _smoke_equivalence()
        _smoke_train()
        return

    cfg = dict(processed_root=a.processed_root, osm_root=a.osm_root, stage0_ckpt=a.stage0_ckpt,
               stage1_ckpt=a.stage1_ckpt, city=a.city, source=a.source, limit_trajs=a.limit_trajs,
               batch=a.batch, steps=a.steps, curriculum_step=a.curriculum_step, seq_len=a.seq_len,
               curriculum_len=a.curriculum_len, lr=a.lr, warmup_steps=a.warmup_steps, beta=a.beta,
               lambda_prior_road=a.lambda_prior_road, lambda_reward=a.lambda_reward,
               road_keep_prob=a.road_keep_prob, norm=a.norm, grad_clip=a.grad_clip,
               out_dir=a.out_dir, ckpt_every=a.ckpt_every, log_every=a.log_every,
               workers=a.workers, resume=a.resume, max_hours=a.max_hours,
               stoch_groups=a.stoch_groups, stoch_classes=a.stoch_classes)

    if HAS_XLA and a.tpu_cores > 1:
        # torch_xla>=2.1: nprocs must be None (or 1) -- xmp.spawn auto-detects
        # and uses every visible TPU core itself, no explicit count accepted.
        # Kaggle kernels pre-set TPU_PROCESS_ADDRESSES/CLOUD_TPU_TASK_ID for the
        # notebook's own single-process TPU access; left in place, xmp.spawn's
        # freshly-forked 8-worker mesh finds a stale 1-worker address config and
        # fails with "Expected 8 worker addresses, got 1" (pytorch/xla's own
        # Kaggle example notebook, contrib/kaggle/distributed-pytorch-xla-basics
        # -with-pjrt.ipynb, pops these before spawn for exactly this reason).
        # start_method="fork" is likewise required in interactive/papermill
        # notebook kernels per that same example.
        os.environ.pop("TPU_PROCESS_ADDRESSES", None)
        os.environ.pop("CLOUD_TPU_TASK_ID", None)
        print("[stage2r_tpu] spawning across all visible TPU cores via xmp.spawn")
        xmp.spawn(_mp_fn, args=(cfg,), start_method="fork")
    else:
        train(**cfg)


if __name__ == "__main__":
    main()
