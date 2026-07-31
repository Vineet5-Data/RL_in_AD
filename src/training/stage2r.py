"""research2 Stage-2R: decoder-light world model pretraining (Rank-1 redesign).

methodology-comparison.md Sec.1/Sec.5 Rank 1: replace the reconstruction-heavy
DreamerV3-style RSSM (Runs 1-3, posterior collapse empirically confirmed at
gate R2, hard-stopped per R3) with a decoder-light world model in the
TD-MPC2/MuDreamer lineage. Differences vs training/stage2.py (kept untouched
for provenance):

  REMOVED  MDN residual GPS head + (h,z) coarse GPS Huber -- the reconstruction
           decoder `h` freeloads on (root cause of collapse) and the dominant-
           gradient head in the 14C MTL competition finding.
  REMOVED  Change 14A/14B/14C patches (z-only aux mirrors, aggressive posterior
           schedule, gradient-competition probe) -- schedule/channel patches
           superseded by the structural fix; keeping them would confound
           attribution of the redesign's effect.
  KEPT     z-only GPS decode, now the ONLY GPS term ("permanently z-only-gate"):
           HeadsLight.gps structurally never sees h. Doubles as the redefined
           cl_gps interpretability probe (cl_gps_z).
  KEPT     road CE (posterior, Change-9 soft targets) as the PRIMARY signal +
           Change-4 prior-side road CE (makes the prior road-predictive, the
           property imagination needs).
  KEPT     Change-13A road-embedding dropout (road_keep_prob) -- retained as
           robustness augmentation for Stage-3 imagination (transitions there
           are conditioned on model-chosen, possibly wrong, segments), no longer
           carrying the anti-collapse burden.
  ADDED    reward-head regression: HeadsLight.reward vs the label-free Sec.2.5
           physics rewards (training/rewards.py) computed per candidate on the
           real sequence -- the DreamerV3 reward channel Stage 3's imagined
           lambda-returns require. Value/reward/action prediction replacing
           observation reconstruction is exactly MuDreamer's recipe.
  ADDED    --norm {batch,layer,none} (default batch) inside prior/post MLPs,
           MuDreamer's anti-constant-code measure (flagged: untested on
           GRU+categorical latents, methodology-comparison Gap 5).

Run:   PYTHONPATH=../data-preprocess python -m training.stage2r --processed-root ../../data/processed --osm-root ../../data/osm --stage0-ckpt ckpt/stage0_porto.pt --stage1-ckpt ckpt/stage1_porto.pt
Smoke: python -m training.stage2r --smoke
"""

from __future__ import annotations

# pyarrow must load before torch (see training/stage0.py -- Windows segfault guard)
import pyarrow.parquet  # noqa: F401

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from models.world_model2 import (RSSM2, HeadsLight, straight_through_sample,
                                      kl_categorical_pergroup, symlog)
    from models.world_model import _mixed_probs, free_bits_kl
    from training.stage2 import (LengthCapped, _stage1_encoder, _soft_road_ce,
                                  extract_frozen_features)
    from training.stage1 import _road_embeddings, _road_encoder_and_graph, build_loader
    from training.rewards import SuccessorTable, candidate_rewards
except ImportError:  # flat layout (Kaggle CODE_ROOT on sys.path)
    from world_model2 import (RSSM2, HeadsLight, straight_through_sample,  # type: ignore
                               kl_categorical_pergroup, symlog)
    from world_model import _mixed_probs, free_bits_kl  # type: ignore
    from stage2 import (LengthCapped, _stage1_encoder, _soft_road_ce,  # type: ignore
                         extract_frozen_features)
    from stage1 import _road_embeddings, _road_encoder_and_graph, build_loader  # type: ignore
    from rewards import SuccessorTable, candidate_rewards  # type: ignore


def extract_features_r(stage1, road_z: torch.Tensor, batch: dict) -> dict:
    """stage2.extract_frozen_features + the raw fields the reward targets need
    (GPS bearing/speed, candidate speed limits)."""
    feats = extract_frozen_features(stage1, road_z, batch)
    feats["bearing_deg"] = batch["bearing_deg"]
    feats["cand_speed_kph"] = batch["cand_speed_kph"]
    return feats


def _road_target_entropy(d_perp: torch.Tensor, cand_mask: torch.Tensor,
                          sigma: float = 10.0) -> torch.Tensor:
    """H(soft road target) per sample [..] -- the irreducible floor of
    _soft_road_ce (excess CE = CE - H = KL(target||pred), zero iff the head
    matches the geometry teacher exactly). Target construction mirrors
    stage2._soft_road_ce -- keep the two in sync (same `sigma`)."""
    d = d_perp.masked_fill(~cand_mask, 1e4)
    p = F.softmax(-(d ** 2) / (2 * sigma ** 2), dim=-1)
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def wm_losses(rssm: RSSM2, heads: HeadsLight, feats: dict, road_z: torch.Tensor,
              succ: SuccessorTable, reward_cfg, beta: float = 1.0,
              lambda_prior_road: float = 0.5, lambda_reward: float = 1.0,
              road_keep_prob: float = 0.6, road_target_sigma: float = 10.0,
              ss_prob: float = 0.0) -> dict:
    """Teacher-forced unroll, decoder-light objective:
        L = zgps_huber + speed_mse + road_ce(post) + lambda_prior_road*road_ce(prior)
            + lambda_reward*reward_mse + beta*KL_freebits
    KL identical to stage2.py (Change 11 per-group free bits, 0.5 dyn / 0.1 rep).

    `road_target_sigma` < 10 sharpens the road-CE soft pseudo-label (ASTRA-
    style confidence sharpening, see _soft_road_ce). `ss_prob` > 0 enables
    scheduled sampling: with this per-sample probability, the previous-step
    road_embed fed into rssm.step is swapped from the ground-truth pos_idx
    pseudo-label to the model's OWN previous road-head argmax pick -- closes
    the train/inference gap diagnosed in diag_jump_vs_priorerror.py (online
    disconnected-jump rate is 4.5x higher right after the model's own choice
    was wrong; teacher-forcing on ground truth never shows it that state)."""
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
    # P0b instrumentation accumulators -- no_grad stats only (no parameters, no
    # RNG), so checkpoints resume unchanged and the gradient path is untouched.
    tgt_ent_sum = post_ent_sum = prior_ent_sum = kl_act_sum = torch.zeros((), device=device)
    kl_groups = 1
    n_valid = n_road = n_rew = 0
    log_speed = torch.log1p(speed_mps.clamp(min=0))
    # previous fix's resolved (pseudo, label-free) candidate heading, per sample
    prev_head_deg = None
    # scheduled sampling: previous step's OWN road-head prediction (set at the
    # bottom of each iteration, consumed at the top of the next one)
    prev_road_logits = prev_cand_mask_t = prev_cand_seg_t = None

    for t in range(L):
        if t > 0:
            re_t = road_embed[:, t - 1]  # previous resolved fix (same shift as stage2.py)
            if rssm.training and ss_prob > 0.0:
                own_logits = prev_road_logits.masked_fill(~prev_cand_mask_t, -1e4)
                own_seg = prev_cand_seg_t.gather(-1, own_logits.argmax(-1, keepdim=True)).squeeze(-1)
                use_own = (torch.rand(B, device=device) < ss_prob).unsqueeze(-1)
                re_t = torch.where(use_own, road_z[own_seg.clamp(min=0)], re_t)
            if rssm.training:
                keep = (torch.rand(B, device=device) < road_keep_prob).unsqueeze(-1)
                re_t = torch.where(keep, re_t, rssm.road_mask_token)
            h = rssm.step(h, z, re_t)
        prior_logits = rssm.prior(h)
        post_logits = rssm.posterior(h, z1[:, t])
        z = straight_through_sample(post_logits).reshape(B, -1)
        gps_hat, speed_hat = heads(h, z)

        road_logits = heads.road(h, z, road_z, cand_seg[:, t], cand_head[:, t],
                                  cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])
        prev_road_logits = road_logits.detach()
        prev_cand_mask_t = cand_mask[:, t]
        prev_cand_seg_t = cand_seg[:, t]
        r_hat = heads.reward(h, z, road_z, cand_seg[:, t], cand_head[:, t],
                              cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])

        vt = valid[:, t]
        if vt.any():
            zgps = zgps + F.huber_loss(gps_hat[vt], coords[:, t][vt], reduction="sum")
            speed_l = speed_l + F.mse_loss(speed_hat[vt], log_speed[:, t][vt], reduction="sum")

            kl_dyn_pg = kl_categorical_pergroup(post_logits.detach(), prior_logits)
            kl_rep_pg = kl_categorical_pergroup(post_logits, prior_logits.detach())
            kl_dyn_raw_sum = kl_dyn_raw_sum + kl_dyn_pg[vt].sum()
            kl_rep_raw_sum = kl_rep_raw_sum + kl_rep_pg[vt].sum()
            kl_t = free_bits_kl(kl_dyn_pg, kl_rep_pg)
            kl_loss = kl_loss + kl_t[vt].sum()

            with torch.no_grad():
                cl_gps_z_sum = cl_gps_z_sum + (gps_hat[vt] - coords[:, t][vt]).norm(dim=-1).sum()
                # P0b: fraction of groups above the 1-nat free-bits floor +
                # posterior/prior categorical entropy (unimix probs, same as KL).
                kl_groups = kl_dyn_pg.size(-1)
                kl_act_sum = kl_act_sum + (kl_dyn_pg[vt] > 1.0).float().sum()
                post_p, prior_p = _mixed_probs(post_logits[vt]), _mixed_probs(prior_logits[vt])
                post_ent_sum = post_ent_sum - (post_p * post_p.log()).sum()
                prior_ent_sum = prior_ent_sum - (prior_p * prior_p.log()).sum()

            # Change 4: prior-side road CE (separate prior sample, same head)
            z_prior = straight_through_sample(prior_logits).reshape(B, -1)
            road_logits_prior = heads.road(h, z_prior, road_z, cand_seg[:, t], cand_head[:, t],
                                            cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])
            hc_kl = vt & has_cand[:, t]
            if hc_kl.any():
                ce_prior = _soft_road_ce(road_logits_prior, cand_dperp[:, t], cand_mask[:, t],
                                          sigma=road_target_sigma)
                prior_road_l = prior_road_l + ce_prior[hc_kl].sum()
            n_valid += int(vt.sum())

        hc = has_cand[:, t]
        if hc.any():
            ce_post = _soft_road_ce(road_logits, cand_dperp[:, t], cand_mask[:, t],
                                     sigma=road_target_sigma)
            road_l = road_l + ce_post[hc].sum()
            n_road += int(hc.sum())
            with torch.no_grad():
                tgt_ent_sum = tgt_ent_sum + _road_target_entropy(
                    cand_dperp[:, t], cand_mask[:, t], sigma=road_target_sigma)[hc].sum()

            # reward targets: label-free physics rewards on the real fix
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
            if rmask.any():
                # symlog targets (DreamerV3): keeps the reward gradient from
                # dominating the shared trunk (see world_model2.symlog docstring)
                reward_l = reward_l + F.mse_loss(r_hat[rmask], symlog(r_tgt)[rmask],
                                                  reduction="sum")
                n_rew += int(rmask.sum())

        prev_head_deg = cand_head[:, t].gather(-1, pos_idx[:, t].unsqueeze(-1)).squeeze(-1)

    n_valid, n_road, n_rew = max(n_valid, 1), max(n_road, 1), max(n_rew, 1)
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
            # P0b: tgt_ent = soft-target entropy (the road-CE floor); road_ex /
            # prior_road_ex = CE - tgt_ent = KL(target||pred), true distance to
            # floor; kl_act = fraction of groups above the 1-nat free-bits floor.
            "tgt_ent": tgt_ent, "road_ex": (road_l - tgt_ent).detach(),
            "prior_road_ex": (prior_road_l - tgt_ent).detach(),
            "post_ent": post_ent, "prior_ent": prior_ent, "kl_act": kl_act}


def _successor_table(osm_root: str, city: str, device) -> SuccessorTable:
    from roadgraph.io import load_pyg
    data = load_pyg(str(osm_root), city)
    return SuccessorTable(data.edge_index, int(data.num_nodes), device)


def train(processed_root, osm_root, stage0_ckpt, stage1_ckpt, city="porto", source="porto",
          limit_trajs=200_000, batch=16, steps=40_000, curriculum_step=10_000, seq_len=64,
          curriculum_len=16, lr=1e-4, warmup_steps=1000, beta=1.0,
          lambda_prior_road=0.5, lambda_reward=1.0, road_keep_prob=0.6, norm="batch",
          grad_clip=100.0, weight_decay=1e-4, out_dir="ckpt",
          ckpt_every=1000, log_every=50, device=None, seed=0, workers=0, resume=None,
          max_hours=None, constant_lr=False, stoch_groups=16, stoch_classes=16,
          unfreeze_encoder=False, road_target_sigma=10.0, ss_max_prob=0.0, ss_ramp_steps=4000):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    from dataset.config import RewardConfig
    reward_cfg = RewardConfig()

    road_enc = road_data = None
    if unfreeze_encoder:
        # Silesia confound probe: recompute road_z from the live RoadGAT every
        # step so road CE / reward-MSE gradients (the two losses that consume
        # road_z outside extract_frozen_features's no_grad block) can reach
        # the encoder -- isolates whether the frozen-encoder-during-retrain
        # confound flagged in the per-city retraining writeup was the bug.
        road_enc, road_data = _road_encoder_and_graph(osm_root, city, stage0_ckpt, device)
        road_z = road_enc(road_data.x, road_data.highway_id, road_data.edge_index)
    else:
        road_z = _road_embeddings(osm_root, city, stage0_ckpt, device)
    stage1 = _stage1_encoder(road_z.size(1), stage1_ckpt, device)
    succ = _successor_table(osm_root, city, device)
    base_ds, _ = build_loader(processed_root, osm_root, city, source, limit_trajs, batch,
                               workers=workers, cache_dir=str(out / "cache"))

    from dataset.trajectories import collate_fn
    from torch.utils.data import DataLoader

    def loader_at(L):
        ds = LengthCapped(base_ds, L)
        return iter(DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_fn,
                                num_workers=workers))

    rssm = RSSM2(obs_dim=road_z.size(1), road_dim=road_z.size(1),
                 groups=stoch_groups, classes=stoch_classes, norm=norm).to(device)
    heads = HeadsLight(h_dim=rssm.h_dim, stoch_dim=stoch_groups * stoch_classes,
                       road_dim=road_z.size(1)).to(device)
    params = list(rssm.parameters()) + list(heads.parameters())
    if unfreeze_encoder:
        params += list(road_enc.parameters())
    opt, sched = _build_optim_sched(params, lr, weight_decay, constant_lr, warmup_steps, steps)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    start_step = 0
    if resume is not None and Path(resume).is_file():
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        rssm.load_state_dict(ckpt["rssm"]); heads.load_state_dict(ckpt["heads"])
        opt.load_state_dict(ckpt["opt"])
        if unfreeze_encoder:
            if "road_enc" in ckpt:
                road_enc.load_state_dict(ckpt["road_enc"])
            else:
                print("[stage2r] resume ckpt has no road_enc state -- "
                      "continuing from the fresh Porto stage0 encoder weights")
        if constant_lr:
            # P0: ignore the saved (dead cosine) sched; run flat LR from here.
            for g in opt.param_groups:
                g["lr"] = lr
        else:
            sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        print(f"[stage2r] resumed from {resume} at step {start_step}"
              + (" (constant-lr, sched reset)" if constant_lr else ""))

    loss_kw = dict(beta=beta, lambda_prior_road=lambda_prior_road,
                   lambda_reward=lambda_reward, road_keep_prob=road_keep_prob,
                   road_target_sigma=road_target_sigma)
    extra_meta = {"road_target_sigma": road_target_sigma, "ss_max_prob": ss_max_prob,
                  "ss_ramp_steps": ss_ramp_steps}
    print(f"[stage2r] device={device} bf16={use_bf16} steps={steps} "
          f"curriculum={curriculum_len}->{seq_len}@{curriculum_step} batch={batch} "
          f"lr={lr:.1e} warmup={warmup_steps} norm={norm} lambda_reward={lambda_reward} "
          f"road_target_sigma={road_target_sigma} ss_max_prob={ss_max_prob} "
          f"ss_ramp_steps={ss_ramp_steps}")
    it = loader_at(seq_len if start_step >= curriculum_step else curriculum_len)
    rssm.train(); heads.train()
    t0 = time.time()
    for step in range(start_step, steps):
        if step == curriculum_step:
            it = loader_at(seq_len)
            print(f"[stage2r] step {step}: curriculum L -> {seq_len}")
        try:
            b = next(it)
        except StopIteration:
            it = loader_at(curriculum_len if step < curriculum_step else seq_len)
            b = next(it)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

        ss_prob = ss_max_prob * min(1.0, step / max(1, ss_ramp_steps))
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if unfreeze_encoder:
                road_z = road_enc(road_data.x, road_data.highway_id, road_data.edge_index)
            feats = extract_features_r(stage1, road_z, b)
            out_l = wm_losses(rssm, heads, feats, road_z, succ, reward_cfg, ss_prob=ss_prob, **loss_kw)
        out_l["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step(); sched.step()

        if step % log_every == 0:
            print(f"step {step:06d}  total {float(out_l['total']):.4f}  zgps {out_l['zgps']:.4f}  "
                  f"speed {out_l['speed']:.4f}  road {out_l['road']:.4f}  "
                  f"prior_road {out_l['prior_road']:.4f}  reward {out_l['reward']:.4f}  "
                  f"kl {out_l['kl']:.4f}  kl_dyn_raw {out_l['kl_dyn_raw']:.4f}  "
                  f"kl_rep_raw {out_l['kl_rep_raw']:.4f}  cl_gps_z_km {out_l['cl_gps_z']:.4f}  "
                  f"tgt_ent {out_l['tgt_ent']:.4f}  road_ex {out_l['road_ex']:.4f}  "
                  f"prior_ex {out_l['prior_road_ex']:.4f}  kl_act {out_l['kl_act']:.3f}  "
                  f"post_ent {out_l['post_ent']:.2f}  prior_ent {out_l['prior_ent']:.2f}  "
                  f"ss_prob {ss_prob:.3f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            _save_and_log(out, city, rssm, heads, opt, sched, step, norm,
                          road_enc=road_enc, extra_meta=extra_meta)
        # session wall-clock guard: Kaggle kills ~12h kernels WITHOUT saving output;
        # save the resume ckpt and exit cleanly instead (Run-2-XL chains sessions via --resume)
        if max_hours is not None and (time.time() - t0) > max_hours * 3600:
            path = _save_and_log(out, city, rssm, heads, opt, sched, step, norm,
                                 road_enc=road_enc, extra_meta=extra_meta, final=True)
            print(f"[stage2r] max-hours {max_hours} reached at step {step}; "
                  f"saved -> {path} (chain next session with --resume)", flush=True)
            return

    path = _save_and_log(out, city, rssm, heads, opt, sched, steps - 1, norm,
                         road_enc=road_enc, extra_meta=extra_meta, final=True)
    print(f"[stage2r] saved -> {path}")


def _save_ckpt(path, rssm, heads, opt, sched, step, norm, road_enc=None, extra_meta=None):
    ckpt = {"rssm": rssm.state_dict(), "heads": heads.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
            "meta": {"norm": norm, "stoch_groups": rssm.groups,
                     "stoch_classes": rssm.classes}}
    if road_enc is not None:
        ckpt["road_enc"] = road_enc.state_dict()  # only present when --unfreeze-encoder
    if extra_meta:
        ckpt["meta"].update(extra_meta)
    torch.save(ckpt, path)


def _build_optim_sched(params, lr, weight_decay, constant_lr, warmup_steps, steps):
    """Shared by train()/train_multi() -- identical optimizer/scheduler setup either way."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if constant_lr:
        # ponytail: P0 LR-tail probe — hold LR flat (no warmup/cosine) so the
        # plateau test isn't confounded by the cosine decaying to ~0.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    else:
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                                                    total_iters=max(1, warmup_steps))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warmup_steps))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                       milestones=[warmup_steps])
    return opt, sched


def _save_and_log(out, tag, rssm, heads, opt, sched, step, norm, road_enc=None,
                   extra_meta=None, final=False):
    """Shared by train()/train_multi() -- `tag` is the city (single) or the
    "multi_<sources>" tag (multi); `final=True` drops the _step{N} suffix, the
    convention both training loops use for their --resume-chained checkpoint."""
    suffix = "" if final else f"_step{step}"
    path = out / f"stage2r_{tag}{suffix}.pt"
    _save_ckpt(path, rssm, heads, opt, sched, step, norm, road_enc=road_enc, extra_meta=extra_meta)
    return path


def train_multi(processed_root, osm_root, stage0_ckpt, stage1_ckpt, city_sources,
                 city_weights=None, limit_trajs=200_000, batch=16, steps=40_000,
                 curriculum_step=10_000, seq_len=64, curriculum_len=16, lr=1e-4,
                 warmup_steps=1000, beta=1.0, lambda_prior_road=0.5, lambda_reward=1.0,
                 road_keep_prob=0.6, norm="batch", grad_clip=100.0, weight_decay=1e-4,
                 out_dir="ckpt", ckpt_every=1000, log_every=50, device=None, seed=0,
                 workers=0, resume=None, max_hours=None, constant_lr=False,
                 stoch_groups=16, stoch_classes=16, road_target_sigma=10.0,
                 ss_max_prob=0.0, ss_ramp_steps=4000):
    """Hybrid-cities training: one shared RSSM/HeadsLight trained across several
    cities' (frozen, precomputed) road graphs. Per step, a city is drawn at
    random (weighted by dataset size, sqrt-scaled by default so Porto's ~1.6M
    segments don't drown out a 15,854-segment Rome or a 1,816-segment
    geolife_car -- see project_summary.md's multi-city data-prep entry for the
    five cities this was built for) and that city's precomputed road_z/
    SuccessorTable/DataLoader are used for the step -- everything else (rssm,
    heads, stage1, optimizer, scheduler) is one shared instance updated by
    every step regardless of which city it came from. `wm_losses`/
    `extract_features_r`/the model classes are used completely unchanged;
    only the city-specific context is swapped per step.

    Out of scope (use single-city train() instead): --unfreeze-encoder (was a
    one-off closed diagnostic, not worth the per-city road_enc complexity here).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    from dataset.config import RewardConfig
    from dataset.trajectories import collate_fn
    from torch.utils.data import DataLoader
    reward_cfg = RewardConfig()

    ctx = {}
    road_dim = None
    for city, source in city_sources:
        road_z_c = _road_embeddings(osm_root, city, stage0_ckpt, device)
        if road_dim is None:
            road_dim = road_z_c.size(1)
        assert road_z_c.size(1) == road_dim, \
            f"{city}/{source}: road_dim {road_z_c.size(1)} != {road_dim} (RoadGAT out_dim mismatch)"
        succ_c = _successor_table(osm_root, city, device)
        base_ds_c, _ = build_loader(processed_root, osm_root, city, source, limit_trajs, batch,
                                     workers=workers, cache_dir=str(out / "cache" / f"{city}__{source}"))
        ctx[(city, source)] = {"road_z": road_z_c, "succ": succ_c, "base_ds": base_ds_c, "it": None,
                                "n": len(base_ds_c), "loss_hist": []}
        print(f"[stage2r] city={city} source={source} segments={road_z_c.size(0)} trajs={len(base_ds_c)}")

    if city_weights is None:
        city_weights = [len(ctx[cs]["base_ds"]) ** 0.5 for cs in city_sources]
    assert len(city_weights) == len(city_sources)
    wsum = sum(city_weights)
    print(f"[stage2r] city weights: " +
          "  ".join(f"{s}={w/wsum:.3f}" for (_, s), w in zip(city_sources, city_weights)))

    stage1 = _stage1_encoder(road_dim, stage1_ckpt, device)

    def loader_at(cs, L):
        ds = LengthCapped(ctx[cs]["base_ds"], L)
        return iter(DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_fn,
                                num_workers=workers))

    rssm = RSSM2(obs_dim=road_dim, road_dim=road_dim,
                 groups=stoch_groups, classes=stoch_classes, norm=norm).to(device)
    heads = HeadsLight(h_dim=rssm.h_dim, stoch_dim=stoch_groups * stoch_classes,
                       road_dim=road_dim).to(device)
    params = list(rssm.parameters()) + list(heads.parameters())
    opt, sched = _build_optim_sched(params, lr, weight_decay, constant_lr, warmup_steps, steps)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    start_step = 0
    if resume is not None and Path(resume).is_file():
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        rssm.load_state_dict(ckpt["rssm"]); heads.load_state_dict(ckpt["heads"])
        opt.load_state_dict(ckpt["opt"])
        if constant_lr:
            for g in opt.param_groups:
                g["lr"] = lr
        else:
            sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        print(f"[stage2r] resumed from {resume} at step {start_step}"
              + (" (constant-lr, sched reset)" if constant_lr else ""))

    loss_kw = dict(beta=beta, lambda_prior_road=lambda_prior_road,
                   lambda_reward=lambda_reward, road_keep_prob=road_keep_prob,
                   road_target_sigma=road_target_sigma)
    tag = "multi_" + "-".join(s for _, s in city_sources)
    print(f"[stage2r] MULTI-CITY device={device} bf16={use_bf16} steps={steps} "
          f"curriculum={curriculum_len}->{seq_len}@{curriculum_step} batch={batch} "
          f"lr={lr:.1e} warmup={warmup_steps} norm={norm} lambda_reward={lambda_reward} "
          f"cities={[f'{c}:{s}' for c, s in city_sources]} road_target_sigma={road_target_sigma} "
          f"ss_max_prob={ss_max_prob} ss_ramp_steps={ss_ramp_steps}")
    for cs in city_sources:
        ctx[cs]["it"] = loader_at(cs, seq_len if start_step >= curriculum_step else curriculum_len)
    rssm.train(); heads.train()
    t0 = time.time()
    extra_meta = {"cities": [f"{c}:{s}" for c, s in city_sources],
                  "city_weights": [w / wsum for w in city_weights],
                  "road_target_sigma": road_target_sigma, "ss_max_prob": ss_max_prob,
                  "ss_ramp_steps": ss_ramp_steps}
    for step in range(start_step, steps):
        if step == curriculum_step:
            for cs in city_sources:
                ctx[cs]["it"] = loader_at(cs, seq_len)
            print(f"[stage2r] step {step}: curriculum L -> {seq_len}")
        cs = rng.choices(city_sources, weights=city_weights, k=1)[0]
        try:
            b = next(ctx[cs]["it"])
        except StopIteration:
            ctx[cs]["it"] = loader_at(cs, curriculum_len if step < curriculum_step else seq_len)
            b = next(ctx[cs]["it"])
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

        ss_prob = ss_max_prob * min(1.0, step / max(1, ss_ramp_steps))
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            feats = extract_features_r(stage1, ctx[cs]["road_z"], b)
            out_l = wm_losses(rssm, heads, feats, ctx[cs]["road_z"], ctx[cs]["succ"],
                               reward_cfg, ss_prob=ss_prob, **loss_kw)
        out_l["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step(); sched.step()

        hist = ctx[cs]["loss_hist"]; hist.append(float(out_l["total"]))
        if len(hist) > 200:
            del hist[:-200]

        if step % log_every == 0:
            per_city = "  ".join(f"{s}:{sum(ctx[(c, s)]['loss_hist'][-50:]) / max(1, len(ctx[(c, s)]['loss_hist'][-50:])):.3f}"
                                  for c, s in city_sources if ctx[(c, s)]["loss_hist"])
            print(f"step {step:06d}  city={cs[1]}  total {float(out_l['total']):.4f}  "
                  f"road {out_l['road']:.4f}  prior_road {out_l['prior_road']:.4f}  "
                  f"reward {out_l['reward']:.4f}  kl {out_l['kl']:.4f}  "
                  f"cl_gps_z_km {out_l['cl_gps_z']:.4f}  kl_act {out_l['kl_act']:.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  |  per-city(last50) {per_city}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            _save_and_log(out, tag, rssm, heads, opt, sched, step, norm, extra_meta=extra_meta)
        if max_hours is not None and (time.time() - t0) > max_hours * 3600:
            path = _save_and_log(out, tag, rssm, heads, opt, sched, step, norm,
                                 extra_meta=extra_meta, final=True)
            print(f"[stage2r] max-hours {max_hours} reached at step {step}; "
                  f"saved -> {path} (chain next session with --resume)", flush=True)
            return

    path = _save_and_log(out, tag, rssm, heads, opt, sched, steps - 1, norm,
                         extra_meta=extra_meta, final=True)
    print(f"[stage2r] saved -> {path}")


def _synthetic_feats(B=4, L=12, obs_dim=256, road_dim=256, K=10, n_roads=50, seed=0):
    """Shared by stage2r/stage3 smoke tests: self-consistent synthetic frozen
    features (pos_idx slot genuinely nearest so soft/physics targets are learnable)."""
    torch.manual_seed(seed)
    road_z = torch.randn(n_roads, road_dim)
    pos_idx = torch.randint(0, K, (B, L))
    cand_segment_id = torch.randint(0, n_roads, (B, L, K))
    cand_d_perp_m = torch.rand(B, L, K) * 40 + 5
    cand_d_perp_m.scatter_(-1, pos_idx.unsqueeze(-1), torch.rand(B, L, 1) * 2)
    road_embed = road_z[cand_segment_id.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)]
    feats = {"z1": torch.randn(B, L, obs_dim), "coords": torch.randn(B, L, 2),
             "pos_idx": pos_idx, "has_cand": torch.ones(B, L) > 0.5,
             "road_embed": road_embed, "valid": torch.ones(B, L) > 0.5,
             "speed_mps": torch.rand(B, L) * 10,
             "seq_mask": torch.ones(B, L),
             "bearing_deg": torch.rand(B, L) * 360,
             "cand_segment_id": cand_segment_id,
             "cand_heading_deg": torch.rand(B, L, K) * 360,
             "cand_speed_kph": torch.rand(B, L, K) * 80 + 20,
             "cand_oneway": torch.randint(0, 2, (B, L, K)),
             "cand_highway_id": torch.randint(0, 50, (B, L, K)),
             "cand_mask": torch.ones(B, L, K) > 0.5,
             "cand_d_perp_m": cand_d_perp_m}
    edge = torch.randint(0, n_roads, (2, 200))
    succ = SuccessorTable(edge, n_roads)
    return feats, road_z, succ


def _smoke():
    """Decoder-light objective drops under gradient descent; every component
    (zgps, road, prior_road, reward MSE, KL) finite and learning."""
    from dataset.config import RewardConfig
    feats, road_z, succ = _synthetic_feats()
    reward_cfg = RewardConfig()
    rssm = RSSM2(obs_dim=256, road_dim=256, norm="batch")
    heads = HeadsLight(h_dim=rssm.h_dim, road_dim=256)
    opt = torch.optim.AdamW(list(rssm.parameters()) + list(heads.parameters()), lr=3e-3)

    first = last = None
    for i in range(150):
        opt.zero_grad()
        out_l = wm_losses(rssm, heads, feats, road_z, succ, reward_cfg)
        out_l["total"].backward()
        torch.nn.utils.clip_grad_norm_(list(rssm.parameters()) + list(heads.parameters()), 100.0)
        opt.step()
        if first is None:
            first = float(out_l["total"])
        last = float(out_l["total"])
    print(f"[smoke] total {first:.3f} -> {last:.3f}  zgps {out_l['zgps']:.3f} "
          f"speed {out_l['speed']:.3f} road {out_l['road']:.3f} "
          f"prior_road {out_l['prior_road']:.3f} reward {out_l['reward']:.3f} "
          f"kl {out_l['kl']:.3f} kl_dyn_raw {out_l['kl_dyn_raw']:.3f} "
          f"kl_rep_raw {out_l['kl_rep_raw']:.3f} cl_gps_z {out_l['cl_gps_z']:.3f} "
          f"tgt_ent {out_l['tgt_ent']:.3f} road_ex {out_l['road_ex']:.3f} "
          f"kl_act {out_l['kl_act']:.3f}")
    # total has a ~9.6-nat irreducible KL free-bits floor (Change 11: 16 groups
    # clamped at >=1 nat, 0.5 dyn + 0.1 rep) -- assert on the above-floor part.
    kl_floor = 9.6
    assert last - kl_floor < 0.6 * (first - kl_floor), \
        f"above-floor loss did not drop enough: {first:.3f} -> {last:.3f}"
    assert float(out_l["reward"]) < 2.0, f"reward head did not learn: {out_l['reward']:.3f}"
    assert float(out_l["prior_road"]) < 2.0, f"prior_road did not learn: {out_l['prior_road']:.3f}"
    print("[smoke] PASS")

    # scheduled sampling + sigma-sharpening: not the default path (ss_prob=0,
    # sigma=10 above), so check separately that both still train without NaNs
    # and that a sharper sigma lowers tgt_ent (the CE floor) as intended.
    rssm2 = RSSM2(obs_dim=256, road_dim=256, norm="batch")
    heads2 = HeadsLight(h_dim=rssm2.h_dim, road_dim=256)
    opt2 = torch.optim.AdamW(list(rssm2.parameters()) + list(heads2.parameters()), lr=3e-3)
    rssm2.train(); heads2.train()
    first2 = last2 = None
    for i in range(150):
        opt2.zero_grad()
        out_l2 = wm_losses(rssm2, heads2, feats, road_z, succ, reward_cfg,
                           road_target_sigma=6.0, ss_prob=0.5)
        assert torch.isfinite(out_l2["total"]), "ss_prob/sharpened-sigma path produced a non-finite loss"
        out_l2["total"].backward()
        torch.nn.utils.clip_grad_norm_(list(rssm2.parameters()) + list(heads2.parameters()), 100.0)
        opt2.step()
        if first2 is None:
            first2 = float(out_l2["total"])
        last2 = float(out_l2["total"])
    assert last2 - kl_floor < 0.6 * (first2 - kl_floor), \
        f"ss_prob=0.5 path did not learn: {first2:.3f} -> {last2:.3f}"
    assert float(out_l2["tgt_ent"]) < float(out_l["tgt_ent"]), \
        f"sigma=6 should sharpen (lower) tgt_ent vs sigma=10: {out_l2['tgt_ent']:.3f} !< {out_l['tgt_ent']:.3f}"
    print(f"[smoke] ss_prob=0.5/sigma=6 total {first2:.3f} -> {last2:.3f}  "
          f"tgt_ent {out_l2['tgt_ent']:.3f} (< default {out_l['tgt_ent']:.3f})")
    print("[smoke] ss_prob/sigma-sharpening PASS")


def main():
    p = argparse.ArgumentParser(description="research2 decoder-light world-model pretraining")
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
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--constant-lr", action="store_true",
                   help="P0 LR-tail probe: hold LR flat (no warmup/cosine); on --resume, ignore saved sched")
    p.add_argument("--stoch-groups", type=int, default=16,
                   help="P2 latent width: categorical groups (16 default, 32 = DreamerV3-XS)")
    p.add_argument("--stoch-classes", type=int, default=16,
                   help="P2 latent width: classes per group")
    p.add_argument("--unfreeze-encoder", action="store_true",
                   help="Silesia confound probe: fine-tune the Stage-0 RoadGAT encoder "
                        "jointly (road CE + reward MSE gradients) instead of using frozen "
                        "precomputed embeddings. --resume restores road_enc too if present. "
                        "Not supported together with --multi-city.")
    p.add_argument("--road-target-sigma", type=float, default=10.0,
                   help="ASTRA-style pseudo-label sharpening: Gaussian bandwidth (m) for the "
                        "road-CE soft target, softmax(-d_perp^2/2*sigma^2). Default 10.0 = "
                        "unchanged GPS-noise-scale target. Smaller = sharper/lower-entropy "
                        "pseudo-label (see literature_papers/astra_self_training_weak_supervision.json).")
    p.add_argument("--ss-max-prob", type=float, default=0.0,
                   help="Scheduled sampling: max per-step probability of feeding the model's "
                        "OWN previous road-head argmax pick instead of the ground-truth pos_idx "
                        "pseudo-label as previous-state conditioning. Default 0.0 = old behavior "
                        "(pure teacher forcing). Diagnosed motivation: diag_jump_vs_priorerror.py "
                        "found online disconnected jumps are 4.5x likelier right after the "
                        "model's own choice was wrong -- this closes that train/inference gap.")
    p.add_argument("--ss-ramp-steps", type=int, default=4000,
                   help="Linear ramp length (in steps) from ss_prob=0 to --ss-max-prob.")
    p.add_argument("--multi-city", nargs="+", default=None, metavar="CITY:SOURCE",
                    help="Hybrid-cities training: space-separated city:source pairs, e.g. "
                         "'porto:porto tdrive:tdrive tdrive:geolife_car cabspotting:cabspotting "
                         "romataxi:romataxi' (city = OSM graph dir, source = processed parquet "
                         "dir -- geolife_car reuses the tdrive graph). One shared RSSM/HeadsLight "
                         "trained across all of them; --city/--source/--unfreeze-encoder ignored.")
    p.add_argument("--city-weights", type=float, nargs="+", default=None,
                    help="one weight per --multi-city entry, same order (default: "
                         "sqrt(dataset size), so the largest city doesn't drown out the rest)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        _smoke(); return
    if a.multi_city:
        city_sources = []
        for tok in a.multi_city:
            city, _, source = tok.partition(":")
            if not source:
                raise SystemExit(f"--multi-city entries must be 'city:source', got {tok!r}")
            city_sources.append((city, source))
        train_multi(a.processed_root, a.osm_root, a.stage0_ckpt, a.stage1_ckpt, city_sources,
                    city_weights=a.city_weights, limit_trajs=a.limit_trajs, batch=a.batch,
                    steps=a.steps, curriculum_step=a.curriculum_step, seq_len=a.seq_len,
                    curriculum_len=a.curriculum_len, lr=a.lr, warmup_steps=a.warmup_steps,
                    beta=a.beta, lambda_prior_road=a.lambda_prior_road,
                    lambda_reward=a.lambda_reward, road_keep_prob=a.road_keep_prob,
                    norm=a.norm, grad_clip=a.grad_clip, out_dir=a.out_dir,
                    ckpt_every=a.ckpt_every, log_every=a.log_every, device=a.device,
                    workers=a.workers, resume=a.resume, max_hours=a.max_hours,
                    constant_lr=a.constant_lr, stoch_groups=a.stoch_groups,
                    stoch_classes=a.stoch_classes, road_target_sigma=a.road_target_sigma,
                    ss_max_prob=a.ss_max_prob, ss_ramp_steps=a.ss_ramp_steps)
        return
    train(a.processed_root, a.osm_root, a.stage0_ckpt, a.stage1_ckpt, a.city, a.source,
          a.limit_trajs, a.batch, a.steps, a.curriculum_step, a.seq_len, a.curriculum_len,
          a.lr, a.warmup_steps, a.beta, a.lambda_prior_road, a.lambda_reward,
          a.road_keep_prob, a.norm, a.grad_clip,
          out_dir=a.out_dir, ckpt_every=a.ckpt_every, log_every=a.log_every,
          device=a.device, workers=a.workers, resume=a.resume, max_hours=a.max_hours,
          constant_lr=a.constant_lr, stoch_groups=a.stoch_groups, stoch_classes=a.stoch_classes,
          unfreeze_encoder=a.unfreeze_encoder, road_target_sigma=a.road_target_sigma,
          ss_max_prob=a.ss_max_prob, ss_ramp_steps=a.ss_ramp_steps)


if __name__ == "__main__":
    main()
