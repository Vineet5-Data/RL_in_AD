"""Stage-2 (P2) pretraining: DreamerV3-style RSSM world model over frozen P1a/P1b.

Label-free (P3): the only "targets" are physical/geometric -- nearest real
candidate road by perpendicular distance, real GPS coords, real speed. No
human-annotated or pre-matched ground truth anywhere in this file.

Frozen inputs, nothing here trains them:
  Stage-0 (P1b) RoadGAT  -> road_z[N, road_dim]           per-segment embedding
  Stage-1 (P1a) encoder  -> z1[B,L,d]                      GPS-road joint latent (enc(o_t))
  real candidates        -> pseudo road id = argmin(d_perp) per fix (label-free)

Recovery run (stage2_correction.md Changes 1/2/3/6): free-bits KL, LR 1e-4 +
warmup, MDN residual GPS head, extra logging columns. Change 7 (de-leak):
7A candidate permutation is loader-level (trajectories.py), 7B replaces the
road head with a permutation-equivariant candidate scorer (world_model.py
RoadScorer), 7E fixes Hit@1 to segment-id comparison (eval-side). Road losses
(posterior + Change-4 prior aux) use Change-9 soft emission targets, ported
from Stage-1's de-leaked retrain. Change 5 is the eval harness
(the retired Track-A eval harness), fixed there separately.

Run:  python -m training.stage2 --processed-root data/processed --osm-root data/osm --stage0-ckpt ckpt/stage0_porto.pt --stage1-ckpt ckpt/stage1_porto.pt
Smoke: python -m training.stage2 --smoke
"""

from __future__ import annotations

# pyarrow must load before torch (see training/stage0.py -- Windows segfault guard)
import pyarrow.parquet  # noqa: F401

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from models.world_model import (RSSM, Heads, straight_through_sample, kl_categorical_pergroup,
                                     free_bits_kl, mdn_nll, mdn_mean)
except ImportError:
    from world_model import (RSSM, Heads, straight_through_sample, kl_categorical_pergroup,  # type: ignore
                              free_bits_kl, mdn_nll, mdn_mean)

try:
    from training.stage1 import _road_embeddings, build_loader
except ImportError:
    from stage1 import _road_embeddings, build_loader  # type: ignore


def _stage1_encoder(road_dim: int, ckpt: str, device: str):
    """Load the frozen Stage-1 (P1a) encoder. Eval mode, no grad -- ponytail: this
    IS the frozen-weights contract, not a shortcut."""
    from models.gps_encoder import Stage1Encoder

    enc = Stage1Encoder(road_dim=road_dim).to(device)
    enc.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class LengthCapped(Dataset):
    """Truncate each item to its first L fixes -- Stage-2's curriculum knob.

    TrajectoryGraphDataset's candidate cache covers the whole df regardless of
    max_len (see `_load_or_build_cache`), so swapping L here is free: no
    recompute, no new cache file, just a shorter slice per __getitem__.
    """

    def __init__(self, base: Dataset, L: int):
        self.base, self.L = base, L

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        item = dict(self.base[i])
        L = min(self.L, item["length"])
        for k, v in item.items():
            if torch.is_tensor(v) and v.dim() >= 1:
                item[k] = v[:L]
        item["length"] = L
        return item


def extract_frozen_features(stage1, road_z: torch.Tensor, batch: dict) -> dict:
    """Glue: run the frozen Stage-1 encoder + geometric nearest-candidate lookup.
    Everything downstream (`rssm_losses`) only needs this dict -- keeps the RSSM
    math testable without a real Stage-1 checkpoint (see `_smoke`).

    Also carries the raw per-fix candidate arrays (segment_id/heading/oneway/
    highway_id/mask/d_perp) needed by the Change 7B RoadScorer + Change 9
    soft-target road losses in `rssm_losses` -- these are NOT the Stage-1
    internal `cand` tensor (that's Stage-1's own d-dim projection), just the
    same raw cache fields Stage-1 itself reads."""
    with torch.no_grad():
        z1, _cand, coords, _mgm, cand_pad = stage1.forward(batch, road_z, apply_mask=False)
    valid = batch["seq_mask"] > 0.5
    dp = batch["cand_d_perp_m"].masked_fill(cand_pad, float("inf"))
    pos_idx = dp.argmin(-1)                                   # [B,L] nearest candidate index
    has_cand = (~cand_pad).any(-1) & valid
    pseudo_seg = batch["cand_segment_id"].gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)
    road_embed = road_z[pseudo_seg.clamp(min=0)]              # [B,L,road_dim] destination road at t
    return {
        "z1": z1, "coords": coords, "pos_idx": pos_idx, "has_cand": has_cand,
        "road_embed": road_embed, "valid": valid,
        "speed_mps": batch["speed_mps"], "seq_mask": batch["seq_mask"],
        "cand_segment_id": batch["cand_segment_id"], "cand_heading_deg": batch["cand_heading_deg"],
        "cand_oneway": batch["cand_oneway"], "cand_highway_id": batch["cand_highway_id"],
        "cand_mask": ~cand_pad, "cand_d_perp_m": batch["cand_d_perp_m"],
    }


def _soft_road_ce(logits: torch.Tensor, d_perp: torch.Tensor, cand_mask: torch.Tensor,
                   sigma: float = 10.0) -> torch.Tensor:
    """Change 9 (ported from Stage-1's gps_encoder.py): CE against a soft
    emission target softmax(-d_perp^2 / 2*sigma^2), sigma=10m (GPS noise
    scale) by default, instead of hard argmin. logits/d_perp/cand_mask:
    [B,K] for a single timestep -> per-sample CE [B]. NOTE: masked_fill uses
    a large finite constant, not +-inf -- soft_tgt is exactly 0 on masked
    slots and logp is ~-inf there; 0 * -inf = NaN, 0 * finite = 0.

    `sigma` < 10 sharpens the target (ASTRA-style confidence sharpening of a
    noisy weak-supervision soft label, literature_papers/
    astra_self_training_weak_supervision.json) -- lowers the pseudo-label's
    entropy floor with zero new data/params. Default unchanged for callers
    that don't pass it (stage2.py's own frozen Track-A path)."""
    logits = logits.masked_fill(~cand_mask, -1e4)
    d = d_perp.masked_fill(~cand_mask, 1e4)
    soft_tgt = F.softmax(-(d ** 2) / (2 * sigma ** 2), dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_tgt * logp).sum(-1)


def rssm_losses(rssm: RSSM, heads: Heads, feats: dict, road_z: torch.Tensor, beta: float = 1.0,
                 lambda_mdn: float = 1.0, lambda_prior_road: float = 0.5,
                 road_keep_prob: float = 0.6, lambda_zonly_road: float = 0.2,
                 lambda_zonly_gps: float = 0.2, return_components: bool = False) -> dict:
    """Unroll the RSSM over the (already truncated) sequence, teacher-forced by
    the posterior (real obs), and accumulate the corrected ELBO
    (stage2_correction.md Changes 1/3/4):
      -log p(o_t|h_t,z_t) ~= GPS Huber (coarse anchor, w=1) + MDN residual NLL
                              (w=lambda_mdn) + speed MSE + posterior road CE
                              (w=1, display-only per Change 0 -- not gated on)
      L_kl = 0.5*kl_dyn_freebits + 0.1*kl_rep_freebits   (Change 11: per-group
             free bits -- each categorical group clamped to >=1 nat, then
             summed, giving the 16x16 latent a 16-nat floor)
      L_prior_road = CE(road_head(h_t, z_prior_t), pos_idx_t) * lambda_prior_road
             (Change 4, MANDATORY for Stage 3: prior-only sample through the
             SAME road head -- the only gradient that makes the GRU/prior
             genuinely road-predictive without seeing real obs; posterior road
             CE above never touches the prior, so it can't do this on its own)
    """
    z1, coords, pos_idx = feats["z1"], feats["coords"], feats["pos_idx"]
    has_cand, road_embed, valid = feats["has_cand"], feats["road_embed"], feats["valid"]
    speed_mps = feats["speed_mps"]
    cand_seg, cand_head = feats["cand_segment_id"], feats["cand_heading_deg"]
    cand_oneway, cand_hw = feats["cand_oneway"], feats["cand_highway_id"]
    cand_mask, cand_dperp = feats["cand_mask"], feats["cand_d_perp_m"]
    device = z1.device
    B, L = valid.shape

    h, z = rssm.initial(B, device)
    recon_gps = recon_speed = recon_road = torch.zeros((), device=device)
    zonly_road_loss = torch.zeros((), device=device)
    zonly_gps_loss = torch.zeros((), device=device)  # Change 14A
    mdn_loss = kl_loss = prior_road_loss = torch.zeros((), device=device)
    kl_dyn_raw_sum = kl_rep_raw_sum = torch.zeros((), device=device)
    cl_gps_coarse_sum = cl_gps_full_sum = torch.zeros((), device=device)
    n_valid = n_road = 0
    log_speed = torch.log1p(speed_mps.clamp(min=0))

    for t in range(L):
        if t > 0:
            # road_embed[:, t-1] (previous, already-resolved fix) -- NOT road_embed[:, t].
            # Both road_embed[:,t] and pos_idx[:,t] are argmin(d_perp) at the SAME real
            # GPS fix t; feeding road_embed[:,t] into h before decoding pos_idx[:,t] from
            # that h is a same-timestep tautology (road loss collapses to ~0 immediately,
            # not genuine learning). Shifting by one makes it a real predict-from-history task.
            re_t = road_embed[:, t - 1]
            if rssm.training:
                keep = (torch.rand(B, device=device) < road_keep_prob).unsqueeze(-1)
                re_t = torch.where(keep, re_t, rssm.road_mask_token)
            h = rssm.step(h, z, re_t)
        prior_logits = rssm.prior(h)
        post_logits = rssm.posterior(h, z1[:, t])
        z_sample = straight_through_sample(post_logits)
        z = z_sample.reshape(B, -1)
        gps_hat, speed_hat, mdn_out = heads(h, z)
        pi_l, mu, log_sig = mdn_out
        # Change 7B: road scorer needs this timestep's raw candidate context
        # (segment_id/heading/oneway/highway_id/mask) -- not part of heads(h,z).
        road_logits = heads.road(h, z, road_z, cand_seg[:, t], cand_head[:, t],
                                  cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])
        road_logits_zonly = heads.road(torch.zeros_like(h), z, road_z, cand_seg[:, t],
                                       cand_head[:, t], cand_oneway[:, t], cand_hw[:, t],
                                       cand_mask[:, t])

        vt = valid[:, t]
        if vt.any():
            recon_gps = recon_gps + F.huber_loss(gps_hat[vt], coords[:, t][vt], reduction="sum")
            recon_speed = recon_speed + F.mse_loss(speed_hat[vt], log_speed[:, t][vt], reduction="sum")

            # Change 14A: z-only coarse GPS decode -- h zeroed so the deterministic
            # path can't freeload on GPS recon (MuDreamer direction). heads.gps is a
            # plain Sequential over the pre-concatenated [h, z_flat]; z is the real
            # posterior sample, h is zeroed. Coarse head ONLY (not the MDN), same
            # vt-masked huber(reduction="sum")/n_valid pattern as recon_gps above.
            gps_hat_zonly = heads.gps(torch.cat([torch.zeros_like(h), z], dim=-1))
            zonly_gps_loss = zonly_gps_loss + F.huber_loss(gps_hat_zonly[vt], coords[:, t][vt], reduction="sum")

            # MDN residual: target = true - coarse.detach() -- MDN never backprops
            # into the coarse gps head, only into its own (pi,mu,log_sigma) params.
            residual = coords[:, t] - gps_hat.detach()
            nll = mdn_nll(pi_l, mu, log_sig, residual)
            mdn_loss = mdn_loss + nll[vt].sum()

            # Change 11: per-group free bits. Clamp every categorical group at
            # >=1 nat, then sum groups so the 16x16 latent gets a 16-nat floor.
            kl_dyn_pg = kl_categorical_pergroup(post_logits.detach(), prior_logits)   # [B,G], trains prior
            kl_rep_pg = kl_categorical_pergroup(post_logits, prior_logits.detach())   # [B,G], trains posterior
            kl_dyn_raw_sum = kl_dyn_raw_sum + kl_dyn_pg[vt].sum()
            kl_rep_raw_sum = kl_rep_raw_sum + kl_rep_pg[vt].sum()
            kl_t = free_bits_kl(kl_dyn_pg, kl_rep_pg)
            kl_loss = kl_loss + kl_t[vt].sum()

            with torch.no_grad():
                cl_gps_coarse_sum = cl_gps_coarse_sum + (gps_hat[vt] - coords[:, t][vt]).norm(dim=-1).sum()
                full_hat = gps_hat.detach() + mdn_mean(pi_l, mu)
                cl_gps_full_sum = cl_gps_full_sum + (full_hat[vt] - coords[:, t][vt]).norm(dim=-1).sum()

            # Change 4: prior-side road auxiliary -- separate sample from the PRIOR
            # (never sees z1/real obs), through the SAME road head (Change 7B
            # scorer, same candidate context), discarded after this loss (does
            # not replace the posterior `z` driving the rollout).
            z_prior = straight_through_sample(prior_logits).reshape(B, -1)
            road_logits_prior = heads.road(h, z_prior, road_z, cand_seg[:, t], cand_head[:, t],
                                            cand_oneway[:, t], cand_hw[:, t], cand_mask[:, t])
            hc_kl = vt & has_cand[:, t]
            if hc_kl.any():
                # Change 9 (ported): soft emission target, not hard argmin.
                ce_prior = _soft_road_ce(road_logits_prior, cand_dperp[:, t], cand_mask[:, t])
                prior_road_loss = prior_road_loss + ce_prior[hc_kl].sum()

            n_valid += int(vt.sum())
        hc = has_cand[:, t]
        if hc.any():
            ce_post = _soft_road_ce(road_logits, cand_dperp[:, t], cand_mask[:, t])
            ce_zonly = _soft_road_ce(road_logits_zonly, cand_dperp[:, t], cand_mask[:, t])
            recon_road = recon_road + ce_post[hc].sum()
            zonly_road_loss = zonly_road_loss + ce_zonly[hc].sum()
            n_road += int(hc.sum())

    n_valid, n_road = max(n_valid, 1), max(n_road, 1)
    recon_gps, recon_speed = recon_gps / n_valid, recon_speed / n_valid
    mdn_loss, kl_loss = mdn_loss / n_valid, kl_loss / n_valid
    kl_dyn_raw, kl_rep_raw = kl_dyn_raw_sum / n_valid, kl_rep_raw_sum / n_valid
    cl_gps_coarse, cl_gps_full = cl_gps_coarse_sum / n_valid, cl_gps_full_sum / n_valid
    recon_road = recon_road / n_road
    prior_road_loss = prior_road_loss / n_road  # same has_cand denominator as posterior road
    zonly_road_loss = zonly_road_loss / n_road
    zonly_gps_loss = zonly_gps_loss / n_valid   # Change 14A: same denominator as coarse gps

    total = (recon_gps + lambda_mdn * mdn_loss + recon_speed + recon_road
             + beta * kl_loss + lambda_prior_road * prior_road_loss
             + lambda_zonly_road * zonly_road_loss + lambda_zonly_gps * zonly_gps_loss)
    out = {"total": total, "gps": recon_gps.detach(), "speed": recon_speed.detach(),
            "road": recon_road.detach(), "kl": kl_loss.detach(),
            "kl_dyn_raw": kl_dyn_raw.detach(), "kl_rep_raw": kl_rep_raw.detach(),
            "prior_road": prior_road_loss.detach(), "mdn_nll": mdn_loss.detach(),
            "zonly_road": zonly_road_loss.detach(), "zonly_gps": zonly_gps_loss.detach(),
            "cl_gps_coarse": cl_gps_coarse.detach(), "cl_gps_full": cl_gps_full.detach()}
    if return_components:
        # Change 14C: undetached loss-group tensors for the gradient-competition
        # probe -- (gps Huber + MDN NLL) vs (road CE + prior_road CE), weighted as
        # in `total`. Only built on request so the normal path stays graph-free.
        out["_gps_group"] = recon_gps + lambda_mdn * mdn_loss
        out["_road_group"] = recon_road + lambda_prior_road * prior_road_loss
    return out


def _grad_competition(rssm, heads, stage1, road_z, b, probe_params, loss_kw):
    """Change 14C: cosine + norm-ratio of the (gps Huber + MDN NLL) vs (road CE +
    prior_road CE) gradients w.r.t. the shared-trunk probe (GRU params + post_net
    first layer). Runs its OWN forward on batch `b` and its own graph via
    torch.autograd.grad(retain_graph=...) -- it never calls backward, so param
    .grad is untouched and the real training gradients are unchanged. Guards zero
    norms. Diagnostic only."""
    feats = extract_frozen_features(stage1, road_z, b)
    out = rssm_losses(rssm, heads, feats, road_z, return_components=True, **loss_kw)
    g_gps = torch.autograd.grad(out["_gps_group"], probe_params, retain_graph=True, allow_unused=True)
    g_road = torch.autograd.grad(out["_road_group"], probe_params, allow_unused=True)

    def _flat(grads):
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(grads, probe_params)])

    a, r = _flat(g_gps), _flat(g_road)
    na, nr = float(a.norm()), float(r.norm())
    eps = 1e-12
    cos = float((a * r).sum() / (na * nr + eps)) if na > eps and nr > eps else 0.0
    ratio = na / (nr + eps) if nr > eps else 0.0
    return cos, ratio


def _z1_variance_preflight(rssm, stage1, road_z, loader_it, device):
    """Change 14D pre-flight: on one batch of different trajectories, per-dim
    variance ACROSS the batch of z1 (frozen Stage-1 latent) and of the RSSM
    posterior probs at t=0. Near-zero mean variance => trajectories collapse to a
    shared code (Stage-1/posterior audit before committing to a full run)."""
    b = next(loader_it)
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.no_grad():
        feats = extract_frozen_features(stage1, road_z, b)
        z1_0 = feats["z1"][:, 0]                                   # [B, obs_dim] first fix
        h0, _ = rssm.initial(z1_0.size(0), device)
        post0 = F.softmax(rssm.posterior(h0, z1_0), dim=-1)        # [B, G, C]
        z1_var = float(z1_0.var(dim=0, unbiased=False).mean())
        post_var = float(post0.var(dim=0, unbiased=False).mean())
    print(f"[stage2] 14D pre-flight (fresh init): z1 cross-traj var(mean)={z1_var:.6f}  "
          f"posterior-prob cross-traj var(mean)={post_var:.6f}  "
          f"(near-zero => Stage-1/posterior-collapse audit)", flush=True)


def _load_compatible(module, state, label):
    current = module.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in current and current[k].shape == v.shape
    }
    skipped = len(state) - len(compatible)
    current.update(compatible)
    module.load_state_dict(current)
    print(f"[stage2] init-compatible {label}: loaded={len(compatible)} skipped={skipped}")


def train(processed_root, osm_root, stage0_ckpt, stage1_ckpt, city="porto", source="porto",
          limit_trajs=200_000, batch=16, steps=40_000, curriculum_step=10_000, seq_len=64,
          curriculum_len=16, lr=1e-4, warmup_steps=1000, beta=1.0, lambda_mdn=1.0,
          lambda_prior_road=0.5, road_keep_prob=0.6, lambda_zonly_road=0.2,
          lambda_zonly_gps=0.2, aggressive_until=3000, n_inner=3,
          grad_clip=100.0, weight_decay=1e-4, out_dir="ckpt",
          ckpt_every=1000, log_every=50, device=None, seed=0, workers=0, resume=None,
          init_compatible_from=None):
    """Curriculum: L=`curriculum_len` for the first `curriculum_step` steps, then
    L=`seq_len` (methodology: L=16 -> L=64 after 10k steps).

    Recovery-run defaults (stage2_correction.md Change 2): lr=1e-4 (was 3e-4),
    linear warmup 0->lr over `warmup_steps` (was none), grad clip unchanged at 100.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    from dataset.config import RewardConfig  # reuse existing sigma_gps_m as the noise-floor reference
    gps_floor_m = RewardConfig().sigma_gps_m
    print(f"[stage2] GPS sensor-noise floor (reference): {gps_floor_m:.1f}m "
          f"({gps_floor_m / 1000:.4f} km) -- cl_gps below this is likely noise-limited, "
          f"not a real model deficiency")

    road_z = _road_embeddings(osm_root, city, stage0_ckpt, device)
    stage1 = _stage1_encoder(road_z.size(1), stage1_ckpt, device)
    base_ds, _ = build_loader(processed_root, osm_root, city, source, limit_trajs, batch,
                               workers=workers, cache_dir=str(out / "cache"))

    from dataset.trajectories import collate_fn
    from torch.utils.data import DataLoader

    def loader_at(L):
        ds = LengthCapped(base_ds, L)
        return iter(DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_fn,
                                num_workers=workers))

    rssm = RSSM(obs_dim=road_z.size(1), road_dim=road_z.size(1)).to(device)
    heads = Heads(h_dim=rssm.h_dim, road_dim=road_z.size(1)).to(device)
    if init_compatible_from is not None:
        ckpt = torch.load(init_compatible_from, map_location=device, weights_only=True)
        _load_compatible(rssm, ckpt["rssm"], "rssm")
        _load_compatible(heads, ckpt["heads"], "heads")
    params = list(rssm.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Change 14B: aggressive posterior training (He et al. 2019, schedule-only).
    # post_params = posterior-side trainable params. The observation encoder is the
    # FROZEN Stage-1 net (not optimized here), so rssm.post_net is the only trainable
    # posterior module -> post_params = post_net. Separate optimizer, lr 1e-4.
    post_params = list(rssm.post_net.parameters())
    opt_post = torch.optim.AdamW(post_params, lr=1e-4, weight_decay=weight_decay)
    # Change 14C: fixed probe = GRU params + post_net first layer (the shared trunk).
    probe_params = list(rssm.gru.parameters()) + [rssm.post_net[0].weight, rssm.post_net[0].bias]
    loss_kw = dict(beta=beta, lambda_mdn=lambda_mdn, lambda_prior_road=lambda_prior_road,
                   road_keep_prob=road_keep_prob, lambda_zonly_road=lambda_zonly_road,
                   lambda_zonly_gps=lambda_zonly_gps)
    # Change 2: linear warmup 0->lr over warmup_steps, then cosine decay to 0 over the rest.
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                                                total_iters=max(1, warmup_steps))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warmup_steps))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                   milestones=[warmup_steps])
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    start_step = 0
    if resume is not None and Path(resume).is_file():
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        rssm.load_state_dict(ckpt["rssm"]); heads.load_state_dict(ckpt["heads"])
        opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        print(f"[stage2] resumed from {resume} at step {start_step}")

    print(f"[stage2] device={device} bf16={use_bf16} steps={steps} curriculum={curriculum_len}"
          f"->{seq_len}@{curriculum_step} batch={batch} lr={lr:.1e} warmup={warmup_steps} "
          f"lambda_mdn={lambda_mdn} lambda_prior_road={lambda_prior_road}")
    it = loader_at(seq_len if start_step >= curriculum_step else curriculum_len)
    # Change 14D pre-flight: z1 / posterior cross-trajectory variance on one fresh
    # batch (default path is fresh init -- --resume/--init-compatible-from unchanged).
    _z1_variance_preflight(rssm, stage1, road_z,
                           loader_at(seq_len if start_step >= curriculum_step else curriculum_len),
                           device)
    rssm.train(); heads.train()
    for step in range(start_step, steps):
        if step == curriculum_step:
            it = loader_at(seq_len)
            print(f"[stage2] step {step}: curriculum L -> {seq_len}")
        try:
            b = next(it)
        except StopIteration:
            L = curriculum_len if step < curriculum_step else seq_len
            it = loader_at(L)
            b = next(it)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

        # Change 14B: aggressive posterior phase. Before each outer update, take
        # n_inner inner steps that update ONLY post_net (via opt_post) on the SAME
        # batch b -- re-forward (simplest; no extra iterator plumbing). After
        # aggressive_until, opt_post is unused and this is plain training.
        if step < aggressive_until:
            for _ in range(n_inner):
                opt.zero_grad(); opt_post.zero_grad()   # zero all grads; only post_net is stepped
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    feats_i = extract_frozen_features(stage1, road_z, b)
                    out_i = rssm_losses(rssm, heads, feats_i, road_z, **loss_kw)
                out_i["total"].backward()
                torch.nn.utils.clip_grad_norm_(post_params, grad_clip)
                opt_post.step()
            if n_inner > 0 and step % log_every == 0:
                print(f"        post_only_loss {float(out_i['total']):.4f}  "
                      f"(aggressive phase, n_inner={n_inner})", flush=True)
        elif step == aggressive_until:
            print(f"[stage2] step {step}: aggressive posterior phase ended "
                  f"(step >= aggressive_until={aggressive_until}); opt_post now unused", flush=True)

        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            feats = extract_frozen_features(stage1, road_z, b)
            out_l = rssm_losses(rssm, heads, feats, road_z, **loss_kw)
        out_l["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step(); sched.step()

        if step % log_every == 0:
            print(f"step {step:06d}  total {float(out_l['total']):.4f}  gps {out_l['gps']:.4f}  "
                  f"mdn_nll {out_l['mdn_nll']:.4f}  speed {out_l['speed']:.4f}  "
                  f"road {out_l['road']:.4f}  prior_road {out_l['prior_road']:.4f}  "
                  f"zonly_road {out_l['zonly_road']:.4f}  zonly_gps {out_l['zonly_gps']:.4f}  "
                  f"kl {out_l['kl']:.4f}  kl_dyn_raw {out_l['kl_dyn_raw']:.4f}  "
                  f"kl_rep_raw {out_l['kl_rep_raw']:.4f}  cl_gps_coarse_km {out_l['cl_gps_coarse']:.4f}  "
                  f"cl_gps_full_km {out_l['cl_gps_full']:.4f}  lr {sched.get_last_lr()[0]:.2e}",
                  flush=True)
            # Change 14C: gradient-competition diagnostic (measure only). Separate
            # forward + its own graph -> cannot perturb the training gradients. No
            # autocast so the cosine/ratio are clean fp32.
            cos, ratio = _grad_competition(rssm, heads, stage1, road_z, b, probe_params, loss_kw)
            print(f"        grad_cos_gpsroad {cos:.4f}  grad_ratio_gpsroad {ratio:.4f}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            _save_ckpt(out / f"stage2_{city}_step{step}.pt", rssm, heads, opt, sched, step)

    _save_ckpt(out / f"stage2_{city}.pt", rssm, heads, opt, sched, steps - 1)
    print(f"[stage2] saved -> {out / f'stage2_{city}.pt'}")


def _save_ckpt(path, rssm, heads, opt, sched, step):
    """rssm/heads state dicts are the frozen-for-Stage-3 artifact; opt/sched/step
    are only there so --resume can restart mid-run (methodology: 'Resume from
    checkpoint mandatory' -- Kaggle/local sessions can die mid-training)."""
    torch.save({"rssm": rssm.state_dict(), "heads": heads.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, path)


def _smoke():
    """RSSM math + training loop, no real Stage-1/data -- synthetic frozen features.
    Validates every corrected term (Change 1/3/4): free-bits KL, MDN NLL, prior_road
    all drop under gradient descent, not just the original gps/speed/road."""
    torch.manual_seed(0)
    B, L, obs_dim, road_dim, K, n_roads = 4, 12, 256, 256, 10, 50

    rssm = RSSM(obs_dim=obs_dim, road_dim=road_dim)
    heads = Heads(h_dim=rssm.h_dim, road_dim=road_dim)
    opt = torch.optim.AdamW(list(rssm.parameters()) + list(heads.parameters()), lr=3e-3)

    # synthetic but self-consistent "frozen features": a fixed random target per
    # trajectory so recon/road losses have a learnable, non-degenerate signal.
    road_z = torch.randn(n_roads, road_dim)
    z1 = torch.randn(B, L, obs_dim)
    coords = torch.randn(B, L, 2)
    pos_idx = torch.randint(0, K, (B, L))
    speed_mps = torch.rand(B, L) * 10
    valid = torch.ones(B, L) > 0.5
    has_cand = torch.ones(B, L) > 0.5
    cand_segment_id = torch.randint(0, n_roads, (B, L, K))
    cand_heading_deg = torch.rand(B, L, K) * 360
    cand_oneway = torch.randint(0, 2, (B, L, K))
    cand_highway_id = torch.randint(0, 50, (B, L, K))
    cand_mask = torch.ones(B, L, K) > 0.5
    # d_perp: make the pos_idx slot genuinely nearest so soft targets are learnable.
    cand_d_perp_m = torch.rand(B, L, K) * 40 + 5
    cand_d_perp_m.scatter_(-1, pos_idx.unsqueeze(-1), torch.rand(B, L, 1) * 2)
    road_embed = road_z[cand_segment_id.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)]
    feats = {"z1": z1, "coords": coords, "pos_idx": pos_idx, "has_cand": has_cand,
             "road_embed": road_embed, "valid": valid, "speed_mps": speed_mps,
             "seq_mask": valid.float(), "cand_segment_id": cand_segment_id,
             "cand_heading_deg": cand_heading_deg, "cand_oneway": cand_oneway,
             "cand_highway_id": cand_highway_id, "cand_mask": cand_mask,
             "cand_d_perp_m": cand_d_perp_m}

    first = last = None
    for i in range(150):
        opt.zero_grad()
        out_l = rssm_losses(rssm, heads, feats, road_z, beta=1.0, lambda_mdn=1.0, lambda_prior_road=0.5)
        out_l["total"].backward()
        torch.nn.utils.clip_grad_norm_(list(rssm.parameters()) + list(heads.parameters()), 100.0)
        opt.step()
        if first is None:
            first = float(out_l["total"])
        last = float(out_l["total"])
    print(f"[smoke] total {first:.3f} -> {last:.3f}  gps {out_l['gps']:.3f} "
          f"mdn_nll {out_l['mdn_nll']:.3f} speed {out_l['speed']:.3f} road {out_l['road']:.3f} "
          f"prior_road {out_l['prior_road']:.3f} zonly_road {out_l['zonly_road']:.3f} kl {out_l['kl']:.3f} "
          f"kl_dyn_raw {out_l['kl_dyn_raw']:.3f} kl_rep_raw {out_l['kl_rep_raw']:.3f} "
          f"cl_gps_coarse {out_l['cl_gps_coarse']:.3f} cl_gps_full {out_l['cl_gps_full']:.3f}")
    assert last < 0.6 * first, f"loss did not drop enough: {first:.3f} -> {last:.3f}"
    assert float(out_l["mdn_nll"]) < 5.0, f"mdn_nll did not converge: {out_l['mdn_nll']:.3f}"
    assert float(out_l["prior_road"]) < 2.0, f"prior_road did not learn: {out_l['prior_road']:.3f}"
    print("[smoke] PASS")


def main():
    p = argparse.ArgumentParser(description="Stage-2 RSSM world-model pretraining")
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
    p.add_argument("--lambda-mdn", type=float, default=1.0)
    p.add_argument("--lambda-prior-road", type=float, default=0.5)
    p.add_argument("--road-keep-prob", type=float, default=0.6)
    p.add_argument("--lambda-zonly-road", type=float, default=0.2)
    p.add_argument("--lambda-zonly-gps", type=float, default=0.2)   # Change 14A (mirror of zonly-road)
    p.add_argument("--aggressive-until", type=int, default=3000)    # Change 14B
    p.add_argument("--n-inner", type=int, default=3)                # Change 14B
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--out-dir", default="ckpt")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume", default=None, help="checkpoint path to resume step/opt/sched from")
    p.add_argument("--init-compatible-from", default=None,
                   help="checkpoint path used only to warm-start tensors with matching names/shapes")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        _smoke(); return
    train(a.processed_root, a.osm_root, a.stage0_ckpt, a.stage1_ckpt, a.city, a.source,
          a.limit_trajs, a.batch, a.steps, a.curriculum_step, a.seq_len, a.curriculum_len,
          a.lr, a.warmup_steps, a.beta, a.lambda_mdn, a.lambda_prior_road,
          a.road_keep_prob, a.lambda_zonly_road,
          lambda_zonly_gps=a.lambda_zonly_gps, aggressive_until=a.aggressive_until,
          n_inner=a.n_inner, grad_clip=a.grad_clip,
          out_dir=a.out_dir, ckpt_every=a.ckpt_every, log_every=a.log_every, device=a.device,
          workers=a.workers, resume=a.resume, init_compatible_from=a.init_compatible_from)


if __name__ == "__main__":
    main()
