"""P2 latent world model (Stage 2, methodology 2.3): DreamerV3-style RSSM.

State s_t = (h_t, z_t):
  h_t in R^512        deterministic GRU hidden
  z_t in {0,1}^16x16   stochastic categorical latent (straight-through), 16x16
                       per the RTX-4060 mitigation (methodology Sec.5), not the
                       16x16 in the pure spec (Sec.2.3); stage2_new implements
                       the stage2_correction.md Run 2 escalation.

Road-conditioned transition: h_{t+1} = GRU(h_t, [z_t, h_{a_t}]); h_{a_t} = frozen
Stage-0 (P1b) embedding of the road segment the trajectory occupies at t+1
(nearest real candidate by perpendicular distance -- label-free, P3).

Frozen inputs only: this module trains nothing from Stage 0/1, both stay eval().
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STOCH_GROUPS = 16
STOCH_CLASSES = 16
STOCH_DIM = STOCH_GROUPS * STOCH_CLASSES  # 256, flattened one-hot code
UNIMIX = 0.01  # DreamerV3 stabilizer: 1% uniform floor on every categorical


def _mixed_probs(logits: torch.Tensor) -> torch.Tensor:
    """softmax(logits) mixed with a uniform floor -- keeps every class's prob >0
    so KL/log terms never blow up and the categorical can't go fully deterministic
    (a real risk we saw empirically: kl dropped to ~0.006 at step 50-150 of the
    first Stage-2 run before recovering -- unimix removes that near-collapse mode)."""
    probs = F.softmax(logits, dim=-1)
    u = 1.0 / probs.size(-1)
    return (1.0 - UNIMIX) * probs + UNIMIX * u


def straight_through_sample(logits: torch.Tensor) -> torch.Tensor:
    """logits [..., G, C] -> one-hot sample [..., G, C], straight-through gradient
    (forward = hard categorical sample, backward = mixed-softmax Jacobian)."""
    probs = _mixed_probs(logits)
    idx = torch.distributions.Categorical(probs=probs).sample()
    hard = F.one_hot(idx, probs.size(-1)).float()
    return hard + probs - probs.detach()


def kl_categorical_pergroup(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per categorical group. p_logits/q_logits: [..., G, C] -> [..., G]."""
    p, q = _mixed_probs(p_logits), _mixed_probs(q_logits)
    return (p * (p.log() - q.log())).sum(-1)  # unimix floor keeps p,q > 0 -- safe log


def kl_categorical(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) summed over categorical groups. p_logits/q_logits: [..., G, C] -> [...]."""
    return kl_categorical_pergroup(p_logits, q_logits).sum(-1)


def free_bits_kl(kl_dyn_pg: torch.Tensor, kl_rep_pg: torch.Tensor,
                  w_dyn: float = 0.5, w_rep: float = 0.1) -> torch.Tensor:
    """DreamerV3 per-group free-bits KL loss: dyn/rep split, 1-nat floor per group,
    summed over groups. kl_dyn_pg/kl_rep_pg: [..., G] -> [...]."""
    return w_dyn * kl_dyn_pg.clamp(min=1.0).sum(-1) + w_rep * kl_rep_pg.clamp(min=1.0).sum(-1)


class RSSM(nn.Module):
    def __init__(self, obs_dim: int = 256, road_dim: int = 256, h_dim: int = 512,
                 groups: int = STOCH_GROUPS, classes: int = STOCH_CLASSES, hidden: int = 256):
        super().__init__()
        self.h_dim, self.groups, self.classes = h_dim, groups, classes
        stoch_dim = groups * classes
        self.gru = nn.GRUCell(stoch_dim + road_dim, h_dim)
        self.road_mask_token = nn.Parameter(torch.zeros(road_dim))
        self.prior_net = nn.Sequential(nn.Linear(h_dim, hidden), nn.SiLU(),
                                        nn.Linear(hidden, stoch_dim))
        self.post_net = nn.Sequential(nn.Linear(h_dim + obs_dim, hidden), nn.SiLU(),
                                       nn.Linear(hidden, stoch_dim))

    def initial(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.h_dim, device=device)
        z = torch.zeros(batch, self.groups * self.classes, device=device)
        return h, z

    def prior(self, h: torch.Tensor) -> torch.Tensor:
        return self.prior_net(h).view(-1, self.groups, self.classes)

    def posterior(self, h: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.post_net(torch.cat([h, obs], dim=-1)).view(-1, self.groups, self.classes)

    def step(self, h: torch.Tensor, z_flat: torch.Tensor, road_embed: torch.Tensor) -> torch.Tensor:
        """h_{t+1} = GRU(h_t, [z_t, h_{a_t}])."""
        return self.gru(torch.cat([z_flat, road_embed], dim=-1), h)


class RoadScorer(nn.Module):
    """Permutation-equivariant candidate road scorer (stage2_correction Change
    7B). Replaces a fixed K-logit MLP over (h,z) -- unlearnable once candidate
    slots are permuted per-fix (Change 7A: slot position carries zero signal
    wrt (h,z)). Scores each candidate independently as a dot-product between a
    query built from (h,z) and a key built from the candidate's own identity/
    geometry, so permuting candidates just permutes the output logits -- there
    is nothing slot-position-dependent left to (mis)learn."""

    def __init__(self, h_dim: int = 512, stoch_dim: int = STOCH_DIM, road_dim: int = 256,
                 d: int = 128, n_highway: int = 64, hw_dim: int = 16):
        super().__init__()
        self.d, self.n_highway = d, n_highway
        self.q_proj = nn.Linear(h_dim + stoch_dim, d)
        self.highway_emb = nn.Embedding(n_highway, hw_dim)
        self.c_proj = nn.Linear(road_dim + hw_dim + 3, d)  # [stage0_embed, hw_emb, sinθ, cosθ, oneway]

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor, road_z: torch.Tensor,
                seg_id: torch.Tensor, heading_deg: torch.Tensor, oneway: torch.Tensor,
                highway_id: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        """h[B,h_dim] z_flat[B,stoch_dim] seg_id/heading_deg/oneway/highway_id/cand_mask[B,K]
        (mask: True = valid candidate) -> logits[B,K]."""
        q = self.q_proj(torch.cat([h, z_flat], dim=-1))                         # [B,d]
        h_e = road_z[seg_id.clamp(min=0)]                                       # [B,K,road_dim]
        hw = self.highway_emb(highway_id.clamp(min=0, max=self.n_highway - 1))  # [B,K,hw_dim]
        head = torch.deg2rad(heading_deg)
        feat = torch.cat([h_e, hw, torch.sin(head).unsqueeze(-1), torch.cos(head).unsqueeze(-1),
                           oneway.float().unsqueeze(-1)], dim=-1)
        e = self.c_proj(feat)                                                   # [B,K,d]
        logits = (q.unsqueeze(1) * e).sum(-1) / (self.d ** 0.5)                 # [B,K]
        return logits.masked_fill(~cand_mask, -1e4)                             # finite, not -inf (soft-target NaN guard)


class Heads(nn.Module):
    """Decoder heads reading (h_t, z_t) -> GPS recon, speed, road-membership
    (methodology 2.3), plus an MDN residual GPS head (stage2_correction Change 3:
    primary fix for the 0.31km coarse-only plateau -- Huber alone under-fits the
    fine-grained GPS distribution, a 5-component mixture over the *residual*
    (true - coarse) captures the multimodal fine structure the coarse head can't).
    `road` is a RoadScorer (Change 7B), a plain submodule (not folded into
    forward()) so both the posterior road CE and Change 4's prior-side
    auxiliary loss can call it directly, each with its own z sample."""

    def __init__(self, h_dim: int = 512, stoch_dim: int = STOCH_DIM, hidden: int = 256,
                 road_dim: int = 256, mdn_k: int = 5):
        super().__init__()
        in_dim = h_dim + stoch_dim
        self.mdn_k = mdn_k
        self.gps = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.speed = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # Change 7B: permutation-equivariant candidate scorer, replaces the old
        # fixed-slot MLP. Called explicitly (not from forward()) since it needs
        # per-timestep candidate context (segment_id/heading/oneway/highway_id/
        # mask) that forward()'s generic (h,z) signature doesn't carry -- same
        # reason it was already "intentionally a plain submodule" pre-7B.
        self.road = RoadScorer(h_dim=h_dim, stoch_dim=stoch_dim, road_dim=road_dim)
        self.mdn = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, mdn_k * 5))  # per component: pi,mu_x,mu_y,ls_x,ls_y
        # zero-init the last layer -> logit_pi starts uniform, mu starts at 0,
        # log_sigma starts at 0 (sigma~=1 in normalized units) -- exactly the spec's init.
        nn.init.zeros_(self.mdn[-1].weight)
        nn.init.zeros_(self.mdn[-1].bias)

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor):
        """GPS/speed/MDN only -- road score is NOT here, call `self.road(...)`
        directly with per-timestep candidate context (see RoadScorer)."""
        x = torch.cat([h, z_flat], dim=-1)
        mdn_out = self.mdn(x).view(-1, self.mdn_k, 5)
        pi_logits = mdn_out[..., 0]
        mu = mdn_out[..., 1:3]
        log_sigma = mdn_out[..., 3:5].clamp(-5.0, 2.0)
        return self.gps(x), self.speed(x).squeeze(-1), (pi_logits, mu, log_sigma)


def mdn_nll(pi_logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of `target` [B,2] under a K-component diagonal-Gaussian
    2D mixture. pi_logits [B,K], mu/log_sigma [B,K,2] -> NLL [B]."""
    log_pi = F.log_softmax(pi_logits, dim=-1)                            # [B,K]
    diff = (target.unsqueeze(-2) - mu) / log_sigma.exp()                 # [B,K,2]
    log_comp = -0.5 * diff.pow(2).sum(-1) - log_sigma.sum(-1) - math.log(2 * math.pi)  # [B,K]
    return -torch.logsumexp(log_pi + log_comp, dim=-1)                   # [B]


def mdn_mean(pi_logits: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Mixture point estimate (weighted mean), for eval/inference -- [B,K],[B,K,2] -> [B,2]."""
    w = F.softmax(pi_logits, dim=-1).unsqueeze(-1)                       # [B,K,1]
    return (w * mu).sum(-2)
