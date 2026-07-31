"""research2: decoder-light latent world model + Dreamer-style actor-critic.

Implements methodology-comparison.md Rank-1 recommendation (TD-MPC2/MuDreamer
lineage): the reconstruction-heavy DreamerV3-style RSSM of Stage-2 Runs 1-3 is
replaced by a decoder-LIGHT world model --

  REMOVED   MDN residual GPS head (the dominant-gradient head in the Change-14C
            MTL competition finding AND the decoder `h` freeloads on -- one
            change addresses both empirically diagnosed failure modes).
  REMOVED   (h,z) coarse GPS head. GPS decode is now permanently z-ONLY
            (`methodology-comparison.md` Sec.5 Rank 1: "remove or permanently
            z-only-gate the MDN GPS reconstruction head"): the head structurally
            cannot see h, so the deterministic path cannot satisfy GPS recon and
            posterior collapse via h-freeloading is removed by construction, while
            z still receives a reconstruction gradient that fights collapse.
  KEPT      RoadScorer (permutation-equivariant, Change 7B) -- road CE is now the
            PRIMARY world-model signal (MuDreamer: value/action prediction
            replaces observation reconstruction).
  ADDED     RewardScorer -- per-candidate physics-reward predictor (DreamerV3
            reward head, needed to compute rewards inside imagination where no
            real GPS exists). Trained on real sequences against the label-free
            Sec.2.5 physics rewards (training/rewards.py).
  ADDED     Actor (a RoadScorer instance, warm-started from the road head per
            methodology Stage 3) + Critic + lambda_return -- the DreamerV3/
            Think2Drive-style discrete categorical actor-critic over the masked
            candidate action space, retained UNCHANGED per methodology-comparison
            Sec.4 ("keep this RL formulation... only redesign the decoder side").
  ADDED     RSSM2: RSSM with a normalization layer inside prior/post MLPs.
            MuDreamer's own caveat: removing reconstruction opens a *different*
            (constant-code) collapse mode that BatchNorm prevents; untested on
            GRU+categorical latents (methodology-comparison Gap 5), so the norm
            is a flag (batch|layer|none, default batch) rather than hardcoded.

Frozen inputs only: nothing here trains Stage 0/1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.world_model import (RSSM, RoadScorer, STOCH_DIM,
                                     straight_through_sample, kl_categorical_pergroup,
                                     kl_categorical, _mixed_probs)
except ImportError:  # flat layout (Kaggle CODE_ROOT on sys.path)
    from world_model import (RSSM, RoadScorer, STOCH_DIM,  # type: ignore # noqa: F401
                              straight_through_sample, kl_categorical_pergroup,
                              kl_categorical, _mixed_probs)

__all__ = ["RSSM2", "HeadsLight", "RewardScorer", "Critic", "lambda_return",
           "RoadScorer", "straight_through_sample", "kl_categorical_pergroup",
           "kl_categorical", "_mixed_probs", "STOCH_DIM", "symlog", "symexp"]


def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 reward/value squashing: sign(x)*log(1+|x|). The physics reward
    spans ~[-11, +3.5] (oneway -5, dist to -5, topo +2); raw-MSE on that scale
    made the reward head the dominant-gradient head on the shared trunk in the
    real-data sanity run -- the exact failure mode (dominant reconstruction-style
    gradient) this redesign removes. Symlog targets are the lineage-faithful fix."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog -- decode predicted rewards back to true scale for
    lambda-returns in imagination."""
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def _norm_layer(kind: str, dim: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind {kind!r} (batch|layer|none)")


class RSSM2(RSSM):
    """RSSM with normalization inside the prior/posterior MLPs (MuDreamer's
    anti-constant-code measure once reconstruction is removed). Everything else
    (GRU transition, road_mask_token, initial/prior/posterior/step) inherited."""

    def __init__(self, obs_dim: int = 256, road_dim: int = 256, h_dim: int = 512,
                 groups: int = 16, classes: int = 16, hidden: int = 256,
                 norm: str = "batch"):
        super().__init__(obs_dim=obs_dim, road_dim=road_dim, h_dim=h_dim,
                         groups=groups, classes=classes, hidden=hidden)
        stoch_dim = groups * classes
        self.prior_net = nn.Sequential(nn.Linear(h_dim, hidden), _norm_layer(norm, hidden),
                                        nn.SiLU(), nn.Linear(hidden, stoch_dim))
        self.post_net = nn.Sequential(nn.Linear(h_dim + obs_dim, hidden), _norm_layer(norm, hidden),
                                       nn.SiLU(), nn.Linear(hidden, stoch_dim))


class RewardScorer(nn.Module):
    """Per-candidate predicted SYMLOG physics reward r_hat[B,K] from state (h,z)
    and each candidate's own identity/geometry -- same permutation-equivariant key
    construction as RoadScorer, but a small MLP over [query, key] instead of a
    dot product (reward regression needs bias/offset capacity a pure dot lacks).
    Trained against symlog(reward); consumers apply symexp() to recover the true
    scale (DreamerV3 convention -- see symlog() docstring for why)."""

    def __init__(self, h_dim: int = 512, stoch_dim: int = STOCH_DIM, road_dim: int = 256,
                 d: int = 128, n_highway: int = 64, hw_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.n_highway = n_highway
        self.q_proj = nn.Linear(h_dim + stoch_dim, d)
        self.highway_emb = nn.Embedding(n_highway, hw_dim)
        self.c_proj = nn.Linear(road_dim + hw_dim + 3, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor, road_z: torch.Tensor,
                seg_id: torch.Tensor, heading_deg: torch.Tensor, oneway: torch.Tensor,
                highway_id: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        """-> predicted reward [B,K] (masked slots forced to 0 -- they are never
        selected by the masked actor, 0 keeps them out of MSE without NaN risk)."""
        q = self.q_proj(torch.cat([h, z_flat], dim=-1))                          # [B,d]
        h_e = road_z[seg_id.clamp(min=0)]                                        # [B,K,road_dim]
        hw = self.highway_emb(highway_id.clamp(min=0, max=self.n_highway - 1))
        head = torch.deg2rad(heading_deg)
        feat = torch.cat([h_e, hw, torch.sin(head).unsqueeze(-1), torch.cos(head).unsqueeze(-1),
                           oneway.float().unsqueeze(-1)], dim=-1)
        e = self.c_proj(feat)                                                    # [B,K,d]
        x = torch.cat([q.unsqueeze(1).expand_as(e), e], dim=-1)                  # [B,K,2d]
        r = self.mlp(x).squeeze(-1)                                              # [B,K]
        return r.masked_fill(~cand_mask, 0.0)


class HeadsLight(nn.Module):
    """Decoder-light heads (Rank-1 redesign):
      road   RoadScorer over (h,z) + candidate context -- PRIMARY signal.
      reward RewardScorer -- physics-reward predictor for imagination.
      gps    z-ONLY coarse decode (structurally blind to h -- the anti-freeload
             gate made permanent; also the redefined cl_gps interpretability
             probe, see methodology-comparison Gap re: cl_gps redefinition).
      speed  kept on (h,z) unchanged (never a diagnosed failure mode; scalar
             target, negligible gradient mass).
    No MDN anywhere."""

    def __init__(self, h_dim: int = 512, stoch_dim: int = STOCH_DIM, hidden: int = 256,
                 road_dim: int = 256):
        super().__init__()
        self.gps = nn.Sequential(nn.Linear(stoch_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.speed = nn.Sequential(nn.Linear(h_dim + stoch_dim, hidden), nn.SiLU(),
                                    nn.Linear(hidden, 1))
        self.road = RoadScorer(h_dim=h_dim, stoch_dim=stoch_dim, road_dim=road_dim)
        self.reward = RewardScorer(h_dim=h_dim, stoch_dim=stoch_dim, road_dim=road_dim)

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor):
        """-> (gps_hat[B,2] from z only, speed_hat[B]). road/reward need
        per-timestep candidate context -- call .road(...)/.reward(...) directly."""
        return self.gps(z_flat), self.speed(torch.cat([h, z_flat], dim=-1)).squeeze(-1)


class Critic(nn.Module):
    """V(h,z) -> scalar. Trained on imagined lambda-returns (Stage 3)."""

    def __init__(self, h_dim: int = 512, stoch_dim: int = STOCH_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h_dim + stoch_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))
        # zero-init last layer: V starts at 0, first lambda-returns are pure
        # predicted-reward sums -- avoids a random-critic bootstrap transient.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z_flat], dim=-1)).squeeze(-1)


def lambda_return(rewards: torch.Tensor, values: torch.Tensor, bootstrap: torch.Tensor,
                  gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """Dreamer lambda-returns (methodology 2.4):
        R_t = r_t + gamma * ((1-lam) * V_{t+1} + lam * R_{t+1}),  R_H = bootstrap
    rewards/values: [T, N] (values[t] = V(s_t), t aligned with rewards),
    bootstrap: [N] = V(s_H). -> returns [T, N]."""
    T = rewards.size(0)
    out = torch.empty_like(rewards)
    nxt = bootstrap
    for t in range(T - 1, -1, -1):
        v_next = values[t + 1] if t + 1 < T else bootstrap
        nxt = rewards[t] + gamma * ((1.0 - lam) * v_next + lam * nxt)
        out[t] = nxt
    return out
