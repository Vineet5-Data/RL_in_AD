"""Label-free physics rewards (methodology 2.5) computed per candidate.

All six constraint terms are functions of raw GPS observables + candidate road
geometry + graph topology only -- no matched ground truth anywhere (P3).
Used two ways:
  stage2r  as regression TARGETS for HeadsLight.reward (the DreamerV3-style
           reward head), computed on real teacher-forced sequences;
  stage3   never computed directly -- imagination has no real GPS, rewards
           there come from the trained reward head.

lambda order [dist, head, oneway, speed, topo, smooth] and constants come from
dataset.config.RewardConfig (already encodes methodology 2.5 verbatim).
"""

from __future__ import annotations

import math

import torch


class SuccessorTable:
    """O(log E) membership test 'is v a topological successor of u' over the
    segment graph, from a PyG edge_index. Symmetrized: continuity in map
    matching means graph connectivity; direction legality is already penalized
    separately by r_oneway. ponytail: symmetrize is a no-op if edge_index
    already stores both directions (typical PyG undirected convention)."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device=None):
        u, v = edge_index[0].long(), edge_index[1].long()
        keys = torch.cat([u * num_nodes + v, v * num_nodes + u])
        self.keys = torch.unique(keys).to(device or edge_index.device)
        self.n = num_nodes

    def contains(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """u,v broadcastable int tensors -> bool tensor of broadcast shape."""
        q = (u.long() * self.n + v.long()).clamp(min=0)
        idx = torch.searchsorted(self.keys, q.reshape(-1)).clamp(max=self.keys.numel() - 1)
        hit = self.keys[idx] == q.reshape(-1)
        return hit.view(q.shape)


def _wrap_deg(d: torch.Tensor) -> torch.Tensor:
    """Wrap angle difference to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def candidate_rewards(cfg, succ: SuccessorTable,
                      d_perp_m: torch.Tensor, cand_heading_deg: torch.Tensor,
                      cand_speed_kph: torch.Tensor, cand_oneway: torch.Tensor,
                      cand_seg: torch.Tensor, cand_mask: torch.Tensor,
                      gps_bearing_deg: torch.Tensor, gps_speed_mps: torch.Tensor,
                      prev_seg: torch.Tensor | None = None,
                      prev_road_heading_deg: torch.Tensor | None = None,
                      prev_gps_bearing_deg: torch.Tensor | None = None) -> torch.Tensor:
    """Combined per-candidate reward r[B,K] for ONE timestep (methodology 2.5).

    Candidate tensors [B,K]; GPS tensors [B]. prev_* are the previous fix's
    resolved segment / its road heading / its GPS bearing ([B] each) -- None on
    the first fix, where r_topo := 1 (no constraint yet) and r_smooth := 0.
    Masked candidate slots return 0."""
    lam = cfg.weights
    sigma, M = cfg.sigma_gps_m, cfg.oneway_penalty

    r_dist = -d_perp_m / sigma

    dh = _wrap_deg(gps_bearing_deg.unsqueeze(-1) - cand_heading_deg)
    r_head = torch.cos(torch.deg2rad(dh))

    r_oneway = -M * ((cand_oneway > 0) & (dh.abs() > 90.0)).float()

    v_lim = cand_speed_kph / 3.6
    known = v_lim > 0.5  # kph<=0 means unknown speed limit -> no penalty
    over = (gps_speed_mps.unsqueeze(-1) - v_lim) / v_lim.clamp(min=1.0)
    r_speed = torch.where(known, -over.clamp(min=0.0) ** 2, torch.zeros_like(over))

    if prev_seg is None:
        r_topo = torch.ones_like(r_dist)
        r_smooth = torch.zeros_like(r_dist)
    else:
        same = cand_seg == prev_seg.unsqueeze(-1)
        r_topo = (same | succ.contains(prev_seg.unsqueeze(-1), cand_seg)).float()
        dth_gps = _wrap_deg(gps_bearing_deg - prev_gps_bearing_deg).unsqueeze(-1)
        dth_road = _wrap_deg(cand_heading_deg - prev_road_heading_deg.unsqueeze(-1))
        r_smooth = -torch.deg2rad(_wrap_deg(dth_gps - dth_road)).abs()

    r = (lam[0] * r_dist + lam[1] * r_head + lam[2] * r_oneway
         + lam[3] * r_speed + lam[4] * r_topo + lam[5] * r_smooth)
    return r.masked_fill(~cand_mask, 0.0)


def _selfcheck():
    """Smallest check that fails if the reward logic breaks."""
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        sigma_gps_m: float = 10.0
        oneway_penalty: float = 5.0
        weights: tuple = (1.0, 0.5, 1.0, 0.3, 2.0, 0.2)

    cfg = Cfg()
    # graph 0->1->2 (stored directed; table symmetrizes)
    succ = SuccessorTable(torch.tensor([[0, 1], [1, 2]]), num_nodes=3)
    assert bool(succ.contains(torch.tensor(0), torch.tensor(1)))
    assert bool(succ.contains(torch.tensor(1), torch.tensor(0)))  # symmetrized
    assert not bool(succ.contains(torch.tensor(0), torch.tensor(2)))

    # 1 sample, 2 candidates: cand0 = perfect (on-road, aligned, connected),
    # cand1 = bad (30m off, opposite one-way, speeding, disconnected).
    r = candidate_rewards(
        cfg, succ,
        d_perp_m=torch.tensor([[0.0, 30.0]]),
        cand_heading_deg=torch.tensor([[90.0, 270.0]]),
        cand_speed_kph=torch.tensor([[50.0, 30.0]]),
        cand_oneway=torch.tensor([[0, 1]]),
        cand_seg=torch.tensor([[1, 2]]),
        cand_mask=torch.tensor([[True, True]]),
        gps_bearing_deg=torch.tensor([90.0]),
        gps_speed_mps=torch.tensor([16.0]),
        prev_seg=torch.tensor([0]),
        prev_road_heading_deg=torch.tensor([90.0]),
        prev_gps_bearing_deg=torch.tensor([90.0]),
    )
    # cand0: dist 0, head +1, topo +2 -> 1.0*0 + 0.5*1 + 2.0*1 = 2.5, speed ok (16<50/3.6=13.9? no ->
    # slight overspeed (16-13.9)/13.9 ~ 0.023 -> -0.3*0.0005 ~ 0) => ~2.499
    # cand1: dist -3, head -1, oneway -5, topo 0, speed (16-8.33)/8.33=0.92 -> -0.253,
    #        smooth -pi -> -0.628 => ~ -3 -0.5 -5 -0.253*... clearly << cand0
    assert r[0, 0] > 2.0 and r[0, 1] < -5.0, r
    # masked slot must come back exactly 0 from candidate_rewards itself
    r_masked = candidate_rewards(
        cfg, succ,
        d_perp_m=torch.tensor([[0.0, 30.0]]),
        cand_heading_deg=torch.tensor([[90.0, 270.0]]),
        cand_speed_kph=torch.tensor([[50.0, 30.0]]),
        cand_oneway=torch.tensor([[0, 1]]),
        cand_seg=torch.tensor([[1, 2]]),
        cand_mask=torch.tensor([[True, False]]),
        gps_bearing_deg=torch.tensor([90.0]),
        gps_speed_mps=torch.tensor([16.0]),
    )
    assert float(r_masked[0, 1]) == 0.0 and float(r_masked[0, 0]) != 0.0
    print("[rewards selfcheck] PASS", r.tolist())


if __name__ == "__main__":
    _selfcheck()
