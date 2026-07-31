"""P1a GPS-Road cross-modal encoder (methodology 2.2 / Stage 1).

Per fix the encoder fuses two streams: a temporal self-attention over the GPS
sequence, and a cross-attention whose keys/values are that fix's <=K candidate
road tokens (frozen Stage-0 RoadGAT embedding h_e + a perpendicular-geometry
positional code). Output z_t in R^d is the GPS-aware road-aligned latent that
Stage 2/3 consume.

Label-free objective = three self-supervised heads (methodology Stage 1):
  MGM      masked-GPS-modeling: hide 15% of fixes, recover their coords (Huber)
  contrast z_t vs its candidate roads, soft target = softmax(-d_perp^2/2*sigma^2)
           (stage1_correction Change 9). Candidate d_perp is a legit deployment
           input (Change 8), zeroed on 30% of fixes (geo_mask) to force context-
           based inference there -- target-side d_perp always read from cache.
  next     autoregressive next-fix coordinate (MSE)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEG_M = 111_320.0  # metres per degree latitude (good enough for centroid-local coords)


def _sinusoid(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding for integer positions [L] -> [L, dim]."""
    device = positions.device
    div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    ang = positions.unsqueeze(-1).float() * div
    pe = torch.zeros(positions.size(0), dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class CrossModalLayer(nn.Module):
    """Pre-norm: temporal self-attn over the sequence, then per-fix cross-attn to roads."""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, cand, seq_pad, cand_pad):
        # x[B,L,d] cand[B,L,K,d] seq_pad[B,L] cand_pad[B,L,K] (True = ignore)
        b, l, d = x.shape
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, key_padding_mask=seq_pad, need_weights=False)[0]
        # per-fix cross-attention: flatten (B,L) into the batch dim, K road tokens as memory
        q = self.n2(x).reshape(b * l, 1, d)
        kv = cand.reshape(b * l, cand.size(2), d)
        cpad = cand_pad.reshape(b * l, cand.size(2))
        c = self.cross_attn(q, kv, kv, key_padding_mask=cpad, need_weights=False)[0]
        c = torch.nan_to_num(c).reshape(b, l, d)  # fixes with 0 valid candidates -> 0 (NaN guard)
        x = x + c
        return x + self.ff(self.n3(x))


class Stage1Encoder(nn.Module):
    def __init__(self, d: int = 256, heads: int = 8, layers: int = 4, ff: int = 1024,
                 road_dim: int = 256, n_hour: int = 24, n_dow: int = 7,
                 mask_ratio: float = 0.15, temp: float = 0.07):
        super().__init__()
        self.d, self.mask_ratio, self.temp = d, mask_ratio, temp
        self.gps_in = nn.Linear(6, d)            # [nlat,nlon,sinθ,cosθ,log1p(v),log1p(dt)]
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.hour_emb = nn.Embedding(n_hour, d)
        self.dow_emb = nn.Embedding(n_dow, d)
        self.road_proj = nn.Linear(road_dim, d)  # adapt frozen Stage-0 embedding
        self.cand_pos = nn.Linear(6, d)          # [d_perp/50 (0 if masked), geo_mask_flag,
                                                  #  sinθ, cosθ, oneway, log1p(speed)]
                                                  # (stage1_correction Change 8: d_perp restored as
                                                  # INPUT -- it's a legit deployment-time observable,
                                                  # not the leak. The leak was target==argmin(visible
                                                  # input) scored on that same input. Circularity is
                                                  # killed by geo_mask: 30% of fixes have d_perp zeroed
                                                  # on the input side while the target still reads the
                                                  # real d_perp from cache, forcing context-based
                                                  # inference on those fixes.)
        self.layers = nn.ModuleList(CrossModalLayer(d, heads, ff) for _ in range(layers))
        self.norm_f = nn.LayerNorm(d)
        self.coord_head = nn.Linear(d, 2)        # MGM
        self.next_head = nn.Linear(d, 2)         # next-point
        self.z_proj = nn.Linear(d, 128)          # contrastive
        self.c_proj = nn.Linear(d, 128)

    def _norm_coords(self, batch):
        """Centroid-local metric coords in km: [B,L,2], masked-mean centroid per trajectory."""
        lat, lon, m = batch["lat"], batch["lon"], batch["seq_mask"]
        w = m / m.sum(1, keepdim=True).clamp(min=1.0)
        lat0 = (lat * w).sum(1, keepdim=True)
        lon0 = (lon * w).sum(1, keepdim=True)
        nlat = (lat - lat0) * DEG_M / 1000.0
        nlon = (lon - lon0) * DEG_M * torch.cos(torch.deg2rad(lat0)) / 1000.0
        return torch.stack([nlat, nlon], -1)

    def _gps_feats(self, batch, coords):
        brg = torch.deg2rad(batch["bearing_deg"])
        return torch.stack([coords[..., 0], coords[..., 1],
                            torch.sin(brg), torch.cos(brg),
                            torch.log1p(batch["speed_mps"].clamp(min=0)),
                            torch.log1p(batch["dt"].clamp(min=0))], -1)

    def _cand_tokens(self, batch, road_z, geo_mask=None):
        # geo_mask: (B,L) bool, True = zero d_perp on the INPUT side for this fix
        # (Change 8 feature-dropout; target-side d_perp untouched, read from cache
        # in compute_losses).
        seg = batch["cand_segment_id"]
        h_e = road_z[seg.clamp(min=0)]                       # [B,L,K,road_dim]
        head = torch.deg2rad(batch["cand_heading_deg"])
        d_perp = batch["cand_d_perp_m"] / 50.0                # [B,L,K] normalized
        if geo_mask is not None:
            gm = geo_mask.unsqueeze(-1).expand_as(d_perp)
            d_perp = torch.where(gm, torch.zeros_like(d_perp), d_perp)
            flag = gm.float()
        else:
            flag = torch.zeros_like(d_perp)
        geo = torch.stack([d_perp, flag, torch.sin(head), torch.cos(head),
                           batch["cand_oneway"].float(),
                           torch.log1p(batch["cand_speed_kph"].clamp(min=0))], -1)
        return self.road_proj(h_e) + self.cand_pos(geo)      # [B,L,K,d]

    def forward(self, batch, road_z, apply_mask: bool, geo_mask=None):
        coords = self._norm_coords(batch)
        g = self.gps_in(self._gps_feats(batch, coords))      # [B,L,d]
        mgm_mask = None
        if apply_mask:
            mgm_mask = (torch.rand_like(batch["seq_mask"]) < self.mask_ratio) * (batch["seq_mask"] > 0)
            g = torch.where(mgm_mask.unsqueeze(-1), self.mask_token, g)

        b, l, _ = g.shape
        x = g + _sinusoid(torch.arange(l, device=g.device), self.d) \
              + self.hour_emb(batch["hour"]) + self.dow_emb(batch["dow"])
        cand = self._cand_tokens(batch, road_z, geo_mask)
        seq_pad = batch["seq_mask"] < 0.5
        cand_pad = (batch["cand_mask"] < 0.5) | (batch["cand_segment_id"] < 0)
        for layer in self.layers:
            x = layer(x, cand, seq_pad, cand_pad)
        return self.norm_f(x), cand, coords, mgm_mask, cand_pad

    def compute_losses(self, batch, road_z, geo_mask=None):
        z, cand, coords, mgm_mask, cand_pad = self.forward(batch, road_z, apply_mask=True, geo_mask=geo_mask)
        valid = batch["seq_mask"] > 0.5

        # MGM: recover coords of hidden fixes
        if mgm_mask is not None and mgm_mask.any():
            mgm = F.huber_loss(self.coord_head(z)[mgm_mask], coords[mgm_mask])
        else:
            mgm = z.sum() * 0.0

        # next-point: predict t+1 coord from z_t
        npair = valid[:, :-1] & valid[:, 1:]
        if npair.any():
            nxt = F.mse_loss(self.next_head(z[:, :-1])[npair], coords[:, 1:][npair])
        else:
            nxt = z.sum() * 0.0

        # contrastive: z_t vs its K candidate roads, soft emission target (Change 9) --
        # softmax(-d_perp^2 / (2*sigma^2)) instead of hard argmin, sigma=10m (GPS noise
        # scale). Handles bidirectional-segment ties and sub-noise margins honestly.
        # Loss computed on ALL valid fixes; split into visible/masked (Change 8) for
        # logging only -- `con` (both combined) is what backprops.
        has_cand = (~cand_pad).any(-1) & valid
        if geo_mask is None:
            geo_mask = torch.zeros_like(valid)
        if has_cand.any():
            zp = F.normalize(self.z_proj(z), dim=-1)             # [B,L,128]
            cp = F.normalize(self.c_proj(cand), dim=-1)          # [B,L,K,128]
            logits = (zp.unsqueeze(2) * cp).sum(-1) / self.temp  # [B,L,K]
            # NOTE: masked_fill with a large finite negative, not -inf. soft_tgt is
            # exactly 0.0 on masked slots and logp is ~-inf there; 0 * -inf = NaN,
            # 0 * finite = 0. Same trick applied to the target-side softmax input.
            NEG = -1e4
            logits = logits.masked_fill(cand_pad, NEG)
            d_perp = batch["cand_d_perp_m"].masked_fill(cand_pad, 1e4)
            sigma = 10.0
            soft_tgt = F.softmax(-(d_perp ** 2) / (2 * sigma ** 2), dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            ce = -(soft_tgt * logp).sum(-1)                      # [B,L]

            vis = has_cand & ~geo_mask
            msk = has_cand & geo_mask
            con = ce[has_cand].mean()
            con_vis = ce[vis].mean() if vis.any() else z.sum() * 0.0
            con_msk = ce[msk].mean() if msk.any() else z.sum() * 0.0
        else:
            con = con_vis = con_msk = z.sum() * 0.0

        total = mgm + con + nxt
        return {"total": total, "mgm": mgm.detach(), "contrast": con.detach(), "next": nxt.detach(),
                "contrast_visible": con_vis.detach(), "contrast_masked": con_msk.detach()}
