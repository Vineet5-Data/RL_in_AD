"""Deployable map-matching + path-prediction API (product roadmap item 1).

Wraps the proven eval-script machinery behind three calls on raw fixes
(lat, lon, unix-seconds t) with the city mode switch the 2026-07-11 OOD
result forces:

  match_offline  batch Viterbi. Trained city -> hybrid WM+NK emission
                 (Porto 0.8655); unseen city -> pure NK (T-Drive 0.7955;
                 hybrid measured WORSE OOD, 0.7871).
  match_online   streaming, no future context. Trained city -> WM road-head
                 closed loop + NK emission/transition terms ("hyb+topo",
                 Porto 0.8122 with 5.9% disconnected transitions vs 0.7690 /
                 14.9% for the bare road head -- eval_online_topo.txt);
                 unseen city -> nearest candidate.
  predict        WM prior rollout over road-graph successors (works zero-shot
                 on unseen cities; degrades gracefully -- OOD hit@5 0.536).

Usage, from .worktrees/research2:
  from matcher import Matcher
  m = Matcher("porto")
  segs = m.match_offline(lat, lon, t)   # int64 per fix, -1 = no road in 50 m
  segs = m.match_online(lat, lon, t)
  future = m.predict(lat, lon, t, horizon=15)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
HERE = Path(__file__).resolve().parent
DP = BASE / ".worktrees" / "data-preprocess"
HMM = BASE / ".worktrees" / "HMM_baseline" / "hmm_baseline"
sys.path = [str(HERE), str(DP), str(HMM),
            *[p for p in sys.path if p not in {str(HERE), str(DP), str(HMM)}]]

import pyarrow.parquet  # noqa: F401,E402  must load before torch on Windows
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import baseline_hmm as hb  # noqa: E402
import eval_research2 as ev  # noqa: E402
from preprocessing.clean import _finalize_segment  # noqa: E402

BETA = 20.0
CHUNK = 128            # Stage-1 transformer cap (SequenceConfig.max_len)
TRAINED_CITIES = {"porto"}


def _road_z(city):
    """Stage-0 Porto road encoder applied to `city`'s graph (zero-shot off-Porto)."""
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT

    data = load_pyg(str(ev.OSM_ROOT), city).to(ev.DEVICE)
    enc = RoadGAT(num_cont=data.x.size(1),
                  num_highway=max(64, int(data.highway_id.max()) + 1)).to(ev.DEVICE)
    enc.load_state_dict(torch.load(str(ev.S0_CKPT), map_location=ev.DEVICE, weights_only=True))
    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.highway_id, data.edge_index)
    return z.detach()


class Matcher:
    def __init__(self, city: str = "porto", ckpt: str = "final"):
        from dataset.candidates import CandidateIndex, build_neighbor_map

        self.city = city
        self.trained = city in TRAINED_CITIES
        self.ci = CandidateIndex.from_city(str(ev.OSM_ROOT), city)
        self.rd = hb.RouteDist(hb.build_road_digraph(city))
        self.road_z = _road_z(city)
        self.stage1 = ev._stage1(self.road_z.size(1))
        ckpt_path = Path(ckpt) if str(ckpt).endswith(".pt") else ev._checkpoint_table()[ckpt]
        self.rssm, self.heads = ev._load_wm(ckpt_path, self.road_z.size(1))
        self.nbr = build_neighbor_map(str(ev.OSM_ROOT), city)
        # id-indexed attribute lookups for predict()'s successor candidate sets
        n = self.road_z.size(0)
        self._heading = np.zeros(n, np.float32)
        self._oneway = np.zeros(n, np.int64)
        self._highway = np.zeros(n, np.int64)
        sid = self.ci.segment_id
        self._heading[sid] = self.ci.heading
        self._oneway[sid] = self.ci.oneway
        self._highway[sid] = self.ci.highway_id

    # ---------- input plumbing ----------

    def _chunks(self, n):
        k = -(-n // CHUNK)  # ceil
        bounds = np.linspace(0, n, k + 1).astype(int)
        return list(zip(bounds[:-1], bounds[1:]))

    def _batch(self, lat, lon, t):
        """One <=CHUNK-fix trajectory -> collated B=1 tensor batch (CPU + device)."""
        from dataset.trajectories import TrajectoryGraphDataset, collate_fn
        from dataset.config import SequenceConfig, RetrievalConfig

        fields = _finalize_segment(np.asarray(lat, np.float64),
                                   np.asarray(lon, np.float64),
                                   np.asarray(t, np.int64))
        df = pd.DataFrame({**fields, "traj_id": 0, "source": self.city})
        ds = TrajectoryGraphDataset(df, {self.city: self.ci},
                                    SequenceConfig(min_len=2), RetrievalConfig(),
                                    cache_path=None, perm_seed=0)
        b_cpu = collate_fn([ds[0]])
        b = {k: v.to(ev.DEVICE) for k, v in b_cpu.items()}
        return b_cpu, b

    @torch.no_grad()
    def _wm_log_emit(self, b):
        """Closed-loop WM road-head log-emissions for a B=1 batch."""
        from models.world_model2 import straight_through_sample

        valid = b["seq_mask"] > 0.5
        batch, length = valid.shape
        z1, _, _, _, _ = self.stage1.forward(b, self.road_z, apply_mask=False)
        cpad = (b["cand_mask"] < 0.5) | (b["cand_segment_id"] < 0)

        out = np.full((length, b["cand_segment_id"].shape[-1]), -np.inf)
        h, _ = self.rssm.initial(batch, ev.DEVICE)
        prev_seg = None
        for t in range(length):
            if t > 0:
                h = self.rssm.step(h, z_q, self.road_z[prev_seg.clamp(min=0)])
            z_q = straight_through_sample(self.rssm.posterior(h, z1[:, t])).view(batch, -1)
            cmask = ~cpad[:, t]
            logits = self.heads.road(h, z_q, self.road_z, b["cand_segment_id"][:, t],
                                     b["cand_heading_deg"][:, t], b["cand_oneway"][:, t],
                                     b["cand_highway_id"][:, t], cmask)
            slot = logits.argmax(-1)
            prev_seg = b["cand_segment_id"][:, t].gather(-1, slot.unsqueeze(-1)).squeeze(-1)
            lp = torch.log_softmax(logits.masked_fill(~cmask, -1e4), dim=-1)
            out[t] = lp[0].double().cpu().numpy()
        return out

    @staticmethod
    def _nk_log_emit(dperp, valid):
        logit = np.where(valid, -(dperp.astype(np.float64) ** 2) / (2 * hb.SIGMA ** 2), -np.inf)
        return hb.log_softmax_rows(logit)

    # ---------- public API ----------

    def match_offline(self, lat, lon, t, mode: str | None = None):
        """Per-fix matched segment id (int64, -1 = unmatched). Batch Viterbi.

        mode: None -> hybrid on trained city, nk elsewhere; or force "hybrid"/"nk".
        """
        mode = mode or ("hybrid" if self.trained else "nk")
        n = len(lat)
        out = np.full(n, -1, np.int64)
        # ponytail: viterbi restarts at each 128-fix chunk boundary; stitch
        # transitions across chunks if long-trajectory accuracy ever matters.
        for s, e in self._chunks(n):
            b_cpu, b = self._batch(lat[s:e], lon[s:e], t[s:e])
            seg = b_cpu["cand_segment_id"][0].numpy()
            dperp = b_cpu["cand_d_perp_m"][0].numpy()
            valid = (b_cpu["cand_mask"][0].numpy() > 0.5) & (seg >= 0)
            dt = b_cpu["dt"][0].numpy()
            le = self._nk_log_emit(dperp, valid)
            if mode == "hybrid":
                le_wm = self._wm_log_emit(b)
                le_wm[~valid] = -np.inf
                le = le + le_wm
            coords_all = np.stack([np.asarray(lat[s:e]), np.asarray(lon[s:e])], axis=1)
            for piece in hb.split_pieces(valid.any(1), dt):
                idx = np.asarray(piece)
                slots = hb.viterbi(le[idx], seg[idx], valid[idx], coords_all[idx],
                                   self.rd, BETA)
                out[s + idx] = seg[idx, slots]
        return out

    @torch.no_grad()
    def match_online(self, lat, lon, t, mode: str | None = None):
        """Per-fix segment id using only past context (streaming protocol).

        mode: None -> "hyb+topo" on trained city (WM log-prob + NK emission +
        NK route-distance transition from own previous choice; topology terms
        reset after a GPS gap, dt > 3x median), "nk" elsewhere; or force
        "base" (bare road-head argmax) / "hyb+topo" / "nk".
        """
        from models.world_model2 import straight_through_sample

        mode = mode or ("hyb+topo" if self.trained else "nk")
        n = len(lat)
        out = np.full(n, -1, np.int64)
        for s, e in self._chunks(n):
            b_cpu, b = self._batch(lat[s:e], lon[s:e], t[s:e])
            seg = b_cpu["cand_segment_id"][0].numpy()
            dperp = b_cpu["cand_d_perp_m"][0].numpy()
            valid = (b_cpu["cand_mask"][0].numpy() > 0.5) & (seg >= 0)
            if mode == "nk":  # nearest candidate == NK online filter
                dp = np.where(valid, dperp, np.inf)
                hit = valid.any(1)
                out[s:e][hit] = seg[np.arange(len(seg)), dp.argmin(1)][hit]
                continue
            la, lo = np.asarray(lat[s:e]), np.asarray(lon[s:e])
            dt = b_cpu["dt"][0].numpy()
            pos = dt[dt > 0]
            med = float(np.median(pos)) if pos.size else 0.0
            le_nk = self._nk_log_emit(dperp, valid)
            cpad = (b["cand_mask"] < 0.5) | (b["cand_segment_id"] < 0)
            z1, _, _, _, _ = self.stage1.forward(b, self.road_z, apply_mask=False)
            h, _ = self.rssm.initial(1, ev.DEVICE)  # ponytail: h resets per chunk
            prev_drive = None   # drives the RSSM (always set)
            prev_topo = None    # anchors the transition term (None after gap/no-cand)
            for i in range(e - s):
                if i > 0:
                    h = self.rssm.step(h, z_q, self.road_z[prev_drive.clamp(min=0)])
                z_q = straight_through_sample(self.rssm.posterior(h, z1[:, i])).view(1, -1)
                cmask = ~cpad[:, i]
                logits = self.heads.road(h, z_q, self.road_z, b["cand_segment_id"][:, i],
                                         b["cand_heading_deg"][:, i], b["cand_oneway"][:, i],
                                         b["cand_highway_id"][:, i], cmask)
                drive_slot = int(logits.argmax(-1).item())
                if valid[i].any():
                    score = logits.masked_fill(~cmask, -1e4)
                    score = torch.log_softmax(score, dim=-1)[0].double().cpu().numpy().copy()
                    if mode == "hyb+topo":
                        score = score + le_nk[i]
                        gap = i == 0 or (med > 0 and dt[i] > 3 * med)
                        if prev_topo is not None and not gap:
                            d_gc = hb.haversine_m(la[i - 1], lo[i - 1], la[i], lo[i])
                            score = score + hb.log_transition(
                                np.array([prev_topo]), np.array([True]),
                                seg[i], valid[i], d_gc, self.rd, BETA)[0]
                    score[~valid[i]] = -np.inf
                    slot = int(np.argmax(score))
                    out[s + i] = prev_topo = int(seg[i, slot])
                    drive_slot = slot
                else:
                    prev_topo = None
                prev_drive = b["cand_segment_id"][:, i].gather(
                    -1, torch.tensor([[drive_slot]], device=ev.DEVICE)).squeeze(-1)
        return out

    @torch.no_grad()
    def predict(self, lat, lon, t, horizon: int = 15):
        """Roll the WM prior `horizon` steps past the last fix; segment ids.

        Candidates per step = graph successors of the previous choice (+ stay),
        so the returned path is connected by construction.
        """
        from models.world_model2 import straight_through_sample

        s = max(0, len(lat) - CHUNK)  # warm up on the trailing window
        b_cpu, b = self._batch(lat[s:], lon[s:], t[s:])
        seg = b_cpu["cand_segment_id"][0].numpy()
        valid = (b_cpu["cand_mask"][0].numpy() > 0.5) & (seg >= 0)
        cpad = (b["cand_mask"] < 0.5) | (b["cand_segment_id"] < 0)
        z1, _, _, _, _ = self.stage1.forward(b, self.road_z, apply_mask=False)

        h, _ = self.rssm.initial(1, ev.DEVICE)
        prev_seg = None
        for i in range(seg.shape[0]):
            if i > 0:
                h = self.rssm.step(h, z_q, self.road_z[prev_seg.clamp(min=0)])
            z_q = straight_through_sample(self.rssm.posterior(h, z1[:, i])).view(1, -1)
            cmask = ~cpad[:, i]
            logits = self.heads.road(h, z_q, self.road_z, b["cand_segment_id"][:, i],
                                     b["cand_heading_deg"][:, i], b["cand_oneway"][:, i],
                                     b["cand_highway_id"][:, i], cmask)
            slot = logits.argmax(-1)
            prev_seg = b["cand_segment_id"][:, i].gather(-1, slot.unsqueeze(-1)).squeeze(-1)

        cur = int(prev_seg.item())
        path = []
        for _ in range(horizon):
            h = self.rssm.step(h, z_q, self.road_z[[max(cur, 0)]])
            z_q = straight_through_sample(self.rssm.prior(h)).view(1, -1)
            cands = sorted(self.nbr.get(cur, set()) | {cur})
            cs = torch.tensor([cands], dtype=torch.long, device=ev.DEVICE)
            logits = self.heads.road(
                h, z_q, self.road_z, cs,
                torch.from_numpy(self._heading[cands][None]).to(ev.DEVICE),
                torch.from_numpy(self._oneway[cands][None]).to(ev.DEVICE),
                torch.from_numpy(self._highway[cands][None]).to(ev.DEVICE),
                torch.ones(1, len(cands), dtype=torch.bool, device=ev.DEVICE))
            cur = cands[int(logits.argmax(-1).item())]
            path.append(cur)
        return np.asarray(path, np.int64)
