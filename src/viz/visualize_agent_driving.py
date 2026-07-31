"""Animate the research2 decoder-light world model (P2) "driving" a held-out
trajectory road-by-road, vs the Track B HMM baseline's Viterbi path and the raw
GPS ground truth.

research2 adaptation of the Run-3 script: loads RSSM2 + HeadsLight (stage2r
checkpoints) and can decode either with the WM road head (No-RL ablation) or,
via --actor-ckpt, with the Stage-3 actor -- the methodology 2.6 unification
("same policy network" for matching and prediction) made visible.

Warm-up (t < --t-warm): posterior on real GPS (Change 5 harness convention) --
dashed, reconstruction not prediction.
Imagination (t >= --t-warm): prior-only autoregressive rollout, self-driven by
the decoder's top-1 over the REAL candidate set at each timestep -- solid, the
actual "agent drives through the latent world model" behavior P2 claims.

Default checkpoint (ckpt/stage2r_porto.pt, road head, no --actor-ckpt) is the
best/optimal weights per methodology-comparison.md §7: "stage2r final + road
head (No-RL)" -- best match@1 (0.695) and hit@15 (0.579) with kl collapse=NO.
The Stage-3 actor underperforms it on matching (finding 3), so it is opt-in
only via --actor-ckpt, never the default.

Default mode is a LIVE interactive simulation window (plt.show()), paced by
each trajectory's real GPS sampling interval (median dt) x --speed, not a
fixed-fps slideshow. Pass --save to render the old headless GIF instead.

Usage:
  cd ~/Desktop/AlphaEvolve_research/.worktrees/research2
  python viz/visualize_agent_driving.py --traj-idx 0,1,2               # live sim, best weights
  python viz/visualize_agent_driving.py --traj-idx 0,1,2 --speed 20    # faster playback
  python viz/visualize_agent_driving.py --traj-idx 0,1,2 --save        # old GIF export
  python viz/visualize_agent_driving.py --actor-ckpt ckpt/stage3_porto.pt --traj-idx 0,1,2
"""
import os, sys, argparse
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/AlphaEvolve_research"))
DP   = BASE / ".worktrees" / "data-preprocess"
HMM  = BASE / ".worktrees" / "HMM_baseline" / "hmm_baseline"
HERE = Path(__file__).resolve().parent.parent  # Stage2_kaggle
sys.path = [str(HERE), str(DP), str(HMM), *[p for p in sys.path
            if p not in {str(HERE), str(DP), str(HMM)}]]

import pyarrow.parquet  # noqa: must precede torch
import numpy as np
import pandas as pd
import torch
import matplotlib
# ponytail: backend must be picked before pyplot import; --save runs headless
# (Agg), default live sim needs a real GUI backend (Tk, already on PATH).
matplotlib.use("Agg" if "--save" in sys.argv else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shapely.wkb
import contextily as cx

# Import research2 models BEFORE baseline_hmm: baseline_hmm.py's module-level
# sys.path.insert(0, kaggle-worktree) would otherwise shadow this worktree's own
# models/ package.
from models.world_model2 import straight_through_sample, RoadScorer
from training.stage3 import load_world_model
import baseline_hmm as hmm  # HMM_baseline/hmm_baseline/baseline_hmm.py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OSM_ROOT = BASE / "data" / "osm"


def load_actor(ckpt_path, road_dim):
    sd = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    actor = RoadScorer(h_dim=512, road_dim=road_dim).to(DEVICE)
    actor.load_state_dict(sd["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def rollout_agent(rssm, decoder, stage1, road_z, batch, t_warm):
    """Per-timestep chosen segment id for the WHOLE trajectory (not eval_stage2's
    fixed horizons). Returns (chosen_seg[L], phase[L] in {"warmup","imagine"},
    coords[L,2] normalized xy, gt_nearest_seg[L], valid[L])."""
    b = {k: v.to(DEVICE) for k, v in batch.items()}
    valid = (b["seq_mask"][0] > 0.5).cpu().numpy()
    L = valid.shape[0]

    z1, _, coords, _, _ = stage1.forward(b, road_z, apply_mask=False)

    cpad_all = (b["cand_mask"] < 0.5) | (b["cand_segment_id"] < 0)
    dp_all = b["cand_d_perp_m"].masked_fill(cpad_all, float("inf"))
    pos_idx_all = dp_all.argmin(-1)
    gt_seg = b["cand_segment_id"].gather(-1, pos_idx_all.unsqueeze(-1)).squeeze(-1)

    chosen_seg = np.full(L, -1, dtype=np.int64)
    phase = np.array(["pad"] * L, dtype=object)

    def pick(h, z, t):
        cmask = ~cpad_all[:, t]
        logits = decoder(h, z, road_z, b["cand_segment_id"][:, t],
                          b["cand_heading_deg"][:, t], b["cand_oneway"][:, t],
                          b["cand_highway_id"][:, t], cmask)
        slot = logits.argmax(-1)
        return int(b["cand_segment_id"][0, t, slot[0]].item())

    h, _ = rssm.initial(1, DEVICE)
    tw = min(t_warm, L)
    for t in range(tw):
        post = rssm.posterior(h, z1[:, t])
        z_q = straight_through_sample(post).view(1, -1)
        chosen_seg[t] = pick(h, z_q, t)
        phase[t] = "warmup"
        h = rssm.step(h, z_q, road_z[gt_seg[:, t].clamp(min=0)])

    for t in range(tw, L):
        prior = rssm.prior(h)
        z_p = straight_through_sample(prior).view(1, -1)
        seg_id = pick(h, z_p, t)
        chosen_seg[t] = seg_id
        phase[t] = "imagine"
        h = rssm.step(h, z_p, road_z[torch.tensor([max(seg_id, 0)], device=DEVICE)])

    return chosen_seg, phase, coords[0].cpu().numpy(), gt_seg[0].cpu().numpy(), valid


def rollout_hmm(batch, beta=20.0, rd=None):
    """nk-emission Viterbi path (Track B's best-performing variant per
    hmm_results.md), mapped back to global timestep -> segment id."""
    seg = batch["cand_segment_id"][0].numpy()
    dperp = batch["cand_d_perp_m"][0].numpy()
    cm = batch["cand_mask"][0].numpy()
    valid = (cm > 0.5) & (seg >= 0)
    dt = batch["dt"][0].numpy()
    lat = batch["lat"][0].numpy(); lon = batch["lon"][0].numpy()

    log_emit = hmm.compute_log_emit("nk", batch, None, None)
    log_emit[~valid] = -np.inf
    valid_fix = valid.any(1)

    L = seg.shape[0]
    chosen_seg = np.full(L, -1, dtype=np.int64)
    for piece in hmm.split_pieces(valid_fix, dt):
        idx = np.asarray(piece)
        coords = np.stack([lat[idx], lon[idx]], axis=1)
        path = hmm.viterbi(log_emit[idx], seg[idx], valid[idx], coords, rd, beta)
        for pos, t in enumerate(idx):
            chosen_seg[t] = seg[t, path[pos]]
    return chosen_seg


def segment_geoms(seg_ids, city="porto"):
    """WKB LineString geometry for a set of segment ids -> {seg_id: [(lon,lat),...]}."""
    seg_ids = set(int(s) for s in seg_ids if s >= 0)
    tbl = pyarrow.parquet.read_table(str(OSM_ROOT / city / "segments.parquet"),
                                      columns=["segment_id", "geometry"])
    df = tbl.to_pandas()
    df = df[df["segment_id"].isin(seg_ids)]
    out = {}
    for sid, wkb in zip(df["segment_id"], df["geometry"]):
        line = shapely.wkb.loads(bytes(wkb))
        out[int(sid)] = list(line.coords)  # [(lon, lat), ...]
    return out


def local_background_segments(lat, lon, city="porto", margin_deg=0.01):
    lat0, lat1 = lat.min() - margin_deg, lat.max() + margin_deg
    lon0, lon1 = lon.min() - margin_deg, lon.max() + margin_deg
    tbl = pyarrow.parquet.read_table(str(OSM_ROOT / city / "segments.parquet"),
                                      columns=["segment_id", "geometry"])
    df = tbl.to_pandas()
    lines = []
    for wkb in df["geometry"]:
        line = shapely.wkb.loads(bytes(wkb))
        xs, ys = line.xy
        if (lon0 <= max(xs) and lon1 >= min(xs) and lat0 <= max(ys) and lat1 >= min(ys)):
            lines.append(list(line.coords))
    return lines


def animate_trajectory(traj_idx, batch, rssm_seg, rssm_phase, hmm_seg, gt_seg, valid,
                        lat, lon, out_path, city="porto", fps=4, interval_ms=250, live=True,
                        substeps=8):
    bg = local_background_segments(lat[valid], lon[valid], city)
    all_seg_ids = set(rssm_seg[valid].tolist()) | set(hmm_seg[valid].tolist()) | set(gt_seg[valid].tolist())
    geoms = segment_geoms(all_seg_ids, city)

    margin = 0.01
    lat_v, lon_v = lat[valid], lon[valid]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(lon_v.min() - margin, lon_v.max() + margin)
    ax.set_ylim(lat_v.min() - margin, lat_v.max() + margin)
    ax.set_autoscale_on(False)
    try:
        # real OSM raster basemap -- ponytail: reuses contextily's own tile
        # cache, no manual tile-fetch/stitch code needed
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik,
                        attribution=False, zorder=-1)
    except Exception as e:
        print(f"[warn] basemap tile fetch failed ({e}); schematic background only")

    for line in bg:
        xs, ys = zip(*line)
        ax.plot(xs, ys, color="0.85", linewidth=0.6, zorder=0)

    gps_line, = ax.plot([], [], "o-", color="black", markersize=6, linewidth=1.0, alpha=0.6,
                         zorder=1, label="raw GPS (ground truth fixes)")
    gt_line, = ax.plot([], [], color="#2ca02c", linewidth=2.0, alpha=0.7, zorder=2,
                        label="nearest-road ground truth")
    hmm_line, = ax.plot([], [], color="#1f77b4", linewidth=2.5, alpha=0.8, zorder=3, label="HMM (nk, β=20)")
    rssm_warmup_line, = ax.plot([], [], color="#d62728", linewidth=2.5, linestyle="--", alpha=0.9, zorder=4,
                                 label="RSSM warm-up (observed)")
    rssm_imagine_line, = ax.plot([], [], color="#d62728", linewidth=3.0, alpha=0.95, zorder=5,
                                  label="RSSM imagination (self-driven)")
    cur_pt, = ax.plot([], [], "*", color="#ffcc00", markeredgecolor="black", markersize=18,
                       zorder=6, label="current fix")

    valid_idx = np.where(valid)[0]
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"traj {traj_idx} -- step 0")

    def frame_segs(seg_arr, up_to):
        pts = []
        for t in valid_idx[:up_to + 1]:
            sid = int(seg_arr[t])
            if sid in geoms:
                pts.extend(geoms[sid])
                pts.append((np.nan, np.nan))
        return pts

    n_fix = len(valid_idx)

    def update(i):
        # i is a sub-frame; t_index is the GPS-fix step, frac in [0,1] is
        # progress toward the next fix -- this is what makes the car glide
        # continuously instead of teleporting once per real GPS sample.
        t_index = min(i // substeps, max(n_fix - 2, 0))
        frac = 0.0 if n_fix <= 1 else min(1.0, (i - t_index * substeps) / substeps)
        t0 = valid_idx[t_index]
        t1 = valid_idx[min(t_index + 1, n_fix - 1)]

        gps_line.set_data(lon[valid_idx[:t_index + 1]], lat[valid_idx[:t_index + 1]])

        if i % substeps == 0:  # road-path redraw only on real fix boundaries
            gp = frame_segs(gt_seg, t_index)
            if gp:
                gx, gy = zip(*gp)
                gt_line.set_data(gx, gy)

            hp = frame_segs(hmm_seg, t_index)
            if hp:
                hx, hy = zip(*hp)
                hmm_line.set_data(hx, hy)

            wm_idx = [j for j in range(t_index + 1) if rssm_phase[valid_idx[j]] == "warmup"]
            im_idx = [j for j in range(t_index + 1) if rssm_phase[valid_idx[j]] == "imagine"]
            if wm_idx:
                wp = frame_segs(rssm_seg, max(wm_idx))
                if wp:
                    wx, wy = zip(*wp)
                    rssm_warmup_line.set_data(wx, wy)
            if im_idx:
                pts = []
                for j in im_idx:
                    sid = int(rssm_seg[valid_idx[j]])
                    if sid in geoms:
                        pts.extend(geoms[sid]); pts.append((np.nan, np.nan))
                if pts:
                    ix, iy = zip(*pts)
                    rssm_imagine_line.set_data(ix, iy)

        # smooth glide of the "car" marker between fix t0 -> t1
        cx_ = lon[t0] + (lon[t1] - lon[t0]) * frac
        cy_ = lat[t0] + (lat[t1] - lat[t0]) * frac
        cur_pt.set_data([cx_], [cy_])

        phase_str = rssm_phase[t0]
        ax.set_title(f"traj {traj_idx} -- step {t_index}/{n_fix-1}  (RSSM: {phase_str})")
        return gps_line, gt_line, hmm_line, rssm_warmup_line, rssm_imagine_line, cur_pt

    total_frames = max(1, (n_fix - 1) * substeps + 1)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False,
                                   repeat=True, interval=interval_ms)
    if live:
        plt.show()
    else:
        ani.save(str(out_path), writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/stage2r_porto.pt",
                    help="stage2r checkpoint; default is the best/optimal weights "
                         "(final ckpt, road head) per methodology-comparison.md §7")
    ap.add_argument("--actor-ckpt", default=None,
                    help="stage3 checkpoint; if set, decode with the actor instead of the WM road head "
                         "(underperforms the road head on matching -- opt-in only, not the default)")
    ap.add_argument("--city", default="porto", choices=["porto", "tdrive"])
    ap.add_argument("--split", default="holdout", choices=["holdout", "dev", "holdout30k"])
    ap.add_argument("--traj-idx", default="0", help="comma-separated indices into the held-out set")
    ap.add_argument("--t-warm", type=int, default=8)
    ap.add_argument("--beta", type=float, default=20.0, help="HMM transition scale (best from hmm_results.md sweep)")
    ap.add_argument("--out-dir", default="viz/out")
    ap.add_argument("--fps", type=int, default=4, help="GIF frame rate, only used with --save")
    ap.add_argument("--save", action="store_true",
                    help="render a headless GIF (old behavior) instead of a live interactive sim")
    ap.add_argument("--speed", type=float, default=8.0,
                    help="live playback speed multiplier vs real GPS sampling interval (dt)")
    ap.add_argument("--substeps", type=int, default=8,
                    help="interpolation sub-frames per real GPS fix (smooth car motion); "
                         "raise for smoother glide, lower for faster GIF export")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds, collate_fn = hmm.load_dataset(args.city, args.split, None)
    road_z = hmm.load_road_z(args.city).to(DEVICE)
    stage1 = hmm.load_stage1(road_z.size(1)).to(DEVICE)
    rssm, heads = load_world_model(args.ckpt, road_z.size(1), DEVICE)
    decoder = load_actor(args.actor_ckpt, road_z.size(1)) if args.actor_ckpt else heads.road
    mode = "actor" if args.actor_ckpt else "roadhead"
    rd = hmm.RouteDist(hmm.build_road_digraph(args.city))

    for idx in [int(x) for x in args.traj_idx.split(",")]:
        batch = collate_fn([ds[idx]])
        rssm_seg, phase, coords, gt_seg, valid = rollout_agent(rssm, decoder, stage1, road_z, batch, args.t_warm)
        hmm_seg = rollout_hmm(batch, beta=args.beta, rd=rd)
        lat = batch["lat"][0].numpy(); lon = batch["lon"][0].numpy()
        out_path = out_dir / f"agent_driving_traj{idx}_{mode}.gif"

        dt_valid = batch["dt"][0].numpy()[valid]
        dt_valid = dt_valid[dt_valid > 0]
        median_dt = float(np.median(dt_valid)) if len(dt_valid) else 1.0
        interval_ms = float(np.clip(median_dt * 1000.0 / (args.speed * args.substeps), 15.0, 500.0))

        animate_trajectory(idx, batch, rssm_seg, phase, hmm_seg, gt_seg, valid,
                            lat, lon, out_path, city=args.city, fps=args.fps,
                            interval_ms=interval_ms, live=not args.save, substeps=args.substeps)


if __name__ == "__main__":
    main()
