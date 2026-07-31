"""Stage-1 (P1a) pretraining: GPS-Road cross-modal encoder, label-free (methodology Stage 1).

Frozen Stage-0 RoadGAT supplies road-segment embeddings z[N,256]; this encoder
learns to align raw GPS trajectories to candidate roads via 3 self-supervised
losses (masked-GPS-modeling + InfoNCE GPS<->road + next-point). No matched labels.

Local default = Porto only (Beijing road graph not built yet -> geolife/tdrive
have no candidate index). bf16 autocast, checkpoint every --ckpt-every steps.

Run:  python -m training.stage1 --processed-root data/processed --osm-root data/osm \
        --stage0-ckpt ckpt/stage0_porto.pt --out-dir ckpt --limit-trajs 4000
Smoke: python -m training.stage1 --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet  # noqa: F401  must precede torch (arrow/torch native-lib load order; segfault otherwise)
import torch

try:
    from models.gps_encoder import Stage1Encoder
except ImportError:
    from gps_encoder import Stage1Encoder  # type: ignore


def _road_embeddings(osm_root, city, stage0_ckpt, device):
    """Frozen Stage-0 road tokens z[N, road_dim]. Row index == cand_segment_id."""
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT

    data = load_pyg(osm_root, city).to(device)
    enc = RoadGAT(num_cont=data.x.size(1), num_highway=max(64, int(data.highway_id.max()) + 1)).to(device)
    enc.load_state_dict(torch.load(stage0_ckpt, map_location=device))
    enc.eval()
    with torch.no_grad():
        z = enc(data.x, data.highway_id, data.edge_index)
    return z.detach()  # [N, road_dim]


def _road_encoder_and_graph(osm_root, city, stage0_ckpt, device):
    """Same load path as _road_embeddings, but returns the live RoadGAT module
    + graph (train mode, gradients enabled) instead of a detached tensor --
    for the encoder-unfrozen retrain probe (isolates the frozen-encoder
    confound flagged in the Silesia per-city retraining writeup)."""
    from roadgraph.io import load_pyg
    from models.road_encoder import RoadGAT

    data = load_pyg(osm_root, city).to(device)
    enc = RoadGAT(num_cont=data.x.size(1), num_highway=max(64, int(data.highway_id.max()) + 1)).to(device)
    enc.load_state_dict(torch.load(stage0_ckpt, map_location=device))
    enc.train()
    return enc, data


def build_loader(processed_root, osm_root, city, source, limit_trajs, batch, workers=0, cache_dir="ckpt/cache"):
    from torch.utils.data import DataLoader
    from dataset.trajectories import load_source_df, TrajectoryGraphDataset, collate_fn
    from dataset.candidates import CandidateIndex
    from dataset.config import SequenceConfig, RetrievalConfig

    df = load_source_df(processed_root, source, limit_trajs)
    ci = CandidateIndex.from_city(osm_root, city)
    retr = RetrievalConfig()
    cache_path = None
    if cache_dir is not None:
        n = limit_trajs if limit_trajs is not None else "all"
        cache_path = Path(cache_dir) / f"{source}_n{n}_r{retr.radius_m:g}_k{retr.k}.npz"
    ds = TrajectoryGraphDataset(df, {source: ci}, SequenceConfig(), retr, cache_path=cache_path)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=workers)
    return ds, loader


def train(processed_root, osm_root, stage0_ckpt, city="porto", source="porto",
          limit_trajs=4000, batch=16, epochs=3, lr=3e-4, layers=4, weight_decay=1e-4,
          out_dir="ckpt", ckpt_every=1000, log_every=50, device=None, seed=0, workers=0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    road_z = _road_embeddings(osm_root, city, stage0_ckpt, device)
    ds, loader = build_loader(processed_root, osm_root, city, source, limit_trajs, batch,
                               workers=workers, cache_dir=str(out / "cache"))
    model = Stage1Encoder(layers=layers, road_dim=road_z.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(loader)))
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    print(f"[stage1] city={city} trajs={len(ds)} roads={road_z.size(0)} "
          f"batch={batch} layers={layers} bf16={use_bf16} device={device}")
    model.train()
    step = 0
    for epoch in range(epochs):
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            # Change 8: 30% of valid fixes get d_perp zeroed on the candidate-token
            # INPUT (geo_mask); target-side d_perp is untouched (read from cache in
            # compute_losses), so masked fixes must be resolved from context/geometry
            # of *other* fixes, not this fix's own answer.
            geo_mask = (torch.rand_like(b["seq_mask"]) < 0.3) & (b["seq_mask"] > 0.5)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out_l = model.compute_losses(b, road_z, geo_mask=geo_mask)
                loss = out_l["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if step % log_every == 0:
                print(f"epoch {epoch} step {step:06d}  total {float(loss):.4f}  "
                      f"mgm {out_l['mgm']:.4f}  contrast {out_l['contrast']:.4f}  "
                      f"con_vis {out_l['contrast_visible']:.4f}  con_msk {out_l['contrast_masked']:.4f}  "
                      f"next {out_l['next']:.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if step and step % ckpt_every == 0:
                torch.save(model.state_dict(), out / f"stage1_{city}_step{step}.pt")
            step += 1
    torch.save(model.state_dict(), out / f"stage1_{city}.pt")
    print(f"[stage1] saved -> {out / f'stage1_{city}.pt'}")
    return model


def _smoke():
    torch.manual_seed(0)
    B, L, K, N, rd = 2, 6, 3, 20, 256
    road_z = torch.randn(N, rd)
    g = torch.Generator().manual_seed(1)
    batch = {
        "seq_mask": torch.ones(B, L),
        "length": torch.full((B,), L, dtype=torch.long),
        "lat": 41.15 + 0.01 * torch.randn(B, L, generator=g),
        "lon": -8.61 + 0.01 * torch.randn(B, L, generator=g),
        "dt": torch.rand(B, L, generator=g) * 5,
        "speed_mps": torch.rand(B, L, generator=g) * 10,
        "bearing_deg": torch.rand(B, L, generator=g) * 360,
        "hour": torch.randint(0, 24, (B, L), generator=g),
        "dow": torch.randint(0, 7, (B, L), generator=g),
        "cand_segment_id": torch.randint(0, N, (B, L, K), generator=g),
        "cand_d_perp_m": torch.rand(B, L, K, generator=g) * 40,
        "cand_heading_deg": torch.rand(B, L, K, generator=g) * 360,
        "cand_speed_kph": torch.rand(B, L, K, generator=g) * 50,
        "cand_oneway": torch.randint(0, 2, (B, L, K), generator=g),
        "cand_highway_id": torch.randint(0, 5, (B, L, K), generator=g),
        "cand_mask": torch.ones(B, L, K),
    }
    model = Stage1Encoder(d=64, heads=4, layers=2, ff=128, road_dim=rd)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = last = None
    for i in range(150):
        geo_mask = (torch.rand(B, L, generator=g) < 0.3) & (batch["seq_mask"] > 0.5)
        opt.zero_grad()
        out = model.compute_losses(batch, road_z, geo_mask=geo_mask)
        out["total"].backward(); opt.step()
        if first is None:
            first = float(out["total"])
        last = float(out["total"])
    print(f"[smoke] total {first:.3f} -> {last:.3f}  "
          f"(mgm {out['mgm']:.3f} contrast {out['contrast']:.3f} "
          f"con_vis {out['contrast_visible']:.3f} con_msk {out['contrast_masked']:.3f} "
          f"next {out['next']:.3f})")
    assert last < 0.7 * first, f"loss did not drop: {first:.3f} -> {last:.3f}"
    print("[smoke] PASS")


def main():
    p = argparse.ArgumentParser(description="Stage-1 GPS-Road cross-modal pretraining")
    p.add_argument("--processed-root", default="data/processed")
    p.add_argument("--osm-root", default="data/osm")
    p.add_argument("--city", default="porto")
    p.add_argument("--source", default="porto")
    p.add_argument("--stage0-ckpt", default="ckpt/stage0_porto.pt")
    p.add_argument("--limit-trajs", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--out-dir", default="ckpt")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        _smoke(); return
    train(a.processed_root, a.osm_root, a.stage0_ckpt, a.city, a.source, a.limit_trajs,
          a.batch, a.epochs, a.lr, a.layers, out_dir=a.out_dir, ckpt_every=a.ckpt_every,
          log_every=a.log_every, device=a.device, workers=a.workers)


if __name__ == "__main__":
    main()
