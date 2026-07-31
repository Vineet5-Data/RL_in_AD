"""Stage 0: GRACE-style graph contrastive pretrain of the P1b road encoder.

Run explicitly (does nothing on import):
    python -m training.stage0 --city porto --osm-root data/osm --epochs 50 --out ckpt/stage0_porto.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from models.road_encoder import ProjectionHead, RoadGAT


def drop_feature(x, p):
    return x * (torch.rand(x.size(1), device=x.device) >= p).float()


def augment(data, p_f, p_e):
    ei, _ = dropout_edge(data.edge_index, p=p_e)
    return drop_feature(data.x, p_f), ei


def nt_xent(z1, z2, tau: float = 0.5):
    z1, z2 = F.normalize(z1), F.normalize(z2)
    n = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, device=z.device) + n,
                         torch.arange(n, device=z.device)], 0)
    return F.cross_entropy(sim, targets)


def step(model, head, data, opt, batch, p_f=0.3, p_e=0.3, tau=0.5):
    model.train()
    x1, e1 = augment(data, p_f, p_e)
    x2, e2 = augment(data, p_f, p_e)
    z1 = head(model(x1, data.highway_id, e1))
    z2 = head(model(x2, data.highway_id, e2))
    idx = torch.randint(0, z1.size(0), (min(batch, z1.size(0)),), device=z1.device)
    # ponytail: O(batch^2) in-batch negatives; full-graph NT-Xent is 154k^2 -- never.
    # Add NeighborLoader subgraph sampling if full-graph encode OOMs on the 4060.
    loss = nt_xent(z1[idx], z2[idx], tau)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


def train(city, osm_root, epochs, batch, lr, out, device):
    from roadgraph.io import load_pyg

    vocab = json.loads((Path(osm_root) / "highway_vocab.json").read_text())
    data = load_pyg(osm_root, city).to(device)
    n_hw = max(len(vocab), int(data.highway_id.max()) + 1)
    model = RoadGAT(num_highway=n_hw).to(device)
    head = ProjectionHead().to(device)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=lr)
    for ep in range(epochs):
        print(f"epoch {ep:03d} nt_xent {step(model, head, data, opt, batch):.4f}", flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 0 graph contrastive pretrain (P1b).")
    ap.add_argument("--city", required=True)
    ap.add_argument("--osm-root", default="data/osm")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    train(a.city, a.osm_root, a.epochs, a.batch, a.lr, a.out, a.device)


if __name__ == "__main__":
    main()
