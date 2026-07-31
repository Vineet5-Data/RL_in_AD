"""Self-check for Stage-0 contrastive logic on a toy graph (no real data, no training run)."""

import torch
from torch_geometric.data import Data

from models.road_encoder import ProjectionHead, RoadGAT
from training.stage0 import nt_xent, step


def _toy(n=40, e=120, dim=13, H=5):
    return Data(x=torch.randn(n, dim), highway_id=torch.randint(0, H, (n,)),
                edge_index=torch.randint(0, n, (2, e)), num_nodes=n)


def test_nt_xent_rewards_matching_views():
    z = torch.randn(16, 8)
    assert nt_xent(z, z.clone()) < nt_xent(z, torch.randn(16, 8))


def test_forward_shape_and_loss_decreases():
    torch.manual_seed(0)
    data = _toy()
    model, head = RoadGAT(num_highway=5), ProjectionHead()
    assert model(data.x, data.highway_id, data.edge_index).shape == (40, 256)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=1e-2)
    losses = [step(model, head, data, opt, batch=32) for _ in range(20)]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-5:]) < sum(losses[:5])
