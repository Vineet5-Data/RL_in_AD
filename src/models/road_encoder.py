"""P1b road-segment encoder: line-graph GAT over x[N,13] + highway_id -> [N,256]."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class RoadGAT(nn.Module):
    def __init__(self, num_cont: int = 13, num_highway: int = 64, hw_dim: int = 16,
                 hidden: int = 64, heads: int = 4, out_dim: int = 256):
        super().__init__()
        self.hw_emb = nn.Embedding(num_highway, hw_dim)
        self.g1 = GATConv(num_cont + hw_dim, hidden, heads=heads)
        self.g2 = GATConv(hidden * heads, out_dim, heads=1)

    def forward(self, x, highway_id, edge_index):
        h = torch.cat([x, self.hw_emb(highway_id)], dim=-1)
        h = F.elu(self.g1(h, edge_index))
        return self.g2(h, edge_index)


class ProjectionHead(nn.Module):
    def __init__(self, dim: int = 256, proj: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj))

    def forward(self, z):
        return self.net(z)
