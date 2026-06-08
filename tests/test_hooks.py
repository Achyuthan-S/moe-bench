"""Tests for router gate discovery on a minimal mock MoE model."""

import torch
import torch.nn as nn

from moe_bench.attach import attach
from moe_bench.hooks import discover_router_gates


class MockMoEBlock(nn.Module):
    def __init__(self, hidden: int = 32, n_experts: int = 8):
        super().__init__()
        self.gate = nn.Linear(hidden, n_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        return x  # simplified


class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(32, 32)
        self.block_sparse_moe = MockMoEBlock()

    def forward(self, x):
        x = self.self_attn(x)
        return self.block_sparse_moe(x)


class MockMoEModel(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_discover_router_gates():
    model = MockMoEModel(n_layers=4)
    specs = discover_router_gates(model)
    assert len(specs) == 4


def test_attach_collects_routing():
    model = MockMoEModel(n_layers=2)
    stats = attach(model, flush_every=0, output_dir=None)
    x = torch.randn(2, 16, 32)
    model(x)
    df = stats.to_dataframe()
    assert len(df) == 2
    assert "mean_entropy" in df.columns
