"""Router logit extraction for tuple outputs (Qwen2-MoE)."""

import torch

from moe_bench.hooks import _extract_router_logits


def test_qwen_tuple_uses_full_logits_not_indices():
    logits = torch.randn(8, 60)
    scores = torch.randn(8, 4)
    indices = torch.randint(0, 60, (8, 4))
    out = _extract_router_logits((logits, scores, indices))
    assert out.shape == (8, 60)
