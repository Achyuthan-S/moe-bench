"""Training step with CE + diversity auxiliary loss (Phase 3 fine-tuning)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_bench.losses import diversity_loss, routing_entropy_torch
from moe_bench.config import MoETrainingConfig


def compute_total_loss(
    model,
    batch: dict[str, torch.Tensor],
    router_logits_by_layer: dict[int, torch.Tensor],
    config: MoETrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    total = ce_loss + lambda * L_div

    Args:
        model: causal LM with labels in batch
        batch: input_ids, attention_mask, labels
        router_logits_by_layer: layer_idx -> (tokens, n_experts) from hooks during forward
    """
    outputs = model(**batch)
    ce_loss = outputs.loss

    layer_entropies = []
    collapsed_mask = []
    for layer_idx in sorted(router_logits_by_layer.keys()):
        logits = router_logits_by_layer[layer_idx]
        h = routing_entropy_torch(logits)
        layer_entropies.append(h)
        probs = F.softmax(logits.float(), dim=-1)
        max_util = probs.max(dim=-1).values.mean()
        collapsed_mask.append(max_util.item() > config.collapse_threshold)

    l_div = diversity_loss(layer_entropies, tau=config.tau, collapsed_mask=collapsed_mask)
    total = ce_loss + config.lambda_div * l_div

    metrics = {
        "ce_loss": ce_loss.item(),
        "l_div": l_div.item(),
        "total_loss": total.item(),
        "mean_entropy": sum(h.item() for h in layer_entropies) / max(len(layer_entropies), 1),
    }
    return total, metrics
