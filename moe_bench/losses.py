"""Auxiliary losses: L_div, router z-loss (ST-MoE), switch load-balancing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diversity_loss(
    layer_entropies: list[torch.Tensor],
    tau: float = 1.5,
    collapsed_mask: list[bool] | None = None,
) -> torch.Tensor:
    """L_div = sum_l max(0, tau - H_l) over collapsed layers."""
    if not layer_entropies:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=layer_entropies[0].device)
    for i, h in enumerate(layer_entropies):
        if collapsed_mask is not None and not collapsed_mask[i]:
            continue
        total = total + F.relu(tau - h)
    return total


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE router z-loss: mean(logsumexp(logits)^2)."""
    log_z = torch.logsumexp(router_logits.float(), dim=-1)
    return (log_z**2).mean()


def switch_load_balancing_loss(
    router_probs: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Switch Transformer auxiliary loss (simplified).
    router_probs: (tokens, num_experts)
    """
    density = router_probs.mean(dim=0)
    load = (router_probs > 0).float().mean(dim=0)
    return num_experts * (density * load).sum()


def routing_entropy_torch(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    log_p = torch.log(probs.clamp(min=eps))
    return -(probs * log_p).sum(dim=-1).mean()
