"""Routing metrics: entropy, utilisation, load imbalance, collapse detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def routing_entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy H = -sum(p * log(p)) over the expert dimension.

    Args:
        probs: routing probabilities, shape (..., num_experts)
        axis: dimension of experts (default last)

    Returns:
        Entropy per token position, shape probs.shape with axis removed.
    """
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def expert_utilisation(
    probs: np.ndarray,
    top_k: int | None = None,
) -> np.ndarray:
    """
    Fraction of tokens assigned to each expert (argmax or top-k mass).

    Args:
        probs: (num_tokens, num_experts) routing weights (post-softmax)
        top_k: if set, count token toward expert if in top-k by weight

    Returns:
        utilisation vector of shape (num_experts,) summing to 1.
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probs, got shape {probs.shape}")

    n_tokens, n_experts = probs.shape
    if n_tokens == 0:
        return np.zeros(n_experts, dtype=np.float64)

    if top_k is None or top_k >= n_experts:
        assigned = np.argmax(probs, axis=-1)
        counts = np.bincount(assigned, minlength=n_experts).astype(np.float64)
        return counts / counts.sum()

    # Top-k: each token contributes 1/k to each selected expert
    top_idx = np.argpartition(-probs, top_k - 1, axis=-1)[:, :top_k]
    counts = np.zeros(n_experts, dtype=np.float64)
    for k in range(top_k):
        np.add.at(counts, top_idx[:, k], 1.0 / top_k)
    return counts / counts.sum()


def load_imbalance_coefficient(utilisation: np.ndarray, eps: float = 1e-12) -> float:
    """max(utilisation) / mean(utilisation)."""
    mean_u = utilisation.mean()
    if mean_u < eps:
        return float("inf")
    return float(utilisation.max() / mean_u)


def collapse_rate(
    layer_utilisations: dict[int, np.ndarray],
    threshold: float = 0.40,
) -> tuple[float, list[int]]:
    """
    Fraction of layers where top-1 expert handles > threshold of tokens.

    Returns:
        (collapse_rate, list of collapsed layer indices)
    """
    collapsed = []
    for layer_idx, util in sorted(layer_utilisations.items()):
        if util.max() > threshold:
            collapsed.append(layer_idx)
    n = len(layer_utilisations)
    rate = len(collapsed) / n if n > 0 else 0.0
    return rate, collapsed


def compute_layer_metrics(
    router_weights: dict[int, list[np.ndarray]],
    collapse_threshold: float = 0.40,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate per-layer metrics from collected router weight tensors.

    Args:
        router_weights: layer_idx -> list of arrays (batch*seq, num_experts)

    Returns:
        DataFrame with columns: layer, mean_entropy, std_entropy,
        expert_0..N utilisation, load_imbalance, max_expert_util, collapsed
    """
    rows = []
    for layer_idx in sorted(router_weights.keys()):
        chunks = router_weights[layer_idx]
        if not chunks:
            continue

        all_probs = np.concatenate(chunks, axis=0)
        ent = routing_entropy(all_probs, axis=-1)
        util = expert_utilisation(all_probs, top_k=top_k)
        imb = load_imbalance_coefficient(util)
        max_util = float(util.max())

        row: dict = {
            "layer": layer_idx,
            "mean_entropy": float(ent.mean()),
            "std_entropy": float(ent.std()),
            "load_imbalance": imb,
            "max_expert_util": max_util,
            "collapsed": max_util > collapse_threshold,
            "num_tokens": all_probs.shape[0],
        }
        for e_idx, u in enumerate(util):
            row[f"expert_{e_idx}"] = float(u)
        rows.append(row)

    return pd.DataFrame(rows)
