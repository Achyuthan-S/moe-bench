"""Bootstrap uncertainty for routing profile metrics (layer-level resampling)."""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_routing_cis(
    per_layer: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    collapse_threshold: float = 0.40,
) -> dict[str, Any]:
    """
    Layer-bootstrap 95% CIs for mean entropy and collapse rate.

    Treats each instrumented router (layer row) as one unit; weights layers equally.
    This is a conservative transparency measure—not token-level Hoeffding bounds.
    """
    if not per_layer:
        return {}

    entropies = np.array([float(row["mean_entropy"]) for row in per_layer], dtype=np.float64)
    collapsed = np.array(
        [bool(row.get("collapsed", row.get("max_expert_util", 0) > collapse_threshold)) for row in per_layer],
        dtype=bool,
    )
    rng = np.random.default_rng(seed)
    n = len(entropies)
    boot_ent = np.empty(n_boot)
    boot_col = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_ent[i] = entropies[idx].mean()
        boot_col[i] = collapsed[idx].mean()

    ent_lo, ent_hi = np.percentile(boot_ent, [2.5, 97.5])
    col_lo, col_hi = np.percentile(boot_col, [2.5, 97.5])
    total_tokens = int(sum(int(row.get("num_tokens", 0)) for row in per_layer))

    return {
        "bootstrap_layers": n,
        "bootstrap_replicates": n_boot,
        "total_token_positions": total_tokens,
        "mean_entropy_ci_95": [float(ent_lo), float(ent_hi)],
        "collapse_rate_ci_95": [float(col_lo), float(col_hi)],
        "ci_method": "layer_bootstrap_equal_weight",
    }
