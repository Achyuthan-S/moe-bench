"""Phase 2: collapse threshold, context length, domain routing statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from moe_bench.metrics import collapse_rate, compute_layer_metrics


def sweep_collapse_thresholds(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Collapse rate at each threshold from per-layer max_expert_util."""
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    expert_cols = [c for c in df.columns if c.startswith("expert_")]
    layer_utils = {
        int(row["layer"]): row[expert_cols].values.astype(float)
        for _, row in df.iterrows()
    }

    rows = []
    for t in thresholds:
        rate, layers = collapse_rate(layer_utils, threshold=t)
        rows.append(
            {
                "threshold": t,
                "collapse_rate": rate,
                "num_collapsed_layers": len(layers),
                "collapsed_layers": layers,
            }
        )
    return pd.DataFrame(rows)


def recommend_collapse_threshold(sweep_df: pd.DataFrame) -> float:
    """
    Pick threshold at largest jump in collapsed-layer count (elbow on layer count).
    """
    counts = sweep_df["num_collapsed_layers"].values
    thresholds = sweep_df["threshold"].values
    if len(counts) < 2:
        return 0.40
    jumps = np.diff(counts.astype(float))
    idx = int(np.argmax(jumps)) + 1 if jumps.max() > 0 else np.argmin(np.abs(thresholds - 0.40))
    return float(thresholds[idx])


def domain_chi_squared(
    domain_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Per-layer chi-squared test: H0 = routing distribution independent of domain.

    Uses expert utilisation vectors as categorical distributions.
    """
    domains = list(domain_dfs.keys())
    if len(domains) < 2:
        raise ValueError("Need at least two domains")

    layers = sorted(domain_dfs[domains[0]]["layer"].unique())
    rows = []
    for layer in layers:
        observed = []
        for d in domains:
            row = domain_dfs[d][domain_dfs[d]["layer"] == layer]
            if row.empty:
                continue
            expert_cols = [c for c in row.columns if c.startswith("expert_")]
            util = row.iloc[0][expert_cols].values.astype(float)
            observed.append(util * 1000)  # scale to pseudo-counts
        if len(observed) < 2:
            continue
        table = np.vstack(observed)
        # chi2_contingency expects independence; use utilisation as multinomial samples
        chi2, p, dof, _ = stats.chi2_contingency(table + 1e-6)
        rows.append({"layer": int(layer), "chi2": float(chi2), "p_value": float(p), "dof": int(dof)})
    return pd.DataFrame(rows)


def compare_context_lengths(
    length_reports: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    """Summarise entropy and collapse vs sequence length."""
    rows = []
    for length, report in sorted(length_reports.items()):
        rows.append(
            {
                "max_length": length,
                "mean_entropy": report.get("mean_entropy"),
                "collapse_rate": report.get("collapse_rate"),
                "avg_load_imbalance": report.get("avg_load_imbalance"),
                "num_collapsed_layers": len(report.get("collapsed_layers", [])),
            }
        )
    return pd.DataFrame(rows)


def load_metrics_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_analysis_report(
    *,
    model: str,
    main_csv: Path,
    threshold_sweep: pd.DataFrame,
    context_df: pd.DataFrame | None,
    domain_chi2: pd.DataFrame | None,
    recommended_threshold: float,
) -> dict[str, Any]:
    df = load_metrics_csv(main_csv)
    return {
        "model": model,
        "recommended_collapse_threshold": recommended_threshold,
        "main_metrics": {
            "mean_entropy": float(df["mean_entropy"].mean()),
            "early_layer_entropy": float(df.head(4)["mean_entropy"].mean()),
            "late_layer_entropy": float(df.tail(4)["mean_entropy"].mean()),
            "max_expert_util_global": float(df["max_expert_util"].max()),
            "worst_layer": int(df.loc[df["max_expert_util"].idxmax(), "layer"]),
        },
        "threshold_sweep": threshold_sweep.to_dict(orient="records"),
        "context_length": context_df.to_dict(orient="records") if context_df is not None else [],
        "domain_chi2_significant_layers": (
            int((domain_chi2["p_value"] < 0.05).sum()) if domain_chi2 is not None and len(domain_chi2) else 0
        ),
        "domain_chi2": domain_chi2.to_dict(orient="records") if domain_chi2 is not None else [],
    }
