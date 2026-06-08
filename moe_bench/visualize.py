"""Figures for MoE-Bench papers and reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def utilisation_heatmap(
    df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    title: str = "Expert Utilisation by Layer",
    figsize: tuple[float, float] = (10, 12),
    dpi: int = 300,
) -> plt.Figure:
    """
    Hero figure: layers x experts heatmap of token routing fraction.

    Expects df from StatsCollector.to_dataframe() with expert_* columns.
    """
    expert_cols = sorted(
        [c for c in df.columns if c.startswith("expert_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not expert_cols:
        raise ValueError("DataFrame has no expert_* columns")

    matrix = df.sort_values("layer")[expert_cols].values
    layers = df.sort_values("layer")["layer"].values
    n_experts = len(expert_cols)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=matrix.max() if matrix.size else 1,
        xticklabels=[str(i + 1) for i in range(n_experts)],
        yticklabels=[str(int(l)) for l in layers],
        cbar_kws={"label": "Fraction of tokens routed"},
    )
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    return fig


def entropy_vs_layer(
    df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    title: str = "Routing Entropy vs Layer Depth",
    dpi: int = 300,
) -> plt.Figure:
    """Line plot of mean routing entropy per layer."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_df = df.sort_values("layer")
    ax.plot(sorted_df["layer"], sorted_df["mean_entropy"], marker="o", linewidth=2)
    ax.fill_between(
        sorted_df["layer"],
        sorted_df["mean_entropy"] - sorted_df["std_entropy"],
        sorted_df["mean_entropy"] + sorted_df["std_entropy"],
        alpha=0.2,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean routing entropy H")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def max_expert_util_distribution(
    df: pd.DataFrame,
    *,
    threshold: float = 0.40,
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Histogram of max expert utilisation per layer (collapse threshold line)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["max_expert_util"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", label=f"collapse τ={threshold}")
    ax.set_xlabel("Max expert utilisation (top-1)")
    ax.set_ylabel("Number of layers")
    ax.set_title("Distribution of per-layer max expert utilisation")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig
