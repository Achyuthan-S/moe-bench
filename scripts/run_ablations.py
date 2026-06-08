#!/usr/bin/env python3
"""Phase 3: λ/τ ablations + baselines (Mac mini-runs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_bench.device import default_device
from moe_bench.train import train_moe_aux

MODEL = "allenai/OLMoE-1B-7B-0924"
RESULTS = ROOT / "results" / "training"
FIGURES = ROOT / "results" / "figures"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = default_device()
    rows = []

    for lam in [0.001, 0.01, 0.05]:
        out = RESULTS / f"ablation_lambda_{lam}.json"
        if not out.exists():
            m = train_moe_aux(MODEL, out, aux_loss="ldiv", lambda_div=lam, tau=1.5, max_samples=24, epochs=1, device=device)
        else:
            m = json.loads(out.read_text())
        rows.append({"method": f"L_div λ={lam}", **m})

    for tau in [1.0, 1.5, 2.0]:
        out = RESULTS / f"ablation_tau_{tau}.json"
        if not out.exists():
            m = train_moe_aux(MODEL, out, aux_loss="ldiv", lambda_div=0.01, tau=tau, max_samples=24, epochs=1, device=device)
        else:
            m = json.loads(out.read_text())
        rows.append({"method": f"L_div τ={tau}", **m})

    for aux in ["zloss", "switch", "none"]:
        out = RESULTS / f"baseline_{aux}.json"
        if not out.exists():
            m = train_moe_aux(MODEL, out, aux_loss=aux, max_samples=24, epochs=1, device=device)
        else:
            m = json.loads(out.read_text())
        rows.append({"method": aux, **m})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "ablation_comparison.csv", index=False)

    sub = df[df["method"].str.contains("λ")]
    if len(sub):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub["mean_entropy"], sub["collapse_rate_40"], s=80)
        for _, r in sub.iterrows():
            ax.annotate(r["method"], (r["mean_entropy"], r["collapse_rate_40"]), fontsize=8)
        ax.set_xlabel("Mean routing entropy")
        ax.set_ylabel("Collapse rate @ 0.40")
        ax.set_title("Entropy vs collapse (λ ablation)")
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "pareto_lambda_ablation.png", dpi=300, bbox_inches="tight")

    print(df[["method", "mean_entropy", "collapse_rate_40"]].to_string(index=False))


if __name__ == "__main__":
    main()
