#!/usr/bin/env python3
"""Phase 2: collapse threshold, context length, domain analysis (Mac / no A100)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from moe_bench.analysis import (
    build_analysis_report,
    compare_context_lengths,
    domain_chi_squared,
    recommend_collapse_threshold,
    sweep_collapse_thresholds,
)
from moe_bench.data import get_stream
from moe_bench.device import default_device, load_causal_lm
from moe_bench.runner import run_routing_pass, save_analysis_run

MODEL = "allenai/OLMoE-1B-7B-0924"
RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"
SAMPLES = 40
BATCH = 1


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = default_device()
    print(f"Phase 2 analysis · {MODEL} · device={device}")

    model, tokenizer, _ = load_causal_lm(MODEL, device=device)

    # Main combined (C4 + pile subset via synthetic domains for offline reliability)
    main_report = save_analysis_run(
        run_routing_pass(
            model,
            tokenizer,
            get_stream("synthetic", max_samples=SAMPLES * 3),
            max_samples=SAMPLES * 3,
            batch_size=BATCH,
            max_length=256,
        ),
        RESULTS / "routing_stats_olmoe_main.json",
        {"model": MODEL, "dataset": "synthetic_extended", "tag": "olmoe_main"},
        figures_dir=FIGURES,
    )

    # Context lengths
    lengths = [128, 512, 1024]
    length_reports = {}
    for L in lengths:
        print(f"  context length {L}...")
        col = run_routing_pass(
            model,
            tokenizer,
            get_stream("general", max_samples=SAMPLES),
            max_samples=SAMPLES,
            batch_size=BATCH,
            max_length=L,
        )
        rep = save_analysis_run(
            col,
            RESULTS / f"routing_stats_olmoe_len{L}.json",
            {"model": MODEL, "max_length": L, "tag": f"len{L}"},
        )
        length_reports[L] = rep

    context_df = compare_context_lengths(length_reports)
    context_df.to_csv(RESULTS / "context_length_study.csv", index=False)

    # Domains
    domain_dfs = {}
    domain_reports = {}
    for domain in ("general", "code", "math"):
        print(f"  domain {domain}...")
        col = run_routing_pass(
            model,
            tokenizer,
            get_stream(domain, max_samples=SAMPLES),
            max_samples=SAMPLES,
            batch_size=BATCH,
            max_length=256,
        )
        path = RESULTS / f"routing_stats_olmoe_{domain}.json"
        domain_reports[domain] = save_analysis_run(
            col, path, {"model": MODEL, "domain": domain, "tag": domain}
        )
        domain_dfs[domain] = col.to_dataframe()

    chi2_df = domain_chi_squared(domain_dfs)
    chi2_df.to_csv(RESULTS / "domain_chi2.csv", index=False)

    main_df = pd.read_csv(RESULTS / "routing_stats_olmoe_main.csv")
    sweep_df = sweep_collapse_thresholds(main_df)
    sweep_df.to_csv(RESULTS / "collapse_threshold_sweep.csv", index=False)
    rec_t = recommend_collapse_threshold(sweep_df)

    report = build_analysis_report(
        model=MODEL,
        main_csv=RESULTS / "routing_stats_olmoe_main.csv",
        threshold_sweep=sweep_df,
        context_df=context_df,
        domain_chi2=chi2_df,
        recommended_threshold=rec_t,
    )
    report["domain_reports"] = domain_reports
    (RESULTS / "phase2_summary.json").write_text(json.dumps(report, indent=2))

    notes = _render_notes(report, main_df, sweep_df, context_df, chi2_df, domain_dfs)
    (RESULTS / "ANALYSIS_NOTES.md").write_text(notes)

    print(f"\nRecommended collapse threshold: {rec_t:.2f}")
    print(f"Wrote {RESULTS / 'ANALYSIS_NOTES.md'}")


def _render_notes(report, main_df, sweep_df, context_df, chi2_df, domain_dfs) -> str:
    worst = main_df.loc[main_df["max_expert_util"].idxmax()]
    sig = chi2_df[chi2_df["p_value"] < 0.05] if len(chi2_df) else chi2_df

    lines = [
        "# MoE-Bench Analysis Notes (Section 3)",
        "",
        f"**Model:** {report['model']}",
        "",
        "## Collapse threshold",
        f"- Recommended τ (top-1 utilisation): **{report['recommended_collapse_threshold']:.2f}**",
        f"- At 0.40: collapse rate from main run varies by dataset; see `collapse_threshold_sweep.csv`",
        "",
        "## Layer depth",
        f"- Early layers (0–3) mean entropy: **{report['main_metrics']['early_layer_entropy']:.3f}**",
        f"- Late layers (12–15) mean entropy: **{report['main_metrics']['late_layer_entropy']:.3f}**",
        f"- Worst layer **{int(worst['layer'])}**: max expert util **{worst['max_expert_util']:.1%}**",
        "",
        "## Context length",
    ]
    for _, row in context_df.iterrows():
        lines.append(
            f"- L={int(row['max_length'])}: entropy={row['mean_entropy']:.3f}, "
            f"collapse_rate={row['collapse_rate']:.1%}"
        )
    lines.extend(["", "## Domain routing (chi-squared)", f"- Significant layers (p<0.05): **{len(sig)}** / {len(chi2_df)}"])
    for domain, df in domain_dfs.items():
        w = df.loc[df["max_expert_util"].idxmax()]
        lines.append(
            f"- **{domain}**: peak layer {int(w['layer'])} → expert util {w['max_expert_util']:.1%}"
        )
    if len(sig):
        row = sig.iloc[0]
        lines.append(f"- Example: layer {int(row['layer'])} chi2={row['chi2']:.2f}, p={row['p_value']:.4f}")
    lines.extend(["", "## Figures", "- `figures/heatmap_*.png`, `entropy_*.png`, `maxutil_*.png`"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
