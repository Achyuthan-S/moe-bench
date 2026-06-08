#!/usr/bin/env python3
"""Phase 1: Mixtral routing analysis on C4 + Pile (run on A100)."""

from pathlib import Path

from moe_bench.cli import run_analysis

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = run_analysis(
        MODEL,
        "combined",
        RESULTS_DIR / "routing_stats_mixtral.json",
        max_samples=10_000,
        batch_size=4,
        max_length=512,
        figures_dir=RESULTS_DIR / "figures",
    )
    print(f"Collapse rate: {report.get('collapse_rate', 'N/A')}")
    print(f"Mean entropy: {report.get('mean_entropy', 'N/A')}")
    print(f"Collapsed layers: {report.get('collapsed_layers', [])}")


if __name__ == "__main__":
    main()
