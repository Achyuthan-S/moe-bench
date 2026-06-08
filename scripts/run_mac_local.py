#!/usr/bin/env python3
"""
Run MoE-Bench on MacBook Pro (Apple Silicon, no A100).

Uses OLMoE-1B-7B (fits ~24GB unified memory in float16 on MPS).
Mixtral-8x7B requires >40GB even quantized — not runnable on this hardware.

Example:
  python scripts/run_mac_local.py
"""

from pathlib import Path

from moe_bench.cli import run_analysis
from moe_bench.device import default_device, is_apple_silicon

# 7B total / 1B active — fits 24GB Mac; real MoE routing (64 experts/layer)
MODEL = "allenai/OLMoE-1B-7B-0924"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Platform: {'Apple Silicon' if is_apple_silicon() else 'other'}, device={default_device()}")
    print(f"Model: {MODEL}")
    print("Dataset: synthetic (offline). Use --dataset c4 for streaming C4 (slower).")

    report = run_analysis(
        MODEL,
        "synthetic",
        RESULTS_DIR / "routing_stats_olmoe_mac.json",
        max_samples=64,
        batch_size=1,
        max_length=128,
        load_in_4bit=False,
        device="mps",
        figures_dir=RESULTS_DIR / "figures",
    )

    print("\n=== MoE-Bench Mac Run Complete ===")
    print(f"Collapse rate: {report.get('collapse_rate', 0):.1%}")
    print(f"Mean entropy:  {report.get('mean_entropy', 0):.3f}")
    print(f"Collapsed layers ({len(report.get('collapsed_layers', []))}): {report.get('collapsed_layers', [])}")
    print(f"CSV: {RESULTS_DIR / 'routing_stats_olmoe_mac.csv'}")
    print(f"Figures: {RESULTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
