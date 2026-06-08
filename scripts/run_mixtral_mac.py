#!/usr/bin/env python3
"""
Best-effort Mixtral routing stats on Mac.

Mixtral-8x7B (~47B total params) usually does NOT fit 24GB unified memory in HF+MPS.
This script tries 4-bit (CUDA/bitsandbytes) or fails with a clear message.

For a Mac-friendly second leaderboard model, use: python scripts/run_jetmoe_mac.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
RESULTS = ROOT / "results"


def main():
    sys.path.insert(0, str(ROOT))
    from moe_bench.cli import run_analysis
    from moe_bench.device import default_device, is_apple_silicon

    device = default_device()
    print(f"Device: {device} (Apple Silicon: {is_apple_silicon()})")

    if is_apple_silicon():
        print(
            "\nWARNING: Mixtral-8x7B typically requires >40GB RAM or cloud GPU.\n"
            "On 24GB Mac this often OOMs. Trying 100 samples @ seq 64 anyway.\n"
            "If it fails, use RunPod (~$0.50/hr) or run scripts/run_jetmoe_mac.py instead.\n"
        )

    try:
        run_analysis(
            MIXTRAL,
            "synthetic",
            RESULTS / "routing_stats_mixtral_mac_attempt.json",
            max_samples=100,
            batch_size=1,
            max_length=64,
            load_in_4bit=(device == "cuda"),
            device=device if device != "cuda" else None,
            figures_dir=RESULTS / "figures",
        )
        print("Success — update leaderboard: python scripts/update_leaderboard.py")
    except Exception as e:
        print(f"\nMixtral run failed (expected on 24GB Mac): {e}\n")
        print("Cloud one-liner:")
        print(
            "  moe-bench analyze --model mistralai/Mixtral-8x7B-Instruct-v0.1 "
            "--dataset synthetic --4bit --max-samples 100 "
            "--output results/routing_stats_mixtral.json"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
