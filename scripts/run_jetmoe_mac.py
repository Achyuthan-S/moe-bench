#!/usr/bin/env python3
"""Second MoE for routing profile table — JetMoE-8B (~2.2B active, fits many 24GB Macs)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "jetmoe/jetmoe-8b"
RESULTS = ROOT / "results"


def main():
    sys.path.insert(0, str(ROOT))
    from moe_bench.cli import run_analysis
    from moe_bench.device import default_device

    print(f"JetMoE routing profile run on {default_device()} (8B total, ~2.2B active/token)")
    run_analysis(
        MODEL,
        "synthetic",
        RESULTS / "routing_stats_jetmoe.json",
        max_samples=100,
        batch_size=1,
        max_length=128,
        load_in_4bit=False,
        device="mps",
        figures_dir=RESULTS / "figures",
    )
    print("Done. Run: python scripts/update_leaderboard.py  # → results/ROUTING_PROFILE_TABLE.md")


if __name__ == "__main__":
    main()
