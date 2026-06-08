#!/usr/bin/env python3
"""
Run entire MoE-Bench project on Mac (excludes Mixtral / A100).

Steps: Phase 2 analysis → Phase 3 mini-train → leaderboard → optional lm-eval
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    py = sys.executable
    run([py, "scripts/run_phase2.py"])
    run([py, "scripts/train_lora_ldiv.py", "--max-samples", "48", "--epochs", "1"])
    run([py, "scripts/run_ablations.py"])
    run([py, "scripts/update_leaderboard.py"])
    try:
        run([py, "scripts/run_lm_eval.py", "--limit", "20"])
    except subprocess.CalledProcessError:
        print("lm-eval skipped or failed (optional)")
    print("\nDone. See results/ANALYSIS_NOTES.md and paper/main.tex")


if __name__ == "__main__":
    main()
