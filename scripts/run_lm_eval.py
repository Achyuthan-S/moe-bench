#!/usr/bin/env python3
"""Phase 3: lm-eval-harness wrapper (optional; run when installed)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL = "allenai/OLMoE-1B-7B-0924"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "lm_eval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tasks", default="hellaswag,winogrande,arc_challenge")
    parser.add_argument("--limit", type=int, default=50, help="Examples per task (Mac smoke test)")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "eval_results.json"

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={args.model},trust_remote_code=True",
        "--tasks",
        args.tasks,
        "--batch_size",
        "1",
        "--limit",
        str(args.limit),
        "--output_path",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Results: {out}")
    except FileNotFoundError:
        print("lm-eval not installed. Run: pip install lm-eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
