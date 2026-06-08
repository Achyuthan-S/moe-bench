#!/usr/bin/env python3
"""Phase 3: LoRA fine-tune OLMoE with L_div (Mac-friendly)."""

import argparse
from pathlib import Path

from moe_bench.device import default_device
from moe_bench.train import train_moe_aux

MODEL = "allenai/OLMoE-1B-7B-0924"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "training"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux-loss", choices=["ldiv", "zloss", "switch", "none"], default="ldiv")
    parser.add_argument("--lambda-div", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=RESULTS / "train_ldiv.json")
    args = parser.parse_args()

    train_moe_aux(
        MODEL,
        args.output,
        aux_loss=args.aux_loss,
        lambda_div=args.lambda_div,
        tau=args.tau,
        max_samples=args.max_samples,
        epochs=args.epochs,
        device=default_device(),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
