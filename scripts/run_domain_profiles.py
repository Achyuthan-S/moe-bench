#!/usr/bin/env python3
"""
Cross-model domain routing profiles (math / code / general).

Matches OLMoE phase-2 settings: 40 samples, max_length=256, batch_size=1.
Run on Mac MPS after: pip install -e .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PROFILES = [
    ("jetmoe/jetmoe-8b", "jetmoe"),
    ("Qwen/Qwen1.5-MoE-A2.7B", "qwen15_moe"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=[m[0] for m in PROFILES],
        help="HuggingFace model ids",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["math", "general", "code"],
        choices=["math", "general", "code"],
    )
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from moe_bench.cli import run_analysis

    slug_map = {m[0]: m[1] for m in PROFILES}

    import gc
    import torch

    for model_id in args.models:
        slug = slug_map.get(model_id, model_id.split("/")[-1].lower().replace("-", "_"))
        device = args.device
        if "qwen" in model_id.lower() and device == "mps":
            # Qwen1.5-MoE-A2.7B often OOMs on 24GB after JetMoE; use CPU unless user overrides.
            device = "cpu"
            print(f"Note: {model_id} → cpu (avoids MPS OOM after prior model loads)")
        for domain in args.domains:
            out = RESULTS / f"routing_stats_{slug}_{domain}.json"
            print(f"\n=== {model_id} · domain={domain} → {out.name} · device={device} ===")
            try:
                run_analysis(
                    model_id,
                    domain,
                    out,
                    max_samples=args.max_samples,
                    batch_size=1,
                    max_length=args.max_length,
                    load_in_4bit=False,
                    device=device,
                    figures_dir=RESULTS / "figures",
                    profile_note="domain-matched (same prompts as OLMoE phase-2)",
                    run_seed=42,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device != "cpu":
                    print(f"MPS OOM — retrying {model_id} on CPU...")
                    gc.collect()
                    if hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                    run_analysis(
                        model_id,
                        domain,
                        out,
                        max_samples=args.max_samples,
                        batch_size=1,
                        max_length=args.max_length,
                        load_in_4bit=False,
                        device="cpu",
                        figures_dir=RESULTS / "figures",
                        profile_note="domain-matched (CPU fallback after MPS OOM)",
                        run_seed=42,
                    )
                else:
                    raise
            gc.collect()
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()

    print("\nDone. Regenerate table: python scripts/update_leaderboard.py")


if __name__ == "__main__":
    main()
