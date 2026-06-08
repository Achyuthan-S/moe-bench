"""CLI: moe-bench analyze --model ... --dataset c4 --output results.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import moe_bench
from moe_bench.data import batch_tokenize, get_stream
from moe_bench.device import default_load_in_4bit, load_causal_lm, model_input_device
from moe_bench.routing_uncertainty import bootstrap_routing_cis
from moe_bench.visualize import entropy_vs_layer, max_expert_util_distribution, utilisation_heatmap


def run_analysis(
    model_id: str,
    dataset: str,
    output: Path,
    *,
    max_samples: int = 10_000,
    batch_size: int = 4,
    max_length: int = 512,
    load_in_4bit: bool | None = None,
    device: str | None = None,
    figures_dir: Path | None = None,
    profile_note: str | None = None,
    run_seed: int = 0,
) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / f".routing_cache_{output.stem}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if load_in_4bit is None:
        load_in_4bit = default_load_in_4bit()

    model, tokenizer, device_name = load_causal_lm(
        model_id, load_in_4bit=load_in_4bit, device=device
    )
    input_device = model_input_device(model)
    stats = moe_bench.attach(model, flush_every=1000, output_dir=cache_dir)

    stream = get_stream(dataset, max_samples=max_samples)

    batch_texts: list[str] = []
    processed = 0

    for sample in tqdm(stream, total=max_samples, desc="Routing analysis"):
        batch_texts.append(sample["text"])
        if len(batch_texts) < batch_size:
            continue

        inputs = batch_tokenize(batch_texts, tokenizer, max_length=max_length)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        batch_texts = []
        processed += batch_size

    if batch_texts:
        inputs = batch_tokenize(batch_texts, tokenizer, max_length=max_length)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        processed += len(batch_texts)

    df = stats.to_dataframe()
    csv_path = output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    summary = stats.summary()
    per_layer = summary.get("per_layer", [])
    uncertainty = bootstrap_routing_cis(per_layer, seed=run_seed)

    note = profile_note
    if note is None:
        if dataset == "synthetic" and processed < 50:
            note = "smoke (synthetic; not comparable to domain runs)"
        elif dataset in ("math", "code", "general"):
            note = "domain-matched (same prompts as OLMoE phase-2)"

    report = {
        "model": model_id,
        "dataset": dataset,
        "domain": dataset if dataset in ("math", "code", "general") else None,
        "device": device_name,
        "samples_processed": processed,
        "batch_size": batch_size,
        "max_length": max_length,
        "load_in_4bit": load_in_4bit,
        "run_seed": run_seed,
        "profile_note": note,
        **summary,
        **uncertainty,
    }
    output.write_text(json.dumps(report, indent=2))

    if figures_dir is None:
        figures_dir = output.parent / "figures"
    figures_dir = Path(figures_dir)
    model_slug = model_id.split("/")[-1]
    utilisation_heatmap(
        df,
        output_path=figures_dir / f"utilisation_heatmap_{model_slug}.png",
        title=f"Expert Utilisation — {model_slug}",
    )
    entropy_vs_layer(
        df,
        output_path=figures_dir / f"entropy_vs_layer_{model_slug}.png",
    )
    max_expert_util_distribution(
        df,
        output_path=figures_dir / f"max_expert_util_{model_slug}.png",
    )

    moe_bench.detach(model)
    del model
    if device_name == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        import gc

        gc.collect()
        torch.mps.empty_cache()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moe-bench",
        description="MoE-Bench: routing instrumentation for HuggingFace MoE models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Collect routing stats on a model")
    analyze.add_argument("--model", required=True, help="HuggingFace model id")
    analyze.add_argument(
        "--dataset",
        choices=["c4", "pile", "combined", "synthetic", "general", "code", "math"],
        default="combined",
    )
    analyze.add_argument("--output", required=True, help="Output JSON path")
    analyze.add_argument("--max-samples", type=int, default=10_000)
    analyze.add_argument("--batch-size", type=int, default=4)
    analyze.add_argument("--max-length", type=int, default=512)
    analyze.add_argument(
        "--4bit",
        dest="use_4bit",
        action="store_true",
        help="Force 4-bit (CUDA only; not supported on Mac MPS)",
    )
    analyze.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    analyze.add_argument("--figures-dir", default=None)

    args = parser.parse_args(argv)

    if args.command == "analyze":
        device = None if args.device == "auto" else args.device
        load_in_4bit = True if args.use_4bit else None
        run_analysis(
            args.model,
            args.dataset,
            Path(args.output),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            load_in_4bit=load_in_4bit,
            device=device,
            figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
