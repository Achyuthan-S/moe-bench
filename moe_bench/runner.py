"""Shared routing analysis runner (single model load, multiple passes)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import moe_bench
from moe_bench.collector import StatsCollector
from moe_bench.data import batch_tokenize, get_stream
from moe_bench.device import load_causal_lm, model_input_device
from moe_bench.visualize import entropy_vs_layer, max_expert_util_distribution, utilisation_heatmap


def run_routing_pass(
    model,
    tokenizer,
    stream: Iterator[dict[str, Any]],
    *,
    max_samples: int,
    batch_size: int = 1,
    max_length: int = 128,
    flush_every: int = 500,
    output_dir: Path | None = None,
) -> StatsCollector:
    input_device = model_input_device(model)
    collector = StatsCollector(flush_every=flush_every, output_dir=output_dir)

    from moe_bench.hooks import register_hooks

    manager = register_hooks(model, collector)
    try:
        batch_texts: list[str] = []
        for sample in tqdm(stream, total=max_samples, desc="routing", leave=False):
            batch_texts.append(sample["text"])
            if len(batch_texts) < batch_size:
                continue
            inputs = batch_tokenize(batch_texts, tokenizer, max_length=max_length)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            batch_texts = []

        if batch_texts:
            inputs = batch_tokenize(batch_texts, tokenizer, max_length=max_length)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
    finally:
        manager.remove()

    return collector


def save_analysis_run(
    collector: StatsCollector,
    output: Path,
    meta: dict[str, Any],
    *,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = collector.to_dataframe()
    df.to_csv(output.with_suffix(".csv"), index=False)

    report = {**meta, **collector.summary()}
    output.write_text(json.dumps(report, indent=2))

    if figures_dir:
        figures_dir = Path(figures_dir)
        slug = meta.get("tag", output.stem)
        utilisation_heatmap(df, output_path=figures_dir / f"heatmap_{slug}.png")
        entropy_vs_layer(df, output_path=figures_dir / f"entropy_{slug}.png")
        max_expert_util_distribution(df, output_path=figures_dir / f"maxutil_{slug}.png")

    return report
