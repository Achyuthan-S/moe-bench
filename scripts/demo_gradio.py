#!/usr/bin/env python3
"""Live Gradio demo: paste text → expert utilisation heatmap (Mac / MPS)."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def analyze_prompt(model, tokenizer, text: str, max_length: int = 256):
    import moe_bench
    from moe_bench.device import model_input_device
    from moe_bench.hooks import register_hooks
    from moe_bench.collector import StatsCollector

    collector = StatsCollector(flush_every=0, output_dir=None)
    manager = register_hooks(model, collector)
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(model_input_device(model)) for k, v in enc.items()}
        import torch

        with torch.no_grad():
            model(**enc)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    finally:
        manager.remove()

    df = collector.to_dataframe()
    summary = collector.summary()
    return df, summary


def df_to_heatmap_bytes(df: pd.DataFrame) -> bytes:
    expert_cols = sorted([c for c in df.columns if c.startswith("expert_")], key=lambda x: int(x.split("_")[1]))
    matrix = df.sort_values("layer")[expert_cols].values
    layers = df.sort_values("layer")["layer"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Token fraction")
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(layers)), [str(int(l)) for l in layers])
    ax.set_title("Expert utilisation")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_ui(model_id: str, device: str):
    import gradio as gr
    from moe_bench.device import load_causal_lm

    print(f"Loading {model_id} on {device}...")
    model, tokenizer, _ = load_causal_lm(model_id, device=device)

    # Warm up MPS (first forward can take 1–2 min on Mac; do it at startup)
    print("Warming up GPU (first forward)...")
    analyze_prompt(model, tokenizer, "Warmup run for MoE routing.", max_length=32)
    print("Ready — submit a prompt in the browser.")

    def run(text: str):
        if not text.strip():
            return None, "Enter some text."
        print(f"Processing ({len(text)} chars)...")
        df, summary = analyze_prompt(model, tokenizer, text, max_length=128)
        print("Done — rendering heatmap.")
        from PIL import Image

        img = Image.open(io.BytesIO(df_to_heatmap_bytes(df)))
        md = (
            f"**Mean entropy:** {summary.get('mean_entropy', 0):.3f}  \n"
            f"**Collapse rate @ 40%:** {summary.get('collapse_rate', 0):.1%}  \n"
            f"**Collapsed layers:** {summary.get('collapsed_layers', [])}"
        )
        return img, md

    return gr.Interface(
        fn=run,
        inputs=gr.Textbox(lines=6, label="Prompt", placeholder="Paste text or code..."),
        outputs=[
            gr.Image(label="Routing heatmap"),
            gr.Markdown(label="Stats"),
        ],
        title="MoE-Bench Live Routing Demo",
        description=f"Model: `{model_id}` · hooks on router gates · runs locally on {device}",
        examples=[
            ["The mixture of experts model routes each token to specialized feed-forward networks."],
            ["def solve(x): return x * x  # python"],
            ["Prove that the sum of the first n integers is n(n+1)/2."],
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(args.model, args.device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
