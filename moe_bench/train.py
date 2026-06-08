"""LoRA fine-tuning with L_div / z-loss / switch baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_bench.data import get_stream
from moe_bench.hooks import discover_router_gates
from moe_bench.losses import (
    diversity_loss,
    router_z_loss,
    routing_entropy_torch,
    switch_load_balancing_loss,
)
from moe_bench.runner import run_routing_pass
from moe_bench.config import MoETrainingConfig


def build_batches(tokenizer, max_samples: int, max_length: int):
    for sample in get_stream("synthetic", max_samples=max_samples):
        enc = tokenizer(
            sample["text"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        yield {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"].clone(),
        }


def train_moe_aux(
    model_id: str,
    output: Path,
    *,
    aux_loss: str = "ldiv",
    lambda_div: float = 0.01,
    tau: float = 1.5,
    max_samples: int = 64,
    epochs: int = 1,
    lr: float = 2e-4,
    device: str = "mps",
) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
    model = model.to(device)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    config = MoETrainingConfig(lambda_div=lambda_div, tau=tau)

    capture: dict[int, list[torch.Tensor]] = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            capture.setdefault(layer_idx, []).append(logits)

        return hook

    handles = [spec.module.register_forward_hook(make_hook(spec.layer_idx)) for spec in discover_router_gates(model)]
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch in tqdm(list(build_batches(tokenizer, max_samples, 128)), desc=f"{aux_loss} e{epoch+1}", leave=False):
            capture.clear()
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            ce = model(**batch).loss

            layer_entropies, collapsed_mask = [], []
            z_total = torch.tensor(0.0, device=device)
            switch_total = torch.tensor(0.0, device=device)
            for layer_idx, chunks in capture.items():
                logits = torch.cat(chunks, dim=0)
                layer_entropies.append(routing_entropy_torch(logits))
                probs = torch.softmax(logits.float(), dim=-1)
                collapsed_mask.append(probs.max(dim=-1).values.mean().item() > config.collapse_threshold)
                z_total = z_total + router_z_loss(logits)
                switch_total = switch_total + switch_load_balancing_loss(probs, probs.shape[-1])

            if aux_loss == "ldiv":
                aux = diversity_loss(layer_entropies, tau=tau, collapsed_mask=collapsed_mask)
                total = ce + lambda_div * aux
            elif aux_loss == "zloss":
                total = ce + 0.001 * z_total / max(len(layer_entropies), 1)
            elif aux_loss == "switch":
                total = ce + 0.01 * switch_total / max(len(layer_entropies), 1)
            else:
                total = ce

            total.backward()
            opt.step()
            epoch_loss += total.item()
            n += 1
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(n, 1)})

    for h in handles:
        h.remove()

    adapter_dir = output.parent / "adapters" / f"{aux_loss}_l{lambda_div}_t{tau}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    model.eval()
    col = run_routing_pass(
        model,
        tokenizer,
        get_stream("synthetic", max_samples=min(32, max_samples)),
        max_samples=min(32, max_samples),
        batch_size=1,
        max_length=128,
    )
    df = col.to_dataframe()
    metrics = {
        "model": model_id,
        "aux_loss": aux_loss,
        "lambda_div": lambda_div,
        "tau": tau,
        "mean_entropy": float(df["mean_entropy"].mean()),
        "collapse_rate_40": float(df["collapsed"].mean()),
        "history": history,
        "adapter_dir": str(adapter_dir),
    }
    output.write_text(json.dumps(metrics, indent=2))
    return metrics
