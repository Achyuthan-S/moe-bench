"""Device and dtype selection for Mac (MPS), CUDA, and CPU."""

from __future__ import annotations

import platform
import sys

import torch


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_load_in_4bit() -> bool:
    """bitsandbytes 4-bit is unreliable on Mac; use float16 on MPS instead."""
    if is_apple_silicon():
        return False
    return torch.cuda.is_available()


def model_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_causal_lm(model_id: str, *, load_in_4bit: bool | None = None, device: str | None = None):
    """
    Load a HuggingFace MoE causal LM with platform-appropriate settings.

    MacBook (24GB): OLMoE-1B-7B in float16 on MPS (~14GB weights).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if load_in_4bit is None:
        load_in_4bit = default_load_in_4bit()
    if device is None:
        device = default_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        model = model.to("mps")
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    else:
        # float16 on Mac CPU (~half RAM vs float32); avoids OOM on Qwen1.5-MoE-A2.7B
        cpu_dtype = torch.float16 if is_apple_silicon() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=cpu_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = model.to("cpu")

    return model, tokenizer, device
