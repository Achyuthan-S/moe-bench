"""Streaming data loaders for C4, The Pile, domains, and local synthetic text."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

DOMAIN_SAMPLES: dict[str, list[str]] = {
    "general": [
        "The European Union promotes economic integration and regulatory cooperation among member states.",
        "Climate scientists study long-term trends in global temperature and precipitation patterns.",
        "Public libraries provide free access to books, internet, and community education programs.",
        "Urban planners design transportation networks to reduce congestion and emissions in cities.",
        "Journalists verify sources before publishing investigative reports on government policy.",
    ] * 8,
    "code": [
        "def route_tokens(hidden, gate):\n    logits = gate(hidden)\n    return torch.topk(logits, k=2, dim=-1)",
        "import torch.nn as nn\nclass MoELayer(nn.Module):\n    def forward(self, x): return self.experts(self.router(x))",
        "for layer in model.layers:\n    h = layer.attn(h)\n    h = layer.moe(h)",
        "git commit -m 'fix router aux loss coefficient'\npush origin main",
        "SELECT user_id, COUNT(*) FROM events GROUP BY user_id HAVING COUNT(*) > 100;",
    ] * 8,
    "math": [
        "Prove that the sum of the first n integers equals n(n+1)/2 by induction.",
        "If f(x) = x^3 - 3x + 1, find all critical points and classify each using the second derivative test.",
        "Compute the integral of e^{-x^2} from negative infinity to positive infinity.",
        "A fair die is rolled twice; what is the probability the sum is at least 10?",
        "Solve the linear system: 2x + y = 5, x - y = 1.",
    ] * 8,
}


def is_english(text: str, min_ascii_ratio: float = 0.85) -> bool:
    if not text or len(text) < 32:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= min_ascii_ratio


def stream_synthetic(max_samples: int = 200) -> Iterator[dict[str, Any]]:
    """Mixed local snippets — no download."""
    all_texts = DOMAIN_SAMPLES["general"] + DOMAIN_SAMPLES["code"] + DOMAIN_SAMPLES["math"]
    for i, text in enumerate(all_texts):
        if i >= max_samples:
            break
        yield {"text": text, "source": "synthetic"}


def stream_domain(domain: str, max_samples: int = 40) -> Iterator[dict[str, Any]]:
    if domain not in DOMAIN_SAMPLES:
        raise ValueError(f"Unknown domain {domain}; choose from {list(DOMAIN_SAMPLES)}")
    for i, text in enumerate(DOMAIN_SAMPLES[domain]):
        if i >= max_samples:
            break
        yield {"text": text, "source": domain}


def stream_c4(
    split: str = "validation",
    max_samples: int = 10_000,
    language: str = "en",
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", language, split=split, streaming=True, trust_remote_code=True)
    count = 0
    for row in ds:
        text = row.get("text", "")
        if not is_english(text):
            continue
        yield {"text": text, "source": "c4"}
        count += 1
        if count >= max_samples:
            break


def stream_pile(split: str = "test", max_samples: int = 10_000) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("monology/pile-uncopyrighted", split=split, streaming=True, trust_remote_code=True)
    count = 0
    for row in ds:
        text = row.get("text", "")
        if not is_english(text):
            continue
        yield {"text": text, "source": "pile"}
        count += 1
        if count >= max_samples:
            break


def stream_combined(*, c4_samples: int = 10_000, pile_samples: int = 10_000) -> Iterator[dict[str, Any]]:
    yield from stream_c4(max_samples=c4_samples)
    yield from stream_pile(max_samples=pile_samples)


def get_stream(dataset: str, max_samples: int) -> Iterator[dict[str, Any]]:
    if dataset == "synthetic":
        return stream_synthetic(max_samples=max_samples)
    if dataset in DOMAIN_SAMPLES:
        return stream_domain(dataset, max_samples=max_samples)
    if dataset == "c4":
        return stream_c4(max_samples=max_samples)
    if dataset == "pile":
        return stream_pile(max_samples=max_samples)
    if dataset == "combined":
        return stream_combined(c4_samples=max_samples // 2, pile_samples=max_samples // 2)
    raise ValueError(f"Unknown dataset: {dataset}")


def batch_tokenize(texts: list[str], tokenizer: Any, max_length: int = 512) -> dict[str, Any]:
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
