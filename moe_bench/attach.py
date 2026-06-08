"""Public API: moe_bench.attach(model) -> StatsCollector."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from moe_bench.collector import StatsCollector
from moe_bench.hooks import MoEHookManager, register_hooks

# Keep managers alive so hooks are not garbage-collected.
_ACTIVE_MANAGERS: dict[int, MoEHookManager] = {}


def attach(
    model: nn.Module,
    *,
    flush_every: int = 1000,
    output_dir: str | Path | None = None,
    collapse_threshold: float = 0.40,
    top_k: int | None = None,
) -> StatsCollector:
    """
    Register forward hooks on all router layers and return a StatsCollector.

    Usage:
        import moe_bench
        stats = moe_bench.attach(model)
        model(input_ids)
        df = stats.to_dataframe()

    Args:
        model: HuggingFace MoE causal LM (Mixtral, OLMoE, DeepSeek-MoE, ...)
        flush_every: flush ring buffer to disk every N token positions
        output_dir: if set, write .npz shards under this directory
        collapse_threshold: top-1 expert fraction defining collapse
        top_k: expert utilisation counting mode (None = argmax)

    Returns:
        StatsCollector attached to the model's router gates.
    """
    collector = StatsCollector(
        flush_every=flush_every,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        top_k=top_k,
    )
    manager = register_hooks(model, collector)
    _ACTIVE_MANAGERS[id(model)] = manager
    collector._manager = manager  # type: ignore[attr-defined]
    return collector


def detach(model: nn.Module) -> None:
    """Remove hooks for a model previously passed to attach()."""
    manager = _ACTIVE_MANAGERS.pop(id(model), None)
    if manager is not None:
        if hasattr(manager, "remove"):
            manager.remove()
        else:
            manager.__exit__(None, None, None)
        manager.collector.flush()
