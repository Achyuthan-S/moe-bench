"""MoE-Bench: routing instrumentation and collapse analysis for sparse MoE LLMs."""

from moe_bench.attach import attach, detach
from moe_bench.collector import StatsCollector
from moe_bench.metrics import (
    collapse_rate,
    compute_layer_metrics,
    load_imbalance_coefficient,
    routing_entropy,
)

__version__ = "0.1.0"

__all__ = [
    "attach",
    "detach",
    "StatsCollector",
    "routing_entropy",
    "load_imbalance_coefficient",
    "collapse_rate",
    "compute_layer_metrics",
]
