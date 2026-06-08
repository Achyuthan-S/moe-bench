"""Forward hooks on MoE router/gate layers across HuggingFace architectures."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from moe_bench.collector import StatsCollector


# Module name patterns for router gates across HF MoE models.
GATE_NAMES = ("gate", "router", "wg", "gating")
MOE_BLOCK_HINTS = (
    "moe",
    "sparse",
    "experts",
    "MixtralSparseMoeBlock",
    "OLMoE",
    "DeepseekMoE",
    "MoEGate",
    "JetMoe",
    "jetmoe",
    "Qwen2Moe",
    "qwen2_moe",
)


@dataclass
class RouterHookSpec:
    layer_idx: int
    module: nn.Module
    qualified_name: str


@dataclass
class MoEHookManager:
    """Registers and removes forward hooks on discovered router gates."""

    collector: StatsCollector
    specs: list[RouterHookSpec] = field(default_factory=list)
    _handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def register(self, model: nn.Module) -> list[RouterHookSpec]:
        self.specs = discover_router_gates(model)
        for spec in self.specs:
            handle = spec.module.register_forward_hook(self._make_hook(spec.layer_idx, spec.qualified_name))
            self._handles.append(handle)
        return self.specs

    def _make_hook(self, layer_idx: int, layer_name: str):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor):
            logits = _extract_router_logits(output)
            self.collector.record(layer_idx, logits, layer_name=layer_name)

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> MoEHookManager:
        return self

    def __exit__(self, *args) -> None:
        self.remove()


def _extract_router_logits(output: torch.Tensor | tuple) -> torch.Tensor:
    """Normalize router outputs across HF MoE architectures (OLMoE, Mixtral, JetMoE, Qwen2-MoE, …)."""
    if isinstance(output, tuple):
        tensors_2d = [t for t in output if torch.is_tensor(t) and t.dim() >= 2]
        if not tensors_2d:
            raise ValueError(f"Could not find router logits in tuple output (len={len(output)})")
        # Qwen2-MoE returns (logits [*, E], top-k scores [*, k], indices [*, k]) — never use indices/scores.
        return max(tensors_2d, key=lambda t: t.shape[-1])
    return output


def _is_moe_router_module(name: str, leaf: str) -> bool:
    """True for MoE router gates (Mixtral Linear, OLMoE TopKRouter at .mlp.gate, etc.)."""
    if leaf not in GATE_NAMES:
        return False
    # Dense FFN weights named gate_proj / gate_up_proj are not routers.
    if "gate_proj" in name or "gate_up_proj" in name:
        return False
    # OLMoE: model.layers.N.mlp.gate ; Mixtral: ...block_sparse_moe.gate
    # JetMoE: model.layers.N.mlp.router (MoE) and ...self_attention.experts.router (MoA)
    if name.endswith((".mlp.gate", ".block_sparse_moe.gate", ".mlp.router", ".experts.router")):
        return True
    name_lower = name.lower()
    return any(hint.lower() in name_lower for hint in MOE_BLOCK_HINTS)


def discover_router_gates(model: nn.Module) -> list[RouterHookSpec]:
    """
    Find MoE router modules in HuggingFace models.

    Supports nn.Linear gates (Mixtral) and custom routers (OLMoE OlmoeTopKRouter).
    """
    specs: list[RouterHookSpec] = []
    seen: set[str] = set()

    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if not _is_moe_router_module(name, leaf):
            continue
        if name in seen:
            continue
        seen.add(name)

        layer_idx = _infer_layer_index(name, len(specs))
        specs.append(RouterHookSpec(layer_idx=layer_idx, module=module, qualified_name=name))

    specs.sort(key=lambda s: (s.layer_idx, s.qualified_name))
    return specs


def _infer_layer_index(qualified_name: str, fallback: int) -> int:
    """Parse layer index from names like model.layers.12.block_sparse_moe.gate."""
    block_idx: int | None = None
    parts = qualified_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                block_idx = int(parts[i + 1])
                break
            except ValueError:
                pass
        if part in ("layer", "h") and i + 1 < len(parts):
            try:
                block_idx = int(parts[i + 1])
                break
            except ValueError:
                pass
    if block_idx is None:
        return fallback
    # JetMoE has MoA + MoE routers per block — offset MoA by +1000 to avoid buffer collisions
    if ".experts.router" in qualified_name or ".self_attention.experts" in qualified_name:
        return 1000 + block_idx
    return block_idx


def register_hooks(model: nn.Module, collector: StatsCollector) -> MoEHookManager:
    """Register hooks and return manager (call manager.remove() when done)."""
    manager = MoEHookManager(collector=collector)
    specs = manager.register(model)
    if not specs:
        raise RuntimeError(
            "No router gate modules found. Ensure this is a HuggingFace MoE model "
            "(Mixtral, OLMoE, DeepSeek-MoE, etc.)."
        )
    return manager
