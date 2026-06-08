"""StatsCollector: ring buffer and disk flush for routing weights."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from moe_bench.metrics import compute_layer_metrics


class StatsCollector:
    """
    Accumulates per-layer router weights from forward hooks.

    Router weights are stored as CPU float32 numpy arrays with shape
    (batch * seq_len, num_experts) per forward pass per layer.
    """

    def __init__(
        self,
        flush_every: int = 1000,
        output_dir: str | Path | None = None,
        collapse_threshold: float = 0.40,
        top_k: int | None = None,
    ):
        self.flush_every = flush_every
        self.output_dir = Path(output_dir) if output_dir else None
        self.collapse_threshold = collapse_threshold
        self.top_k = top_k

        self._buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        self._sample_count = 0
        self._flush_count = 0
        self.layer_names: dict[int, str] = {}
        self.num_experts: dict[int, int] = {}

    def record(
        self,
        layer_idx: int,
        router_logits: torch.Tensor,
        layer_name: str = "",
    ) -> None:
        """
        Record one forward pass of router logits.

        Args:
            router_logits: (batch, seq, n_experts) or (batch*seq, n_experts)
        """
        with torch.no_grad():
            if router_logits.dim() == 3:
                b, s, e = router_logits.shape
                flat = router_logits.reshape(b * s, e)
            elif router_logits.dim() == 2:
                flat = router_logits
            else:
                raise ValueError(f"Unexpected router_logits shape: {router_logits.shape}")

            probs = torch.softmax(flat.float(), dim=-1).cpu().numpy()
            self._buffers[layer_idx].append(probs)
            self.layer_names[layer_idx] = layer_name or f"layer_{layer_idx}"
            self.num_experts[layer_idx] = probs.shape[-1]
            self._sample_count += probs.shape[0]

        if self.flush_every > 0 and self._sample_count >= self.flush_every * (self._flush_count + 1):
            self.flush()

    def flush(self) -> Path | None:
        """Write buffered weights to disk as .npz shards."""
        if self.output_dir is None or not self._buffers:
            self._flush_count += 1
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"routing_shard_{self._flush_count:05d}.npz"
        payload = {f"layer_{k}": np.concatenate(v, axis=0) for k, v in self._buffers.items() if v}
        np.savez_compressed(path, **payload)
        meta_path = self.output_dir / f"routing_shard_{self._flush_count:05d}_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "layer_names": {str(k): v for k, v in self.layer_names.items()},
                    "num_experts": {str(k): v for k, v in self.num_experts.items()},
                },
                indent=2,
            )
        )
        self._buffers.clear()
        self._flush_count += 1
        return path

    def get_router_weights(self) -> dict[int, list[np.ndarray]]:
        """Return accumulated buffers (not yet flushed)."""
        return dict(self._buffers)

    def to_dataframe(self) -> pd.DataFrame:
        """Compute per-layer routing metrics as a pandas DataFrame."""
        weights = self.get_router_weights()
        if self.output_dir:
            for shard in sorted(self.output_dir.glob("routing_shard_*.npz")):
                if "_meta" in shard.name:
                    continue
                data = np.load(shard)
                for key in data.files:
                    layer_idx = int(key.replace("layer_", ""))
                    weights.setdefault(layer_idx, []).append(data[key])

        return compute_layer_metrics(
            weights,
            collapse_threshold=self.collapse_threshold,
            top_k=self.top_k,
        )

    def summary(self) -> dict[str, Any]:
        """High-level summary for JSON reports."""
        df = self.to_dataframe()
        if df.empty:
            return {"error": "no data collected"}

        from moe_bench.metrics import collapse_rate

        layer_utils = {
            int(row["layer"]): np.array([row[c] for c in df.columns if c.startswith("expert_")])
            for _, row in df.iterrows()
        }
        rate, collapsed_layers = collapse_rate(layer_utils, self.collapse_threshold)

        return {
            "num_layers": len(df),
            "mean_entropy": float(df["mean_entropy"].mean()),
            "collapse_rate": rate,
            "collapsed_layers": collapsed_layers,
            "avg_load_imbalance": float(df["load_imbalance"].mean()),
            "per_layer": df.to_dict(orient="records"),
        }

    def detach(self) -> None:
        """No-op placeholder; hooks removed via MoEHookManager."""
        self.flush()
