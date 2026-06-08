# API Reference

## `moe_bench.attach(model, ...)`

Registers forward hooks on all discovered router `gate` / `router` Linear layers.

Returns a `StatsCollector`. After forward passes:

```python
df = stats.to_dataframe()
summary = stats.summary()
```

## Metrics

| Function | Description |
|----------|-------------|
| `routing_entropy(probs)` | H = −Σ p log p per token |
| `expert_utilisation(probs)` | Fraction of tokens per expert |
| `load_imbalance_coefficient(util)` | max / mean utilisation |
| `collapse_rate(layer_utils, threshold=0.4)` | Fraction of collapsed layers |

## Visualization

```python
from moe_bench.visualize import utilisation_heatmap
utilisation_heatmap(df, output_path="heatmap.png")
```
