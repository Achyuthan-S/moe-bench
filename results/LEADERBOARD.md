# MoE-Bench routing profile table

> **Deprecated:** use [`ROUTING_PROFILE_TABLE.md`](ROUTING_PROFILE_TABLE.md).


Per-model **routing** statistics (entropy, collapse, load imbalance)—not downstream accuracy.
Auto-generated from `results/routing_stats_*.json`.

† **smoke** = synthetic stream, often fewer tokens; compare **domain-matched** rows (math/code/general) across models.
CIs = layer-bootstrap (95%), equal weight per hooked router module; see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

| Model | Condition | n samples | Token positions | Mean entropy [95% CI] | Collapse rate [95% CI] | Load imb. | Note |
|-------|-----------|-----------|-----------------|------------------------|-------------------------|-----------|------|
| Qwen/Qwen1.5-MoE-A2.7B | math | 40 | 40744 | 3.300 [3.198, 3.409] | 0.00% [0.000, 0.000] | 6.35 | domain-matched (same prompts as OLMoE phase-2) |
| allenai/OLMoE-1B-7B-0924 | code | — | 18176 | 3.664 [3.558, 3.764] | 25.00% [0.062, 0.500] | 24.06 | domain-matched (same prompts as OLMoE phase-2) |
| allenai/OLMoE-1B-7B-0924 | general | — | 8960 | 3.666 [3.584, 3.751] | 0.00% [0.000, 0.000] | 10.40 | domain-matched (same prompts as OLMoE phase-2) |
| allenai/OLMoE-1B-7B-0924 | ? | — | 8960 | 3.666 [3.584, 3.751] | 0.00% [0.000, 0.000] | 10.40 | — |
| allenai/OLMoE-1B-7B-0924 | ? | — | 8960 | 3.666 [3.584, 3.751] | 0.00% [0.000, 0.000] | 10.40 | — |
| allenai/OLMoE-1B-7B-0924 | ? | — | 8960 | 3.666 [3.584, 3.751] | 0.00% [0.000, 0.000] | 10.40 | — |
| allenai/OLMoE-1B-7B-0924 | math | — | 13184 | 3.658 [3.567, 3.749] | 50.00% [0.250, 0.750] | 26.21 | domain-matched (same prompts as OLMoE phase-2) |
| allenai/OLMoE-1B-7B-0924 | synthetic | 64 | 27406 | 3.664 [3.580, 3.751] | 0.00% [0.000, 0.000] | 8.65 | smoke† |
| allenai/OLMoE-1B-7B-0924 | synthetic_extended | — | 40320 | 3.662 [3.571, 3.755] | 6.25% [0.000, 0.188] | 14.96 | — |
| jetmoe/jetmoe-8b | math | 40 | 65364 | 1.842 [1.812, 1.871] | 0.00% [0.000, 0.000] | 1.91 | domain-matched (same prompts as OLMoE phase-2); JetMoE dual-router (MoA+MoE) |
| jetmoe/jetmoe-8b | synthetic | 100 | 120087 | 1.868 [1.842, 1.891] | 0.00% [0.000, 0.000] | 1.79 | ; JetMoE dual-router (MoA+MoE) |

## Cross-model domain comparison (headline)

Compare **math** rows across models (same local prompt set, τ=0.40).

| Model | Collapse rate | Mean entropy | Peak layer util |
|-------|---------------|--------------|-----------------|
| Qwen/Qwen1.5-MoE-A2.7B | 0.0% | 3.300 | 15.8% |
| allenai/OLMoE-1B-7B-0924 | 50.0% | 3.658 | 82.5% |
| jetmoe/jetmoe-8b | 0.0% | 1.842 | 36.4% |
