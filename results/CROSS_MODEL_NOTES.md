# Cross-model routing notes (domain-matched)

Same local prompt sets as OLMoE phase-2 (`moe_bench.data.DOMAIN_SAMPLES`). Settings: **40 samples**, **max_length=256**, **batch_size=1**, collapse threshold **τ=0.40**.

## Math domain (headline comparison)

| Model | Collapse rate | Mean entropy | Peak top-expert util | Notes |
|-------|---------------|--------------|----------------------|-------|
| OLMoE-1B-7B | 50% (8/16 layers) | 3.658 | ~82.5% (layer 5) | Single `.mlp.gate` per layer |
| JetMoE-8B | 0% (0/48 router hooks) | 1.842 | ~19% | Dual router (MoA+MoE) per block |
| Qwen1.5-MoE-A2.7B | 0% (0/24 layers) | 3.300 | ~16% | 60 experts; CPU float16 (corrected hooks) |

**Interpretation:** OLMoE exhibits strong **math-specific collapse** on this benchmark; JetMoE and Qwen on the **same math strings** do not hit the τ=0.40 collapse criterion on any hooked router. Qwen’s full-router entropy (~3.3) is comparable to OLMoE (~3.7) once hooks read 60-way logits (not top-k indices). An earlier reported Qwen entropy of 0.08 was a measurement artifact—see README footnote.

This supports MoE-Bench’s goal—measuring routing behaviour that varies by architecture—not ranking model quality.

## How to extend

```bash
python scripts/run_domain_profiles.py --domains math general code
python scripts/update_leaderboard.py
```

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for adding a fourth model.
