# MoE-Bench

**Open benchmark for expert collapse and routing efficiency in sparse MoE LLMs**

MoE-Bench is a **pip-installable toolkit** for routing entropy, expert utilisation, and collapse—**not** a downstream-accuracy leaderboard. It ships with three Mac-validated MoE profiles (OLMoE, JetMoE, Qwen1.5-MoE), domain-matched comparisons, bootstrap routing uncertainty, optional \(L_{\mathrm{div}}\) training on OLMoE, MMLU sanity checks, and a Gradio demo.

**Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) · **Paper:** [`paper/main.tex`](paper/main.tex)

## At a glance

| | OLMoE-1B-7B | JetMoE-8B | Qwen1.5-MoE-A2.7B |
|--|-------------|-----------|-------------------|
| Role | Full analysis + \(L_{\mathrm{div}}\) training | Routing profile (dual router) | Routing profile (3rd architecture) |
| **Math domain** collapse (τ=0.40) | **50%** | **0%** | **0%** |
| Math mean entropy | 3.66 | 1.84 | **3.30**‡ |
| Peak top-expert util (any layer) | ~82% | ~19% | ~16%‡ |
| Hooks | `.mlp.gate` | `.mlp.router` + `.experts.router` | `.mlp.gate` |
| Run device (math) | MPS | MPS | CPU (float16) |

All three use the **same local math prompts** (40 samples, 256 tokens). OLMoE shows strong layer-collapse; JetMoE and Qwen do not at τ=0.40—**collapse is architecture-dependent**, not a universal law.

‡ Earlier Qwen entropy **0.08** was a **hook bug** (we softmaxed top-k expert *indices*, not 60-way logits). Fixed in `hooks.py`; corrected re-run: **3.30** H, comparable to OLMoE.

Full table with **95% layer-bootstrap CIs**: [`results/ROUTING_PROFILE_TABLE.md`](results/ROUTING_PROFILE_TABLE.md)

## Project status

| Phase | Status |
|-------|--------|
| Toolkit + hooks (OLMoE, JetMoE, Qwen2-MoE) | ✅ |
| OLMoE phase-2 (length, domains, χ²) | ✅ |
| Cross-model domain profiles | ✅ OLMoE + JetMoE + Qwen math |
| \(L_{\mathrm{div}}\) + MMLU (OLMoE only) | ✅ |
| Paper (toolkit-first) | ✅ |

This release targets **24GB Mac** validation. Large cloud MoE runs (e.g. Mixtral-8x7B at paper scale) are **out of scope here**—not a deferred roadmap item. The CLI already supports them via `--4bit` when you have GPU access ([`scripts/run_mixtral_mac.py`](scripts/run_mixtral_mac.py) documents the Mac OOM case).

### Scope (read before citing)

- **\(L_{\mathrm{div}}\)** LoRA training, ablations, and MMLU are **OLMoE-only**.
- **JetMoE** and **Qwen** are **routing-profile-only** baselines (no \(L_{\mathrm{div}}\) fine-tune).
- Compare models on **domain-matched** rows (`math` / `code` / `general`), not synthetic smoke rows.

## MMLU subset (OLMoE, directional)

12 subjects · 5-shot · 25 examples/subject · merged LoRA weights.

| Model | Macro accuracy | 95% CI |
|-------|----------------|--------|
| Base OLMoE | **51.7%** | [44.4%, 59.0%] |
| L_div LoRA | **52.0%** | [46.4%, 57.6%] |
| Δ | **+0.33 pp** | CIs overlap → **not significant** |

We report this to show that \(L_{\mathrm{div}}\) LoRA **does not hurt** downstream accuracy on a cheap sanity check—the point is *no degradation*, not a benchmark win. A full 57-task MMLU run would be needed before any accuracy claim.

Details: [`results/lm_eval/mmlu_ldiv_comparison.json`](results/lm_eval/mmlu_ldiv_comparison.json)

## Related work: how \(L_{\mathrm{div}}\) differs

| Loss | Limitation | \(L_{\mathrm{div}}\) |
|------|------------|----------------------|
| Switch aux | Load balance ≠ high entropy | Entropy floor on collapsed layers |
| Router z-loss | Stabilises logits, not diversity | \(\max(0, \tau - H_l)\) per layer |
| Load-balancing | Few experts can still dominate | Shannon \(H\) per layer |

Routing ablations: [`ablation_comparison.csv`](results/ablation_comparison.csv) (smoke-scale synthetic LoRA).

## Reproduce on Mac

```bash
cd /path/to/MOE
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,eval,demo,dev]"
```

```bash
# Cross-model domain profiles (OLMoE phase-2 matched settings)
python scripts/run_domain_profiles.py --domains math general code

# Regenerate table + bootstrap CIs on all JSONs
python scripts/enrich_routing_reports.py
python scripts/update_leaderboard.py

# OLMoE-only: MMLU, Gradio, full pipeline
python scripts/run_mmlu_ldiv_comparison.py --ldiv-only
python scripts/demo_gradio.py
python scripts/run_full_project.py
```

**Note:** Qwen1.5-MoE math was run on **CPU in float16** (~35 min) to avoid MPS OOM after JetMoE; `moe_bench.device` uses float16 on Apple Silicon CPU automatically.

## Install & quick start

```bash
pip install -e ".[train,eval,demo,dev]"
```

```python
import moe_bench
from moe_bench.device import load_causal_lm, model_input_device

model, tokenizer, _ = load_causal_lm("allenai/OLMoE-1B-7B-0924", device="mps")
stats = moe_bench.attach(model)
inputs = tokenizer("Hello", return_tensors="pt").to(model_input_device(model))
model(**inputs)
print(stats.to_dataframe()[["layer", "mean_entropy", "collapsed"]])
moe_bench.detach(model)
```

## CLI

```bash
moe-bench analyze --model allenai/OLMoE-1B-7B-0924 --dataset math \
  --device mps --max-samples 40 --max-length 256 \
  --output results/routing_stats_olmoe_math.json
```

## Key outputs

| Path | Content |
|------|---------|
| [`ROUTING_PROFILE_TABLE.md`](results/ROUTING_PROFILE_TABLE.md) | All profiles + CIs + notes |
| [`routing_stats_jetmoe_math.json`](results/routing_stats_jetmoe_math.json) | JetMoE math (cross-model) |
| [`routing_stats_qwen15_moe_math.json`](results/routing_stats_qwen15_moe_math.json) | Qwen1.5-MoE math (CPU) |
| [`ANALYSIS_NOTES.md`](results/ANALYSIS_NOTES.md) | OLMoE deep dive |
| [`CROSS_MODEL_NOTES.md`](results/CROSS_MODEL_NOTES.md) | OLMoE vs JetMoE math comparison |
| [`mmlu_ldiv_comparison.json`](results/lm_eval/mmlu_ldiv_comparison.json) | MMLU base vs L_div |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Hook guide for new models |

## Layout

```
moe_bench/       attach, hooks, metrics, routing_uncertainty, train
scripts/         run_domain_profiles, run_jetmoe_mac, update_leaderboard
results/         routing_stats_*.json, ROUTING_PROFILE_TABLE.md
paper/           main.tex
CONTRIBUTING.md
```

## Hardware

| Platform | Models |
|----------|--------|
| Mac MPS 24GB | OLMoE, JetMoE (sequential); Qwen often CPU after JetMoE |
| Cloud GPU (optional) | Mixtral-8x7B via `moe-bench analyze --4bit` — not validated in this release |

## Citation

If you use MoE-Bench, cite the software release. Add your repository URL to `url` after you publish to GitHub.

```bibtex
@misc{sivasankar2026moebench,
  title        = {MoE-Bench: Open Benchmark for Expert Collapse and Routing Efficiency in Sparse MoE LLMs},
  author       = {Sivasankar, Achyuthan and {MoE-Bench Contributors}},
  year         = {2026},
  howpublished = {Software},
  note         = {Routing benchmark toolkit; MMLU and cloud-scale Mixtral results reported as directional or out-of-scope}
}
```

Apache-2.0
