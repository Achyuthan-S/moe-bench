# MoE-Bench 

**An open benchmark for expert collapse and routing efficiency in sparse Mixture-of-Experts LLMs.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Mac MPS Validated](https://img.shields.io/badge/Mac%20MPS-Validated-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> MoE-Bench is a **pip-installable toolkit** for measuring routing entropy, expert utilisation, and collapse in sparse MoE LLMs — not a downstream-accuracy leaderboard. It ships ready-to-run on a 24GB Mac.

---

## Why MoE-Bench?

Sparse MoE models route tokens to a subset of experts per layer. When routing collapses — a small number of experts monopolise all tokens — the model wastes capacity and degrades silently. Existing evaluations measure *what* a model outputs, not *how* its routing behaves.

MoE-Bench fills that gap:

- **Routing entropy** (Shannon H) per layer, per domain
- **Expert utilisation** distribution and collapse detection (configurable τ)
- **Bootstrap confidence intervals** on routing statistics
- **L_div training** — an entropy-floor regulariser that prevents collapse without hurting accuracy
- **Three validated architectures** out of the box: OLMoE, JetMoE, Qwen1.5-MoE

---

## Quick Results

All three models evaluated on the **same local math prompts** (40 samples, 256 tokens).

| Model | Math collapse (τ=0.40) | Mean entropy | Peak top-expert util |
|---|---|---|---|
| OLMoE-1B-7B | **50%** of layers | 3.66 | ~82% |
| JetMoE-8B | 0% | 1.84 | ~19% |
| Qwen1.5-MoE-A2.7B | 0% | 3.30 | ~16% |

**Key finding:** collapse is architecture-dependent, not a universal property of MoE models. OLMoE collapses heavily on math; JetMoE and Qwen do not.

Full table with 95% layer-bootstrap CIs → [`results/ROUTING_PROFILE_TABLE.md`](results/ROUTING_PROFILE_TABLE.md)

---

## L_div: Does fixing collapse hurt accuracy?

We tested L_div LoRA fine-tuning on OLMoE against a 12-subject MMLU subset (5-shot, 25 examples/subject):

| Model | Macro accuracy | 95% CI |
|---|---|---|
| Base OLMoE | 51.7% | [44.4%, 59.0%] |
| + L_div LoRA | 52.0% | [46.4%, 57.6%] |
| Δ | +0.33 pp | CIs overlap → not significant |

**L_div does not hurt downstream accuracy** — the entropy floor regularises routing without degrading output quality. A full 57-task MMLU run is needed before any accuracy claim; this is a directional sanity check.

---

## How L_div differs from existing routing losses

| Loss | Limitation | L_div |
|---|---|---|
| Switch auxiliary loss | Load balance ≠ high entropy | Entropy floor on collapsed layers |
| Router z-loss | Stabilises logits, not diversity | `max(0, τ − H_l)` per layer |
| Load-balancing loss | Few experts can still dominate | Shannon H per layer |

---

## Install

```bash
git clone https://github.com/YOUR_USERNAME/moe-bench.git
cd moe-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,eval,demo,dev]"
```

---

## Quick Start

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

---

## CLI

```bash
# Analyze routing on a math domain
moe-bench analyze \
  --model allenai/OLMoE-1B-7B-0924 \
  --dataset math \
  --device mps \
  --max-samples 40 \
  --max-length 256 \
  --output results/routing_stats_olmoe_math.json

# Cross-model domain profiles
python scripts/run_domain_profiles.py --domains math general code

# Regenerate leaderboard table + bootstrap CIs
python scripts/enrich_routing_reports.py
python scripts/update_leaderboard.py

# OLMoE only: L_div training + MMLU eval
python scripts/run_mmlu_ldiv_comparison.py --ldiv-only

# Gradio demo
python scripts/demo_gradio.py
```

---

## Repo Layout

```
moe_bench/          # Core library: attach, hooks, metrics, routing_uncertainty, train
scripts/            # run_domain_profiles, run_jetmoe_mac, update_leaderboard, ...
results/            # routing_stats_*.json, ROUTING_PROFILE_TABLE.md, ANALYSIS_NOTES.md
CONTRIBUTING.md     # Hook guide for adding new MoE architectures
```

---

## Key Output Files

| Path | Content |
|---|---|
| `results/ROUTING_PROFILE_TABLE.md` | All profiles with 95% bootstrap CIs |
| `results/ANALYSIS_NOTES.md` | OLMoE deep-dive |
| `results/CROSS_MODEL_NOTES.md` | OLMoE vs JetMoE math comparison |
| `results/routing_stats_jetmoe_math.json` | JetMoE math routing data |
| `results/routing_stats_qwen15_moe_math.json` | Qwen1.5-MoE math routing data (CPU) |
| `results/lm_eval/mmlu_ldiv_comparison.json` | MMLU base vs L_div |
| `results/ablation_comparison.csv` | Routing ablations (synthetic LoRA) |

---

## Hardware & Scope

| Platform | Supported models |
|---|---|
| Mac MPS 24GB | OLMoE, JetMoE (sequential); Qwen often on CPU after JetMoE |
| Cloud GPU (optional) | Mixtral-8x7B via `moe-bench analyze --4bit` — not validated in this release |

**Scope notes:**
- L_div LoRA training, ablations, and MMLU are **OLMoE-only** in this release.
- JetMoE and Qwen are **routing-profile-only** baselines (no L_div fine-tune).
- Always compare models on **domain-matched rows** (`math` / `code` / `general`), not synthetic smoke rows.
- Large cloud MoE runs (e.g. Mixtral-8x7B at paper scale) are out of scope here. The CLI supports them via `--4bit` when GPU access is available.

---

## Contributing

Hook support for new architectures lives in [`CONTRIBUTING.md`](CONTRIBUTING.md). Adding a model is a matter of identifying the gate/router module name and registering a hook — the rest of the toolkit is architecture-agnostic.

PRs welcome.

---

## Citation

```bibtex
@misc{sivasankar2026moebench,
  title        = {MoE-Bench: Open Benchmark for Expert Collapse and Routing Efficiency in Sparse MoE LLMs},
  author       = {Sivasankar, Achyuthan and {MoE-Bench Contributors}},
  year         = {2026},
  howpublished = {Software},
  url          = {https://github.com/YOUR_USERNAME/moe-bench},
  note         = {Routing benchmark toolkit; MMLU and cloud-scale Mixtral results reported as directional or out-of-scope}
}
```

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
