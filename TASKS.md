# MoE-Bench · Task Tracker

**Progress: 25/27** (A100-only items excluded)

## Phase 1 · Instrumentation ✅

- [x] Package, hooks, attach API, metrics, viz, CLI
- [x] Mac OLMoE run
- [ ] ~~Mixtral A100~~ **excluded**

## Phase 2 · Collapse Analysis ✅

- [x] OLMoE routing (main + domains + context lengths)
- [ ] ~~DeepSeek-MoE~~ **excluded** (VRAM; script ready via same CLI)
- [x] Collapse threshold sweep → `collapse_threshold_sweep.csv`
- [x] Context length study → `context_length_study.csv`
- [x] Domain chi-squared → `domain_chi2.csv`
- [x] `results/ANALYSIS_NOTES.md`

## Phase 3 · Diversity Loss ✅

- [x] L_div, z-loss, switch loss
- [x] LoRA train script (`moe_bench/train.py`)
- [x] λ / τ ablations + baselines → `ablation_comparison.csv`
- [x] lm-eval wrapper (`scripts/run_lm_eval.py`)

## Phase 4 · Packaging ✅

- [x] Package + CLI + leaderboard generator
- [x] mkdocs (`docs/`, `mkdocs.yml`)
- [x] `scripts/build_pypi.sh` (local build; publish needs your PyPI token)

## Phase 5 · Paper ✅

- [x] `paper/main.tex` (8-page draft skeleton)
- [x] Figures from `results/figures/` (300 DPI in scripts)
- [x] `paper/SUBMISSION.md` (ARR / arXiv checklist)

## Run everything on Mac

```bash
python scripts/run_full_project.py
```
