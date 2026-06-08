# Contributing to MoE-Bench

MoE-Bench is an **open routing benchmark**: contributions are new model profiles, hook coverage, and reproducible analysis scripts—not leaderboard accuracy scores.

## Quick contribution paths

| Goal | Steps |
|------|--------|
| Add a model profile | Run `moe-bench analyze` (see below) → open PR with JSON + regenerated `results/ROUTING_PROFILE_TABLE.md` |
| Fix hook coverage | Extend [`moe_bench/hooks.py`](moe_bench/hooks.py) → add test in `tests/` |
| New metric | Extend [`moe_bench/metrics.py`](moe_bench/metrics.py) + document in README |

## The `attach()` contract

```python
import moe_bench

stats = moe_bench.attach(
    model,
    flush_every=1000,          # optional disk shards
    output_dir="cache/",       # optional
    collapse_threshold=0.40,   # top-1 util threshold
)
model(**inputs)                # forward pass(es)
df = stats.to_dataframe()      # per-router metrics
moe_bench.detach(model)        # remove hooks
```

**Requirements for a valid profile**

1. Hooks must fire on every MoE (and MoA, if applicable) router in the model.
2. Run at least one **domain-matched** condition (`math`, `code`, or `general`) with the same settings as our reference scripts (40 samples, `max_length=256`).
3. Include `profile_note` in the JSON (auto-set by CLI) so readers know smoke vs domain runs.

## Adding hooks for a new architecture

1. **Find router modules** after loading the model:

```python
for name, mod in model.named_modules():
    if any(x in name.lower() for x in ("moe", "gate", "router", "expert")):
        print(name, type(mod).__name__)
```

2. **Update** `_is_moe_router_module()` in [`moe_bench/hooks.py`](moe_bench/hooks.py):
   - Exclude dense FFN names (`gate_proj`, `gate_up_proj`).
   - Prefer suffixes like `.mlp.gate`, `.block_sparse_moe.gate`, `.mlp.router`, `.experts.router`.

3. **Update** `_extract_router_logits()` if the router returns a tuple (e.g. JetMoE, Qwen2-MoE). Pick the 2D tensor of shape `(num_tokens, num_experts)`.

4. **Layer indices**: use `_infer_layer_index()`; if a block has multiple routers (JetMoE MoA+MoE), offset MoA indices (we use `1000 + block_idx`).

5. **Verify**:

```bash
pytest tests/ -q
moe-bench analyze --model <hf_id> --dataset math --max-samples 8 \
  --output results/routing_stats_<slug>_math.json --device mps
```

## Standard profile commands

```bash
pip install -e ".[dev]"

# Domain-matched (comparable across models)
python scripts/run_domain_profiles.py --domains math general code

# Regenerate table
python scripts/enrich_routing_reports.py
python scripts/update_leaderboard.py
```

## What we merge

- New `results/routing_stats_*.json` with clear `profile_note` and bootstrap CIs
- Hook fixes with tests
- Docs only if they match measured artifacts in `results/`

## What we do not merge

- Downstream accuracy claims without lm-eval logs
- Profiles with no domain-matched row when comparing collapse across models
- Renaming the routing profile table back to “leaderboard”

## Code style

- Match existing `moe_bench` patterns; keep scripts thin wrappers around `run_analysis()`.
- Run `pytest tests/` before opening a PR.

## Questions

Open a GitHub issue with the model id, device (MPS/CUDA), and hook discovery output from step 1 above.
