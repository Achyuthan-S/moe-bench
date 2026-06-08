# Contributing a Model

1. Run analysis on an A100 (or GPU with enough VRAM for 4-bit load):

```bash
moe-bench analyze \
  --model <huggingface_model_id> \
  --dataset combined \
  --output results/routing_stats_<name>.json
```

2. Regenerate the routing profile table:

```bash
python scripts/update_leaderboard.py
```

3. Open a PR adding your JSON under `results/` and the updated `ROUTING_PROFILE_TABLE.md`.

Supported architectures: any HuggingFace MoE with `gate` / `router` modules (Mixtral, OLMoE, DeepSeek-MoE, Qwen-MoE, …).
