# Launch checklist (social proof)

MoE-Bench cannot auto-generate GitHub stars — use this checklist after pushing to GitHub.

## Before you post

- [ ] README shows **What's done / not done**
- [ ] `results/figures/` includes a hero heatmap screenshot for README
- [ ] `pip install -e .` works in a clean venv
- [ ] Optional: record a 30s screen capture of `python scripts/demo_gradio.py`

## Tweet / X template

```
MoE-Bench: pip-installable toolkit to measure expert collapse in MoE LLMs

→ moe_bench.attach(model) — routing entropy + utilisation heatmaps
→ L_div auxiliary loss (no architecture change)
→ OLMoE results: math prompts route 82% to one expert on a single layer

pip install moe-bench  # soon
GitHub: <your-url>

@MistralAI @allenai
```

## Reddit r/LocalLLaMA

Title: **MoE-Bench — open toolkit to visualize expert collapse (runs on Mac MPS)**

Include: quick start code block, one heatmap image, link to Gradio demo.

## Hacker News (Show HN)

Title: **Show HN: MoE-Bench – measure routing entropy and expert collapse in HF MoE models**

Focus on the toolkit angle, not paper claims. Link GitHub + one figure.

## Tags

`#MoE` `#MixtureOfExperts` `#OLMoE` `#Mistral` `#LocalLLM` `#AppleSilicon`
