# MoE-Bench Analysis Notes (Section 3)

**Model:** allenai/OLMoE-1B-7B-0924

## Collapse threshold
- Recommended τ (top-1 utilisation): **0.40**
- At 0.40: collapse rate from main run varies by dataset; see `collapse_threshold_sweep.csv`

## Layer depth
- Early layers (0–3) mean entropy: **3.822**
- Late layers (12–15) mean entropy: **3.395**
- Worst layer **5**: max expert util **42.5%**

## Context length
- L=128: entropy=3.666, collapse_rate=0.0%
- L=512: entropy=3.666, collapse_rate=0.0%
- L=1024: entropy=3.666, collapse_rate=0.0%

## Domain routing (chi-squared)
- Significant layers (p<0.05): **16** / 16
- **general**: peak layer 7 → expert util 31.4%
- **code**: peak layer 6 → expert util 57.7%
- **math**: peak layer 5 → expert util 82.5%
- Example: layer 0 chi2=2266.85, p=0.0000

## Figures
- `figures/heatmap_*.png`, `entropy_*.png`, `maxutil_*.png`
