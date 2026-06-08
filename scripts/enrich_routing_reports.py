#!/usr/bin/env python3
"""Add bootstrap CIs and profile notes to existing routing_stats_*.json files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT))
from moe_bench.routing_uncertainty import bootstrap_routing_cis


def enrich(path: Path) -> None:
    data = json.loads(path.read_text())
    per_layer = data.get("per_layer", [])
    if not per_layer:
        return
    data.update(bootstrap_routing_cis(per_layer, seed=int(data.get("run_seed", 0))))
    dataset = data.get("dataset") or data.get("domain")
    if "profile_note" not in data:
        if dataset == "synthetic" and data.get("samples_processed", 0) < 50:
            data["profile_note"] = "smoke (synthetic; not comparable to domain runs)"
        elif dataset in ("math", "code", "general") or data.get("domain") in ("math", "code", "general"):
            data["profile_note"] = "domain-matched (same prompts as OLMoE phase-2)"
    if "jetmoe" in data.get("model", "").lower() and "dual-router" not in (data.get("profile_note") or ""):
        base = (data.get("profile_note") or "smoke (synthetic)").rstrip("; ")
        data["profile_note"] = f"{base}; JetMoE dual-router (MoA+MoE)"
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    for path in sorted(RESULTS.glob("routing_stats_*.json")):
        enrich(path)
        print(f"enriched {path.name}")
    print("Run: python scripts/update_leaderboard.py")


if __name__ == "__main__":
    main()
