#!/usr/bin/env python3
"""Generate results/ROUTING_PROFILE_TABLE.md from routing stats JSON files."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PROFILE_TABLE_PATH = RESULTS_DIR / "ROUTING_PROFILE_TABLE.md"
LEGACY_LEADERBOARD_PATH = RESULTS_DIR / "LEADERBOARD.md"


def load_reports() -> list[dict]:
    reports = []
    for path in sorted(RESULTS_DIR.glob("routing_stats_*.json")):
        data = json.loads(path.read_text())
        data["_path"] = path.name
        reports.append(data)
    return reports


def _profile_key(r: dict) -> tuple:
    model = r.get("model", "?")
    label = r.get("domain") or r.get("dataset") or r.get("tag") or r["_path"]
    return (model, str(label))


def _dedupe_latest(reports: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for r in reports:
        key = _profile_key(r)
        prev = by_key.get(key)
        if prev is None or r.get("_path", "") >= prev.get("_path", ""):
            by_key[key] = r
    return sorted(by_key.values(), key=lambda x: (_profile_key(x)))


def _fmt_ci(pair: list[float] | None) -> str:
    if not pair or len(pair) != 2:
        return "—"
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def _fmt_note(r: dict) -> str:
    note = r.get("profile_note") or ""
    if not note and r.get("dataset") == "synthetic":
        note = "smoke†"
    return note.replace("|", "/")[:80] if note else "—"


def main() -> None:
    reports = _dedupe_latest(load_reports())
    lines = [
        "# MoE-Bench routing profile table",
        "",
        "Per-model **routing** statistics (entropy, collapse, load imbalance)—not downstream accuracy.",
        "Auto-generated from `results/routing_stats_*.json`.",
        "",
        "† **smoke** = synthetic stream, often fewer tokens; compare **domain-matched** rows (math/code/general) across models.",
        "CIs = layer-bootstrap (95%), equal weight per hooked router module; see [`CONTRIBUTING.md`](../CONTRIBUTING.md).",
        "",
        "| Model | Condition | n samples | Token positions | Mean entropy [95% CI] | Collapse rate [95% CI] | Load imb. | Note |",
        "|-------|-----------|-----------|-----------------|------------------------|-------------------------|-----------|------|",
    ]
    for r in reports:
        cond = r.get("domain") or r.get("dataset") or "?"
        ent_ci = _fmt_ci(r.get("mean_entropy_ci_95"))
        col_ci = _fmt_ci(r.get("collapse_rate_ci_95"))
        lines.append(
            f"| {r.get('model', '?')} "
            f"| {cond} "
            f"| {r.get('samples_processed', '—')} "
            f"| {r.get('total_token_positions', '—')} "
            f"| {r.get('mean_entropy', 0):.3f} {ent_ci} "
            f"| {r.get('collapse_rate', 0):.2%} {col_ci} "
            f"| {r.get('avg_load_imbalance', 0):.2f} "
            f"| {_fmt_note(r)} |"
        )
    if not reports:
        lines.append("| _No results yet_ | — | — | — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Cross-model domain comparison (headline)",
            "",
            "Compare **math** rows across models (same local prompt set, τ=0.40).",
            "",
        ]
    )
    math_rows = [r for r in reports if (r.get("domain") or r.get("dataset")) == "math"]
    if math_rows:
        lines.append("| Model | Collapse rate | Mean entropy | Peak layer util |")
        lines.append("|-------|---------------|--------------|-----------------|")
        for r in math_rows:
            per = r.get("per_layer") or []
            peak = max((p.get("max_expert_util", 0) for p in per), default=0.0)
            lines.append(
                f"| {r.get('model', '?')} | {r.get('collapse_rate', 0):.1%} "
                f"| {r.get('mean_entropy', 0):.3f} | {peak:.1%} |"
            )
    else:
        lines.append("_Run `python scripts/run_domain_profiles.py` to populate math domain rows._")

    text = "\n".join(lines) + "\n"
    PROFILE_TABLE_PATH.write_text(text)
    LEGACY_LEADERBOARD_PATH.write_text(
        text.replace(
            "# MoE-Bench routing profile table",
            "# MoE-Bench routing profile table\n\n"
            "> **Deprecated:** use [`ROUTING_PROFILE_TABLE.md`](ROUTING_PROFILE_TABLE.md).\n",
            1,
        )
    )
    print(f"Wrote {PROFILE_TABLE_PATH} ({len(reports)} profiles)")


if __name__ == "__main__":
    main()
