#!/usr/bin/env python3
"""
5-shot MMLU subset on Mac: OLMoE base vs LoRA fine-tuned with L_div.

Directional evidence for the +MMLU claim (not full 57-task suite).
Requires: pip install lm-eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lm_eval"
MODEL = "allenai/OLMoE-1B-7B-0924"
# 12 MMLU subjects (~directional macro average on Mac in ~30-60 min)
MMLU_SUBSET = (
    "mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,"
    "mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,"
    "mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,"
    "mmlu_college_physics,mmlu_computer_security"
)


def run_eval(name: str, model_args: str, limit: int, num_fewshot: int) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / name
    out_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        MMLU_SUBSET,
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        "1",
        "--limit",
        str(limit),
        "--output_path",
        str(out_dir),
    ]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # lm-eval writes results_*.json under out_dir
    json_files = sorted(out_dir.glob("**/results*.json"), key=lambda p: p.stat().st_mtime)
    if not json_files:
        json_files = sorted(out_dir.glob("*.json"))
    data = json.loads(json_files[-1].read_text())
    return data


def extract_mmlu_avg(results: dict) -> float:
    """Macro-average acc across MMLU tasks in results."""
    scores = []
    for key, val in results.get("results", {}).items():
        if "mmlu" in key and isinstance(val, dict):
            acc = val.get("acc,none") or val.get("acc_norm,none") or val.get("acc")
            if acc is not None:
                scores.append(float(acc))
    return sum(scores) / len(scores) if scores else 0.0


def _adapter_module_to_weight_key(module_key: str, state_keys: set[str]) -> str | None:
    """Map PEFT adapter module path to a key in the base model state_dict."""
    candidates = [
        f"{module_key}.weight",
        module_key.replace("model.model.", "model.", 1) + ".weight",
        module_key.replace("model.", "", 1) + ".weight",
    ]
    for key in candidates:
        if key in state_keys:
            return key
    return None


def merge_adapter(model_id: str, adapter: Path, out_dir: Path) -> Path:
    """Merge LoRA into base weights without PeftModel (peft 0.19 + transformers 5.x clash)."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scaling = float(cfg["lora_alpha"]) / float(cfg["r"])
    adapter_state = load_file(str(adapter / "adapter_model.safetensors"))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging LoRA manually (scaling={scaling:.4f}) → {out_dir} ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, trust_remote_code=True
    )
    state = model.state_dict()
    state_keys = set(state.keys())

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter_state.items():
        if ".lora_A." in key:
            mod = key.replace("base_model.", "").split(".lora_A.")[0]
            pairs.setdefault(mod, {})["A"] = tensor
        elif ".lora_B." in key:
            mod = key.replace("base_model.", "").split(".lora_B.")[0]
            pairs.setdefault(mod, {})["B"] = tensor

    merged_count = 0
    for mod_key, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            raise RuntimeError(f"Incomplete LoRA pair for {mod_key}")
        weight_key = _adapter_module_to_weight_key(mod_key, state_keys)
        if weight_key is None:
            raise RuntimeError(f"No base weight for adapter module {mod_key!r}")
        delta = (ab["B"] @ ab["A"]) * scaling
        state[weight_key] = state[weight_key] + delta.to(dtype=state[weight_key].dtype)
        merged_count += 1

    model.load_state_dict(state)
    model.save_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.save_pretrained(out_dir)
    print(f"Merged {merged_count} LoRA modules into base weights.")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25, help="Examples per MMLU subject")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=RESULTS.parent / "training" / "adapters" / "ldiv_l0.01_t1.5",
    )
    parser.add_argument("--train-first", action="store_true", help="Run mini L_div train if no adapter")
    parser.add_argument("--base-only", action="store_true", help="Skip L_div eval (use existing base results)")
    parser.add_argument(
        "--ldiv-only",
        action="store_true",
        help="Skip base re-eval; merge adapter and run L_div only (needs existing mmlu_base results)",
    )
    args = parser.parse_args()
    if args.base_only and args.ldiv_only:
        parser.error("Use only one of --base-only or --ldiv-only")

    if args.train_first and not args.adapter.exists():
        sys.path.insert(0, str(ROOT))
        from moe_bench.device import default_device
        from moe_bench.train import train_moe_aux

        print("Training LoRA + L_div adapter (mini run)...")
        train_moe_aux(
            MODEL,
            RESULTS.parent / "training" / "train_for_mmlu.json",
            aux_loss="ldiv",
            lambda_div=0.01,
            tau=1.5,
            max_samples=48,
            epochs=1,
            device=default_device(),
        )

    if not args.adapter.exists():
        print(f"Adapter not found: {args.adapter}")
        print("Run: python scripts/train_lora_ldiv.py --max-samples 48")
        sys.exit(1)

    base_args = f"pretrained={MODEL},trust_remote_code=True,dtype=float16"
    base_json = sorted((RESULTS / "mmlu_base").glob("**/results*.json"))
    if base_json and (args.base_only or args.ldiv_only):
        base_res = json.loads(base_json[-1].read_text())
        print(f"Using existing base results: {base_json[-1]}")
    else:
        base_res = run_eval("mmlu_base", base_args, args.limit, args.num_fewshot)

    ldiv_res = None
    if not args.base_only:
        merged_dir = RESULTS / "olmoe_ldiv_merged"
        merge_adapter(MODEL, args.adapter, merged_dir)
        ldiv_args = f"pretrained={merged_dir},trust_remote_code=True,dtype=float16"
        ldiv_res = run_eval("mmlu_ldiv", ldiv_args, args.limit, args.num_fewshot)

    base_avg = extract_mmlu_avg(base_res)
    ldiv_avg = extract_mmlu_avg(ldiv_res) if ldiv_res else None
    delta = (ldiv_avg - base_avg) if ldiv_avg is not None else None

    report = {
        "model": MODEL,
        "mmlu_subjects": 12,
        "num_fewshot": args.num_fewshot,
        "limit_per_subject": args.limit,
        "base_macro_acc": base_avg,
        "ldiv_macro_acc": ldiv_avg,
        "delta": delta,
        "note": "Mac subset run; not full 57-task MMLU. L_div uses merged LoRA weights.",
    }
    out = RESULTS / "mmlu_ldiv_comparison.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n=== MMLU subset (macro avg over 12 subjects) ===")
    print(f"Base:     {base_avg:.4f}")
    if ldiv_avg is not None:
        print(f"L_div:    {ldiv_avg:.4f}")
        print(f"Delta:    {delta:+.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
