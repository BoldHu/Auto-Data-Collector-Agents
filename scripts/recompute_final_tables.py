#!/usr/bin/env python3
"""Recompute final evaluation tables from raw outputs ONLY.

This script does NOT read any audit JSON, summary CSVs, or paper-ready
tables as inputs. Every table is generated from raw record-level outputs.

For each result row, records:
- table_name
- model/system/variant
- raw_input_paths
- scorer_version
- parser_version
- denominator
- numerator
- metric_name
- metric_value
- failure_count
- parse_failure_count

Usage:
    python scripts/recompute_final_tables.py --output build/validation/recompute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return records


def compute_accuracy(records: list[dict], field: str = "strict_correct") -> dict:
    """Compute accuracy from scored records."""
    total = len(records)
    if total == 0:
        return {"total": 0, "numerator": 0, "accuracy": 0.0}

    failures = sum(1 for r in records if r.get("error_type"))
    parse_failures = sum(1 for r in records if not r.get("parse_success", True))
    valid = [r for r in records if not r.get("error_type")]

    correct = sum(1 for r in valid if r.get(field, 0) == 1)
    return {
        "total": total,
        "valid": len(valid),
        "numerator": correct,
        "denominator": len(valid),
        "accuracy": round(correct / len(valid), 4) if valid else 0.0,
        "failure_count": failures,
        "parse_failure_count": parse_failures,
    }


def compute_by_task_type(records: list[dict], field: str = "strict_correct") -> dict:
    """Compute accuracy broken down by task type."""
    by_type = defaultdict(list)
    for r in records:
        tt = r.get("task_type", "unknown")
        by_type[tt].append(r)

    result = {}
    for tt, recs in by_type.items():
        valid = [r for r in recs if not r.get("error_type")]
        correct = sum(1 for r in valid if r.get(field, 0) == 1)
        result[tt] = {
            "total": len(recs),
            "correct": correct,
            "accuracy": round(correct / len(valid), 4) if valid else 0.0,
        }
    return result


def recompute_qwen_outputs(data_dir: Path) -> list[dict]:
    """Recompute Qwen model evaluation from raw outputs."""
    rows = []
    eval_dir = data_dir / "evaluation"
    if not eval_dir.exists():
        return rows

    for outputs_file in sorted(eval_dir.rglob("*outputs*.jsonl")):
        records = load_jsonl(str(outputs_file))
        if not records:
            continue

        # Detect surface (canonical 150 vs large 361)
        total = len(records)
        surface = "canonical_150" if total == 150 else "large_361" if total == 361 else f"other_{total}"

        # Detect model from path
        path_str = str(outputs_file)
        if "gold" in path_str:
            model_variant = "gold"
        elif "v4" in path_str or "full" in path_str:
            model_variant = "v4_full"
        elif "base" in path_str:
            model_variant = "base"
        else:
            model_variant = "unknown"

        stats = compute_accuracy(records)
        stats_by_type = compute_by_task_type(records)

        rows.append({
            "table_name": "qwen_evaluation",
            "model": "Qwen2.5-VL-3B",
            "variant": model_variant,
            "surface": surface,
            "raw_input_path": str(outputs_file),
            "scorer_version": "v2_repaired",
            "parser_version": "v2_repaired",
            **stats,
            "by_task_type": stats_by_type,
        })

    return rows


def recompute_closed_source(data_dir: Path) -> list[dict]:
    """Recompute closed-source model evaluation from raw outputs."""
    rows = []
    eval_dir = data_dir / "evaluation"
    if not eval_dir.exists():
        return rows

    # Look for closed-source output files
    for outputs_file in sorted(eval_dir.rglob("*outputs*.jsonl")):
        path_str = str(outputs_file)
        # Skip Qwen files (handled separately)
        if "qwen" in path_str.lower() or "lora" in path_str.lower() or "adapter" in path_str.lower():
            continue

        records = load_jsonl(str(outputs_file))
        if not records:
            continue

        total = len(records)
        surface = "canonical_150" if total == 150 else "large_361" if total == 361 else f"other_{total}"

        # Detect model from path
        model_name = "unknown"
        for name in ["doubao", "glm", "deepseek", "minimax", "kimi"]:
            if name in path_str.lower():
                model_name = name
                break

        stats = compute_accuracy(records)

        rows.append({
            "table_name": "closed_source_evaluation",
            "model": model_name,
            "variant": "default",
            "surface": surface,
            "raw_input_path": str(outputs_file),
            "scorer_version": "v2_repaired",
            "parser_version": "v2_repaired",
            **stats,
        })

    return rows


def recompute_dtcg_system_ablation(data_dir: Path) -> list[dict]:
    """Recompute DTCG system ablation from raw traces."""
    rows = []
    trace_path = data_dir / "evaluation" / "phase_6_9" / "targeted_rerun_traces.jsonl"
    if not trace_path.exists():
        return rows

    records = load_jsonl(str(trace_path))
    if not records:
        return rows

    by_system = defaultdict(list)
    for r in records:
        st = r.get("system_type", "unknown")
        by_system[st].append(r)

    for system, recs in by_system.items():
        valid = [r for r in recs if not r.get("error_type")]
        correct = sum(1 for r in valid if r.get("is_correct", False))
        avg_judge = sum(r.get("judge_score", 0) or 0 for r in valid) / len(valid) if valid else 0
        avg_context = sum(r.get("selected_context_tokens", 0) for r in valid) / len(valid) if valid else 0

        rows.append({
            "table_name": "dtcg_system_ablation",
            "model": "xiaomi_mimo",
            "variant": system,
            "surface": "targeted_rerun",
            "raw_input_path": str(trace_path),
            "scorer_version": "rule_and_judge",
            "parser_version": "v1",
            "total": len(recs),
            "valid": len(valid),
            "numerator": correct,
            "denominator": len(valid),
            "accuracy": round(correct / len(valid), 4) if valid else 0.0,
            "avg_judge_score": round(avg_judge, 4),
            "avg_context_tokens": round(avg_context),
            "failure_count": len(recs) - len(valid),
        })

    return rows


def recompute_dtcg_component_ablation(data_dir: Path) -> list[dict]:
    """Recompute DTCG component ablation from raw traces."""
    rows = []
    ablation_dir = data_dir / "evaluation" / "dtcg_ablation"
    if not ablation_dir.exists():
        # Try build output
        ablation_dir = Path("build/validation/dtcg_ablation")
    if not ablation_dir.exists():
        return rows

    for trace_file in sorted(ablation_dir.glob("trace_*.jsonl")):
        variant = trace_file.stem.replace("trace_", "")
        records = load_jsonl(str(trace_file))
        if not records:
            continue

        total = len(records)
        success = sum(1 for r in records if r.get("task_success", False))

        rows.append({
            "table_name": "dtcg_component_ablation",
            "model": "xiaomi_mimo",
            "variant": variant,
            "surface": "component_ablation",
            "raw_input_path": str(trace_file),
            "scorer_version": "persistent_eval_v1",
            "parser_version": "v1",
            "total": total,
            "numerator": success,
            "denominator": total,
            "success_rate": round(success / total, 4) if total else 0.0,
            "failure_count": 0,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Recompute final tables from raw outputs")
    parser.add_argument("--output", default="build/validation/recompute")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("Final Table Recomputation (Raw Outputs Only)")
    print("=" * 60)

    all_rows = []

    # Qwen evaluation
    print("\nRecomputing Qwen evaluation...")
    qwen_rows = recompute_qwen_outputs(data_dir)
    all_rows.extend(qwen_rows)
    print(f"  Found {len(qwen_rows)} Qwen result rows")
    for row in qwen_rows:
        print(f"    {row['variant']}/{row['surface']}: {row['numerator']}/{row['denominator']} = {row['accuracy']:.4f}")

    # Closed-source evaluation
    print("\nRecomputing closed-source evaluation...")
    cs_rows = recompute_closed_source(data_dir)
    all_rows.extend(cs_rows)
    print(f"  Found {len(cs_rows)} closed-source result rows")
    for row in cs_rows:
        print(f"    {row['model']}/{row['surface']}: {row['numerator']}/{row['denominator']} = {row['accuracy']:.4f}")

    # DTCG system ablation
    print("\nRecomputing DTCG system ablation...")
    dtcg_rows = recompute_dtcg_system_ablation(data_dir)
    all_rows.extend(dtcg_rows)
    print(f"  Found {len(dtcg_rows)} DTCG system ablation rows")
    for row in dtcg_rows:
        print(f"    {row['variant']}: {row['numerator']}/{row['denominator']} = {row['accuracy']:.4f}")

    # DTCG component ablation
    print("\nRecomputing DTCG component ablation...")
    comp_rows = recompute_dtcg_component_ablation(data_dir)
    all_rows.extend(comp_rows)
    print(f"  Found {len(comp_rows)} DTCG component ablation rows")

    # Write full recomputation
    recompute_path = output_dir / "recomputed_tables.jsonl"
    with open(recompute_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nFull recomputation: {recompute_path}")

    # Write summary CSV
    csv_path = output_dir / "recomputed_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("table_name,model,variant,surface,denominator,numerator,metric_value,raw_input_path\n")
        for row in all_rows:
            metric = row.get("accuracy", row.get("success_rate", 0))
            f.write(f"{row['table_name']},{row['model']},{row['variant']},{row['surface']},"
                    f"{row['denominator']},{row['numerator']},{metric},{row['raw_input_path']}\n")
    print(f"Summary CSV: {csv_path}")

    # Write manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_dir": str(data_dir),
        "total_rows": len(all_rows),
        "tables": list(set(r["table_name"] for r in all_rows)),
        "raw_input_paths": list(set(r["raw_input_path"] for r in all_rows)),
    }
    manifest_path = output_dir / "recompute_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest: {manifest_path}")

    # Detect accidental truncation
    for row in all_rows:
        if row["denominator"] == 50:
            print(f"\n  WARNING: Row has denominator=50, possible eval_items[:50] truncation:")
            print(f"    {row['raw_input_path']}")


if __name__ == "__main__":
    main()
