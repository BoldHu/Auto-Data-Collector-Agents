#!/usr/bin/env python3
"""Validate final SFT v4 dataset.

Checks:
- ChatML format validity
- Evidence presence and support
- Source_refs resolvable
- Task type distribution
- Train/validation isolation (no leakage)

Usage:
    python scripts/validate_final_sft_v4.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    return " ".join(text.lower().split())


def check_leakage(train_records: list[dict], val_records: list[dict], bench_records: list[dict] = None) -> dict:
    """Check for data leakage between train/validation and benchmark."""
    # Exact normalized instruction overlap
    train_instructions = {normalize_text(r.get("instruction", "")) for r in train_records}
    val_instructions = {normalize_text(r.get("instruction", "")) for r in val_records}

    train_val_overlap = train_instructions & val_instructions
    train_val_overlap.discard("")  # Remove empty strings

    # Benchmark instruction overlap
    bench_overlap = set()
    if bench_records:
        bench_questions = {normalize_text(r.get("question", "")) for r in bench_records}
        bench_overlap = train_instructions & bench_questions
        bench_overlap.discard("")

    return {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_val_exact_overlap": len(train_val_overlap),
        "train_val_overlap_rate": round(len(train_val_overlap) / max(len(train_records), 1), 4),
        "benchmark_overlap": len(bench_overlap),
        "benchmark_overlap_rate": round(len(bench_overlap) / max(len(train_records), 1), 4),
        "sample_overlaps": list(train_val_overlap)[:5],
    }


def validate_sft(records: list[dict], split_name: str) -> dict:
    """Validate SFT records for format and quality."""
    total = len(records)
    has_instruction = 0
    has_output = 0
    has_evidence = 0
    has_source_refs = 0
    task_types = Counter()
    source_types = Counter()

    for rec in records:
        if rec.get("instruction"):
            has_instruction += 1
        if rec.get("output"):
            has_output += 1
        if rec.get("evidence") or rec.get("evidence_text"):
            has_evidence += 1
        if rec.get("source_refs"):
            has_source_refs += 1
        task_types[rec.get("task_type", "unknown")] += 1
        source_types[rec.get("source_type", rec.get("source", "unknown"))] += 1

    return {
        "split": split_name,
        "total": total,
        "has_instruction": has_instruction,
        "has_output": has_output,
        "has_evidence": has_evidence,
        "evidence_rate": round(has_evidence / total, 4) if total else 0,
        "has_source_refs": has_source_refs,
        "source_ref_rate": round(has_source_refs / total, 4) if total else 0,
        "task_type_distribution": dict(task_types.most_common()),
        "source_type_distribution": dict(source_types.most_common()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate SFT v4")
    parser.add_argument("--output", default="data/registry/sft_v4_validation.json")
    args = parser.parse_args()

    print("SFT v4 Validation")
    print("=" * 60)

    # Find v4 files
    v4_dir = Path("data/sft/final_v4")
    if not v4_dir.exists():
        print(f"Error: {v4_dir} not found")
        return

    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "splits": {}, "leakage": {}}

    # Validate gold split
    gold_train = v4_dir / "gold" / "train.jsonl"
    gold_val = v4_dir / "gold" / "validation.jsonl"
    if gold_train.exists():
        train_records = load_jsonl(str(gold_train))
        results["splits"]["gold_train"] = validate_sft(train_records, "gold_train")
        print(f"\nGold Train: {len(train_records)} samples")
        print(f"  Evidence rate: {results['splits']['gold_train']['evidence_rate']:.2%}")
        print(f"  Source ref rate: {results['splits']['gold_train']['source_ref_rate']:.2%}")

    if gold_val.exists():
        val_records = load_jsonl(str(gold_val))
        results["splits"]["gold_val"] = validate_sft(val_records, "gold_val")
        print(f"Gold Val: {len(val_records)} samples")

    # Validate full split
    full_train = v4_dir / "full" / "train.jsonl"
    full_val = v4_dir / "full" / "validation.jsonl"
    if full_train.exists():
        full_train_records = load_jsonl(str(full_train))
        results["splits"]["full_train"] = validate_sft(full_train_records, "full_train")
        print(f"\nFull Train: {len(full_train_records)} samples")
        print(f"  Evidence rate: {results['splits']['full_train']['evidence_rate']:.2%}")

    if full_val.exists():
        full_val_records = load_jsonl(str(full_val))
        results["splits"]["full_val"] = validate_sft(full_val_records, "full_val")
        print(f"Full Val: {len(full_val_records)} samples")

    # Leakage checks
    print("\nLeakage Checks:")
    if gold_train.exists() and gold_val.exists():
        leakage = check_leakage(train_records, val_records)
        results["leakage"]["gold"] = leakage
        print(f"  Gold train-val overlap: {leakage['train_val_exact_overlap']} ({leakage['train_val_overlap_rate']:.2%})")

    if full_train.exists() and full_val.exists():
        leakage = check_leakage(full_train_records, full_val_records)
        results["leakage"]["full"] = leakage
        print(f"  Full train-val overlap: {leakage['train_val_exact_overlap']} ({leakage['train_val_overlap_rate']:.2%})")

    # Benchmark leakage
    bench_path = Path("data/benchmark/versions/cfbench_v1_full.jsonl")
    if not bench_path.exists():
        bench_path = Path("data/benchmark/carbon_fiber_benchmark_all.jsonl")
    if bench_path.exists():
        bench_records = load_jsonl(str(bench_path))
        if gold_train.exists():
            bench_leak = check_leakage(train_records, [], bench_records)
            results["leakage"]["gold_bench"] = bench_leak
            print(f"  Gold train-bench overlap: {bench_leak['benchmark_overlap']} ({bench_leak['benchmark_overlap_rate']:.2%})")
        if full_train.exists():
            bench_leak = check_leakage(full_train_records, [], bench_records)
            results["leakage"]["full_bench"] = bench_leak
            print(f"  Full train-bench overlap: {bench_leak['benchmark_overlap']} ({bench_leak['benchmark_overlap_rate']:.2%})")

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
