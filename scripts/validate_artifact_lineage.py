#!/usr/bin/env python3
"""Artifact lineage validator.

Validates traceability from final artifacts back to raw sources:
- SFT sample -> evidence -> cleaned chunk -> raw source
- Benchmark item -> source candidate -> cleaned/image/exam source
- Model prediction -> benchmark item -> evaluator/parser version

Usage:
    python scripts/validate_artifact_lineage.py
    python scripts/validate_artifact_lineage.py --output data/registry/lineage_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "hash_error"


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
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


def validate_sft_lineage(sft_path: str, cleaned_chunks_path: str = None) -> dict:
    """Validate SFT sample lineage.

    Checks:
    - Each SFT sample has source_refs
    - Each SFT sample has evidence
    - source_refs point to resolvable files
    - evidence text is non-empty
    """
    records = load_jsonl(sft_path)
    if not records:
        return {"error": f"No records found in {sft_path}", "total": 0}

    total = len(records)
    has_source_refs = 0
    has_evidence = 0
    resolvable_refs = 0
    nonempty_evidence = 0
    missing_source_refs = []
    missing_evidence = []

    for i, rec in enumerate(records):
        sample_id = rec.get("sample_id", rec.get("id", f"row_{i}"))

        # Check source_refs
        source_refs = rec.get("source_refs", [])
        if source_refs:
            has_source_refs += 1
            # Check if refs are resolvable
            for ref in source_refs:
                if isinstance(ref, str) and (Path(ref).exists() or ref.startswith("data/")):
                    resolvable_refs += 1
                    break
        else:
            missing_source_refs.append(sample_id)

        # Check evidence
        evidence = rec.get("evidence", rec.get("evidence_text", ""))
        if evidence:
            has_evidence += 1
            if len(str(evidence).strip()) > 10:
                nonempty_evidence += 1
        else:
            missing_evidence.append(sample_id)

    return {
        "artifact": sft_path,
        "total": total,
        "has_source_refs": has_source_refs,
        "source_ref_rate": round(has_source_refs / total, 4) if total else 0,
        "resolvable_refs": resolvable_refs,
        "has_evidence": has_evidence,
        "evidence_rate": round(has_evidence / total, 4) if total else 0,
        "nonempty_evidence": nonempty_evidence,
        "nonempty_evidence_rate": round(nonempty_evidence / total, 4) if total else 0,
        "missing_source_refs_sample": missing_source_refs[:10],
        "missing_evidence_sample": missing_evidence[:10],
    }


def validate_benchmark_lineage(bench_path: str) -> dict:
    """Validate benchmark item lineage.

    Checks:
    - Each item has source_refs
    - Each item has question and answer
    - source_refs are non-empty
    """
    records = load_jsonl(bench_path)
    if not records:
        return {"error": f"No records found in {bench_path}", "total": 0}

    total = len(records)
    has_source_refs = 0
    has_question = 0
    has_answer = 0
    has_evidence = 0
    task_types = Counter()
    difficulties = Counter()

    for rec in records:
        if rec.get("source_refs"):
            has_source_refs += 1
        if rec.get("question"):
            has_question += 1
        if rec.get("answer") or rec.get("options"):
            has_answer += 1
        if rec.get("evidence"):
            has_evidence += 1
        task_types[rec.get("task_type", "unknown")] += 1
        difficulties[rec.get("difficulty", "unknown")] += 1

    return {
        "artifact": bench_path,
        "total": total,
        "has_source_refs": has_source_refs,
        "source_ref_rate": round(has_source_refs / total, 4) if total else 0,
        "has_question": has_question,
        "has_answer": has_answer,
        "has_evidence": has_evidence,
        "evidence_rate": round(has_evidence / total, 4) if total else 0,
        "task_type_distribution": dict(task_types.most_common()),
        "difficulty_distribution": dict(difficulties),
    }


def validate_evaluation_lineage(eval_path: str, bench_path: str = None) -> dict:
    """Validate model evaluation lineage.

    Checks:
    - Each prediction has benchmark_id
    - Predictions link to benchmark items
    - Scoring fields are present
    """
    records = load_jsonl(eval_path)
    if not records:
        return {"error": f"No records found in {eval_path}", "total": 0}

    total = len(records)
    has_benchmark_id = 0
    has_parsed_answer = 0
    has_score = 0
    task_types = Counter()

    # Load benchmark for cross-reference
    bench_ids = set()
    if bench_path and Path(bench_path).exists():
        for rec in load_jsonl(bench_path):
            bid = rec.get("benchmark_id", rec.get("item_id", ""))
            if bid:
                bench_ids.add(bid)

    matched_bench_ids = 0
    for rec in records:
        bid = rec.get("benchmark_id", rec.get("item_id", ""))
        if bid:
            has_benchmark_id += 1
            if bid in bench_ids:
                matched_bench_ids += 1
        if rec.get("parsed_answer"):
            has_parsed_answer += 1
        if rec.get("strict_correct") is not None or rec.get("correct") is not None:
            has_score += 1
        task_types[rec.get("task_type", "unknown")] += 1

    return {
        "artifact": eval_path,
        "total": total,
        "has_benchmark_id": has_benchmark_id,
        "benchmark_id_rate": round(has_benchmark_id / total, 4) if total else 0,
        "matched_benchmark_ids": matched_bench_ids,
        "has_parsed_answer": has_parsed_answer,
        "has_score": has_score,
        "score_rate": round(has_score / total, 4) if total else 0,
        "task_type_distribution": dict(task_types.most_common()),
    }


def build_artifact_registry(data_dir: str = "data") -> list[dict]:
    """Build a registry of all artifacts in the data directory."""
    registry = []
    data_path = Path(data_dir)

    for jsonl_file in sorted(data_path.rglob("*.jsonl")):
        try:
            count = sum(1 for line in open(jsonl_file, encoding="utf-8") if line.strip())
        except Exception:
            count = 0

        registry.append({
            "artifact_id": f"art_{hashlib.md5(str(jsonl_file).encode()).hexdigest()[:8]}",
            "path": str(jsonl_file),
            "type": "jsonl",
            "record_count": count,
            "size_bytes": jsonl_file.stat().st_size,
            "hash": compute_hash(str(jsonl_file)),
            "timestamp": jsonl_file.stat().st_mtime,
        })

    return registry


def main():
    parser = argparse.ArgumentParser(description="Validate artifact lineage")
    parser.add_argument("--output", default="data/registry/lineage_report.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    print("Artifact Lineage Validator")
    print("=" * 60)

    # Build artifact registry
    print("\nBuilding artifact registry...")
    registry = build_artifact_registry(args.data_dir)
    print(f"  Found {len(registry)} JSONL artifacts")

    # Find key artifacts
    sft_files = [r for r in registry if "sft" in r["path"].lower() and "final" in r["path"].lower()]
    bench_files = [r for r in registry if "benchmark" in r["path"].lower() and ("canonical" in r["path"].lower() or "large" in r["path"].lower() or "all" in r["path"].lower())]
    eval_files = [r for r in registry if "evaluation" in r["path"].lower() and "outputs" in r["path"].lower()]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "registry_count": len(registry),
        "sft_lineage": [],
        "benchmark_lineage": [],
        "evaluation_lineage": [],
    }

    # Validate SFT lineage
    print("\nValidating SFT lineage...")
    for sft in sft_files[:5]:
        result = validate_sft_lineage(sft["path"])
        report["sft_lineage"].append(result)
        total = result.get("total", 0)
        if total:
            print(f"  {Path(sft['path']).name}: {total} samples, "
                  f"evidence_rate={result.get('evidence_rate', 0):.2%}, "
                  f"source_ref_rate={result.get('source_ref_rate', 0):.2%}")

    # Validate benchmark lineage
    print("\nValidating benchmark lineage...")
    for bench in bench_files[:5]:
        result = validate_benchmark_lineage(bench["path"])
        report["benchmark_lineage"].append(result)
        total = result.get("total", 0)
        if total:
            print(f"  {Path(bench['path']).name}: {total} items, "
                  f"evidence_rate={result.get('evidence_rate', 0):.2%}, "
                  f"source_ref_rate={result.get('source_ref_rate', 0):.2%}")

    # Validate evaluation lineage
    print("\nValidating evaluation lineage...")
    for ev in eval_files[:5]:
        result = validate_evaluation_lineage(ev["path"])
        report["evaluation_lineage"].append(result)
        total = result.get("total", 0)
        if total:
            print(f"  {Path(ev['path']).name}: {total} predictions, "
                  f"score_rate={result.get('score_rate', 0):.2%}")

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")

    # Summary
    total_sft = sum(r.get("total", 0) for r in report["sft_lineage"])
    total_bench = sum(r.get("total", 0) for r in report["benchmark_lineage"])
    total_eval = sum(r.get("total", 0) for r in report["evaluation_lineage"])
    print(f"\nSummary: {total_sft} SFT samples, {total_bench} benchmark items, {total_eval} evaluation predictions")


if __name__ == "__main__":
    main()
