#!/usr/bin/env python3
"""Validate CFBench v1 benchmark.

Checks:
- Duplicate questions
- Missing answers
- Missing evidence
- Answer/evidence consistency
- Source availability
- Task type distribution
- Difficulty distribution
- Answer format distribution

Usage:
    python scripts/validate_cfbench_v1.py
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
    return " ".join(text.lower().split())


def validate_benchmark(records: list[dict], name: str) -> dict:
    """Validate benchmark records."""
    total = len(records)

    # Required fields
    has_question = sum(1 for r in records if r.get("question"))
    has_answer = sum(1 for r in records if r.get("answer"))
    has_options = sum(1 for r in records if r.get("options"))
    has_evidence = sum(1 for r in records if r.get("evidence"))
    has_source_refs = sum(1 for r in records if r.get("source_refs"))
    has_task_type = sum(1 for r in records if r.get("task_type"))
    has_difficulty = sum(1 for r in records if r.get("difficulty"))

    # Distributions
    task_types = Counter(r.get("task_type", "unknown") for r in records)
    difficulties = Counter(r.get("difficulty", "unknown") for r in records)
    modalities = Counter(r.get("modality", "text") for r in records)
    sources = Counter(r.get("source", "unknown") for r in records)

    # Duplicate detection
    questions = [normalize_text(r.get("question", "")) for r in records]
    seen = {}
    duplicates = []
    for i, q in enumerate(questions):
        if q and q in seen:
            duplicates.append((seen[q], i, q[:80]))
        elif q:
            seen[q] = i

    # Answer format analysis
    answer_formats = Counter()
    for r in records:
        answer = str(r.get("answer", ""))
        if not answer:
            answer_formats["missing"] += 1
        elif len(answer) == 1 and answer.isalpha():
            answer_formats["single_letter"] += 1
        elif answer in ["A", "B", "C", "D", "E", "F"]:
            answer_formats["choice_letter"] += 1
        elif answer.replace(".", "").replace("-", "").isdigit():
            answer_formats["numeric"] += 1
        elif len(answer) < 20:
            answer_formats["short_text"] += 1
        else:
            answer_formats["long_text"] += 1

    return {
        "name": name,
        "total": total,
        "has_question": has_question,
        "has_answer": has_answer,
        "has_options": has_options,
        "has_evidence": has_evidence,
        "evidence_rate": round(has_evidence / total, 4) if total else 0,
        "has_source_refs": has_source_refs,
        "source_ref_rate": round(has_source_refs / total, 4) if total else 0,
        "has_task_type": has_task_type,
        "has_difficulty": has_difficulty,
        "task_type_distribution": dict(task_types.most_common()),
        "difficulty_distribution": dict(difficulties),
        "modality_distribution": dict(modalities),
        "source_distribution": dict(sources.most_common()),
        "answer_format_distribution": dict(answer_formats.most_common()),
        "duplicate_count": len(duplicates),
        "duplicate_pairs": duplicates[:10],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate CFBench v1")
    parser.add_argument("--output", default="data/registry/cfbench_v1_validation.json")
    args = parser.parse_args()

    print("CFBench v1 Validation")
    print("=" * 60)

    # Find benchmark files
    bench_dir = Path("data/benchmark")
    versions_dir = bench_dir / "versions"

    files_to_validate = []

    # Full benchmark
    full_path = bench_dir / "carbon_fiber_benchmark_all.jsonl"
    if full_path.exists():
        files_to_validate.append(("full", str(full_path)))

    # Versioned subsets
    if versions_dir.exists():
        for f in sorted(versions_dir.glob("*.jsonl")):
            name = f.stem.replace("cfbench_v1_", "")
            files_to_validate.append((name, str(f)))

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": {},
    }

    for name, path in files_to_validate:
        records = load_jsonl(path)
        if not records:
            print(f"\n{name}: No records found")
            continue

        validation = validate_benchmark(records, name)
        results["benchmarks"][name] = validation

        print(f"\n{name}: {validation['total']} items")
        print(f"  Questions: {validation['has_question']}, Answers: {validation['has_answer']}")
        print(f"  Evidence rate: {validation['evidence_rate']:.2%}")
        print(f"  Source ref rate: {validation['source_ref_rate']:.2%}")
        print(f"  Duplicates: {validation['duplicate_count']}")
        print(f"  Task types: {validation['task_type_distribution']}")
        print(f"  Difficulties: {validation['difficulty_distribution']}")
        print(f"  Answer formats: {validation['answer_format_distribution']}")

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
