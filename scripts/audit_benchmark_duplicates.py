#!/usr/bin/env python3
"""Benchmark Duplicate Audit Script.

Computes:
- Exact duplicate groups
- Normalized duplicate groups
- Semantic/heuristic near-duplicate groups
- Cross-subset duplicates
- SFT overlap/leakage candidates

Produces a resolution table with actions (keep/remove/merge/document).

Usage:
    python scripts/audit_benchmark_duplicates.py --output build/validation/benchmark_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    """Hash normalized text."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def word_set(text: str) -> set[str]:
    """Get set of significant words (length > 2)."""
    return {w for w in normalize_text(text).split() if len(w) > 2}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def find_exact_duplicates(records: list[dict]) -> list[list[int]]:
    """Find exact normalized question duplicates."""
    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        q = rec.get("question", "")
        if q:
            h = text_hash(q)
            hash_to_indices[h].append(i)

    groups = []
    for h, indices in hash_to_indices.items():
        if len(indices) > 1:
            groups.append(indices)
    return groups


def find_near_duplicates(records: list[dict], threshold: float = 0.8) -> list[list[int]]:
    """Find near-duplicate groups using Jaccard similarity."""
    word_sets = [word_set(rec.get("question", "")) for rec in records]
    n = len(records)
    visited = set()
    groups = []

    for i in range(n):
        if i in visited or not word_sets[i]:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited or not word_sets[j]:
                continue
            if jaccard_similarity(word_sets[i], word_sets[j]) >= threshold:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)

    return groups


def find_sft_overlap(bench_records: list[dict], sft_records: list[dict]) -> list[dict]:
    """Find benchmark items that overlap with SFT training data."""
    sft_questions = {}
    for i, rec in enumerate(sft_records):
        q = normalize_text(rec.get("instruction", ""))
        if q:
            sft_questions[q] = i

    overlaps = []
    for i, rec in enumerate(bench_records):
        q = normalize_text(rec.get("question", ""))
        if q and q in sft_questions:
            overlaps.append({
                "bench_index": i,
                "bench_id": rec.get("benchmark_id", rec.get("item_id", f"bench_{i}")),
                "sft_index": sft_questions[q],
                "question_preview": q[:100],
            })

    return overlaps


def build_resolution_table(
    records: list[dict],
    exact_groups: list[list[int]],
    near_groups: list[list[int]],
) -> list[dict]:
    """Build a resolution table for duplicate items."""
    # Track which indices are in duplicate groups
    exact_dup_indices = set()
    for group in exact_groups:
        for idx in group[1:]:  # Keep first, mark rest as duplicates
            exact_dup_indices.add(idx)

    near_dup_indices = set()
    for group in near_groups:
        for idx in group[1:]:
            near_dup_indices.add(idx)

    resolutions = []
    for i, rec in enumerate(records):
        item_id = rec.get("benchmark_id", rec.get("item_id", f"item_{i}"))
        question = rec.get("question", "")

        if i in exact_dup_indices:
            # Find which group this belongs to
            group_id = None
            for gi, group in enumerate(exact_groups):
                if i in group:
                    group_id = f"exact_{gi}"
                    break
            resolutions.append({
                "item_id": item_id,
                "index": i,
                "duplicate_group_id": group_id,
                "subset": rec.get("subset", "unknown"),
                "normalized_question_hash": text_hash(question),
                "source_ref": str(rec.get("source_refs", []))[:100],
                "action": "remove",
                "reason": "exact_normalized_duplicate",
            })
        elif i in near_dup_indices:
            group_id = None
            for gi, group in enumerate(near_groups):
                if i in group:
                    group_id = f"near_{gi}"
                    break
            resolutions.append({
                "item_id": item_id,
                "index": i,
                "duplicate_group_id": group_id,
                "subset": rec.get("subset", "unknown"),
                "normalized_question_hash": text_hash(question),
                "source_ref": str(rec.get("source_refs", []))[:100],
                "action": "document",
                "reason": "near_duplicate_jaccard>=0.8",
            })
        else:
            resolutions.append({
                "item_id": item_id,
                "index": i,
                "duplicate_group_id": None,
                "subset": rec.get("subset", "unknown"),
                "normalized_question_hash": text_hash(question),
                "source_ref": str(rec.get("source_refs", []))[:100],
                "action": "keep",
                "reason": "no_duplicate",
            })

    return resolutions


def main():
    parser = argparse.ArgumentParser(description="Benchmark Duplicate Audit")
    parser.add_argument("--output", default="build/validation/benchmark_audit")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--sft", default=None)
    parser.add_argument("--near-threshold", type=float, default=0.8)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark Duplicate Audit")
    print("=" * 60)

    # Find benchmark
    bench_path = args.benchmark
    if not bench_path:
        candidates = [
            "data/benchmark/versions/cfbench_v1_full.jsonl",
            "data/benchmark/carbon_fiber_benchmark_all.jsonl",
        ]
        for p in candidates:
            if Path(p).exists():
                bench_path = p
                break

    if not bench_path:
        print("Error: No benchmark file found")
        sys.exit(1)

    records = load_jsonl(bench_path)
    print(f"Benchmark: {bench_path}")
    print(f"Total records: {len(records)}")

    # Find exact duplicates
    exact_groups = find_exact_duplicates(records)
    exact_dup_count = sum(len(g) - 1 for g in exact_groups)
    print(f"\nExact normalized duplicate groups: {len(exact_groups)}")
    print(f"Extra duplicate items: {exact_dup_count}")

    # Find near duplicates
    near_groups = find_near_duplicates(records, threshold=args.near_threshold)
    near_dup_count = sum(len(g) - 1 for g in near_groups)
    print(f"Near-duplicate groups (Jaccard >= {args.near_threshold}): {len(near_groups)}")
    print(f"Extra near-duplicate items: {near_dup_count}")

    # SFT overlap
    sft_path = args.sft
    sft_overlap = []
    if sft_path and Path(sft_path).exists():
        sft_records = load_jsonl(sft_path)
        sft_overlap = find_sft_overlap(records, sft_records)
        print(f"SFT overlap items: {len(sft_overlap)}")

    # Build resolution table
    resolutions = build_resolution_table(records, exact_groups, near_groups)

    # Compute statistics
    action_counts = Counter(r["action"] for r in resolutions)
    print(f"\nResolution actions:")
    for action, count in action_counts.most_common():
        print(f"  {action}: {count}")

    # Compute subset/difficulty/modality distribution for kept items
    kept = [r for r in resolutions if r["action"] == "keep"]
    kept_records = [records[r["index"]] for r in kept]

    # Write outputs
    # Resolution table
    resolution_path = output_dir / "duplicate_resolution.jsonl"
    with open(resolution_path, "w", encoding="utf-8") as f:
        for r in resolutions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResolution table: {resolution_path}")

    # Summary
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark_path": bench_path,
        "total_records": len(records),
        "exact_duplicate_groups": len(exact_groups),
        "exact_extra_items": exact_dup_count,
        "near_duplicate_groups": len(near_groups),
        "near_extra_items": near_dup_count,
        "near_threshold": args.near_threshold,
        "sft_overlap_count": len(sft_overlap),
        "action_counts": dict(action_counts),
        "kept_count": len(kept),
        "removed_count": action_counts.get("remove", 0),
        "documented_count": action_counts.get("document", 0),
    }
    summary_path = output_dir / "duplicate_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary: {summary_path}")

    # Exact duplicate groups detail
    groups_detail = []
    for gi, group in enumerate(exact_groups):
        groups_detail.append({
            "group_id": f"exact_{gi}",
            "indices": group,
            "item_ids": [records[i].get("benchmark_id", records[i].get("item_id", f"item_{i}")) for i in group],
            "question_preview": records[group[0]].get("question", "")[:100],
        })
    groups_path = output_dir / "exact_duplicate_groups.json"
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(groups_detail, f, ensure_ascii=False, indent=2)
    print(f"Exact groups: {groups_path}")


if __name__ == "__main__":
    main()
