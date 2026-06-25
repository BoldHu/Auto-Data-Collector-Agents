#!/usr/bin/env python3
"""Create Versioned Benchmark.

Reads the duplicate audit resolution table and creates a versioned benchmark
with duplicates removed. Produces:
- Versioned benchmark file (cfbench_v1_1.jsonl)
- Old-to-new ID mapping
- Changelog
- Updated manifest

Usage:
    python scripts/create_versioned_benchmark.py --audit-dir build/validation/benchmark_audit --output data/benchmark/versions
"""

from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Create versioned benchmark")
    parser.add_argument("--audit-dir", default="build/validation/benchmark_audit")
    parser.add_argument("--output", default="data/benchmark/versions")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--version", default="v1.1")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating CFBench {args.version}")
    print("=" * 60)

    # Load resolution table
    resolution_path = audit_dir / "duplicate_resolution.jsonl"
    if not resolution_path.exists():
        print(f"Error: {resolution_path} not found. Run audit_benchmark_duplicates.py first.")
        sys.exit(1)

    resolutions = load_jsonl(str(resolution_path))

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
    print(f"Source: {bench_path}")
    print(f"Original records: {len(records)}")

    # Filter: keep only records with action=keep
    keep_indices = {r["index"] for r in resolutions if r["action"] == "keep"}
    kept_records = []
    id_mapping = []

    for i, rec in enumerate(records):
        old_id = rec.get("benchmark_id", rec.get("item_id", f"item_{i}"))
        if i in keep_indices:
            new_id = f"cfbench_{args.version}_{len(kept_records):04d}"
            new_rec = dict(rec)
            new_rec["benchmark_id"] = new_id
            new_rec["original_id"] = old_id
            new_rec["version"] = args.version
            kept_records.append(new_rec)
            id_mapping.append({
                "old_id": old_id,
                "new_id": new_id,
                "action": "kept",
            })
        else:
            id_mapping.append({
                "old_id": old_id,
                "new_id": None,
                "action": "removed",
                "reason": next((r["reason"] for r in resolutions if r["index"] == i), "unknown"),
            })

    print(f"Kept records: {len(kept_records)}")
    print(f"Removed records: {len(records) - len(kept_records)}")

    # Write versioned benchmark
    versioned_path = output_dir / f"cfbench_{args.version}_full.jsonl"
    with open(versioned_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nVersioned benchmark: {versioned_path}")

    # Write ID mapping
    mapping_path = output_dir / f"cfbench_{args.version}_id_mapping.jsonl"
    with open(mapping_path, "w", encoding="utf-8") as f:
        for m in id_mapping:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"ID mapping: {mapping_path}")

    # Compute distribution
    task_types = Counter(r.get("task_type", "unknown") for r in kept_records)
    difficulties = Counter(r.get("difficulty", "unknown") for r in kept_records)
    modalities = Counter(r.get("modality", "text") for r in kept_records)
    sources = Counter(r.get("source", "unknown") for r in kept_records)

    # Write manifest
    manifest = {
        "version": args.version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_benchmark": bench_path,
        "audit_dir": str(audit_dir),
        "original_count": len(records),
        "kept_count": len(kept_records),
        "removed_count": len(records) - len(kept_records),
        "task_type_distribution": dict(task_types.most_common()),
        "difficulty_distribution": dict(difficulties),
        "modality_distribution": dict(modalities),
        "source_distribution": dict(sources.most_common()),
        "versioned_path": str(versioned_path),
        "id_mapping_path": str(mapping_path),
    }
    manifest_path = output_dir / f"cfbench_{args.version}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest: {manifest_path}")

    # Write changelog
    changelog = {
        "version": args.version,
        "previous_version": "v1",
        "changes": [
            f"Removed {len(records) - len(kept_records)} exact normalized duplicate items",
            f"Reassigned benchmark IDs to {args.version} format",
            "Added version field to each record",
            "Preserved original_id for traceability",
        ],
        "distribution": {
            "task_types": dict(task_types.most_common()),
            "difficulties": dict(difficulties),
            "modalities": dict(modalities),
        },
    }
    changelog_path = output_dir / f"cfbench_{args.version}_changelog.json"
    with open(changelog_path, "w", encoding="utf-8") as f:
        json.dump(changelog, f, ensure_ascii=False, indent=2)
    print(f"Changelog: {changelog_path}")

    # Print distribution
    print(f"\nDistribution ({args.version}):")
    print(f"  Task types: {dict(task_types.most_common())}")
    print(f"  Difficulties: {dict(difficulties)}")
    print(f"  Modalities: {dict(modalities)}")
    print(f"  Sources: {dict(sources.most_common())}")


if __name__ == "__main__":
    main()
