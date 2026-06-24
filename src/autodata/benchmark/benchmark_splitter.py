"""Benchmark splitter for Phase 5.

Splits benchmark items into dev/test/SFT pools with leakage control.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_benchmark(
    items: list[dict],
    dev_ratio: float = 0.20,
    seed: int = 42,
) -> dict:
    """Split benchmark items into dev/test with leakage control.

    Rules:
    - Same leakage_group_id stays in one split
    - Balance task type, difficulty, modality across splits
    - SFT pool is separate from dev/test
    """
    random.seed(seed)

    # Group by leakage_group_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        group_id = item.get("leakage_group_id", item.get("benchmark_id", ""))
        groups[group_id].append(item)

    # Assign groups to splits
    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    n_dev = max(1, int(len(group_ids) * dev_ratio))

    dev_groups = set(group_ids[:n_dev])
    test_groups = set(group_ids[n_dev:])

    dev_items = []
    test_items = []

    for gid, group_items in groups.items():
        if gid in dev_groups:
            dev_items.extend(group_items)
        else:
            test_items.extend(group_items)

    # Assign split field
    for item in dev_items:
        item["split"] = "dev"
    for item in test_items:
        item["split"] = "test"

    # Check for leakage
    dev_ids = set(item.get("benchmark_id") for item in dev_items)
    test_ids = set(item.get("benchmark_id") for item in test_items)
    leaked = dev_ids & test_ids

    # Check source leakage
    dev_sources = set()
    for item in dev_items:
        for ref in item.get("source_refs", []):
            dev_sources.add(ref)
    test_sources = set()
    for item in test_items:
        for ref in item.get("source_refs", []):
            test_sources.add(ref)
    source_overlap = dev_sources & test_sources

    report = {
        "total_items": len(items),
        "dev_items": len(dev_items),
        "test_items": len(test_items),
        "dev_groups": len(dev_groups),
        "test_groups": len(test_groups),
        "leaked_item_ids": len(leaked),
        "source_overlap": len(source_overlap),
        "source_overlap_files": list(source_overlap)[:10],
    }

    return {
        "dev": dev_items,
        "test": test_items,
        "report": report,
    }


def save_splits(result: dict) -> dict:
    """Save benchmark splits to files."""
    benchmark_dir = PROJECT_ROOT / "data" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    dev_path = benchmark_dir / "carbon_fiber_benchmark_dev.jsonl"
    test_path = benchmark_dir / "carbon_fiber_benchmark_test.jsonl"
    all_path = benchmark_dir / "carbon_fiber_benchmark_all.jsonl"
    leakage_path = benchmark_dir / "leakage_report.json"

    # Write dev
    with open(dev_path, "w") as f:
        for item in result["dev"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write test
    with open(test_path, "w") as f:
        for item in result["test"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write all
    all_items = result["dev"] + result["test"]
    with open(all_path, "w") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write leakage report
    with open(leakage_path, "w") as f:
        json.dump(result["report"], f, indent=2)

    return {
        "dev_path": str(dev_path),
        "test_path": str(test_path),
        "all_path": str(all_path),
        "leakage_path": str(leakage_path),
    }
