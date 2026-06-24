"""Ablation subset builder for Phase 6.6.

Selects ~100 items from benchmark subsets emphasizing data construction tasks.
"""

from __future__ import annotations

import json
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


def build_ablation_subset() -> tuple[list[dict], dict]:
    """Build ablation subset from benchmark files.

    Returns:
        (subset_items, statistics)
    """
    subsets_dir = PROJECT_ROOT / "data" / "benchmark" / "subsets"

    # Load source subsets
    agenttask = load_jsonl(subsets_dir / "cfbench_agenttask_test.jsonl")
    text = load_jsonl(subsets_dir / "cfbench_text_test.jsonl")
    core = load_jsonl(subsets_dir / "cfbench_core_test.jsonl")
    hard = load_jsonl(subsets_dir / "cfbench_hard_test.jsonl")

    subset = []

    # 1. All agent task items (up to 127)
    subset.extend(agenttask[:127])

    # 2. Source-grounded reasoning from text
    sgr = [t for t in text if "source_grounded" in t.get("task_type", "")]
    subset.extend(sgr[:50])

    # 3. Process reasoning from core
    pr = [c for c in core if "process" in c.get("task_type", "")]
    subset.extend(pr[:50])

    # 4. Constraint satisfaction
    cs = [c for c in core if "constraint" in c.get("task_type", "").lower()]
    subset.extend(cs[:30])

    # 5. Hard text items
    hard_text = [h for h in hard if h.get("modality") == "text"]
    subset.extend(hard_text[:50])

    # 6. Domain knowledge QA
    dk = [t for t in text if "domain_knowledge" in t.get("task_type", "")]
    subset.extend(dk[:30])

    # Deduplicate by benchmark_id
    seen = set()
    unique = []
    for item in subset:
        bid = item.get("benchmark_id", "")
        if bid not in seen:
            seen.add(bid)
            unique.append(item)

    # Statistics
    from collections import Counter
    task_dist = dict(Counter(i.get("task_type", "unknown") for i in unique).most_common())
    diff_dist = dict(Counter(i.get("difficulty", "unknown") for i in unique).most_common())
    src_dist = dict(Counter(i.get("source_type", "unknown") for i in unique).most_common())

    stats = {
        "total_items": len(unique),
        "source_subsets": {
            "cfbench_agenttask": len(agenttask),
            "cfbench_text": len(text),
            "cfbench_core": len(core),
            "cfbench_hard": len(hard),
        },
        "task_distribution": task_dist,
        "difficulty_distribution": diff_dist,
        "source_distribution": src_dist,
    }

    return unique, stats


def save_ablation_subset(items: list[dict], stats: dict) -> tuple[Path, Path, Path]:
    """Save ablation subset and statistics."""
    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_6"
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = output_dir / "ablation_subset.jsonl"
    stats_json_path = output_dir / "ablation_subset_statistics.json"
    stats_md_path = output_dir / "ablation_subset_statistics.md"

    with open(subset_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with open(stats_md_path, "w") as f:
        f.write("# Ablation Subset Statistics\n\n")
        f.write(f"Total items: {stats['total_items']}\n\n")
        f.write("## Source Subsets\n\n| Subset | Count |\n|--------|-------|\n")
        for k, v in stats["source_subsets"].items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Task Distribution\n\n| Task Type | Count |\n|-----------|-------|\n")
        for k, v in stats["task_distribution"].items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Difficulty Distribution\n\n| Difficulty | Count |\n|------------|-------|\n")
        for k, v in stats["difficulty_distribution"].items():
            f.write(f"| {k} | {v} |\n")

    return subset_path, stats_json_path, stats_md_path
