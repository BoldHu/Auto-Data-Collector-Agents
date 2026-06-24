"""Long-context subset expander for Phase 6.8.

Creates expanded long-context items by adding distractor evidence.
"""

from __future__ import annotations

import json
import random
from collections import Counter
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


def expand_long_context_subset(
    target_count: int = 100,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Build expanded long-context subset.

    Takes agent-task and text items, adds distractor contexts.

    Returns:
        (items, statistics)
    """
    random.seed(seed)

    # Load source items
    agenttask = load_jsonl(PROJECT_ROOT / "data" / "benchmark" / "subsets" / "cfbench_agenttask_test.jsonl")
    text = load_jsonl(PROJECT_ROOT / "data" / "benchmark" / "subsets" / "cfbench_text_test.jsonl")
    core = load_jsonl(PROJECT_ROOT / "data" / "benchmark" / "subsets" / "cfbench_core_test.jsonl")

    # Load distractor pool from knowledge units and pretraining corpus
    distractor_pool = []
    ku_path = PROJECT_ROOT / "data" / "processed" / "knowledge_units" / "knowledge_units_pilot.jsonl"
    if ku_path.exists():
        for item in load_jsonl(ku_path):
            distractor_pool.append(item.get("claim", "") + " " + item.get("evidence_text", ""))

    corpus_path = PROJECT_ROOT / "data" / "processed" / "pretraining_corpus" / "pretraining_corpus_reclean.jsonl"
    if corpus_path.exists():
        for item in load_jsonl(corpus_path)[:500]:
            distractor_pool.append(item.get("text", "")[:500])

    # Filter distractor pool to non-empty
    distractor_pool = [d for d in distractor_pool if d and len(d) > 50]

    # Select candidate items
    candidates = []

    # Agent tasks (best for long-context)
    for item in agenttask:
        item["_source_subset"] = "agenttask"
        candidates.append(item)

    # Text items with evidence
    for item in text:
        if item.get("evidence") and len(str(item.get("evidence", ""))) > 100:
            item["_source_subset"] = "text"
            candidates.append(item)

    # Core items with explanations
    for item in core:
        if item.get("explanation") and len(item.get("explanation", "")) > 200:
            item["_source_subset"] = "core"
            candidates.append(item)

    random.shuffle(candidates)
    selected = candidates[:target_count]

    # Expand each item with distractors
    expanded = []
    for item in selected:
        # Determine number of distractors based on context length bucket
        num_distractors = random.randint(3, 8)

        # Sample distractors
        distractors = random.sample(distractor_pool, min(num_distractors, len(distractor_pool)))

        # Build expanded item
        expanded_item = dict(item)
        expanded_item["benchmark_id"] = f"lc_{item.get('benchmark_id', '')}"
        expanded_item["original_benchmark_id"] = item.get("benchmark_id", "")

        # Merge evidence with distractors
        original_evidence = item.get("evidence", [])
        if isinstance(original_evidence, str):
            original_evidence = [original_evidence]

        expanded_item["evidence"] = original_evidence
        expanded_item["distractors"] = distractors
        expanded_item["constraints"] = item.get("constraints", item.get("required_knowledge", []))

        # Determine context length bucket
        total_context = sum(len(str(e)) for e in original_evidence) + sum(len(d) for d in distractors)
        if total_context > 10000:
            expanded_item["context_length_bucket"] = "very_long"
        elif total_context > 5000:
            expanded_item["context_length_bucket"] = "long"
        else:
            expanded_item["context_length_bucket"] = "medium"

        expanded_item["expected_reasoning_steps"] = item.get("reasoning_type", [])
        expanded_item["source_refs"] = item.get("source_refs", [])
        expanded_item["metadata"] = {
            "is_long_context_wrapper": True,
            "num_relevant_evidence": len(original_evidence),
            "num_distractors": len(distractors),
            "total_context_chars": total_context,
        }

        expanded.append(expanded_item)

    # Statistics
    stats = {
        "total_items": len(expanded),
        "source_distribution": dict(Counter(i.get("_source_subset", "unknown") for i in expanded).most_common()),
        "context_length_distribution": dict(Counter(i.get("context_length_bucket", "unknown") for i in expanded).most_common()),
        "avg_distractors": sum(len(i.get("distractors", [])) for i in expanded) / len(expanded) if expanded else 0,
        "avg_evidence": sum(len(i.get("evidence", [])) for i in expanded) / len(expanded) if expanded else 0,
    }

    return expanded, stats


def save_expanded_subset(items: list[dict], stats: dict) -> tuple[Path, Path, Path]:
    """Save expanded subset and statistics."""
    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_8"
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = output_dir / "long_context_expanded.jsonl"
    stats_json = PROJECT_ROOT / "data" / "reports" / "phase_6_8_evidence_consolidation" / "long_context_expansion_statistics.json"
    stats_md = PROJECT_ROOT / "data" / "reports" / "phase_6_8_evidence_consolidation" / "long_context_expansion_statistics.md"

    stats_json.parent.mkdir(parents=True, exist_ok=True)

    with open(subset_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with open(stats_md, "w") as f:
        f.write("# Long-Context Expansion Statistics\n\n")
        f.write(f"Total items: {stats['total_items']}\n\n")
        f.write("## Source Distribution\n\n| Source | Count |\n|--------|-------|\n")
        for k, v in stats["source_distribution"].items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Context Length Distribution\n\n| Bucket | Count |\n|--------|-------|\n")
        for k, v in stats["context_length_distribution"].items():
            f.write(f"| {k} | {v} |\n")
        f.write(f"\n- Avg distractors per item: {stats['avg_distractors']:.1f}\n")
        f.write(f"- Avg evidence per item: {stats['avg_evidence']:.1f}\n")

    return subset_path, stats_json, stats_md
