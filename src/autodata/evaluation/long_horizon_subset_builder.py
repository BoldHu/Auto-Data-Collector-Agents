"""Long-horizon stress-test subset builder for Phase 6.7.

Builds three subsets:
1. Standard: comparable to Phase 6.6
2. Long-context: more evidence, longer source context
3. Stress: multi-step planning, evidence selection, constraint satisfaction
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
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


def build_stress_subsets(seed: int = 42) -> dict:
    """Build three ablation subsets.

    Returns:
        Dict with 'standard', 'long_context', 'stress' item lists and statistics.
    """
    random.seed(seed)

    agenttask = load_jsonl(PROJECT_ROOT / "data" / "benchmark" / "subsets" / "cfbench_agenttask_test.jsonl")
    text = load_jsonl(PROJECT_ROOT / "data" / "benchmark" / "subsets" / "cfbench_text_test.jsonl")
    core = load_jsonl(PROJECT_ROOT / "data" / "benchmark" / "subsets" / "cfbench_core_test.jsonl")
    hard = load_jsonl(PROJECT_ROOT / "data" / "benchmark" / "subsets" / "cfbench_hard_test.jsonl")

    # ── Standard subset (100 items) ──────────────────────────────
    standard = []
    standard.extend(agenttask[:50])
    # Add some text items
    text_items = [t for t in text if t.get("modality") == "text"]
    standard.extend(text_items[:30])
    # Add some core items
    core_text = [c for c in core if c.get("modality") == "text"]
    standard.extend(core_text[:20])
    random.shuffle(standard)
    standard = standard[:100]

    # Add metadata
    for item in standard:
        item["_subset"] = "standard"
        item["_context_length_bucket"] = "medium"
        item["_num_distractors"] = 0

    # ── Long-context subset (100 items) ──────────────────────────
    # Items with longer evidence, more distractors
    long_candidates = []
    for item in agenttask:
        evidence_len = sum(len(str(e)) for e in item.get("evidence", []))
        if evidence_len > 500:  # has substantial evidence
            long_candidates.append(item)

    # Also get text items with explanations
    for item in text:
        if item.get("explanation") and len(item.get("explanation", "")) > 200:
            long_candidates.append(item)

    # Get hard items
    for item in hard:
        if item.get("modality") == "text":
            long_candidates.append(item)

    random.shuffle(long_candidates)
    long_context = long_candidates[:100]

    # Add distractors from other items
    distractor_pool = [c for c in core if c.get("modality") == "text"]
    for item in long_context:
        # Add 2-5 distractor contexts
        num_distractors = random.randint(2, 5)
        distractors = random.sample(distractor_pool, min(num_distractors, len(distractor_pool)))
        distractor_texts = [d.get("question", "")[:200] for d in distractors]
        item["_distractors"] = distractor_texts
        item["_num_distractors"] = len(distractors)
        item["_subset"] = "long_context"
        item["_context_length_bucket"] = "long" if len(distractors) <= 3 else "very_long"

    # ── Stress subset (150 items) ────────────────────────────────
    # Items requiring multi-step planning, evidence selection, constraint satisfaction
    stress_candidates = []

    # All agent tasks (require planning/coordination)
    for item in agenttask:
        item["_stress_type"] = "agent_task"
        stress_candidates.append(item)

    # Process reasoning from core
    for item in core:
        if "process" in item.get("task_type", "").lower():
            item["_stress_type"] = "process_reasoning"
            stress_candidates.append(item)

    # Constraint satisfaction
    for item in core:
        if "constraint" in item.get("task_type", "").lower():
            item["_stress_type"] = "constraint_satisfaction"
            stress_candidates.append(item)

    # Source-grounded reasoning
    for item in text:
        if "source_grounded" in item.get("task_type", ""):
            item["_stress_type"] = "source_grounded"
            stress_candidates.append(item)

    # Hard items
    for item in hard:
        if item.get("modality") == "text":
            item["_stress_type"] = "hard_text"
            stress_candidates.append(item)

    random.shuffle(stress_candidates)
    stress = stress_candidates[:150]

    for item in stress:
        item["_subset"] = "stress"
        item["_context_length_bucket"] = "long"
        item["_num_distractors"] = random.randint(1, 3)

    # Statistics
    stats = {
        "standard": {
            "count": len(standard),
            "task_dist": dict(Counter(i.get("task_type", "unknown") for i in standard).most_common()),
            "diff_dist": dict(Counter(i.get("difficulty", "unknown") for i in standard).most_common()),
        },
        "long_context": {
            "count": len(long_context),
            "task_dist": dict(Counter(i.get("task_type", "unknown") for i in long_context).most_common()),
            "diff_dist": dict(Counter(i.get("difficulty", "unknown") for i in long_context).most_common()),
            "avg_distractors": sum(i.get("_num_distractors", 0) for i in long_context) / len(long_context) if long_context else 0,
        },
        "stress": {
            "count": len(stress),
            "stress_type_dist": dict(Counter(i.get("_stress_type", "unknown") for i in stress).most_common()),
            "task_dist": dict(Counter(i.get("task_type", "unknown") for i in stress).most_common()),
            "diff_dist": dict(Counter(i.get("difficulty", "unknown") for i in stress).most_common()),
        },
    }

    return {"standard": standard, "long_context": long_context, "stress": stress, "stats": stats}


def save_subsets(subsets: dict) -> dict:
    """Save subsets to files."""
    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name in ["standard", "long_context", "stress"]:
        path = output_dir / f"ablation_subset_{name}.jsonl"
        with open(path, "w") as f:
            for item in subsets[name]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        paths[name] = str(path)

    # Save statistics
    stats_path = PROJECT_ROOT / "data" / "reports" / "phase_6_7_ablation_robustness"
    stats_path.mkdir(parents=True, exist_ok=True)
    with open(stats_path / "stress_subset_statistics.json", "w") as f:
        json.dump(subsets["stats"], f, indent=2, ensure_ascii=False)

    return paths
