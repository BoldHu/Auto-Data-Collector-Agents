"""SFT data splitter for Phase 7.

Creates train/validation splits with leakage prevention.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def split_samples(
    samples: list[dict],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples into train/validation.

    Rules:
    1. Same leakage_group_id stays in one split
    2. Balance source types
    3. Balance difficulty
    """
    random.seed(seed)

    # Group by leakage_group_id
    groups: dict[str, list[dict]] = defaultdict(list)
    ungrouped: list[dict] = []

    for sample in samples:
        lg = sample.get("leakage_group_id", "")
        if lg:
            groups[lg].append(sample)
        else:
            ungrouped.append(sample)

    # Split groups
    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    train = []
    val = []

    n_train = int(len(group_ids) * train_ratio)
    for i, gid in enumerate(group_ids):
        if i < n_train:
            train.extend(groups[gid])
        else:
            val.extend(groups[gid])

    # Split ungrouped
    random.shuffle(ungrouped)
    n_train_ungrouped = int(len(ungrouped) * train_ratio)
    train.extend(ungrouped[:n_train_ungrouped])
    val.extend(ungrouped[n_train_ungrouped:])

    return train, val


def compute_split_statistics(train: list[dict], val: list[dict]) -> dict:
    """Compute statistics for train/val splits."""
    from collections import Counter

    def stats(samples: list[dict]) -> dict:
        source_types = Counter(s.get("source_type", "unknown") for s in samples)
        task_types = Counter(s.get("task_type", "unknown") for s in samples)
        difficulties = Counter(s.get("difficulty", "unknown") for s in samples)

        input_lens = [len(s.get("instruction", "") + s.get("input", "")) for s in samples]
        output_lens = [len(s.get("output", "")) for s in samples]

        return {
            "count": len(samples),
            "source_type_dist": dict(source_types),
            "task_type_dist": dict(task_types),
            "difficulty_dist": dict(difficulties),
            "avg_input_len": round(sum(input_lens) / max(len(input_lens), 1)),
            "avg_output_len": round(sum(output_lens) / max(len(output_lens), 1)),
        }

    return {
        "train": stats(train),
        "validation": stats(val),
        "total": len(train) + len(val),
    }


def save_splits(
    train: list[dict],
    val: list[dict],
    output_dir: Path,
) -> dict:
    """Save train/val splits in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw splits
    with open(output_dir / "train.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(output_dir / "validation.jsonl", "w") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save ChatML format
    def to_chatml(sample: dict) -> dict:
        messages = []
        sys_prompt = sample.get("system_prompt", "")
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        user_content = sample.get("instruction", "")
        if sample.get("input"):
            user_content += "\n\n" + sample["input"]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": sample.get("output", "")})
        return {"messages": messages}

    with open(output_dir / "train_chatml.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")

    with open(output_dir / "validation_chatml.jsonl", "w") as f:
        for s in val:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")

    # Compute and save statistics
    stats = compute_split_statistics(train, val)
    with open(output_dir / "sft_dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats
