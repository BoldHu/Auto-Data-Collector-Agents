"""Benchmark subset builder for Phase 5.5.

Constructs official benchmark subsets for fair evaluation.
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


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_subsets() -> dict:
    """Build all official benchmark subsets."""
    all_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_all.jsonl"
    dev_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_dev.jsonl"
    test_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"

    all_items = load_jsonl(all_path)
    dev_items = load_jsonl(dev_path)
    test_items = load_jsonl(test_path)

    subsets_dir = PROJECT_ROOT / "data" / "benchmark" / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    # 1. CFBench-Text: text-only items
    text_items = [i for i in all_items if i.get("modality") == "text"]
    text_dev = [i for i in dev_items if i.get("modality") == "text"]
    text_test = [i for i in test_items if i.get("modality") == "text"]
    save_jsonl(text_dev, subsets_dir / "cfbench_text_dev.jsonl")
    save_jsonl(text_test, subsets_dir / "cfbench_text_test.jsonl")
    report["cfbench_text"] = {"dev": len(text_dev), "test": len(text_test), "total": len(text_items)}

    # 2. CFBench-MM: multimodal items
    mm_items = [i for i in all_items if i.get("modality") == "multimodal"]
    mm_dev = [i for i in dev_items if i.get("modality") == "multimodal"]
    mm_test = [i for i in test_items if i.get("modality") == "multimodal"]
    save_jsonl(mm_dev, subsets_dir / "cfbench_mm_dev.jsonl")
    save_jsonl(mm_test, subsets_dir / "cfbench_mm_test.jsonl")
    report["cfbench_mm"] = {"dev": len(mm_dev), "test": len(mm_test), "total": len(mm_items)}

    # 3. CFBench-Exam: exam-derived items
    exam_items = [i for i in all_items if i.get("source_type") == "exam"]
    exam_dev = [i for i in dev_items if i.get("source_type") == "exam"]
    exam_test = [i for i in test_items if i.get("source_type") == "exam"]
    save_jsonl(exam_dev, subsets_dir / "cfbench_exam_dev.jsonl")
    save_jsonl(exam_test, subsets_dir / "cfbench_exam_test.jsonl")
    report["cfbench_exam"] = {"dev": len(exam_dev), "test": len(exam_test), "total": len(exam_items)}

    # 4. CFBench-Hard: hard difficulty items
    hard_items = [i for i in all_items if i.get("difficulty") == "hard"]
    hard_test = [i for i in test_items if i.get("difficulty") == "hard"]
    save_jsonl(hard_test, subsets_dir / "cfbench_hard_test.jsonl")
    report["cfbench_hard"] = {"test": len(hard_test), "total": len(hard_items)}

    # 5. CFBench-AgentTask: agent task items (will be populated after generation)
    agent_path = PROJECT_ROOT / "data" / "benchmark_candidates" / "agent_task" / "agent_task_candidates_validated.jsonl"
    if agent_path.exists():
        agent_items = load_jsonl(agent_path)
        save_jsonl(agent_items, subsets_dir / "cfbench_agenttask_test.jsonl")
        report["cfbench_agenttask"] = {"test": len(agent_items)}
    else:
        report["cfbench_agenttask"] = {"test": 0, "note": "pending generation"}

    # 6. CFBench-Core: balanced subset
    random.seed(42)
    core_items = []
    task_buckets = defaultdict(list)
    for item in all_items:
        task_buckets[item.get("task_type", "unknown")].append(item)

    target_per_task = 75  # ~20 types * 75 = 1500
    for task, bucket in task_buckets.items():
        sampled = random.sample(bucket, min(target_per_task, len(bucket)))
        core_items.extend(sampled)

    # Split core into dev/test
    random.shuffle(core_items)
    n_dev = max(20, int(len(core_items) * 0.20))
    core_dev = core_items[:n_dev]
    core_test = core_items[n_dev:]
    for item in core_dev:
        item["split"] = "dev"
    for item in core_test:
        item["split"] = "test"

    save_jsonl(core_dev, subsets_dir / "cfbench_core_dev.jsonl")
    save_jsonl(core_test, subsets_dir / "cfbench_core_test.jsonl")
    report["cfbench_core"] = {"dev": len(core_dev), "test": len(core_test), "total": len(core_items)}

    # 7. CFBench-Full: all items
    save_jsonl(dev_items, subsets_dir / "cfbench_full_dev.jsonl")
    save_jsonl(test_items, subsets_dir / "cfbench_full_test.jsonl")
    report["cfbench_full"] = {"dev": len(dev_items), "test": len(test_items), "total": len(all_items)}

    return report


def save_subset_report(report: dict) -> tuple[Path, Path]:
    """Save subset report."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_5_benchmark_refinement"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "subset_report.json"
    md_path = report_dir / "subset_report.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# Phase 5.5 基准子集报告\n\n")
        f.write("| 子集 | Dev | Test | 总计 |\n|------|-----|------|------|\n")
        for name, data in report.items():
            f.write(f"| {name} | {data.get('dev', '-')} | {data.get('test', '-')} | {data.get('total', '-')} |\n")

    return json_path, md_path
