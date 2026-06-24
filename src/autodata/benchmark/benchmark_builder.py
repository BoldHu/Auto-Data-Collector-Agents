"""Benchmark builder for Phase 5.

Assembles all source pools into a unified benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.autodata.benchmark.benchmark_schema import BenchmarkItem
from src.autodata.benchmark.benchmark_taxonomy import (
    map_exam_task_type,
    map_multimodal_task_type,
    MAX_ITEMS_PER_TASK,
)

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


def convert_exam_to_benchmark(exam: dict) -> BenchmarkItem:
    """Convert an exam question to a BenchmarkItem."""
    task_type = map_exam_task_type(exam.get("question_type", "unknown"))

    return BenchmarkItem(
        benchmark_id=BenchmarkItem.generate_id("exam", exam.get("question_id", "")),
        source_type="exam",
        task_type=task_type,
        modality="text",
        question=exam.get("question_text", ""),
        options=exam.get("options", []),
        answer=exam.get("answer", ""),
        explanation=exam.get("explanation", ""),
        evidence=[exam.get("raw_evidence", "")],
        source_refs=[exam.get("source_file", "")],
        required_knowledge=exam.get("knowledge_points", []),
        reasoning_type=["exam_reasoning"],
        difficulty=exam.get("difficulty", "medium"),
        quality_scores={
            "clarity": exam.get("clarity"),
            "completeness": exam.get("completeness"),
            "answerability": exam.get("answerability"),
            "domain_relevance": exam.get("domain_relevance"),
        },
        leakage_group_id=exam.get("source_file", ""),
        validation_status="passed",
    )


def convert_multimodal_to_benchmark(mm: dict) -> BenchmarkItem:
    """Convert a multimodal candidate to a BenchmarkItem."""
    task_type = map_multimodal_task_type(mm.get("task_type", "unknown"))

    return BenchmarkItem(
        benchmark_id=BenchmarkItem.generate_id("multimodal", mm.get("candidate_id", "")),
        source_type="multimodal",
        task_type=task_type,
        modality="multimodal",
        question=mm.get("question", ""),
        options=mm.get("options", []),
        answer=mm.get("answer", ""),
        explanation=mm.get("explanation", ""),
        evidence=mm.get("visual_evidence", []),
        source_refs=mm.get("source_refs", []),
        image_refs=[mm.get("image_id", "")],
        required_knowledge=mm.get("required_knowledge", []),
        reasoning_type=mm.get("reasoning_steps", []),
        difficulty=mm.get("difficulty", "medium"),
        quality_scores={
            "answerability": mm.get("answerability_score"),
            "hallucination_risk": mm.get("hallucination_risk"),
        },
        leakage_group_id=mm.get("image_id", ""),
        validation_status=mm.get("validation_status", "passed"),
    )


def convert_ku_to_benchmark(ku: dict) -> BenchmarkItem:
    """Convert a knowledge unit to a BenchmarkItem."""
    question = f"关于{ku.get('topic', '碳纤维')}的知识：{ku.get('claim', '')}"
    answer = ku.get("evidence_text", ku.get("claim", ""))

    return BenchmarkItem(
        benchmark_id=BenchmarkItem.generate_id("ku", ku.get("unit_id", "")),
        source_type="text",
        task_type="domain_knowledge_qa",
        modality="text",
        question=question,
        answer=answer,
        evidence=[ku.get("evidence_text", "")],
        source_refs=[ref if isinstance(ref, str) else ref.get("source_file", "") for ref in ku.get("source_refs", [])],
        required_knowledge=[e.get("name", "") if isinstance(e, dict) else str(e) for e in ku.get("entities", [])],
        reasoning_type=["knowledge_retrieval"],
        difficulty="medium",
        quality_scores={
            "domain_relevance": ku.get("quality_score"),
        },
        leakage_group_id=ku.get("source_chunk_id", ""),
        validation_status="passed",
    )


def convert_sft_to_benchmark(sft: dict) -> BenchmarkItem:
    """Convert an SFT candidate to a BenchmarkItem."""
    return BenchmarkItem(
        benchmark_id=BenchmarkItem.generate_id("sft", sft.get("sample_id", "")),
        source_type="text",
        task_type=sft.get("task_type", "domain_knowledge_qa"),
        modality="text",
        question=sft.get("instruction", ""),
        answer=sft.get("output", ""),
        evidence=[sft.get("evidence_text", "")],
        source_refs=[ref if isinstance(ref, str) else ref.get("source_file", "") for ref in sft.get("source_refs", [])],
        reasoning_type=[sft.get("task_type", "reasoning")],
        difficulty=sft.get("difficulty", "medium"),
        quality_scores={
            "domain_relevance": sft.get("quality_score"),
        },
        leakage_group_id=sft.get("source_chunk_id", ""),
        validation_status="passed",
    )


def build_benchmark() -> dict:
    """Build the final benchmark from all source pools.

    Returns:
        Build report dict.
    """
    # Load source pools
    exam_path = PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_benchmark_ready_candidates.jsonl"
    mm_path = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_final_pool.jsonl"
    ku_path = PROJECT_ROOT / "data" / "processed" / "knowledge_units" / "knowledge_units_pilot.jsonl"
    sft_path = PROJECT_ROOT / "data" / "processed" / "sft_candidates" / "sft_candidates_pilot.jsonl"

    exam_items = load_jsonl(exam_path)
    mm_items = load_jsonl(mm_path)
    ku_items = load_jsonl(ku_path)
    sft_items = load_jsonl(sft_path)

    # Convert to BenchmarkItems
    all_items = []

    for exam in exam_items:
        all_items.append(convert_exam_to_benchmark(exam))

    for mm in mm_items:
        all_items.append(convert_multimodal_to_benchmark(mm))

    for ku in ku_items:
        all_items.append(convert_ku_to_benchmark(ku))

    for sft in sft_items:
        all_items.append(convert_sft_to_benchmark(sft))

    # Balance by task type
    from collections import Counter
    task_counts = Counter(item.task_type for item in all_items)

    # Cap per task type
    balanced_items = []
    task_buckets: dict[str, list] = {}
    for item in all_items:
        task = item.task_type
        if task not in task_buckets:
            task_buckets[task] = []
        task_buckets[task].append(item)

    for task, bucket in task_buckets.items():
        balanced_items.extend(bucket[:MAX_ITEMS_PER_TASK])

    # Write all candidates
    output_dir = PROJECT_ROOT / "data" / "benchmark" / "final_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "benchmark_candidates_all.jsonl"
    with open(all_path, "w") as f:
        for item in balanced_items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    report = {
        "total_exam": len(exam_items),
        "total_mm": len(mm_items),
        "total_ku": len(ku_items),
        "total_sft": len(sft_items),
        "total_converted": len(all_items),
        "total_balanced": len(balanced_items),
        "task_distribution": dict(task_counts),
        "output_path": str(all_path),
    }

    return report


def save_build_report(report: dict) -> Path:
    """Save build report."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_benchmark_construction"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "benchmark_build_report.json"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return json_path
