"""Benchmark source pool auditor for Phase 5.

Audits all candidate source pools before final benchmark construction.
"""

from __future__ import annotations

import json
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


def check_jsonl_valid(path: Path) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    json.loads(line)
        return True
    except Exception:
        return False


def check_provenance(records: list[dict], required_fields: list[str]) -> tuple[bool, list[str]]:
    missing = []
    for i, r in enumerate(records[:100]):
        for field in required_fields:
            if field not in r or not r[field]:
                missing.append(f"record {i}: missing {field}")
                break
    return len(missing) == 0, missing[:10]


def check_duplicate_ids(records: list[dict], id_field: str) -> tuple[bool, int]:
    ids = [r.get(id_field, "") for r in records]
    dup_count = len(ids) - len(set(ids))
    return dup_count == 0, dup_count


def audit_source_pools() -> dict:
    """Audit all source pools."""
    report = {
        "phase": "phase_5_source_audit",
        "pools": {},
        "findings": [],
        "warnings": [],
    }

    # 1. Exam pool
    exam_ready = load_jsonl(PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_benchmark_ready_candidates.jsonl")
    exam_unique = load_jsonl(PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_unique.jsonl")
    exam_validated = load_jsonl(PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_validated.jsonl")

    exam_ids = [r.get("question_id", "") for r in exam_ready]
    exam_dup_count = len(exam_ids) - len(set(exam_ids))

    report["pools"]["exam"] = {
        "ready_count": len(exam_ready),
        "unique_count": len(exam_unique),
        "validated_count": len(exam_validated),
        "duplicate_ids": exam_dup_count,
        "has_answer": sum(1 for r in exam_ready if r.get("answer")),
        "has_source_file": sum(1 for r in exam_ready if r.get("source_file")),
    }

    # 2. Multimodal pool
    mm_candidates = load_jsonl(PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_validated_full_normalized.jsonl")
    mm_validation = load_jsonl(PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_candidate_validation_full.jsonl")

    mm_passed = [r for r in mm_candidates if r.get("validation_status") == "passed"]
    mm_failed = [r for r in mm_candidates if r.get("validation_status") == "failed"]
    mm_no_val = [r for r in mm_candidates if r.get("validation_status") == "no_validation"]

    mm_ids = [r.get("candidate_id", "") for r in mm_candidates]
    mm_dup_count = len(mm_ids) - len(set(mm_ids))

    # Check image_id linkage
    image_ids = set()
    image_labels_path = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_labels_full.jsonl"
    if image_labels_path.exists():
        with open(image_labels_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    image_ids.add(r.get("image_id", ""))

    mm_linked = sum(1 for r in mm_candidates if r.get("image_id") in image_ids)

    report["pools"]["multimodal"] = {
        "total_candidates": len(mm_candidates),
        "passed": len(mm_passed),
        "failed": len(mm_failed),
        "no_validation": len(mm_no_val),
        "duplicate_ids": mm_dup_count,
        "linked_to_image": mm_linked,
        "task_type_distribution": dict(Counter(r.get("task_type", "unknown") for r in mm_passed).most_common()),
        "difficulty_distribution": dict(Counter(r.get("difficulty", "unknown") for r in mm_passed).most_common()),
    }

    # 3. Text pool (knowledge units + SFT candidates)
    ku_path = PROJECT_ROOT / "data" / "processed" / "knowledge_units" / "knowledge_units_pilot.jsonl"
    sft_path = PROJECT_ROOT / "data" / "processed" / "sft_candidates" / "sft_candidates_pilot.jsonl"

    ku_records = load_jsonl(ku_path)
    sft_records = load_jsonl(sft_path)

    report["pools"]["text"] = {
        "knowledge_units": len(ku_records),
        "sft_candidates": len(sft_records),
        "ku_types": dict(Counter(r.get("knowledge_type", "unknown") for r in ku_records).most_common()),
        "sft_types": dict(Counter(r.get("task_type", "unknown") for r in sft_records).most_common()),
    }

    # 4. Findings
    if exam_dup_count > 0:
        report["findings"].append(f"Exam pool has {exam_dup_count} duplicate IDs")

    if mm_dup_count > 0:
        report["findings"].append(f"Multimodal pool has {mm_dup_count} duplicate IDs")

    if len(mm_no_val) > 0:
        report["findings"].append(f"{len(mm_no_val)} multimodal candidates lack validation")

    if len(mm_failed) > 0:
        report["findings"].append(f"{len(mm_failed)} multimodal candidates failed validation")

    if mm_linked < len(mm_candidates):
        report["warnings"].append(f"{len(mm_candidates) - mm_linked} multimodal candidates have unlinked image_id")

    # 5. Check API keys in outputs
    api_key_leak = False
    for path in [
        PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_benchmark_ready_candidates.jsonl",
        PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_validated_full_normalized.jsonl",
    ]:
        if path.exists():
            with open(path) as f:
                content = f.read(10000)
                if "tp-" in content or "API_KEY" in content:
                    api_key_leak = True
                    report["findings"].append(f"API key leak detected in {path.name}")

    if not api_key_leak:
        report["findings"].append("No API key leaks detected")

    # Summary
    report["summary"] = {
        "total_exam_ready": len(exam_ready),
        "total_mm_passed": len(mm_passed),
        "total_ku": len(ku_records),
        "total_sft": len(sft_records),
        "total_benchmark_pool": len(exam_ready) + len(mm_passed) + len(ku_records) + len(sft_records),
    }

    return report


def save_audit_report(report: dict) -> tuple[Path, Path]:
    """Save audit report as JSON and MD."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_benchmark_construction"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "source_pool_audit.json"
    md_path = report_dir / "source_pool_audit.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# Phase 5 源池审计报告\n\n")

        f.write("## 源池统计\n\n")
        f.write("| 源池 | 数量 |\n|------|------|\n")
        for pool_name, pool_data in report["pools"].items():
            if isinstance(pool_data, dict):
                for k, v in pool_data.items():
                    if isinstance(v, int):
                        f.write(f"| {pool_name}.{k} | {v} |\n")

        f.write("\n## 总计\n\n")
        for k, v in report["summary"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## 发现\n\n")
        for finding in report["findings"]:
            f.write(f"- {finding}\n")

        if report["warnings"]:
            f.write("\n## 警告\n\n")
            for warning in report["warnings"]:
                f.write(f"- {warning}\n")

    return json_path, md_path
