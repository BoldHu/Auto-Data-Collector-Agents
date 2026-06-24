"""Artifact lineage auditor for Phase 6.55.

Verifies that outputs from one stage are used as inputs to later stages.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def audit_artifact_lineage() -> dict:
    """Audit artifact lineage across all phases."""
    report = {"lineages": {}, "summary": {}}

    # Text lineage
    text_lineage = {
        "raw_text": {
            "path": "text_raw_data/",
            "exists": (PROJECT_ROOT / "text_raw_data").exists(),
        },
        "cleaned_chunks": {
            "path": "data/processed/pretraining_corpus/",
            "exists": (PROJECT_ROOT / "data" / "processed" / "pretraining_corpus").exists(),
        },
        "knowledge_units": {
            "path": "data/processed/knowledge_units/",
            "exists": (PROJECT_ROOT / "data" / "processed" / "knowledge_units").exists(),
        },
        "sft_candidates": {
            "path": "data/processed/sft_candidates/",
            "exists": (PROJECT_ROOT / "data" / "processed" / "sft_candidates").exists(),
        },
        "text_quality": {
            "path": "data/processed/text_quality/",
            "exists": (PROJECT_ROOT / "data" / "processed" / "text_quality").exists(),
        },
    }

    # Check record counts
    for stage, info in text_lineage.items():
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            jsonl_files = list(p.glob("*.jsonl"))
            total_records = 0
            for f in jsonl_files:
                with open(f) as fh:
                    total_records += sum(1 for line in fh if line.strip())
            info["jsonl_files"] = len(jsonl_files)
            info["total_records"] = total_records
        else:
            info["jsonl_files"] = 0
            info["total_records"] = 0

    report["lineages"]["text"] = text_lineage

    # Image lineage
    image_lineage = {
        "raw_images": {
            "path": "imgs_raw_data/carbon_fiber_mm/",
            "exists": (PROJECT_ROOT / "imgs_raw_data" / "carbon_fiber_mm").exists(),
        },
        "image_manifest": {
            "path": "data/processed/image_corpus/image_unique_manifest.jsonl",
            "exists": (PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_unique_manifest.jsonl").exists(),
        },
        "image_labels": {
            "path": "data/processed/image_corpus/image_labels_full.jsonl",
            "exists": (PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_labels_full.jsonl").exists(),
        },
        "image_captions": {
            "path": "data/processed/image_corpus/image_captions_full.jsonl",
            "exists": (PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_captions_full.jsonl").exists(),
        },
        "mm_candidates": {
            "path": "data/benchmark_candidates/multimodal/mm_benchmark_candidates_full.jsonl",
            "exists": (PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_full.jsonl").exists(),
        },
        "mm_validation": {
            "path": "data/benchmark_candidates/multimodal/mm_candidate_validation_full.jsonl",
            "exists": (PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_candidate_validation_full.jsonl").exists(),
        },
    }

    for stage, info in image_lineage.items():
        p = PROJECT_ROOT / info["path"]
        if p.exists() and p.suffix == ".jsonl":
            with open(p) as f:
                info["records"] = sum(1 for line in f if line.strip())
        elif p.exists() and p.is_dir():
            info["records"] = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))

    report["lineages"]["image"] = image_lineage

    # Exam lineage
    exam_lineage = {
        "raw_exams": {
            "path": "exam_raw_data/",
            "exists": (PROJECT_ROOT / "exam_raw_data").exists(),
        },
        "text_blocks": {
            "path": "data/interim/exam_extracted_text/exam_text_blocks.jsonl",
            "exists": (PROJECT_ROOT / "data" / "interim" / "exam_extracted_text" / "exam_text_blocks.jsonl").exists(),
        },
        "raw_questions": {
            "path": "data/processed/exam_questions/exam_questions_raw.jsonl",
            "exists": (PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_raw.jsonl").exists(),
        },
        "validated_questions": {
            "path": "data/processed/exam_questions/exam_questions_validated.jsonl",
            "exists": (PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_validated.jsonl").exists(),
        },
        "benchmark_ready": {
            "path": "data/processed/exam_questions/exam_questions_benchmark_ready_candidates.jsonl",
            "exists": (PROJECT_ROOT / "data" / "processed" / "exam_questions" / "exam_questions_benchmark_ready_candidates.jsonl").exists(),
        },
    }

    for stage, info in exam_lineage.items():
        p = PROJECT_ROOT / info["path"]
        if p.exists() and p.suffix == ".jsonl":
            with open(p) as f:
                info["records"] = sum(1 for line in f if line.strip())
        elif p.exists() and p.is_dir():
            info["records"] = len(list(p.iterdir()))

    report["lineages"]["exam"] = exam_lineage

    # Benchmark lineage
    benchmark_lineage = {
        "all_candidates": {
            "path": "data/benchmark/final_candidates/benchmark_candidates_all.jsonl",
            "exists": (PROJECT_ROOT / "data" / "benchmark" / "final_candidates" / "benchmark_candidates_all.jsonl").exists(),
        },
        "benchmark_dev": {
            "path": "data/benchmark/carbon_fiber_benchmark_dev.jsonl",
            "exists": (PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_dev.jsonl").exists(),
        },
        "benchmark_test": {
            "path": "data/benchmark/carbon_fiber_benchmark_test.jsonl",
            "exists": (PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl").exists(),
        },
        "evaluation_predictions": {
            "path": "data/evaluation/phase_6_5/parsed_predictions/",
            "exists": (PROJECT_ROOT / "data" / "evaluation" / "phase_6_5" / "parsed_predictions").exists(),
        },
    }

    for stage, info in benchmark_lineage.items():
        p = PROJECT_ROOT / info["path"]
        if p.exists() and p.suffix == ".jsonl":
            with open(p) as f:
                info["records"] = sum(1 for line in f if line.strip())
        elif p.exists() and p.is_dir():
            jsonl_files = list(p.glob("*.jsonl"))
            total = 0
            for f in jsonl_files:
                with open(f) as fh:
                    total += sum(1 for line in fh if line.strip())
            info["records"] = total
            info["files"] = len(jsonl_files)

    report["lineages"]["benchmark"] = benchmark_lineage

    # Summary
    all_stages = []
    for lineage in report["lineages"].values():
        all_stages.extend(lineage.values())
    report["summary"] = {
        "total_stages": len(all_stages),
        "stages_with_data": sum(1 for s in all_stages if s.get("exists")),
        "stages_without_data": sum(1 for s in all_stages if not s.get("exists")),
    }

    return report
