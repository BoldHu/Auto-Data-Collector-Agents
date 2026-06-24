"""Build DTCG trace for Phase 4 exam extraction pipeline.

Usage:
    python scripts/phase_4_dtcg_trace.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_4_exam_extraction"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "exam_questions"
TRACE_PATH = REPORT_DIR / "dtcg_exam_extraction_trace.json"
PACKAGES_PATH = REPORT_DIR / "context_packages_exam_extraction.jsonl"


def load_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def build_trace():
    nodes = [
        {"id": "agent_inventory", "type": "agent", "name": "ExamInventoryAgent"},
        {"id": "agent_extraction", "type": "agent", "name": "ExamExtractionAgent"},
        {"id": "agent_quality", "type": "agent", "name": "ExamQualityAgent"},
        {"id": "agent_dedup", "type": "agent", "name": "ExamDedupAgent"},
        {"id": "task_conversion", "type": "task", "name": "Document Conversion", "status": "completed"},
        {"id": "task_extraction", "type": "task", "name": "Question Extraction", "status": "completed"},
        {"id": "task_quality", "type": "task", "name": "Quality Verification", "status": "completed"},
        {"id": "task_dedup", "type": "task", "name": "Deduplication", "status": "completed"},
        {"id": "artifact_raw_files", "type": "artifact", "name": "Raw exam files (22)"},
        {"id": "artifact_text_blocks", "type": "artifact", "name": "exam_text_blocks.jsonl"},
        {"id": "artifact_raw_questions", "type": "artifact", "name": "exam_questions_raw.jsonl"},
        {"id": "artifact_validated", "type": "artifact", "name": "exam_questions_validated.jsonl"},
        {"id": "artifact_quality_scores", "type": "artifact", "name": "exam_question_quality_scores.jsonl"},
        {"id": "artifact_duplicates", "type": "artifact", "name": "exam_question_duplicates.jsonl"},
        {"id": "artifact_unique", "type": "artifact", "name": "exam_questions_unique.jsonl"},
        {"id": "artifact_benchmark_ready", "type": "artifact", "name": "exam_questions_benchmark_ready_candidates.jsonl"},
        {"id": "tool_model_pool", "type": "tool", "name": "ModelPool (API_KEY1 only)"},
        {"id": "tool_doc_converter", "type": "tool", "name": "DocumentConverter"},
        {"id": "constraint_api_key1", "type": "constraint", "name": "Use API_KEY1 only"},
        {"id": "constraint_no_hallucinate", "type": "constraint", "name": "No hallucinated answers"},
    ]

    edges = [
        {"source": "agent_inventory", "target": "task_conversion", "type": "agent_assignment"},
        {"source": "agent_extraction", "target": "task_extraction", "type": "agent_assignment"},
        {"source": "agent_quality", "target": "task_quality", "type": "agent_assignment"},
        {"source": "agent_dedup", "target": "task_dedup", "type": "agent_assignment"},
        {"source": "task_conversion", "target": "artifact_text_blocks", "type": "artifact_derived_from"},
        {"source": "task_extraction", "target": "artifact_raw_questions", "type": "artifact_derived_from"},
        {"source": "task_quality", "target": "artifact_validated", "type": "artifact_derived_from"},
        {"source": "task_quality", "target": "artifact_quality_scores", "type": "artifact_derived_from"},
        {"source": "task_dedup", "target": "artifact_unique", "type": "artifact_derived_from"},
        {"source": "task_dedup", "target": "artifact_benchmark_ready", "type": "artifact_derived_from"},
        {"source": "artifact_raw_files", "target": "task_conversion", "type": "task_dependency"},
        {"source": "artifact_text_blocks", "target": "task_extraction", "type": "task_dependency"},
        {"source": "artifact_raw_questions", "target": "task_quality", "type": "task_dependency"},
        {"source": "artifact_validated", "target": "task_dedup", "type": "task_dependency"},
        {"source": "agent_extraction", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "agent_quality", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "task_conversion", "target": "tool_doc_converter", "type": "tool_usage"},
        {"source": "constraint_api_key1", "target": "agent_extraction", "type": "quality_feedback"},
        {"source": "constraint_no_hallucinate", "target": "agent_extraction", "type": "quality_feedback"},
    ]

    stats = {
        "text_blocks": load_jsonl_count(PROJECT_ROOT / "data" / "interim" / "exam_extracted_text" / "exam_text_blocks.jsonl"),
        "raw_questions": load_jsonl_count(OUTPUT_DIR / "exam_questions_raw.jsonl"),
        "validated": load_jsonl_count(OUTPUT_DIR / "exam_questions_validated.jsonl"),
        "quality_scores": load_jsonl_count(OUTPUT_DIR / "exam_question_quality_scores.jsonl"),
        "unique": load_jsonl_count(OUTPUT_DIR / "exam_questions_unique.jsonl"),
        "benchmark_ready": load_jsonl_count(OUTPUT_DIR / "exam_questions_benchmark_ready_candidates.jsonl"),
    }

    trace = {
        "phase": "phase_4_exam_extraction",
        "timestamp": time.time(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "statistics": stats,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRACE_PATH, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    # Context packages
    packages = []
    for task in [n for n in nodes if n["type"] == "task"]:
        pkg = {
            "agent_name": "ExamExtractionPipeline",
            "task_id": task["id"],
            "current_goal": task["name"],
            "allowed_tools": ["DocumentConverter", "ModelPool"],
            "relevant_plan": "Phase 4 exam extraction pipeline",
            "selected_memory": [],
            "selected_artifacts": [n["id"] for n in nodes if n["type"] == "artifact"],
            "constraints": ["use_api_key1_only", "no_hallucinated_answers"],
            "quality_requirements": ["domain_relevance >= 0.7", "answer_consistency >= 0.7"],
            "output_schema": {"jsonl": True},
            "forbidden_actions": ["use_api_key2", "hallucinate_answers"],
        }
        packages.append(pkg)

    with open(PACKAGES_PATH, "w") as f:
        for pkg in packages:
            f.write(json.dumps(pkg, ensure_ascii=False) + "\n")

    print(f"DTCG trace: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Trace: {TRACE_PATH}")
    print(f"Packages: {PACKAGES_PATH}")
    print(f"Statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    build_trace()
