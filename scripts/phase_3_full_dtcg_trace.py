"""Build DTCG trace for Phase 3.9 full image labeling pipeline.

Usage:
    python scripts/phase_3_full_dtcg_trace.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_full_image_labeling"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_labels_full.jsonl"
CAPTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_captions_full.jsonl"
QUALITY_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_quality_scores_full.jsonl"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_full.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_candidate_validation_full.jsonl"
TRACE_PATH = REPORT_DIR / "dtcg_full_image_labeling_trace.json"
PACKAGES_PATH = REPORT_DIR / "context_packages_full_image_labeling.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_trace():
    """Build DTCG trace graph for the full labeling pipeline."""
    nodes = []
    edges = []

    # Agent nodes
    agents = [
        {"id": "agent_central_planner", "type": "agent", "name": "Central Planning Agent"},
        {"id": "agent_data_labeling", "type": "agent", "name": "Data Labeling Agent"},
        {"id": "agent_benchmark_gen", "type": "agent", "name": "Benchmark Generation Agent"},
        {"id": "agent_quality_verifier", "type": "agent", "name": "Quality Verification Agent"},
    ]
    nodes.extend(agents)

    # Task nodes
    tasks = [
        {"id": "task_stage1_labeling", "type": "task", "name": "Stage 1: Image Labeling", "status": "completed"},
        {"id": "task_stage2_candidates", "type": "task", "name": "Stage 2: Benchmark Candidates", "status": "completed"},
        {"id": "task_stage3_validation", "type": "task", "name": "Stage 3: Critic Validation", "status": "completed"},
    ]
    nodes.extend(tasks)

    # Artifact nodes - output files
    artifacts = [
        {"id": "artifact_labels", "type": "artifact", "name": "image_labels_full.jsonl", "path": str(LABELS_PATH)},
        {"id": "artifact_captions", "type": "artifact", "name": "image_captions_full.jsonl", "path": str(CAPTIONS_PATH)},
        {"id": "artifact_quality", "type": "artifact", "name": "image_quality_scores_full.jsonl", "path": str(QUALITY_PATH)},
        {"id": "artifact_candidates", "type": "artifact", "name": "mm_benchmark_candidates_full.jsonl", "path": str(CANDIDATES_PATH)},
        {"id": "artifact_validation", "type": "artifact", "name": "mm_candidate_validation_full.jsonl", "path": str(VALIDATION_PATH)},
    ]
    nodes.extend(artifacts)

    # Tool nodes
    tools = [
        {"id": "tool_model_pool", "type": "tool", "name": "ModelPool (dual API keys)"},
        {"id": "tool_writer_queue", "type": "tool", "name": "WriterQueue (concurrent JSONL writer)"},
        {"id": "tool_adaptive_concurrency", "type": "tool", "name": "AdaptiveConcurrencyController"},
        {"id": "tool_progress_tracker", "type": "tool", "name": "ImageProgressTracker"},
    ]
    nodes.extend(tools)

    # Constraint nodes
    constraints = [
        {"id": "constraint_no_modify_raw", "type": "constraint", "name": "Do not modify raw images"},
        {"id": "constraint_no_delete_dup", "type": "constraint", "name": "Do not delete duplicates"},
        {"id": "constraint_domain_relevance", "type": "constraint", "name": "Domain relevance >= 0.7"},
        {"id": "constraint_quality_keep", "type": "constraint", "name": "Quality status == keep"},
    ]
    nodes.extend(constraints)

    # Memory nodes
    memory = [
        {"id": "memory_pilot_results", "type": "memory", "name": "Pilot Phase 3 results (282 images)"},
        {"id": "memory_checkpoint_stage1", "type": "memory", "name": "Stage 1 checkpoint (11,308 IDs)"},
        {"id": "memory_checkpoint_stage2", "type": "memory", "name": "Stage 2 checkpoint"},
    ]
    nodes.extend(memory)

    # Edges
    edges.extend([
        {"source": "agent_central_planner", "target": "task_stage1_labeling", "type": "agent_assignment"},
        {"source": "agent_central_planner", "target": "task_stage2_candidates", "type": "agent_assignment"},
        {"source": "agent_central_planner", "target": "task_stage3_validation", "type": "agent_assignment"},
        {"source": "agent_data_labeling", "target": "task_stage1_labeling", "type": "agent_assignment"},
        {"source": "agent_benchmark_gen", "target": "task_stage2_candidates", "type": "agent_assignment"},
        {"source": "agent_quality_verifier", "target": "task_stage3_validation", "type": "agent_assignment"},
        {"source": "task_stage1_labeling", "target": "artifact_labels", "type": "artifact_derived_from"},
        {"source": "task_stage1_labeling", "target": "artifact_captions", "type": "artifact_derived_from"},
        {"source": "task_stage1_labeling", "target": "artifact_quality", "type": "artifact_derived_from"},
        {"source": "task_stage2_candidates", "target": "artifact_candidates", "type": "artifact_derived_from"},
        {"source": "task_stage3_validation", "target": "artifact_validation", "type": "artifact_derived_from"},
        {"source": "artifact_labels", "target": "task_stage2_candidates", "type": "task_dependency"},
        {"source": "artifact_quality", "target": "task_stage2_candidates", "type": "task_dependency"},
        {"source": "artifact_candidates", "target": "task_stage3_validation", "type": "task_dependency"},
        {"source": "agent_data_labeling", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "agent_data_labeling", "target": "tool_writer_queue", "type": "tool_usage"},
        {"source": "agent_benchmark_gen", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "agent_quality_verifier", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "task_stage1_labeling", "target": "tool_adaptive_concurrency", "type": "tool_usage"},
        {"source": "task_stage1_labeling", "target": "tool_progress_tracker", "type": "tool_usage"},
        {"source": "memory_pilot_results", "target": "task_stage1_labeling", "type": "context_relevance"},
        {"source": "memory_checkpoint_stage1", "target": "task_stage1_labeling", "type": "context_relevance"},
        {"source": "constraint_no_modify_raw", "target": "task_stage1_labeling", "type": "quality_feedback"},
        {"source": "constraint_domain_relevance", "target": "task_stage2_candidates", "type": "quality_feedback"},
        {"source": "constraint_quality_keep", "target": "task_stage2_candidates", "type": "quality_feedback"},
    ])

    # Load statistics
    stats = {}
    for name, path in [("labels", LABELS_PATH), ("captions", CAPTIONS_PATH), ("quality", QUALITY_PATH), ("candidates", CANDIDATES_PATH), ("validation", VALIDATION_PATH)]:
        if path.exists():
            with open(path) as f:
                count = sum(1 for _ in f)
            stats[name] = count
        else:
            stats[name] = 0

    trace = {
        "phase": "3.9_full_image_labeling",
        "timestamp": time.time(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "statistics": stats,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRACE_PATH, "w") as f:
        json.dump(trace, f, indent=2)

    # Build context packages
    packages = []
    for task in tasks:
        pkg = {
            "agent_name": "Central Planning Agent",
            "task_id": task["id"],
            "current_goal": task["name"],
            "allowed_tools": ["ModelPool", "WriterQueue", "ImageProgressTracker"],
            "relevant_plan": "Phase 3.9 full-scale image labeling pipeline",
            "selected_memory": ["memory_pilot_results", "memory_checkpoint_stage1"],
            "selected_artifacts": [a["id"] for a in artifacts],
            "constraints": [c["name"] for c in constraints],
            "quality_requirements": ["caption_faithfulness >= 90%", "label_reasonableness >= 95%", "domain_relevance >= 0.7"],
            "output_schema": {"jsonl": True},
            "forbidden_actions": ["modify_raw_images", "delete_duplicates", "run_finetuning"],
        }
        packages.append(pkg)

    with open(PACKAGES_PATH, "w") as f:
        for pkg in packages:
            f.write(json.dumps(pkg) + "\n")

    print(f"DTCG trace built: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Trace: {TRACE_PATH}")
    print(f"Packages: {PACKAGES_PATH}")
    print(f"Statistics: {json.dumps(stats, indent=2)}")

    return {"node_count": len(nodes), "edge_count": len(edges), "trace_path": str(TRACE_PATH), "packages_path": str(PACKAGES_PATH), "statistics": stats}


if __name__ == "__main__":
    build_trace()
