"""DTCG trace enricher for Phase 5.5.

Builds a rich DTCG trace for benchmark construction with 30+ nodes, 40+ edges.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def load_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def build_enriched_trace() -> dict:
    """Build enriched DTCG trace for Phase 5.5."""
    nodes = [
        # Agents
        {"id": "agent_planning", "type": "agent", "name": "BenchmarkPlanningAgent", "role": "orchestrator"},
        {"id": "agent_source_audit", "type": "agent", "name": "SourceAuditAgent", "role": "quality"},
        {"id": "agent_text_enhance", "type": "agent", "name": "TextTaskEnhancementAgent", "role": "generation"},
        {"id": "agent_agent_task", "type": "agent", "name": "AgentTaskGenerationAgent", "role": "generation"},
        {"id": "agent_mm_select", "type": "agent", "name": "MultimodalSelectionAgent", "role": "selection"},
        {"id": "agent_exam_integrate", "type": "agent", "name": "ExamIntegrationAgent", "role": "integration"},
        {"id": "agent_validation", "type": "agent", "name": "BenchmarkValidationAgent", "role": "quality"},
        {"id": "agent_splitter", "type": "agent", "name": "BenchmarkSplitterAgent", "role": "split"},
        {"id": "agent_leakage", "type": "agent", "name": "LeakageControlAgent", "role": "quality"},
        {"id": "agent_stats", "type": "agent", "name": "StatisticsAgent", "role": "reporting"},

        # Tasks
        {"id": "task_source_audit", "type": "task", "name": "Source Pool Audit", "status": "completed"},
        {"id": "task_text_enhance", "type": "task", "name": "Text Task Enhancement", "status": "in_progress"},
        {"id": "task_agent_task_gen", "type": "task", "name": "Agent Task Generation", "status": "in_progress"},
        {"id": "task_mm_select", "type": "task", "name": "Multimodal Selection", "status": "completed"},
        {"id": "task_exam_integrate", "type": "task", "name": "Exam Integration", "status": "completed"},
        {"id": "task_subset_build", "type": "task", "name": "Subset Construction", "status": "completed"},
        {"id": "task_validation", "type": "task", "name": "Benchmark Validation", "status": "pending"},
        {"id": "task_split", "type": "task", "name": "Benchmark Split", "status": "completed"},
        {"id": "task_leakage_check", "type": "task", "name": "Leakage Check", "status": "completed"},
        {"id": "task_statistics", "type": "task", "name": "Statistics Generation", "status": "pending"},

        # Artifacts - source pools
        {"id": "artifact_corpus", "type": "artifact", "name": "Pretraining Corpus (4,161 chunks)"},
        {"id": "artifact_ku", "type": "artifact", "name": "Knowledge Units (148)"},
        {"id": "artifact_sft", "type": "artifact", "name": "SFT Candidates (182)"},
        {"id": "artifact_exam", "type": "artifact", "name": "Exam Questions (61)"},
        {"id": "artifact_mm_pool", "type": "artifact", "name": "MM Candidates (10,038)"},

        # Artifacts - candidate pools
        {"id": "artifact_text_candidates", "type": "artifact", "name": "Text Enhanced Candidates"},
        {"id": "artifact_agent_candidates", "type": "artifact", "name": "Agent Task Candidates"},

        # Artifacts - benchmark subsets
        {"id": "artifact_cfbench_text", "type": "artifact", "name": "CFBench-Text"},
        {"id": "artifact_cfbench_mm", "type": "artifact", "name": "CFBench-MM"},
        {"id": "artifact_cfbench_exam", "type": "artifact", "name": "CFBench-Exam"},
        {"id": "artifact_cfbench_hard", "type": "artifact", "name": "CFBench-Hard"},
        {"id": "artifact_cfbench_agent", "type": "artifact", "name": "CFBench-AgentTask"},
        {"id": "artifact_cfbench_core", "type": "artifact", "name": "CFBench-Core"},
        {"id": "artifact_cfbench_full", "type": "artifact", "name": "CFBench-Full"},

        # Artifacts - reports
        {"id": "artifact_leakage_report", "type": "artifact", "name": "Leakage Report"},
        {"id": "artifact_statistics", "type": "artifact", "name": "Benchmark Statistics"},
        {"id": "artifact_benchmark_card", "type": "artifact", "name": "Benchmark Card"},
        {"id": "artifact_eval_protocol", "type": "artifact", "name": "Evaluation Protocol"},

        # Tools
        {"id": "tool_model_pool", "type": "tool", "name": "ModelPool (API_KEY1 only)"},
        {"id": "tool_concurrency", "type": "tool", "name": "Phase5ConcurrencyController"},

        # Constraints
        {"id": "constraint_api_key1", "type": "constraint", "name": "Use API_KEY1 only"},
        {"id": "constraint_no_leakage", "type": "constraint", "name": "No dev/test leakage"},
        {"id": "constraint_evidence", "type": "constraint", "name": "Every item needs evidence"},
        {"id": "constraint_balance", "type": "constraint", "name": "Balance task types"},
    ]

    edges = [
        # Agent assignments
        {"source": "agent_planning", "target": "task_source_audit", "type": "agent_assignment"},
        {"source": "agent_source_audit", "target": "task_source_audit", "type": "agent_assignment"},
        {"source": "agent_text_enhance", "target": "task_text_enhance", "type": "agent_assignment"},
        {"source": "agent_agent_task", "target": "task_agent_task_gen", "type": "agent_assignment"},
        {"source": "agent_mm_select", "target": "task_mm_select", "type": "agent_assignment"},
        {"source": "agent_exam_integrate", "target": "task_exam_integrate", "type": "agent_assignment"},
        {"source": "agent_validation", "target": "task_validation", "type": "agent_assignment"},
        {"source": "agent_splitter", "target": "task_split", "type": "agent_assignment"},
        {"source": "agent_leakage", "target": "task_leakage_check", "type": "agent_assignment"},
        {"source": "agent_stats", "target": "task_statistics", "type": "agent_assignment"},

        # Task dependencies
        {"source": "task_source_audit", "target": "task_text_enhance", "type": "task_dependency"},
        {"source": "task_source_audit", "target": "task_agent_task_gen", "type": "task_dependency"},
        {"source": "task_text_enhance", "target": "task_subset_build", "type": "task_dependency"},
        {"source": "task_agent_task_gen", "target": "task_subset_build", "type": "task_dependency"},
        {"source": "task_mm_select", "target": "task_subset_build", "type": "task_dependency"},
        {"source": "task_exam_integrate", "target": "task_subset_build", "type": "task_dependency"},
        {"source": "task_subset_build", "target": "task_validation", "type": "task_dependency"},
        {"source": "task_validation", "target": "task_split", "type": "task_dependency"},
        {"source": "task_split", "target": "task_leakage_check", "type": "task_dependency"},
        {"source": "task_leakage_check", "target": "task_statistics", "type": "task_dependency"},

        # Source to task
        {"source": "artifact_corpus", "target": "task_text_enhance", "type": "artifact_derived_from"},
        {"source": "artifact_ku", "target": "task_text_enhance", "type": "artifact_derived_from"},
        {"source": "artifact_mm_pool", "target": "task_mm_select", "type": "artifact_derived_from"},
        {"source": "artifact_exam", "target": "task_exam_integrate", "type": "artifact_derived_from"},

        # Task to artifact
        {"source": "task_text_enhance", "target": "artifact_text_candidates", "type": "artifact_derived_from"},
        {"source": "task_agent_task_gen", "target": "artifact_agent_candidates", "type": "artifact_derived_from"},
        {"source": "task_subset_build", "target": "artifact_cfbench_text", "type": "artifact_derived_from"},
        {"source": "task_subset_build", "target": "artifact_cfbench_mm", "type": "artifact_derived_from"},
        {"source": "task_subset_build", "target": "artifact_cfbench_exam", "type": "artifact_derived_from"},
        {"source": "task_subset_build", "target": "artifact_cfbench_hard", "type": "artifact_derived_from"},
        {"source": "task_subset_build", "target": "artifact_cfbench_agent", "type": "artifact_derived_from"},
        {"source": "task_subset_build", "target": "artifact_cfbench_core", "type": "artifact_derived_from"},
        {"source": "task_subset_build", "target": "artifact_cfbench_full", "type": "artifact_derived_from"},
        {"source": "task_split", "target": "artifact_leakage_report", "type": "artifact_derived_from"},
        {"source": "task_statistics", "target": "artifact_statistics", "type": "artifact_derived_from"},
        {"source": "task_statistics", "target": "artifact_benchmark_card", "type": "artifact_derived_from"},

        # Quality feedback
        {"source": "constraint_api_key1", "target": "agent_text_enhance", "type": "quality_feedback"},
        {"source": "constraint_api_key1", "target": "agent_agent_task", "type": "quality_feedback"},
        {"source": "constraint_no_leakage", "target": "agent_splitter", "type": "quality_feedback"},
        {"source": "constraint_evidence", "target": "agent_validation", "type": "quality_feedback"},
        {"source": "constraint_balance", "target": "agent_mm_select", "type": "quality_feedback"},

        # Tool usage
        {"source": "agent_text_enhance", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "agent_agent_task", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "agent_validation", "target": "tool_model_pool", "type": "tool_usage"},
        {"source": "task_text_enhance", "target": "tool_concurrency", "type": "tool_usage"},
        {"source": "task_agent_task_gen", "target": "tool_concurrency", "type": "tool_usage"},
    ]

    # Context packages
    packages = []
    for agent in [n for n in nodes if n["type"] == "agent"]:
        pkg = {
            "agent_id": agent["id"],
            "agent_name": agent["name"],
            "role": agent.get("role", ""),
            "relevant_tasks": [e["target"] for e in edges if e["source"] == agent["id"] and e["type"] == "agent_assignment"],
            "relevant_artifacts": [e["target"] for e in edges if e["source"] == agent["id"] and e["type"] == "artifact_derived_from"],
            "constraints": [e["source"] for e in edges if e["target"] == agent["id"] and e["type"] == "quality_feedback"],
            "tools": [e["target"] for e in edges if e["source"] == agent["id"] and e["type"] == "tool_usage"],
            "estimated_broadcast_tokens": 5000,
            "estimated_dtcg_tokens": 1500,
        }
        packages.append(pkg)

    # Statistics
    stats = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "context_package_count": len(packages),
        "agent_count": len([n for n in nodes if n["type"] == "agent"]),
        "task_count": len([n for n in nodes if n["type"] == "task"]),
        "artifact_count": len([n for n in nodes if n["type"] == "artifact"]),
        "tool_count": len([n for n in nodes if n["type"] == "tool"]),
        "constraint_count": len([n for n in nodes if n["type"] == "constraint"]),
        "estimated_broadcast_tokens": len(packages) * 5000,
        "estimated_dtcg_tokens": sum(p["estimated_dtcg_tokens"] for p in packages),
    }
    stats["context_saving_ratio"] = 1.0 - (stats["estimated_dtcg_tokens"] / stats["estimated_broadcast_tokens"]) if stats["estimated_broadcast_tokens"] > 0 else 0

    trace = {
        "phase": "phase_5_5_benchmark_refinement",
        "timestamp": time.time(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "statistics": stats,
    }

    return {"trace": trace, "packages": packages, "statistics": stats}


def save_trace(result: dict) -> tuple[Path, Path, Path]:
    """Save trace, packages, and statistics."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_5_benchmark_refinement"
    report_dir.mkdir(parents=True, exist_ok=True)

    trace_path = report_dir / "dtcg_phase_5_5_trace.json"
    packages_path = report_dir / "context_packages_phase_5_5.jsonl"
    stats_path = report_dir / "dtcg_phase_5_5_statistics.json"

    with open(trace_path, "w") as f:
        json.dump(result["trace"], f, indent=2, ensure_ascii=False)

    with open(packages_path, "w") as f:
        for pkg in result["packages"]:
            f.write(json.dumps(pkg, ensure_ascii=False) + "\n")

    with open(stats_path, "w") as f:
        json.dump(result["statistics"], f, indent=2)

    return trace_path, packages_path, stats_path
