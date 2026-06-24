"""Run Phase 5 benchmark construction.

Usage:
    python scripts/run_phase_5_benchmark_construction.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_benchmark_construction"
    report_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {"phase": "phase_5_benchmark_construction", "stages": {}}

    # Step 1: Source audit
    print("Step 1: Source pool audit...")
    from src.autodata.pipelines.benchmark_source_auditor import audit_source_pools, save_audit_report
    audit = audit_source_pools()
    save_audit_report(audit)
    results["stages"]["source_audit"] = audit["summary"]
    print(f"  Exam: {audit['summary']['total_exam_ready']}, MM: {audit['summary']['total_mm_passed']}, Total: {audit['summary']['total_benchmark_pool']}")

    # Step 2: MM candidate repair
    print("Step 2: MM candidate repair...")
    from src.autodata.pipelines.multimodal_candidate_repair import repair_multimodal_candidates, save_repair_report
    mm_report = repair_multimodal_candidates()
    save_repair_report(mm_report)
    results["stages"]["mm_repair"] = mm_report
    print(f"  Final pool: {mm_report['final_pool_size']}")

    # Step 3: Build benchmark
    print("Step 3: Building benchmark...")
    from src.autodata.benchmark.benchmark_builder import build_benchmark, save_build_report
    build_report = build_benchmark()
    save_build_report(build_report)
    results["stages"]["build"] = build_report
    print(f"  Total balanced: {build_report['total_balanced']}")

    # Step 4: Split benchmark
    print("Step 4: Splitting benchmark...")
    from src.autodata.benchmark.benchmark_splitter import load_jsonl, split_benchmark, save_splits
    all_items = load_jsonl(Path(build_report["output_path"]))
    split_result = split_benchmark(all_items)
    split_paths = save_splits(split_result)
    results["stages"]["split"] = split_result["report"]
    print(f"  Dev: {split_result['report']['dev_items']}, Test: {split_result['report']['test_items']}")

    # Step 5: Statistics
    print("Step 5: Generating statistics...")
    from src.autodata.benchmark.benchmark_statistics import compute_statistics, save_statistics, generate_benchmark_card
    stats = compute_statistics(all_items)
    save_statistics(stats)
    card_path = generate_benchmark_card(stats)
    results["stages"]["statistics"] = {
        "total_items": stats["total_items"],
        "task_types": len(stats.get("task_type_distribution", {})),
        "source_files": stats.get("source_file_count", 0),
    }
    print(f"  Total items: {stats['total_items']}")

    # Step 6: DTCG trace (runtime integration)
    print("Step 6: Building DTCG trace...")
    from src.autodata.context_graph.pipeline_dtcg_integration import PipelineDTCG
    dtcg = PipelineDTCG("phase_5_benchmark_construction", report_dir)

    # Agents
    agent_planner = dtcg.add_agent("BenchmarkPlanningAgent", role="orchestrator")
    agent_builder = dtcg.add_agent("BenchmarkBuilder", role="assembly")
    agent_validator = dtcg.add_agent("BenchmarkValidator", role="quality")
    agent_splitter = dtcg.add_agent("BenchmarkSplitter", role="splitting")

    # Tools and constraints
    tool_pool = dtcg.add_tool("ModelPool", api="xiaomi0")
    constraint_key = dtcg.add_constraint("use_api_key1_only")

    # Source artifacts
    art_exam = dtcg.add_artifact("exam_candidates", path="data/processed/exam_questions/exam_questions_benchmark_ready_candidates.jsonl")
    art_mm = dtcg.add_artifact("mm_candidates", path="data/benchmark_candidates/multimodal/mm_benchmark_candidates_final_pool.jsonl")
    art_ku = dtcg.add_artifact("knowledge_units", path="data/processed/knowledge_units/knowledge_units_pilot.jsonl")
    art_sft = dtcg.add_artifact("sft_candidates", path="data/processed/sft_candidates/sft_candidates_pilot.jsonl")

    # Tasks
    task_audit = dtcg.add_task("source_pool_audit", status="completed")
    task_build = dtcg.add_task("benchmark_assembly", status="completed")
    task_split = dtcg.add_task("benchmark_split", status="completed")
    task_stats = dtcg.add_task("statistics_generation", status="completed")

    # Output artifacts
    art_all = dtcg.add_artifact("benchmark_candidates_all.jsonl", path=str(Path(build_report["output_path"])))
    art_dev = dtcg.add_artifact("carbon_fiber_benchmark_dev.jsonl", path="data/benchmark/carbon_fiber_benchmark_dev.jsonl")
    art_test = dtcg.add_artifact("carbon_fiber_benchmark_test.jsonl", path="data/benchmark/carbon_fiber_benchmark_test.jsonl")

    # Edges
    dtcg.connect_agent_to_task(agent_planner, task_audit)
    dtcg.connect_agent_to_task(agent_builder, task_build)
    dtcg.connect_agent_to_task(agent_splitter, task_split)
    dtcg.connect_artifact_derived(art_exam, task_build)
    dtcg.connect_artifact_derived(art_mm, task_build)
    dtcg.connect_artifact_derived(art_ku, task_build)
    dtcg.connect_artifact_derived(art_sft, task_build)
    dtcg.connect_artifact_derived(task_build, art_all)
    dtcg.connect_artifact_derived(task_split, art_dev)
    dtcg.connect_artifact_derived(task_split, art_test)
    dtcg.connect_quality_feedback(constraint_key, agent_builder)
    dtcg.connect_tool_usage(tool_pool, task_build)

    # Save DTCG runtime trace
    dtcg.save()
    print(f"  DTCG: runtime trace saved")

    # Save total elapsed
    results["total_elapsed_seconds"] = time.time() - start_time
    results["total_elapsed_formatted"] = f"{int(results['total_elapsed_seconds'] // 3600):02d}:{int((results['total_elapsed_seconds'] % 3600) // 60):02d}:{int(results['total_elapsed_seconds'] % 60):02d}"

    metadata_path = report_dir / "run_metadata_phase_5.json"
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Phase 5 Benchmark Construction Complete ===")
    print(f"Total elapsed: {results['total_elapsed_formatted']}")
    print(f"Total items: {stats['total_items']}")
    print(f"Dev: {split_result['report']['dev_items']}, Test: {split_result['report']['test_items']}")
    print(f"Report: {metadata_path}")


if __name__ == "__main__":
    main()
