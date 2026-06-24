"""Phase 6.55 smoke test.

Verifies agent instantiation, DTCG graph creation, and context package generation.
Does NOT make expensive LLM calls.

Usage:
    python scripts/run_phase_6_55_smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_55_pipeline_audit"
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "phase": "phase_6_55_smoke_test",
        "timestamp": time.time(),
        "tests": [],
    }

    def record_test(name, passed, details=""):
        results["tests"].append({
            "name": name,
            "passed": passed,
            "details": details,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {details}")

    print("=== Phase 6.55 Smoke Test ===\n")

    # Test 1: Import DTCG modules
    try:
        from src.autodata.context_graph.graph_schema import DynamicTaskContextGraph, NodeType, EdgeType
        record_test("import_dtcg_graph", True, "DynamicTaskContextGraph imported")
    except Exception as e:
        record_test("import_dtcg_graph", False, str(e)[:100])

    # Test 2: Import ContextSelector
    try:
        from src.autodata.context_graph.context_selector import ContextSelector
        record_test("import_context_selector", True, "ContextSelector imported")
    except Exception as e:
        record_test("import_context_selector", False, str(e)[:100])

    # Test 3: Import MessageStore
    try:
        from src.autodata.context_graph.message_store import MessageStore
        record_test("import_message_store", True, "MessageStore imported")
    except Exception as e:
        record_test("import_message_store", False, str(e)[:100])

    # Test 4: Import LocalCache
    try:
        from src.autodata.context_graph.local_cache import LocalCache
        record_test("import_local_cache", True, "LocalCache imported")
    except Exception as e:
        record_test("import_local_cache", False, str(e)[:100])

    # Test 5: Import BaseAgent
    try:
        from src.autodata.agents.base_agent import BaseAgent
        record_test("import_base_agent", True, "BaseAgent imported")
    except Exception as e:
        record_test("import_base_agent", False, str(e)[:100])

    # Test 6: Import CentralPlanningAgent
    try:
        from src.autodata.agents.planning_agent import CentralPlanningAgent
        record_test("import_central_planner", True, "CentralPlanningAgent imported")
    except Exception as e:
        record_test("import_central_planner", False, str(e)[:100])

    # Test 7: Import DataCleaningAgent
    try:
        from src.autodata.agents.data_cleaning_agent import DataCleaningAgent
        record_test("import_data_cleaning_agent", True, "DataCleaningAgent imported")
    except Exception as e:
        record_test("import_data_cleaning_agent", False, str(e)[:100])

    # Test 8: Import QualityVerificationAgent
    try:
        from src.autodata.agents.quality_verification_agent import QualityVerificationAgent
        record_test("import_quality_agent", True, "QualityVerificationAgent imported")
    except Exception as e:
        record_test("import_quality_agent", False, str(e)[:100])

    # Test 9: Import ExamExtractionAgent
    try:
        from src.autodata.agents.exam_extraction_agent import ExamExtractionAgent
        record_test("import_exam_extraction_agent", True, "ExamExtractionAgent imported")
    except Exception as e:
        record_test("import_exam_extraction_agent", False, str(e)[:100])

    # Test 10: Import pipelines
    try:
        from src.autodata.pipelines.text_cleaning_pipeline import TextCleaningPipeline
        record_test("import_text_cleaning_pipeline", True, "TextCleaningPipeline imported")
    except Exception as e:
        record_test("import_text_cleaning_pipeline", False, str(e)[:100])

    try:
        from src.autodata.pipelines.full_image_labeling_pipeline import FullImageLabelingPipeline
        record_test("import_image_labeling_pipeline", True, "FullImageLabelingPipeline imported")
    except Exception as e:
        record_test("import_image_labeling_pipeline", False, str(e)[:100])

    try:
        from src.autodata.pipelines.exam_question_extraction_pipeline import ExamQuestionExtractionPipeline
        record_test("import_exam_pipeline", True, "ExamQuestionExtractionPipeline imported")
    except Exception as e:
        record_test("import_exam_pipeline", False, str(e)[:100])

    # Test 11: Create synthetic DTCG graph
    try:
        from src.autodata.context_graph.graph_schema import DynamicTaskContextGraph, NodeType, EdgeType, Node, Edge
        graph = DynamicTaskContextGraph()

        # Add nodes using Node dataclass
        nodes = [
            Node(node_id="agent_planner", node_type=NodeType.AGENT, name="CentralPlanningAgent"),
            Node(node_id="agent_cleaner", node_type=NodeType.AGENT, name="DataCleaningAgent"),
            Node(node_id="agent_verifier", node_type=NodeType.AGENT, name="QualityVerificationAgent"),
            Node(node_id="task_clean", node_type=NodeType.TASK, name="Clean Text"),
            Node(node_id="task_verify", node_type=NodeType.TASK, name="Verify Quality"),
            Node(node_id="artifact_raw", node_type=NodeType.ARTIFACT, name="Raw Text"),
            Node(node_id="artifact_cleaned", node_type=NodeType.ARTIFACT, name="Cleaned Text"),
            Node(node_id="constraint_1", node_type=NodeType.CONSTRAINT, name="Preserve Formulas"),
        ]
        for n in nodes:
            graph.add_node(n)

        # Add edges using Edge dataclass
        edges = [
            Edge(edge_id="e1", source_id="agent_planner", target_id="task_clean", edge_type=EdgeType.AGENT_ASSIGNMENT),
            Edge(edge_id="e2", source_id="agent_planner", target_id="task_verify", edge_type=EdgeType.AGENT_ASSIGNMENT),
            Edge(edge_id="e3", source_id="task_clean", target_id="task_verify", edge_type=EdgeType.TASK_DEPENDENCY),
            Edge(edge_id="e4", source_id="artifact_raw", target_id="task_clean", edge_type=EdgeType.ARTIFACT_DERIVED_FROM),
            Edge(edge_id="e5", source_id="task_clean", target_id="artifact_cleaned", edge_type=EdgeType.ARTIFACT_DERIVED_FROM),
            Edge(edge_id="e6", source_id="constraint_1", target_id="task_clean", edge_type=EdgeType.QUALITY_FEEDBACK),
        ]
        for e in edges:
            graph.add_edge(e)

        record_test("create_dtcg_graph", True,
                    f"Created graph with {len(graph.to_dict()['nodes'])} nodes, {len(graph.to_dict()['edges'])} edges")
    except Exception as e:
        record_test("create_dtcg_graph", False, str(e)[:100])

    # Test 12: Generate context packages
    try:
        from src.autodata.context_graph.context_selector import ContextSelector, ContextSelectorConfig
        config = ContextSelectorConfig(default_token_budget=4000)
        selector = ContextSelector(config)

        pkg = selector.select_context(
            graph=graph,
            agent_node_id="agent_cleaner",
            task_id="task_clean",
            current_goal="Clean raw text chunks",
        )

        record_test("generate_context_package", True,
                    f"Package for DataCleaningAgent: {len(pkg.selected_memory)} memory, {len(pkg.selected_artifacts)} artifacts, ~{pkg.total_token_estimate} tokens")
    except Exception as e:
        record_test("generate_context_package", False, str(e)[:100])

    # Test 13: Write messages to MessageStore
    try:
        from src.autodata.context_graph.message_store import MessageStore, Message, MessageType, Visibility
        store = MessageStore()

        msg = Message.new(
            sender="CentralPlanningAgent",
            receiver="DataCleaningAgent",
            task_id="task_clean",
            message_type=MessageType.PLAN,
            content="Clean the raw text chunk about carbon fiber production",
        )
        store.add(msg)

        messages = store.get_by_receiver("DataCleaningAgent")
        record_test("write_message_store", True,
                    f"Wrote 1 message, retrieved {len(messages)} for DataCleaningAgent")
    except Exception as e:
        record_test("write_message_store", False, str(e)[:100])

    # Test 14: Update LocalCache
    try:
        from src.autodata.context_graph.local_cache import LocalCache, CacheEntry, CacheEntryType
        cache = LocalCache(agent_name="DataCleaningAgent", max_entries=50, max_tokens=2000)

        entry = CacheEntry.new(
            entry_type=CacheEntryType.OBSERVATION,
            content="Raw text contains OCR noise in formula regions",
            relevance_tags=["ocr", "formula"],
            importance=0.8,
        )
        cache.add(entry)

        context_str = cache.to_context_string()
        record_test("update_local_cache", True,
                    f"Cache has {len(cache.entries)} entries, context string length: {len(context_str)}")
    except Exception as e:
        record_test("update_local_cache", False, str(e)[:100])

    # Test 15: Serialize DTCG trace
    try:
        trace_data = graph.to_dict()
        trace_path = report_dir / "smoke_test_dtcg_trace.json"
        with open(trace_path, "w") as f:
            json.dump(trace_data, f, indent=2)
        record_test("serialize_dtcg_trace", True, f"Trace saved to {trace_path.name}")
    except Exception as e:
        record_test("serialize_dtcg_trace", False, str(e)[:100])

    # Test 16: Check Xiaomi API availability (lightweight)
    try:
        from src.autodata.utils.model_pool import get_model_pool
        pool = get_model_pool(use_key2=False)
        record_test("xiaomi_pool_available", True, f"Pool has {len(pool.endpoints)} endpoints")
    except Exception as e:
        record_test("xiaomi_pool_available", False, str(e)[:100])

    # Summary
    passed = sum(1 for t in results["tests"] if t["passed"])
    total = len(results["tests"])
    results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0,
    }

    print(f"\n=== Smoke Test Complete ===")
    print(f"Passed: {passed}/{total} ({results['summary']['pass_rate']:.0%})")

    # Save results
    with open(report_dir / "smoke_test_result.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    main()
