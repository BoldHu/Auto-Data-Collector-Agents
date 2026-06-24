"""Phase 3 DTCG trace — register image labeling agents, artifacts, and edges.

Registers Phase 3 image labeling pipeline nodes and edges into the
Dynamic Task-Context Graph (DTCG).

Nodes:
- AGENT nodes: ImageIndexAgent, ImageDedupAgent, ImageLabelingAgent, ImageCaptionAgent,
  ImageQualityAgent, BenchmarkCandidateAgent, QualityVerifierAgent
- TASK nodes: phase_3_1, phase_3_2, phase_3_3, phase_3_4, phase_3_5
- ARTIFACT nodes: image_index.jsonl, image_dedup.jsonl, image_labels_pilot.jsonl,
  image_captions_pilot.jsonl, image_quality_scores_pilot.jsonl,
  mm_benchmark_candidates_pilot.jsonl, mm_candidate_validation_pilot.jsonl
- TOOL nodes: model_pool_multimodal, image_utils, phash
- CONSTRAINT nodes: no_raw_image_modification, provenance_preservation, json_output_schema

Edges:
- task_dependency: phase 3.1→3.2→3.3→3.4→3.5 (sequential)
- agent_assignment: each agent assigned to its task
- artifact_derived_from: each output derived from its input
- quality_feedback: verifier feedback on candidates and labels
- duplication_conflict: dedup marking duplicates
- benchmark_source: image → benchmark candidate tracing
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    EdgeType,
    NodeType,
    TaskStatus,
    Node,
    Edge,
)
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("phase_3_trace")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_image_labeling"


def build_phase_3_trace() -> DynamicTaskContextGraph:
    """Build the DTCG trace for Phase 3 image labeling pipeline."""
    graph = DynamicTaskContextGraph()
    ts = time.time()

    # ── Agent Nodes ─────────────────────────────────────────────────
    agents = {
        "img_idx": Node.new(NodeType.AGENT, "ImageIndexAgent", role="worker", framework="single-pass", phase="3.1", created_at=ts),
        "img_dd": Node.new(NodeType.AGENT, "ImageDedupAgent", role="worker", framework="single-pass", phase="3.2", created_at=ts),
        "img_lbl": Node.new(NodeType.AGENT, "ImageLabelingAgent", role="worker", framework="threadpool+modelpool", phase="3.3", created_at=ts),
        "img_cap": Node.new(NodeType.AGENT, "ImageCaptionAgent", role="worker", framework="threadpool+modelpool", phase="3.3", created_at=ts),
        "img_qa": Node.new(NodeType.AGENT, "ImageQualityAgent", role="worker", framework="threadpool+modelpool", phase="3.3", created_at=ts),
        "bm_gen": Node.new(NodeType.AGENT, "BenchmarkCandidateAgent", role="worker", framework="threadpool+modelpool", phase="3.4", created_at=ts),
        "q_ver": Node.new(NodeType.AGENT, "QualityVerifierAgent", role="worker", framework="threadpool+modelpool", phase="3.5", created_at=ts),
    }
    for node in agents.values():
        graph.add_node(node)

    # ── Task Nodes ──────────────────────────────────────────────────
    tasks = {
        "t3_1": Node.new(NodeType.TASK, "phase_3_1_image_indexing", status=TaskStatus.COMPLETED.value, phase="3.1", created_at=ts),
        "t3_2": Node.new(NodeType.TASK, "phase_3_2_image_dedup", status=TaskStatus.COMPLETED.value, phase="3.2", created_at=ts),
        "t3_3": Node.new(NodeType.TASK, "phase_3_3_image_labeling_pilot", status=TaskStatus.COMPLETED.value, phase="3.3", created_at=ts),
        "t3_4": Node.new(NodeType.TASK, "phase_3_4_benchmark_generation", status=TaskStatus.PENDING.value, phase="3.4", created_at=ts),
        "t3_5": Node.new(NodeType.TASK, "phase_3_5_quality_verification", status=TaskStatus.PENDING.value, phase="3.5", created_at=ts),
    }
    for node in tasks.values():
        graph.add_node(node)

    # ── Artifact Nodes ──────────────────────────────────────────────
    artifacts = {
        "a_idx": Node.new(NodeType.ARTIFACT, "image_index.jsonl", path="data/interim/image_index/image_index.jsonl", records=15859, phase="3.1", created_at=ts),
        "a_dd": Node.new(NodeType.ARTIFACT, "image_dedup.jsonl", path="data/interim/image_dedup/image_dedup.jsonl", records=15859, phase="3.2", created_at=ts),
        "a_lbl": Node.new(NodeType.ARTIFACT, "image_labels_pilot.jsonl", path="data/interim/image_labeled/image_labels_pilot.jsonl", records=282, phase="3.3", created_at=ts),
        "a_cap": Node.new(NodeType.ARTIFACT, "image_captions_pilot.jsonl", path="data/interim/image_labeled/image_captions_pilot.jsonl", records=282, phase="3.3", created_at=ts),
        "a_qs": Node.new(NodeType.ARTIFACT, "image_quality_scores_pilot.jsonl", path="data/interim/image_labeled/image_quality_scores_pilot.jsonl", records=282, phase="3.3", created_at=ts),
        "a_bm": Node.new(NodeType.ARTIFACT, "mm_benchmark_candidates_pilot.jsonl", path="data/benchmark_candidates/multimodal/mm_benchmark_candidates_pilot.jsonl", phase="3.4", created_at=ts),
        "a_cv": Node.new(NodeType.ARTIFACT, "mm_candidate_validation_pilot.jsonl", path="data/benchmark_candidates/multimodal/mm_candidate_validation_pilot.jsonl", phase="3.5", created_at=ts),
    }
    for node in artifacts.values():
        graph.add_node(node)

    # ── Tool Nodes ──────────────────────────────────────────────────
    tools = {
        "mp_mm": Node.new(NodeType.TOOL, "ModelPool_chat_multimodal", models="mimo-v2-omni,mimo-v2.5", role="multimodal_caller", created_at=ts),
        "mp_q": Node.new(NodeType.TOOL, "ModelPool_chat_quality", models="mimo-v2.5-pro", role="quality_caller", created_at=ts),
        "iu": Node.new(NodeType.TOOL, "image_utils", functions="encode_base64,resize_for_api,build_multimodal_message,validate_image", created_at=ts),
        "ph": Node.new(NodeType.TOOL, "phash_dedup", library="imagehash", threshold=8, created_at=ts),
    }
    for node in tools.values():
        graph.add_node(node)

    # ── Constraint Nodes ────────────────────────────────────────────
    constraints = {
        "c_nrm": Node.new(NodeType.CONSTRAINT, "no_raw_image_modification", description="Never modify original image files, only resize in-memory copies", severity="critical", created_at=ts),
        "c_prov": Node.new(NodeType.CONSTRAINT, "provenance_preservation", description="Every output must preserve image_id, file_path, keyword, metadata", severity="critical", created_at=ts),
        "c_json": Node.new(NodeType.CONSTRAINT, "json_output_schema", description="All LLM responses must follow strict JSON schema", severity="high", created_at=ts),
        "c_no_spec": Node.new(NodeType.CONSTRAINT, "no_speculation_in_caption", description="Captions must be 100% visually grounded, no inferred content", severity="high", created_at=ts),
        "c_dedup_h": Node.new(NodeType.CONSTRAINT, "dedup_hamming_threshold", description="Perceptual hash hamming distance ≤ 8 for near-duplicate grouping", severity="medium", created_at=ts),
    }
    for node in constraints.values():
        graph.add_node(node)

    # ── Edges ───────────────────────────────────────────────────────

    # Task dependencies (sequential)
    dep_pairs = [("t3_1", "t3_2"), ("t3_2", "t3_3"), ("t3_3", "t3_4"), ("t3_4", "t3_5")]
    for src, tgt in dep_pairs:
        e = Edge.new(tasks[src].node_id, tasks[tgt].node_id, EdgeType.TASK_DEPENDENCY,
                      dependency_type="sequential", strength=1.0)
        e.dependency_score = 1.0
        e.compute_weight()
        graph.add_edge(e)

    # Agent assignments
    assignments = [
        ("img_idx", "t3_1"), ("img_dd", "t3_2"), ("img_lbl", "t3_3"),
        ("img_cap", "t3_3"), ("img_qa", "t3_3"), ("bm_gen", "t3_4"), ("q_ver", "t3_5"),
    ]
    for agent_key, task_key in assignments:
        e = Edge.new(agents[agent_key].node_id, tasks[task_key].node_id, EdgeType.AGENT_ASSIGNMENT)
        e.relevance_score = 0.9
        e.compute_weight()
        graph.add_edge(e)

    # Artifact derived-from (provenance)
    derivations = [
        ("a_idx", "a_dd"), ("a_dd", "a_lbl"), ("a_dd", "a_cap"), ("a_dd", "a_qs"),
        ("a_lbl", "a_bm"), ("a_qs", "a_bm"), ("a_bm", "a_cv"),
    ]
    for src_art, tgt_art in derivations:
        e = Edge.new(artifacts[src_art].node_id, artifacts[tgt_art].node_id, EdgeType.ARTIFACT_DERIVED_FROM)
        e.trust_score = 0.8
        e.compute_weight()
        graph.add_edge(e)

    # Quality feedback: verifier -> candidates, verifier -> labels
    qf_edges = [
        ("q_ver", "a_bm"), ("q_ver", "a_lbl"), ("q_ver", "a_qs"),
    ]
    for agent_key, art_key in qf_edges:
        e = Edge.new(agents[agent_key].node_id, artifacts[art_key].node_id, EdgeType.QUALITY_FEEDBACK)
        e.trust_score = 0.9
        e.compute_weight()
        graph.add_edge(e)

    # Tool usage
    tool_usages = [
        ("img_idx", "iu"), ("img_dd", "ph"), ("img_lbl", "mp_mm"), ("img_lbl", "iu"),
        ("img_cap", "mp_mm"), ("img_cap", "iu"), ("img_qa", "mp_mm"), ("img_qa", "iu"),
        ("bm_gen", "mp_mm"), ("bm_gen", "iu"), ("q_ver", "mp_q"),
    ]
    for agent_key, tool_key in tool_usages:
        e = Edge.new(agents[agent_key].node_id, tools[tool_key].node_id, EdgeType.TOOL_USAGE)
        e.relevance_score = 0.7
        e.compute_weight()
        graph.add_edge(e)

    # Duplication conflict: dedup agent -> dedup artifact
    e = Edge.new(agents["img_dd"].node_id, artifacts["a_dd"].node_id, EdgeType.DUPLICATION_CONFLICT,
                  duplicate_count=4235, unique_count=11624)
    e.relevance_score = 0.8
    e.compute_weight()
    graph.add_edge(e)

    # Benchmark source: labeled images -> benchmark candidates
    e = Edge.new(artifacts["a_lbl"].node_id, artifacts["a_bm"].node_id, EdgeType.BENCHMARK_SOURCE)
    e.relevance_score = 0.9
    e.compute_weight()
    graph.add_edge(e)

    # Constraint enforcement
    constraint_enforced = [
        ("img_lbl", "c_nrm"), ("img_lbl", "c_prov"), ("img_lbl", "c_json"),
        ("img_cap", "c_no_spec"), ("img_dd", "c_dedup_h"), ("bm_gen", "c_prov"),
    ]
    for agent_key, c_key in constraint_enforced:
        e = Edge.new(agents[agent_key].node_id, constraints[c_key].node_id, EdgeType.CONTEXT_RELEVANCE,
                      constraint_enforcement=True)
        e.relevance_score = 0.6
        e.compute_weight()
        graph.add_edge(e)

    logger.info(f"Phase 3 DTCG trace built: {graph.node_count} nodes, {graph.edge_count} edges")
    return graph


def write_trace(graph: DynamicTaskContextGraph, output_dir: Path = REPORT_DIR) -> dict[str, str]:
    """Write DTCG trace to JSON and context packages to JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write full trace JSON
    trace_path = output_dir / "dtcg_image_labeling_trace.json"
    trace_data = graph.to_dict()
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Written DTCG trace to {trace_path}")

    # Write context packages JSONL (one per agent)
    packages_path = output_dir / "context_packages_image_labeling.jsonl"
    with open(packages_path, "w", encoding="utf-8") as f:
        for node in graph.get_nodes_by_type(NodeType.AGENT):
            neighbors = graph.get_neighbors(node.node_id)
            edges = graph.get_edges_of(node.node_id)

            package = {
                "agent_name": node.name,
                "agent_id": node.node_id,
                "task": node.properties.get("phase", ""),
                "neighbor_count": len(neighbors),
                "edge_count": len(edges),
                "relevant_artifacts": [n.name for n in neighbors if n.node_type == NodeType.ARTIFACT],
                "relevant_tasks": [n.name for n in neighbors if n.node_type == NodeType.TASK],
                "relevant_tools": [n.name for n in neighbors if n.node_type == NodeType.TOOL],
                "relevant_constraints": [n.name for n in neighbors if n.node_type == NodeType.CONSTRAINT],
                "properties": node.properties,
            }
            f.write(json.dumps(package, ensure_ascii=False) + "\n")

    logger.info(f"Written context packages to {packages_path}")

    return {
        "trace_path": str(trace_path),
        "packages_path": str(packages_path),
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
    }


def main() -> dict:
    """Build and write the Phase 3 DTCG trace."""
    logger.info("Building Phase 3 DTCG trace...")
    graph = build_phase_3_trace()
    result = write_trace(graph)
    logger.info(f"Phase 3 DTCG trace complete: {result}")
    return result