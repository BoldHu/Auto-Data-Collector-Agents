"""Unit tests for Phase 2.5 DTCG trace recording."""

import pytest

from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
    TaskStatus,
)


class TestDTCGNodes:
    def test_all_node_types_defined(self):
        """Verify all required node types are in NodeType enum."""
        required = ["agent", "task", "artifact", "memory", "tool", "constraint"]
        for nt in required:
            assert nt in [e.value for e in NodeType], f"Missing node type: {nt}"

    def test_agent_node_creation(self):
        """Verify agent nodes can be created."""
        graph = DynamicTaskContextGraph()
        agent_node = Node(
            node_id="agent_test",
            node_type=NodeType.AGENT,
            name="TestAgent",
            properties={"framework": "react", "model": "mimo-v2.5-pro"},
        )
        graph.add_node(agent_node)
        assert graph.node_count == 1
        assert graph.get_node("agent_test").node_type == NodeType.AGENT

    def test_artifact_node_creation(self):
        """Verify artifact nodes for chunks, KU, SFT can be created."""
        graph = DynamicTaskContextGraph()
        # Raw chunk artifact
        raw_node = Node(
            node_id="art_raw_test",
            node_type=NodeType.ARTIFACT,
            name="Raw chunk from test.json p.10",
            properties={"source_file": "test.json", "page_number": 10},
        )
        graph.add_node(raw_node)
        # Cleaned chunk artifact
        cleaned_node = Node(
            node_id="art_cleaned_test",
            node_type=NodeType.ARTIFACT,
            name="Cleaned chunk test_chunk_001",
            properties={"source_file": "test.json", "chunk_type": "body"},
        )
        graph.add_node(cleaned_node)
        # KU artifact
        ku_node = Node(
            node_id="art_ku_test",
            node_type=NodeType.ARTIFACT,
            name="Knowledge unit ku_001: carbon fiber",
            properties={"knowledge_type": "definition"},
        )
        graph.add_node(ku_node)
        # SFT artifact
        sft_node = Node(
            node_id="art_sft_test",
            node_type=NodeType.ARTIFACT,
            name="SFT candidate sft_001: qa",
            properties={"task_type": "qa"},
        )
        graph.add_node(sft_node)

        assert graph.node_count == 4
        artifacts = graph.get_nodes_by_type(NodeType.ARTIFACT)
        assert len(artifacts) == 4

    def test_constraint_node_creation(self):
        """Verify constraint nodes for cleaning and quality can be created."""
        graph = DynamicTaskContextGraph()
        constraints = [
            ("c1", "No hallucinated facts allowed", "quality"),
            ("c2", "Every output must preserve provenance", "provenance"),
            ("c3", "Formulas must not be simplified", "domain"),
        ]
        for cid, desc, category in constraints:
            node = Node(
                node_id=cid,
                node_type=NodeType.CONSTRAINT,
                name=desc,
                properties={"category": category, "active": True},
            )
            graph.add_node(node)

        assert len(graph.get_nodes_by_type(NodeType.CONSTRAINT)) == 3

    def test_memory_node_creation(self):
        """Verify memory nodes for phase summaries."""
        graph = DynamicTaskContextGraph()
        mem_node = Node(
            node_id="mem_inventory",
            node_type=NodeType.MEMORY,
            name="Phase 2 inventory summary",
            properties={"phase": "phase_2"},
        )
        graph.add_node(mem_node)
        assert len(graph.get_nodes_by_type(NodeType.MEMORY)) == 1

    def test_tool_node_creation(self):
        """Verify tool nodes can be created."""
        graph = DynamicTaskContextGraph()
        tool_node = Node(
            node_id="tool_llm",
            node_type=NodeType.TOOL,
            name="Xiaomi LLM API caller",
            properties={"tool_type": "mimo-v2.5-pro"},
        )
        graph.add_node(tool_node)
        assert len(graph.get_nodes_by_type(NodeType.TOOL)) == 1


class TestDTCGEdges:
    def test_all_edge_types_defined(self):
        """Verify all required edge types are in EdgeType enum."""
        required = [
            "task_dependency", "agent_assignment", "artifact_derived_from",
            "context_relevance", "quality_feedback", "tool_usage",
            "benchmark_source",
        ]
        for et in required:
            assert et in [e.value for e in EdgeType], f"Missing edge type: {et}"

    def test_task_dependency_edge(self):
        """Verify task dependency edges can be created."""
        graph = DynamicTaskContextGraph()
        t1 = Node(node_id="task_clean", node_type=NodeType.TASK, name="Clean text")
        t2 = Node(node_id="task_verify", node_type=NodeType.TASK, name="Verify quality")
        graph.add_node(t1)
        graph.add_node(t2)
        edge = Edge.new(
            source_id="task_verify",
            target_id="task_clean",
            edge_type=EdgeType.TASK_DEPENDENCY,
            dependency_score=0.8,
        )
        graph.add_edge(edge)
        assert graph.edge_count == 1
        assert graph.get_edges_by_type(EdgeType.TASK_DEPENDENCY)[0].edge_type == EdgeType.TASK_DEPENDENCY

    def test_agent_assignment_edge(self):
        """Verify agent assignment edges can be created."""
        graph = DynamicTaskContextGraph()
        agent = Node(node_id="agent_dc", node_type=NodeType.AGENT, name="DataCleaningAgent")
        task = Node(node_id="task_clean", node_type=NodeType.TASK, name="Clean text")
        graph.add_node(agent)
        graph.add_node(task)
        edge = Edge.new(
            source_id="agent_dc",
            target_id="task_clean",
            edge_type=EdgeType.AGENT_ASSIGNMENT,
            relevance_score=1.0,
        )
        graph.add_edge(edge)
        assert graph.edge_count == 1
        assert len(graph.get_edges_by_type(EdgeType.AGENT_ASSIGNMENT)) == 1

    def test_artifact_derived_from_edge(self):
        """Verify artifact derived-from edges (raw → cleaned → KU/SFT)."""
        graph = DynamicTaskContextGraph()
        raw = Node(node_id="art_raw_001", node_type=NodeType.ARTIFACT, name="Raw chunk")
        cleaned = Node(node_id="art_cleaned_001", node_type=NodeType.ARTIFACT, name="Cleaned chunk")
        graph.add_node(raw)
        graph.add_node(cleaned)
        edge = Edge.new(
            source_id="art_cleaned_001",
            target_id="art_raw_001",
            edge_type=EdgeType.ARTIFACT_DERIVED_FROM,
            relevance_score=0.9,
        )
        graph.add_edge(edge)
        assert graph.edge_count == 1
        derived_edges = graph.get_edges_by_type(EdgeType.ARTIFACT_DERIVED_FROM)
        assert len(derived_edges) == 1
        assert derived_edges[0].source_id == "art_cleaned_001"

    def test_quality_feedback_edge(self):
        """Verify quality feedback edges (verifier → cleaned chunk)."""
        graph = DynamicTaskContextGraph()
        verifier = Node(node_id="agent_qv", node_type=NodeType.AGENT, name="QualityVerificationAgent")
        cleaned = Node(node_id="art_cleaned_001", node_type=NodeType.ARTIFACT, name="Cleaned chunk")
        graph.add_node(verifier)
        graph.add_node(cleaned)
        edge = Edge.new(
            source_id="agent_qv",
            target_id="art_cleaned_001",
            edge_type=EdgeType.QUALITY_FEEDBACK,
            trust_score=0.85,
            properties={"verdict": "passed", "average_score": 0.85},
        )
        graph.add_edge(edge)
        assert graph.edge_count == 1
        qf_edges = graph.get_edges_by_type(EdgeType.QUALITY_FEEDBACK)
        assert len(qf_edges) == 1
        assert qf_edges[0].source_id == "agent_qv"

    def test_tool_usage_edge(self):
        """Verify tool usage edges."""
        graph = DynamicTaskContextGraph()
        agent = Node(node_id="agent_dc", node_type=NodeType.AGENT, name="DataCleaningAgent")
        tool = Node(node_id="tool_llm", node_type=NodeType.TOOL, name="Xiaomi LLM")
        graph.add_node(agent)
        graph.add_node(tool)
        edge = Edge.new(
            source_id="agent_dc",
            target_id="tool_llm",
            edge_type=EdgeType.TOOL_USAGE,
        )
        graph.add_edge(edge)
        assert len(graph.get_edges_by_type(EdgeType.TOOL_USAGE)) == 1

    def test_graph_serialization(self):
        """Verify DTCG graph can serialize to dict with nodes and edges."""
        graph = DynamicTaskContextGraph()
        agent = Node(node_id="agent_1", node_type=NodeType.AGENT, name="TestAgent")
        task = Node(node_id="task_1", node_type=NodeType.TASK, name="TestTask")
        graph.add_node(agent)
        graph.add_node(task)
        edge = Edge.new(
            source_id="agent_1",
            target_id="task_1",
            edge_type=EdgeType.AGENT_ASSIGNMENT,
        )
        graph.add_edge(edge)
        d = graph.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][edge.edge_id]["edge_type"] == "agent_assignment"

    def test_edge_weight_computation(self):
        """Verify dynamic edge weight can be computed."""
        edge = Edge(
            edge_id="test_edge",
            source_id="agent_1",
            target_id="task_1",
            edge_type=EdgeType.AGENT_ASSIGNMENT,
            relevance_score=0.8,
            dependency_score=0.5,
            recency_score=0.3,
            trust_score=0.9,
            redundancy_score=0.1,
            cost_score=0.2,
        )
        weight = edge.compute_weight()
        assert 0 < weight < 1  # sigmoid output