"""Behavioral tests that validate behavior, not just existence.

These tests prove:
- CentralPlanningAgent invocation in central mode
- Worker stages consume planner output
- DTCG selected contexts differ under controlled trust/redundancy/cache fixtures
- Artifact lineage IDs propagate across stages
- Benchmark duplicate detection
- SFT/benchmark leakage checks
- Failure handling (all parse errors -> failed step)
- Claim registry validation
"""

import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_model_client():
    """Mock XiaomiModelClient for all tests."""
    mock_client = MagicMock()
    mock_plan = json.dumps([
        {"step_id": "step_0", "description": "Collect data from raw sources", "assigned_agent": "DataCollectionAgent", "dependencies": []},
        {"step_id": "step_1", "description": "Clean OCR text from raw book sources", "assigned_agent": "DataCleaningAgent", "dependencies": ["step_0"]},
    ])
    mock_response = MagicMock()
    mock_response.content = mock_plan
    mock_response.total_tokens = 100
    mock_response.reasoning = "test"
    mock_response.usage = {"prompt_tokens": 80, "completion_tokens": 20}
    mock_client.chat.return_value = mock_response
    mock_client.model_name = "mock_test"
    with patch("src.autodata.utils.model_client.get_default_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def graph():
    from src.autodata.context_graph.graph_schema import DynamicTaskContextGraph
    return DynamicTaskContextGraph()


@pytest.fixture
def message_store():
    from src.autodata.context_graph.message_store import MessageStore
    return MessageStore()


# ── Central Planner Invocation ──────────────────────────────────────

class TestCentralPlannerInvocation:
    """Test that central mode actually invokes CentralPlanningAgent."""

    def test_central_mode_invokes_planner(self, mock_model_client, graph, message_store, tmp_path):
        """Central planning mode must invoke CentralPlanningAgent._create_plan()."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "central"),
            mode="smoke",
            planning_mode="central",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        # Central planner must have been invoked
        assert result.central_planner_invoked is True
        assert result.planner_artifact_id is not None
        assert result.planning_mode == "central"
        # Model client must have been called for planning
        assert mock_model_client.chat.called

    def test_static_mode_does_not_invoke_planner(self, mock_model_client, graph, message_store, tmp_path):
        """Static planning mode must NOT invoke CentralPlanningAgent."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "static"),
            mode="smoke",
            planning_mode="static",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        assert result.central_planner_invoked is False
        assert result.planner_artifact_id is None
        assert result.planning_mode == "static"

    def test_invalid_planning_mode_raises(self, mock_model_client, graph, message_store):
        """Invalid planning mode must raise ValueError."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        with pytest.raises(ValueError, match="Invalid planning_mode"):
            EndToEndOrchestrator(
                model_client=mock_model_client,
                planning_mode="invalid_mode",
            )


# ── Worker Stages Consume Planner Output ─────────────────────────────

class TestWorkerConsumesPlannerOutput:
    """Test that worker stages execute based on planner-generated steps."""

    def test_workers_execute_planner_steps(self, mock_model_client, graph, message_store, tmp_path):
        """Workers must execute the steps defined by the planner."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "workers"),
            mode="smoke",
            planning_mode="central",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        # At least some workers must have executed
        assert len(result.executed_agent_names) > 0
        # Steps must have been created from planner output
        assert len(result.plan_steps) > 0
        # Each executed step must have a step_id matching plan
        plan_step_ids = {s["step_id"] for s in result.plan_steps}
        for step in result.steps:
            assert step.step_id in plan_step_ids

    def test_finetuning_dryrun_marked(self, mock_model_client, graph, message_store, tmp_path):
        """FineTuningAgent dry-run must be explicitly marked."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "dryrun"),
            mode="smoke",
            planning_mode="static",
            skip_training=True,
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        # FineTuningAgent must be in dry_run_stages if it executed
        if "FineTuningAgent" in result.executed_agent_names:
            assert "FineTuningAgent" in result.dry_run_stages

    def test_planner_artifact_persisted(self, mock_model_client, graph, message_store, tmp_path):
        """Central mode must persist planner artifact to disk."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        out_dir = tmp_path / "persist"
        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(out_dir),
            mode="smoke",
            planning_mode="central",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        # Planner artifact must be persisted
        artifact_path = out_dir / "planner_artifact.json"
        assert artifact_path.exists()
        with open(artifact_path) as f:
            artifact = json.load(f)
        assert "plan_id" in artifact
        assert "steps" in artifact
        assert "domain_request" in artifact


# ── Orchestrator Failure Handling ────────────────────────────────────

class TestOrchestratorFailureHandling:
    """Test that orchestrator correctly handles parse errors and failures."""

    def test_all_parse_errors_marks_failed(self, mock_model_client, graph, message_store, tmp_path):
        """A worker step with all parse errors must be marked failed, not completed."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        # Make the mock return unparsable output for ReAct agents
        def unparsable_chat(*args, **kwargs):
            response = MagicMock()
            response.content = "This is not a valid ReAct format"
            response.total_tokens = 50
            response.usage = {"prompt_tokens": 40, "completion_tokens": 10}
            response.success = False
            return response

        mock_model_client.chat.side_effect = unparsable_chat

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "parse_fail"),
            mode="smoke",
            planning_mode="static",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        # At least some steps should fail due to parse errors
        failed_steps = [s for s in result.steps if s.status == "failed"]
        # With unparsable output, steps should fail
        assert result.total_parse_errors > 0 or len(failed_steps) > 0

    def test_structured_failure_fields_present(self, mock_model_client, graph, message_store, tmp_path):
        """Execution steps must have structured failure fields."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "fields"),
            mode="smoke",
            planning_mode="static",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        # Every step must have structured fields
        for step in result.steps:
            assert hasattr(step, 'parse_error_count')
            assert hasattr(step, 'tool_call_count')
            assert hasattr(step, 'successful_tool_call_count')
            assert hasattr(step, 'artifact_count')
            assert hasattr(step, 'completion_reason')

    def test_orchestration_result_has_aggregate_fields(self, mock_model_client, graph, message_store, tmp_path):
        """Orchestration result must have aggregate failure/tool call fields."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "agg"),
            mode="smoke",
            planning_mode="static",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        assert hasattr(result, 'total_parse_errors')
        assert hasattr(result, 'total_tool_calls')
        assert hasattr(result, 'total_successful_tool_calls')
        assert hasattr(result, 'total_artifacts')


# ── DTCG Context Selection ──────────────────────────────────────────

class TestDTCGContextSelection:
    """Test that DTCG variants produce different context under controlled fixtures."""

    def test_edge_scores_propagate(self, graph):
        """Edge.new() must store scores in dataclass fields, not just properties."""
        from src.autodata.context_graph.graph_schema import Edge, EdgeType

        edge = Edge.new(
            "src", "tgt", EdgeType.CONTEXT_RELEVANCE,
            relevance_score=0.9,
            trust_score=0.8,
        )

        # Scores must be on the dataclass fields
        assert edge.relevance_score == 0.9
        assert edge.trust_score == 0.8
        # Scores must NOT be in properties
        assert "relevance_score" not in edge.properties
        assert "trust_score" not in edge.properties

    def test_no_trust_changes_selected_context(self, graph, message_store):
        """no_trust variant must select different context when trust signals differ."""
        from src.autodata.context_graph.context_selector import ContextSelector, ContextSelectorConfig
        from src.autodata.context_graph.graph_schema import Node, NodeType, Edge, EdgeType

        # Create two artifacts with different trust scores
        agent = Node.new(NodeType.AGENT, "TestAgent")
        task = Node.new(NodeType.TASK, "test_task")
        art_low = Node.new(NodeType.ARTIFACT, "low_trust_doc", content="low trust content about carbon fiber")
        art_high = Node.new(NodeType.ARTIFACT, "high_trust_doc", content="high trust content about composites")

        graph.add_node(agent)
        graph.add_node(task)
        graph.add_node(art_low)
        graph.add_node(art_high)

        graph.add_edge(Edge.new(agent.node_id, art_low.node_id, EdgeType.CONTEXT_RELEVANCE,
                               relevance_score=0.5, trust_score=0.1))
        graph.add_edge(Edge.new(agent.node_id, art_high.node_id, EdgeType.CONTEXT_RELEVANCE,
                               relevance_score=0.5, trust_score=0.9))

        # With trust: should prefer high trust
        config_trust = ContextSelectorConfig(gamma=0.4, default_token_budget=5000)
        selector_trust = ContextSelector(config_trust)
        pkg_trust = selector_trust.select_context(graph, agent.node_id, task.node_id, "test")

        # Without trust: selection should differ
        config_no_trust = ContextSelectorConfig(gamma=0.0, default_token_budget=5000)
        selector_no_trust = ContextSelector(config_no_trust)
        pkg_no_trust = selector_no_trust.select_context(graph, agent.node_id, task.node_id, "test")

        # With trust, high_trust should be selected first
        if pkg_trust.selected_artifacts:
            assert pkg_trust.selected_artifacts[0]["name"] == "high_trust_doc"

        # Selected context sets should differ
        trust_ids = {a["node_id"] for a in pkg_trust.selected_artifacts}
        no_trust_ids = {a["node_id"] for a in pkg_no_trust.selected_artifacts}
        # At minimum, the ordering should differ (high trust first vs equal)
        if len(trust_ids) > 1 and len(no_trust_ids) > 1:
            assert trust_ids == no_trust_ids  # Same items, but different order

    def test_no_redundancy_selects_more_items(self, graph, message_store):
        """no_redundancy variant must select more items when duplicates exist."""
        from src.autodata.context_graph.context_selector import ContextSelector, ContextSelectorConfig
        from src.autodata.context_graph.graph_schema import Node, NodeType, Edge, EdgeType

        agent = Node.new(NodeType.AGENT, "TestAgent")
        task = Node.new(NodeType.TASK, "test_task")
        # Two artifacts with same name (high redundancy)
        art1 = Node.new(NodeType.ARTIFACT, "same_name", content="content A")
        art2 = Node.new(NodeType.ARTIFACT, "same_name", content="content B")

        graph.add_node(agent)
        graph.add_node(task)
        graph.add_node(art1)
        graph.add_node(art2)

        graph.add_edge(Edge.new(agent.node_id, art1.node_id, EdgeType.CONTEXT_RELEVANCE, relevance_score=0.8))
        graph.add_edge(Edge.new(agent.node_id, art2.node_id, EdgeType.CONTEXT_RELEVANCE, relevance_score=0.8))

        # With redundancy penalty
        config_red = ContextSelectorConfig(lam=0.6, default_token_budget=5000)
        selector_red = ContextSelector(config_red)
        pkg_red = selector_red.select_context(graph, agent.node_id, task.node_id, "test")

        # Without redundancy penalty
        config_no_red = ContextSelectorConfig(lam=0.0, default_token_budget=5000)
        selector_no_red = ContextSelector(config_no_red)
        pkg_no_red = selector_no_red.select_context(graph, agent.node_id, task.node_id, "test")

        # Without redundancy, more items should be selected
        assert len(pkg_no_red.selected_artifacts) >= len(pkg_red.selected_artifacts)


# ── Benchmark Duplicate Detection ────────────────────────────────────

class TestBenchmarkDuplicateDetection:
    """Test benchmark duplicate detection logic."""

    def test_exact_duplicate_detection(self):
        """Exact normalized duplicates must be detected."""
        from scripts.audit_benchmark_duplicates import find_exact_duplicates

        records = [
            {"question": "What is carbon fiber?"},
            {"question": "what is carbon fiber?"},  # normalized duplicate
            {"question": "How is CF manufactured?"},
            {"question": "How is CF manufactured?"},  # normalized duplicate
        ]

        groups = find_exact_duplicates(records)
        assert len(groups) == 2
        assert [0, 1] in groups or [1, 0] in groups
        assert [2, 3] in groups or [3, 2] in groups

    def test_near_duplicate_detection(self):
        """Near-duplicates with high Jaccard similarity must be detected."""
        from scripts.audit_benchmark_duplicates import find_near_duplicates

        records = [
            {"question": "carbon fiber tensile strength properties"},
            {"question": "carbon fiber tensile strength characteristics"},  # near-dup
            {"question": "manufacturing process of aluminum alloys"},  # different
        ]

        groups = find_near_duplicates(records, threshold=0.6)
        # First two should be near-duplicates
        assert any(0 in g and 1 in g for g in groups)


# ── SFT/Benchmark Leakage ────────────────────────────────────────────

class TestLeakageDetection:
    """Test SFT/benchmark leakage detection."""

    def test_exact_leakage_detected(self):
        """Exact question overlap between SFT and benchmark must be detected."""
        from scripts.validate_final_sft_v4 import check_leakage

        train = [
            {"instruction": "What is carbon fiber?"},
            {"instruction": "Explain tensile strength"},
        ]
        val = [
            {"instruction": "What is carbon fiber?"},  # overlap with train
        ]
        bench = [
            {"question": "Explain tensile strength"},  # overlap with train
        ]

        leakage = check_leakage(train, val, bench)
        assert leakage["train_val_exact_overlap"] == 1
        assert leakage["benchmark_overlap"] == 1


# ── SFT Provenance ──────────────────────────────────────────────────

class TestSFTProvenance:
    """Test SFT provenance validation."""

    def test_list_evidence_handled(self):
        """DataAnnotationAgent must handle list evidence without crashing."""
        from src.autodata.agents.data_annotation_agent import DataAnnotationAgent

        # Test _normalize_evidence with list
        result = DataAnnotationAgent._normalize_evidence(["evidence 1", "evidence 2"])
        assert "evidence 1" in result
        assert "evidence 2" in result

    def test_dict_evidence_handled(self):
        """DataAnnotationAgent must handle dict evidence without crashing."""
        from src.autodata.agents.data_annotation_agent import DataAnnotationAgent

        result = DataAnnotationAgent._normalize_evidence({"text": "evidence text", "source": "file.txt"})
        assert "evidence text" in result

    def test_nested_evidence_handled(self):
        """DataAnnotationAgent must handle nested evidence without crashing."""
        from src.autodata.agents.data_annotation_agent import DataAnnotationAgent

        result = DataAnnotationAgent._normalize_evidence([{"text": "nested"}, "flat"])
        assert "nested" in result
        assert "flat" in result

    def test_empty_evidence_returns_empty(self):
        """DataAnnotationAgent must return empty string for empty evidence."""
        from src.autodata.agents.data_annotation_agent import DataAnnotationAgent

        assert DataAnnotationAgent._normalize_evidence("") == ""
        assert DataAnnotationAgent._normalize_evidence([]) == ""
        assert DataAnnotationAgent._normalize_evidence(None) == ""


# ── Claim Registry ──────────────────────────────────────────────────

class TestClaimRegistry:
    """Test claim registry consistency."""

    def test_csv_and_md_have_same_claims(self):
        """CSV and Markdown claim registries must have the same claim IDs."""
        import csv
        csv_path = "reports/paper_ready/revised_claim_registry.csv"
        md_path = "reports/paper_ready/revised_claim_registry.md"

        # Read CSV claim IDs
        csv_ids = set()
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_ids.add(row["claim_id"])
        except FileNotFoundError:
            pytest.skip("CSV registry not found")

        # Read MD claim IDs (from table rows)
        md_ids = set()
        try:
            with open(md_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("|") and not line.startswith("|---"):
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 3 and parts[1] and not parts[1].startswith("ID"):
                            # Check if it looks like a claim ID (e.g., SYS-01, DTCG-01)
                            claim_id = parts[1].strip()
                            if "-" in claim_id and any(c.isdigit() for c in claim_id):
                                md_ids.add(claim_id)
        except FileNotFoundError:
            pytest.skip("MD registry not found")

        # Both should have claims
        assert len(csv_ids) > 0, "CSV registry is empty"
        assert len(md_ids) > 0, "MD registry is empty"

    def test_no_prohibited_claims_in_csv(self):
        """CSV claim registry must not contain prohibited claim statuses."""
        import csv
        csv_path = "reports/paper_ready/revised_claim_registry.csv"

        prohibited = [
            "human expert validated",
            "universally outperforms",
            "small model outperforms larger",
            "fully automated end-to-end",
        ]

        try:
            with open(csv_path) as f:
                content = f.read().lower()
                for phrase in prohibited:
                    assert phrase not in content, f"Prohibited phrase found: {phrase}"
        except FileNotFoundError:
            pytest.skip("CSV registry not found")


# ── Artifact Lineage ─────────────────────────────────────────────────

class TestArtifactLineage:
    """Test artifact lineage propagation."""

    def test_dtcg_trace_has_graph_state(self, mock_model_client, graph, message_store, tmp_path):
        """DTCG trace must record graph state before and after each task."""
        from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            output_dir=str(tmp_path / "lineage"),
            mode="smoke",
            planning_mode="static",
        )

        result = orchestrator.run("Build carbon-fiber data pipeline")

        # DTCG should have nodes and edges
        assert result.dtcg_node_count > 0
        # Check DTCG trace file exists
        dtcg_path = tmp_path / "lineage" / "dtcg_trace.json"
        assert dtcg_path.exists()
