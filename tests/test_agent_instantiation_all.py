"""Tests that all advertised agents can be instantiated cleanly.

These tests use mock model clients so no external API calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch


# Mock the model client to avoid real API calls
@pytest.fixture(autouse=True)
def mock_model_client():
    """Mock XiaomiModelClient for all agent tests."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Thought: I should finish this task.\nAction: [finish] [done]"
    mock_response.total_tokens = 100
    mock_response.reasoning = "test reasoning"
    mock_client.chat.return_value = mock_response
    with patch("src.autodata.utils.model_client.get_default_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def graph():
    """Create a fresh DynamicTaskContextGraph."""
    from src.autodata.context_graph.graph_schema import DynamicTaskContextGraph
    return DynamicTaskContextGraph()


@pytest.fixture
def message_store():
    """Create a fresh MessageStore."""
    from src.autodata.context_graph.message_store import MessageStore
    return MessageStore()


class TestAgentInstantiation:
    """Test that all agents instantiate without errors."""

    def test_data_collection_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.data_collection_agent import DataCollectionAgent
        agent = DataCollectionAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        assert agent.name == "DataCollectionAgent"
        assert agent.tool_registry.has_tool("register_manifest")
        assert agent.tool_registry.has_tool("validate_source")
        assert agent.tool_registry.has_tool("finish")

    def test_data_cleaning_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.data_cleaning_agent import DataCleaningAgent
        agent = DataCleaningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        assert agent.name == "DataCleaningAgent"
        assert agent.tool_registry.has_tool("text_cleaner")
        assert agent.tool_registry.has_tool("finish")

    def test_data_annotation_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.data_annotation_agent import DataAnnotationAgent
        agent = DataAnnotationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        assert agent.name == "DataAnnotationAgent"
        assert agent.tool_registry.has_tool("generate_sft_samples")
        assert agent.tool_registry.has_tool("create_knowledge_units")
        assert agent.tool_registry.has_tool("finish")

    def test_quality_verification_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.quality_verification_agent import QualityVerificationAgent
        agent = QualityVerificationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        assert agent.name == "QualityVerificationAgent"
        assert agent.tool_registry.has_tool("quality_verifier")
        assert agent.tool_registry.has_tool("finish")

    def test_benchmark_generation_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.benchmark_generation_agent import BenchmarkGenerationAgent
        agent = BenchmarkGenerationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        assert agent.name == "BenchmarkGenerationAgent"
        assert agent.tool_registry.has_tool("load_candidates")
        assert agent.tool_registry.has_tool("validate_items")
        assert agent.tool_registry.has_tool("compute_statistics")
        assert agent.tool_registry.has_tool("finish")

    def test_model_evaluation_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.model_evaluation_agent import ModelEvaluationAgent
        agent = ModelEvaluationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        assert agent.name == "ModelEvaluationAgent"
        assert agent.tool_registry.has_tool("load_predictions")
        assert agent.tool_registry.has_tool("score_predictions")
        assert agent.tool_registry.has_tool("compute_metrics")
        assert agent.tool_registry.has_tool("finish")

    def test_finetuning_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.finetuning_agent import FineTuningAgent
        agent = FineTuningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            skip_training=True,
        )
        assert agent.name == "FineTuningAgent"
        assert agent.tool_registry.has_tool("prepare_training_data")
        assert agent.tool_registry.has_tool("configure_training")
        assert agent.tool_registry.has_tool("run_training")
        assert agent.tool_registry.has_tool("evaluate_adapter")
        assert agent.tool_registry.has_tool("finish")
        assert agent.skip_training is True

    def test_central_planning_agent(self, mock_model_client, graph, message_store):
        from src.autodata.agents.planning_agent import CentralPlanningAgent
        agent = CentralPlanningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        assert agent.name == "CentralPlanningAgent"


class TestAgentGraphRegistration:
    """Test that agents register themselves in the DTCG."""

    def test_data_collection_registers_in_graph(self, mock_model_client, graph, message_store):
        from src.autodata.agents.data_collection_agent import DataCollectionAgent
        agent = DataCollectionAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        # Agent should have registered a node
        node_ids = [n.node_id for n in graph.nodes.values() if n.name == "DataCollectionAgent"]
        assert len(node_ids) >= 1

    def test_data_cleaning_registers_in_graph(self, mock_model_client, graph, message_store):
        from src.autodata.agents.data_cleaning_agent import DataCleaningAgent
        agent = DataCleaningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        node_ids = [n.node_id for n in graph.nodes.values() if n.name == "DataCleaningAgent"]
        assert len(node_ids) >= 1

    def test_benchmark_generation_registers_in_graph(self, mock_model_client, graph, message_store):
        from src.autodata.agents.benchmark_generation_agent import BenchmarkGenerationAgent
        agent = BenchmarkGenerationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        node_ids = [n.node_id for n in graph.nodes.values() if n.name == "BenchmarkGenerationAgent"]
        assert len(node_ids) >= 1

    def test_model_evaluation_registers_in_graph(self, mock_model_client, graph, message_store):
        from src.autodata.agents.model_evaluation_agent import ModelEvaluationAgent
        agent = ModelEvaluationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        node_ids = [n.node_id for n in graph.nodes.values() if n.name == "ModelEvaluationAgent"]
        assert len(node_ids) >= 1

    def test_finetuning_registers_in_graph(self, mock_model_client, graph, message_store):
        from src.autodata.agents.finetuning_agent import FineTuningAgent
        agent = FineTuningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        node_ids = [n.node_id for n in graph.nodes.values() if n.name == "FineTuningAgent"]
        assert len(node_ids) >= 1
