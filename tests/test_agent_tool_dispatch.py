"""Tests that agent tools dispatch correctly and produce real output.

Verifies tools are not placeholder strings and produce traceable artifacts.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_model_client():
    """Mock XiaomiModelClient for all tests."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"cleaned_text": "test output", "confidence": 0.8}'
    mock_response.total_tokens = 100
    mock_response.reasoning = "test"
    mock_client.chat.return_value = mock_response
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


class TestDataCollectionToolDispatch:
    """Test DataCollectionAgent tool dispatch."""

    def test_validate_source_existing_file(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.data_collection_agent import DataCollectionAgent
        agent = DataCollectionAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        result = agent._validate_source_tool(str(test_file))
        assert "Source valid" in result

    def test_validate_source_missing_file(self, mock_model_client, graph, message_store):
        from src.autodata.agents.data_collection_agent import DataCollectionAgent
        agent = DataCollectionAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        result = agent._validate_source_tool("/nonexistent/path")
        assert "not found" in result

    def test_list_sources(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.data_collection_agent import DataCollectionAgent
        agent = DataCollectionAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        result = agent._list_sources_tool(str(tmp_path))
        assert "file1.txt" in result
        assert "file2.txt" in result

    def test_register_manifest(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.data_collection_agent import DataCollectionAgent
        agent = DataCollectionAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        # Create a test manifest
        manifest = tmp_path / "test.jsonl"
        manifest.write_text('{"id": 1}\n{"id": 2}\n{"id": 3}\n')
        result = agent._register_manifest_tool(str(manifest))
        assert "3 records" in result

    def test_deduplicate(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.data_collection_agent import DataCollectionAgent
        agent = DataCollectionAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        metadata = tmp_path / "meta.jsonl"
        metadata.write_text(
            '{"content_hash": "aaa"}\n'
            '{"content_hash": "bbb"}\n'
            '{"content_hash": "aaa"}\n'
        )
        result = agent._deduplicate_tool(str(metadata))
        assert "3 records" in result
        assert "1 duplicates" in result


class TestBenchmarkToolDispatch:
    """Test BenchmarkGenerationAgent tool dispatch."""

    def test_load_candidates(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.benchmark_generation_agent import BenchmarkGenerationAgent
        agent = BenchmarkGenerationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        bench = tmp_path / "bench.jsonl"
        bench.write_text(
            '{"question": "Q1", "answer": "A1", "task_type": "qa"}\n'
            '{"question": "Q2", "answer": "A2", "task_type": "mc"}\n'
        )
        result = agent._load_candidates_tool(str(bench))
        assert "2 candidates" in result
        assert "qa" in result

    def test_validate_items(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.benchmark_generation_agent import BenchmarkGenerationAgent
        agent = BenchmarkGenerationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        bench = tmp_path / "bench.jsonl"
        bench.write_text(
            '{"question": "Q1", "answer": "A1"}\n'
            '{"question": "Q2"}\n'  # missing answer
        )
        result = agent._validate_items_tool(str(bench))
        # Result is a formatted string, parse key info
        assert "2 items" in result
        assert "1 valid" in result

    def test_compute_statistics(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.benchmark_generation_agent import BenchmarkGenerationAgent
        agent = BenchmarkGenerationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        bench = tmp_path / "bench.jsonl"
        bench.write_text(
            '{"question": "Q1", "task_type": "qa", "difficulty": "easy"}\n'
            '{"question": "Q2", "task_type": "qa", "difficulty": "hard"}\n'
            '{"question": "Q3", "task_type": "mc", "difficulty": "medium"}\n'
        )
        result = agent._compute_statistics_tool(str(bench))
        parsed = json.loads(result)
        assert parsed["total"] == 3
        assert parsed["task_type_distribution"]["qa"] == 2


class TestModelEvaluationToolDispatch:
    """Test ModelEvaluationAgent tool dispatch."""

    def test_load_predictions(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.model_evaluation_agent import ModelEvaluationAgent
        agent = ModelEvaluationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        preds = tmp_path / "preds.jsonl"
        preds.write_text(
            '{"strict_correct": 1, "parsed_answer": "A"}\n'
            '{"strict_correct": 0, "parsed_answer": "B"}\n'
        )
        result = agent._load_predictions_tool(str(preds))
        assert "2 predictions" in result

    def test_compute_metrics(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.model_evaluation_agent import ModelEvaluationAgent
        agent = ModelEvaluationAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        preds = tmp_path / "scored.jsonl"
        preds.write_text(
            '{"strict_correct": 1, "normalized_correct": 1, "letter_correct": 1, "parse_success": 1, "task_type": "qa"}\n'
            '{"strict_correct": 0, "normalized_correct": 0, "letter_correct": 0, "parse_success": 1, "task_type": "qa"}\n'
            '{"strict_correct": 1, "normalized_correct": 1, "letter_correct": 0, "parse_success": 1, "task_type": "mc"}\n'
        )
        result = agent._compute_metrics_tool(str(preds))
        parsed = json.loads(result)
        assert parsed["total"] == 3
        assert parsed["strict_accuracy"] == pytest.approx(2 / 3, abs=0.01)
        assert "qa" in parsed["task_type_metrics"]


class TestFineTuningToolDispatch:
    """Test FineTuningAgent tool dispatch."""

    def test_prepare_training_data(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.finetuning_agent import FineTuningAgent
        agent = FineTuningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            skip_training=True,
        )
        sft = tmp_path / "sft.jsonl"
        sft.write_text(
            '{"instruction": "What is CF?", "output": "Carbon fiber...", "task_type": "qa"}\n'
            '{"instruction": "Explain tensile", "output": "Tensile strength...", "task_type": "qa"}\n'
        )
        result = agent._prepare_data_tool(str(sft))
        parsed = json.loads(result)
        assert parsed["total_records"] == 2
        assert parsed["valid_records"] == 2

    def test_configure_training(self, mock_model_client, graph, message_store):
        from src.autodata.agents.finetuning_agent import FineTuningAgent
        agent = FineTuningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
            skip_training=True,
        )
        config = json.dumps({
            "model_name": "Qwen2.5-VL-3B",
            "sft_data_path": "/tmp/sft.jsonl",
            "output_dir": "/tmp/output",
        })
        result = agent._configure_tool(config)
        parsed = json.loads(result)
        assert parsed["status"] == "configured"
        assert parsed["skip_training"] is True

    def test_list_adapters(self, mock_model_client, graph, message_store, tmp_path):
        from src.autodata.agents.finetuning_agent import FineTuningAgent
        agent = FineTuningAgent(
            model_client=mock_model_client,
            graph=graph,
            message_store=message_store,
        )
        # Create mock adapter directory
        adapter_dir = tmp_path / "adapter_1"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text('{"r": 16}')
        result = agent._list_adapters_tool(str(tmp_path))
        parsed = json.loads(result)
        assert parsed["total"] == 1
        assert parsed["adapters"][0]["status"] == "incomplete"  # no safetensors
