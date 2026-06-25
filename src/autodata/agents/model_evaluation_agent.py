"""Model Evaluation Agent — runs model evaluation on benchmark items.

Loads benchmark manifests, runs model inference, computes metrics,
and generates evaluation result files.
Inherits from ReActAgent for DTCG integration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from src.autodata.agents.react_agent import ReActAgent
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    EdgeType,
    Node,
    NodeType,
)
from src.autodata.context_graph.local_cache import CacheEntryType
from src.autodata.context_graph.message_store import MessageType, MessageStore, Visibility
from src.autodata.utils.logging_utils import get_logger
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client

logger = get_logger("model_evaluation_agent")


class ModelEvaluationAgent(ReActAgent):
    """Agent for evaluating models on the benchmark.

    Capabilities:
    - Load benchmark manifests (canonical150, large361, full)
    - Score model predictions against reference answers
    - Compute accuracy metrics (strict, normalized, letter)
    - Generate evaluation result files

    This agent operates on existing raw output files from prior evaluation runs.
    """

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store: Optional[MessageStore] = None,
        run_id: str = "model_evaluation",
        output_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="ModelEvaluationAgent",
            model_client=model_client,
            message_store=message_store,
            max_iterations=10,
        )
        self.graph = graph or DynamicTaskContextGraph()
        self.run_id = run_id
        self.output_path = output_path
        self._items_scored = 0

        # Register tools
        self.tool_registry.register(
            "load_predictions",
            "Load model prediction outputs from a JSONL file",
            self._load_predictions_tool,
        )
        self.tool_registry.register(
            "score_predictions",
            "Score predictions against reference answers",
            self._score_predictions_tool,
        )
        self.tool_registry.register(
            "compute_metrics",
            "Compute aggregate metrics from scored predictions",
            self._compute_metrics_tool,
        )
        self.tool_registry.register(
            "compare_models",
            "Compare metrics across multiple model outputs",
            self._compare_models_tool,
        )
        self.tool_registry.register(
            "finish",
            "Mark evaluation as complete",
            self._finish_tool,
        )

        # Register agent node in DTCG
        self._register_in_graph()

    def _register_in_graph(self) -> None:
        """Register this agent as a node in the DTCG."""
        node = Node(
            node_id=self.graph_node_id,
            node_type=NodeType.AGENT,
            name=self.name,
            properties={
                "framework": "react",
                "model": self.model,
                "role": "model_evaluation",
            },
        )
        self.graph.add_node(node)

    def _load_predictions_tool(self, jsonl_path: str) -> str:
        """Load model prediction outputs from a JSONL file."""
        path = Path(jsonl_path)
        if not path.exists():
            return f"Error: file not found at {jsonl_path}"

        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        # Register artifact node
        artifact_node = Node(
            node_id=f"art_predictions_{path.stem}",
            node_type=NodeType.ARTIFACT,
            name=f"Predictions: {path.name}",
            properties={
                "path": str(path),
                "record_count": len(records),
                "source_type": "model_predictions",
            },
        )
        self.graph.add_node(artifact_node)

        # Check for key fields
        has_correct = sum(1 for r in records if "correct" in r or "strict_correct" in r)
        has_parsed = sum(1 for r in records if "parsed_answer" in r)

        return f"Loaded {len(records)} predictions from {path.name}. Scored: {has_correct}, Parsed: {has_parsed}"

    def _score_predictions_tool(self, params: str) -> str:
        """Score predictions against reference answers."""
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            return f"Error: invalid JSON config: {params[:100]}"

        predictions_path = config.get("predictions_path", "")
        path = Path(predictions_path)
        if not path.exists():
            return f"Error: file not found at {predictions_path}"

        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        # Count existing scores
        strict_correct = sum(1 for r in records if r.get("strict_correct", 0) == 1)
        normalized_correct = sum(1 for r in records if r.get("normalized_correct", 0) == 1)
        total = len(records)
        self._items_scored += total

        return json.dumps({
            "total": total,
            "strict_correct": strict_correct,
            "strict_accuracy": round(strict_correct / total, 4) if total > 0 else 0,
            "normalized_correct": normalized_correct,
            "normalized_accuracy": round(normalized_correct / total, 4) if total > 0 else 0,
        })

    def _compute_metrics_tool(self, scored_path: str) -> str:
        """Compute aggregate metrics from scored predictions."""
        path = Path(scored_path)
        if not path.exists():
            return f"Error: file not found at {scored_path}"

        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        if not records:
            return "No records to analyze"

        total = len(records)
        strict_correct = sum(1 for r in records if r.get("strict_correct", 0) == 1)
        normalized_correct = sum(1 for r in records if r.get("normalized_correct", 0) == 1)
        letter_correct = sum(1 for r in records if r.get("letter_correct", 0) == 1)
        parse_success = sum(1 for r in records if r.get("parse_success", 1) == 1)

        # Task type breakdown
        task_metrics = {}
        for r in records:
            tt = r.get("task_type", "unknown")
            if tt not in task_metrics:
                task_metrics[tt] = {"total": 0, "correct": 0}
            task_metrics[tt]["total"] += 1
            if r.get("strict_correct", 0) == 1:
                task_metrics[tt]["correct"] += 1

        for tt in task_metrics:
            t = task_metrics[tt]["total"]
            c = task_metrics[tt]["correct"]
            task_metrics[tt]["accuracy"] = round(c / t, 4) if t > 0 else 0

        return json.dumps({
            "total": total,
            "strict_accuracy": round(strict_correct / total, 4),
            "normalized_accuracy": round(normalized_correct / total, 4),
            "letter_accuracy": round(letter_correct / total, 4),
            "parse_success_rate": round(parse_success / total, 4),
            "task_type_metrics": task_metrics,
        }, ensure_ascii=False, indent=2)

    def _compare_models_tool(self, config: str) -> str:
        """Compare metrics across multiple model outputs."""
        try:
            model_paths = json.loads(config)
        except json.JSONDecodeError:
            return f"Error: expected JSON dict of model_name -> path, got: {config[:100]}"

        results = {}
        for model_name, path_str in model_paths.items():
            path = Path(path_str)
            if not path.exists():
                results[model_name] = {"error": "file not found"}
                continue

            records = []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except Exception:
                results[model_name] = {"error": "read error"}
                continue

            total = len(records)
            if total == 0:
                results[model_name] = {"total": 0}
                continue

            strict_correct = sum(1 for r in records if r.get("strict_correct", 0) == 1)
            results[model_name] = {
                "total": total,
                "strict_correct": strict_correct,
                "strict_accuracy": round(strict_correct / total, 4),
            }

        return json.dumps(results, ensure_ascii=False, indent=2)

    def _finish_tool(self, _: str) -> str:
        """Mark task as complete."""
        return f"TASK_COMPLETE: Model evaluation finished. {self._items_scored} items scored."

    def run(self, task: str, context: Optional[dict] = None) -> list:
        """Execute evaluation task using the ReAct loop."""
        return super().run(task=task, context=context)
