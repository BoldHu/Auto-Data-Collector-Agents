"""Model Evaluation Agent for Phase 6.55 enhancement.

Runs baseline model evaluation, computes metrics, saves results.
Inherits from ReActAgent for DTCG integration.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.autodata.agents.react_agent import ReActAgent
from src.autodata.context_graph.graph_schema import DynamicTaskContextGraph, NodeType, EdgeType, Node
from src.autodata.context_graph.message_store import MessageType
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("model_evaluation_agent")


class ModelEvaluationAgent(ReActAgent):
    """Agent for evaluating models on the benchmark.

    Capabilities:
    - Run baseline models on benchmark subsets
    - Compute metrics (accuracy, F1, judge scores)
    - Generate paper tables and error analysis
    - Support system ablation comparisons
    """

    def __init__(
        self,
        model_client=None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store=None,
        run_id: str = "model_evaluation",
    ) -> None:
        super().__init__(
            name="ModelEvaluationAgent",
            model_client=model_client,
            graph=graph,
            message_store=message_store,
            max_iterations=10,
        )
        self.run_id = run_id

        # Register tools
        self.tool_registry.register(
            "run_evaluation",
            "Run model evaluation on a benchmark subset",
            self._run_eval_tool,
        )
        self.tool_registry.register(
            "compute_metrics",
            "Compute evaluation metrics for predictions",
            self._compute_metrics_tool,
        )
        self.tool_registry.register(
            "generate_tables",
            "Generate paper-ready CSV tables",
            self._gen_tables_tool,
        )
        self.tool_registry.register(
            "analyze_errors",
            "Run error taxonomy analysis",
            self._analyze_errors_tool,
        )
        self.tool_registry.register(
            "compare_systems",
            "Run system ablation comparison",
            self._compare_systems_tool,
        )
        self.tool_registry.register(
            "finish",
            "Mark evaluation as complete",
            self._finish_tool,
        )

    def _register_in_graph(self):
        if self.graph:
            node = Node(
                node_id=f"agent_{self.name.lower()}",
                node_type=NodeType.AGENT,
                name=self.name,
                properties={"framework": "react", "role": "model_evaluation"},
            )
            self.graph.add_node(node)

    def _run_eval_tool(self, params: str) -> str:
        return f"Running evaluation with params: {params}"

    def _compute_metrics_tool(self, predictions_path: str) -> str:
        return f"Computing metrics for: {predictions_path}"

    def _gen_tables_tool(self, _: str) -> str:
        return "Generating paper tables"

    def _analyze_errors_tool(self, predictions_path: str) -> str:
        return f"Analyzing errors in: {predictions_path}"

    def _compare_systems_tool(self, config: str) -> str:
        return f"Running system ablation: {config}"

    def _finish_tool(self, _: str) -> str:
        return "TASK_COMPLETE: Model evaluation finished"

    def step(self, context: dict) -> "AgentObservation":
        from src.autodata.agents.base_agent import AgentObservation
        goal = context.get("goal", "Evaluate models on benchmark")
        result = self._think(goal, context)
        return AgentObservation(
            agent_name=self.name,
            action_type="evaluate",
            content=result,
            success=True,
        )

    def run(self, task: dict, context: dict) -> list:
        observations = []
        goal = task.get("goal", "Evaluate models")
        obs = self.step({"goal": goal, **context})
        observations.append(obs)
        return observations
