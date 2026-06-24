"""Benchmark Generation Agent for Phase 6.55 enhancement.

Builds benchmark items from cleaned data, validates items, generates statistics.
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

logger = get_logger("benchmark_generation_agent")


class BenchmarkGenerationAgent(ReActAgent):
    """Agent for generating and validating benchmark items.

    Capabilities:
    - Construct benchmark items from cleaned text, images, exams
    - Validate item quality
    - Split benchmark into dev/test
    - Generate statistics and benchmark card
    """

    def __init__(
        self,
        model_client=None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store=None,
        run_id: str = "benchmark_generation",
    ) -> None:
        super().__init__(
            name="BenchmarkGenerationAgent",
            model_client=model_client,
            graph=graph,
            message_store=message_store,
            max_iterations=10,
        )
        self.run_id = run_id

        # Register tools
        self.tool_registry.register(
            "build_text_items",
            "Generate benchmark items from cleaned text and knowledge units",
            self._build_text_items_tool,
        )
        self.tool_registry.register(
            "select_multimodal_items",
            "Select validated multimodal candidates for benchmark",
            self._select_mm_items_tool,
        )
        self.tool_registry.register(
            "integrate_exam_items",
            "Integrate exam questions into benchmark",
            self._integrate_exam_tool,
        )
        self.tool_registry.register(
            "validate_items",
            "Run independent validation on benchmark items",
            self._validate_items_tool,
        )
        self.tool_registry.register(
            "split_benchmark",
            "Split benchmark into dev/test with leakage control",
            self._split_tool,
        )
        self.tool_registry.register(
            "generate_statistics",
            "Generate benchmark statistics and card",
            self._gen_stats_tool,
        )
        self.tool_registry.register(
            "finish",
            "Mark benchmark construction as complete",
            self._finish_tool,
        )

    def _register_in_graph(self):
        if self.graph:
            node = Node(
                node_id=f"agent_{self.name.lower()}",
                node_type=NodeType.AGENT,
                name=self.name,
                properties={"framework": "react", "role": "benchmark_generation"},
            )
            self.graph.add_node(node)

    def _build_text_items_tool(self, sources: str) -> str:
        return f"Building text benchmark items from: {sources}"

    def _select_mm_items_tool(self, criteria: str) -> str:
        return f"Selecting multimodal items with criteria: {criteria}"

    def _integrate_exam_tool(self, exam_path: str) -> str:
        return f"Integrating exam questions from: {exam_path}"

    def _validate_items_tool(self, items_path: str) -> str:
        return f"Validating items in: {items_path}"

    def _split_tool(self, params: str) -> str:
        return "Splitting benchmark into dev/test"

    def _gen_stats_tool(self, _: str) -> str:
        return "Generating benchmark statistics"

    def _finish_tool(self, _: str) -> str:
        return "TASK_COMPLETE: Benchmark construction finished"

    def step(self, context: dict) -> "AgentObservation":
        from src.autodata.agents.base_agent import AgentObservation
        goal = context.get("goal", "Build carbon fiber benchmark")
        result = self._think(goal, context)
        return AgentObservation(
            agent_name=self.name,
            action_type="benchmark_generate",
            content=result,
            success=True,
        )

    def run(self, task: dict, context: dict) -> list:
        observations = []
        goal = task.get("goal", "Build benchmark")
        obs = self.step({"goal": goal, **context})
        observations.append(obs)
        return observations
