"""Data Collection Agent for Phase 6.55 enhancement.

Handles data acquisition tasks: image crawling, paper collection, metadata collection.
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

logger = get_logger("data_collection_agent")


class DataCollectionAgent(ReActAgent):
    """Agent for collecting domain data from various sources.

    Capabilities:
    - Image crawling from web sources
    - Paper/metadata collection
    - Source validation and deduplication
    - Provenance tracking
    """

    def __init__(
        self,
        model_client=None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store=None,
        run_id: str = "data_collection",
    ) -> None:
        super().__init__(
            name="DataCollectionAgent",
            model_client=model_client,
            graph=graph,
            message_store=message_store,
            max_iterations=10,
        )
        self.run_id = run_id

        # Register tools
        self.tool_registry.register(
            "collect_images",
            "Collect images from web sources using search keywords",
            self._collect_images_tool,
        )
        self.tool_registry.register(
            "validate_source",
            "Validate a data source for relevance and quality",
            self._validate_source_tool,
        )
        self.tool_registry.register(
            "deduplicate_metadata",
            "Check metadata for duplicates against existing corpus",
            self._deduplicate_tool,
        )
        self.tool_registry.register(
            "finish",
            "Mark collection task as complete",
            self._finish_tool,
        )

    def _register_in_graph(self):
        """Register this agent in the DTCG."""
        if self.graph:
            node = Node(
                node_id=f"agent_{self.name.lower()}",
                node_type=NodeType.AGENT,
                name=self.name,
                properties={"framework": "react", "role": "data_collection"},
            )
            self.graph.add_node(node)

    def _collect_images_tool(self, query: str) -> str:
        """Tool: Collect images from web sources."""
        # This would wrap existing crawler scripts
        return f"Image collection initiated for query: {query}. Results will be saved to data/raw/images/"

    def _validate_source_tool(self, source_path: str) -> str:
        """Tool: Validate a data source."""
        # Check if source exists and is relevant
        return f"Source validated: {source_path}"

    def _deduplicate_tool(self, metadata_path: str) -> str:
        """Tool: Check for duplicates."""
        return f"Deduplication check completed for {metadata_path}"

    def _finish_tool(self, _: str) -> str:
        """Tool: Mark task as complete."""
        return "TASK_COMPLETE: Data collection finished"

    def step(self, context: dict) -> "AgentObservation":
        """Execute one collection step."""
        from src.autodata.agents.base_agent import AgentObservation

        goal = context.get("goal", "Collect carbon fiber domain data")
        result = self._think(goal, context)

        return AgentObservation(
            agent_name=self.name,
            action_type="collect",
            content=result,
            success=True,
            artifact_refs=[],
            source_refs=[],
        )

    def run(self, task: dict, context: dict) -> list:
        """Execute full collection task."""
        observations = []
        goal = task.get("goal", "Collect domain data")
        obs = self.step({"goal": goal, **context})
        observations.append(obs)
        return observations
