"""Data Collection Agent — handles data acquisition tasks.

Collects images, papers, and metadata from available sources.
Registers existing crawler tools and source manifests.
Inherits from ReActAgent for DTCG integration.
"""

from __future__ import annotations

import json
import os
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

logger = get_logger("data_collection_agent")


class DataCollectionAgent(ReActAgent):
    """Agent for collecting domain data from various sources.

    Capabilities:
    - Register existing image manifests and metadata
    - Validate source files for existence and relevance
    - Check metadata for duplicates against existing corpus
    - Track provenance for collected artifacts

    This agent wraps existing crawler/index/manifest logic rather than
    performing live web crawling, which requires external credentials.
    """

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store: Optional[MessageStore] = None,
        run_id: str = "data_collection",
        output_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="DataCollectionAgent",
            model_client=model_client,
            message_store=message_store,
            max_iterations=10,
        )
        self.graph = graph or DynamicTaskContextGraph()
        self.run_id = run_id
        self.output_path = output_path
        self._collected_count = 0

        # Register tools
        self.tool_registry.register(
            "register_manifest",
            "Register an existing data manifest (JSONL) as a collection artifact",
            self._register_manifest_tool,
        )
        self.tool_registry.register(
            "validate_source",
            "Validate a data source file exists and is non-empty",
            self._validate_source_tool,
        )
        self.tool_registry.register(
            "list_sources",
            "List available raw data sources in a directory",
            self._list_sources_tool,
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
                "role": "data_collection",
            },
        )
        self.graph.add_node(node)

    def _register_manifest_tool(self, manifest_path: str) -> str:
        """Register an existing data manifest as a collection artifact.

        Args:
            manifest_path: Path to a JSONL manifest file.

        Returns:
            Summary of registered records.
        """
        path = Path(manifest_path)
        if not path.exists():
            return f"Error: manifest not found at {manifest_path}"

        count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        count += 1
        except Exception as e:
            return f"Error reading manifest: {str(e)[:100]}"

        # Register artifact node in DTCG
        artifact_node = Node(
            node_id=f"art_manifest_{path.stem}",
            node_type=NodeType.ARTIFACT,
            name=f"Data manifest: {path.name}",
            properties={
                "path": str(path),
                "record_count": count,
                "source_type": "manifest",
            },
        )
        self.graph.add_node(artifact_node)

        self._collected_count += count

        # Send message
        self.send_message(
            receiver="CentralPlanningAgent",
            content=f"Registered manifest {path.name}: {count} records",
            task_id=self.run_id,
            message_type=MessageType.OBSERVATION,
            visibility=Visibility.LOCAL,
        )

        return f"Registered manifest {path.name} with {count} records"

    def _validate_source_tool(self, source_path: str) -> str:
        """Validate a data source file exists and is non-empty."""
        path = Path(source_path)
        if not path.exists():
            return f"Source not found: {source_path}"
        if path.is_file() and path.stat().st_size == 0:
            return f"Source is empty: {source_path}"
        if path.is_dir():
            files = list(path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            return f"Directory {source_path}: {file_count} files found"
        return f"Source valid: {source_path} ({path.stat().st_size} bytes)"

    def _list_sources_tool(self, directory: str) -> str:
        """List available raw data sources in a directory."""
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return f"Directory not found: {directory}"

        entries = []
        for item in sorted(path.iterdir()):
            if item.is_file():
                entries.append(f"  file: {item.name} ({item.stat().st_size} bytes)")
            elif item.is_dir():
                sub_count = sum(1 for _ in item.rglob("*") if _.is_file())
                entries.append(f"  dir:  {item.name}/ ({sub_count} files)")

        if not entries:
            return f"Directory {directory} is empty"

        return f"Sources in {directory}:\n" + "\n".join(entries[:50])

    def _deduplicate_tool(self, metadata_path: str) -> str:
        """Check metadata for duplicates against existing corpus."""
        path = Path(metadata_path)
        if not path.exists():
            return f"Metadata file not found: {metadata_path}"

        seen_hashes = set()
        duplicates = 0
        total = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        record = json.loads(line)
                        h = record.get("content_hash") or record.get("image_hash")
                        if h and h in seen_hashes:
                            duplicates += 1
                        elif h:
                            seen_hashes.add(h)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            return f"Error reading metadata: {str(e)[:100]}"

        return f"Deduplication: {total} records, {duplicates} duplicates found ({len(seen_hashes)} unique)"

    def _finish_tool(self, _: str) -> str:
        """Mark task as complete."""
        return f"TASK_COMPLETE: Data collection finished. {self._collected_count} records registered."

    def run(self, task: str, context: Optional[dict] = None) -> list:
        """Execute collection task using the ReAct loop.

        Args:
            task: Task description (e.g., "Register image manifests").
            context: Optional context from DTCG.

        Returns:
            List of observations from the ReAct loop.
        """
        # Call parent ReActAgent.run() for proper ReAct loop
        return super().run(task=task, context=context)
