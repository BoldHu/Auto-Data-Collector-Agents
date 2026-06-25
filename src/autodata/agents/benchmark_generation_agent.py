"""Benchmark Generation Agent — builds and validates benchmark items.

Assembles benchmark items from cleaned text, image labels, and exam questions.
Validates item quality, splits into subsets, and generates statistics.
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

logger = get_logger("benchmark_generation_agent")


class BenchmarkGenerationAgent(ReActAgent):
    """Agent for generating and validating benchmark items.

    Capabilities:
    - Load benchmark candidate items from phase artifacts
    - Validate items for completeness and quality
    - Split benchmark into evaluation subsets
    - Generate statistics and distribution reports

    This agent operates on existing artifacts produced by prior pipeline phases.
    """

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store: Optional[MessageStore] = None,
        run_id: str = "benchmark_generation",
        output_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="BenchmarkGenerationAgent",
            model_client=model_client,
            message_store=message_store,
            max_iterations=10,
        )
        self.graph = graph or DynamicTaskContextGraph()
        self.run_id = run_id
        self.output_path = output_path
        self._items_processed = 0

        # Register tools
        self.tool_registry.register(
            "load_candidates",
            "Load benchmark candidate items from a JSONL file",
            self._load_candidates_tool,
        )
        self.tool_registry.register(
            "validate_items",
            "Validate benchmark items for required fields and quality",
            self._validate_items_tool,
        )
        self.tool_registry.register(
            "compute_statistics",
            "Compute distribution statistics for benchmark items",
            self._compute_statistics_tool,
        )
        self.tool_registry.register(
            "split_benchmark",
            "Split benchmark into evaluation subsets by criteria",
            self._split_tool,
        )
        self.tool_registry.register(
            "finish",
            "Mark benchmark construction as complete",
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
                "role": "benchmark_generation",
            },
        )
        self.graph.add_node(node)

    def _load_candidates_tool(self, jsonl_path: str) -> str:
        """Load benchmark candidate items from a JSONL file."""
        path = Path(jsonl_path)
        if not path.exists():
            return f"Error: file not found at {jsonl_path}"

        items = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        self._items_processed += len(items)

        # Register artifact node
        artifact_node = Node(
            node_id=f"art_bench_candidates_{path.stem}",
            node_type=NodeType.ARTIFACT,
            name=f"Benchmark candidates: {path.name}",
            properties={
                "path": str(path),
                "record_count": len(items),
                "source_type": "benchmark_candidates",
            },
        )
        self.graph.add_node(artifact_node)

        # Compute basic stats
        task_types = {}
        for item in items:
            tt = item.get("task_type", "unknown")
            task_types[tt] = task_types.get(tt, 0) + 1

        stats = ", ".join(f"{k}: {v}" for k, v in sorted(task_types.items(), key=lambda x: -x[1])[:10])
        return f"Loaded {len(items)} candidates from {path.name}. Task types: {stats}"

    def _validate_items_tool(self, items_path: str) -> str:
        """Validate benchmark items for required fields and quality."""
        path = Path(items_path)
        if not path.exists():
            return f"Error: file not found at {items_path}"

        total = 0
        valid = 0
        issues = {"missing_question": 0, "missing_answer": 0, "missing_task_type": 0, "invalid_json": 0}

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        issues["invalid_json"] += 1
                        continue

                    if not item.get("question"):
                        issues["missing_question"] += 1
                    if not item.get("answer") and not item.get("options"):
                        issues["missing_answer"] += 1
                    if not item.get("task_type"):
                        issues["missing_task_type"] += 1

                    if item.get("question") and (item.get("answer") or item.get("options")):
                        valid += 1
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        issue_summary = ", ".join(f"{k}: {v}" for k, v in issues.items() if v > 0)
        return f"Validation: {total} items, {valid} valid, issues: {issue_summary or 'none'}"

    def _compute_statistics_tool(self, items_path: str) -> str:
        """Compute distribution statistics for benchmark items."""
        path = Path(items_path)
        if not path.exists():
            return f"Error: file not found at {items_path}"

        items = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        if not items:
            return "No items to analyze"

        # Compute distributions
        task_types = {}
        difficulties = {}
        modalities = {}
        sources = {}

        for item in items:
            tt = item.get("task_type", "unknown")
            task_types[tt] = task_types.get(tt, 0) + 1

            diff = item.get("difficulty", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1

            mod = item.get("modality", "text")
            modalities[mod] = modalities.get(mod, 0) + 1

            src = item.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        result = {
            "total": len(items),
            "task_type_distribution": dict(sorted(task_types.items(), key=lambda x: -x[1])),
            "difficulty_distribution": difficulties,
            "modality_distribution": modalities,
            "source_distribution": dict(sorted(sources.items(), key=lambda x: -x[1])),
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _split_tool(self, params: str) -> str:
        """Split benchmark into evaluation subsets."""
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            config = {"input_path": params}

        input_path = config.get("input_path", "")
        split_ratio = config.get("split_ratio", 0.2)
        seed = config.get("seed", 42)

        path = Path(input_path)
        if not path.exists():
            return f"Error: file not found at {input_path}"

        items = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        # Simple deterministic split
        import random
        rng = random.Random(seed)
        indices = list(range(len(items)))
        rng.shuffle(indices)

        split_idx = int(len(items) * (1 - split_ratio))
        train_indices = sorted(indices[:split_idx])
        test_indices = sorted(indices[split_idx:])

        return json.dumps({
            "total": len(items),
            "train_size": len(train_indices),
            "test_size": len(test_indices),
            "split_ratio": split_ratio,
            "seed": seed,
        })

    def _finish_tool(self, _: str) -> str:
        """Mark task as complete."""
        return f"TASK_COMPLETE: Benchmark generation finished. {self._items_processed} items processed."

    def run(self, task: str, context: Optional[dict] = None) -> list:
        """Execute benchmark generation task using the ReAct loop."""
        return super().run(task=task, context=context)
