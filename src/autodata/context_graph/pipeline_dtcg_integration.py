"""DTCG Pipeline Integration for Phase 6.55 enhancement.

Lightweight integration layer that existing pipelines can use to generate
runtime DTCG traces without requiring full agent framework adoption.

Usage:
    from src.autodata.context_graph.pipeline_dtcg_integration import PipelineDTCG

    dtcg = PipelineDTCG("phase_3_image_labeling")
    dtcg.add_agent("ImageLabelingAgent", role="labeling")
    dtcg.add_task("label_images", status="in_progress")
    dtcg.add_artifact("image_labels.jsonl", derived_from="image_manifest.jsonl")
    dtcg.connect_agent_to_task("ImageLabelingAgent", "label_images")
    dtcg.save()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    NodeType,
    EdgeType,
    Node,
    Edge,
)
from src.autodata.context_graph.context_selector import (
    ContextSelector,
    ContextSelectorConfig,
    ContextPackage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


class PipelineDTCG:
    """Lightweight DTCG integration for pipelines.

    Provides a simple API for pipelines to create DTCG nodes, edges,
    and context packages without requiring full agent framework adoption.
    """

    def __init__(self, phase_name: str, report_dir: Optional[Path] = None):
        self.phase_name = phase_name
        self.graph = DynamicTaskContextGraph()
        self.report_dir = report_dir or (PROJECT_ROOT / "data" / "reports" / phase_name)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._node_counter = 0
        self._edge_counter = 0

    def _next_node_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter:04d}"

    def _next_edge_id(self) -> str:
        self._edge_counter += 1
        return f"e_{self._edge_counter:04d}"

    # ── Node creation ─────────────────────────────────────────────

    def add_agent(self, name: str, role: str = "", **props) -> str:
        """Add an agent node. Returns node_id."""
        node_id = self._next_node_id("agent")
        props["role"] = role
        self.graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.AGENT,
            name=name,
            properties=props,
        ))
        return node_id

    def add_task(self, name: str, status: str = "pending", **props) -> str:
        """Add a task node. Returns node_id."""
        node_id = self._next_node_id("task")
        props["status"] = status
        self.graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.TASK,
            name=name,
            properties=props,
        ))
        return node_id

    def add_artifact(self, name: str, path: str = "", derived_from: str = "", **props) -> str:
        """Add an artifact node. Returns node_id."""
        node_id = self._next_node_id("artifact")
        props["path"] = path
        if derived_from:
            props["derived_from"] = derived_from
        self.graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.ARTIFACT,
            name=name,
            properties=props,
        ))
        return node_id

    def add_memory(self, name: str, content: str = "", **props) -> str:
        """Add a memory node. Returns node_id."""
        node_id = self._next_node_id("memory")
        props["content"] = content[:200]
        self.graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.MEMORY,
            name=name,
            properties=props,
        ))
        return node_id

    def add_tool(self, name: str, **props) -> str:
        """Add a tool node. Returns node_id."""
        node_id = self._next_node_id("tool")
        self.graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.TOOL,
            name=name,
            properties=props,
        ))
        return node_id

    def add_constraint(self, name: str, **props) -> str:
        """Add a constraint node. Returns node_id."""
        node_id = self._next_node_id("constraint")
        self.graph.add_node(Node(
            node_id=node_id,
            node_type=NodeType.CONSTRAINT,
            name=name,
            properties=props,
        ))
        return node_id

    # ── Edge creation ─────────────────────────────────────────────

    def connect_agent_to_task(self, agent_node_id: str, task_node_id: str) -> str:
        """Connect agent to task via agent_assignment edge."""
        edge_id = self._next_edge_id()
        self.graph.add_edge(Edge(
            edge_id=edge_id,
            source_id=agent_node_id,
            target_id=task_node_id,
            edge_type=EdgeType.AGENT_ASSIGNMENT,
        ))
        return edge_id

    def connect_task_dependency(self, from_task: str, to_task: str) -> str:
        """Connect tasks via task_dependency edge."""
        edge_id = self._next_edge_id()
        self.graph.add_edge(Edge(
            edge_id=edge_id,
            source_id=from_task,
            target_id=to_task,
            edge_type=EdgeType.TASK_DEPENDENCY,
        ))
        return edge_id

    def connect_artifact_derived(self, source: str, target: str) -> str:
        """Connect artifact derived_from relationship."""
        edge_id = self._next_edge_id()
        self.graph.add_edge(Edge(
            edge_id=edge_id,
            source_id=source,
            target_id=target,
            edge_type=EdgeType.ARTIFACT_DERIVED_FROM,
        ))
        return edge_id

    def connect_quality_feedback(self, constraint_id: str, task_id: str) -> str:
        """Connect constraint quality feedback to task."""
        edge_id = self._next_edge_id()
        self.graph.add_edge(Edge(
            edge_id=edge_id,
            source_id=constraint_id,
            target_id=task_id,
            edge_type=EdgeType.QUALITY_FEEDBACK,
        ))
        return edge_id

    def connect_tool_usage(self, tool_id: str, task_id: str) -> str:
        """Connect tool usage to task."""
        edge_id = self._next_edge_id()
        self.graph.add_edge(Edge(
            edge_id=edge_id,
            source_id=tool_id,
            target_id=task_id,
            edge_type=EdgeType.TOOL_USAGE,
        ))
        return edge_id

    # ── Context selection ─────────────────────────────────────────

    def select_context(
        self,
        agent_node_id: str,
        task_id: str,
        current_goal: str,
        token_budget: int = 4000,
    ) -> ContextPackage:
        """Select context for an agent using DTCG."""
        config = ContextSelectorConfig(default_token_budget=token_budget)
        selector = ContextSelector(config)
        return selector.select_context(
            graph=self.graph,
            agent_node_id=agent_node_id,
            task_id=task_id,
            current_goal=current_goal,
        )

    # ── Save/Export ───────────────────────────────────────────────

    def save(self) -> tuple[str, str]:
        """Save DTCG trace and context packages.

        Returns:
            (trace_path, packages_path)
        """
        # Save trace
        trace_data = self.graph.to_dict()
        trace_path = self.report_dir / f"dtcg_{self.phase_name}_trace.json"
        with open(trace_path, "w") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        # Generate and save context packages for all agent nodes
        packages = []
        config = ContextSelectorConfig(default_token_budget=4000)
        selector = ContextSelector(config)

        trace_dict = self.graph.to_dict()
        for node_id, node in trace_dict.get("nodes", {}).items():
            if node.get("node_type") == "agent":
                # Find associated tasks
                for edge_id, edge in trace_dict.get("edges", {}).items():
                    if edge.get("source_id") == node_id and edge.get("edge_type") == "agent_assignment":
                        try:
                            pkg = selector.select_context(
                                graph=self.graph,
                                agent_node_id=node_id,
                                task_id=edge["target_id"],
                                current_goal=node.get("properties", {}).get("role", ""),
                            )
                            packages.append({
                                "agent": node["name"],
                                "task": edge["target_id"],
                                "memory": len(pkg.selected_memory),
                                "artifacts": len(pkg.selected_artifacts),
                                "token_estimate": pkg.total_token_estimate,
                            })
                        except Exception:
                            pass

        packages_path = self.report_dir / f"context_packages_{self.phase_name}.jsonl"
        with open(packages_path, "w") as f:
            for pkg in packages:
                f.write(json.dumps(pkg, ensure_ascii=False) + "\n")

        # Save statistics
        stats = {
            "phase": self.phase_name,
            "timestamp": time.time(),
            "node_count": len(trace_data["nodes"]),
            "edge_count": len(trace_data["edges"]),
            "context_packages": len(packages),
        }
        stats_path = self.report_dir / f"dtcg_{self.phase_name}_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return str(trace_path), str(packages_path)
