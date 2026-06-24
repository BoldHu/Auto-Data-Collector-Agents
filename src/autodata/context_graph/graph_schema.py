"""Dynamic Task-Context Graph (DTCG) — Graph Schema.

Defines the heterogeneous graph structure G_t = (V_t, E_t) with
typed nodes, typed edges, and dynamic edge weights.

Node types:
  - agent: worker agents and central planner
  - task: current, subtask, pending, completed
  - artifact: files, documents, samples, benchmark items, evaluation results
  - memory: summarized context, prior decisions, constraints, error reports
  - tool: crawler, OCR, parser, LLM API caller, evaluator, deduplicator
  - constraint: domain constraints, quality rules, benchmark rules, provenance rules

Edge types:
  - task_dependency: task -> task dependency
  - agent_assignment: agent <-> task assignment
  - artifact_derived_from: artifact provenance
  - context_relevance: semantic/contextual links
  - quality_feedback: quality verification feedback
  - tool_usage: agent/tool usage links
  - duplication_conflict: redundancy or conflict detection
  - benchmark_source: benchmark item source tracing
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Node Types ───────────────────────────────────────────────────────

class NodeType(str, Enum):
    AGENT = "agent"
    TASK = "task"
    ARTIFACT = "artifact"
    MEMORY = "memory"
    TOOL = "tool"
    CONSTRAINT = "constraint"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Edge Types ───────────────────────────────────────────────────────

class EdgeType(str, Enum):
    TASK_DEPENDENCY = "task_dependency"
    AGENT_ASSIGNMENT = "agent_assignment"
    ARTIFACT_DERIVED_FROM = "artifact_derived_from"
    CONTEXT_RELEVANCE = "context_relevance"
    QUALITY_FEEDBACK = "quality_feedback"
    TOOL_USAGE = "tool_usage"
    DUPLICATION_CONFLICT = "duplication_conflict"
    BENCHMARK_SOURCE = "benchmark_source"


# ── Node ─────────────────────────────────────────────────────────────

@dataclass
class Node:
    """A node in the Dynamic Task-Context Graph."""
    node_id: str
    node_type: NodeType
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    embedding_id: Optional[str] = None

    @staticmethod
    def new(node_type: NodeType, name: str, **properties) -> "Node":
        """Create a new node with auto-generated ID."""
        return Node(
            node_id=uuid.uuid4().hex[:12],
            node_type=node_type,
            name=name,
            properties=properties,
        )


# ── Edge with Dynamic Weight ────────────────────────────────────────

@dataclass
class Edge:
    """An edge in the Dynamic Task-Context Graph with dynamic weight.

    The weight is computed as:

        w_ij^(t) = sigmoid(
            α1 * Rel(i,j)
          + α2 * Dep(i,j)
          + α3 * Rec(i,j,t)
          + α4 * Trust(i,j)
          - α5 * Red(i,j)
          - α6 * Cost(j)
        )

    Where:
        Rel(i,j)  = semantic relevance between agent/task query and node j
        Dep(i,j)  = task dependency strength
        Rec(i,j,t)= recency with time decay
        Trust(i,j)= source quality / verification score
        Red(i,j)  = redundancy with already-selected context
        Cost(j)   = estimated token or computation cost
    """
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # Weight component scores (populated by context_selector)
    relevance_score: float = 0.0     # Rel(i,j)
    dependency_score: float = 0.0    # Dep(i,j)
    recency_score: float = 0.0       # Rec(i,j,t)
    trust_score: float = 0.0         # Trust(i,j)
    redundancy_score: float = 0.0    # Red(i,j)
    cost_score: float = 0.0          # Cost(j)

    # Weight hyperparameters
    ALPHA_RELEVANCE: float = 1.0     # α1
    ALPHA_DEPENDENCY: float = 0.5    # α2
    ALPHA_RECENCY: float = 0.3       # α3
    ALPHA_TRUST: float = 0.4         # α4
    ALPHA_REDUNDANCY: float = 0.6    # α5
    ALPHA_COST: float = 0.2          # α6

    @staticmethod
    def new(
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        **properties,
    ) -> "Edge":
        """Create a new edge with auto-generated ID."""
        return Edge(
            edge_id=uuid.uuid4().hex[:12],
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=properties,
        )

    def compute_weight(self, current_time: Optional[float] = None) -> float:
        """Compute the dynamic edge weight using the sigmoid formula."""
        import math

        if current_time is None:
            current_time = time.time()

        logit = (
            self.ALPHA_RELEVANCE * self.relevance_score
            + self.ALPHA_DEPENDENCY * self.dependency_score
            + self.ALPHA_RECENCY * self.recency_score
            + self.ALPHA_TRUST * self.trust_score
            - self.ALPHA_REDUNDANCY * self.redundancy_score
            - self.ALPHA_COST * self.cost_score
        )
        self.weight = 1.0 / (1.0 + math.exp(-logit))
        return self.weight


# ── Dynamic Heterogeneous Graph ──────────────────────────────────────

class DynamicTaskContextGraph:
    """Dynamic Task-Context Graph G_t = (V_t, E_t).

    Maintains the evolving graph structure across time steps,
    supporting node/edge insertion, deletion, and neighborhood queries.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        # Adjacency: node_id -> set of edge_ids
        self.adj: dict[str, set[str]] = {}
        # Time step counter
        self.t: int = 0

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj:
            self.adj[node.node_id] = set()

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph (both directions tracked)."""
        self.edges[edge.edge_id] = edge
        self.adj.setdefault(edge.source_id, set()).add(edge.edge_id)
        self.adj.setdefault(edge.target_id, set()).add(edge.edge_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id in self.adj:
            edge_ids = list(self.adj[node_id])
            for eid in edge_ids:
                self.remove_edge(eid)
            del self.adj[node_id]
        self.nodes.pop(node_id, None)

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge from the graph."""
        edge = self.edges.pop(edge_id, None)
        if edge:
            self.adj.get(edge.source_id, set()).discard(edge_id)
            self.adj.get(edge.target_id, set()).discard(edge_id)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> list[Node]:
        """Get all neighbor nodes of a given node."""
        neighbor_ids = set()
        for eid in self.adj.get(node_id, set()):
            edge = self.edges.get(eid)
            if edge:
                if edge.source_id == node_id:
                    neighbor_ids.add(edge.target_id)
                else:
                    neighbor_ids.add(edge.source_id)
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_edges_of(self, node_id: str) -> list[Edge]:
        """Get all edges incident to a node."""
        return [
            self.edges[eid]
            for eid in self.adj.get(node_id, set())
            if eid in self.edges
        ]

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Get all nodes of a given type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_edges_by_type(self, edge_type: EdgeType) -> list[Edge]:
        """Get all edges of a given type."""
        return [e for e in self.edges.values() if e.edge_type == edge_type]

    def advance_time(self) -> None:
        """Advance the graph to the next time step."""
        self.t += 1

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a dictionary for persistence."""
        return {
            "t": self.t,
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "node_type": n.node_type.value,
                    "name": n.name,
                    "properties": n.properties,
                    "created_at": n.created_at,
                    "updated_at": n.updated_at,
                    "embedding_id": n.embedding_id,
                }
                for nid, n in self.nodes.items()
            },
            "edges": {
                eid: {
                    "edge_id": e.edge_id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                    "properties": e.properties,
                    "relevance_score": e.relevance_score,
                    "dependency_score": e.dependency_score,
                    "recency_score": e.recency_score,
                    "trust_score": e.trust_score,
                    "redundancy_score": e.redundancy_score,
                    "cost_score": e.cost_score,
                }
                for eid, e in self.edges.items()
            },
        }
