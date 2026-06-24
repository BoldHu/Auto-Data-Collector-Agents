"""Dynamic Task-Context Graph (DTCG) — Context Selector.

Implements the token-budgeted context selection algorithm:

For each agent a at step t, select a context subset S_a^t from the
graph neighborhood N(a,t) under a token budget B_a:

    Maximize:
        Σ relevance(v) + β Σ dependency(v) + γ Σ trust(v)
      - λ redundancy(S) - μ token_cost(S)

    Subject to:
        Σ token_cost(v) <= B_a

The selection uses a greedy MMR/knapsack approximation:
    1. Retrieve candidate nodes from the agent-task neighborhood
    2. Rank by relevance, dependency, recency, and trust
    3. Penalize redundancy with already-selected context
    4. Select until token budget is reached
    5. Compress selected nodes into a role-specific context package
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    EdgeType,
    Node,
    NodeType,
    Edge,
)


# ── Context Package Schema ───────────────────────────────────────────

@dataclass
class ContextPackage:
    """A context package constructed for a specific agent invocation.

    Contains only graph-selected context — agents should never receive
    full conversation history.
    """
    agent_name: str
    task_id: str
    current_goal: str
    allowed_tools: list[str] = field(default_factory=list)
    relevant_plan: str = ""
    selected_memory: list[dict[str, Any]] = field(default_factory=list)
    selected_artifacts: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    quality_requirements: list[dict[str, Any]] = field(default_factory=list)
    output_schema: dict[str, Any] = field(default_factory=dict)
    forbidden_actions: list[str] = field(default_factory=list)
    total_token_estimate: int = 0


# ── Context Selector ─────────────────────────────────────────────────

@dataclass
class ContextSelectorConfig:
    """Configuration for the context selection algorithm."""
    # Objective weights
    beta: float = 0.5       # dependency weight
    gamma: float = 0.4      # trust weight
    lam: float = 0.6        # redundancy penalty
    mu: float = 0.2         # token cost penalty

    # Edge weight hyperparameters (forwarded to Edge computation)
    alpha_relevance: float = 1.0
    alpha_dependency: float = 0.5
    alpha_recency: float = 0.3
    alpha_trust: float = 0.4
    alpha_redundancy: float = 0.6
    alpha_cost: float = 0.2

    # Recency half-life in seconds (for time decay)
    recency_half_life: float = 3600.0  # 1 hour

    # Default token budget per agent
    default_token_budget: int = 8000

    # Maximum number of context items
    max_context_items: int = 50


class ContextSelector:
    """Selects context for each agent from the DTCG neighborhood.

    Implements the greedy MMR/knapsack approximation for token-budgeted
    context selection, replacing broadcast-style multi-agent communication.
    """

    def __init__(self, config: Optional[ContextSelectorConfig] = None) -> None:
        self.config = config or ContextSelectorConfig()

    def select_context(
        self,
        graph: DynamicTaskContextGraph,
        agent_node_id: str,
        task_id: str,
        current_goal: str,
        token_budget: Optional[int] = None,
    ) -> ContextPackage:
        """Select context for an agent from its graph neighborhood.

        Steps:
        1. Retrieve candidate nodes from the agent-task neighborhood
        2. Compute edge weights (relevance, dependency, recency, trust)
        3. Score and rank candidates
        4. Apply greedy selection with redundancy penalty
        5. Compress selected nodes into a context package

        Args:
            graph: The current Dynamic Task-Context Graph
            agent_node_id: The agent's node ID in the graph
            task_id: The current task ID
            current_goal: The agent's current goal description
            token_budget: Maximum tokens for the context package

        Returns:
            A ContextPackage with graph-selected context
        """
        budget = token_budget or self.config.default_token_budget

        # Step 1: Get candidate nodes from neighborhood
        candidates = self._get_candidates(graph, agent_node_id)

        # Step 2: Score each candidate
        scored = self._score_candidates(graph, candidates, agent_node_id)

        # Step 3: Greedy selection with redundancy penalty
        selected = self._greedy_select(scored, budget)

        # Step 4: Build context package
        return self._build_package(
            graph=graph,
            agent_node_id=agent_node_id,
            task_id=task_id,
            current_goal=current_goal,
            selected_nodes=selected,
            token_budget=budget,
        )

    def _get_candidates(
        self,
        graph: DynamicTaskContextGraph,
        agent_node_id: str,
    ) -> list[Node]:
        """Retrieve candidate context nodes from the agent's neighborhood.

        Expands to 2-hop neighborhood: direct neighbors and their neighbors,
        then filters by relevance to the agent's current task.
        """
        visited = {agent_node_id}
        candidates = []

        # 1-hop neighbors
        direct = graph.get_neighbors(agent_node_id)
        for neighbor in direct:
            if neighbor.node_id not in visited:
                visited.add(neighbor.node_id)
                candidates.append(neighbor)

            # 2-hop neighbors
            for hop2 in graph.get_neighbors(neighbor.node_id):
                if hop2.node_id not in visited:
                    visited.add(hop2.node_id)
                    candidates.append(hop2)

        return candidates

    def _score_candidates(
        self,
        graph: DynamicTaskContextGraph,
        candidates: list[Node],
        agent_node_id: str,
    ) -> list[tuple[Node, float, dict[str, float]]]:
        """Score each candidate node using the objective function.

        For each candidate v, compute:
            score(v) = relevance(v) + β*dependency(v) + γ*trust(v)

        The detailed edge weight components are also computed.
        """
        current_time = time.time()
        scored = []

        for node in candidates:
            # Get the edge between agent and this candidate
            edges = graph.get_edges_of(agent_node_id)
            best_edge = None
            for e in edges:
                if e.source_id == node.node_id or e.target_id == node.node_id:
                    best_edge = e
                    break

            # Compute component scores
            relevance = 0.5  # Default; will be replaced by embedding similarity
            dependency = 0.0
            recency = 0.0
            trust = 0.0
            redundancy = 0.0
            cost = 0.0

            if best_edge:
                # Recency: exponential time decay
                age = current_time - node.updated_at
                recency = math.exp(
                    -math.log(2) * age / self.config.recency_half_life
                )

                # Use stored edge scores or defaults
                relevance = best_edge.relevance_score or relevance
                dependency = best_edge.dependency_score or dependency
                trust = best_edge.trust_score or trust
                cost = best_edge.cost_score or cost

                # Update edge scores
                best_edge.recency_score = recency
                best_edge.relevance_score = relevance
                best_edge.dependency_score = dependency
                best_edge.trust_score = trust
                best_edge.cost_score = cost

                # Compute the edge weight
                best_edge.compute_weight(current_time)

            # Node-level trust (from properties)
            if "quality_score" in node.properties:
                trust = max(trust, node.properties["quality_score"])

            # Estimate token cost
            estimated_tokens = self._estimate_token_cost(node)
            cost = estimated_tokens / self.config.default_token_budget

            # Objective: relevance + β*dependency + γ*trust
            score = (
                relevance
                + self.config.beta * dependency
                + self.config.gamma * trust
            )

            scored.append((node, score, {
                "relevance": relevance,
                "dependency": dependency,
                "recency": recency,
                "trust": trust,
                "redundancy": redundancy,
                "cost": cost,
                "estimated_tokens": estimated_tokens,
            }))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _greedy_select(
        self,
        scored: list[tuple[Node, float, dict[str, float]]],
        token_budget: int,
    ) -> list[tuple[Node, dict[str, float]]]:
        """Greedy MMR/knapsack selection with redundancy penalty.

        At each step, select the candidate with highest marginal gain,
        penalizing redundancy with already-selected items.

        Marginal gain for candidate v:
            gain(v) = score(v) - λ * max_redundancy(v, S)
                    - μ * token_cost(v)

        Where max_redundancy(v, S) is the maximum redundancy between v
        and any already-selected item in S.
        """
        selected: list[tuple[Node, dict[str, float]]] = []
        total_tokens = 0
        selected_names = set()

        for node, base_score, components in scored:
            est_tokens = components["estimated_tokens"]

            # Check token budget
            if total_tokens + est_tokens > token_budget:
                continue

            # Compute redundancy with selected items
            max_red = 0.0
            for sel_node, _ in selected:
                # Simple name-based redundancy check
                # In full implementation, use embedding cosine similarity
                if sel_node.name == node.name:
                    max_red = 1.0
                elif sel_node.node_type == node.node_type:
                    # Same type has some inherent redundancy
                    name_overlap = len(
                        set(sel_node.name.split()) & set(node.name.split())
                    )
                    max_red = max(max_red, min(name_overlap / max(len(node.name.split()), 1), 1.0))

            # Marginal gain with redundancy and cost penalty
            marginal_gain = (
                base_score
                - self.config.lam * max_red
                - self.config.mu * components["cost"]
            )

            # Only include if marginal gain is positive
            if marginal_gain > 0 and node.name not in selected_names:
                components["redundancy"] = max_red
                selected.append((node, components))
                total_tokens += est_tokens
                selected_names.add(node.name)

            if len(selected) >= self.config.max_context_items:
                break

        return selected

    def _estimate_token_cost(self, node: Node) -> int:
        """Estimate the token cost of including a node in context.

        Rough heuristic:
        - Agent nodes: ~200 tokens (name + role)
        - Task nodes: ~300 tokens (description + status)
        - Artifact nodes: ~500 tokens (path + summary)
        - Memory nodes: ~400 tokens (summary)
        - Tool nodes: ~150 tokens (name + description)
        - Constraint nodes: ~200 tokens (rule description)
        """
        base_costs = {
            NodeType.AGENT: 200,
            NodeType.TASK: 300,
            NodeType.ARTIFACT: 500,
            NodeType.MEMORY: 400,
            NodeType.TOOL: 150,
            NodeType.CONSTRAINT: 200,
        }
        return base_costs.get(node.node_type, 300)

    def _build_package(
        self,
        graph: DynamicTaskContextGraph,
        agent_node_id: str,
        task_id: str,
        current_goal: str,
        selected_nodes: list[tuple[Node, dict[str, float]]],
        token_budget: int,
    ) -> ContextPackage:
        """Construct a ContextPackage from selected nodes."""
        agent_node = graph.get_node(agent_node_id)
        agent_name = agent_node.name if agent_node else "unknown"

        memory_items = []
        artifact_items = []
        constraints = []
        quality_reqs = []
        tools = []
        total_tokens = 0

        for node, components in selected_nodes:
            item = {
                "node_id": node.node_id,
                "name": node.name,
                "type": node.node_type.value,
                "properties": node.properties,
                "scores": {k: round(v, 4) for k, v in components.items()},
            }
            total_tokens += components.get("estimated_tokens", 300)

            # Also expose content at top level for easy access
            if "content" in node.properties:
                item["content"] = node.properties["content"]

            if node.node_type == NodeType.MEMORY:
                memory_items.append(item)
            elif node.node_type == NodeType.ARTIFACT:
                artifact_items.append(item)
            elif node.node_type == NodeType.CONSTRAINT:
                constraints.append(item)
            elif node.node_type == NodeType.TOOL:
                tools.append(item)

        # Extract quality requirements from constraint nodes
        for c in constraints:
            if c.get("properties", {}).get("category") == "quality":
                quality_reqs.append(c)

        # Extract allowed tools
        allowed_tools = [t["name"] for t in tools]

        return ContextPackage(
            agent_name=agent_name,
            task_id=task_id,
            current_goal=current_goal,
            allowed_tools=allowed_tools,
            selected_memory=memory_items,
            selected_artifacts=artifact_items,
            constraints=constraints,
            quality_requirements=quality_reqs,
            total_token_estimate=total_tokens,
        )
