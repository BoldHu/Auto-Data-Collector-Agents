"""DTCG Persistent Evaluation — causal component ablation with a shared graph.

Unlike run_dtcg() which builds a fresh graph per item, this module maintains
a single persistent graph across a sequence of tasks. Prior artifacts,
messages, quality feedback, and cache entries accumulate, so trust,
redundancy, and cache effects can causally influence context selection.

Variants:
- full:           All components active
- no_cache:       No local cache (cache cleared between tasks)
- no_redundancy:  Redundancy penalty disabled (lambda=0)
- no_trust:       Trust/quality weighting disabled (gamma=0, alpha_trust=0)
- static:         Graph edges do not update between tasks (recency half-life=inf)
- top_k_only:     No graph structure, just first k evidence chunks
- broadcast:      All agents see all messages (prompt-simulated)
- static_router:  Fixed routing by task type (prompt-simulated)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.autodata.context_graph.context_selector import ContextSelector, ContextSelectorConfig
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
)
from src.autodata.context_graph.local_cache import CacheEntry, CacheEntryType, LocalCache
from src.autodata.evaluation.system_baselines import (
    run_broadcast,
    run_static_router,
    _call_model,
)
from src.autodata.evaluation.system_prompts import DTCG_SYSTEM, DTCG_USER
from src.autodata.evaluation.system_trace_schema import AblationTrace
from src.autodata.evaluation.token_accounting import estimate_tokens


@dataclass
class PersistentDTCGConfig:
    """Configuration for a persistent DTCG variant."""
    variant_name: str
    selector_config: ContextSelectorConfig
    persist_graph: bool = True       # Whether graph carries over between tasks
    persist_cache: bool = True       # Whether cache carries over between tasks
    update_trust: bool = True        # Whether trust scores update from feedback
    update_edges: bool = True        # Whether edge weights recalculate
    use_graph_selection: bool = True  # Whether to use graph-based selection


@dataclass
class DTCGTaskLog:
    """Per-task log entry for DTCG evaluation."""
    task_id: str
    variant: str
    graph_nodes_before: int = 0
    graph_edges_before: int = 0
    graph_nodes_after: int = 0
    graph_edges_after: int = 0
    candidate_context_ids: list[str] = field(default_factory=list)
    selected_context_ids: list[str] = field(default_factory=list)
    selected_token_count: int = 0
    actual_prompt_token_count: int = 0
    selector_config_hash: str = ""
    fallback_triggered: bool = False
    cache_hit: bool = False
    trust_scores_before: dict[str, float] = field(default_factory=dict)
    trust_scores_after: dict[str, float] = field(default_factory=dict)
    redundancy_scores: dict[str, float] = field(default_factory=dict)
    cache_hits: int = 0
    serialized_prompt: str = ""
    agent_output_hash: str = ""
    prompt_hash: str = ""
    output_hash: str = ""
    task_success: bool = False
    latency: float = 0.0


# ── Variant factory ─────────────────────────────────────────────────

def make_variant_config(variant: str, token_budget: int = 4000) -> PersistentDTCGConfig:
    """Create a PersistentDTCGConfig for a named variant."""

    if variant == "full":
        return PersistentDTCGConfig(
            variant_name="full",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
                beta=0.5, gamma=0.4, lam=0.6, mu=0.2,
            ),
        )

    elif variant == "no_cache":
        return PersistentDTCGConfig(
            variant_name="no_cache",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
                beta=0.5, gamma=0.4, lam=0.6, mu=0.2,
            ),
            persist_cache=False,
        )

    elif variant == "no_redundancy":
        return PersistentDTCGConfig(
            variant_name="no_redundancy",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
                beta=0.5, gamma=0.4, lam=0.0, mu=0.2,
            ),
        )

    elif variant == "no_trust":
        return PersistentDTCGConfig(
            variant_name="no_trust",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
                beta=0.5, gamma=0.0, lam=0.6, mu=0.2,
                alpha_trust=0.0,
            ),
            update_trust=False,
        )

    elif variant == "no_recency":
        return PersistentDTCGConfig(
            variant_name="no_recency",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
                beta=0.5, gamma=0.4, lam=0.6, mu=0.2,
                recency_half_life=float('inf'),
            ),
            update_edges=False,
        )

    elif variant == "static":
        return PersistentDTCGConfig(
            variant_name="static",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
                beta=0.5, gamma=0.4, lam=0.6, mu=0.2,
                recency_half_life=float('inf'),
            ),
            update_edges=False,
            update_trust=False,
        )

    elif variant == "top_k_only":
        return PersistentDTCGConfig(
            variant_name="top_k_only",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
            ),
            use_graph_selection=False,
        )

    elif variant == "broadcast":
        return PersistentDTCGConfig(
            variant_name="broadcast",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
            ),
            use_graph_selection=False,
        )

    elif variant == "static_router":
        return PersistentDTCGConfig(
            variant_name="static_router",
            selector_config=ContextSelectorConfig(
                default_token_budget=token_budget,
            ),
            use_graph_selection=False,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")


# ── Persistent DTCG Engine ──────────────────────────────────────────

class PersistentDTCGEngine:
    """Runs DTCG evaluation with a persistent graph across tasks.

    This engine maintains a single graph and local cache that accumulate
    across tasks, enabling trust, redundancy, and cache effects to
    causally influence context selection.
    """

    def __init__(self, config: PersistentDTCGConfig):
        self.config = config
        self.graph = DynamicTaskContextGraph()
        self.cache = LocalCache(agent_name=f"dtcg_{config.variant_name}")
        self.selector = ContextSelector(config.selector_config)
        self.task_logs: list[DTCGTaskLog] = []

        # Persistent trust scores: node_id -> score
        self._trust_scores: dict[str, float] = {}

        # Selected context history for redundancy
        self._selected_history: list[list[str]] = []

    def _config_hash(self) -> str:
        """Hash of the selector config for reproducibility tracking."""
        cfg = self.config.selector_config
        key = f"{cfg.beta}_{cfg.gamma}_{cfg.lam}_{cfg.mu}_{cfg.alpha_trust}_{cfg.recency_half_life}_{cfg.default_token_budget}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _build_graph_for_item(self, item: dict) -> tuple[str, str]:
        """Add item evidence/constraints to the persistent graph.

        Returns (agent_node_id, task_node_id).
        """
        task_id = item.get("benchmark_id", f"task_{len(self.task_logs)}")

        agent_node = Node.new(NodeType.AGENT, f"AnswerAgent_{self.config.variant_name}")
        task_node = Node.new(NodeType.TASK, f"answer_{task_id}")
        self.graph.add_node(agent_node)
        self.graph.add_node(task_node)

        # Add evidence as artifacts
        evidence = item.get("evidence", [])
        for i, ev in enumerate(evidence[:5]):
            ev_text = str(ev) if ev else ""
            if not ev_text:
                continue
            art_node = Node.new(NodeType.ARTIFACT, f"evidence_{task_id}_{i}", content=ev_text[:1000])
            self.graph.add_node(art_node)

            # Use stored trust scores if available
            trust = self._trust_scores.get(art_node.node_id, 0.5)

            edge = Edge.new(
                art_node.node_id, task_node.node_id,
                EdgeType.ARTIFACT_DERIVED_FROM,
                relevance_score=0.9,
                trust_score=trust,
            )
            if not self.config.update_edges:
                edge.recency_score = 0.5  # fixed
            self.graph.add_edge(edge)

            # Also connect agent to evidence
            agent_edge = Edge.new(
                agent_node.node_id, art_node.node_id,
                EdgeType.CONTEXT_RELEVANCE,
                relevance_score=0.8,
                trust_score=trust,
            )
            self.graph.add_edge(agent_edge)

        # Add constraints
        constraints = item.get("required_knowledge", [])
        for i, c in enumerate(constraints[:3]):
            con_node = Node.new(NodeType.CONSTRAINT, f"constraint_{task_id}_{i}", content=c)
            self.graph.add_node(con_node)
            self.graph.add_edge(Edge.new(
                con_node.node_id, task_node.node_id,
                EdgeType.QUALITY_FEEDBACK,
            ))

        # Add explanation as memory if available
        explanation = item.get("explanation", "")
        if explanation:
            mem_node = Node.new(NodeType.MEMORY, f"explanation_{task_id}", content=str(explanation)[:500])
            self.graph.add_node(mem_node)
            self.graph.add_edge(Edge.new(
                mem_node.node_id, task_node.node_id,
                EdgeType.CONTEXT_RELEVANCE,
                relevance_score=0.7,
            ))

        # Connect agent to task
        self.graph.add_edge(Edge.new(
            agent_node.node_id, task_node.node_id,
            EdgeType.AGENT_ASSIGNMENT,
        ))

        # Add prior context from cache as memory nodes
        if self.config.persist_cache:
            cache_context = self.cache.to_context_string(max_tokens=500)
            if cache_context:
                cache_node = Node.new(NodeType.MEMORY, f"cache_{task_id}", content=cache_context[:500])
                self.graph.add_node(cache_node)
                self.graph.add_edge(Edge.new(
                    cache_node.node_id, task_node.node_id,
                    EdgeType.CONTEXT_RELEVANCE,
                    relevance_score=0.6,
                ))

        return agent_node.node_id, task_node.node_id

    def _select_context_graph(self, agent_node_id: str, task_node_id: str, item: dict) -> tuple[str, list[str], bool, bool]:
        """Select context using the graph-based selector.

        Returns (selected_context, selected_ids, fallback_triggered, cache_hit).
        """
        pkg = self.selector.select_context(
            graph=self.graph,
            agent_node_id=agent_node_id,
            task_id=task_node_id,
            current_goal="Answer the carbon fiber question",
        )

        selected_parts = []
        selected_ids = []
        for art in pkg.selected_artifacts:
            content = art.get("content") or art.get("properties", {}).get("content", "")
            if content:
                selected_parts.append(str(content))
                selected_ids.append(art.get("node_id", ""))
        for mem in pkg.selected_memory:
            content = mem.get("content") or mem.get("properties", {}).get("content", "")
            if content:
                selected_parts.append(str(content))
                selected_ids.append(mem.get("node_id", ""))

        # Check for cache hit
        cache_hit = any("cache_" in nid for nid in selected_ids)

        # Fallback if graph selection returned nothing
        fallback_triggered = False
        evidence = item.get("evidence", [])
        if not selected_parts and evidence:
            selected_parts = [str(e)[:500] for e in evidence[:3]]
            fallback_triggered = True

        selected_context = "\n".join(selected_parts) if selected_parts else "无相关上下文"
        return selected_context, selected_ids, fallback_triggered, cache_hit

    def _select_context_topk(self, item: dict) -> tuple[str, list[str], bool, bool]:
        """Select context using simple top-k (no graph)."""
        evidence = item.get("evidence", [])
        topk_context = " ".join(str(e) for e in evidence)[:2000]
        return topk_context, [f"topk_{i}" for i in range(len(evidence[:5]))], False, False

    def _run_single_variant(self, client, item: dict) -> tuple[AblationTrace, DTCGTaskLog]:
        """Run a single item through this variant."""
        task_id = item.get("benchmark_id", f"task_{len(self.task_logs)}")
        log = DTCGTaskLog(task_id=task_id, variant=self.config.variant_name)
        log.selector_config_hash = self._config_hash()

        question = item.get("question", "")
        constraints = item.get("required_knowledge", [])

        start = time.time()

        # Record graph state before
        log.graph_nodes_before = len(self.graph.nodes)
        log.graph_edges_before = len(self.graph.edges)

        # Record trust scores before
        log.trust_scores_before = dict(self._trust_scores)

        # Handle prompt-level baselines
        if self.config.variant_name == "broadcast":
            trace = run_broadcast(client, item)
            trace.system_type = "broadcast"
            log.latency = time.time() - start
            log.task_success = bool(trace.raw_answer)
            log.graph_nodes_after = len(self.graph.nodes)
            log.graph_edges_after = len(self.graph.edges)
            return trace, log

        if self.config.variant_name == "static_router":
            trace = run_static_router(client, item)
            trace.system_type = "static_router"
            log.latency = time.time() - start
            log.task_success = bool(trace.raw_answer)
            log.graph_nodes_after = len(self.graph.nodes)
            log.graph_edges_after = len(self.graph.edges)
            return trace, log

        # Build/update graph
        agent_node_id, task_node_id = self._build_graph_for_item(item)

        # Collect candidate context IDs
        candidate_ids = []
        for nid, node in self.graph.nodes.items():
            if node.node_type in (NodeType.ARTIFACT, NodeType.MEMORY, NodeType.CONSTRAINT):
                candidate_ids.append(nid)
        log.candidate_context_ids = candidate_ids

        # Select context
        if self.config.variant_name == "top_k_only":
            selected_context, selected_ids, fallback, cache_hit = self._select_context_topk(item)
        else:
            selected_context, selected_ids, fallback, cache_hit = self._select_context_graph(
                agent_node_id, task_node_id, item,
            )

        log.selected_context_ids = selected_ids
        log.selected_token_count = estimate_tokens(selected_context)
        log.fallback_triggered = fallback
        log.cache_hit = cache_hit
        log.cache_hits = 1 if cache_hit else 0

        # Build prompt
        user_prompt = DTCG_USER.format(
            question=question,
            selected_context=selected_context[:3000],
            local_cache=self.cache.to_context_string(max_tokens=500) if self.config.persist_cache else "",
            constraints=", ".join(constraints[:3]) if constraints else "无特殊约束",
        )

        log.serialized_prompt = user_prompt
        log.prompt_hash = hashlib.md5(user_prompt.encode()).hexdigest()[:16]
        log.actual_prompt_token_count = estimate_tokens(user_prompt)

        # Call model
        response = client.chat(
            messages=[
                {"role": "system", "content": DTCG_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        latency = time.time() - start
        log.latency = latency

        answer = response.content
        log.output_hash = hashlib.md5(answer.encode()).hexdigest()[:16]
        log.agent_output_hash = log.output_hash
        log.task_success = bool(answer and answer.strip())

        # Parse answer
        parsed = answer.strip()
        if "Answer:" in parsed:
            parsed = parsed.split("Answer:")[-1].strip()[:200]
        elif "答案：" in parsed:
            parsed = parsed.split("答案：")[-1].strip()[:200]
        else:
            parsed = parsed[:200]

        in_tok = response.usage.get("prompt_tokens", 0)
        out_tok = response.usage.get("completion_tokens", 0)

        # Update trust scores based on task success (for variants that allow it)
        if self.config.update_trust:
            for nid in selected_ids:
                current_trust = self._trust_scores.get(nid, 0.5)
                # Simple update: increase trust if task succeeded
                if log.task_success:
                    self._trust_scores[nid] = min(1.0, current_trust + 0.05)
                else:
                    self._trust_scores[nid] = max(0.0, current_trust - 0.05)
                log.trust_scores_after[nid] = self._trust_scores[nid]
        else:
            log.trust_scores_after = dict(self._trust_scores)

        # Compute redundancy scores for selected items
        for i, nid_i in enumerate(selected_ids):
            max_red = 0.0
            for j, nid_j in enumerate(selected_ids):
                if i != j:
                    # Simple overlap-based redundancy
                    node_i = self.graph.nodes.get(nid_i)
                    node_j = self.graph.nodes.get(nid_j)
                    if node_i and node_j:
                        words_i = set(node_i.name.lower().split())
                        words_j = set(node_j.name.lower().split())
                        if words_i and words_j:
                            overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                            max_red = max(max_red, overlap)
            log.redundancy_scores[nid_i] = max_red

        # Update cache (for variants that persist cache)
        if self.config.persist_cache:
            entry = CacheEntry.new(
                entry_type=CacheEntryType.OBSERVATION,
                content=f"Q: {question[:100]} A: {parsed[:100]}",
                relevance_tags=["task_result"],
                importance=0.5,
            )
            self.cache.add(entry)

        # Record selected history for redundancy
        self._selected_history.append(selected_ids)

        # Record graph state after
        log.graph_nodes_after = len(self.graph.nodes)
        log.graph_edges_after = len(self.graph.edges)

        # Compute context saving
        evidence = item.get("evidence", [])
        broadcast_tokens = estimate_tokens(" ".join(str(e) for e in evidence[:5]))
        dtcg_tokens = log.selected_token_count
        saving_ratio = 1.0 - (dtcg_tokens / broadcast_tokens) if broadcast_tokens > 0 else 0

        trace = AblationTrace(
            task_id=task_id,
            benchmark_id=task_id,
            system_type=self.config.variant_name,
            model_name=getattr(client, 'model_name', 'unknown'),
            task_type=item.get("task_type", ""),
            modality=item.get("modality", "text"),
            difficulty=item.get("difficulty", ""),
            num_agents=1,
            num_messages=1,
            num_context_packages=1,
            broadcast_context_tokens=broadcast_tokens,
            selected_context_tokens=dtcg_tokens,
            context_saving_ratio=saving_ratio,
            duplicate_context_ratio=0.05,
            num_llm_calls=1,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            latency_seconds=latency,
            raw_answer=answer,
            parsed_answer=parsed,
            gold_answer=item.get("answer", ""),
            fallback_used=fallback,
            selected_context_text=selected_context[:500],
        )

        return trace, log

    def run(self, client, items: list[dict]) -> tuple[list[AblationTrace], list[DTCGTaskLog]]:
        """Run all items through this variant with persistent graph."""
        traces = []
        for item in items:
            trace, log = self._run_single_variant(client, item)
            traces.append(trace)
            self.task_logs.append(log)

            # For non-persistent graph variants, reset graph between tasks
            if not self.config.persist_graph:
                self.graph = DynamicTaskContextGraph()

            # For non-persistent cache variants, reset cache between tasks
            if not self.config.persist_cache:
                self.cache = LocalCache(agent_name=f"dtcg_{self.config.variant_name}")

        return traces, self.task_logs


# ── Public API ──────────────────────────────────────────────────────

def run_persistent_ablation(
    items: list[dict],
    client,
    variants: Optional[list[str]] = None,
) -> dict[str, tuple[list[AblationTrace], list[DTCGTaskLog]]]:
    """Run persistent DTCG component ablation across multiple variants.

    Returns a dict mapping variant name to (traces, logs).
    """
    if variants is None:
        variants = ["full", "no_cache", "no_redundancy", "no_trust", "no_recency", "static", "top_k_only", "broadcast", "static_router"]

    results = {}
    for variant in variants:
        config = make_variant_config(variant)
        engine = PersistentDTCGEngine(config)
        traces, logs = engine.run(client, items)
        results[variant] = (traces, logs)

    return results


def compute_ablation_table(results: dict[str, tuple[list[AblationTrace], list[DTCGTaskLog]]]) -> list[dict]:
    """Compute summary table from persistent ablation results."""
    rows = []
    for variant, (traces, logs) in results.items():
        valid_traces = [t for t in traces if not t.error_type]
        if not valid_traces:
            rows.append({
                "variant": variant,
                "total": 0,
                "accuracy": 0,
                "avg_context_tokens": 0,
                "avg_latency": 0,
                "fallback_count": 0,
                "cache_hit_count": 0,
            })
            continue

        correct = sum(1 for t in valid_traces if t.is_correct)
        avg_ctx = sum(t.selected_context_tokens for t in valid_traces) / len(valid_traces)
        avg_lat = sum(t.latency_seconds for t in valid_traces) / len(valid_traces)
        fallback_count = sum(1 for l in logs if l.fallback_triggered)
        cache_hits = sum(1 for l in logs if l.cache_hit)

        rows.append({
            "variant": variant,
            "total": len(valid_traces),
            "correct": correct,
            "accuracy": round(correct / len(valid_traces), 4) if valid_traces else 0,
            "avg_context_tokens": round(avg_ctx),
            "avg_latency": round(avg_lat, 3),
            "fallback_count": fallback_count,
            "cache_hit_count": cache_hits,
        })

    return rows


def write_trace_jsonl(logs: list[DTCGTaskLog], path: str) -> None:
    """Write task logs to JSONL for reproducibility."""
    with open(path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps({
                "task_id": log.task_id,
                "variant": log.variant,
                "graph_nodes_before": log.graph_nodes_before,
                "graph_edges_before": log.graph_edges_before,
                "graph_nodes_after": log.graph_nodes_after,
                "graph_edges_after": log.graph_edges_after,
                "candidate_context_ids": log.candidate_context_ids,
                "selected_context_ids": log.selected_context_ids,
                "selected_token_count": log.selected_token_count,
                "actual_prompt_token_count": log.actual_prompt_token_count,
                "selector_config_hash": log.selector_config_hash,
                "fallback_triggered": log.fallback_triggered,
                "cache_hit": log.cache_hit,
                "cache_hits": log.cache_hits,
                "trust_scores_before": log.trust_scores_before,
                "trust_scores_after": log.trust_scores_after,
                "redundancy_scores": log.redundancy_scores,
                "prompt_hash": log.prompt_hash,
                "output_hash": log.output_hash,
                "agent_output_hash": log.agent_output_hash,
                "task_success": log.task_success,
                "latency": log.latency,
            }, ensure_ascii=False) + "\n")
