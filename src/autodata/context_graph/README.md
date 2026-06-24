# Dynamic Task-Context Graph (DTCG)

A graph-based context-management framework for long-horizon multi-agent data construction.

## Problem

Traditional multi-agent systems use broadcast-style communication: every agent receives all historical messages. This causes:

- **Context overload**: each agent's prompt grows unboundedly
- **High token cost**: redundant information is re-processed by every agent
- **Duplicated reasoning**: agents re-derive conclusions already reached by others
- **Poor scalability**: adding agents multiplies the communication cost

## Solution

DTCG replaces broadcast communication with a dynamic heterogeneous graph that routes only relevant context to each agent under a token budget.

---

## 1. Graph Definition

At time step `t`, define a dynamic heterogeneous graph:

```
G_t = (V_t, E_t)
```

### 1.1 Node Types

| Node Type | Description | Examples |
|-----------|-------------|---------|
| **Agent** | Worker agents and central planner | CentralPlanningAgent, DataCleaningAgent |
| **Task** | Current, subtask, pending, completed | "clean_text_v2", "extract_exam_questions" |
| **Artifact** | Files, documents, generated samples | cleaned_corpus.json, benchmark_items.jsonl |
| **Memory** | Summarized context, prior decisions | "text_cleaning_strategy_v1", "error_report_ocr" |
| **Tool** | Crawler, OCR, parser, evaluator | bing_crawler, pdf_parser, deduplicator |
| **Constraint** | Domain constraints, quality rules | "min_clarity_score=0.7", "preserve_provenance" |

### 1.2 Edge Types

| Edge Type | Semantics |
|-----------|-----------|
| `task_dependency` | Task → Task dependency ordering |
| `agent_assignment` | Agent ↔ Task assignment |
| `artifact_derived_from` | Artifact provenance chain |
| `context_relevance` | Semantic/contextual links between nodes |
| `quality_feedback` | Quality verification feedback |
| `tool_usage` | Agent → Tool usage links |
| `duplication_conflict` | Redundancy or conflict detection |
| `benchmark_source` | Benchmark item → source tracing |

### 1.3 Dynamic Edge Weight

Each edge carries a dynamic weight computed as:

```
w_ij^(t) = sigmoid(
    α₁ · Rel(i,j)    // semantic relevance between agent/task query and node j
  + α₂ · Dep(i,j)    // task dependency strength
  + α₃ · Rec(i,j,t)  // recency with exponential time decay
  + α₄ · Trust(i,j)  // source quality / verification score
  - α₅ · Red(i,j)    // redundancy with already-selected context
  - α₆ · Cost(j)     // estimated token or computation cost
)
```

Where:
- **Rel(i,j)**: Semantic relevance computed via embedding cosine similarity between the current agent/task query and node j's content
- **Dep(i,j)**: Task dependency measured by transitive closure on task_dependency edges; direct dependencies score 1.0, k-hop dependencies score 1/k
- **Rec(i,j,t)**: Recency with exponential decay: `Rec(i,j,t) = exp(-ln(2) · age / half_life)`
- **Trust(i,j)**: Source quality score from the Quality Verification Agent, or historical reliability of the producing agent
- **Red(i,j)**: Redundancy measured by maximum pairwise similarity between candidate j and already-selected context items
- **Cost(j)**: Normalized token estimate of including node j in context, divided by the agent's total budget

---

## 2. Context Selection Objective

For each agent `a` at step `t`, select a context subset `S_a^t` from graph neighborhood `N(a,t)` under a token budget `B_a`:

### Objective

```
Maximize:
    Σ relevance(v) + β Σ dependency(v) + γ Σ trust(v)
  - λ redundancy(S) - μ token_cost(S)

Subject to:
    Σ token_cost(v) ≤ B_a
    S_a^t ⊆ N(a,t)
```

### Greedy MMR/Knapsack Approximation

Since the problem is NP-hard (reduction from knapsack), we use a greedy approximation:

1. **Retrieve candidates**: Expand the agent's 2-hop neighborhood in G_t
2. **Score candidates**: For each candidate node v, compute the marginal objective value
3. **Penalize redundancy**: For each candidate, compute `max_redundancy(v, S)` — the maximum pairwise redundancy with already-selected items
4. **Select greedily**: At each step, pick the candidate with the highest marginal gain:
   ```
   gain(v) = score(v) - λ · max_red(v, S) - μ · cost(v)
   ```
5. **Stop when budget exhausted**: Continue until adding the next best candidate would exceed B_a
6. **Compress into context package**: Serialize selected nodes into a role-specific ContextPackage

The greedy algorithm achieves a `(1-1/e)` approximation ratio for the submodular portion of the objective.

---

## 3. Local Cache

Each agent `a` maintains a compact, role-specific local cache:

```
C_a^t = {
    recent observations,
    tool results,
    verified facts,
    unresolved issues,
    agent-specific summaries
}
```

Properties:
- **Compact**: bounded by `max_entries` and `max_tokens`
- **Role-specific**: content tailored to the agent's function
- **Updated** after each action
- **Periodically summarized**: when entries exceed the threshold, low-importance entries are compressed into summary entries
- **Linked to graph nodes**: each cache entry can reference a graph node for cross-agent retrieval
- **Retrievable**: by type, tag, recency, importance, and semantic similarity

The cache is distinct from the global message store: it holds only the information relevant to this specific agent's role and recent actions, not the full conversation history.

---

## 4. Context Package Schema

Before invoking an agent, the system constructs a context package containing only graph-selected context:

```json
{
  "agent_name": "DataCleaningAgent",
  "task_id": "clean_text_v2",
  "current_goal": "Clean OCR text from carbon fiber books",
  "allowed_tools": ["pdf_parser", "text_cleaner", "deduplicator"],
  "relevant_plan": "Phase 1: Clean all book text files...",
  "selected_memory": [
    {"node_id": "mem_001", "name": "cleaning_strategy_v1", "scores": {"relevance": 0.9}}
  ],
  "selected_artifacts": [
    {"node_id": "art_042", "name": "book_001_raw.json", "scores": {"relevance": 0.85}}
  ],
  "constraints": [
    {"node_id": "con_003", "name": "preserve_provenance", "properties": {"category": "domain"}}
  ],
  "quality_requirements": [
    {"clarity": 0.7, "completeness": 0.8, "consistency": 0.75}
  ],
  "output_schema": {"cleaned_text": "str", "quality_scores": "dict"},
  "forbidden_actions": ["delete_raw_data", "modify_source_files"],
  "total_token_estimate": 4200
}
```

Agents **never** receive full conversation history. They receive only graph-selected context.

---

## 5. Message Schema

All inter-agent messages follow a structured format:

```json
{
  "message_id": "msg_abc123",
  "timestamp": 1716200000.0,
  "sender_agent": "DataCleaningAgent",
  "receiver_agent": "QualityVerificationAgent",
  "task_id": "clean_text_v2",
  "message_type": "observation",
  "content": "Cleaned 218 pages from book 碳纤维入门, OCR noise reduced by 67%",
  "artifact_refs": ["art_042", "art_043"],
  "source_refs": ["src_book_001"],
  "embedding_id": "emb_xyz",
  "quality_score": null,
  "relevance_tags": ["text_cleaning", "ocr", "chinese"],
  "token_estimate": 85,
  "visibility": "local"
}
```

Message types: `plan`, `observation`, `tool_result`, `critique`, `constraint`, `summary`, `decision`

Visibility levels: `global` (all agents), `local` (sender + receiver), `restricted` (specific agents only)

---

## 6. Why DTCG Beats Broadcast Communication

| Aspect | Broadcast Multi-Agent | DTCG Multi-Agent |
|--------|----------------------|------------------|
| **Context per agent** | Full history (grows unboundedly) | Graph-selected subset (bounded by B_a) |
| **Token cost** | O(A² · M) where A=agents, M=messages | O(A · B_a) where B_a is the budget |
| **Redundancy** | High: each agent re-processes all messages | Low: redundancy explicitly penalized |
| **Scalability** | Poor: adding agents quadratically increases cost | Good: each agent's cost bounded by B_a |
| **Relevance** | Irrelevant messages dilute agent focus | Only relevant context is included |
| **Quality control** | No mechanism to route quality feedback to the right agent | Quality_feedback edges route critiques to producers |
| **Task dependency** | Implicit and error-prone | Explicit task_dependency edges |
| **Provenance** | Lost in broadcast noise | artifact_derived_from edges trace lineage |
| **Long-horizon** | Degrades as history grows | Stable: graph pruning + cache summarization |

### Theoretical Argument

In a broadcast system with A agents running for T steps, each producing ~M messages per step, the total context seen by each agent grows as O(A·T·M). Under DTCG, each agent sees at most B_a tokens per step, giving O(T·B_a) total. The savings ratio is:

```
Savings = (A · T · M · avg_msg_tokens) / (T · B_a) = A · M · avg_msg_tokens / B_a
```

For typical values (A=6, M=3, avg_msg_tokens=200, B_a=8000), DTCG reduces token cost by approximately **450×** per agent per step compared to broadcast.

---

## 7. Implementation Modules

| Module | File | Description |
|--------|------|-------------|
| `graph_schema` | `graph_schema.py` | Node, Edge, DynamicTaskContextGraph classes |
| `context_selector` | `context_selector.py` | ContextSelector with greedy MMR/knapsack |
| `message_store` | `message_store.py` | JSONL-based structured message store |
| `local_cache` | `local_cache.py` | Agent-local cache with summarization |
