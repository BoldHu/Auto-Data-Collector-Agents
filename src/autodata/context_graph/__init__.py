"""Dynamic Task-Context Graph (DTCG) module.

Core components:
  - graph_schema: Heterogeneous graph with typed nodes, edges, and dynamic weights
  - context_selector: Token-budgeted context selection via greedy MMR/knapsack
  - message_store: Structured JSONL message store for inter-agent communication
  - local_cache: Agent-specific compact cache with periodic summarization
"""

from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    Node,
    Edge,
    NodeType,
    EdgeType,
    TaskStatus,
)
from src.autodata.context_graph.context_selector import (
    ContextSelector,
    ContextSelectorConfig,
    ContextPackage,
)
from src.autodata.context_graph.message_store import (
    Message,
    MessageStore,
    MessageType,
    Visibility,
)
from src.autodata.context_graph.local_cache import (
    CacheEntry,
    CacheEntryType,
    LocalCache,
)

__all__ = [
    "DynamicTaskContextGraph",
    "Node",
    "Edge",
    "NodeType",
    "EdgeType",
    "TaskStatus",
    "ContextSelector",
    "ContextSelectorConfig",
    "ContextPackage",
    "Message",
    "MessageStore",
    "MessageType",
    "Visibility",
    "CacheEntry",
    "CacheEntryType",
    "LocalCache",
]
