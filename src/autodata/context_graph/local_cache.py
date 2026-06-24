"""Dynamic Task-Context Graph (DTCG) — Local Cache.

Each agent maintains a compact, role-specific local cache:

    C_a^t = {
        recent observations,
        tool results,
        verified facts,
        unresolved issues,
        agent-specific summaries
    }

Properties:
  - Compact: limited by a configurable size
  - Role-specific: content tailored to the agent's function
  - Updated after each action
  - Periodically summarized
  - Linked to graph nodes
  - Retrievable by semantic similarity and graph dependency
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Cache Entry Types ────────────────────────────────────────────────

class CacheEntryType(str, Enum):
    OBSERVATION = "observation"
    TOOL_RESULT = "tool_result"
    VERIFIED_FACT = "verified_fact"
    UNRESOLVED_ISSUE = "unresolved_issue"
    SUMMARY = "summary"
    DECISION = "decision"


# ── Cache Entry ──────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A single entry in an agent's local cache."""
    entry_id: str
    entry_type: CacheEntryType
    content: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    graph_node_id: Optional[str] = None
    embedding_id: Optional[str] = None
    relevance_tags: list[str] = field(default_factory=list)
    token_estimate: int = 0
    access_count: int = 0
    importance: float = 0.5

    @staticmethod
    def new(
        entry_type: CacheEntryType,
        content: str,
        **kwargs,
    ) -> "CacheEntry":
        """Create a new cache entry with auto-generated ID."""
        return CacheEntry(
            entry_id=uuid.uuid4().hex[:12],
            entry_type=entry_type,
            content=content,
            **kwargs,
        )

    def touch(self) -> None:
        """Mark this entry as accessed (updates recency)."""
        self.access_count += 1
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "graph_node_id": self.graph_node_id,
            "embedding_id": self.embedding_id,
            "relevance_tags": self.relevance_tags,
            "token_estimate": self.token_estimate,
            "access_count": self.access_count,
            "importance": self.importance,
        }


# ── Local Cache ──────────────────────────────────────────────────────

class LocalCache:
    """An agent's local cache C_a^t.

    Maintains a compact, role-specific cache that is:
    - Updated after each agent action
    - Periodically summarized to prevent unbounded growth
    - Linked to graph nodes for cross-agent retrieval
    - Retrievable by type, tags, and recency
    """

    def __init__(
        self,
        agent_name: str,
        max_entries: int = 100,
        max_tokens: int = 4000,
        summary_threshold: float = 0.8,
    ) -> None:
        """Initialize the local cache.

        Args:
            agent_name: Name of the owning agent
            max_entries: Maximum number of entries before summarization
            max_tokens: Approximate maximum total token count
            summary_threshold: Fraction of max_entries that triggers summarization
        """
        self.agent_name = agent_name
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold

        self.entries: dict[str, CacheEntry] = {}
        self._by_type: dict[CacheEntryType, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        """Current total estimated token count."""
        return self._total_tokens

    @property
    def entry_count(self) -> int:
        """Current number of entries."""
        return len(self.entries)

    def add(self, entry: CacheEntry) -> None:
        """Add an entry to the cache.

        If the cache exceeds the summary threshold, triggers summarization.
        """
        # Check capacity and summarize if needed
        if self.entry_count >= int(self.max_entries * self.summary_threshold):
            self._summarize()

        self.entries[entry.entry_id] = entry
        self._total_tokens += entry.token_estimate

        # Index by type
        self._by_type.setdefault(entry.entry_type, []).append(entry.entry_id)

        # Index by tags
        for tag in entry.relevance_tags:
            self._by_tag.setdefault(tag, []).append(entry.entry_id)

    def get(self, entry_id: str) -> Optional[CacheEntry]:
        """Retrieve an entry by ID and mark it as accessed."""
        entry = self.entries.get(entry_id)
        if entry:
            entry.touch()
        return entry

    def get_by_type(self, entry_type: CacheEntryType) -> list[CacheEntry]:
        """Retrieve all entries of a given type."""
        ids = self._by_type.get(entry_type, [])
        return [self.entries[eid] for eid in ids if eid in self.entries]

    def get_by_tag(self, tag: str) -> list[CacheEntry]:
        """Retrieve all entries matching a tag."""
        ids = self._by_tag.get(tag, [])
        return [self.entries[eid] for eid in ids if eid in self.entries]

    def get_recent(self, n: int = 10) -> list[CacheEntry]:
        """Retrieve the N most recently updated entries."""
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.updated_at,
            reverse=True,
        )
        return sorted_entries[:n]

    def get_important(self, n: int = 10) -> list[CacheEntry]:
        """Retrieve the N most important entries."""
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.importance,
            reverse=True,
        )
        return sorted_entries[:n]

    def search_by_similarity(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[CacheEntry]:
        """Retrieve entries by semantic similarity to a query.

        Currently uses simple keyword overlap. In the full implementation,
        this will use embedding cosine similarity via the embedding_id field.
        """
        query_terms = set(query.lower().split())
        scored = []
        for entry in self.entries.values():
            content_terms = set(entry.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]

    def remove(self, entry_id: str) -> None:
        """Remove an entry from the cache."""
        entry = self.entries.pop(entry_id, None)
        if entry:
            self._total_tokens -= entry.token_estimate
            # Clean up indices
            type_list = self._by_type.get(entry.entry_type, [])
            if entry_id in type_list:
                type_list.remove(entry_id)
            for tag in entry.relevance_tags:
                tag_list = self._by_tag.get(tag, [])
                if entry_id in tag_list:
                    tag_list.remove(entry_id)

    def _summarize(self) -> None:
        """Summarize old/low-importance entries to free cache space.

        Strategy:
        1. Group entries by type
        2. For each group, create a summary entry
        3. Remove the original entries
        4. Add the summary entry
        """
        # Sort entries by importance (ascending) and recency (ascending)
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: (e.importance, e.updated_at),
        )

        # Summarize the bottom 50% of entries
        n_to_summarize = len(sorted_entries) // 2
        if n_to_summarize == 0:
            return

        to_summarize = sorted_entries[:n_to_summarize]

        # Group by type for structured summaries
        type_groups: dict[CacheEntryType, list[CacheEntry]] = {}
        for entry in to_summarize:
            type_groups.setdefault(entry.entry_type, []).append(entry)

        # Create summary entries
        for entry_type, group in type_groups.items():
            summary_content = f"[Summary of {len(group)} {entry_type.value} entries]\n"
            for entry in group:
                summary_content += f"- {entry.content[:100]}\n"
                self.remove(entry.entry_id)

            summary = CacheEntry.new(
                entry_type=CacheEntryType.SUMMARY,
                content=summary_content,
                relevance_tags=["auto_summary"],
                token_estimate=len(summary_content.split()) * 2,
                importance=0.3,
            )
            self.add(summary)

    def to_context_string(self, max_tokens: int = 2000) -> str:
        """Serialize the cache into a compact context string for agent prompts.

        Prioritizes important and recent entries, respecting the token budget.
        """
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: (e.importance, e.updated_at),
            reverse=True,
        )

        lines = []
        token_count = 0
        for entry in sorted_entries:
            line = f"[{entry.entry_type.value}] {entry.content}"
            est = len(line.split()) * 2  # rough token estimate
            if token_count + est > max_tokens:
                break
            lines.append(line)
            token_count += est

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire cache to a dictionary."""
        return {
            "agent_name": self.agent_name,
            "max_entries": self.max_entries,
            "max_tokens": self.max_tokens,
            "total_tokens": self._total_tokens,
            "entry_count": self.entry_count,
            "entries": {
                eid: entry.to_dict() for eid, entry in self.entries.items()
            },
        }
