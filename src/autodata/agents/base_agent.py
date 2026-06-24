"""Base agent abstraction — shared interface for all AutoData agents.

Every agent:
- Has a name, model, and framework type
- Maintains a local cache (LocalCache)
- Is registered as a node in the DTCG (DynamicTaskContextGraph)
- Communicates via the MessageStore
- Executes actions and returns structured observations
- Tracks token usage and call counts
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.autodata.context_graph.local_cache import CacheEntry, CacheEntryType, LocalCache
from src.autodata.context_graph.message_store import Message, MessageType, MessageStore, Visibility
from src.autodata.utils.logging_utils import get_logger, safe_serialize


logger = get_logger("agents")


# ── Framework types ────────────────────────────────────────────────

class AgentFramework(str, Enum):
    PLAN_AND_EXECUTE = "plan_and_execute"
    REACT = "react"


# ── Agent observation ──────────────────────────────────────────────

@dataclass
class AgentObservation:
    """Structured output from an agent action."""
    agent_name: str
    action_type: str
    content: str
    success: bool = True
    artifact_refs: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    token_usage: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Base Agent ──────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base class for all AutoData agents.

    Subclasses must implement:
    - step(): execute one action/step
    - run(): execute the full task (multiple steps)
    """

    def __init__(
        self,
        name: str,
        framework: AgentFramework,
        model: str = "mimo-v2.5-pro",
        max_iterations: int = 15,
        context_budget: int = 6000,
        graph_node_id: Optional[str] = None,
        message_store: Optional[MessageStore] = None,
    ) -> None:
        self.name = name
        self.framework = framework
        self.model = model
        self.max_iterations = max_iterations
        self.context_budget = context_budget
        self.graph_node_id = graph_node_id or f"agent_{uuid.uuid4().hex[:8]}"

        # Local cache — role-specific, bounded
        self.cache = LocalCache(
            agent_name=name,
            max_entries=100,
            max_tokens=context_budget,
        )

        # Message store reference (shared across agents)
        self._message_store = message_store

        # Execution tracking
        self._step_count = 0
        self._total_tokens_used = 0
        self._observations: list[AgentObservation] = []

    @abstractmethod
    def step(self, context: dict[str, Any]) -> AgentObservation:
        """Execute one action step given the current context.

        Args:
            context: Context package from DTCG context selection.

        Returns:
            AgentObservation with action result.
        """
        ...

    @abstractmethod
    def run(self, task: str, context: Optional[dict] = None) -> list[AgentObservation]:
        """Execute the full task (multiple steps).

        Args:
            task: Task description.
            context: Initial context package (from DTCG).

        Returns:
            List of AgentObservation from all steps.
        """
        ...

    def send_message(
        self,
        receiver: str,
        content: str,
        task_id: Optional[str] = None,
        message_type: MessageType = MessageType.OBSERVATION,
        artifact_refs: Optional[list[str]] = None,
        source_refs: Optional[list[str]] = None,
        visibility: Visibility = Visibility.LOCAL,
    ) -> Message:
        """Send an inter-agent message via the message store."""
        msg = Message(
            message_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            sender_agent=self.name,
            receiver_agent=receiver,
            task_id=task_id or "",
            message_type=message_type,
            content=content,
            artifact_refs=artifact_refs or [],
            source_refs=source_refs or [],
            relevance_tags=[self.framework.value, self.name],
            token_estimate=len(content.split()) * 2,
            visibility=visibility,
        )

        if self._message_store:
            self._message_store.add(msg)

        logger.info(
            f"Agent {self.name} sent message to {receiver}: "
            + str(safe_serialize({"type": message_type.value, "content_len": len(content)}))
        )
        return msg

    def receive_messages(
        self,
        task_id: Optional[str] = None,
    ) -> list[Message]:
        """Retrieve messages addressed to this agent."""
        if self._message_store is None:
            return []
        return self._message_store.get_by_receiver(self.name)

    def add_to_cache(
        self,
        entry_type: CacheEntryType,
        content: str,
        relevance_tags: Optional[list[str]] = None,
        importance: float = 0.5,
    ) -> CacheEntry:
        """Add an observation to the local cache."""
        entry = CacheEntry.new(
            entry_type=entry_type,
            content=content,
            relevance_tags=relevance_tags or [],
            token_estimate=len(content.split()) * 2,
            importance=importance,
            graph_node_id=self.graph_node_id,
        )
        self.cache.add(entry)
        return entry

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    @property
    def observations(self) -> list[AgentObservation]:
        return self._observations

    def _record_observation(self, obs: AgentObservation) -> None:
        """Record an observation and update counters."""
        self._observations.append(obs)
        self._total_tokens_used += obs.token_usage
        self._step_count += 1