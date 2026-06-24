"""Dynamic Task-Context Graph (DTCG) — Message Store.

All inter-agent messages are stored in structured JSONL format.
Each message has a well-defined schema with sender, receiver,
task context, artifact references, quality scores, and visibility.

Message types:
  - plan: planning directives from central planner
  - observation: agent observations and tool results
  - tool_result: outputs from tool invocations
  - critique: quality feedback from verification agent
  - constraint: domain or quality constraints
  - summary: context summaries and compressions
  - decision: final decisions or task completions
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ── Message Types ─────────────────────────────────────────────────────

class MessageType(str, Enum):
    PLAN = "plan"
    OBSERVATION = "observation"
    TOOL_RESULT = "tool_result"
    CRITIQUE = "critique"
    CONSTRAINT = "constraint"
    SUMMARY = "summary"
    DECISION = "decision"


class Visibility(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    RESTRICTED = "restricted"


# ── Message Schema ───────────────────────────────────────────────────

@dataclass
class Message:
    """A structured inter-agent message.

    Schema:
    {
      "message_id": str,
      "timestamp": float,
      "sender_agent": str,
      "receiver_agent": str,
      "task_id": str,
      "message_type": MessageType,
      "content": str,
      "artifact_refs": list[str],
      "source_refs": list[str],
      "embedding_id": Optional[str],
      "quality_score": Optional[float],
      "relevance_tags": list[str],
      "token_estimate": int,
      "visibility": Visibility
    }
    """
    message_id: str
    timestamp: float
    sender_agent: str
    receiver_agent: str
    task_id: str
    message_type: MessageType
    content: str
    artifact_refs: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    embedding_id: Optional[str] = None
    quality_score: Optional[float] = None
    relevance_tags: list[str] = field(default_factory=list)
    token_estimate: int = 0
    visibility: Visibility = Visibility.LOCAL

    @staticmethod
    def new(
        sender: str,
        receiver: str,
        task_id: str,
        message_type: MessageType,
        content: str,
        **kwargs,
    ) -> "Message":
        """Create a new message with auto-generated ID and timestamp."""
        return Message(
            message_id=uuid.uuid4().hex[:16],
            timestamp=time.time(),
            sender_agent=sender,
            receiver_agent=receiver,
            task_id=task_id,
            message_type=message_type,
            content=content,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender_agent": self.sender_agent,
            "receiver_agent": self.receiver_agent,
            "task_id": self.task_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "artifact_refs": self.artifact_refs,
            "source_refs": self.source_refs,
            "embedding_id": self.embedding_id,
            "quality_score": self.quality_score,
            "relevance_tags": self.relevance_tags,
            "token_estimate": self.token_estimate,
            "visibility": self.visibility.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        """Deserialize from a dictionary."""
        return cls(
            message_id=d["message_id"],
            timestamp=d["timestamp"],
            sender_agent=d["sender_agent"],
            receiver_agent=d["receiver_agent"],
            task_id=d["task_id"],
            message_type=MessageType(d["message_type"]),
            content=d["content"],
            artifact_refs=d.get("artifact_refs", []),
            source_refs=d.get("source_refs", []),
            embedding_id=d.get("embedding_id"),
            quality_score=d.get("quality_score"),
            relevance_tags=d.get("relevance_tags", []),
            token_estimate=d.get("token_estimate", 0),
            visibility=Visibility(d.get("visibility", "local")),
        )


# ── Message Store ────────────────────────────────────────────────────

class MessageStore:
    """Persistent JSONL-based message store for inter-agent communication.

    All messages are appended to a JSONL file for durability and audit.
    In-memory index allows efficient retrieval by sender, receiver, task, or type.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self.messages: list[Message] = []
        self._by_sender: dict[str, list[int]] = {}
        self._by_receiver: dict[str, list[int]] = {}
        self._by_task: dict[str, list[int]] = {}
        self._by_type: dict[MessageType, list[int]] = {}
        self._store_path = Path(store_path) if store_path else None
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing messages from the JSONL store file."""
        if self._store_path and self._store_path.exists():
            with open(self._store_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            msg = Message.from_dict(d)
                            self._index_message(len(self.messages), msg)
                            self.messages.append(msg)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

    def _index_message(self, idx: int, msg: Message) -> None:
        """Add a message to the in-memory indices."""
        self._by_sender.setdefault(msg.sender_agent, []).append(idx)
        self._by_receiver.setdefault(msg.receiver_agent, []).append(idx)
        self._by_task.setdefault(msg.task_id, []).append(idx)
        self._by_type.setdefault(msg.message_type, []).append(idx)

    def add(self, message: Message) -> None:
        """Add a message to the store and persist it."""
        idx = len(self.messages)
        self.messages.append(message)
        self._index_message(idx, message)

        if self._store_path:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def get_by_sender(self, sender: str) -> list[Message]:
        """Retrieve all messages from a specific sender."""
        return [self.messages[i] for i in self._by_sender.get(sender, [])]

    def get_by_receiver(self, receiver: str) -> list[Message]:
        """Retrieve all messages to a specific receiver."""
        return [self.messages[i] for i in self._by_receiver.get(receiver, [])]

    def get_by_task(self, task_id: str) -> list[Message]:
        """Retrieve all messages for a specific task."""
        return [self.messages[i] for i in self._by_task.get(task_id, [])]

    def get_by_type(self, message_type: MessageType) -> list[Message]:
        """Retrieve all messages of a specific type."""
        return [self.messages[i] for i in self._by_type.get(message_type, [])]

    def get_recent(self, n: int = 10) -> list[Message]:
        """Retrieve the N most recent messages."""
        return self.messages[-n:]

    def get_for_agent(
        self,
        agent_name: str,
        task_id: Optional[str] = None,
        max_messages: int = 50,
    ) -> list[Message]:
        """Retrieve messages relevant to a specific agent.

        Returns messages where the agent is either sender or receiver,
        optionally filtered by task_id.
        """
        sent = set(self._by_sender.get(agent_name, []))
        received = set(self._by_receiver.get(agent_name, []))
        all_indices = sorted(sent | received, reverse=True)

        if task_id:
            task_indices = set(self._by_task.get(task_id, []))
            all_indices = sorted(all_indices & task_indices, reverse=True)

        messages = [self.messages[i] for i in all_indices[:max_messages]]
        return messages

    def count(self) -> int:
        """Return the total number of stored messages."""
        return len(self.messages)
