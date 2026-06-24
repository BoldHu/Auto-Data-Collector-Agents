"""Trace schema for Phase 6.6 system ablation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AblationTrace:
    """Trace for a single system-task evaluation."""
    run_id: str = "phase_6_6_system_ablation"
    task_id: str = ""
    benchmark_id: str = ""
    system_type: str = ""
    model_name: str = ""
    task_type: str = ""
    modality: str = ""
    difficulty: str = ""

    # Agent coordination
    num_agents: int = 0
    num_messages: int = 0
    num_context_packages: int = 0

    # Context efficiency
    broadcast_context_tokens: int = 0
    selected_context_tokens: int = 0
    context_saving_ratio: float = 0.0
    duplicate_context_ratio: float = 0.0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Runtime
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    latency_seconds: float = 0.0

    # Results
    raw_answer: str = ""
    parsed_answer: str = ""
    gold_answer: str = ""

    # Scores
    judge_score: Optional[float] = None
    is_correct: Optional[bool] = None
    evidence_support: Optional[float] = None
    constraint_satisfaction: Optional[float] = None
    hallucination_flag: bool = False
    format_valid: bool = True

    # Error tracking
    error_type: Optional[str] = None
    trace_refs: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    # DTCG-specific
    fallback_used: bool = False
    selected_context_text: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "benchmark_id": self.benchmark_id,
            "system_type": self.system_type,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "modality": self.modality,
            "difficulty": self.difficulty,
            "num_agents": self.num_agents,
            "num_messages": self.num_messages,
            "num_context_packages": self.num_context_packages,
            "broadcast_context_tokens": self.broadcast_context_tokens,
            "selected_context_tokens": self.selected_context_tokens,
            "context_saving_ratio": self.context_saving_ratio,
            "duplicate_context_ratio": self.duplicate_context_ratio,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_llm_calls": self.num_llm_calls,
            "num_tool_calls": self.num_tool_calls,
            "latency_seconds": self.latency_seconds,
            "raw_answer": self.raw_answer[:500],
            "parsed_answer": self.parsed_answer[:200],
            "gold_answer": self.gold_answer[:200],
            "judge_score": self.judge_score,
            "is_correct": self.is_correct,
            "evidence_support": self.evidence_support,
            "constraint_satisfaction": self.constraint_satisfaction,
            "hallucination_flag": self.hallucination_flag,
            "format_valid": self.format_valid,
            "error_type": self.error_type,
            "trace_refs": self.trace_refs,
            "timestamp": self.timestamp,
            "fallback_used": self.fallback_used,
            "selected_context_text": self.selected_context_text[:500],
        }
