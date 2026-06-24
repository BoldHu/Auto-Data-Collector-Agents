"""Evaluation schema for Phase 6.

Defines dataclasses for evaluation results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvaluationResult:
    """Result of evaluating one model on one benchmark item."""
    run_id: str
    model_name: str
    benchmark_id: str
    subset: str
    task_type: str
    modality: str
    prompt: str
    raw_response: str
    parsed_answer: str
    gold_answer: str
    is_correct: Optional[bool] = None
    exact_match: Optional[bool] = None
    f1: Optional[float] = None
    numeric_score: Optional[float] = None
    llm_judge_score: Optional[float] = None
    rubric_score: Optional[float] = None
    format_valid: bool = True
    hallucination_flag: bool = False
    latency_seconds: Optional[float] = None
    token_usage: dict = field(default_factory=dict)
    cost_estimate: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: str(time.time()))

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "benchmark_id": self.benchmark_id,
            "subset": self.subset,
            "task_type": self.task_type,
            "modality": self.modality,
            "prompt": self.prompt[:500],
            "raw_response": self.raw_response[:500],
            "parsed_answer": self.parsed_answer,
            "gold_answer": self.gold_answer,
            "is_correct": self.is_correct,
            "exact_match": self.exact_match,
            "f1": self.f1,
            "numeric_score": self.numeric_score,
            "llm_judge_score": self.llm_judge_score,
            "rubric_score": self.rubric_score,
            "format_valid": self.format_valid,
            "hallucination_flag": self.hallucination_flag,
            "latency_seconds": self.latency_seconds,
            "token_usage": self.token_usage,
            "cost_estimate": self.cost_estimate,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class JudgeResult:
    """Result of LLM judge evaluation."""
    correctness: float = 0.0
    evidence_support: float = 0.0
    reasoning_quality: float = 0.0
    hallucination: float = 0.0
    format_validity: float = 0.0
    final_score: float = 0.0
    verdict: str = "incorrect"
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "evidence_support": self.evidence_support,
            "reasoning_quality": self.reasoning_quality,
            "hallucination": self.hallucination,
            "format_validity": self.format_validity,
            "final_score": self.final_score,
            "verdict": self.verdict,
            "rationale": self.rationale,
        }
