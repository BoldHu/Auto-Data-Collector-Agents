"""Exam question schemas for Phase 4.

Defines dataclasses for exam questions, quality scores, and related enums.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"
    CALCULATION = "calculation"
    CASE_ANALYSIS = "case_analysis"
    UNKNOWN = "unknown"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AnswerSource(str, Enum):
    EXPLICIT_ANSWER_KEY = "explicit_answer_key"
    INLINE_SOLUTION = "inline_solution"
    MODEL_INFERRED = "model_inferred"
    MISSING = "missing"


class QualityStatus(str, Enum):
    KEEP = "keep"
    REVIEW = "review"
    DROP = "drop"


@dataclass
class ExamOption:
    """A single option in a choice question."""
    key: str
    text: str

    def to_dict(self) -> dict:
        return {"key": self.key, "text": self.text}


@dataclass
class ExamQuestion:
    """An extracted exam question with full provenance."""
    question_id: str
    source_file: str
    source_page: list[int] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    question_number: str = ""
    question_type: str = "unknown"
    question_text: str = ""
    options: list[dict] = field(default_factory=list)
    answer: str = ""
    answer_source: str = "missing"
    explanation: str = ""
    knowledge_points: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    requires_calculation: bool = False
    contains_formula: bool = False
    contains_table: bool = False
    contains_image_reference: bool = False
    domain_relevance: float = 0.0
    extraction_confidence: float = 0.0
    uncertainty_notes: list[str] = field(default_factory=list)
    raw_evidence: str = ""
    run_id: str = "phase_4_exam_extraction"
    extraction_model: str = ""
    prompt_version: str = "v1"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "source_file": self.source_file,
            "source_page": self.source_page,
            "source_block_ids": self.source_block_ids,
            "question_number": self.question_number,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "options": self.options,
            "answer": self.answer,
            "answer_source": self.answer_source,
            "explanation": self.explanation,
            "knowledge_points": self.knowledge_points,
            "difficulty": self.difficulty,
            "requires_calculation": self.requires_calculation,
            "contains_formula": self.contains_formula,
            "contains_table": self.contains_table,
            "contains_image_reference": self.contains_image_reference,
            "domain_relevance": self.domain_relevance,
            "extraction_confidence": self.extraction_confidence,
            "uncertainty_notes": self.uncertainty_notes,
            "raw_evidence": self.raw_evidence,
            "run_id": self.run_id,
            "extraction_model": self.extraction_model,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExamQuestion:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def generate_id(source_file: str, question_number: str) -> str:
        """Generate deterministic question ID."""
        content = f"{source_file}:{question_number}"
        return f"exam_q_{hashlib.md5(content.encode()).hexdigest()[:12]}"


@dataclass
class ExamQualityScore:
    """Quality verification result for an exam question."""
    question_id: str
    quality_status: str = "review"
    clarity: float = 0.0
    completeness: float = 0.0
    answerability: float = 0.0
    option_integrity: float = 0.0
    answer_consistency: float = 0.0
    domain_relevance: float = 0.0
    difficulty_reasonableness: float = 0.0
    benchmark_usefulness: float = 0.0
    detected_issues: list[str] = field(default_factory=list)
    revision_suggestion: str = ""
    verifier_model: str = ""
    run_id: str = "phase_4_exam_extraction"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "quality_status": self.quality_status,
            "clarity": self.clarity,
            "completeness": self.completeness,
            "answerability": self.answerability,
            "option_integrity": self.option_integrity,
            "answer_consistency": self.answer_consistency,
            "domain_relevance": self.domain_relevance,
            "difficulty_reasonableness": self.difficulty_reasonableness,
            "benchmark_usefulness": self.benchmark_usefulness,
            "detected_issues": self.detected_issues,
            "revision_suggestion": self.revision_suggestion,
            "verifier_model": self.verifier_model,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExamQualityScore:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TextBlock:
    """A block of extracted text with provenance."""
    block_id: str
    source_file: str
    page_number: int = 0
    paragraph_id: int = 0
    table_id: str = ""
    text: str = ""
    extraction_method: str = ""
    content_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "paragraph_id": self.paragraph_id,
            "table_id": self.table_id,
            "text": self.text,
            "extraction_method": self.extraction_method,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def generate_id(source_file: str, page: int, para: int) -> str:
        content = f"{source_file}:p{page}:para{para}"
        return f"tb_{hashlib.md5(content.encode()).hexdigest()[:12]}"
