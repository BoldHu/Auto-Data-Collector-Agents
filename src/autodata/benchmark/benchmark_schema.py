"""Benchmark item schema for Phase 5.

Defines the unified BenchmarkItem dataclass and related enums.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SourceType(str, Enum):
    TEXT = "text"
    EXAM = "exam"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class TaskCategory(str, Enum):
    # Text-only tasks
    DOMAIN_KNOWLEDGE_QA = "domain_knowledge_qa"
    EXAM_SINGLE_CHOICE = "exam_single_choice"
    EXAM_MULTIPLE_CHOICE = "exam_multiple_choice"
    EXAM_TRUE_FALSE = "exam_true_false"
    EXAM_FILL_BLANK = "exam_fill_blank"
    EXAM_SHORT_ANSWER = "exam_short_answer"
    EXAM_CALCULATION = "exam_calculation"
    INFORMATION_EXTRACTION = "information_extraction"
    PROCESS_REASONING = "process_reasoning"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    CAUSAL_REASONING = "causal_reasoning"
    ERROR_DIAGNOSIS = "error_diagnosis"
    SOURCE_GROUNDED_REASONING = "source_grounded_reasoning"
    # Multimodal tasks
    VISUAL_QA = "visual_qa"
    IMAGE_CLASSIFICATION = "image_classification"
    MATERIAL_FORM_RECOGNITION = "material_form_recognition"
    PROCESS_STAGE_IDENTIFICATION = "process_stage_identification"
    EQUIPMENT_FUNCTION_REASONING = "equipment_function_reasoning"
    DEFECT_DIAGNOSIS = "defect_diagnosis"
    CHART_READING = "chart_reading"
    DIAGRAM_REASONING = "diagram_reasoning"
    OCR_REASONING = "ocr_reasoning"
    CROSS_MODAL_REASONING = "cross_modal_reasoning"
    MULTI_STEP_VISUAL_REASONING = "multi_step_visual_reasoning"


class BenchmarkModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class BenchmarkSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    SFT_POOL = "sft_pool"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW = "review"


@dataclass
class QualityScores:
    """Quality scores for a benchmark item."""
    clarity: Optional[float] = None
    completeness: Optional[float] = None
    answerability: Optional[float] = None
    domain_relevance: Optional[float] = None
    reasoning_depth: Optional[float] = None
    hallucination_risk: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "clarity": self.clarity,
            "completeness": self.completeness,
            "answerability": self.answerability,
            "domain_relevance": self.domain_relevance,
            "reasoning_depth": self.reasoning_depth,
            "hallucination_risk": self.hallucination_risk,
        }


@dataclass
class BenchmarkItem:
    """A unified benchmark item for the carbon-fiber benchmark."""
    benchmark_id: str
    source_type: str
    task_type: str
    modality: str
    question: str
    options: list[dict] = field(default_factory=list)
    answer: str = ""
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)
    required_knowledge: list[str] = field(default_factory=list)
    reasoning_type: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    quality_scores: dict = field(default_factory=dict)
    split: str = "test"
    leakage_group_id: str = ""
    validation_status: str = "passed"
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "source_type": self.source_type,
            "task_type": self.task_type,
            "modality": self.modality,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "source_refs": self.source_refs,
            "image_refs": self.image_refs,
            "required_knowledge": self.required_knowledge,
            "reasoning_type": self.reasoning_type,
            "difficulty": self.difficulty,
            "quality_scores": self.quality_scores,
            "split": self.split,
            "leakage_group_id": self.leakage_group_id,
            "validation_status": self.validation_status,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkItem:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def generate_id(source_type: str, source_id: str) -> str:
        import uuid
        content = f"{source_type}:{source_id}:{uuid.uuid4().hex[:8]}"
        return f"bench_{hashlib.md5(content.encode()).hexdigest()[:16]}"
