"""Text corpus structured schemas with provenance preservation.

Every output item preserves source file, folder, page number,
content hashes, language, model name, timestamp, and run metadata.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Language enum ────────────────────────────────────────────────

class Language(str, Enum):
    ZH = "zh"
    EN = "en"
    UNKNOWN = "unknown"


# ── Knowledge type enum ─────────────────────────────────────────

class KnowledgeType(str, Enum):
    DEFINITION = "definition"
    PROPERTY = "property"
    PROCESS = "process"
    MECHANISM = "mechanism"
    APPLICATION = "application"
    MEASUREMENT = "measurement"
    DEFECT = "defect"
    COMPARISON = "comparison"
    EQUATION = "equation"
    TABLE = "table"
    OTHER = "other"


# ── SFT task type enum ──────────────────────────────────────────

class SFTTaskType(str, Enum):
    QA = "qa"
    EXPLANATION = "explanation"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    COMPARISON = "comparison"
    PROCESS_REASONING = "process_reasoning"


# ── Difficulty enum ─────────────────────────────────────────────

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ── Quality verdict enum ────────────────────────────────────────

class QualityVerdict(str, Enum):
    PASSED = "passed"
    NEEDS_REVISION = "needs_revision"
    FAILED = "failed"


# ── Helper: content hash ────────────────────────────────────────

def content_hash(text: str) -> str:
    """SHA-256 hash of text content for provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Raw document ────────────────────────────────────────────────

@dataclass
class RawDocument:
    """A raw OCR/book document loaded from text_raw_data."""
    file_name: str
    source_folder: str  # "books" or "en_books"
    page_count: int
    file_size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[RawPage] = field(default_factory=list)
    language: Language = Language.UNKNOWN


@dataclass
class RawPage:
    """A single page from a raw document."""
    page_number: int
    content: str
    clean_content: Optional[str] = None
    has_formula_guess: bool = False
    source: str = "ocr"
    content_hash: str = ""
    source_file: str = ""
    source_folder: str = ""

    def __post_init__(self):
        if self.content_hash == "" and self.content:
            self.content_hash = content_hash(self.content)


# ── Cleaned chunk ────────────────────────────────────────────────

@dataclass
class CleanedChunk:
    """A cleaned text chunk from the pipeline (v2.0 schema).

    Key v2.0 additions over v1.0:
    - keep_for_corpus: whether this chunk should enter pretraining corpus
    - removed_noise_types: list of noise types removed during cleaning
    - ocr_repairs: specific OCR corrections made
    - technical_content_types: types of technical content present
    - uncertainty_notes: where the model is uncertain
    - drop_reason: reason for not including in corpus (if keep_for_corpus=False)
    - enriched_notes: model-generated annotations (separate from source-faithful cleaned_text)
    """
    source_file: str
    source_folder: str
    page_numbers: list[int]
    language: Language
    original_text: str
    cleaned_text: str
    original_content_hash: str = ""
    cleaned_content_hash: str = ""
    chunk_id: str = ""
    cleaning_model: str = "mimo-v2-omni"
    cleaning_prompt_version: str = "v2.0"
    cleaning_timestamp: float = field(default_factory=time.time)
    run_id: str = ""
    quality_score: Optional[QualityScore] = None
    chunk_type: str = "body"  # body, header_footer, formula, table, mixed, table_uncertain, empty
    metadata: dict[str, Any] = field(default_factory=dict)
    # v2.0 fields — enrichment separation and domain filtering
    keep_for_corpus: bool = True
    removed_noise_types: list[str] = field(default_factory=list)
    ocr_repairs: list[dict[str, str]] = field(default_factory=list)
    technical_content_types: list[str] = field(default_factory=list)
    uncertainty_notes: str = ""
    drop_reason: str = ""
    enriched_notes: str = ""

    def __post_init__(self):
        if self.original_content_hash == "" and self.original_text:
            self.original_content_hash = content_hash(self.original_text)
        if self.cleaned_content_hash == "" and self.cleaned_text:
            self.cleaned_content_hash = content_hash(self.cleaned_text)
        if self.chunk_id == "":
            self.chunk_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "source_folder": self.source_folder,
            "page_numbers": self.page_numbers,
            "language": self.language.value,
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "original_content_hash": self.original_content_hash,
            "cleaned_content_hash": self.cleaned_content_hash,
            "cleaning_model": self.cleaning_model,
            "cleaning_prompt_version": self.cleaning_prompt_version,
            "cleaning_timestamp": self.cleaning_timestamp,
            "run_id": self.run_id,
            "chunk_type": self.chunk_type,
            "quality_score": self.quality_score.to_dict() if self.quality_score else None,
            "metadata": self.metadata,
            "keep_for_corpus": self.keep_for_corpus,
            "removed_noise_types": self.removed_noise_types,
            "ocr_repairs": self.ocr_repairs,
            "technical_content_types": self.technical_content_types,
            "uncertainty_notes": self.uncertainty_notes,
            "drop_reason": self.drop_reason,
            "enriched_notes": self.enriched_notes,
        }


# ── Knowledge unit ──────────────────────────────────────────────

@dataclass
class KnowledgeUnit:
    """An atomic carbon-fiber knowledge unit extracted from cleaned text."""
    unit_id: str
    source_chunk_id: str
    language: Language
    topic: str
    subtopic: str = ""
    knowledge_type: KnowledgeType = KnowledgeType.OTHER
    claim: str = ""
    evidence_text: str = ""
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    numeric_values: list[dict[str, Any]] = field(default_factory=list)
    quality_score: Optional[QualityScore] = None
    source_refs: list[str] = field(default_factory=list)
    extraction_model: str = "mimo-v2.5-pro"
    extraction_timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def __post_init__(self):
        if self.unit_id == "":
            self.unit_id = f"ku_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_chunk_id": self.source_chunk_id,
            "language": self.language.value,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "knowledge_type": self.knowledge_type.value,
            "claim": self.claim,
            "evidence_text": self.evidence_text,
            "entities": self.entities,
            "relations": self.relations,
            "conditions": self.conditions,
            "numeric_values": self.numeric_values,
            "quality_score": self.quality_score.to_dict() if self.quality_score else None,
            "source_refs": self.source_refs,
            "extraction_model": self.extraction_model,
            "extraction_timestamp": self.extraction_timestamp,
            "run_id": self.run_id,
        }


# ── SFT candidate ───────────────────────────────────────────────

@dataclass
class SFTCandidate:
    """A candidate supervised fine-tuning sample."""
    sample_id: str
    source_chunk_id: str
    task_type: SFTTaskType = SFTTaskType.QA
    instruction: str = ""
    input: str = ""
    output: str = ""
    evidence_text: str = ""
    difficulty: Difficulty = Difficulty.MEDIUM
    quality_score: Optional[QualityScore] = None
    source_refs: list[str] = field(default_factory=list)
    generation_model: str = "mimo-v2.5-pro"
    generation_timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def __post_init__(self):
        if self.sample_id == "":
            self.sample_id = f"sft_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_chunk_id": self.source_chunk_id,
            "task_type": self.task_type.value,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "evidence_text": self.evidence_text,
            "difficulty": self.difficulty.value,
            "quality_score": self.quality_score.to_dict() if self.quality_score else None,
            "source_refs": self.source_refs,
            "generation_model": self.generation_model,
            "generation_timestamp": self.generation_timestamp,
            "run_id": self.run_id,
        }


# ── Quality score ───────────────────────────────────────────────

@dataclass
class QualityScore:
    """Quality scores for a cleaned chunk, knowledge unit, or SFT candidate."""
    clarity: float = 0.0       # 0-1
    completeness: float = 0.0  # 0-1
    consistency: float = 0.0   # 0-1
    feasibility: float = 0.0   # 0-1
    complexity: float = 0.0    # 0-1
    domain_relevance: float = 0.0  # 0-1
    verdict: QualityVerdict = QualityVerdict.PASSED
    issues: list[str] = field(default_factory=list)
    verification_model: str = "mimo-v2.5-pro"
    verification_timestamp: float = field(default_factory=time.time)

    @property
    def average(self) -> float:
        scores = [self.clarity, self.completeness, self.consistency,
                  self.feasibility, self.domain_relevance]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clarity": self.clarity,
            "completeness": self.completeness,
            "consistency": self.consistency,
            "feasibility": self.feasibility,
            "complexity": self.complexity,
            "domain_relevance": self.domain_relevance,
            "average": self.average,
            "verdict": self.verdict.value,
            "issues": self.issues,
            "verification_model": self.verification_model,
            "verification_timestamp": self.verification_timestamp,
        }


# ── Cleaning run metadata ──────────────────────────────────────

@dataclass
class CleaningRunMetadata:
    """Metadata for a complete cleaning run."""
    run_id: str
    mode: str = "pilot"  # pilot or full
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    model_name: str = "mimo-v2.5-pro"
    model: str = "mimo-v2.5-pro"  # backward compat
    prompt_version: str = "v1.0"
    language_filter: str = "all"
    max_files: Optional[int] = None
    max_pages_per_file: Optional[int] = None
    total_raw_files_seen: int = 0
    total_files_processed: int = 0
    total_pages_processed: int = 0
    total_raw_chunks: int = 0
    total_cleaned_chunks: int = 0
    total_quality_scores: int = 0
    total_chunks_created: int = 0
    total_chunks_passed: int = 0
    total_chunks_needs_revision: int = 0
    total_chunks_failed: int = 0
    total_knowledge_units: int = 0
    total_sft_candidates: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    total_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.run_id == "":
            self.run_id = f"run_{uuid.uuid4().hex[:8]}_{int(self.start_time)}"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id, "mode": self.mode,
            "start_time": self.start_time, "end_time": self.end_time,
            "model_name": self.model_name, "model": self.model,
            "prompt_version": self.prompt_version,
            "language_filter": self.language_filter,
            "max_files": self.max_files, "max_pages_per_file": self.max_pages_per_file,
            "total_raw_files_seen": self.total_raw_files_seen,
            "total_files_processed": self.total_files_processed,
            "total_pages_processed": self.total_pages_processed,
            "total_raw_chunks": self.total_raw_chunks,
            "total_cleaned_chunks": self.total_cleaned_chunks,
            "total_quality_scores": self.total_quality_scores,
            "total_chunks_created": self.total_chunks_created,
            "total_chunks_passed": self.total_chunks_passed,
            "total_chunks_needs_revision": self.total_chunks_needs_revision,
            "total_chunks_failed": self.total_chunks_failed,
            "total_knowledge_units": self.total_knowledge_units,
            "total_sft_candidates": self.total_sft_candidates,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "total_latency_ms": self.total_latency_ms,
            "errors": self.errors,
        }
        return d