"""Image corpus structured schemas with provenance preservation.

Every output item preserves image path, keyword folder, metadata,
content hashes, model name, timestamp, and run metadata.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.autodata.pipelines.text_schema import (
    QualityScore,
    QualityVerdict,
    Difficulty,
    content_hash,
)


# ── Image category enum ────────────────────────────────────────────

class ImageCategory(str, Enum):
    FIBER = "fiber"
    FABRIC = "fabric"
    PREPREG = "prepreg"
    COMPOSITE_PART = "composite_part"
    MICROSTRUCTURE = "microstructure"
    EQUIPMENT = "equipment"
    PROCESS = "process"
    APPLICATION = "application"
    TESTING = "testing"
    DEFECT = "defect"
    CHART_DIAGRAM = "chart_diagram"
    PAPER_SCREENSHOT = "paper_screenshot"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


# ── Image modality enum ────────────────────────────────────────────

class ImageModality(str, Enum):
    PHOTO = "photo"
    MICROSCOPY = "microscopy"
    DIAGRAM = "diagram"
    CHART = "chart"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# ── Material form enum ─────────────────────────────────────────────

class MaterialForm(str, Enum):
    RAW_FIBER = "raw_fiber"
    TOW = "tow"
    FABRIC = "fabric"
    PREPREG = "prepreg"
    LAMINATE = "laminate"
    CFRP_PART = "cfrp_part"
    POWDER = "powder"
    UNKNOWN = "unknown"


# ── Process stage enum ─────────────────────────────────────────────

class ProcessStage(str, Enum):
    PRECURSOR = "precursor"
    SPINNING = "spinning"
    STABILIZATION = "stabilization"
    CARBONIZATION = "carbonization"
    GRAPHITIZATION = "graphitization"
    SURFACE_TREATMENT = "surface_treatment"
    SIZING = "sizing"
    WEAVING = "weaving"
    LAYUP = "layup"
    CURING = "curing"
    TESTING = "testing"
    APPLICATION = "application"
    UNKNOWN = "unknown"


# ── Application domain enum ───────────────────────────────────────

class ApplicationDomain(str, Enum):
    AEROSPACE = "aerospace"
    AUTOMOTIVE = "automotive"
    SPORTS = "sports"
    CIVIL_ENGINEERING = "civil_engineering"
    ENERGY = "energy"
    INDUSTRIAL = "industrial"
    BIOMEDICAL = "biomedical"
    UNKNOWN = "unknown"


# ── Visual task type enum ─────────────────────────────────────────

class VisualTaskType(str, Enum):
    CLASSIFICATION = "classification"
    CAPTIONING = "captioning"
    DEFECT_RECOGNITION = "defect_recognition"
    PROCESS_IDENTIFICATION = "process_identification"
    EQUIPMENT_IDENTIFICATION = "equipment_identification"
    MATERIAL_FORM_RECOGNITION = "material_form_recognition"
    CHART_READING = "chart_reading"
    DIAGRAM_REASONING = "diagram_reasoning"
    OCR_REASONING = "ocr_reasoning"
    CROSS_MODAL_REASONING = "cross_modal_reasoning"


# ── Benchmark task type enum ──────────────────────────────────────

class BenchmarkTaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    VISUAL_QA = "visual_qa"
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    PROCESS_REASONING = "process_reasoning"
    DEFECT_DIAGNOSIS = "defect_diagnosis"
    CHART_READING = "chart_reading"
    DIAGRAM_REASONING = "diagram_reasoning"
    OCR_REASONING = "ocr_reasoning"
    CROSS_MODAL_REASONING = "cross_modal_reasoning"


# ── Answerability type enum ───────────────────────────────────────

class AnswerabilityType(str, Enum):
    IMAGE_ONLY = "image_only"
    IMAGE_PLUS_METADATA = "image_plus_metadata"
    IMAGE_PLUS_DOMAIN_KNOWLEDGE = "image_plus_domain_knowledge"
    NOT_ANSWERABLE = "not_answerable"


# ── Hallucination risk enum ───────────────────────────────────────

class HallucinationRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Source status enum ────────────────────────────────────────────

class SourceStatus(str, Enum):
    METADATA_MATCHED = "metadata_matched"
    METADATA_MISSING = "metadata_missing"
    PATH_REPAIRED = "path_repaired"


# ── Dedup status enum ─────────────────────────────────────────────

class DedupStatus(str, Enum):
    UNIQUE = "unique"
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    UNKNOWN = "unknown"


# ── Quality status enum ───────────────────────────────────────────

class QualityStatus(str, Enum):
    KEEP = "keep"
    REVIEW = "review"
    DROP = "drop"


# ── Image manifest item ───────────────────────────────────────────

@dataclass
class ImageManifestItem:
    """A single image entry in the manifest."""
    image_id: str = ""
    file_path: str = ""
    relative_path: str = ""
    folder_keyword: str = ""
    folder_index: str = ""
    metadata_index: Optional[int] = None
    metadata_title: str = ""
    metadata_keyword: str = ""
    metadata_keyword_labels: list[str] = field(default_factory=list)
    image_url: str = ""
    file_size: int = 0
    width: int = 0
    height: int = 0
    format: str = "jpg"
    source_status: SourceStatus = SourceStatus.METADATA_MISSING
    hash: str = ""
    phash: str = ""
    created_at: float = field(default_factory=time.time)
    run_id: str = ""

    def __post_init__(self):
        if self.image_id == "":
            self.image_id = f"img_{uuid.uuid4().hex[:12]}"
        if self.hash == "" and self.file_path:
            try:
                with open(self.file_path, "rb") as f:
                    self.hash = hashlib.sha256(f.read()).hexdigest()[:16]
            except (OSError, IOError):
                self.hash = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "folder_keyword": self.folder_keyword,
            "folder_index": self.folder_index,
            "metadata_index": self.metadata_index,
            "metadata_title": self.metadata_title,
            "metadata_keyword": self.metadata_keyword,
            "metadata_keyword_labels": self.metadata_keyword_labels,
            "image_url": self.image_url,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "source_status": self.source_status.value,
            "hash": self.hash,
            "phash": self.phash,
            "created_at": self.created_at,
            "run_id": self.run_id,
        }


# ── Image caption record ──────────────────────────────────────────

@dataclass
class ImageCaptionRecord:
    """Caption and description for a single image."""
    image_id: str = ""
    short_caption: str = ""
    technical_caption: str = ""
    visible_objects: list[str] = field(default_factory=list)
    visible_materials: list[str] = field(default_factory=list)
    visible_processes: list[str] = field(default_factory=list)
    visible_equipment: list[str] = field(default_factory=list)
    visible_text: list[str] = field(default_factory=list)
    visual_evidence: list[str] = field(default_factory=list)
    inferred_domain_context: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    caption_model: str = "mimo-v2-omni"
    caption_prompt_version: str = "v1.0"
    caption_status: QualityVerdict = QualityVerdict.PASSED
    source_refs: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "short_caption": self.short_caption,
            "technical_caption": self.technical_caption,
            "visible_objects": self.visible_objects,
            "visible_materials": self.visible_materials,
            "visible_processes": self.visible_processes,
            "visible_equipment": self.visible_equipment,
            "visible_text": self.visible_text,
            "visual_evidence": self.visual_evidence,
            "inferred_domain_context": self.inferred_domain_context,
            "uncertainty_notes": self.uncertainty_notes,
            "caption_model": self.caption_model,
            "caption_prompt_version": self.caption_prompt_version,
            "caption_status": self.caption_status.value,
            "source_refs": self.source_refs,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


# ── Image label record ────────────────────────────────────────────

@dataclass
class ImageLabelRecord:
    """Domain labels and classification for a single image."""
    image_id: str = ""
    primary_category: ImageCategory = ImageCategory.UNKNOWN
    secondary_categories: list[str] = field(default_factory=list)
    material_form: MaterialForm = MaterialForm.UNKNOWN
    process_stage: ProcessStage = ProcessStage.UNKNOWN
    application_domain: ApplicationDomain = ApplicationDomain.UNKNOWN
    visual_task_type: list[VisualTaskType] = field(default_factory=list)
    domain_relevance: float = 0.0
    label_confidence: float = 0.0
    requires_human_review: bool = False
    label_model: str = "mimo-v2-omni"
    label_prompt_version: str = "v1.0"
    source_refs: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "primary_category": self.primary_category.value,
            "secondary_categories": self.secondary_categories,
            "material_form": self.material_form.value,
            "process_stage": self.process_stage.value,
            "application_domain": self.application_domain.value,
            "visual_task_type": [t.value for t in self.visual_task_type],
            "domain_relevance": self.domain_relevance,
            "label_confidence": self.label_confidence,
            "requires_human_review": self.requires_human_review,
            "label_model": self.label_model,
            "label_prompt_version": self.label_prompt_version,
            "source_refs": self.source_refs,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


# ── Image quality score ───────────────────────────────────────────

@dataclass
class ImageQualityScore:
    """Quality scores for a labeled image."""
    image_id: str = ""
    clarity: float = 0.0
    domain_relevance: float = 0.0
    visual_informativeness: float = 0.0
    captionability: float = 0.0
    reasoning_potential: float = 0.0
    metadata_completeness: float = 0.0
    dedup_status: DedupStatus = DedupStatus.UNKNOWN
    quality_status: QualityStatus = QualityStatus.REVIEW
    drop_reason: Optional[str] = None
    quality_model: str = "mimo-v2.5-pro"
    source_refs: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    @property
    def average(self) -> float:
        scores = [self.clarity, self.domain_relevance, self.visual_informativeness,
                  self.captionability, self.reasoning_potential]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "clarity": self.clarity,
            "domain_relevance": self.domain_relevance,
            "visual_informativeness": self.visual_informativeness,
            "captionability": self.captionability,
            "reasoning_potential": self.reasoning_potential,
            "metadata_completeness": self.metadata_completeness,
            "average": self.average,
            "dedup_status": self.dedup_status.value,
            "quality_status": self.quality_status.value,
            "drop_reason": self.drop_reason,
            "quality_model": self.quality_model,
            "source_refs": self.source_refs,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


# ── Multimodal benchmark candidate ────────────────────────────────

@dataclass
class MultimodalBenchmarkCandidate:
    """A candidate benchmark item from an image."""
    candidate_id: str = ""
    image_id: str = ""
    task_type: BenchmarkTaskType = BenchmarkTaskType.VISUAL_QA
    question: str = ""
    options: list[str] = field(default_factory=list)
    answer: str = ""
    explanation: str = ""
    visual_evidence: list[str] = field(default_factory=list)
    required_knowledge: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    answerability: AnswerabilityType = AnswerabilityType.IMAGE_ONLY
    hallucination_risk: HallucinationRisk = HallucinationRisk.LOW
    validation_status: QualityVerdict = QualityVerdict.PASSED
    critic_notes: str = ""
    source_refs: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def __post_init__(self):
        if self.candidate_id == "":
            self.candidate_id = f"mm_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "image_id": self.image_id,
            "task_type": self.task_type.value,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "explanation": self.explanation,
            "visual_evidence": self.visual_evidence,
            "required_knowledge": self.required_knowledge,
            "reasoning_steps": self.reasoning_steps,
            "difficulty": self.difficulty.value,
            "answerability": self.answerability.value,
            "hallucination_risk": self.hallucination_risk.value,
            "validation_status": self.validation_status.value,
            "critic_notes": self.critic_notes,
            "source_refs": self.source_refs,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


# ── Candidate validation record ───────────────────────────────────

@dataclass
class CandidateValidationRecord:
    """Independent critic validation of a benchmark candidate."""
    candidate_id: str = ""
    image_id: str = ""
    validation_status: QualityVerdict = QualityVerdict.PASSED
    answerability_score: float = 0.0
    visual_grounding_score: float = 0.0
    domain_reasoning_score: float = 0.0
    hallucination_risk: HallucinationRisk = HallucinationRisk.LOW
    ambiguity_score: float = 0.0
    critic_notes: str = ""
    revision_suggestion: str = ""
    critic_model: str = "mimo-v2.5-pro"
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "image_id": self.image_id,
            "validation_status": self.validation_status.value,
            "answerability_score": self.answerability_score,
            "visual_grounding_score": self.visual_grounding_score,
            "domain_reasoning_score": self.domain_reasoning_score,
            "hallucination_risk": self.hallucination_risk.value,
            "ambiguity_score": self.ambiguity_score,
            "critic_notes": self.critic_notes,
            "revision_suggestion": self.revision_suggestion,
            "critic_model": self.critic_model,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


# ── Image labeling run metadata ───────────────────────────────────

@dataclass
class ImageLabelingRunMetadata:
    """Metadata for a complete image labeling run."""
    run_id: str = ""
    mode: str = "pilot"  # pilot or full
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    model_name: str = "ModelPool(multimodal=mimo-v2-omni, quality=mimo-v2.5-pro)"
    prompt_version: str = "v1.0"
    total_images_scanned: int = 0
    total_images_indexed: int = 0
    total_metadata_matched: int = 0
    total_metadata_missing: int = 0
    total_duplicate_groups: int = 0
    total_unique_images: int = 0
    total_images_labeled: int = 0
    total_images_captioned: int = 0
    total_images_kept: int = 0
    total_images_review: int = 0
    total_images_dropped: int = 0
    total_benchmark_candidates: int = 0
    total_candidates_validated: int = 0
    total_candidates_passed: int = 0
    total_candidates_failed: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.run_id == "":
            self.run_id = f"img_run_{uuid.uuid4().hex[:8]}_{int(self.start_time)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "total_images_scanned": self.total_images_scanned,
            "total_images_indexed": self.total_images_indexed,
            "total_metadata_matched": self.total_metadata_matched,
            "total_metadata_missing": self.total_metadata_missing,
            "total_duplicate_groups": self.total_duplicate_groups,
            "total_unique_images": self.total_unique_images,
            "total_images_labeled": self.total_images_labeled,
            "total_images_captioned": self.total_images_captioned,
            "total_images_kept": self.total_images_kept,
            "total_images_review": self.total_images_review,
            "total_images_dropped": self.total_images_dropped,
            "total_benchmark_candidates": self.total_benchmark_candidates,
            "total_candidates_validated": self.total_candidates_validated,
            "total_candidates_passed": self.total_candidates_passed,
            "total_candidates_failed": self.total_candidates_failed,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "errors": self.errors,
        }