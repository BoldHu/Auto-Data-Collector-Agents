"""Tests for image schema dataclasses and enums."""

import json
import time
from pathlib import Path

from src.autodata.pipelines.image_schema import (
    ImageCategory,
    ImageModality,
    MaterialForm,
    ProcessStage,
    ApplicationDomain,
    QualityStatus,
    AnswerabilityType,
    BenchmarkTaskType,
    Difficulty,
    HallucinationRisk,
    QualityVerdict,
    ImageManifestItem,
    ImageLabelRecord,
    ImageCaptionRecord,
    ImageQualityScore,
    MultimodalBenchmarkCandidate,
    CandidateValidationRecord,
)


def test_image_category_enum():
    """ImageCategory enum values are correct."""
    assert ImageCategory.FIBER.value == "fiber"
    assert ImageCategory.FABRIC.value == "fabric"
    assert ImageCategory.PREPREG.value == "prepreg"
    assert ImageCategory.COMPOSITE_PART.value == "composite_part"
    assert ImageCategory.MICROSTRUCTURE.value == "microstructure"
    assert ImageCategory.EQUIPMENT.value == "equipment"
    assert ImageCategory.PROCESS.value == "process"
    assert ImageCategory.APPLICATION.value == "application"
    assert ImageCategory.TESTING.value == "testing"
    assert ImageCategory.DEFECT.value == "defect"
    assert ImageCategory.CHART_DIAGRAM.value == "chart_diagram"
    assert ImageCategory.PAPER_SCREENSHOT.value == "paper_screenshot"
    assert ImageCategory.IRRELEVANT.value == "irrelevant"


def test_image_category_from_string():
    """ImageCategory can be constructed from string value."""
    assert ImageCategory("fiber") == ImageCategory.FIBER
    assert ImageCategory("defect") == ImageCategory.DEFECT


def test_image_category_unknown_fallback():
    """ImageCategory UNKNOWN exists for unmapped values."""
    assert ImageCategory.UNKNOWN.value == "unknown"


def test_modality_enum():
    """ImageModality enum covers expected values."""
    assert ImageModality.PHOTO.value == "photo"
    assert ImageModality.MICROSCOPY.value == "microscopy"
    assert ImageModality.DIAGRAM.value == "diagram"
    assert ImageModality.CHART.value == "chart"
    assert ImageModality.UNKNOWN.value == "unknown"


def test_material_form_enum():
    """MaterialForm enum covers expected values."""
    assert MaterialForm.RAW_FIBER.value == "raw_fiber"
    assert MaterialForm.TOW.value == "tow"
    assert MaterialForm.FABRIC.value == "fabric"
    assert MaterialForm.PREPREG.value == "prepreg"
    assert MaterialForm.LAMINATE.value == "laminate"
    assert MaterialForm.UNKNOWN.value == "unknown"


def test_process_stage_enum():
    """ProcessStage enum covers expected values."""
    assert ProcessStage.CARBONIZATION.value == "carbonization"
    assert ProcessStage.CURING.value == "curing"
    assert ProcessStage.UNKNOWN.value == "unknown"


def test_quality_verdict_enum():
    """QualityVerdict has PASSED, NEEDS_REVISION, FAILED."""
    assert QualityVerdict.PASSED.value == "passed"
    assert QualityVerdict.NEEDS_REVISION.value == "needs_revision"
    assert QualityVerdict.FAILED.value == "failed"


def test_difficulty_enum():
    """Difficulty enum covers easy, medium, hard."""
    assert Difficulty.EASY.value == "easy"
    assert Difficulty.MEDIUM.value == "medium"
    assert Difficulty.HARD.value == "hard"


def test_hallucination_risk_enum():
    """HallucinationRisk covers low, medium, high."""
    assert HallucinationRisk.LOW.value == "low"
    assert HallucinationRisk.MEDIUM.value == "medium"
    assert HallucinationRisk.HIGH.value == "high"


def test_benchmark_task_type_enum():
    """BenchmarkTaskType covers all task types."""
    assert BenchmarkTaskType.VISUAL_QA.value == "visual_qa"
    assert BenchmarkTaskType.MULTIPLE_CHOICE.value == "multiple_choice"
    assert BenchmarkTaskType.SHORT_ANSWER.value == "short_answer"
    assert BenchmarkTaskType.PROCESS_REASONING.value == "process_reasoning"
    assert BenchmarkTaskType.DEFECT_DIAGNOSIS.value == "defect_diagnosis"
    assert BenchmarkTaskType.CHART_READING.value == "chart_reading"


def test_answerability_type_enum():
    """AnswerabilityType covers all types."""
    assert AnswerabilityType.IMAGE_ONLY.value == "image_only"
    assert AnswerabilityType.IMAGE_PLUS_METADATA.value == "image_plus_metadata"
    assert AnswerabilityType.IMAGE_PLUS_DOMAIN_KNOWLEDGE.value == "image_plus_domain_knowledge"


def test_image_manifest_item():
    """ImageManifestItem creates correctly and converts to dict."""
    item = ImageManifestItem(
        image_id="img_001",
        file_path="/path/to/nonexistent.jpg",
        folder_keyword="carbon_fiber_tow",
        file_size=144000,
        width=800,
        height=600,
    )
    d = item.to_dict()
    assert d["image_id"] == "img_001"
    assert d["folder_keyword"] == "carbon_fiber_tow"
    assert d["file_size"] == 144000
    assert d["width"] == 800


def test_image_manifest_item_auto_fields():
    """ImageManifestItem auto-generates image_id when empty."""
    item = ImageManifestItem()
    assert item.image_id.startswith("img_")


def test_image_label_record():
    """ImageLabelRecord creates correctly and converts to dict."""
    rec = ImageLabelRecord(
        image_id="img_001",
        primary_category=ImageCategory.FIBER,
        domain_relevance=0.85,
        label_confidence=0.9,
    )
    d = rec.to_dict()
    assert d["image_id"] == "img_001"
    assert d["primary_category"] == "fiber"
    assert d["domain_relevance"] == 0.85
    assert d["label_confidence"] == 0.9


def test_image_caption_record():
    """ImageCaptionRecord creates correctly."""
    rec = ImageCaptionRecord(
        image_id="img_001",
        short_caption="碳纤维丝束",
        technical_caption="可见碳纤维丝束，表面光滑",
    )
    d = rec.to_dict()
    assert d["short_caption"] == "碳纤维丝束"
    assert d["technical_caption"] == "可见碳纤维丝束，表面光滑"


def test_image_quality_score():
    """ImageQualityScore creates correctly."""
    rec = ImageQualityScore(
        image_id="img_001",
        clarity=0.9,
        domain_relevance=0.85,
        quality_status=QualityStatus.KEEP,
    )
    d = rec.to_dict()
    assert d["clarity"] == 0.9
    assert d["quality_status"] == "keep"


def test_multimodal_benchmark_candidate():
    """MultimodalBenchmarkCandidate creates correctly with auto ID."""
    cand = MultimodalBenchmarkCandidate(
        image_id="img_001",
        task_type=BenchmarkTaskType.VISUAL_QA,
        question="图中展示了什么材料？",
        answer="碳纤维丝束",
        explanation="可见碳纤维丝束特征纹理",
        difficulty=Difficulty.MEDIUM,
        hallucination_risk=HallucinationRisk.LOW,
    )
    assert cand.candidate_id.startswith("mm_")
    d = cand.to_dict()
    assert d["task_type"] == "visual_qa"
    assert d["difficulty"] == "medium"


def test_candidate_validation_record():
    """CandidateValidationRecord creates correctly."""
    rec = CandidateValidationRecord(
        candidate_id="mm_abc123",
        image_id="img_001",
        validation_status=QualityVerdict.PASSED,
        answerability_score=0.95,
        visual_grounding_score=0.9,
        domain_reasoning_score=0.85,
        hallucination_risk=HallucinationRisk.LOW,
        ambiguity_score=0.1,
        critic_notes="题目质量良好",
    )
    d = rec.to_dict()
    assert d["validation_status"] == "passed"
    assert d["answerability_score"] == 0.95


def test_candidate_validation_record_needs_revision():
    """CandidateValidationRecord with NEEDS_REVISION status."""
    rec = CandidateValidationRecord(
        candidate_id="mm_xyz",
        image_id="img_002",
        validation_status=QualityVerdict.NEEDS_REVISION,
        answerability_score=0.5,
        critic_notes="题目存在歧义",
        revision_suggestion="增加限定条件",
    )
    d = rec.to_dict()
    assert d["validation_status"] == "needs_revision"


def test_quality_verdict_mapping_for_review():
    """Map 'review' string to NEEDS_REVISION enum."""
    verdict_map = {e.value: e for e in QualityVerdict}
    verdict_map["review"] = QualityVerdict.NEEDS_REVISION
    assert verdict_map.get("review") == QualityVerdict.NEEDS_REVISION
    assert verdict_map.get("passed") == QualityVerdict.PASSED
    assert verdict_map.get("failed") == QualityVerdict.FAILED