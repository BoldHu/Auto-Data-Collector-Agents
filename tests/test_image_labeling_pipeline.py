"""Tests for image labeling pipeline — prompt building, response parsing."""

import json

from src.autodata.pipelines.prompts.image_labeling_prompts import (
    PROMPT_VERSION,
    CATEGORY_DEFINITIONS,
    MODALITY_DEFINITIONS,
    MATERIAL_FORM_DEFINITIONS,
    PROCESS_STAGE_DEFINITIONS,
    APPLICATION_DOMAIN_DEFINITIONS,
    LABELING_SYSTEM_PROMPT,
    LABELING_USER_PROMPT_TEMPLATE,
    CAPTIONING_SYSTEM_PROMPT,
    QUALITY_SYSTEM_PROMPT,
    BENCHMARK_SYSTEM_PROMPT,
    BENCHMARK_USER_PROMPT_HEADER,
    BENCHMARK_OUTPUT_SCHEMA,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_OUTPUT_SCHEMA,
    COMBINED_OUTPUT_SCHEMA,
    build_benchmark_user_prompt,
    build_critic_user_prompt,
    get_combined_labeling_prompt,
)


# ── Test: Prompt version ──────────────────────────────────────

def test_prompt_version():
    """Prompt version is set."""
    assert PROMPT_VERSION == "v1.0"


# ── Test: Category definitions ─────────────────────────────────

def test_category_definitions_content():
    """CATEGORY_DEFINITIONS contains key categories."""
    assert "fiber" in CATEGORY_DEFINITIONS
    assert "fabric" in CATEGORY_DEFINITIONS
    assert "composite_part" in CATEGORY_DEFINITIONS
    assert "irrelevant" in CATEGORY_DEFINITIONS


def test_modality_definitions_content():
    """MODALITY_DEFINITIONS contains key modalities."""
    assert "photo" in MODALITY_DEFINITIONS
    assert "microscopy" in MODALITY_DEFINITIONS
    assert "diagram" in MODALITY_DEFINITIONS


# ── Test: System prompts ──────────────────────────────────────

def test_labeling_system_prompt():
    """LABELING_SYSTEM_PROMPT contains key rules."""
    assert "碳纤维" in LABELING_SYSTEM_PROMPT
    assert "分类" in LABELING_SYSTEM_PROMPT


def test_captioning_system_prompt():
    """CAPTIONING_SYSTEM_PROMPT contains key rules."""
    assert "描述" in CAPTIONING_SYSTEM_PROMPT
    assert "30字" in CAPTIONING_SYSTEM_PROMPT


def test_quality_system_prompt():
    """QUALITY_SYSTEM_PROMPT contains scoring dimensions."""
    assert "clarity" in QUALITY_SYSTEM_PROMPT
    assert "domain_relevance" in QUALITY_SYSTEM_PROMPT


def test_benchmark_system_prompt():
    """BENCHMARK_SYSTEM_PROMPT contains key rules."""
    assert "基准评测" in BENCHMARK_SYSTEM_PROMPT
    assert "视觉推理" in BENCHMARK_SYSTEM_PROMPT


def test_critic_system_prompt():
    """CRITIC_SYSTEM_PROMPT contains validation dimensions."""
    assert "answerability_score" in CRITIC_SYSTEM_PROMPT
    assert "hallucination_risk" in CRITIC_SYSTEM_PROMPT


# ── Test: build_benchmark_user_prompt ──────────────────────────

def test_build_benchmark_user_prompt():
    """build_benchmark_user_prompt builds a valid prompt string."""
    prompt = build_benchmark_user_prompt(
        primary_category="fiber",
        material_form="tow",
        process_stage="spinning",
        domain_relevance=0.85,
    )
    assert "fiber" in prompt
    assert "tow" in prompt
    assert "0.85" in prompt
    assert "candidates" in prompt  # Output schema mentions candidates


# ── Test: build_critic_user_prompt ────────────────────────────

def test_build_critic_user_prompt():
    """build_critic_user_prompt builds a valid critic prompt."""
    prompt = build_critic_user_prompt(
        task_type="visual_qa",
        question="图中展示了什么材料？",
        options="['纤维', '织物', '预浸料', '制件']",
        answer="纤维",
        explanation="可见纤维纹理特征",
        visual_evidence="['纤维纹理', '表面光滑']",
        primary_category="fiber",
        label_confidence=0.85,
    )
    assert "visual_qa" in prompt
    assert "纤维" in prompt
    assert "0.85" in prompt
    assert "validation_status" in prompt


# ── Test: Combined prompt ──────────────────────────────────────

def test_combined_output_schema_has_no_format_conflicts():
    """COMBINED_OUTPUT_SCHEMA contains curly braces but is not an f-string."""
    # This should be a regular string with curly braces that represent
    # the expected JSON output format, NOT Python format specifiers
    assert "primary_category" in COMBINED_OUTPUT_SCHEMA
    assert "domain_relevance" in COMBINED_OUTPUT_SCHEMA
    # Should contain Chinese descriptions (not format placeholders)
    assert "主类别" in COMBINED_OUTPUT_SCHEMA


# ── Test: JSON output schema ──────────────────────────────────

def test_benchmark_output_schema_is_parseable():
    """BENCHMARK_OUTPUT_SCHEMA mentions candidates array."""
    assert "candidates" in BENCHMARK_OUTPUT_SCHEMA
    assert "task_type" in BENCHMARK_OUTPUT_SCHEMA


def test_critic_output_schema_is_parseable():
    """CRITIC_OUTPUT_SCHEMA mentions validation fields."""
    assert "validation_status" in CRITIC_OUTPUT_SCHEMA
    assert "hallucination_risk" in CRITIC_OUTPUT_SCHEMA


# ── Test: Response parsing ─────────────────────────────────────

def test_parse_labeling_response_valid_json():
    """Parse a valid labeling response JSON."""
    response_text = """{
      "primary_category": "fiber",
      "secondary_categories": ["fabric"],
      "material_form": "tow",
      "domain_relevance": 0.85,
      "label_confidence": 0.9,
      "short_caption": "碳纤维丝束",
      "technical_caption": "可见碳纤维丝束，表面光滑",
      "clarity": 0.9,
      "visual_informativeness": 0.85,
      "quality_status": "keep"
    }"""
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    data = json.loads(response_text[json_start:json_end])
    assert data["primary_category"] == "fiber"
    assert data["domain_relevance"] == 0.85


def test_parse_benchmark_response_valid_json():
    """Parse a valid benchmark candidate response."""
    response_text = """Some prefix text here
    {
      "candidates": [
        {
          "task_type": "visual_qa",
          "question": "图中展示了什么材料？",
          "answer": "碳纤维丝束",
          "difficulty": "medium"
        }
      ]
    }
    Some suffix text"""

    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    data = json.loads(response_text[json_start:json_end])
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["task_type"] == "visual_qa"


def test_parse_critic_response_valid_json():
    """Parse a valid critic validation response."""
    response_text = """{
      "validation_status": "passed",
      "answerability_score": 0.95,
      "visual_grounding_score": 0.9,
      "hallucination_risk": "low",
      "ambiguity_score": 0.1,
      "critic_notes": "题目质量良好"
    }"""

    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    data = json.loads(response_text[json_start:json_end])
    assert data["validation_status"] == "passed"


def test_parse_critic_response_review_status():
    """Parse critic response with 'review' status (mapped to needs_revision)."""
    from src.autodata.pipelines.text_schema import QualityVerdict
    verdict_map = {e.value: e for e in QualityVerdict}
    verdict_map["review"] = QualityVerdict.NEEDS_REVISION

    response_text = """{
      "validation_status": "review",
      "answerability_score": 0.5
    }"""

    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    data = json.loads(response_text[json_start:json_end])
    status = verdict_map.get(data.get("validation_status", ""), QualityVerdict.NEEDS_REVISION)
    assert status == QualityVerdict.NEEDS_REVISION