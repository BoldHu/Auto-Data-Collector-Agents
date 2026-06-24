"""Tests for multimodal benchmark candidate pipeline."""

import json

from src.autodata.pipelines.image_schema import (
    BenchmarkTaskType,
    Difficulty,
    HallucinationRisk,
    AnswerabilityType,
    QualityVerdict,
    MultimodalBenchmarkCandidate,
)
from src.autodata.pipelines.image_benchmark_generator import BenchmarkCandidateGenerator


# ── Test: benchmark candidate generation ──────────────────────

def test_benchmark_task_type_values():
    """All BenchmarkTaskType values are valid."""
    expected = {"visual_qa", "multiple_choice", "short_answer", "process_reasoning",
                "defect_diagnosis", "chart_reading", "diagram_reasoning",
                "ocr_reasoning", "cross_modal_reasoning", "image_classification"}
    actual = {e.value for e in BenchmarkTaskType}
    assert actual == expected


def test_select_suitable_images_filtering():
    """BenchmarkCandidateGenerator.select_suitable_images filters by domain_relevance and quality."""
    labels = {
        "img_001": {"primary_category": "fiber", "material_form": "tow",
                     "process_stage": "spinning", "domain_relevance": 0.85,
                     "label_confidence": 0.9},
        "img_002": {"primary_category": "irrelevant", "material_form": "unknown",
                     "process_stage": "unknown", "domain_relevance": 0.2,
                     "label_confidence": 0.3},
        "img_003": {"primary_category": "fabric", "material_form": "fabric",
                     "process_stage": "weaving", "domain_relevance": 0.75,
                     "label_confidence": 0.8},
    }
    quality = {
        "img_001": {"quality_status": "keep"},
        "img_002": {"quality_status": "drop"},
        "img_003": {"quality_status": "review"},
    }
    dedup_paths = {
        "img_001": "/path/img_001.jpg",
        "img_002": "/path/img_002.jpg",
        "img_003": "/path/img_003.jpg",
    }

    gen = BenchmarkCandidateGenerator.__new__(BenchmarkCandidateGenerator)
    gen.min_domain_relevance = 0.6

    suitable = gen.select_suitable_images(labels, quality, dedup_paths)
    # img_002 excluded: domain_relevance=0.2 < 0.6 AND quality_status=drop
    assert len(suitable) == 2
    suitable_ids = {s["image_id"] for s in suitable}
    assert "img_001" in suitable_ids
    assert "img_003" in suitable_ids
    assert "img_002" not in suitable_ids


def test_parse_benchmark_response_valid():
    """Parse benchmark response with valid candidates."""
    gen = BenchmarkCandidateGenerator.__new__(BenchmarkCandidateGenerator)

    response_text = """{
      "candidates": [
        {
          "task_type": "visual_qa",
          "question": "图中展示的是什么？",
          "answer": "碳纤维丝束",
          "difficulty": "medium",
          "hallucination_risk": "low"
        },
        {
          "task_type": "multiple_choice",
          "question": "该材料属于哪种形态？",
          "options": ["纤维原料", "织物", "预浸料", "制件"],
          "answer": "纤维原料",
          "difficulty": "easy"
        }
      ]
    }"""

    candidates = gen.parse_benchmark_response(response_text, "img_001")
    assert len(candidates) == 2
    assert candidates[0]["task_type"] == "visual_qa"
    assert candidates[1]["task_type"] == "multiple_choice"


def test_parse_benchmark_response_invalid_json():
    """Parse benchmark response with invalid JSON returns empty list."""
    gen = BenchmarkCandidateGenerator.__new__(BenchmarkCandidateGenerator)

    response_text = "This is not valid JSON"
    candidates = gen.parse_benchmark_response(response_text, "img_001")
    assert candidates == []


def test_parse_benchmark_response_empty_candidates():
    """Parse benchmark response with empty candidates list."""
    gen = BenchmarkCandidateGenerator.__new__(BenchmarkCandidateGenerator)

    response_text = """{"candidates": []}"""
    candidates = gen.parse_benchmark_response(response_text, "img_001")
    assert candidates == []


def test_candidate_record_build():
    """Build MultimodalBenchmarkCandidate from parsed response."""
    raw = {
        "task_type": "visual_qa",
        "question": "图中展示了什么材料？",
        "options": [],
        "answer": "碳纤维丝束",
        "explanation": "可见碳纤维丝束特征纹理",
        "visual_evidence": ["纤维纹理"],
        "required_knowledge": ["碳纤维识别"],
        "reasoning_steps": ["观察纹理→判断为纤维"],
        "difficulty": "medium",
        "answerability": "image_only",
        "hallucination_risk": "low",
    }

    task_type_map = {e.value: e for e in BenchmarkTaskType}
    difficulty_map = {e.value: e for e in Difficulty}
    answerability_map = {e.value: e for e in AnswerabilityType}
    hallucination_map = {e.value: e for e in HallucinationRisk}

    cand = MultimodalBenchmarkCandidate(
        image_id="img_001",
        task_type=task_type_map.get(raw.get("task_type", ""), BenchmarkTaskType.VISUAL_QA),
        question=raw.get("question", ""),
        options=raw.get("options", []),
        answer=raw.get("answer", ""),
        explanation=raw.get("explanation", ""),
        visual_evidence=raw.get("visual_evidence", []),
        required_knowledge=raw.get("required_knowledge", []),
        reasoning_steps=raw.get("reasoning_steps", []),
        difficulty=difficulty_map.get(raw.get("difficulty", ""), Difficulty.MEDIUM),
        answerability=answerability_map.get(raw.get("answerability", ""), AnswerabilityType.IMAGE_ONLY),
        hallucination_risk=hallucination_map.get(raw.get("hallucination_risk", ""), HallucinationRisk.LOW),
    )

    assert cand.candidate_id.startswith("mm_")
    assert cand.task_type == BenchmarkTaskType.VISUAL_QA
    assert cand.difficulty == Difficulty.MEDIUM
    d = cand.to_dict()
    assert d["task_type"] == "visual_qa"
    assert d["difficulty"] == "medium"