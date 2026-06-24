"""Benchmark taxonomy for Phase 5.

Defines task categories, mappings, and balancing rules.
"""

from __future__ import annotations

# Task category definitions
TEXT_TASKS = [
    "domain_knowledge_qa",
    "exam_single_choice",
    "exam_multiple_choice",
    "exam_true_false",
    "exam_fill_blank",
    "exam_short_answer",
    "exam_calculation",
    "information_extraction",
    "process_reasoning",
    "constraint_satisfaction",
    "causal_reasoning",
    "error_diagnosis",
    "source_grounded_reasoning",
]

MULTIMODAL_TASKS = [
    "visual_qa",
    "image_classification",
    "material_form_recognition",
    "process_stage_identification",
    "equipment_function_reasoning",
    "defect_diagnosis",
    "chart_reading",
    "diagram_reasoning",
    "ocr_reasoning",
    "cross_modal_reasoning",
    "multi_step_visual_reasoning",
]

ALL_TASKS = TEXT_TASKS + MULTIMODAL_TASKS

# Mapping from source task types to benchmark task types
MULTIMODAL_TASK_MAP = {
    "process_reasoning": "process_reasoning",
    "multiple_choice": "exam_single_choice",
    "visual_qa": "visual_qa",
    "short_answer": "exam_short_answer",
    "defect_diagnosis": "defect_diagnosis",
    "diagram_reasoning": "diagram_reasoning",
    "cross_modal_reasoning": "cross_modal_reasoning",
    "chart_reading": "chart_reading",
    "ocr_reasoning": "ocr_reasoning",
}

EXAM_TASK_MAP = {
    "single_choice": "exam_single_choice",
    "multiple_choice": "exam_multiple_choice",
    "true_false": "exam_true_false",
    "fill_blank": "exam_fill_blank",
    "short_answer": "exam_short_answer",
    "calculation": "exam_calculation",
    "case_analysis": "source_grounded_reasoning",
    "unknown": "domain_knowledge_qa",
}

# Difficulty balance targets
DIFFICULTY_TARGETS = {
    "easy": 0.30,
    "medium": 0.50,
    "hard": 0.20,
}

# Modality balance targets
MODALITY_TARGETS = {
    "text": 0.30,
    "image": 0.10,
    "multimodal": 0.60,
}

# Minimum items per task type
MIN_ITEMS_PER_TASK = 10

# Maximum items per task type (to prevent domination)
MAX_ITEMS_PER_TASK = 300


def map_multimodal_task_type(source_task_type: str) -> str:
    """Map multimodal source task type to benchmark task type."""
    return MULTIMODAL_TASK_MAP.get(source_task_type, "visual_qa")


def map_exam_task_type(source_question_type: str) -> str:
    """Map exam question type to benchmark task type."""
    return EXAM_TASK_MAP.get(source_question_type, "domain_knowledge_qa")
