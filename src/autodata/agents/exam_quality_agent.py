"""Exam quality verification agent for Phase 4.

Independent critic that verifies extracted exam questions.
Uses API_KEY1 only via ModelPool.
"""

from __future__ import annotations

import json
from typing import Any

from src.autodata.pipelines.exam_schema import ExamQualityScore
from src.autodata.pipelines.prompts.exam_extraction_prompts import (
    QUALITY_SYSTEM_PROMPT,
    get_quality_prompt,
)
from src.autodata.utils.model_pool import ModelPool


class ExamQualityAgent:
    """Verifies quality of extracted exam questions.

    Uses API_KEY1 only via ModelPool with chat_quality() for text-only model.
    """

    def __init__(
        self,
        pool: ModelPool,
        run_id: str = "phase_4_exam_extraction",
    ) -> None:
        self.pool = pool
        self.run_id = run_id

    def verify_question(
        self,
        question_dict: dict,
        raw_evidence: str = "",
    ) -> ExamQualityScore:
        """Verify a single extracted question.

        Args:
            question_dict: Question as dict (from ExamQuestion.to_dict())
            raw_evidence: Original text evidence for context

        Returns:
            ExamQualityScore with quality assessment.
        """
        question_id = question_dict.get("question_id", "unknown")

        # Build prompt
        question_json = json.dumps(question_dict, ensure_ascii=False, indent=2)
        user_prompt = get_quality_prompt(question_json, raw_evidence[:2000])

        # Call LLM (text-only quality model)
        try:
            response = self.pool.chat_quality(
                messages=[
                    {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=2048,
            )
            response_text = response.content
        except Exception:
            return ExamQualityScore(
                question_id=question_id,
                quality_status="review",
                detected_issues=["LLM call failed"],
                run_id=self.run_id,
            )

        # Parse response
        result = self._parse_response(response_text, question_id)
        return result

    def _parse_response(self, response_text: str, question_id: str) -> ExamQualityScore:
        """Parse LLM response into ExamQualityScore."""
        # Try to find JSON object
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(response_text[json_start:json_end])
                return ExamQualityScore(
                    question_id=question_id,
                    quality_status=data.get("quality_status", "review"),
                    clarity=data.get("clarity", 0.0),
                    completeness=data.get("completeness", 0.0),
                    answerability=data.get("answerability", 0.0),
                    option_integrity=data.get("option_integrity", 0.0),
                    answer_consistency=data.get("answer_consistency", 0.0),
                    domain_relevance=data.get("domain_relevance", 0.0),
                    difficulty_reasonableness=data.get("difficulty_reasonableness", 0.0),
                    benchmark_usefulness=data.get("benchmark_usefulness", 0.0),
                    detected_issues=data.get("detected_issues", []),
                    revision_suggestion=data.get("revision_suggestion", ""),
                    run_id=self.run_id,
                )
            except (json.JSONDecodeError, Exception):
                pass

        # Fallback: return review status
        return ExamQualityScore(
            question_id=question_id,
            quality_status="review",
            detected_issues=["Failed to parse quality response"],
            run_id=self.run_id,
        )
