"""Exam extraction agent for Phase 4.

Uses LLM (API_KEY1 only) to extract exam questions from text blocks.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from src.autodata.pipelines.exam_schema import ExamQuestion
from src.autodata.pipelines.prompts.exam_extraction_prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    get_extraction_prompt,
    is_domain_relevant,
)
from src.autodata.utils.model_pool import ModelPool


class ExamExtractionAgent:
    """Extracts exam questions from text blocks using LLM.

    Uses API_KEY1 only via ModelPool.
    """

    def __init__(
        self,
        pool: ModelPool,
        run_id: str = "phase_4_exam_extraction",
    ) -> None:
        self.pool = pool
        self.run_id = run_id

    def extract_questions(
        self,
        text_blocks: list[dict],
        source_file: str,
    ) -> list[ExamQuestion]:
        """Extract questions from text blocks.

        Processes blocks in batches to handle large files.
        Each batch is ~6000 chars to leave room for prompt overhead.

        Args:
            text_blocks: List of text block dicts with 'text', 'block_id', etc.
            source_file: Source file name for provenance.

        Returns:
            List of extracted ExamQuestion objects.
        """
        # Filter out error/empty blocks
        valid_blocks = [b for b in text_blocks if b.get("text") and b.get("extraction_method") != "error"]

        if not valid_blocks:
            return []

        # Check domain relevance on first batch
        sample_text = "\n\n".join(b.get("text", "") for b in valid_blocks[:20])
        if not is_domain_relevant(sample_text):
            return []

        # Split into batches of ~6000 chars
        batches = self._split_into_batches(valid_blocks, max_chars=6000)

        all_questions = []
        seen_ids = set()

        for batch_blocks in batches:
            combined_text = "\n\n".join(b.get("text", "") for b in batch_blocks)
            block_ids = [b.get("block_id", "") for b in batch_blocks]

            # Build prompt
            user_prompt = get_extraction_prompt(source_file, combined_text)

            # Call LLM
            try:
                response = self.pool.chat(
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=8192,
                )
                response_text = response.content
            except Exception:
                continue

            # Parse response
            raw_questions = self._parse_response(response_text)

            # Build ExamQuestion objects
            for raw in raw_questions:
                try:
                    qid = ExamQuestion.generate_id(source_file, str(raw.get("question_number", "")))
                    if qid in seen_ids:
                        continue
                    seen_ids.add(qid)

                    q = ExamQuestion(
                        question_id=qid,
                        source_file=source_file,
                        source_block_ids=block_ids,
                        question_number=str(raw.get("question_number", "")),
                        question_type=raw.get("question_type", "unknown"),
                        question_text=raw.get("question_text", ""),
                        options=raw.get("options", []),
                        answer=raw.get("answer", ""),
                        answer_source=raw.get("answer_source", "missing"),
                        explanation=raw.get("explanation", ""),
                        knowledge_points=raw.get("knowledge_points", []),
                        difficulty=raw.get("difficulty", "medium"),
                        requires_calculation=raw.get("requires_calculation", False),
                        contains_formula=raw.get("contains_formula", False),
                        contains_table=raw.get("contains_table", False),
                        contains_image_reference=raw.get("contains_image_reference", False),
                        domain_relevance=raw.get("domain_relevance", 0.0),
                        extraction_confidence=raw.get("extraction_confidence", 0.0),
                        uncertainty_notes=raw.get("uncertainty_notes", []),
                        raw_evidence=combined_text[:500],
                        run_id=self.run_id,
                        extraction_model=response.usage.get("model", "unknown") if hasattr(response, "usage") else "unknown",
                    )
                    all_questions.append(q)
                except Exception:
                    continue

        return all_questions

    def _split_into_batches(self, blocks: list[dict], max_chars: int = 6000) -> list[list[dict]]:
        """Split text blocks into batches of approximately max_chars."""
        batches = []
        current_batch = []
        current_size = 0

        for block in blocks:
            text = block.get("text", "")
            text_len = len(text)

            if current_size + text_len > max_chars and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0

            current_batch.append(block)
            current_size += text_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def _parse_response(self, response_text: str) -> list[dict]:
        """Parse LLM response to extract question dicts."""
        # Try to find JSON array
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(response_text[json_start:json_end])
            except json.JSONDecodeError:
                pass

        # Try to find single JSON object and wrap in array
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            try:
                obj = json.loads(response_text[json_start:json_end])
                return [obj]
            except json.JSONDecodeError:
                pass

        return []
