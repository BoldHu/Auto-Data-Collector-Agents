"""Unit tests for SFT candidate generator module."""

import pytest

from src.autodata.pipelines.sft_candidate_generator import (
    generate_sft_candidates,
    _parse_sft_candidates,
)
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    SFTCandidate,
    SFTTaskType,
    Difficulty,
    Language,
)


class TestSFTCandidateParsing:
    def test_parse_valid_json(self):
        response = """[
            {
                "task_type": "qa",
                "instruction": "什么是碳纤维？",
                "input": "",
                "output": "碳纤维是一种含碳量在90%以上的高强度纤维材料",
                "evidence_text": "碳纤维含碳量超过90%",
                "difficulty": "easy"
            }
        ]"""
        chunk = CleanedChunk(
            source_file="test.json", source_folder="books",
            page_numbers=[10], language=Language.ZH,
            original_text="test", cleaned_text="碳纤维含碳量超过90%",
        )
        candidates = _parse_sft_candidates(response, chunk, "test_run")
        assert len(candidates) == 1
        assert candidates[0].task_type == SFTTaskType.QA
        assert candidates[0].difficulty == Difficulty.EASY

    def test_parse_invalid_json_fallback(self):
        response = "Not JSON"
        chunk = CleanedChunk(
            source_file="test.json", source_folder="books",
            page_numbers=[10], language=Language.ZH,
            original_text="碳纤维是一种含碳量在90%以上的高强度纤维材料，广泛应用于航空航天、汽车、体育器材等多个领域，具有重要的工业价值。",
            cleaned_text="碳纤维是一种含碳量在90%以上的高强度纤维材料，广泛应用于航空航天、汽车、体育器材等多个领域，具有重要的工业价值。",
        )
        candidates = _parse_sft_candidates(response, chunk, "test_run")
        # Should create a fallback candidate since cleaned_text is >50 chars
        assert len(candidates) == 1

    def test_empty_chunk_returns_empty(self):
        chunk = CleanedChunk(
            source_file="test.json", source_folder="books",
            page_numbers=[1], language=Language.ZH,
            original_text="", cleaned_text="",
            chunk_type="empty",
        )
        candidates = generate_sft_candidates(chunk)
        assert len(candidates) == 0

    def test_short_chunk_returns_empty(self):
        chunk = CleanedChunk(
            source_file="test.json", source_folder="books",
            page_numbers=[1], language=Language.ZH,
            original_text="短文本", cleaned_text="短文本",
        )
        candidates = generate_sft_candidates(chunk)
        assert len(candidates) == 0  # less than 50 chars