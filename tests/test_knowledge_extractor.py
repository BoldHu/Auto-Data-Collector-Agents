"""Unit tests for knowledge extractor module."""

import pytest

from src.autodata.pipelines.knowledge_extractor import (
    extract_knowledge_units,
    _parse_knowledge_units,
)
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    KnowledgeUnit,
    KnowledgeType,
    Language,
)


class TestKnowledgeUnitParsing:
    def test_parse_valid_json(self):
        response = """[
            {
                "topic": "碳纤维制备",
                "subtopic": "PAN基碳纤维",
                "knowledge_type": "process",
                "claim": "PAN基碳纤维通过预氧化和碳化两个步骤制备",
                "evidence_text": "PAN纤维首先在200-300°C下预氧化",
                "entities": ["PAN纤维", "预氧化", "碳化"],
                "relations": ["PAN纤维→预氧化→碳化→碳纤维"],
                "conditions": ["温度200-300°C"],
                "numeric_values": [{"value": 200, "unit": "°C", "context": "预氧化温度"}]
            }
        ]"""
        chunk = CleanedChunk(
            source_file="test.json", source_folder="books",
            page_numbers=[10], language=Language.ZH,
            original_text="test", cleaned_text="test",
        )
        units = _parse_knowledge_units(response, chunk, "test_run")
        assert len(units) == 1
        assert units[0].topic == "碳纤维制备"
        assert units[0].knowledge_type == KnowledgeType.PROCESS

    def test_parse_invalid_json_fallback(self):
        response = "This is not JSON at all"
        chunk = CleanedChunk(
            source_file="test.json", source_folder="books",
            page_numbers=[10], language=Language.ZH,
            original_text="碳纤维材料", cleaned_text="碳纤维材料",
        )
        units = _parse_knowledge_units(response, chunk, "test_run")
        # Should create a fallback unit
        assert len(units) == 1
        assert units[0].knowledge_type == KnowledgeType.OTHER

    def test_empty_chunk_returns_empty(self):
        chunk = CleanedChunk(
            source_file="test.json", source_folder="books",
            page_numbers=[1], language=Language.ZH,
            original_text="", cleaned_text="",
            chunk_type="empty",
        )
        units = extract_knowledge_units(chunk)
        assert len(units) == 0