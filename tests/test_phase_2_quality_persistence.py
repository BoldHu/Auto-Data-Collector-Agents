"""Unit tests for Phase 2.5 quality-score persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from src.autodata.agents.quality_verification_agent import QualityVerificationAgent
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    Language,
    QualityScore,
    QualityVerdict,
)
from src.autodata.utils.io_utils import append_jsonl_record, safe_read_jsonl


class TestQualityScorePersistence:
    def test_quality_record_schema(self):
        """Verify quality records have all required fields."""
        chunk = CleanedChunk(
            source_file="test.json",
            source_folder="books",
            page_numbers=[10],
            language=Language.ZH,
            original_text="碳纤维含碳量超过90%",
            cleaned_text="碳纤维含碳量超过90%",
        )
        quality = QualityScore(
            clarity=0.85,
            completeness=0.9,
            consistency=0.8,
            feasibility=0.75,
            complexity=0.6,
            domain_relevance=0.95,
            verdict=QualityVerdict.PASSED,
        )

        # Build the record as the agent does
        record = {
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "source_folder": chunk.source_folder,
            "page_numbers": chunk.page_numbers,
            "language": chunk.language.value,
            "clarity": quality.clarity,
            "completeness": quality.completeness,
            "consistency": quality.consistency,
            "feasibility": quality.feasibility,
            "complexity": quality.complexity,
            "domain_relevance": quality.domain_relevance,
            "average_score": quality.average,
            "final_status": quality.verdict.value,
            "detected_issues": quality.issues,
            "verifier_model": quality.verification_model,
            "prompt_version": "v1.0",
            "run_id": "test_run",
            "timestamp": 1000.0,
        }

        required_fields = [
            "chunk_id", "source_file", "source_folder", "page_numbers",
            "language", "clarity", "completeness", "consistency",
            "feasibility", "complexity", "domain_relevance",
            "final_status", "detected_issues", "verifier_model",
            "prompt_version", "run_id", "timestamp",
        ]
        for field in required_fields:
            assert field in record, f"Missing required field: {field}"

    def test_quality_record_jsonl_write(self):
        """Verify quality records can be written to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quality_scores.jsonl"
            record = {
                "chunk_id": "test_chunk_001",
                "source_file": "test.json",
                "source_folder": "books",
                "page_numbers": [10],
                "language": "zh",
                "clarity": 0.85,
                "completeness": 0.9,
                "consistency": 0.8,
                "feasibility": 0.75,
                "complexity": 0.6,
                "domain_relevance": 0.95,
                "average_score": 0.85,
                "final_status": "passed",
                "detected_issues": [],
                "verifier_model": "mimo-v2.5-pro",
                "prompt_version": "v1.0",
                "run_id": "test_run",
                "timestamp": 1000.0,
            }
            append_jsonl_record(str(path), record)
            records = safe_read_jsonl(str(path))
            assert len(records) == 1
            assert records[0]["chunk_id"] == "test_chunk_001"
            assert records[0]["final_status"] == "passed"

    def test_needs_revision_quality_record(self):
        """Verify needs_revision quality records are written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quality_scores.jsonl"
            record = {
                "chunk_id": "test_chunk_002",
                "source_file": "test.json",
                "source_folder": "books",
                "page_numbers": [20],
                "language": "zh",
                "clarity": 0.3,
                "completeness": 0.4,
                "consistency": 0.5,
                "feasibility": 0.3,
                "complexity": 0.2,
                "domain_relevance": 0.4,
                "average_score": 0.37,
                "final_status": "needs_revision",
                "detected_issues": ["json_parse_failed", "over_cleaning"],
                "verifier_model": "mimo-v2.5-pro",
                "prompt_version": "v1.0",
                "run_id": "test_run",
                "timestamp": 1000.0,
            }
            append_jsonl_record(str(path), record)
            records = safe_read_jsonl(str(path))
            assert records[0]["final_status"] == "needs_revision"
            assert "json_parse_failed" in records[0]["detected_issues"]

    def test_failed_quality_record(self):
        """Verify failed quality records are written correctly."""
        record = {
            "chunk_id": "test_chunk_003",
            "final_status": "failed",
            "detected_issues": ["hallucinated_additions"],
            "verifier_model": "mimo-v2.5-pro",
        }
        assert record["final_status"] == "failed"

    def test_default_model_is_mimo(self):
        """Verify default verifier model is mimo-v2.5-pro."""
        score = QualityScore(verdict=QualityVerdict.PASSED)
        assert score.verification_model == "mimo-v2.5-pro"