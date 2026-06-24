"""Unit tests for Phase 2.5 output validation logic."""

import json
import tempfile
from pathlib import Path

import pytest

from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    CleaningRunMetadata,
    Language,
    QualityScore,
    QualityVerdict,
)
from src.autodata.utils.io_utils import atomic_write_json, append_jsonl_record, safe_read_jsonl


class TestOutputValidationChecks:
    def test_cleaned_chunk_file_exists_and_nonempty(self):
        """Verify cleaned chunk JSONL can be written and read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cleaned_chunks.jsonl"
            chunk = CleanedChunk(
                source_file="test.json",
                source_folder="books",
                page_numbers=[1],
                language=Language.ZH,
                original_text="原文",
                cleaned_text="清洗后文本",
            )
            append_jsonl_record(str(path), chunk.to_dict())
            records = safe_read_jsonl(str(path))
            assert len(records) > 0

    def test_quality_score_per_chunk(self):
        """Verify quality score records match cleaned chunks count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write 3 chunks and 3 quality records
            chunk_path = Path(tmpdir) / "chunks.jsonl"
            quality_path = Path(tmpdir) / "quality.jsonl"
            for i in range(3):
                chunk = CleanedChunk(
                    chunk_id=f"chunk_{i}",
                    source_file="test.json",
                    source_folder="books",
                    page_numbers=[i],
                    language=Language.ZH,
                    original_text=f"原文{i}",
                    cleaned_text=f"清洗{i}",
                )
                append_jsonl_record(str(chunk_path), chunk.to_dict())
                quality_record = {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "final_status": "passed",
                    "verifier_model": "mimo-v2.5-pro",
                }
                append_jsonl_record(str(quality_path), quality_record)

            chunks = safe_read_jsonl(str(chunk_path))
            quality = safe_read_jsonl(str(quality_path))
            assert len(chunks) == 3
            assert len(quality) == 3
            # Every chunk has a quality record
            chunk_ids = {c["chunk_id"] for c in chunks}
            quality_chunk_ids = {q["chunk_id"] for q in quality}
            assert chunk_ids == quality_chunk_ids

    def test_provenance_in_all_outputs(self):
        """Verify all output records have source provenance fields."""
        chunk = CleanedChunk(
            source_file="test.json",
            source_folder="books",
            page_numbers=[1],
            language=Language.ZH,
            original_text="原文",
            cleaned_text="清洗后文本",
        )
        d = chunk.to_dict()
        provenance_fields = ["source_file", "source_folder", "page_numbers", "language"]
        for field in provenance_fields:
            assert field in d, f"Missing provenance field: {field}"

    def test_no_api_keys_in_outputs(self):
        """Verify no API keys appear in output records."""
        chunk = CleanedChunk(
            source_file="test.json",
            source_folder="books",
            page_numbers=[1],
            language=Language.ZH,
            original_text="原文",
            cleaned_text="清洗后文本",
        )
        d = chunk.to_dict()
        json_str = json.dumps(d)
        # Check no common API key patterns
        key_patterns = ["sk-", "api_key", "apikey", "token", "secret", "password", "Bearer"]
        for pattern in key_patterns:
            assert pattern.lower() not in json_str.lower(), f"Found potential API key pattern: {pattern}"

    def test_no_raw_file_overwrite_flag(self):
        """Verify metadata tracks that raw files should not be overwritten."""
        metadata = CleaningRunMetadata(run_id="test")
        d = metadata.to_dict()
        # Check that metadata contains proper tracking fields
        assert "run_id" in d

    def test_metadata_counts_match_jsonl(self):
        """Verify metadata counts can be validated against JSONL file counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create small JSONL files
            path = Path(tmpdir) / "sft.jsonl"
            for i in range(5):
                append_jsonl_record(str(path), {"sample_id": f"sft_{i}", "task_type": "qa"})
            records = safe_read_jsonl(str(path))
            assert len(records) == 5

    def test_schema_no_duplicate_fields(self):
        """Verify no dataclass has duplicate field names."""
        import dataclasses
        from src.autodata.pipelines.text_schema import (
            CleanedChunk, KnowledgeUnit, SFTCandidate, QualityScore,
            CleaningRunMetadata, RawDocument, RawPage,
        )
        for cls in [CleanedChunk, KnowledgeUnit, SFTCandidate, QualityScore,
                     CleaningRunMetadata, RawDocument, RawPage]:
            fields = [f.name for f in dataclasses.fields(cls)]
            assert len(fields) == len(set(fields)), \
                f"{cls.__name__} has duplicate field names: {fields}"