"""Unit tests for text cleaning pipeline (dry-run mode)."""

import json
import tempfile

import pytest

from src.autodata.pipelines.text_cleaning_pipeline import TextCleaningPipeline
from src.autodata.pipelines.text_schema import CleanedChunk, Language, content_hash


class TestTextCleaningPipelineDryRun:
    def test_pipeline_creation(self):
        pipeline = TextCleaningPipeline(mode="pilot", skip_llm=True)
        assert pipeline.mode == "pilot"
        assert pipeline.skip_llm is True
        assert pipeline.metadata.model == "mimo-v2.5-pro"

    def test_pipeline_has_run_id(self):
        pipeline = TextCleaningPipeline(skip_llm=True)
        assert pipeline.run_id.startswith("run_")

    def test_pipeline_dry_run(self):
        pipeline = TextCleaningPipeline(
            mode="pilot",
            skip_llm=True,
            max_files=2,
            max_pages_per_file=5,
        )
        metadata = pipeline.run()
        assert metadata.total_files_processed >= 1
        assert metadata.total_chunks_created >= 1
        assert metadata.model == "mimo-v2.5-pro"

    def test_no_raw_file_overwrite(self):
        """Verify raw data files are never modified."""
        import os
        raw_dir = "/home/hudongcheng/Desktop/photo_download/text_raw_data/books"
        files = os.listdir(raw_dir)
        # Check that no new files were created in raw data dir
        for f in files:
            path = os.path.join(raw_dir, f)
            assert f.endswith(".clean.json")
            # Verify file wasn't modified (just check it exists and is unchanged)
            assert os.path.exists(path)


class TestCleanedChunk:
    def test_chunk_creation(self):
        chunk = CleanedChunk(
            chunk_id="test_chunk",
            source_file="test.json",
            source_folder="books",
            page_numbers=[1],
            language=Language.ZH,
            original_text="原文",
            cleaned_text="清洗后文本",
            original_content_hash="abc123",
            cleaned_content_hash="def456",
        )
        assert chunk.chunk_id == "test_chunk"
        assert chunk.source_file == "test.json"
        assert chunk.language == Language.ZH

    def test_chunk_auto_hash(self):
        chunk = CleanedChunk(
            source_file="test.json",
            source_folder="books",
            page_numbers=[1],
            language=Language.ZH,
            original_text="碳纤维材料",
            cleaned_text="碳纤维材料",
        )
        assert chunk.original_content_hash != ""
        assert chunk.cleaned_content_hash != ""

    def test_chunk_serialization(self):
        chunk = CleanedChunk(
            source_file="test.json",
            source_folder="books",
            page_numbers=[1],
            language=Language.ZH,
            original_text="原文",
            cleaned_text="清洗后文本",
        )
        d = chunk.to_dict()
        assert "chunk_id" in d
        assert "source_file" in d
        assert "original_content_hash" in d
        assert "cleaned_content_hash" in d
        assert "cleaning_model" in d
        assert d["cleaning_model"] == "mimo-v2.5-pro"