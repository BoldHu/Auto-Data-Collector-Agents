"""Unit tests for Phase 2.5 metadata accounting."""

import pytest

from src.autodata.pipelines.text_schema import CleaningRunMetadata


class TestMetadataAccounting:
    def test_metadata_has_all_required_fields(self):
        """Verify metadata has all required fields per Phase 2.5 spec."""
        metadata = CleaningRunMetadata(run_id="test_run")
        d = metadata.to_dict()
        required_fields = [
            "run_id", "start_time", "end_time", "model_name",
            "total_raw_files_seen", "total_files_processed",
            "total_pages_processed", "total_raw_chunks",
            "total_cleaned_chunks", "total_quality_scores",
            "total_chunks_passed", "total_chunks_needs_revision",
            "total_chunks_failed", "total_knowledge_units",
            "total_sft_candidates", "total_llm_calls",
            "total_tokens_used", "total_api_calls",
            "prompt_version", "language_filter",
        ]
        for field in required_fields:
            assert field in d, f"Missing metadata field: {field}"

    def test_metadata_default_model(self):
        """Verify default model is mimo-v2.5-pro."""
        metadata = CleaningRunMetadata(run_id="test_run")
        assert metadata.model_name == "mimo-v2.5-pro"
        assert metadata.model == "mimo-v2.5-pro"
        d = metadata.to_dict()
        assert d["model_name"] == "mimo-v2.5-pro"
        assert d["model"] == "mimo-v2.5-pro"

    def test_metadata_counters_increment(self):
        """Verify counters can be incremented correctly."""
        metadata = CleaningRunMetadata(run_id="test_run")
        metadata.total_raw_files_seen = 4
        metadata.total_files_processed = 4
        metadata.total_pages_processed = 120
        metadata.total_raw_chunks = 45
        metadata.total_cleaned_chunks = 45
        metadata.total_quality_scores = 45
        metadata.total_chunks_passed = 40
        metadata.total_chunks_needs_revision = 4
        metadata.total_chunks_failed = 1
        metadata.total_knowledge_units = 30
        metadata.total_sft_candidates = 25
        metadata.total_llm_calls = 100

        d = metadata.to_dict()
        assert d["total_raw_files_seen"] == 4
        assert d["total_cleaned_chunks"] == 45
        assert d["total_quality_scores"] == 45
        assert d["total_knowledge_units"] == 30
        assert d["total_sft_candidates"] == 25

    def test_metadata_serialization_roundtrip(self):
        """Verify metadata can serialize and deserialize."""
        import json
        metadata = CleaningRunMetadata(
            run_id="test_run",
            model_name="mimo-v2.5-pro",
            total_knowledge_units=10,
            total_sft_candidates=5,
        )
        d = metadata.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        assert d2["run_id"] == "test_run"
        assert d2["total_knowledge_units"] == 10
        assert d2["total_sft_candidates"] == 5