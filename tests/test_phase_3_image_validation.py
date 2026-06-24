"""Tests for Phase 3 image output validation script."""

import json
import tempfile
from pathlib import Path

from scripts.validate_phase_3_image_outputs import (
    check_file_exists,
    check_json_valid,
    check_required_fields,
    check_image_id_traceability,
    check_quality_thresholds,
)


# ── Test: check_file_exists ────────────────────────────────────

def test_check_file_exists_real():
    """check_file_exists returns True for existing non-empty file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write("test content\n")
        f.flush()
        assert check_file_exists(Path(f.name)) is True
    import os
    os.unlink(f.name)


def test_check_file_exists_empty():
    """check_file_exists returns False for empty file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.flush()
        assert check_file_exists(Path(f.name)) is False
    import os
    os.unlink(f.name)


def test_check_file_exists_nonexistent():
    """check_file_exists returns False for nonexistent file."""
    assert check_file_exists(Path("/tmp/nonexistent_999.jsonl")) is False


# ── Test: check_json_valid ─────────────────────────────────────

def test_check_json_valid_good_jsonl():
    """check_json_valid returns True for valid JSONL."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps({"a": 1}) + "\n")
        f.write(json.dumps({"b": 2}) + "\n")
        f.flush()
        assert check_json_valid(Path(f.name)) is True
    import os
    os.unlink(f.name)


def test_check_json_valid_bad_jsonl():
    """check_json_valid returns False for invalid JSON line."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps({"a": 1}) + "\n")
        f.write("not valid json\n")
        f.flush()
        assert check_json_valid(Path(f.name)) is False
    import os
    os.unlink(f.name)


def test_check_json_valid_nonexistent():
    """check_json_valid returns False for nonexistent file."""
    assert check_json_valid(Path("/tmp/nonexistent_999.jsonl")) is False


# ── Test: check_required_fields ────────────────────────────────

def test_check_required_fields_all_present():
    """check_required_fields returns True when all required fields present."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps({"image_id": "img1", "primary_category": "fiber"}) + "\n")
        f.write(json.dumps({"image_id": "img2", "primary_category": "fabric"}) + "\n")
        f.flush()
        result = check_required_fields(Path(f.name), ["image_id", "primary_category"])
        assert result is True
    import os
    os.unlink(f.name)


def test_check_required_fields_missing():
    """check_required_fields returns False when required field missing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps({"image_id": "img1", "primary_category": "fiber"}) + "\n")
        f.write(json.dumps({"image_id": "img2"}) + "\n")  # missing primary_category
        f.flush()
        result = check_required_fields(Path(f.name), ["image_id", "primary_category"])
        assert result is False
    import os
    os.unlink(f.name)


def test_check_required_fields_empty_value():
    """check_required_fields returns False when required field is empty string."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps({"image_id": "img1", "primary_category": ""}) + "\n")
        f.flush()
        result = check_required_fields(Path(f.name), ["image_id", "primary_category"])
        assert result is False
    import os
    os.unlink(f.name)


# ── Test: check_image_id_traceability ──────────────────────────

def test_check_traceability_matching_ids():
    """check_image_id_traceability returns True when IDs match."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as label_f:
        label_f.write(json.dumps({"image_id": "img1"}) + "\n")
        label_f.write(json.dumps({"image_id": "img2"}) + "\n")
        label_f.flush()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as caption_f:
        caption_f.write(json.dumps({"image_id": "img1"}) + "\n")
        caption_f.write(json.dumps({"image_id": "img2"}) + "\n")
        caption_f.flush()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as quality_f:
        quality_f.write(json.dumps({"image_id": "img1"}) + "\n")
        quality_f.write(json.dumps({"image_id": "img2"}) + "\n")
        quality_f.flush()

    result = check_image_id_traceability(
        Path(label_f.name), Path(caption_f.name), Path(quality_f.name)
    )
    assert result is True

    import os
    os.unlink(label_f.name)
    os.unlink(caption_f.name)
    os.unlink(quality_f.name)


def test_check_traceability_mismatched_ids():
    """check_image_id_traceability returns False when IDs don't match."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as label_f:
        label_f.write(json.dumps({"image_id": "img1"}) + "\n")
        label_f.write(json.dumps({"image_id": "img2"}) + "\n")
        label_f.flush()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as caption_f:
        caption_f.write(json.dumps({"image_id": "img1"}) + "\n")
        caption_f.write(json.dumps({"image_id": "img3"}) + "\n")  # mismatch
        caption_f.flush()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as quality_f:
        quality_f.write(json.dumps({"image_id": "img1"}) + "\n")
        quality_f.write(json.dumps({"image_id": "img2"}) + "\n")
        quality_f.flush()

    result = check_image_id_traceability(
        Path(label_f.name), Path(caption_f.name), Path(quality_f.name)
    )
    assert result is False

    import os
    os.unlink(label_f.name)
    os.unlink(caption_f.name)
    os.unlink(quality_f.name)


# ── Test: check_quality_thresholds ──────────────────────────────

def test_quality_thresholds_pass():
    """check_quality_thresholds passes when thresholds are met."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as label_f:
        for i in range(100):
            label_f.write(json.dumps({
                "image_id": f"img{i}",
                "domain_relevance": 0.85,
                "label_confidence": 0.8,
            }) + "\n")
        label_f.flush()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as quality_f:
        for i in range(100):
            quality_f.write(json.dumps({
                "image_id": f"img{i}",
                "quality_status": "keep",
            }) + "\n")
        quality_f.flush()

    thresholds = check_quality_thresholds(Path(label_f.name), Path(quality_f.name))
    assert thresholds["caption_faithful_pass"] is True  # 100% keep >= 85%
    assert thresholds["label_reasonable_pass"] is True  # 100% >= 0.5 confidence
    assert thresholds["avg_relevance_pass"] is True  # avg 0.85 >= 0.6

    import os
    os.unlink(label_f.name)
    os.unlink(quality_f.name)


def test_quality_thresholds_fail():
    """check_quality_thresholds fails when thresholds not met."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as label_f:
        for i in range(100):
            label_f.write(json.dumps({
                "image_id": f"img{i}",
                "domain_relevance": 0.3,
                "label_confidence": 0.2,
            }) + "\n")
        label_f.flush()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as quality_f:
        for i in range(100):
            quality_f.write(json.dumps({
                "image_id": f"img{i}",
                "quality_status": "drop",
            }) + "\n")
        quality_f.flush()

    thresholds = check_quality_thresholds(Path(label_f.name), Path(quality_f.name))
    assert thresholds["caption_faithful_pass"] is False  # 0% keep < 85%
    assert thresholds["label_reasonable_pass"] is False  # 0% >= 0.5
    assert thresholds["avg_relevance_pass"] is False  # avg 0.3 < 0.6

    import os
    os.unlink(label_f.name)
    os.unlink(quality_f.name)