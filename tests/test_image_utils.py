"""Tests for image utilities — base64 encoding, resizing, validation."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.autodata.utils.image_utils import (
    encode_image_to_base64,
    resize_image_for_api,
    validate_image_file,
    get_image_dimensions,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_image_path():
    """Create a small sample JPG image for testing."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(f.name, format="JPEG")
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def large_image_path():
    """Create a large image (>1024x1024) to test resizing."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (2000, 1500), color=(128, 128, 128))
        img.save(f.name, format="JPEG")
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def nonexistent_path():
    """Return path to a file that doesn't exist."""
    return "/tmp/nonexistent_image_99999.jpg"


# ── Test: validate_image_file ─────────────────────────────────

def test_validate_image_file_valid(sample_image_path):
    """validate_image_file returns valid=True for a valid JPG."""
    result = validate_image_file(sample_image_path)
    assert result["valid"] is True


def test_validate_image_file_nonexistent(nonexistent_path):
    """validate_image_file returns valid=False for nonexistent file."""
    result = validate_image_file(nonexistent_path)
    assert result["valid"] is False
    assert result["reason"] == "file_not_found"


def test_validate_image_file_not_image():
    """validate_image_file returns valid=False for a non-image file."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not an image")
        f.flush()
        result = validate_image_file(f.name)
        assert result["valid"] is False
    os.unlink(f.name)


# ── Test: get_image_dimensions ─────────────────────────────

def test_get_dimensions_small(sample_image_path):
    """get_image_dimensions returns correct dimensions."""
    w, h = get_image_dimensions(sample_image_path)
    assert w == 100
    assert h == 100


def test_get_dimensions_nonexistent(nonexistent_path):
    """get_image_dimensions returns (0, 0) for nonexistent file."""
    w, h = get_image_dimensions(nonexistent_path)
    assert w == 0
    assert h == 0


# ── Test: encode_image_to_base64 ──────────────────────────────

def test_encode_to_base64(sample_image_path):
    """encode_image_to_base64 produces a valid base64 string."""
    result = encode_image_to_base64(sample_image_path)
    assert result.startswith("data:image/jpeg;base64,")
    # Verify the base64 part is valid
    b64_part = result.split(",", 1)[1]
    import base64
    decoded = base64.b64decode(b64_part)
    assert len(decoded) > 0


def test_encode_to_base64_nonexistent(nonexistent_path):
    """encode_image_to_base64 raises FileNotFoundError for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        encode_image_to_base64(nonexistent_path)


# ── Test: resize_image_for_api ─────────────────────────────────

def test_resize_small_image(sample_image_path):
    """resize_image_for_api returns base64 for small image (no resize needed)."""
    result = resize_image_for_api(sample_image_path)
    assert result.startswith("data:image/jpeg;base64,")


def test_resize_large_image(large_image_path):
    """resize_image_for_api resizes large images to fit API limits."""
    result = resize_image_for_api(large_image_path)
    assert result.startswith("data:image/jpeg;base64,")
    # Verify the resulting image is within bounds
    b64_part = result.split(",", 1)[1]
    import base64
    from PIL import Image
    import io
    decoded = base64.b64decode(b64_part)
    img = Image.open(io.BytesIO(decoded))
    assert max(img.width, img.height) <= 1024


def test_resize_nonexistent(nonexistent_path):
    """resize_image_for_api raises FileNotFoundError for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        resize_image_for_api(nonexistent_path)