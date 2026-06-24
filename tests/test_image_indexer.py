"""Tests for image indexer pipeline — filesystem scan, metadata merge, path repair."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.autodata.pipelines.image_indexer import ImageIndexer
from src.autodata.pipelines.image_schema import SourceStatus


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create a temp directory structure mimicking the project layout.

    The metadata local_paths use ./imgs/ prefix (which needs repair),
    and after repair they should resolve to the actual temp image paths.
    """
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Create fake image folders under carbon_fiber_mm
        img_dir = root / "imgs_raw_data" / "carbon_fiber_mm"
        img_dir.mkdir(parents=True)

        # Folder name must match what's in metadata local_path after repair
        # Metadata has ./imgs/carbon_fiber_mm/0061_碳纤维_热压罐/img_X.jpg
        # After repair: ./imgs_raw_data/carbon_fiber_mm/0061_碳纤维_热压罐/img_X.jpg
        folder_name = "0061_碳纤维_热压罐"
        folder_dir = img_dir / folder_name
        folder_dir.mkdir()
        from PIL import Image
        for i in range(3):
            img_path = folder_dir / f"img_{i}.jpg"
            img = Image.new("RGB", (50, 50), color=(i * 80, i * 80, i * 80))
            img.save(img_path, format="JPEG")

        # Second folder with no metadata coverage (orphan images)
        orphan_folder = img_dir / "0075_carbon_fiber_precursor"
        orphan_folder.mkdir()
        for i in range(3):
            img_path = orphan_folder / f"img_{i}.jpg"
            img = Image.new("RGB", (50, 50), color=(i * 60, i * 60, i * 60))
            img.save(img_path, format="JPEG")

        # Create fake metadata JSONL — paths use ./imgs/ prefix (needs repair)
        # After repair, paths become ./imgs_raw_data/carbon_fiber_mm/0061_碳纤维_热压罐/img_X.jpg
        meta_path = root / "imgs_raw_data" / "carbon_fiber_corpus_5911_6000.jsonl"
        with open(meta_path, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "index": i,
                    "local_path": f"./imgs/carbon_fiber_mm/{folder_name}/img_{i}.jpg",
                    "title": f"Test title {i}",
                    "keyword": "碳纤维 热压罐",
                    "keyword_labels": ["lang:zh", "碳纤维", "热压罐"],
                    "image_url": f"http://example.com/img_{i}",
                }) + "\n")

        # Create output dirs
        out_dir = root / "data" / "interim" / "image_index"
        out_dir.mkdir(parents=True)
        report_dir = root / "data" / "reports" / "phase_3_image_labeling"
        report_dir.mkdir(parents=True)

        yield root


# ── Test: scan_filesystem ──────────────────────────────────────

def test_scan_filesystem(temp_dir):
    """ImageIndexer.scan_filesystem finds JPG files in subfolders."""
    img_dir = temp_dir / "imgs_raw_data" / "carbon_fiber_mm"
    with patch("src.autodata.pipelines.image_indexer.PROJECT_ROOT", temp_dir):
        indexer = ImageIndexer(
            image_dir=img_dir,
            metadata_path=temp_dir / "imgs_raw_data" / "carbon_fiber_corpus_5911_6000.jsonl",
            output_dir=temp_dir / "data" / "interim" / "image_index",
            report_dir=temp_dir / "data" / "reports" / "phase_3_image_labeling",
        )
        fs_images = indexer.scan_filesystem()
    assert len(fs_images) == 6  # 3 images per folder, 2 folders


# ── Test: load_metadata ──────────────────────────────────────

def test_load_metadata(temp_dir):
    """ImageIndexer.load_metadata loads and repairs metadata from JSONL."""
    with patch("src.autodata.pipelines.image_indexer.PROJECT_ROOT", temp_dir):
        indexer = ImageIndexer(
            image_dir=temp_dir / "imgs_raw_data" / "carbon_fiber_mm",
            metadata_path=temp_dir / "imgs_raw_data" / "carbon_fiber_corpus_5911_6000.jsonl",
            output_dir=temp_dir / "data" / "interim" / "image_index",
            report_dir=temp_dir / "data" / "reports" / "phase_3_image_labeling",
        )
        metadata = indexer.load_metadata()
    assert len(metadata) == 3
    # Verify path was repaired: ./imgs/ → ./imgs_raw_data/
    for abs_path, meta in metadata.items():
        assert "imgs_raw_data" in abs_path


def test_load_metadata_nonexistent():
    """ImageIndexer.load_metadata returns empty dict for nonexistent metadata."""
    indexer = ImageIndexer(
        image_dir=Path("/tmp/nonexistent_dir"),
        metadata_path=Path("/tmp/nonexistent_meta.jsonl"),
        output_dir=Path("/tmp/out"),
        report_dir=Path("/tmp/report"),
    )
    metadata = indexer.load_metadata()
    assert len(metadata) == 0


# ── Test: build_index ──────────────────────────────────────────

def test_build_index(temp_dir):
    """ImageIndexer.build_index merges filesystem and metadata data."""
    with patch("src.autodata.pipelines.image_indexer.PROJECT_ROOT", temp_dir):
        indexer = ImageIndexer(
            image_dir=temp_dir / "imgs_raw_data" / "carbon_fiber_mm",
            metadata_path=temp_dir / "imgs_raw_data" / "carbon_fiber_corpus_5911_6000.jsonl",
            output_dir=temp_dir / "data" / "interim" / "image_index",
            report_dir=temp_dir / "data" / "reports" / "phase_3_image_labeling",
        )
        fs_images = indexer.scan_filesystem()
        metadata = indexer.load_metadata()
        items = indexer.build_index(fs_images, metadata)

    # 6 images total
    assert len(items) == 6

    # Count source statuses
    repaired = sum(1 for i in items if i.source_status == SourceStatus.PATH_REPAIRED)
    orphan = sum(1 for i in items if i.source_status == SourceStatus.METADATA_MISSING)

    # 3 images in 0061 folder matched metadata (paths repaired)
    assert repaired == 3
    # 3 images in 0075 folder have no metadata
    assert orphan == 3


# ── Test: full index run ──────────────────────────────────────

def test_full_index_run(temp_dir):
    """Run full indexing pipeline on temp directory."""
    with patch("src.autodata.pipelines.image_indexer.PROJECT_ROOT", temp_dir):
        indexer = ImageIndexer(
            image_dir=temp_dir / "imgs_raw_data" / "carbon_fiber_mm",
            metadata_path=temp_dir / "imgs_raw_data" / "carbon_fiber_corpus_5911_6000.jsonl",
            output_dir=temp_dir / "data" / "interim" / "image_index",
            report_dir=temp_dir / "data" / "reports" / "phase_3_image_labeling",
        )
        result = indexer.run()

    assert result["total_indexed"] == 6

    # Check output file exists
    output_path = temp_dir / "data" / "interim" / "image_index" / "image_index.jsonl"
    assert output_path.exists()

    # Verify JSONL is valid
    with open(output_path) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 6
    for r in records:
        assert "image_id" in r
        assert "file_path" in r
        assert "source_status" in r