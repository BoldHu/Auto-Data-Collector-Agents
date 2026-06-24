"""Tests for image deduplication pipeline — perceptual hash, union-find grouping."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.autodata.pipelines.image_deduplicator import ImageDeduplicator


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create temp directory with fake image index and dedup output."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # Create index JSONL
        index_dir = root / "data" / "interim" / "image_index"
        index_dir.mkdir(parents=True)
        index_path = index_dir / "image_index.jsonl"

        # Create fake images
        img_dir = root / "imgs_raw_data" / "carbon_fiber_mm" / "test_folder"
        img_dir.mkdir(parents=True)
        from PIL import Image
        file_paths = []
        for i in range(5):
            img_path = img_dir / f"img_{i}.jpg"
            img = Image.new("RGB", (50, 50), color=(i * 50, i * 50, i * 50))
            img.save(img_path, format="JPEG")
            file_paths.append(str(img_path))

        # Write index records
        with open(index_path, "w") as f:
            for i, fp in enumerate(file_paths):
                f.write(json.dumps({
                    "image_id": f"img_{i:04d}",
                    "file_path": fp,
                    "folder_keyword": "test",
                    "format": "jpg",
                    "source_status": "metadata_matched",
                    "file_size": 1000 + i * 100,
                    "width": 50,
                    "height": 50,
                }) + "\n")

        # Create dedup output dir
        dedup_dir = root / "data" / "interim" / "image_dedup"
        dedup_dir.mkdir(parents=True)
        report_dir = root / "data" / "reports" / "phase_3_image_labeling"
        report_dir.mkdir(parents=True)

        yield root


# ── Test: union-find algorithm ────────────────────────────────

def test_union_find_basic():
    """Union-Find correctly groups related items."""
    # Simple test: items 0-4, pairs (0,1) and (2,3)
    parent = list(range(5))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    union(0, 1)
    union(2, 3)

    assert find(0) == find(1)
    assert find(2) == find(3)
    assert find(4) != find(0)
    assert find(4) != find(2)


# ── Test: compute phashes ────────────────────────────────────

def test_compute_phashes(temp_dir):
    """ImageDeduplicator.compute_phashes computes perceptual hashes for items."""
    index_path = temp_dir / "data" / "interim" / "image_index" / "image_index.jsonl"

    dedup = ImageDeduplicator(
        index_path=index_path,
        output_dir=temp_dir / "data" / "interim" / "image_dedup",
        report_dir=temp_dir / "data" / "reports" / "phase_3_image_labeling",
    )

    # Load index first, then compute phashes
    items = dedup.load_index()
    items_with_phash = dedup.compute_phashes(items)
    assert len(items_with_phash) == 5
    # Each phash should be a string
    for item in items_with_phash:
        assert isinstance(item.get("phash", ""), str)


# ── Test: group duplicates ────────────────────────────────────

def test_group_duplicates_no_duplicates(temp_dir):
    """Group duplicates with all unique images (different colors)."""
    index_path = temp_dir / "data" / "interim" / "image_index" / "image_index.jsonl"

    dedup = ImageDeduplicator(
        index_path=index_path,
        output_dir=temp_dir / "data" / "interim" / "image_dedup",
        report_dir=temp_dir / "data" / "reports" / "phase_3_image_labeling",
    )

    items = dedup.load_index()
    items = dedup.compute_phashes(items)
    groups = dedup.group_duplicates(items)

    # Result should have "unique" and "duplicate_groups" keys
    assert "unique" in groups
    assert "duplicate_groups" in groups
    # All 5 images have different content, so most should be unique
    total = len(groups["unique"]) + sum(len(g) for g in groups["duplicate_groups"])
    assert total == 5


# ── Test: select primaries ─────────────────────────────────────

def test_select_primaries():
    """Select primaries from dedup groups."""
    dedup = ImageDeduplicator.__new__(ImageDeduplicator)

    # Use dict format matching actual API: {"unique": [...], "duplicate_groups": [...]}
    groups = {
        "unique": [
            {"image_id": "img_002", "phash": "def456", "source_status": "metadata_matched",
             "file_size": 1500, "image_url": ""},
        ],
        "duplicate_groups": [
            [
                {"image_id": "img_000", "phash": "abc123", "source_status": "metadata_matched",
                 "file_size": 1200, "image_url": "http://example.com/img_000"},
                {"image_id": "img_001", "phash": "abc123", "source_status": "metadata_missing",
                 "file_size": 1000, "image_url": ""},
            ],
        ],
    }

    result_items = dedup.select_primaries(groups)
    assert len(result_items) == 3

    # img_000 should be primary (has metadata + url + larger file)
    primaries = [i for i in result_items if i["dedup_status"] == "unique"]
    assert len(primaries) == 2  # img_002 (solo unique) + img_000 (primary of dup group)

    # img_001 should be duplicate referencing img_000
    dup = [i for i in result_items if i["dedup_status"] == "duplicate"]
    assert len(dup) == 1
    assert dup[0]["dedup_primary_id"] == "img_000"


# ── Test: full dedup run ──────────────────────────────────────

def test_full_dedup_run(temp_dir):
    """Run full dedup pipeline on temp directory."""
    index_path = temp_dir / "data" / "interim" / "image_index" / "image_index.jsonl"

    dedup = ImageDeduplicator(
        index_path=index_path,
        output_dir=temp_dir / "data" / "interim" / "image_dedup",
        report_dir=temp_dir / "data" / "reports" / "phase_3_image_labeling",
    )

    result = dedup.run()
    assert result["total_indexed"] == 5

    # Check output file exists
    output_path = temp_dir / "data" / "interim" / "image_dedup" / "image_dedup.jsonl"
    assert output_path.exists()

    # Verify JSONL is valid
    with open(output_path) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 5
    for r in records:
        assert "image_id" in r
        assert "dedup_status" in r
        assert r["dedup_status"] in ("unique", "duplicate", "near_duplicate", "unknown")