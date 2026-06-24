"""Image indexer — scan filesystem, load metadata, merge, detect orphans, compute hashes.

Phase 3.1: Build a complete image index by:
1. Scanning all JPG files in imgs_raw_data/carbon_fiber_mm/
2. Loading metadata from carbon_fiber_corpus_5911_6000.jsonl
3. Repairing local_path prefix (./imgs/ → ./imgs_raw_data/)
4. Merging filesystem data with metadata (matched vs orphan)
5. Computing file sizes, dimensions via PIL, SHA-256 hashes
6. Writing image_index.jsonl with ImageManifestItem records

Output:
    data/interim/image_index/image_index.jsonl
    data/reports/phase_3_image_labeling/phase_3_1_index_report.json
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from src.autodata.pipelines.image_schema import (
    ImageManifestItem,
    SourceStatus,
)
from src.autodata.utils.image_utils import get_image_dimensions, validate_image_file
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("image_indexer")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = PROJECT_ROOT / "imgs_raw_data" / "carbon_fiber_mm"
METADATA_PATH = IMAGE_DIR / "carbon_fiber_corpus_5911_6000.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "image_index"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_image_labeling"


class ImageIndexer:
    """Scan filesystem, load metadata, merge into unified image index."""

    def __init__(
        self,
        image_dir: Path = IMAGE_DIR,
        metadata_path: Path = METADATA_PATH,
        output_dir: Path = OUTPUT_DIR,
        report_dir: Path = REPORT_DIR,
    ) -> None:
        self.image_dir = image_dir
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.start_time = time.time()

        # Ensure output dirs exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def scan_filesystem(self) -> dict[str, dict]:
        """Scan all JPG files in image_dir, return dict keyed by absolute path.

        For each image, record:
            - file_path (absolute)
            - relative_path (relative to project root)
            - folder_keyword (folder name)
            - folder_index (first 4-digit number from folder name)
            - file_size
            - width, height (from PIL)
        """
        logger.info(f"Scanning filesystem: {self.image_dir}")
        fs_images = {}
        total_scanned = 0
        empty_folders = 0
        non_empty_folders = 0

        for folder_name in sorted(os.listdir(self.image_dir)):
            folder_path = self.image_dir / folder_name
            if not folder_path.is_dir():
                continue

            jpgs = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".jpg")
            ])

            if len(jpgs) == 0:
                empty_folders += 1
                continue

            non_empty_folders += 1

            # Extract keyword from folder name
            # Format: "0061_碳纤维_热压罐_固化_现场" or "0075_carbon_fiber_precursor_factory"
            folder_keyword = folder_name
            folder_index = ""
            parts = folder_name.split("_")
            if parts and parts[0].isdigit():
                folder_index = parts[0]

            for jpg_name in jpgs:
                abs_path = folder_path / jpg_name
                rel_path = abs_path.relative_to(PROJECT_ROOT)
                size = abs_path.stat().st_size
                w, h = get_image_dimensions(abs_path)

                fs_images[str(abs_path)] = {
                    "file_path": str(abs_path),
                    "relative_path": str(rel_path),
                    "folder_keyword": folder_keyword,
                    "folder_index": folder_index,
                    "file_size": size,
                    "width": w,
                    "height": h,
                }
                total_scanned += 1

        logger.info(
            f"Scanned {total_scanned} images across {non_empty_folders} folders "
            f"({empty_folders} empty folders)"
        )
        return fs_images

    def load_metadata(self) -> dict[str, dict]:
        """Load metadata JSONL, repair local_path prefix, return dict keyed by absolute path.

        Repairs: ./imgs/ → ./imgs_raw_data/ in local_path field.
        """
        logger.info(f"Loading metadata: {self.metadata_path}")
        metadata = {}
        total_loaded = 0

        if not self.metadata_path.exists():
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            return metadata

        with open(self.metadata_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                local_path = record.get("local_path", "")

                # Repair path prefix: ./imgs/ → ./imgs_raw_data/
                if local_path.startswith("./imgs/"):
                    local_path = local_path.replace("./imgs/", "./imgs_raw_data/", 1)

                # Convert to absolute path
                abs_path = str(PROJECT_ROOT / local_path)

                metadata[abs_path] = {
                    "metadata_index": record.get("index", 0),
                    "metadata_title": record.get("title", ""),
                    "metadata_keyword": record.get("keyword", ""),
                    "metadata_keyword_labels": record.get("keyword_labels", []),
                    "image_url": record.get("image_url", ""),
                    "local_path_repaired": local_path,
                }
                total_loaded += 1

        logger.info(f"Loaded {total_loaded} metadata records")
        return metadata

    def build_index(
        self,
        fs_images: dict[str, dict],
        metadata: dict[str, dict],
    ) -> list[ImageManifestItem]:
        """Merge filesystem data with metadata to build image index.

        Three categories:
        1. METADATA_MATCHED: image on disk AND in metadata → best case
        2. METADATA_MISSING: image on disk but NOT in metadata → orphan
        3. PATH_REPAIRED: metadata path was repaired (./imgs/ → ./imgs_raw_data/)
        """
        logger.info("Building image index...")
        items = []
        matched = 0
        orphan = 0
        repaired = 0
        invalid = 0

        # Process all filesystem images
        for abs_path, fs_data in fs_images.items():
            meta_data = metadata.get(abs_path, {})

            # Determine source status
            if meta_data:
                local_repaired = meta_data.get("local_path_repaired", "")
                original = meta_data.get("local_path_repaired", "")
                # Check if path was actually repaired (original had ./imgs/)
                if "./imgs_raw_data/" in local_repaired:
                    source_status = SourceStatus.PATH_REPAIRED
                    repaired += 1
                else:
                    source_status = SourceStatus.METADATA_MATCHED
                matched += 1
            else:
                source_status = SourceStatus.METADATA_MISSING
                orphan += 1

            # Validate image file
            validation = validate_image_file(abs_path)

            # Compute hash
            try:
                with open(abs_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            except (OSError, IOError):
                file_hash = ""
                invalid += 1

            # Determine format from extension
            ext = Path(abs_path).suffix.lower().lstrip(".")
            if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
                ext = "jpg"

            item = ImageManifestItem(
                file_path=fs_data["file_path"],
                relative_path=fs_data["relative_path"],
                folder_keyword=fs_data["folder_keyword"],
                folder_index=fs_data["folder_index"],
                metadata_index=meta_data.get("metadata_index"),
                metadata_title=meta_data.get("metadata_title", ""),
                metadata_keyword=meta_data.get("metadata_keyword", ""),
                metadata_keyword_labels=meta_data.get("metadata_keyword_labels", []),
                image_url=meta_data.get("image_url", ""),
                file_size=fs_data["file_size"],
                width=fs_data["width"] if fs_data["width"] > 0 else validation.get("width", 0),
                height=fs_data["height"] if fs_data["height"] > 0 else validation.get("height", 0),
                format=ext,
                source_status=source_status,
                hash=file_hash,
                run_id=f"phase_3_1_{int(self.start_time)}",
            )

            items.append(item)

        logger.info(
            f"Index built: {len(items)} images "
            f"(matched={matched}, orphan={orphan}, "
            f"repaired={repaired}, invalid={invalid})"
        )
        return items

    def write_index(self, items: list[ImageManifestItem]) -> Path:
        """Write image index to JSONL file."""
        output_path = self.output_dir / "image_index.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Written {len(items)} items to {output_path}")
        return output_path

    def write_report(
        self,
        items: list[ImageManifestItem],
        fs_images: dict,
        metadata: dict,
    ) -> Path:
        """Write Phase 3.1 index report."""
        # Statistics
        matched = sum(1 for i in items if i.source_status == SourceStatus.METADATA_MATCHED)
        repaired = sum(1 for i in items if i.source_status == SourceStatus.PATH_REPAIRED)
        orphan = sum(1 for i in items if i.source_status == SourceStatus.METADATA_MISSING)

        # Size distribution
        sizes = [i.file_size for i in items]
        size_stats = {
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "mean": sum(sizes) / len(sizes) if sizes else 0,
            "median": sorted(sizes)[len(sizes)//2] if sizes else 0,
        }

        # Dimension stats
        widths = [i.width for i in items if i.width > 0]
        heights = [i.height for i in items if i.height > 0]
        dim_stats = {
            "width_mean": sum(widths) / len(widths) if widths else 0,
            "height_mean": sum(heights) / len(heights) if heights else 0,
            "width_max": max(widths) if widths else 0,
            "height_max": max(heights) if heights else 0,
        }

        # Folder keyword distribution
        keyword_counts = {}
        for item in items:
            kw = item.folder_keyword
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        # Top 20 keywords
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        # Language distribution from metadata
        lang_zh = sum(1 for i in items if any("lang:zh" in l for l in i.metadata_keyword_labels))
        lang_en = sum(1 for i in items if any("lang:en" in l for l in i.metadata_keyword_labels))
        lang_unknown = len(items) - lang_zh - lang_en

        report = {
            "phase": "3.1",
            "run_id": f"phase_3_1_{int(self.start_time)}",
            "timestamp": time.time(),
            "total_images_scanned": len(fs_images),
            "total_metadata_records": len(metadata),
            "total_images_indexed": len(items),
            "metadata_matched": matched,
            "metadata_missing": orphan,
            "path_repaired": repaired,
            "size_stats_bytes": size_stats,
            "dimension_stats": dim_stats,
            "language_distribution": {
                "zh": lang_zh,
                "en": lang_en,
                "unknown": lang_unknown,
            },
            "top_20_keywords": [
                {"keyword": k, "count": c} for k, c in top_keywords
            ],
            "total_keyword_folders": len(keyword_counts),
            "errors": [],
        }

        report_path = self.report_dir / "phase_3_1_index_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Written report to {report_path}")
        return report_path

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline."""
        logger.info("=== Phase 3.1: Image Indexing ===")

        # Step 1: Scan filesystem
        fs_images = self.scan_filesystem()

        # Step 2: Load metadata
        metadata = self.load_metadata()

        # Step 3: Build merged index
        items = self.build_index(fs_images, metadata)

        # Step 4: Write index JSONL
        index_path = self.write_index(items)

        # Step 5: Write report
        report_path = self.write_report(items, fs_images, metadata)

        elapsed = time.time() - self.start_time
        logger.info(f"=== Phase 3.1 complete in {elapsed:.1f}s ===")

        return {
            "index_path": str(index_path),
            "report_path": str(report_path),
            "total_indexed": len(items),
            "elapsed_seconds": elapsed,
        }