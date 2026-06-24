"""Generate unique image manifest from deduplicated data.

Reads image_dedup.jsonl (15,859 records), splits into:
- image_unique_manifest.jsonl: 11,624 records with dedup_status=="unique"
- duplicate_mapping.jsonl: 4,235 records with dedup_status=="duplicate"

The unique manifest is the canonical input for the full labeling pipeline.
The duplicate mapping preserves duplicate -> primary references for later use.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.utils.logging_utils import get_logger

logger = get_logger("generate_manifest")

DEDUP_PATH = PROJECT_ROOT / "data" / "interim" / "image_dedup" / "image_dedup.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_unique_manifest.jsonl"
DUPLICATE_MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "duplicate_mapping.jsonl"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_full_image_labeling"


def generate_manifest() -> dict:
    """Generate unique manifest and duplicate mapping from dedup data."""
    # Ensure output dirs
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading dedup data from {DEDUP_PATH}")

    unique_count = 0
    duplicate_count = 0
    source_status_counts = {}
    category_counts = {}

    with open(DEDUP_PATH, encoding="utf-8") as fin, \
         open(MANIFEST_PATH, "w", encoding="utf-8") as fout_unique, \
         open(DUPLICATE_MAPPING_PATH, "w", encoding="utf-8") as fout_dup:

        for line in fin:
            record = json.loads(line)
            status = record.get("dedup_status", "unknown")
            source_status = record.get("source_status", "unknown")
            source_status_counts[source_status] = source_status_counts.get(source_status, 0) + 1

            if status == "unique":
                fout_unique.write(line)
                unique_count += 1
            elif status in ("duplicate", "near_duplicate"):
                fout_dup.write(line)
                duplicate_count += 1
            else:
                # Unknown status: treat as unique
                fout_unique.write(line)
                unique_count += 1

    logger.info(f"Unique images: {unique_count}")
    logger.info(f"Duplicate images: {duplicate_count}")
    logger.info(f"Source status distribution: {source_status_counts}")

    # Verify: count source_status in unique manifest
    metadata_missing = source_status_counts.get("metadata_missing", 0)
    path_repaired = source_status_counts.get("path_repaired", 0)
    metadata_matched = source_status_counts.get("metadata_matched", 0)

    # Write generation report
    report = {
        "phase": "3.9_manifest_generation",
        "timestamp": time.time(),
        "input_dedup_path": str(DEDUP_PATH),
        "output_manifest_path": str(MANIFEST_PATH),
        "output_duplicate_mapping_path": str(DUPLICATE_MAPPING_PATH),
        "total_dedup_records": unique_count + duplicate_count,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "source_status_distribution": source_status_counts,
        "metadata_missing_in_unique": metadata_missing,
        "path_repaired_in_unique": path_repaired,
        "metadata_matched_in_unique": metadata_matched,
    }

    report_path = REPORT_DIR / "manifest_generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report written to {report_path}")

    return report


if __name__ == "__main__":
    result = generate_manifest()
    print(f"\n=== Manifest Generation Complete ===")
    print(f"Unique images: {result['unique_count']}")
    print(f"Duplicate images: {result['duplicate_count']}")
    print(f"Metadata-missing unique: {result['metadata_missing_in_unique']}")
    print(f"Path-repaired unique: {result['path_repaired_in_unique']}")
    print(f"Manifest: {result['output_manifest_path']}")
    print(f"Duplicate mapping: {result['output_duplicate_mapping_path']}")