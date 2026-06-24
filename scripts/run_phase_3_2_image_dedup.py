"""Run Phase 3.2: Image deduplication using perceptual hashing.

Usage:
    python scripts/run_phase_3_2_image_dedup.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.image_deduplicator import ImageDeduplicator, OUTPUT_DIR
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_3_2")


def main():
    logger.info("Starting Phase 3.2: Image Deduplication")

    # Use precomputed phash values from the previous run (phash computation takes ~22 min)
    precomputed = OUTPUT_DIR / "image_dedup.jsonl"

    dedup = ImageDeduplicator(precomputed_dedup_path=precomputed)
    result = dedup.run()

    logger.info(f"Result: {result}")
    print(f"\n=== Phase 3.2 Complete ===")
    print(f"Total indexed: {result['total_indexed']}")
    print(f"Dedup file: {result['dedup_path']}")
    print(f"Report file: {result['report_path']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()