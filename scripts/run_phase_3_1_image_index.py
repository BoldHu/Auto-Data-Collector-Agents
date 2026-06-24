"""Run Phase 3.1: Image indexing and metadata repair.

Usage:
    python scripts/run_phase_3_1_image_index.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.image_indexer import ImageIndexer
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_3_1")


def main():
    logger.info("Starting Phase 3.1: Image Indexing")
    indexer = ImageIndexer()
    result = indexer.run()

    logger.info(f"Result: {result}")
    print(f"\n=== Phase 3.1 Complete ===")
    print(f"Total indexed: {result['total_indexed']}")
    print(f"Index file: {result['index_path']}")
    print(f"Report file: {result['report_path']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()