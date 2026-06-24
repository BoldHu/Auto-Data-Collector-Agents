"""Run Phase 3.3: Image labeling, captioning, and quality assessment (pilot 300).

Usage:
    python scripts/run_phase_3_3_image_labeling.py --max_images 300 --num_workers 10
    python scripts/run_phase_3_3_image_labeling.py --max_images 300 --num_workers 10 --no_stratified
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.image_labeling_pipeline import ImageLabelingPipeline
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_3_3")


def main():
    parser = argparse.ArgumentParser(description="Phase 3.3: Image labeling pilot")
    parser.add_argument("--max_images", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--no_stratified", action="store_true", default=False)
    args = parser.parse_args()

    logger.info(f"Starting Phase 3.3: Image Labeling (pilot {args.max_images})")
    pipeline = ImageLabelingPipeline(
        max_images=args.max_images,
        num_workers=args.num_workers,
        stratified=not args.no_stratified,
    )
    result = pipeline.run()

    logger.info(f"Result: {json.dumps({k: v for k, v in result.items() if k != 'pool_stats'}, indent=2)}")
    print(f"\n=== Phase 3.3 Pilot Complete ===")
    print(f"Total processed: {result['total_processed']}")
    print(f"Total failed: {result['total_failed']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
    print(f"Labels file: {result['labels_file']}")
    print(f"Captions file: {result['captions_file']}")
    print(f"Quality file: {result['quality_file']}")


if __name__ == "__main__":
    main()