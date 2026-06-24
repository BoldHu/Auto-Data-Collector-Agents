"""Run Phase 3.5: Independent critic validation of benchmark candidates.

Usage:
    python scripts/run_phase_3_5_quality_verification.py --num_workers 10
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.image_quality_verifier import ImageQualityVerifier
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_3_5")


def main():
    parser = argparse.ArgumentParser(description="Phase 3.5: Quality verification")
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    logger.info("Starting Phase 3.5: Quality Verification")
    verifier = ImageQualityVerifier(num_workers=args.num_workers)
    result = verifier.run()

    logger.info(f"Result: {result}")
    print(f"\n=== Phase 3.5 Complete ===")
    print(f"Total validated: {result['total_validated']}")
    print(f"Total passed: {result['total_passed']}")
    print(f"Total failed: {result['total_failed']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()