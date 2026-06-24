"""Run Phase 3.4: Multimodal benchmark candidate generation.

Usage:
    python scripts/run_phase_3_4_benchmark_generation.py --num_workers 10
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.image_benchmark_generator import BenchmarkCandidateGenerator
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_3_4")


def main():
    parser = argparse.ArgumentParser(description="Phase 3.4: Benchmark candidate generation")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--min_relevance", type=float, default=0.6)
    args = parser.parse_args()

    logger.info("Starting Phase 3.4: Benchmark Candidate Generation")
    generator = BenchmarkCandidateGenerator(
        num_workers=args.num_workers,
        min_domain_relevance=args.min_relevance,
    )
    result = generator.run()

    logger.info(f"Result: {result}")
    print(f"\n=== Phase 3.4 Complete ===")
    print(f"Total candidates: {result['total_candidates']}")
    print(f"Images processed: {result['total_images_processed']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()