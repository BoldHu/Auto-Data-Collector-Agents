"""Run Phase 4 exam question extraction pipeline.

Usage:
    python scripts/run_phase_4_exam_extraction.py \
        --run_id phase_4_exam_extraction \
        --max_workers 16 \
        --extraction_workers 8 \
        --quality_workers 8 \
        --start_stage 1 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_4")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 exam extraction")
    parser.add_argument("--run_id", type=str, default="phase_4_exam_extraction")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--extraction_workers", type=int, default=8)
    parser.add_argument("--quality_workers", type=int, default=8)
    parser.add_argument("--start_stage", type=int, default=1)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    exam_dir = PROJECT_ROOT / "exam_raw_data"
    output_dir = PROJECT_ROOT / "data" / "processed" / "exam_questions"
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_4_exam_extraction"

    # Verify exam directory exists
    if not exam_dir.exists():
        logger.error(f"Exam directory not found: {exam_dir}")
        sys.exit(1)

    # Count exam files
    exam_files = [f for f in exam_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    logger.info(f"Found {len(exam_files)} exam files in {exam_dir}")

    from src.autodata.pipelines.exam_question_extraction_pipeline import ExamQuestionExtractionPipeline

    pipeline = ExamQuestionExtractionPipeline(
        exam_dir=exam_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        run_id=args.run_id,
        max_workers=args.max_workers,
        extraction_workers=args.extraction_workers,
        quality_workers=args.quality_workers,
        use_key2=False,  # API_KEY1 only
    )

    logger.info(f"Starting pipeline: run_id={args.run_id}, start_stage={args.start_stage}")
    logger.info(f"Workers: max={args.max_workers}, extraction={args.extraction_workers}, quality={args.quality_workers}")
    logger.info(f"API policy: use_key1_only (use_key2=False)")

    results = pipeline.run(
        start_stage=args.start_stage,
        max_files=args.max_files,
    )

    # Print results summary
    print(f"\n=== Phase 4 Exam Extraction Complete ===")
    print(f"Run ID: {results.get('run_id')}")
    print(f"Total elapsed: {results.get('total_elapsed_formatted', 'N/A')}")

    for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
        if stage_key in results:
            stage = results[stage_key]
            print(f"\n{stage_key}:")
            for k, v in stage.items():
                if k != "controller_stats":
                    print(f"  {k}: {v}")

    # Print final output counts
    output_dir = PROJECT_ROOT / "data" / "processed" / "exam_questions"
    for name in ["exam_questions_raw", "exam_questions_validated", "exam_question_quality_scores",
                  "exam_questions_unique", "exam_questions_benchmark_ready_candidates"]:
        path = output_dir / f"{name}.jsonl"
        if path.exists():
            count = sum(1 for _ in open(path))
            print(f"  {name}.jsonl: {count} records")


if __name__ == "__main__":
    main()
