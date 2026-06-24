#!/usr/bin/env python3
"""Phase 2 text cleaning pipeline runner.

Usage:
  python scripts/run_phase_2_text_cleaning.py --pilot          # Small pilot (2 zh + 2 en, 20 pages)
  python scripts/run_phase_2_text_cleaning.py --pilot --skip_llm  # Dry-run pilot
  python scripts/run_phase_2_text_cleaning.py --full           # Full-scale (not recommended yet)
  python scripts/run_phase_2_text_cleaning.py --resume         # Resume from checkpoint
  python scripts/run_phase_2_text_cleaning.py --language zh    # Only Chinese
  python scripts/run_phase_2_text_cleaning.py --language en    # Only English

Default behavior: pilot mode with LLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.text_cleaning_pipeline import TextCleaningPipeline
from src.autodata.utils.logging_utils import setup_logging, get_logger
from src.autodata.utils.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Text cleaning pipeline")
    parser.add_argument("--pilot", action="store_true", default=True, help="Pilot mode (default)")
    parser.add_argument("--full", action="store_true", help="Full-scale mode (use with caution)")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process")
    parser.add_argument("--max_pages_per_file", type=int, default=None, help="Max pages per file")
    parser.add_argument("--language", choices=["zh", "en", "all"], default="all", help="Language filter")
    parser.add_argument("--skip_llm", action="store_true", help="Dry-run (no LLM calls)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--run_id", type=str, default=None, help="Custom run ID")
    parser.add_argument("--enable_dtcg_trace", action="store_true", default=True, help="Enable DTCG trace recording")
    parser.add_argument("--enable_quality_persistence", action="store_true", default=True, help="Enable quality-score persistence")
    parser.add_argument("--enable_context_package_dump", action="store_true", default=True, help="Enable context package dump")
    parser.add_argument("--enable_progress_monitor", action="store_true", default=False, help="Enable real-time progress monitoring")
    parser.add_argument("--use_key2", action="store_true", default=False, help="Use API_KEY2 (second Xiaomi key, unlimited quota)")
    parser.add_argument("--model", type=str, default=None, help="Model name override (e.g. mimo-v2.5, mimo-v2-pro)")
    parser.add_argument("--skip_file_indices", type=str, default=None, help="Comma-separated file indices to skip (for parallel processing)")
    args = parser.parse_args()

    # Setup logging
    log_dir = str(PROJECT_ROOT / "data" / "reports" / "phase_2_text_cleaning")
    ensure_dir(log_dir)
    setup_logging(level="INFO", log_dir=log_dir, log_file="phase_2_pipeline.log")
    logger = get_logger("phase_2_runner")

    mode = "full" if args.full else "pilot"
    if args.full:
        logger.warning("Running in FULL mode — this processes all 64 books. Ensure this is intended.")

    logger.info(f"Phase 2 pipeline: mode={mode}, language={args.language}, skip_llm={args.skip_llm}")

    # Parse skip_file_indices
    skip_indices = None
    if args.skip_file_indices:
        skip_indices = [int(i) for i in args.skip_file_indices.split(",")]

    # Create and run pipeline
    pipeline = TextCleaningPipeline(
        mode=mode,
        max_files=args.max_files,
        max_pages_per_file=args.max_pages_per_file,
        language_filter=args.language,
        skip_llm=args.skip_llm,
        resume=args.resume,
        run_id=args.run_id,
        enable_dtcg_trace=args.enable_dtcg_trace,
        enable_quality_persistence=args.enable_quality_persistence,
        enable_context_package_dump=args.enable_context_package_dump,
        enable_progress_monitor=args.enable_progress_monitor,
        use_key2=args.use_key2,
        model_name=args.model,
        skip_file_indices=skip_indices,
    )

    start_time = time.time()
    metadata = pipeline.run()
    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("Phase 2 — Text Cleaning Pipeline Results")
    print("=" * 60)
    print(f"  Run ID:       {metadata.run_id}")
    print(f"  Mode:         {metadata.mode}")
    print(f"  Model:        {metadata.model}")
    print(f"  Language:     {metadata.language_filter}")
    print(f"  Files:        {metadata.total_files_processed}")
    print(f"  Pages:        {metadata.total_pages_processed}")
    print(f"  Chunks:       {metadata.total_chunks_created}")
    print(f"  Passed:       {metadata.total_chunks_passed}")
    print(f"  Needs rev:    {metadata.total_chunks_needs_revision}")
    print(f"  Failed:       {metadata.total_chunks_failed}")
    print(f"  Knowledge:    {metadata.total_knowledge_units}")
    print(f"  SFT cand:     {metadata.total_sft_candidates}")
    print(f"  Tokens:       {metadata.total_tokens_used}")
    print(f"  API calls:    {metadata.total_api_calls}")
    print(f"  Total time:   {total_time:.1f}s")
    print("=" * 60)

    # Save metadata
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_text_cleaning"
    with open(report_dir / "phase_2_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved to: {report_dir / 'phase_2_run_metadata.json'}")


if __name__ == "__main__":
    main()