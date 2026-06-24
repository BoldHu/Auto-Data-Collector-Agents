"""Run Phase 3.9 full-scale image labeling pipeline.

Orchestration:
1. Check stale processes
2. Generate manifest (if needed)
3. Preflight validation
4. Sanity check (50 images)
5. Full 3-stage pipeline (labeling -> candidates -> validation)

Usage:
    python scripts/run_phase_3_full_image_labeling.py \
        --initial_workers 8 \
        --max_workers 16 \
        --start_stage 1 \
        --skip_preflight \
        --skip_sanity_check

For background run:
    nohup python scripts/run_phase_3_full_image_labeling.py \
        --initial_workers 8 --max_workers 16 > /dev/null 2>&1 &

Monitor:
    tail -f data/reports/phase_3_full_image_labeling/progress_full.log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.full_image_labeling_pipeline import FullImageLabelingPipeline
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_3_full")


def check_stale_processes() -> list[str]:
    """Check for stale Phase 3 processes."""
    result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
    stale = [line for line in result.stdout.splitlines()
             if any(kw in line for kw in ["run_phase_3", "image_labeling", "benchmark_generation"])
             and "grep" not in line and "run_phase_3_full" not in line]
    return stale


def run_preflight() -> bool:
    """Run preflight checks."""
    from scripts.preflight_phase_3_full import run_preflight
    result = run_preflight()
    return result.get("all_checks_pass", False)


def run_sanity_check(pipeline: FullImageLabelingPipeline) -> bool:
    """Run 50-image sanity check."""
    logger.info("=== Running 50-image sanity check ===")

    result = pipeline.run_stage_1(max_images=50)

    completed = result.get("total_processed", 0)
    failed = result.get("total_failed", 0)

    if completed < 45:  # Allow up to 5 failures out of 50
        logger.error(f"Sanity check FAILED: only {completed} out of 50 completed")
        return False

    # Verify output files
    labels_path = pipeline.labels_file
    quality_path = pipeline.quality_file

    if labels_path.exists():
        labels = []
        with open(labels_path) as f:
            for line in f:
                labels.append(json.loads(line))

        # Check JSON validity
        if len(labels) < 45:
            logger.error(f"Sanity check FAILED: only {len(labels)} label records")
            return False

        # Check domain relevance average
        avg_relevance = sum(l.get("domain_relevance", 0) for l in labels) / len(labels)
        if avg_relevance < 0.5:
            logger.error(f"Sanity check FAILED: avg domain relevance {avg_relevance:.2f} < 0.5")
            return False

        logger.info(f"Sanity check PASSED: {completed} completed, avg relevance {avg_relevance:.2f}")
        return True

    logger.error("Sanity check FAILED: labels file not found")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3.9 full image labeling")
    parser.add_argument("--initial_workers", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--start_stage", type=int, default=1, help="Start from stage 1, 2, or 3")
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--skip_sanity_check", action="store_true")
    parser.add_argument("--run_id", type=str, default="phase_3_full_image_labeling")
    args = parser.parse_args()

    # Step 1: Check stale processes
    stale = check_stale_processes()
    if stale:
        logger.warning(f"Found {len(stale)} stale processes. Please stop them before proceeding.")
        for line in stale:
            logger.warning(f"  {line.strip()}")

    # Step 2: Generate manifest if needed
    manifest_path = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_unique_manifest.jsonl"
    if not manifest_path.exists():
        logger.info("Manifest not found, generating...")
        from scripts.generate_unique_manifest import generate_manifest
        generate_manifest()

    # Step 3: Preflight
    if not args.skip_preflight:
        logger.info("Running preflight checks...")
        passed = run_preflight()
        if not passed:
            logger.error("Preflight FAILED. Cannot start full labeling.")
            sys.exit(1)

    # Step 4: Create pipeline
    pipeline = FullImageLabelingPipeline(
        initial_workers=args.initial_workers,
        max_workers=args.max_workers,
        run_id=args.run_id,
    )

    # Step 5: Sanity check (only if starting from stage 1)
    if args.start_stage == 1 and not args.skip_sanity_check:
        passed = run_sanity_check(pipeline)
        if not passed:
            logger.error("Sanity check FAILED. Do not proceed with full labeling.")
            sys.exit(1)
        logger.info("Sanity check PASSED. Proceeding with full labeling.")

    # Step 6: Run full pipeline
    logger.info(f"Starting full pipeline: initial_workers={args.initial_workers}, "
                f"max_workers={args.max_workers}, start_stage={args.start_stage}")

    # Reset pool stats for fresh measurement
    pipeline.pool.reset_stats()

    results = pipeline.run(start_stage=args.start_stage)

    # Step 7: Save results
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_3_full_image_labeling"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "run_metadata_full.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Full pipeline run complete!")
    logger.info(f"Results: {json.dumps(results, indent=2, default=str)[:200]}")

    return results


if __name__ == "__main__":
    main()