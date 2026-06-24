"""Run Phase 5.5 agent task generation.

Generates benchmark items evaluating long-horizon domain agent capabilities.
Uses API_KEY1 only.

Usage:
    python scripts/run_phase_5_5_agent_task_generation.py \
        --target_candidates 500 \
        --target_passed 300 \
        --max_workers 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.benchmark.agent_task_generator import generate_agent_tasks, validate_agent_task
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("agent_task_gen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_candidates", type=int, default=500)
    parser.add_argument("--target_passed", type=int, default=300)
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    from src.autodata.utils.model_pool import get_model_pool
    pool = get_model_pool(use_key2=False)

    output_dir = PROJECT_ROOT / "data" / "benchmark_candidates" / "agent_task"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "agent_task_candidates.jsonl"
    validated_path = output_dir / "agent_task_candidates_validated.jsonl"

    all_candidates = []
    all_validated = []
    start_time = time.time()

    # Generation phase - generate in batches of 10
    logger.info("Starting agent task generation...")
    batches_needed = (args.target_candidates + 9) // 10

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for _ in range(batches_needed):
            future = executor.submit(generate_agent_tasks, pool, 10)
            futures[future] = True

        for future in as_completed(futures):
            try:
                items = future.result()
                for item in items:
                    item["benchmark_id"] = f"agent_{hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]}"
                    item["source_type"] = "agent_task"
                    item["modality"] = "text"
                    item["task_type"] = "agent_task"
                    all_candidates.append(item)
            except Exception:
                continue

            if len(all_candidates) >= args.target_candidates:
                break

    logger.info(f"Generated {len(all_candidates)} candidates")

    # Save candidates
    with open(candidates_path, "w") as f:
        for c in all_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Validation phase
    logger.info("Starting validation...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for item in all_candidates:
            future = executor.submit(validate_agent_task, pool, item)
            futures[future] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                score = future.result()
                if score.get("quality_status") == "keep":
                    item["quality_scores"] = score
                    item["validation_status"] = "passed"
                    all_validated.append(item)
            except Exception:
                continue

            if len(all_validated) >= args.target_passed:
                break

    logger.info(f"Validated {len(all_validated)} items")

    # Save validated
    with open(validated_path, "w") as f:
        for v in all_validated:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    report = {
        "total_candidates": len(all_candidates),
        "total_validated": len(all_validated),
        "elapsed_seconds": elapsed,
    }

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_5_benchmark_refinement"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "agent_task_generation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Agent Task Generation Complete ===")
    print(f"Candidates: {len(all_candidates)}")
    print(f"Validated: {len(all_validated)}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
