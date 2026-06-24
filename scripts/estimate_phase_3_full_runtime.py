"""Estimate Phase 3.9 full runtime based on pilot throughput data.

Uses actual pilot measurements to extrapolate for different worker counts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_full_image_labeling"

# Pilot data (from Phase 3 report)
PILOT_IMAGES_COMPLETED = 282
PILOT_RUNTIME_MINUTES = 142  # approximate
PILOT_WORKERS = 2
PILOT_LLM_CALLS = 1208  # approximate (labeling ~282 + candidates ~282 + validation ~644)
FULL_UNIQUE_IMAGES = 11624


def estimate_runtime(
    full_images: int,
    pilot_images: int,
    pilot_runtime_seconds: float,
    pilot_workers: int,
    target_workers: int,
    overhead_factor: float = 1.15,
) -> dict:
    """Estimate runtime for a given worker count.

    Formula: (full/pilot) * pilot_time * (pilot_workers/target_workers) * overhead
    """
    estimated_seconds = (
        (full_images / pilot_images)
        * pilot_runtime_seconds
        * (pilot_workers / target_workers)
        * overhead_factor
    )
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60

    return {
        "target_workers": target_workers,
        "overhead_factor": overhead_factor,
        "estimated_seconds": round(estimated_seconds, 1),
        "estimated_minutes": round(estimated_minutes, 1),
        "estimated_hours": round(estimated_hours, 2),
    }


def generate_estimates():
    """Generate runtime estimates for multiple worker counts."""
    pilot_runtime_seconds = PILOT_RUNTIME_MINUTES * 60

    worker_counts = [4, 8, 12, 16]
    overhead_scenarios = {
        "optimistic": 1.15,
        "conservative": 1.50,
        "pessimistic": 2.00,
    }

    estimates = {}
    for scenario, overhead in overhead_scenarios.items():
        estimates[scenario] = []
        for workers in worker_counts:
            est = estimate_runtime(
                full_images=FULL_UNIQUE_IMAGES,
                pilot_images=PILOT_IMAGES_COMPLETED,
                pilot_runtime_seconds=pilot_runtime_seconds,
                pilot_workers=PILOT_WORKERS,
                target_workers=workers,
                overhead_factor=overhead,
            )
            estimates[scenario].append(est)

    report = {
        "phase": "3.9_runtime_estimation",
        "timestamp": time.time(),
        "pilot_data": {
            "images_completed": PILOT_IMAGES_COMPLETED,
            "runtime_minutes": PILOT_RUNTIME_MINUTES,
            "workers": PILOT_WORKERS,
            "llm_calls_approx": PILOT_LLM_CALLS,
        },
        "full_target": {
            "unique_images": FULL_UNIQUE_IMAGES,
            "approx_llm_calls_labeling": FULL_UNIQUE_IMAGES,
            "approx_llm_calls_candidates": "variable (depends on quality filter)",
            "approx_llm_calls_validation": "variable (depends on candidates generated)",
            "total_llm_calls_estimate": "~30000-50000",
        },
        "estimates": estimates,
        "warning": "Estimated runtime is LONG. Even at 16 workers with optimistic overhead, "
                   "Stage 1 alone is estimated to take many hours. Plan accordingly.",
    }

    # Write JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "runtime_estimate_full.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write MD
    md_path = REPORT_DIR / "runtime_estimate_full.md"
    with open(md_path, "w") as f:
        f.write("# Phase 3.9 Runtime Estimation\n\n")
        f.write(f"## Pilot Data\n")
        f.write(f"- Images completed: {PILOT_IMAGES_COMPLETED}\n")
        f.write(f"- Runtime: {PILOT_RUNTIME_MINUTES} minutes\n")
        f.write(f"- Workers: {PILOT_WORKERS}\n")
        f.write(f"- LLM calls: ~{PILOT_LLM_CALLS}\n\n")
        f.write(f"## Full Target\n")
        f.write(f"- Unique images: {FULL_UNIQUE_IMAGES}\n")
        f.write(f"- Estimated total LLM calls: ~30,000-50,000\n\n")
        f.write(f"## Estimated Runtime (Stage 1: Labeling Only)\n\n")
        f.write(f"| Workers | Optimistic | Conservative | Pessimistic |\n")
        f.write(f"|---------|------------|--------------|-------------|\n")
        for workers in worker_counts:
            opt = estimates["optimistic"][worker_counts.index(workers)]
            con = estimates["conservative"][worker_counts.index(workers)]
            pes = estimates["pessimistic"][worker_counts.index(workers)]
            f.write(f"| {workers} | {opt['estimated_hours']:.1f}h | {con['estimated_hours']:.1f}h | {pes['estimated_hours']:.1f}h |\n")
        f.write(f"\n**WARNING**: Even the optimistic estimate is many hours. ")
        f.write(f"Stage 2 (candidates) and Stage 3 (validation) add additional hours.\n")
        f.write(f"Plan for the full run to take potentially 10-20+ hours depending on worker count and API stability.\n\n")
        f.write(f"## Recommended Strategy\n")
        f.write(f"- Start with 8 workers\n")
        f.write(f"- Monitor error rate every 5 minutes\n")
        f.write(f"- Scale up to 12-16 if stable\n")
        f.write(f"- Use checkpoint/resume to handle interruptions\n")
        f.write(f"- Run in background: `nohup python scripts/run_phase_3_full_image_labeling.py &`\n")
        f.write(f"- Monitor: `tail -f data/reports/phase_3_full_image_labeling/progress_full.log`\n")

    print(f"\n=== Runtime Estimation ===")
    print(f"Full target: {FULL_UNIQUE_IMAGES} images")
    for scenario, ests in estimates.items():
        print(f"\n{scenario.capitalize()} scenario:")
        for est in ests:
            print(f"  {est['target_workers']} workers: {est['estimated_hours']:.1f} hours")
    print(f"\nReports written to {REPORT_DIR}")

    return report


if __name__ == "__main__":
    generate_estimates()