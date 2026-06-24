"""Run Phase 5 multimodal candidate repair.

Usage:
    python scripts/run_phase_5_mm_candidate_repair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.multimodal_candidate_repair import repair_multimodal_candidates, save_repair_report


def main():
    print("Repairing multimodal candidates...")
    report = repair_multimodal_candidates()
    json_path = save_repair_report(report)

    print(f"\n=== Multimodal Candidate Repair Complete ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nReport: {json_path}")


if __name__ == "__main__":
    main()
