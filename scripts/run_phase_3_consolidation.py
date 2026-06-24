"""Run Phase 3 output consistency consolidation.

Usage:
    python scripts/run_phase_3_consolidation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.phase3_consolidation import run_consolidation, save_consolidation_report
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("phase3_consolidation")


def main():
    logger.info("Starting Phase 3 output consistency consolidation...")
    report = run_consolidation()
    json_path, md_path = save_consolidation_report(report)

    print(f"\n=== Phase 3 Consolidation Complete ===")
    print(f"Findings: {len(report['findings'])}")
    for f in report["findings"]:
        print(f"  {f['id']}: {f['description']}")
    print(f"Fixes: {len(report['fixes'])}")
    for fix in report["fixes"]:
        print(f"  {fix['id']}: {fix['description']}")
    print(f"\nReport: {json_path}")
    print(f"Report MD: {md_path}")

    # Print key statistics
    stats = report["statistics"]
    print(f"\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
