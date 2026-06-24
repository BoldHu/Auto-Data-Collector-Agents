"""Phase 7: Run preflight checks.

Usage:
    python scripts/run_phase_7_preflight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.autodata.finetuning.phase7_preflight import run_preflight, save_preflight_report

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_finetuning_preparation"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Running Phase 7 preflight checks...")
    checks = run_preflight(PROJECT_ROOT)

    save_preflight_report(checks, report_dir)

    print(f"Status: {checks['status']}")
    for k, v in checks.items():
        if k not in ("status", "all_critical_passed") and isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    if checks["status"] == "FAIL":
        print("\nSome critical checks failed. See report for details.")
        sys.exit(1)
    else:
        print("\nAll critical checks passed.")


if __name__ == "__main__":
    main()
