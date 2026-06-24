"""Run Phase 5.5 detailed benchmark audit.

Usage:
    python scripts/run_phase_5_5_benchmark_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.benchmark.benchmark_auditor import audit_benchmark_detailed, save_audit_report


def main():
    print("Running detailed benchmark audit...")
    report = audit_benchmark_detailed()
    json_path, md_path = save_audit_report(report)

    print(f"\n=== Detailed Audit Complete ===")
    for dim, data in report["dimensions"].items():
        if isinstance(data, dict) and "status" in data:
            print(f"  {dim}: {data['status']}")
        elif isinstance(data, dict):
            print(f"  {dim}: {list(data.keys())[:3]}...")
        else:
            print(f"  {dim}: {data}")
    print(f"\nJSON: {json_path}")
    print(f"MD: {md_path}")


if __name__ == "__main__":
    main()
