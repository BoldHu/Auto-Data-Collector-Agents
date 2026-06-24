"""Run Phase 5 source pool audit.

Usage:
    python scripts/run_phase_5_source_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.benchmark_source_auditor import audit_source_pools, save_audit_report


def main():
    print("Running source pool audit...")
    report = audit_source_pools()
    json_path, md_path = save_audit_report(report)

    print(f"\n=== Source Pool Audit Complete ===")
    print(f"Exam ready: {report['summary']['total_exam_ready']}")
    print(f"MM passed: {report['summary']['total_mm_passed']}")
    print(f"Knowledge units: {report['summary']['total_ku']}")
    print(f"SFT candidates: {report['summary']['total_sft']}")
    print(f"Total pool: {report['summary']['total_benchmark_pool']}")
    print(f"\nFindings:")
    for f in report["findings"]:
        print(f"  - {f}")
    print(f"\nJSON: {json_path}")
    print(f"MD: {md_path}")


if __name__ == "__main__":
    main()
