"""Run Phase 6.55 pipeline implementation audit.

Usage:
    python scripts/run_phase_6_55_pipeline_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.autodata.audit.pipeline_auditor import run_full_audit

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_55_pipeline_audit"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 6.55 Pipeline Implementation Audit ===\n")

    # Run full audit
    report = run_full_audit()

    # Save main report
    with open(report_dir / "preflight_phase_6_55.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save individual sections
    for section_name, section_data in report["sections"].items():
        with open(report_dir / f"{section_name}_audit.json", "w") as f:
            json.dump(section_data, f, indent=2, ensure_ascii=False)

    # Save readiness score
    with open(report_dir / "readiness_score.json", "w") as f:
        json.dump(report["readiness_score"], f, indent=2)

    # Print summary
    print(f"\n=== Audit Complete ===")
    print(f"Sections audited: {len(report['sections'])}")

    score = report["readiness_score"]
    print(f"\nReadiness Score: {score['total_score']}/{score['max_score']} ({score['classification']})")
    for dim, val in score["scores"].items():
        print(f"  {dim}: {val}")

    # Agent summary
    agent_summary = report["sections"].get("agent_usage", {}).get("summary", {})
    print(f"\nAgent Implementation:")
    for k, v in agent_summary.items():
        print(f"  {k}: {v}")

    # DTCG summary
    dtcg_conclusions = report["sections"].get("dtcg", {}).get("conclusions", {})
    print(f"\nDTCG:")
    for k, v in dtcg_conclusions.items():
        print(f"  {k}: {v}")

    print(f"\nReports saved to: {report_dir}")


if __name__ == "__main__":
    main()
