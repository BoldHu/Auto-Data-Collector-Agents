"""Validate Phase 6.8 outputs are complete.

Usage:
    python scripts/validate_phase_6_8_evidence_consolidation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_8_evidence_consolidation"
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_8"

    checks = []
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        checks.append(f"[{status}] {name}" + (f" - {detail}" if detail else ""))

    # 1. Long-context expanded subset exists with traces
    lc_traces_path = eval_dir / "long_context_expanded" / "system_ablation_traces.jsonl"
    lc_traces = load_jsonl(lc_traces_path)
    check("Long-context expanded traces exist", len(lc_traces) > 0, f"{len(lc_traces)} traces")

    # 2. Stress subset exists with traces
    stress_traces_path = eval_dir / "stress" / "system_ablation_traces.jsonl"
    stress_traces = load_jsonl(stress_traces_path)
    check("Stress traces exist", len(stress_traces) > 0, f"{len(stress_traces)} traces")

    # 3. All 6 systems represented in each subset
    def get_systems(traces):
        return set(t.get("system_type", "unknown") for t in traces)

    expected_systems = {"direct_llm", "single_react", "plan_execute", "broadcast", "static_router", "dtcg"}
    lc_systems = get_systems(lc_traces)
    stress_systems = get_systems(stress_traces)
    check("Long-context has all 6 systems", expected_systems.issubset(lc_systems), f"found: {lc_systems}")
    check("Stress has all 6 systems", expected_systems.issubset(stress_systems), f"found: {stress_systems}")

    # 4. Combined results exist
    combined_path = report_dir / "phase_6_8_combined_results.json"
    check("Combined results file exists", combined_path.exists())

    # 5. Statistical significance analysis exists
    sig_path = report_dir / "statistical_significance.json"
    check("Statistical significance file exists", sig_path.exists())

    # 6. DTCG component ablation scores exist
    dtcg_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "dtcg_component_ablation_scores.csv"
    check("DTCG component ablation scores exist", dtcg_path.exists())

    # 7. Report exists
    report_path = report_dir / "PHASE_6_8_REPORT.md"
    check("Phase 6.8 report exists", report_path.exists())

    # 8. Paper tables exist
    paper_path = PROJECT_ROOT / "reports" / "paper_ready" / "phase_6_8_tables.tex"
    check("LaTeX tables exist", paper_path.exists())

    # 9. Case studies exist
    cases_path = report_dir / "case_studies.json"
    check("Case studies exist", cases_path.exists())

    # 10. Check DTCG component ablation has nonzero dtcg_full
    if dtcg_path.exists():
        with open(dtcg_path) as f:
            lines = f.readlines()
            dtcg_full_line = [l for l in lines if l.startswith("dtcg_full,")]
            if dtcg_full_line:
                parts = dtcg_full_line[0].strip().split(",")
                total = int(parts[1])
                check("dtcg_full has nonzero records", total > 0, f"total={total}")
            else:
                check("dtcg_full has nonzero records", False, "not found in CSV")

    # Print results
    print("=== Phase 6.8 Validation ===\n")
    for check_line in checks:
        print(check_line)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    if failed == 0:
        print("\nAll checks passed! Phase 6.8 is complete.")
    else:
        print(f"\n{failed} check(s) failed. Please review.")


if __name__ == "__main__":
    main()
