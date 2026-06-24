"""Main pipeline auditor for Phase 6.55.

Coordinates all sub-auditors and generates readiness score.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def run_full_audit() -> dict:
    """Run the full pipeline audit."""
    from src.autodata.audit.code_structure_auditor import audit_code_structure
    from src.autodata.audit.dtcg_auditor import audit_dtcg_implementation
    from src.autodata.audit.agent_usage_auditor import audit_agent_usage
    from src.autodata.audit.artifact_lineage_auditor import audit_artifact_lineage
    from src.autodata.audit.project_agent_verifier import verify_project_agents

    report = {
        "phase": "phase_6_55_pipeline_audit",
        "timestamp": time.time(),
        "sections": {},
    }

    # Run each auditor
    print("  Running code structure audit...")
    report["sections"]["code_structure"] = audit_code_structure()

    print("  Running DTCG audit...")
    report["sections"]["dtcg"] = audit_dtcg_implementation()

    print("  Running agent usage audit...")
    report["sections"]["agent_usage"] = audit_agent_usage()

    print("  Running artifact lineage audit...")
    report["sections"]["artifact_lineage"] = audit_artifact_lineage()

    print("  Running project agent verification...")
    report["sections"]["project_agent_verifier"] = verify_project_agents()

    # Compute readiness score
    report["readiness_score"] = compute_readiness_score(report)

    return report


def compute_readiness_score(report: dict) -> dict:
    """Compute readiness score for Phase 6.6 ablation."""
    scores = {}

    # 1. Agent implementation completeness (0-20)
    agent_summary = report["sections"].get("agent_usage", {}).get("summary", {})
    implemented = agent_summary.get("implemented_and_used", 0) + agent_summary.get("implemented_but_not_used", 0)
    total = agent_summary.get("total_agents", 1)
    scores["agent_implementation"] = min(20, int(20 * implemented / total))

    # 2. DTCG implementation completeness (0-20)
    dtcg_impl = report["sections"].get("dtcg", {}).get("implementation", {})
    dtcg_complete = sum(1 for m in dtcg_impl.values() if m.get("exists") and m.get("has_expected_classes"))
    scores["dtcg_implementation"] = min(20, int(20 * dtcg_complete / max(1, len(dtcg_impl))))

    # 3. DTCG actual usage in pipelines (0-20)
    dtcg_conclusions = report["sections"].get("dtcg", {}).get("conclusions", {})
    pipelines_using = dtcg_conclusions.get("pipelines_using_dtcg", 0)
    scores["dtcg_usage"] = min(20, int(20 * pipelines_using / 3))  # At least 3 pipelines should use DTCG

    # 4. Artifact lineage completeness (0-15)
    lineage_summary = report["sections"].get("artifact_lineage", {}).get("summary", {})
    stages_with_data = lineage_summary.get("stages_with_data", 0)
    total_stages = lineage_summary.get("total_stages", 1)
    scores["artifact_lineage"] = min(15, int(15 * stages_with_data / total_stages))

    # 5. Script/config/report reproducibility (0-10)
    code_summary = report["sections"].get("code_structure", {}).get("summary", {})
    total_files = code_summary.get("total_files", 0)
    scores["reproducibility"] = min(10, int(10 * total_files / 50))  # At least 50 files expected

    # 6. Evaluation pipeline completeness (0-10)
    eval_dir = PROJECT_ROOT / "src" / "autodata" / "evaluation"
    eval_files = len(list(eval_dir.glob("*.py"))) if eval_dir.exists() else 0
    scores["evaluation_completeness"] = min(10, int(10 * eval_files / 10))  # At least 10 eval files

    # 7. Code-level independence from Claude Code (0-5)
    verifier = report["sections"].get("project_agent_verifier", {})
    conclusions = verifier.get("conclusions", {})
    if conclusions.get("central_planner_used") and conclusions.get("context_selector_used"):
        scores["code_independence"] = 5
    elif conclusions.get("central_planner_used"):
        scores["code_independence"] = 3
    else:
        scores["code_independence"] = 1

    total_score = sum(scores.values())

    if total_score >= 85:
        classification = "ready_for_phase_6_6"
    elif total_score >= 70:
        classification = "mostly_ready_minor_fixes"
    elif total_score >= 50:
        classification = "partially_implemented"
    else:
        classification = "not_ready"

    return {
        "scores": scores,
        "total_score": total_score,
        "max_score": 100,
        "classification": classification,
    }
