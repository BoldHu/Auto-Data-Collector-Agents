"""Phase 6.9: Audit Phase 6.8 evidence for inconsistencies.

Usage:
    python scripts/run_phase_6_9_evidence_audit.py
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


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_9_dtcg_diagnosis"
    report_dir.mkdir(parents=True, exist_ok=True)

    findings = []

    def finding(category: str, description: str, severity: str = "warning"):
        findings.append({
            "category": category,
            "description": description,
            "severity": severity,
        })

    # 1. Load Phase 6.8 report claims
    report_path = PROJECT_ROOT / "data" / "reports" / "phase_6_8_evidence_consolidation" / "PHASE_6_8_REPORT.md"
    report_text = report_path.read_text() if report_path.exists() else ""

    # 2. Load component ablation scores (updated CSV)
    csv_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "dtcg_component_ablation_scores.csv"
    component_scores = {}
    if csv_path.exists():
        with open(csv_path) as f:
            header = None
            for line in f:
                parts = line.strip().split(",")
                if header is None:
                    header = parts
                    continue
                component_scores[parts[0]] = {
                    "total": int(parts[1]),
                    "correct": int(parts[2]),
                    "accuracy": float(parts[3]),
                    "avg_judge_score": float(parts[4]),
                    "avg_context_tokens": int(parts[5]),
                }

    # 3. Check claim: "40% vs 10% for topk/static"
    if "40% vs 10%" in report_text:
        actual_full = component_scores.get("dtcg_full", {}).get("accuracy", 0)
        actual_topk = component_scores.get("dtcg_topk", {}).get("accuracy", 0)
        actual_static = component_scores.get("dtcg_static", {}).get("accuracy", 0)
        if actual_full != 0.40 or actual_topk != 0.10:
            finding(
                "claim_contradiction",
                f"Report claims '40% vs 10%' but actual: dtcg_full={actual_full:.1%}, dtcg_topk={actual_topk:.1%}, dtcg_static={actual_static:.1%}",
                "high"
            )

    # 4. Check claim: "DTCG outperforms broadcast on long-context"
    lc_traces_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_8" / "long_context_expanded" / "system_ablation_traces.jsonl"
    if lc_traces_path.exists():
        lc_traces = load_jsonl(lc_traces_path)
        lc_by_system = {}
        for t in lc_traces:
            s = t.get("system_type")
            if s not in lc_by_system:
                lc_by_system[s] = {"total": 0, "correct": 0}
            lc_by_system[s]["total"] += 1
            if t.get("is_correct"):
                lc_by_system[s]["correct"] += 1

        dtcg_acc = lc_by_system.get("dtcg", {}).get("correct", 0) / max(lc_by_system.get("dtcg", {}).get("total", 1), 1)
        broadcast_acc = lc_by_system.get("broadcast", {}).get("correct", 0) / max(lc_by_system.get("broadcast", {}).get("total", 1), 1)

        if "outperforms broadcast" in report_text and dtcg_acc <= broadcast_acc:
            finding(
                "claim_unsupported",
                f"Report claims 'DTCG outperforms broadcast' but dtcg={dtcg_acc:.1%} vs broadcast={broadcast_acc:.1%}",
                "high"
            )

    # 5. Check DTCG empty context rate
    all_traces_paths = [
        (PROJECT_ROOT / "data" / "evaluation" / "phase_6_8" / "long_context_expanded" / "system_ablation_traces.jsonl", "long_context_expanded"),
        (PROJECT_ROOT / "data" / "evaluation" / "phase_6_8" / "stress" / "system_ablation_traces.jsonl", "stress"),
    ]

    for traces_path, subset_name in all_traces_paths:
        if not traces_path.exists():
            continue
        traces = load_jsonl(traces_path)
        dtcg_traces = [t for t in traces if t.get("system_type") == "dtcg"]
        empty_count = 0
        for t in dtcg_traces:
            answer = t.get("raw_answer", "")
            if "无" in answer[:80] or "空" in answer[:80] or "未选择" in answer[:80] or "未提供" in answer[:80]:
                empty_count += 1
        empty_rate = empty_count / max(len(dtcg_traces), 1)
        if empty_rate > 0.5:
            finding(
                "empty_context",
                f"{subset_name}: {empty_count}/{len(dtcg_traces)} ({empty_rate:.0%}) DTCG traces have empty/missing context in answer",
                "critical"
            )

    # 6. Check Avg Context Tokens = 0 for systems that should have prompts
    if "Avg Context Tokens |" in report_text:
        for line in report_text.split("\n"):
            if "| direct_llm" in line or "| single_react" in line or "| broadcast" in line:
                if "| 0 |" in line:
                    finding(
                        "token_accounting",
                        f"System shows Avg Context Tokens = 0 but should have prompt tokens: {line.strip()}",
                        "medium"
                    )

    # 7. Check component ablation sample size
    for variant, data in component_scores.items():
        if data["total"] < 20:
            finding(
                "sample_size",
                f"Component ablation {variant} has only {data['total']} items (need >=20 for reliability)",
                "medium"
            )

    # 8. Check statistical significance support
    sig_path = PROJECT_ROOT / "data" / "reports" / "phase_6_8_evidence_consolidation" / "statistical_significance.json"
    sig_data = load_json(sig_path)
    for key in ["phase68_long_context_expanded", "phase68_stress"]:
        if key in sig_data:
            for pair, comp in sig_data[key].get("pairwise_comparisons", {}).items():
                mc = comp.get("mcnemar", {})
                if "broadcast" in pair and mc.get("significant", False):
                    finding(
                        "claim_supported",
                        f"Statistical significance confirmed: {pair} (p={mc.get('p_value')})",
                        "info"
                    )
                elif "broadcast" in pair and not mc.get("significant", False):
                    finding(
                        "claim_unsupported",
                        f"No statistical significance: {pair} (p={mc.get('p_value')})",
                        "high"
                    )

    # Classify claims
    claims_classification = {
        "strongly_supported": [
            "DTCG significantly outperforms Plan-and-Execute (p=0.0001 on stress, p=0.04 on long-context)",
            "DTCG provides a traceable graph-based context selection mechanism",
        ],
        "weakly_supported": [
            "DTCG reduces context redundancy compared with broadcast (requires corrected token accounting)",
            "Graph structure matters (component ablation sample size only 10-39)",
        ],
        "unsupported": [
            "DTCG universally outperforms Single-ReAct (not significant, p>0.5)",
            "DTCG universally outperforms Broadcast (not significant, p=1.0)",
            "DTCG is always more accurate than Broadcast (actual: DTCG 18% vs Broadcast 17% on long-context, 26% vs 29% on stress)",
        ],
        "contradicted_by_results": [
            "DTCG outperforms broadcast on stress tasks (actual: DTCG 26% < Broadcast 29%)",
        ],
        "requires_rerun": [
            "DTCG component ablation (94% of DTCG traces had empty context due to bug, now fixed)",
            "DTCG vs Broadcast comparison (context injection was broken, rerun needed)",
            "DTCG vs Static Router comparison (context injection was broken, rerun needed)",
        ],
    }

    # Save audit
    audit = {
        "phase": "6.9",
        "audit_type": "evidence_consistency",
        "total_findings": len(findings),
        "critical_findings": len([f for f in findings if f["severity"] == "critical"]),
        "high_findings": len([f for f in findings if f["severity"] == "high"]),
        "findings": findings,
        "claims_classification": claims_classification,
        "component_scores": component_scores,
    }

    with open(report_dir / "evidence_audit.json", "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    # Write markdown
    with open(report_dir / "evidence_audit.md", "w") as f:
        f.write("# Phase 6.9 Evidence Audit\n\n")
        f.write(f"Total findings: {len(findings)}\n")
        f.write(f"Critical: {len([x for x in findings if x['severity'] == 'critical'])}\n")
        f.write(f"High: {len([x for x in findings if x['severity'] == 'high'])}\n\n")

        f.write("## Findings\n\n")
        for i, fi in enumerate(findings):
            f.write(f"### {i+1}. [{fi['severity'].upper()}] {fi['category']}\n")
            f.write(f"{fi['description']}\n\n")

        f.write("## Claims Classification\n\n")
        for category, items in claims_classification.items():
            f.write(f"### {category}\n")
            for item in items:
                f.write(f"- {item}\n")
            f.write("\n")

    print(f"Audit complete: {len(findings)} findings")
    for fi in findings:
        if fi["severity"] in ("critical", "high"):
            print(f"  [{fi['severity']}] {fi['description'][:100]}")


if __name__ == "__main__":
    main()
