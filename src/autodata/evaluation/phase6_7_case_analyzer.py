"""Case analyzer for Phase 6.7.

Extracts representative cases for paper discussion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def extract_cases(traces_by_system: dict[str, list[dict]], items: list[dict]) -> dict:
    """Extract representative cases.

    Returns:
        Dict with case lists.
    """
    items_by_id = {i.get("benchmark_id", ""): i for i in items}

    cases = {
        "dtcg_success_broadcast_fail": [],
        "dtcg_success_single_react_fail": [],
        "broadcast_success_dtcg_fail": [],
        "single_react_success_dtcg_fail": [],
        "context_saving_success": [],
        "dtcg_missing_context": [],
        "broadcast_overload": [],
    }

    dtcg = traces_by_system.get("dtcg", [])
    broadcast = traces_by_system.get("broadcast", [])
    single_react = traces_by_system.get("single_react", [])

    # Build lookup by benchmark_id
    dtcg_by_id = {t.get("benchmark_id", ""): t for t in dtcg}
    broadcast_by_id = {t.get("benchmark_id", ""): t for t in broadcast}
    react_by_id = {t.get("benchmark_id", ""): t for t in single_react}

    for bid in dtcg_by_id:
        d = dtcg_by_id.get(bid, {})
        b = broadcast_by_id.get(bid, {})
        s = react_by_id.get(bid, {})
        item = items_by_id.get(bid, {})

        if not d or not b:
            continue

        d_correct = d.get("is_correct", False)
        b_correct = b.get("is_correct", False)
        s_correct = s.get("is_correct", False) if s else False

        case = {
            "benchmark_id": bid,
            "question": item.get("question", "")[:200],
            "gold_answer": str(item.get("answer", ""))[:100],
            "task_type": item.get("task_type", ""),
            "difficulty": item.get("difficulty", ""),
            "dtcg_answer": d.get("parsed_answer", "")[:100],
            "dtcg_correct": d_correct,
            "dtcg_context_tokens": d.get("selected_context_tokens", 0),
            "dtcg_judge": d.get("judge_score"),
            "broadcast_answer": b.get("parsed_answer", "")[:100],
            "broadcast_correct": b_correct,
            "broadcast_context_tokens": b.get("broadcast_context_tokens", 0),
        }

        # DTCG correct, Broadcast wrong
        if d_correct and not b_correct:
            cases["dtcg_success_broadcast_fail"].append(case)

        # DTCG correct, Single-ReAct wrong
        if d_correct and not s_correct:
            cases["dtcg_success_single_react_fail"].append(case)

        # Broadcast correct, DTCG wrong
        if b_correct and not d_correct:
            cases["broadcast_success_dtcg_fail"].append(case)

        # Single-ReAct correct, DTCG wrong
        if s_correct and not d_correct:
            cases["single_react_success_dtcg_fail"].append(case)

        # DTCG saves >70% context with correct answer
        if d_correct and d.get("context_saving_ratio", 0) > 0.7:
            cases["context_saving_success"].append({
                **case,
                "saving_ratio": d.get("context_saving_ratio", 0),
            })

        # DTCG fails due to missing context
        if not d_correct and d.get("selected_context_tokens", 0) < 100:
            cases["dtcg_missing_context"].append(case)

        # Broadcast fails (context overload)
        if not b_correct and b.get("broadcast_context_tokens", 0) > 1000:
            cases["broadcast_overload"].append(case)

    # Limit each case list
    for key in cases:
        cases[key] = cases[key][:20]

    return cases


def save_cases(cases: dict, output_dir: Path = None):
    """Save case studies."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "case_studies"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary table
    summary = []
    for case_type, items in cases.items():
        for item in items[:5]:
            summary.append({"case_type": case_type, **{k: str(v)[:80] for k, v in item.items()}})

    with open(output_dir / "summary_case_table.csv", "w") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

    # Save individual case files
    for case_type, items in cases.items():
        if items:
            md_path = output_dir / f"{case_type}.md"
            with open(md_path, "w") as f:
                f.write(f"# {case_type.replace('_', ' ').title()}\n\n")
                f.write(f"Total cases: {len(items)}\n\n")
                for i, case in enumerate(items[:10], 1):
                    f.write(f"## Case {i}\n\n")
                    f.write(f"- **Question**: {case.get('question', '')}\n")
                    f.write(f"- **Gold Answer**: {case.get('gold_answer', '')}\n")
                    f.write(f"- **DTCG Answer**: {case.get('dtcg_answer', '')} (correct={case.get('dtcg_correct')})\n")
                    f.write(f"- **Broadcast Answer**: {case.get('broadcast_answer', '')} (correct={case.get('broadcast_correct')})\n")
                    f.write(f"- **DTCG Context Tokens**: {case.get('dtcg_context_tokens', 0)}\n")
                    f.write(f"- **Broadcast Context Tokens**: {case.get('broadcast_context_tokens', 0)}\n\n")

import csv
