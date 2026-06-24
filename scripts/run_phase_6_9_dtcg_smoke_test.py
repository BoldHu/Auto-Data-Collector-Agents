"""Phase 6.9: DTCG smoke test after repair.

Runs 20 tasks to verify DTCG context injection is working.

Usage:
    python scripts/run_phase_6_9_dtcg_smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time
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
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient
    from src.autodata.evaluation.system_baselines import run_dtcg, run_broadcast, run_single_react

    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_9"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_9_dtcg_diagnosis"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load 10 long-context + 10 stress items
    lc_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_8" / "long_context_expanded.jsonl"
    stress_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "ablation_subset_stress.jsonl"

    items = []
    if lc_path.exists():
        lc_items = load_jsonl(lc_path)
        items.extend(("long_context", it) for it in lc_items[:10])
    if stress_path.exists():
        stress_items = load_jsonl(stress_path)
        items.extend(("stress", it) for it in stress_items[:10])

    if not items:
        print("ERROR: No items found")
        return

    print(f"Smoke test: {len(items)} items")

    client = UnifiedModelClient(model_name="deepseek-v4-flash")

    traces = []
    results = {
        "total": len(items),
        "dtcg_context_nonempty": 0,
        "dtcg_fallback_used": 0,
        "dtcg_context_in_prompt": 0,
        "dtcg_nonzero_tokens": 0,
        "systems": {},
    }

    for i, (subset, item) in enumerate(items):
        print(f"  [{i+1}/{len(items)}] {subset} - {item.get('benchmark_id', '')[:20]}...")

        # Run DTCG
        dtcg_trace = run_dtcg(client, item)
        traces.append(dtcg_trace.to_dict())

        # Check context injection
        answer = dtcg_trace.raw_answer
        has_context = "无" not in answer[:80] and "空" not in answer[:80] and "未选择" not in answer[:80] and "未提供" not in answer[:80]
        if has_context:
            results["dtcg_context_nonempty"] += 1
        if dtcg_trace.fallback_used:
            results["dtcg_fallback_used"] += 1
        if dtcg_trace.selected_context_text:
            results["dtcg_context_in_prompt"] += 1
        if dtcg_trace.selected_context_tokens > 0:
            results["dtcg_nonzero_tokens"] += 1

        # Run broadcast for comparison
        broadcast_trace = run_broadcast(client, item)
        traces.append(broadcast_trace.to_dict())

        # Run single_react for comparison
        react_trace = run_single_react(client, item)
        traces.append(react_trace.to_dict())

        # Track per-system
        for trace in [dtcg_trace, broadcast_trace, react_trace]:
            s = trace.system_type
            if s not in results["systems"]:
                results["systems"][s] = {"total": 0, "context_empty": 0, "avg_ctx_tokens": 0}
            results["systems"][s]["total"] += 1
            if "无" in trace.raw_answer[:80] or "空" in trace.raw_answer[:80]:
                results["systems"][s]["context_empty"] += 1
            results["systems"][s]["avg_ctx_tokens"] += trace.selected_context_tokens

    # Compute averages
    for s in results["systems"]:
        total = results["systems"][s]["total"]
        results["systems"][s]["avg_ctx_tokens"] = round(results["systems"][s]["avg_ctx_tokens"] / max(total, 1))
        results["systems"][s]["context_empty_rate"] = round(results["systems"][s]["context_empty"] / max(total, 1), 3)

    # Save traces
    with open(output_dir / "dtcg_smoke_test_traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Save results
    with open(report_dir / "dtcg_smoke_test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Write report
    with open(report_dir / "dtcg_smoke_test_results.md", "w") as f:
        f.write("# DTCG Smoke Test Results\n\n")
        f.write(f"Total items: {results['total']}\n\n")
        f.write("## DTCG Context Injection\n\n")
        f.write(f"- Context non-empty in answer: {results['dtcg_context_nonempty']}/{results['total']}\n")
        f.write(f"- Fallback used: {results['dtcg_fallback_used']}/{results['total']}\n")
        f.write(f"- Context text in prompt: {results['dtcg_context_in_prompt']}/{results['total']}\n")
        f.write(f"- Nonzero token count: {results['dtcg_nonzero_tokens']}/{results['total']}\n\n")

        f.write("## Per-System Comparison\n\n")
        f.write("| System | Total | Context Empty Rate | Avg Context Tokens |\n")
        f.write("|--------|-------|--------------------|--------------------|\n")
        for s, data in results["systems"].items():
            f.write(f"| {s} | {data['total']} | {data['context_empty_rate']:.1%} | {data['avg_ctx_tokens']} |\n")

    # Print summary
    print(f"\nSmoke Test Results:")
    print(f"  DTCG context non-empty: {results['dtcg_context_nonempty']}/{results['total']}")
    print(f"  DTCG fallback used: {results['dtcg_fallback_used']}/{results['total']}")
    print(f"  DTCG context in prompt: {results['dtcg_context_in_prompt']}/{results['total']}")

    passed = results["dtcg_context_in_prompt"] >= results["total"] * 0.5
    print(f"\nSmoke test {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
