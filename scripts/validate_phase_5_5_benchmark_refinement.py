"""Validate Phase 5.5 benchmark refinement outputs.

Usage:
    python scripts/validate_phase_5_5_benchmark_refinement.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_5_5_benchmark_refinement"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
SUBSETS_DIR = BENCHMARK_DIR / "subsets"


def check_jsonl_valid(path: Path) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    json.loads(line)
        return True
    except Exception:
        return False


def main():
    checks = {}

    print("1. Checking file existence...")
    files = {
        "audit_json": REPORT_DIR / "benchmark_audit_detailed.json",
        "subset_report": REPORT_DIR / "subset_report.json",
        "dtcg_trace": REPORT_DIR / "dtcg_phase_5_5_trace.json",
        "dtcg_packages": REPORT_DIR / "context_packages_phase_5_5.jsonl",
        "dtcg_stats": REPORT_DIR / "dtcg_phase_5_5_statistics.json",
        "eval_protocol_json": BENCHMARK_DIR / "EVALUATION_PROTOCOL.json",
        "eval_protocol_md": BENCHMARK_DIR / "EVALUATION_PROTOCOL.md",
        "api_key_policy": REPORT_DIR / "api_key_policy_phase_5_5.json",
    }
    for name, path in files.items():
        exists = path.exists()
        checks[f"exists_{name}"] = exists
        print(f"  {name}: {'OK' if exists else 'MISSING'}")

    print("\n2. Checking subsets...")
    subset_files = list(SUBSETS_DIR.glob("cfbench_*.jsonl"))
    checks["subset_count"] = len(subset_files)
    print(f"  Subset files: {len(subset_files)}")
    for sf in sorted(subset_files):
        count = sum(1 for line in open(sf) if line.strip())
        print(f"    {sf.name}: {count}")

    print("\n3. Checking DTCG trace...")
    trace_path = REPORT_DIR / "dtcg_phase_5_5_trace.json"
    if trace_path.exists():
        with open(trace_path) as f:
            trace = json.load(f)
        checks["dtcg_nodes"] = trace.get("node_count", 0)
        checks["dtcg_edges"] = trace.get("edge_count", 0)
        print(f"  Nodes: {trace.get('node_count', 0)}")
        print(f"  Edges: {trace.get('edge_count', 0)}")

    print("\n4. Checking evaluation protocol...")
    protocol_path = BENCHMARK_DIR / "EVALUATION_PROTOCOL.json"
    if protocol_path.exists():
        with open(protocol_path) as f:
            protocol = json.load(f)
        checks["protocol_subsets"] = len(protocol.get("subsets", {}))
        checks["protocol_metrics"] = len(protocol.get("metrics", {}))
        print(f"  Subsets: {len(protocol.get('subsets', {}))}")
        print(f"  Metrics: {len(protocol.get('metrics', {}))}")

    print("\n5. Checking API key policy...")
    policy_path = REPORT_DIR / "api_key_policy_phase_5_5.json"
    if policy_path.exists():
        with open(policy_path) as f:
            policy = json.load(f)
        checks["api_key1_only"] = policy.get("api_key_policy") == "use_key1_only"
        checks["api_key2_disabled"] = policy.get("api_key2_disabled", False)
        print(f"  Policy: {policy.get('api_key_policy')}")
        print(f"  Key2 disabled: {policy.get('api_key2_disabled')}")

    print("\n6. Checking enhanced candidates...")
    text_candidates = PROJECT_ROOT / "data" / "benchmark_candidates" / "text_enhanced" / "text_enhanced_candidates.jsonl"
    text_validated = PROJECT_ROOT / "data" / "benchmark_candidates" / "text_enhanced" / "text_enhanced_candidates_validated.jsonl"
    agent_candidates = PROJECT_ROOT / "data" / "benchmark_candidates" / "agent_task" / "agent_task_candidates.jsonl"
    agent_validated = PROJECT_ROOT / "data" / "benchmark_candidates" / "agent_task" / "agent_task_candidates_validated.jsonl"

    for name, path in [("text_candidates", text_candidates), ("text_validated", text_validated),
                       ("agent_candidates", agent_candidates), ("agent_validated", agent_validated)]:
        if path.exists():
            count = sum(1 for line in open(path) if line.strip())
            checks[f"count_{name}"] = count
            print(f"  {name}: {count}")
        else:
            print(f"  {name}: NOT YET CREATED")

    # Overall
    critical = [
        checks.get("exists_audit_json", False),
        checks.get("exists_dtcg_trace", False),
        checks.get("exists_eval_protocol_json", False),
        checks.get("exists_api_key_policy", False),
        checks.get("api_key1_only", False),
        checks.get("api_key2_disabled", False),
    ]
    all_passed = all(critical)
    checks["validation_result"] = "PASS" if all_passed else "FAIL"
    checks["timestamp"] = time.time()

    print(f"\n=== Validation Result: {'PASS' if all_passed else 'FAIL'} ===")

    with open(REPORT_DIR / "validation_phase_5_5.json", "w") as f:
        json.dump(checks, f, indent=2)

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
