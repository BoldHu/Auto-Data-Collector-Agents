"""Validate Phase 5 benchmark outputs.

Usage:
    python scripts/validate_phase_5_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_5_benchmark_construction"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


def check_jsonl_valid(path: Path) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    json.loads(line)
        return True
    except Exception:
        return False


def check_unique_ids(path: Path, id_field: str) -> tuple[bool, int]:
    ids = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.append(r.get(id_field, ""))
    dup_count = len(ids) - len(set(ids))
    return dup_count == 0, dup_count


def check_provenance(path: Path) -> bool:
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not r.get("source_refs") and not r.get("source_type"):
                    return False
    return True


def check_no_api_keys(path: Path) -> bool:
    with open(path) as f:
        content = f.read(50000)
        return "tp-" not in content and "API_KEY" not in content


def main():
    checks = {}

    print("1. Checking file existence...")
    files = {
        "all_candidates": BENCHMARK_DIR / "final_candidates" / "benchmark_candidates_all.jsonl",
        "benchmark_dev": BENCHMARK_DIR / "carbon_fiber_benchmark_dev.jsonl",
        "benchmark_test": BENCHMARK_DIR / "carbon_fiber_benchmark_test.jsonl",
        "benchmark_all": BENCHMARK_DIR / "carbon_fiber_benchmark_all.jsonl",
        "leakage_report": BENCHMARK_DIR / "leakage_report.json",
        "statistics": BENCHMARK_DIR / "benchmark_statistics.json",
        "benchmark_card": BENCHMARK_DIR / "CARBON_FIBER_BENCHMARK_CARD.md",
    }
    for name, path in files.items():
        exists = path.exists()
        checks[f"exists_{name}"] = exists
        print(f"  {name}: {'OK' if exists else 'MISSING'}")

    print("\n2. Checking JSON validity...")
    for name in ["all_candidates", "benchmark_dev", "benchmark_test", "benchmark_all"]:
        path = files.get(name)
        if path and path.exists() and path.suffix == ".jsonl":
            valid = check_jsonl_valid(path)
            checks[f"valid_{name}"] = valid
            print(f"  {name}: {'OK' if valid else 'INVALID'}")

    print("\n3. Checking unique IDs...")
    for name in ["benchmark_dev", "benchmark_test"]:
        path = files.get(name)
        if path and path.exists():
            unique, dups = check_unique_ids(path, "benchmark_id")
            checks[f"unique_{name}"] = unique
            print(f"  {name}: {'OK' if unique else f'{dups} DUPLICATES'}")

    print("\n4. Checking provenance...")
    for name in ["benchmark_dev", "benchmark_test"]:
        path = files.get(name)
        if path and path.exists():
            ok = check_provenance(path)
            checks[f"provenance_{name}"] = ok
            print(f"  {name}: {'OK' if ok else 'MISSING PROVENANCE'}")

    print("\n5. Checking no API keys...")
    for name, path in files.items():
        if path and path.exists() and path.suffix in (".jsonl", ".json"):
            ok = check_no_api_keys(path)
            checks[f"no_api_keys_{name}"] = ok
            print(f"  {name}: {'OK' if ok else 'API KEY FOUND'}")

    print("\n6. Checking DTCG trace...")
    trace_path = REPORT_DIR / "dtcg_phase_5_trace.json"
    if trace_path.exists():
        with open(trace_path) as f:
            trace = json.load(f)
        checks["dtcg_nodes"] = trace.get("node_count", 0)
        checks["dtcg_edges"] = trace.get("edge_count", 0)
        print(f"  Trace: {trace.get('node_count', 0)} nodes, {trace.get('edge_count', 0)} edges")
    else:
        checks["dtcg_exists"] = False
        print(f"  Trace: MISSING")

    print("\n7. Counting records...")
    for name in ["benchmark_dev", "benchmark_test", "benchmark_all"]:
        path = files.get(name)
        if path and path.exists():
            count = sum(1 for line in open(path) if line.strip())
            checks[f"count_{name}"] = count
            print(f"  {name}: {count}")

    # Overall
    critical = [
        checks.get("exists_benchmark_dev", False),
        checks.get("exists_benchmark_test", False),
        checks.get("valid_benchmark_dev", False),
        checks.get("valid_benchmark_test", False),
        checks.get("unique_benchmark_dev", False),
        checks.get("unique_benchmark_test", False),
    ]
    all_passed = all(critical)
    checks["validation_result"] = "PASS" if all_passed else "FAIL"
    checks["timestamp"] = time.time()

    print(f"\n=== Validation Result: {'PASS' if all_passed else 'FAIL'} ===")

    with open(REPORT_DIR / "validation_phase_5.json", "w") as f:
        json.dump(checks, f, indent=2)

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
