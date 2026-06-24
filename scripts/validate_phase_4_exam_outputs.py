"""Validate Phase 4 exam extraction outputs.

Usage:
    python scripts/validate_phase_4_exam_outputs.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_4_exam_extraction"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "exam_questions"
EXAM_DIR = PROJECT_ROOT / "exam_raw_data"


def check_json_valid(path: Path) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    json.loads(line)
        return True
    except Exception:
        return False


def check_required_fields(path: Path, fields: list[str]) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    for field in fields:
                        if field not in r:
                            return False
        return True
    except Exception:
        return False


def check_source_provenance(path: Path) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if not r.get("source_file"):
                        return False
        return True
    except Exception:
        return False


def check_no_api_keys(path: Path) -> bool:
    try:
        with open(path) as f:
            content = f.read()
            if "tp-" in content or "API_KEY" in content:
                return False
        return True
    except Exception:
        return True


def main():
    checks = {}

    print("1. Checking file existence...")
    files_to_check = {
        "text_blocks": PROJECT_ROOT / "data" / "interim" / "exam_extracted_text" / "exam_text_blocks.jsonl",
        "raw_questions": OUTPUT_DIR / "exam_questions_raw.jsonl",
        "validated": OUTPUT_DIR / "exam_questions_validated.jsonl",
        "quality_scores": OUTPUT_DIR / "exam_question_quality_scores.jsonl",
        "unique": OUTPUT_DIR / "exam_questions_unique.jsonl",
        "benchmark_ready": OUTPUT_DIR / "exam_questions_benchmark_ready_candidates.jsonl",
        "failures": OUTPUT_DIR / "exam_extraction_failures.jsonl",
    }
    for name, path in files_to_check.items():
        exists = path.exists()
        checks[f"file_exists_{name}"] = exists
        print(f"  {name}: {'OK' if exists else 'MISSING'}")

    print("\n2. Checking JSON validity...")
    for name, path in files_to_check.items():
        if path.exists() and path.suffix == ".jsonl":
            valid = check_json_valid(path)
            checks[f"json_valid_{name}"] = valid
            print(f"  {name}: {'OK' if valid else 'INVALID'}")

    print("\n3. Checking source provenance...")
    for name in ["raw_questions", "validated", "unique"]:
        path = files_to_check.get(name)
        if path and path.exists():
            ok = check_source_provenance(path)
            checks[f"provenance_{name}"] = ok
            print(f"  {name}: {'OK' if ok else 'MISSING PROVENANCE'}")

    print("\n4. Checking no API keys in outputs...")
    for name, path in files_to_check.items():
        if path.exists():
            ok = check_no_api_keys(path)
            checks[f"no_api_keys_{name}"] = ok
            print(f"  {name}: {'OK' if ok else 'API KEY FOUND'}")

    print("\n5. Checking DTCG trace...")
    trace_path = REPORT_DIR / "dtcg_exam_extraction_trace.json"
    if trace_path.exists():
        with open(trace_path) as f:
            trace = json.load(f)
        checks["dtcg_exists"] = True
        checks["dtcg_nodes"] = trace.get("node_count", 0)
        checks["dtcg_edges"] = trace.get("edge_count", 0)
        print(f"  Trace: {trace.get('node_count', 0)} nodes, {trace.get('edge_count', 0)} edges")
    else:
        checks["dtcg_exists"] = False
        print(f"  Trace: MISSING")

    print("\n6. Checking context packages...")
    packages_path = REPORT_DIR / "context_packages_exam_extraction.jsonl"
    checks["packages_exists"] = packages_path.exists()
    print(f"  Packages: {'OK' if packages_path.exists() else 'MISSING'}")

    print("\n7. Checking API key policy...")
    policy_path = REPORT_DIR / "api_key_policy_phase4.json"
    if policy_path.exists():
        with open(policy_path) as f:
            policy = json.load(f)
        checks["api_key_policy"] = policy.get("api_key_policy") == "use_key1_only"
        print(f"  Policy: {policy.get('api_key_policy', 'unknown')}")
    else:
        checks["api_key_policy"] = False
        print(f"  Policy: MISSING")

    print("\n8. Counting records...")
    for name, path in files_to_check.items():
        if path.exists() and path.suffix == ".jsonl":
            count = sum(1 for line in open(path) if line.strip())
            checks[f"count_{name}"] = count
            print(f"  {name}: {count}")

    # Overall result
    critical = [
        checks.get("file_exists_raw_questions", False),
        checks.get("file_exists_validated", False),
        checks.get("json_valid_raw_questions", False),
        checks.get("json_valid_validated", False),
        checks.get("api_key_policy", False),
        checks.get("dtcg_exists", False),
    ]
    all_passed = all(critical)
    checks["validation_result"] = "PASS" if all_passed else "FAIL"
    checks["critical_checks_passed"] = sum(critical)
    checks["timestamp"] = time.time()

    print(f"\n=== Validation Result: {'PASS' if all_passed else 'FAIL'} ===")
    print(f"Critical checks: {sum(critical)}/{len(critical)}")

    # Save report
    with open(REPORT_DIR / "validation_exam_extraction.json", "w") as f:
        json.dump(checks, f, indent=2)

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
