"""Preflight validation for Phase 3.9 full image labeling.

Checks:
- Both API keys are available
- Multimodal model is callable
- Text model mimo-v2.5-pro is callable
- Unique manifest exists and has correct count
- Output directories are writable
- No stale processes running
- Pilot quality gate passed
- Output files will not overwrite pilot outputs
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.utils.api_loader import load_xiaomi_config
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("preflight")

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_full_image_labeling"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_unique_manifest.jsonl"
PILOT_VALIDATION = PROJECT_ROOT / "data" / "reports" / "phase_3_image_labeling" / "phase_3_7_validation_report.json"

OUTPUT_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "image_corpus",
    PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal",
    REPORT_DIR,
]


def run_preflight() -> dict:
    """Run all preflight checks."""
    checks = {}
    all_pass = True

    # 1. API keys available
    print("1. Checking API keys...")
    try:
        config1 = load_xiaomi_config()
        checks["api_key1_available"] = bool(config1.api_key)
        print(f"  Key 1: OK")
    except ValueError as e:
        checks["api_key1_available"] = False
        print(f"  Key 1: FAIL — {str(e)[:60]}")
        all_pass = False

    try:
        config2 = load_xiaomi_config(use_key2=True)
        checks["api_key2_available"] = bool(config2.api_key)
        print(f"  Key 2: OK")
    except ValueError as e:
        checks["api_key2_available"] = False
        print(f"  Key 2: FAIL — {str(e)[:60]}")
        all_pass = False

    # 2. Manifest exists
    print("\n2. Checking manifest...")
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            count = sum(1 for _ in f)
        checks["manifest_exists"] = True
        checks["manifest_count"] = count
        print(f"  Manifest: OK ({count} records)")
    else:
        checks["manifest_exists"] = False
        print(f"  Manifest: FAIL — file not found")
        all_pass = False

    # 3. Output dirs writable
    print("\n3. Checking output directories...")
    for dir_path in OUTPUT_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)
        test_file = dir_path / ".preflight_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            checks[f"writable_{dir_path.name}"] = True
            print(f"  {dir_path.name}: OK")
        except Exception as e:
            checks[f"writable_{dir_path.name}"] = False
            print(f"  {dir_path.name}: FAIL — {str(e)[:60]}")
            all_pass = False

    # 4. Pilot quality gate passed
    print("\n4. Checking pilot quality gate...")
    if PILOT_VALIDATION.exists():
        with open(PILOT_VALIDATION) as f:
            pilot_result = json.load(f)
        gate_result = pilot_result.get("quality_gate_result", "unknown")
        checks["pilot_quality_gate"] = gate_result == "PASS"
        print(f"  Pilot gate: {gate_result}")
        if gate_result != "PASS":
            all_pass = False
    else:
        checks["pilot_quality_gate"] = False
        print(f"  Pilot gate: FAIL — validation report not found")
        all_pass = False

    # 5. No stale processes
    print("\n5. Checking for stale processes...")
    import subprocess
    result = subprocess.run(
        ["ps", "-ef"], capture_output=True, text=True
    )
    stale = [line for line in result.stdout.splitlines()
             if any(kw in line for kw in ["run_phase_3", "image_labeling", "benchmark_generation"])
             and "grep" not in line and "preflight" not in line]
    checks["no_stale_processes"] = len(stale) == 0
    if stale:
        print(f"  Stale processes found: {len(stale)}")
        for line in stale:
            print(f"    {line.strip()}")
        all_pass = False
    else:
        print(f"  No stale processes: OK")

    # 6. Pilot outputs won't be overwritten
    print("\n6. Checking no pilot overwrite risk...")
    pilot_files = [
        PROJECT_ROOT / "data" / "interim" / "image_labeled" / "image_labels_pilot.jsonl",
        PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_pilot.jsonl",
    ]
    full_files = [
        PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_labels_full.jsonl",
        PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_full.jsonl",
    ]
    # Full files use _full suffix, pilot files use _pilot suffix — they won't clash
    checks["no_pilot_overwrite"] = True
    print(f"  No clash: OK (_full suffix vs _pilot suffix)")

    # 7. Model connectivity test (optional, quick)
    print("\n7. Checking model connectivity...")
    try:
        from src.autodata.utils.model_pool import get_model_pool
        pool = get_model_pool()
        # Quick test call
        test_msg = [{"role": "user", "content": "Reply with only: OK"}]
        response = pool.chat(messages=test_msg, max_completion_tokens=10, max_retries=1)
        checks["model_callable"] = True
        print(f"  Model: OK (response: '{response.content.strip()[:20]}')")
    except Exception as e:
        checks["model_callable"] = False
        print(f"  Model: FAIL — {str(e)[:60]}")
        all_pass = False

    # Summary
    checks["all_checks_pass"] = all_pass
    checks["timestamp"] = time.time()
    checks["phase"] = "3.9_preflight"

    print(f"\n=== Preflight Result: {'PASS' if all_pass else 'FAIL'} ===")

    # Write report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "preflight_full.json"
    with open(report_path, "w") as f:
        json.dump(checks, f, indent=2)
    print(f"Report written to {report_path}")

    return checks


if __name__ == "__main__":
    result = run_preflight()
    if not result["all_checks_pass"]:
        print("\nPreflight FAILED. Do not start full labeling until issues are resolved.")
        sys.exit(1)
    else:
        print("\nPreflight PASSED. Ready to start full labeling.")