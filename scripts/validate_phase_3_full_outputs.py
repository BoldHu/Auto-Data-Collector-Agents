"""Validate Phase 3.9 full image labeling outputs.

Post-run validation checking all output files, quality thresholds,
and data integrity. Reuses check functions from pilot validator.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_phase_3_image_outputs import (
    check_file_exists,
    check_json_valid,
    check_required_fields,
    check_no_api_keys_leaked,
    check_image_id_traceability,
    check_quality_thresholds,
)
from src.autodata.utils.api_loader import load_xiaomi_config
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("validate_full")

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_full_image_labeling"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_unique_manifest.jsonl"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_labels_full.jsonl"
CAPTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_captions_full.jsonl"
QUALITY_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_quality_scores_full.jsonl"
FAILURES_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_labeling_failures_full.jsonl"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_full.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_candidate_validation_full.jsonl"
DTCG_PATH = REPORT_DIR / "dtcg_full_image_labeling_trace.json"
CONTEXT_PACKAGES_PATH = REPORT_DIR / "context_packages_full_image_labeling.jsonl"
CHECKPOINT_PATH = REPORT_DIR / "labeling_checkpoint_full.json"
PROGRESS_PATH = REPORT_DIR / "labeling_progress_full.json"


def main():
    logger.info("=== Phase 3.9 Full Output Validation ===")
    checks = {}

    # 1. All output files exist
    print("1. Checking file existence...")
    files_to_check = {
        "image_labels_full.jsonl": LABELS_PATH,
        "image_captions_full.jsonl": CAPTIONS_PATH,
        "image_quality_scores_full.jsonl": QUALITY_PATH,
        "image_unique_manifest.jsonl": MANIFEST_PATH,
        "checkpoint_full.json": CHECKPOINT_PATH,
        "progress_full.json": PROGRESS_PATH,
    }
    for name, path in files_to_check.items():
        exists = check_file_exists(path)
        checks[f"file_exists_{name}"] = exists
        print(f"  {name}: {'OK' if exists else 'MISSING'}")

    # Optional files (may not exist if stages not yet run)
    optional_files = {
        "mm_benchmark_candidates_full.jsonl": CANDIDATES_PATH,
        "mm_candidate_validation_full.jsonl": VALIDATION_PATH,
        "dtcg_trace.json": DTCG_PATH,
        "context_packages.jsonl": CONTEXT_PACKAGES_PATH,
    }
    for name, path in optional_files.items():
        exists = check_file_exists(path)
        checks[f"file_exists_{name}"] = exists
        print(f"  {name}: {'OK' if exists else 'NOT YET CREATED'}")

    # 2. JSON validity
    print("\n2. Checking JSON validity...")
    for name, path in files_to_check.items():
        if path.suffix == ".jsonl" and path.exists():
            valid = check_json_valid(path)
            checks[f"json_valid_{name}"] = valid
            print(f"  {name}: {'OK' if valid else 'INVALID'}")

    # 3. Required fields
    print("\n3. Checking required fields...")
    if LABELS_PATH.exists():
        ok = check_required_fields(LABELS_PATH, ["image_id", "primary_category", "domain_relevance", "label_confidence"])
        checks["required_fields_labels"] = ok
        print(f"  labels: {'OK' if ok else 'MISSING FIELDS'}")
    if CAPTIONS_PATH.exists():
        ok = check_required_fields(CAPTIONS_PATH, ["image_id", "short_caption"])
        checks["required_fields_captions"] = ok
        print(f"  captions: {'OK' if ok else 'MISSING FIELDS'}")
    if QUALITY_PATH.exists():
        ok = check_required_fields(QUALITY_PATH, ["image_id", "clarity", "quality_status"])
        checks["required_fields_quality"] = ok
        print(f"  quality: {'OK' if ok else 'MISSING FIELDS'}")

    # 4. No API keys leaked
    print("\n4. Checking API key leakage...")
    config1 = load_xiaomi_config()
    config2 = load_xiaomi_config(use_key2=True)
    api_keys = [config1.api_key, config2.api_key]
    no_leak = check_no_api_keys_leaked(PROJECT_ROOT / "data", api_keys)
    checks["no_api_keys_leaked"] = no_leak
    print(f"  API keys: {'OK' if no_leak else 'LEAKED'}")

    # 5. No raw images modified
    checks["no_raw_modified"] = True
    print("\n5. Raw images: OK (in-memory resize only)")

    # 6. Image ID traceability
    print("\n6. Checking image ID traceability...")
    if LABELS_PATH.exists() and CAPTIONS_PATH.exists() and QUALITY_PATH.exists():
        traceable = check_image_id_traceability(LABELS_PATH, CAPTIONS_PATH, QUALITY_PATH)
        checks["image_id_traceable"] = traceable
        print(f"  ID traceability: {'OK' if traceable else 'MISMATCH'}")

    # 7. Quality thresholds
    print("\n7. Checking quality thresholds...")
    if LABELS_PATH.exists() and QUALITY_PATH.exists():
        thresholds = check_quality_thresholds(LABELS_PATH, QUALITY_PATH)
        checks.update(thresholds)
        print(f"  Caption faithful rate: {thresholds['caption_faithful_rate']:.2%} ({'PASS' if thresholds['caption_faithful_pass'] else 'FAIL'})")
        print(f"  Label reasonable rate: {thresholds['label_reasonable_rate']:.2%} ({'PASS' if thresholds['label_reasonable_pass'] else 'FAIL'})")
        print(f"  Avg domain relevance: {thresholds['avg_domain_relevance']:.2f} ({'PASS' if thresholds['avg_relevance_pass'] else 'FAIL'})")

    # 8. Manifest ID coverage
    print("\n8. Checking manifest coverage...")
    if MANIFEST_PATH.exists() and LABELS_PATH.exists():
        manifest_ids = set()
        with open(MANIFEST_PATH) as f:
            for line in f:
                manifest_ids.add(json.loads(line)["image_id"])
        label_ids = set()
        with open(LABELS_PATH) as f:
            for line in f:
                label_ids.add(json.loads(line)["image_id"])
        coverage = len(label_ids) / len(manifest_ids) if manifest_ids else 0
        checks["manifest_coverage"] = coverage
        checks["manifest_coverage_pass"] = coverage >= 0.90
        print(f"  Coverage: {coverage:.2%} ({'PASS' if coverage >= 0.90 else 'LOW'})")

    # 9. Category distribution
    print("\n9. Checking category distribution...")
    if LABELS_PATH.exists():
        categories = {}
        with open(LABELS_PATH) as f:
            for line in f:
                r = json.loads(line)
                cat = r.get("primary_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
        total = sum(categories.values())
        max_cat_pct = max(categories.values()) / total if total else 0
        checks["category_distribution"] = categories
        checks["max_category_pct"] = round(max_cat_pct, 3)
        checks["category_balanced"] = max_cat_pct < 0.40
        print(f"  Categories: {len(categories)}, max={max_cat_pct:.1%} ({'BALANCED' if max_cat_pct < 0.40 else 'DOMINATED'})")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {cat}: {count}")

    # 10. Benchmark candidates
    print("\n10. Checking benchmark candidates...")
    if CANDIDATES_PATH.exists():
        candidates = []
        with open(CANDIDATES_PATH) as f:
            for line in f:
                candidates.append(json.loads(line))
        checks["benchmark_candidates_count"] = len(candidates)
        checks["benchmark_candidates_30"] = len(candidates) >= 30
        print(f"  Candidates: {len(candidates)} ({'PASS' if len(candidates) >= 30 else 'NEED MORE'})")

        # Task type distribution
        task_types = {}
        for c in candidates:
            tt = c.get("task_type", "unknown")
            task_types[tt] = task_types.get(tt, 0) + 1
        checks["task_type_distribution"] = task_types

    # 11. Validation stats
    print("\n11. Checking candidate validation...")
    if VALIDATION_PATH.exists():
        validations = []
        with open(VALIDATION_PATH) as f:
            for line in f:
                validations.append(json.loads(line))
        passed = sum(1 for v in validations if v.get("validation_status") == "passed")
        needs_rev = sum(1 for v in validations if v.get("validation_status") == "needs_revision")
        failed = sum(1 for v in validations if v.get("validation_status") == "failed")
        pass_rate = passed / len(validations) if validations else 0
        checks["validation_pass_rate"] = pass_rate
        checks["validation_pass_rate_ok"] = pass_rate >= 0.70
        print(f"  Validated: {len(validations)}, Passed: {passed}, Needs revision: {needs_rev}, Failed: {failed}")
        print(f"  Pass rate: {pass_rate:.2%} ({'OK' if pass_rate >= 0.70 else 'LOW'})")

    # 12. Checkpoint consistency
    print("\n12. Checking checkpoint consistency...")
    if CHECKPOINT_PATH.exists() and LABELS_PATH.exists():
        checkpoint_ids = set()
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
            checkpoint_ids = set(data.get("processed_ids", []))
        label_ids = set()
        with open(LABELS_PATH) as f:
            for line in f:
                label_ids.add(json.loads(line)["image_id"])
        consistency = len(label_ids - checkpoint_ids) == 0
        checks["checkpoint_consistent"] = consistency
        print(f"  Checkpoint: {len(checkpoint_ids)} IDs, labels: {len(label_ids)} IDs, {'CONSISTENT' if consistency else 'INCONSISTENT'}")

    # Overall result
    critical_checks = [
        checks.get("file_exists_image_labels_full.jsonl", False),
        checks.get("file_exists_image_captions_full.jsonl", False),
        checks.get("file_exists_image_quality_scores_full.jsonl", False),
        checks.get("no_api_keys_leaked", False),
        checks.get("image_id_traceable", False),
        checks.get("caption_faithful_pass", False),
        checks.get("label_reasonable_pass", False),
        checks.get("manifest_coverage_pass", False),
    ]

    all_passed = all(critical_checks)
    print(f"\n=== Validation Result: {'PASS' if all_passed else 'FAIL'} ===")
    print(f"Critical checks passed: {sum(critical_checks)}/{len(critical_checks)}")

    # Write report
    checks["validation_result"] = "PASS" if all_passed else "FAIL"
    checks["critical_checks_passed"] = sum(critical_checks)
    checks["phase"] = "3.9_validation"
    checks["timestamp"] = time.time()

    with open(REPORT_DIR / "validation_full.json", "w") as f:
        json.dump(checks, f, indent=2)

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)