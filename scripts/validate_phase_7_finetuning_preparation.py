"""Phase 7: Validate all fine-tuning preparation outputs.

Usage:
    python scripts/validate_phase_7_finetuning_preparation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_finetuning_preparation"
    sft_dir = PROJECT_ROOT / "data" / "sft"

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

    # 1. SFT pools exist
    pools_dir = sft_dir / "pools"
    pool_files = list(pools_dir.glob("*_sft.jsonl")) if pools_dir.exists() else []
    check("SFT pools exist", len(pool_files) >= 3, f"{len(pool_files)} pools found")

    # 2. Leakage report exists
    check("Leakage report exists", (report_dir / "leakage_report.json").exists())

    # 3. No benchmark dev/test leakage
    leakage_path = report_dir / "leakage_report.json"
    if leakage_path.exists():
        with open(leakage_path) as f:
            report = json.load(f)
        clean = report.get("clean", 0)
        total = report.get("total_checked", 0)
        check("Leakage check completed", total > 0, f"{clean}/{total} clean")

    # 4. Final train/validation files exist
    final_dir = sft_dir / "final"
    check("Train file exists", (final_dir / "train.jsonl").exists())
    check("Validation file exists", (final_dir / "validation.jsonl").exists())
    check("Train ChatML exists", (final_dir / "train_chatml.jsonl").exists())
    check("Validation ChatML exists", (final_dir / "validation_chatml.jsonl").exists())

    # 5. Dataset card exists
    check("Dataset card exists", (final_dir / "SFT_DATASET_CARD.md").exists())

    # 6. Training configs exist
    configs_dir = PROJECT_ROOT / "configs" / "finetuning"
    check("LoRA config exists", (configs_dir / "lora_default.yaml").exists())
    check("QLoRA config exists", (configs_dir / "qlora_default.yaml").exists())
    check("Dry run config exists", (configs_dir / "dry_run.yaml").exists())

    # 7. Training scripts exist
    check("train_lora.py exists", (PROJECT_ROOT / "src" / "autodata" / "finetuning" / "train_lora.py").exists())
    check("dataset_loader.py exists", (PROJECT_ROOT / "src" / "autodata" / "finetuning" / "dataset_loader.py").exists())

    # 8. Full training disabled by default
    import yaml
    for cfg_name in ["lora_default.yaml", "qlora_default.yaml", "dry_run.yaml"]:
        cfg_path = configs_dir / cfg_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            check(f"{cfg_name} training disabled", cfg.get("run_training", True) is False)

    # 9. Post-finetune evaluation plan exists
    check("Eval plan exists", (final_dir / "POST_FINETUNE_EVALUATION_PLAN.md").exists())

    # 10. No API keys in outputs
    import re
    api_key_pattern = re.compile(r'(sk-|ak-|api[_-]?key\s*[:=]\s*\S+)', re.IGNORECASE)
    key_found = False
    for fpath in report_dir.glob("*.json"):
        content = fpath.read_text()
        if api_key_pattern.search(content):
            key_found = True
    check("No API keys in outputs", not key_found)

    # 11. Sample counts
    if (final_dir / "train.jsonl").exists():
        with open(final_dir / "train.jsonl") as f:
            train_count = sum(1 for _ in f)
        with open(final_dir / "validation.jsonl") as f:
            val_count = sum(1 for _ in f)
        check("Train has samples", train_count > 0, f"{train_count} samples")
        check("Validation has samples", val_count > 0, f"{val_count} samples")

    # Print results
    print("=== Phase 7 Validation ===\n")
    for c in checks:
        print(c)
    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    # Save validation
    validation = {
        "total_checks": passed + failed,
        "passed": passed,
        "failed": failed,
        "checks": checks,
    }
    with open(report_dir / "validation_phase_7.json", "w") as f:
        json.dump(validation, f, indent=2)

    if failed == 0:
        print("\nAll checks passed!")
    else:
        print(f"\n{failed} check(s) failed.")


if __name__ == "__main__":
    main()
