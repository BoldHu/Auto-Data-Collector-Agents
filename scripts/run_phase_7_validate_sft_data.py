"""Phase 7: Validate SFT data quality.

Usage:
    python scripts/run_phase_7_validate_sft_data.py
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
                if line.strip():
                    records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    from src.autodata.finetuning.sft_quality_filter import filter_samples, deduplicate_samples
    from src.autodata.finetuning.sft_validator import validate_batch

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_finetuning_preparation"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load validated samples (after leakage check)
    validated_path = PROJECT_ROOT / "data" / "sft" / "validated" / "sft_validated_all.jsonl"
    samples = load_jsonl(validated_path)
    print(f"Loaded {len(samples)} samples after leakage check")

    # Deduplicate
    deduped = deduplicate_samples(samples)
    print(f"After dedup: {len(deduped)} (removed {len(samples) - len(deduped)})")

    # Quality filter
    passed, rejected_filter = filter_samples(deduped)
    print(f"Quality filter: passed={len(passed)}, rejected={len(rejected_filter)}")

    # Validator
    validated, rejected_validate = validate_batch(passed)
    print(f"Validation: passed={len(validated)}, rejected={len(rejected_validate)}")

    # Save
    save_jsonl(validated, PROJECT_ROOT / "data" / "sft" / "validated" / "sft_validated_all.jsonl")
    save_jsonl(rejected_filter + rejected_validate, PROJECT_ROOT / "data" / "sft" / "validated" / "sft_rejected.jsonl")

    # Report
    report = {
        "input_samples": len(samples),
        "after_dedup": len(deduped),
        "quality_passed": len(passed),
        "validation_passed": len(validated),
        "total_rejected": len(rejected_filter) + len(rejected_validate),
    }
    with open(report_dir / "sft_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal: {len(validated)} validated samples")
    print("Validation report saved.")


if __name__ == "__main__":
    main()
