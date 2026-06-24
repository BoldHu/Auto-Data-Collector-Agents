"""Phase 7.5: Validate all outputs.

Usage:
    python scripts/validate_phase_7_5_sft_expansion.py
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


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_5_sft_expansion"
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

    # 1. Phase 7 statistics audit
    check("Statistics audit exists", (report_dir / "phase7_statistics_audit.json").exists())

    # 2. Embedding leakage report
    check("Embedding leakage report exists", (report_dir / "embedding_leakage_report.json").exists())

    # 3. Source pool exists
    pool_path = sft_dir / "source_pool" / "sft_source_pool.jsonl"
    check("Source pool exists", pool_path.exists())
    if pool_path.exists():
        pool = load_jsonl(pool_path)
        check("Source pool has samples", len(pool) > 0, f"{len(pool)} sources")

    # 4. Expanded candidates exist
    expanded_path = sft_dir / "expanded" / "sft_expanded_candidates.jsonl"
    check("Expanded candidates exist", expanded_path.exists())
    if expanded_path.exists():
        expanded = load_jsonl(expanded_path)
        check("Expanded has samples", len(expanded) > 0, f"{len(expanded)} candidates")

    # 5. Expanded validation
    check("Expanded validated exists", (sft_dir / "expanded" / "sft_expanded_validated.jsonl").exists())

    # 6. Final v2 files
    final_v2 = sft_dir / "final_v2"
    check("Final v2 train exists", (final_v2 / "train.jsonl").exists())
    check("Final v2 validation exists", (final_v2 / "validation.jsonl").exists())
    check("Final v2 train ChatML exists", (final_v2 / "train_chatml.jsonl").exists())
    check("Final v2 validation ChatML exists", (final_v2 / "validation_chatml.jsonl").exists())

    # 7. Sample counts
    if (final_v2 / "train.jsonl").exists():
        train = load_jsonl(final_v2 / "train.jsonl")
        val = load_jsonl(final_v2 / "validation.jsonl")
        total = len(train) + len(val)
        check("Final v2 has >= 1500 samples", total >= 1500, f"{total} total")
        check("Train/val split reasonable", len(val) >= 10, f"val={len(val)}")

        # Check ChatML matches
        train_chatml = load_jsonl(final_v2 / "train_chatml.jsonl")
        val_chatml = load_jsonl(final_v2 / "validation_chatml.jsonl")
        check("ChatML counts match", len(train) == len(train_chatml) and len(val) == len(val_chatml))

    # 8. No benchmark leakage in final
    if (final_v2 / "train.jsonl").exists():
        bench_ids = set()
        for fname in ["carbon_fiber_benchmark_dev.jsonl", "carbon_fiber_benchmark_test.jsonl"]:
            for item in load_jsonl(PROJECT_ROOT / "data" / "benchmark" / fname):
                bench_ids.add(item.get("benchmark_id", ""))

        train_ids = set(s.get("benchmark_id", "") for s in load_jsonl(final_v2 / "train.jsonl"))
        overlap = train_ids & bench_ids
        check("No benchmark ID leakage", len(overlap) == 0, f"{len(overlap)} overlaps")

    # 9. Scaling subsets
    subsets_dir = final_v2 / "subsets"
    check("Scaling subset train_100 exists", (subsets_dir / "train_100.jsonl").exists())
    check("Scaling subset train_500 exists", (subsets_dir / "train_500.jsonl").exists())
    check("Scaling subset train_1000 exists", (subsets_dir / "train_1000.jsonl").exists())

    # 10. Training configs v2
    configs_dir = PROJECT_ROOT / "configs" / "finetuning"
    check("LoRA v2 config exists", (configs_dir / "lora_v2.yaml").exists())
    check("QLoRA v2 config exists", (configs_dir / "qlora_v2.yaml").exists())

    # 11. Dataset card
    check("Dataset card v2 exists", (final_v2 / "SFT_DATASET_CARD.md").exists())

    # 12. No API keys
    import re
    api_key_pattern = re.compile(r'(sk-|ak-|api[_-]?key\s*[:=]\s*\S+)', re.IGNORECASE)
    key_found = False
    for fpath in report_dir.glob("*.json"):
        content = fpath.read_text()
        if api_key_pattern.search(content):
            key_found = True
    check("No API keys in outputs", not key_found)

    # Print results
    print("=== Phase 7.5 Validation ===\n")
    for c in checks:
        print(c)
    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    # Save
    with open(report_dir / "validation_phase_7_5.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)

    if failed == 0:
        print("\nAll checks passed!")
    else:
        print(f"\n{failed} check(s) failed.")


if __name__ == "__main__":
    main()
