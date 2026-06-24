"""Phase 7: Run leakage check on SFT data.

Usage:
    python scripts/run_phase_7_leakage_check.py
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
    from src.autodata.finetuning.leakage_detector import LeakageDetector

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_finetuning_preparation"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark dev/test...")
    detector = LeakageDetector()
    dev_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_dev.jsonl"
    test_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    detector.load_benchmark(dev_path, test_path)

    print(f"  Benchmark IDs: {len(detector.benchmark_ids)}")
    print(f"  Benchmark questions: {len(detector.benchmark_questions)}")

    # Check SFT pools
    sft_dir = PROJECT_ROOT / "data" / "sft" / "pools"
    all_samples = []
    for pool_file in sft_dir.glob("*_sft.jsonl"):
        samples = load_jsonl(pool_file)
        all_samples.extend(samples)
        print(f"  {pool_file.name}: {len(samples)} samples")

    print(f"\nChecking {len(all_samples)} samples for leakage...")
    clean, leaked = detector.filter_samples(all_samples)

    print(f"  Clean: {len(clean)}")
    print(f"  Leaked: {len(leaked)}")
    print(f"  Leakage rate: {len(leaked)/max(len(all_samples),1):.1%}")

    # Save results
    save_jsonl(clean, PROJECT_ROOT / "data" / "sft" / "validated" / "sft_validated_all.jsonl")
    save_jsonl(leaked, PROJECT_ROOT / "data" / "sft" / "leakage_removed_samples.jsonl")

    # Leakage report
    report = {
        "total_checked": len(all_samples),
        "clean": len(clean),
        "leaked": len(leaked),
        "leakage_rate": len(leaked) / max(len(all_samples), 1),
        "leaked_examples": [
            {"sample_id": s.get("sample_id", ""), "reasons": s.get("_leakage_result", {}).get("reasons", [])}
            for s in leaked[:20]
        ],
    }
    with open(report_dir / "leakage_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(report_dir / "leakage_report.md", "w") as f:
        f.write("# Leakage Report\n\n")
        f.write(f"- Total checked: {len(all_samples)}\n")
        f.write(f"- Clean: {len(clean)}\n")
        f.write(f"- Leaked: {len(leaked)}\n")
        f.write(f"- Leakage rate: {len(leaked)/max(len(all_samples),1):.1%}\n")

    print("\nLeakage report saved.")


if __name__ == "__main__":
    main()
