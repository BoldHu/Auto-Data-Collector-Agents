"""Phase 7: Split SFT data into train/validation.

Usage:
    python scripts/run_phase_7_split_sft_data.py
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
    from src.autodata.finetuning.sft_splitter import split_samples, save_splits

    validated_path = PROJECT_ROOT / "data" / "sft" / "validated" / "sft_validated_all.jsonl"
    samples = load_jsonl(validated_path)
    print(f"Loaded {len(samples)} validated samples")

    train, val = split_samples(samples, train_ratio=0.9)
    print(f"Split: train={len(train)}, val={len(val)}")

    stats = save_splits(train, val, PROJECT_ROOT / "data" / "sft" / "final")
    print(f"\nStatistics:")
    print(f"  Train source types: {stats['train']['source_type_dist']}")
    print(f"  Val source types: {stats['validation']['source_type_dist']}")
    print(f"\nFiles saved to data/sft/final/")


if __name__ == "__main__":
    main()
