"""Phase 7: Build SFT data pools.

Usage:
    python scripts/run_phase_7_build_sft_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.autodata.finetuning.sft_data_builder import SFTDataBuilder

    print("Building SFT data pools...")
    builder = SFTDataBuilder(PROJECT_ROOT)
    pools = builder.build_all_pools()

    sft_dir = PROJECT_ROOT / "data" / "sft" / "pools"
    counts = builder.save_pools(pools, sft_dir)

    total = sum(counts.values())
    print(f"\nSFT pools built:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print(f"  Total: {total}")


if __name__ == "__main__":
    main()
