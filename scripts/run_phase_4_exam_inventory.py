"""Run Phase 4 exam data inventory.

Usage:
    python scripts/run_phase_4_exam_inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.exam_inventory import build_inventory, save_inventory
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("exam_inventory")

EXAM_DIR = PROJECT_ROOT / "exam_raw_data"


def main():
    logger.info(f"Scanning exam directory: {EXAM_DIR}")
    inventory = build_inventory(EXAM_DIR)
    json_path, md_path = save_inventory(inventory)

    print(f"\n=== Exam Inventory Complete ===")
    print(f"Total files: {len(inventory)}")

    types = {}
    for item in inventory:
        t = item["file_type"]
        types[t] = types.get(t, 0) + 1
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")

    scanned = sum(1 for item in inventory if item["scanned"])
    answers = sum(1 for item in inventory if item["has_answer_key"])
    print(f"  Scanned: {scanned}")
    print(f"  With answers: {answers}")

    print(f"\nJSON: {json_path}")
    print(f"MD: {md_path}")


if __name__ == "__main__":
    main()
