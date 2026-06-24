"""List models for Phase 6 evaluation.

Usage:
    python scripts/run_phase_6_list_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.evaluation.model_registry import load_model_registry, save_model_registry


def main():
    models = load_model_registry()
    json_path, md_path = save_model_registry(models)

    print(f"=== Phase 6 Model Registry ===")
    print(f"Total models: {len(models)}\n")
    for m in models:
        print(f"  {m['display_name']} ({m['provider']}) - text_only={m['text_only']}, thinking={m['supports_thinking']}")
    print(f"\nJSON: {json_path}")
    print(f"MD: {md_path}")


if __name__ == "__main__":
    main()
