"""Phase 7: LoRA fine-tuning (wrapper script).

Usage:
    python scripts/run_phase_7_train_lora.py \
        --config configs/finetuning/lora_default.yaml \
        --train_file data/sft/final/train_chatml.jsonl \
        --validation_file data/sft/final/validation_chatml.jsonl \
        --base_model <model_path> \
        --run_training false
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Delegate to the training module
if __name__ == "__main__":
    from src.autodata.finetuning.train_lora import main
    main()
