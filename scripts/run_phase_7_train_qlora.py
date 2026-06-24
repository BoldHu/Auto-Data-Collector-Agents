"""Phase 7: QLoRA fine-tuning (wrapper script).

Usage:
    python scripts/run_phase_7_train_qlora.py \
        --config configs/finetuning/qlora_default.yaml \
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

# QLoRA uses the same training script with quantization config
if __name__ == "__main__":
    from src.autodata.finetuning.train_lora import main
    main()
