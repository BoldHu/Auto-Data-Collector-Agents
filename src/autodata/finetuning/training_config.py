"""Training configuration loader for Phase 7."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    learning_rate: float = 2.0e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    num_epochs: int = 3
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass
class QLoRAConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class FinetuningConfig:
    base_model: str = ""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    qlora: Optional[QLoRAConfig] = None
    output_dir: str = "data/finetuning_outputs/default"
    run_name: str = "cf_default"
    run_training: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "FinetuningConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()
        config.base_model = data.get("base_model", "")
        config.output_dir = data.get("output_dir", config.output_dir)
        config.run_name = data.get("run_name", config.run_name)
        config.run_training = data.get("run_training", False)

        if "lora" in data:
            for k, v in data["lora"].items():
                if hasattr(config.lora, k):
                    setattr(config.lora, k, v)

        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        if "qlora" in data:
            config.qlora = QLoRAConfig(**data["qlora"])

        return config
