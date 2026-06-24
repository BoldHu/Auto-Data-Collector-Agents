"""LoRA fine-tuning script for Phase 7.

Full training is disabled by default. Use --run_training true to enable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train ChatML JSONL")
    parser.add_argument("--validation_file", type=str, default="", help="Path to val ChatML JSONL")
    parser.add_argument("--base_model", type=str, default="", help="Override base model path")
    parser.add_argument("--run_training", type=str, default="false", help="Enable full training")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples for dry run")
    args = parser.parse_args()

    from src.autodata.finetuning.training_config import FinetuningConfig

    config = FinetuningConfig.from_yaml(Path(args.config))
    if args.base_model:
        config.base_model = args.base_model

    run_training = args.run_training.lower() == "true"

    print(f"Config: {args.config}")
    print(f"Base model: {config.base_model}")
    print(f"Train file: {args.train_file}")
    print(f"Run training: {run_training}")
    print(f"Max samples: {args.max_samples}")

    # Check if model is available
    if not config.base_model:
        print("\nWARNING: No base model specified. Set --base_model or base_model in config.")
        print("Skipping training. Pipeline validation only.")
        _validate_pipeline(args.train_file, args.max_samples)
        return

    # Try to load model
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"\nLoading tokenizer from {config.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model from {config.base_model}...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if not run_training:
            print("\nDry run mode. Validating one forward pass...")
            _dry_run_forward(model, tokenizer, args.train_file, args.max_samples or 4)
            print("Dry run complete. Pass --run_training true to start training.")
            return

        # Full training
        print("\nStarting full training...")
        _run_training(model, tokenizer, config, args.train_file, args.validation_file)

    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install: pip install transformers peft accelerate bitsandbytes")
        _validate_pipeline(args.train_file, args.max_samples)
    except Exception as e:
        print(f"\nError: {e}")
        _validate_pipeline(args.train_file, args.max_samples)


def _validate_pipeline(train_file: str, max_samples: int):
    """Validate training data pipeline without model."""
    import json

    print("\n=== Pipeline Validation ===")
    samples = []
    with open(train_file) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    if samples:
        sample = samples[0]
        messages = sample.get("messages", [])
        print(f"Sample messages: {len(messages)}")
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")[:100]
            print(f"  {role}: {content}...")

    print("Pipeline validation passed.")


def _dry_run_forward(model, tokenizer, train_file: str, max_samples: int):
    """Run one forward pass to verify loss computation."""
    import json
    import torch

    samples = []
    with open(train_file) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if line.strip():
                samples.append(json.loads(line))

    model.train()
    total_loss = 0
    for sample in samples:
        messages = sample.get("messages", [])
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
            text = "\n".join(parts)

        encoding = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        print(f"  Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(len(samples), 1)
    print(f"Average loss: {avg_loss:.4f}")


def _run_training(model, tokenizer, config, train_file: str, val_file: str):
    """Run full training."""
    from transformers import TrainingArguments, Trainer
    from src.autodata.finetuning.dataset_loader import SFTDataset

    train_dataset = SFTDataset(Path(train_file), tokenizer, config.training.max_seq_length)
    val_dataset = SFTDataset(Path(val_file), tokenizer, config.training.max_seq_length) if val_file else None

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        fp16=config.training.mixed_precision == "fp16",
        bf16=config.training.mixed_precision == "bf16",
        gradient_checkpointing=config.training.gradient_checkpointing,
        seed=config.training.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
