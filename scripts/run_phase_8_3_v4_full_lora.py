"""Phase 8.3: SFT v4 full LoRA training and evaluation.

Usage:
    python scripts/run_phase_8_3_v4_full_lora.py \
        --model_path models/qwen/Qwen2.5-VL-3B-Instruct \
        --epochs 2 \
        --batch_size 1 \
        --grad_accum 8 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 8.3 v4 full LoRA")
    parser.add_argument("--model_path", type=str, default="models/qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_3_v4_full_lora"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_v4_full_lora"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_3.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.3: V4 Full LoRA ===")

    # Step 1: Preflight
    log("Step 1: Preflight checks...")
    preflight = _run_preflight(args.model_path, log)
    save_json(preflight, report_dir / "preflight_phase_8_3.json")

    if preflight.get("status") != "pass":
        log("Preflight failed, stopping.")
        return

    # Step 2: Load frozen baseline scores
    log("Step 2: Loading frozen baseline scores...")
    baseline = _load_baseline_scores(log)
    save_json(baseline, report_dir / "frozen_baseline_scores.json")

    # Step 3: Train v4 full LoRA
    v4_adapter = PROJECT_ROOT / "outputs" / "phase_8_3_v4_full_lora" / "lora_v4_full"
    if not args.skip_training:
        log("Step 3: Training v4 full LoRA...")
        training_results = _train_v4_full(args, log)
        save_json(training_results, report_dir / "lora_v4_full_training_summary.json")
    else:
        log("Step 3: Skipping training")
        training_results = {"status": "skipped"}

    # Step 4: Evaluate v4 full on canonical 150 items
    if v4_adapter.exists():
        log("Step 4: Evaluating v4 full adapter...")
        v4_results = _evaluate_v4_full(args.model_path, str(v4_adapter), args, log)
        save_json(v4_results, report_dir / "v4_full_eval_report.json")
    else:
        log("Step 4: Skipping evaluation (adapter not found)")
        v4_results = {"status": "skipped"}

    # Step 5: Unified comparison
    log("Step 5: Unified comparison...")
    comparison = _unified_comparison(baseline, v4_results, log)
    save_json(comparison, report_dir / "unified_comparison_report.json")

    # Step 6: Generate report
    log("Step 6: Generating report...")
    _generate_report(preflight, baseline, training_results, v4_results, comparison, report_dir, log)

    log("=== Phase 8.3 Complete ===")


def _run_preflight(model_path: str, log) -> dict:
    """Run preflight checks."""
    checks = {
        "model_exists": os.path.exists(model_path),
        "canonical_manifest_exists": (PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "canonical_eval_manifest_150.jsonl").exists(),
        "baseline_scores_exist": (PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "base_vs_all_adapters_150.csv").exists(),
        "v4_train_exists": (PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "train_chatml.jsonl").exists(),
        "v4_val_exists": (PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "validation_chatml.jsonl").exists(),
        "gold100_adapter_exists": (PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100").exists(),
        "goldfull_adapter_exists": (PROJECT_ROOT / "outputs" / "phase_8_1b_gold_full_lora" / "lora_gold_full").exists(),
        "formataligned_adapter_exists": (PROJECT_ROOT / "outputs" / "phase_8_2_lora_degradation_debug" / "lora_format_aligned_200").exists(),
    }

    all_pass = all(checks.values())
    checks["status"] = "pass" if all_pass else "fail"

    for k, v in checks.items():
        if k != "status":
            log(f"  {k}: {'PASS' if v else 'FAIL'}")

    return checks


def _load_baseline_scores(log) -> dict:
    """Load frozen baseline scores from Phase 8.2.5."""
    csv_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "base_vs_all_adapters_150.csv"

    scores = {}
    if csv_path.exists():
        with open(csv_path) as f:
            header = None
            for line in f:
                parts = line.strip().split(",")
                if header is None:
                    header = parts
                    continue
                if len(parts) >= 5:
                    scores[parts[0]] = {
                        "total": int(parts[1]),
                        "scored": int(parts[2]),
                        "correct": int(parts[3]),
                        "accuracy": float(parts[4]),
                        "vs_base": float(parts[5]) if len(parts) > 5 else 0,
                    }

    log(f"  Loaded baseline scores: {list(scores.keys())}")
    return scores


def _train_v4_full(args, log) -> dict:
    """Train v4 full LoRA."""
    import torch

    model_path = args.model_path
    train_file = str(PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "train_chatml.jsonl")
    val_file = str(PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "validation_chatml.jsonl")
    output_dir = str(PROJECT_ROOT / "outputs" / "phase_8_3_v4_full_lora" / "lora_v4_full")

    result = {
        "model_path": model_path,
        "train_file": train_file,
        "val_file": val_file,
        "output_dir": output_dir,
        "status": "unknown",
    }

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType

        log("  Loading model...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        # LoRA config
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=target_modules, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()

        result["trainable_parameters"] = trainable
        result["total_parameters"] = total
        result["trainable_ratio"] = round(trainable / total, 4)

        log(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

        # Load dataset
        train_samples = load_jsonl(Path(train_file))
        val_samples = load_jsonl(Path(val_file))

        result["train_count"] = len(train_samples)
        result["val_count"] = len(val_samples)

        log(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

        # Create dataset
        class SimpleDataset:
            def __init__(self, samples, processor, max_length):
                self.samples = samples
                self.processor = processor
                self.max_length = max_length

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                messages = sample.get("messages", [])
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                encoding = self.processor.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
                input_ids = encoding["input_ids"].squeeze()
                attention_mask = encoding["attention_mask"].squeeze()
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        train_dataset = SimpleDataset(train_samples, processor, args.max_seq_length)
        val_dataset = SimpleDataset(val_samples, processor, args.max_seq_length)

        # Training arguments - disable gradient_checkpointing for LoRA compatibility
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            save_steps=50,
            eval_steps=50,
            logging_steps=10,
            bf16=True,
            gradient_checkpointing=False,  # Disabled for LoRA compatibility
            report_to="none",
            save_total_limit=2,
            remove_unused_columns=False,
            max_grad_norm=1.0,
        )

        # Custom trainer
        class LoggingTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.train_losses = []
                self.val_losses = []

            def log(self, logs, *args, **kwargs):
                super().log(logs, *args, **kwargs)
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                if "eval_loss" in logs:
                    self.val_losses.append(logs["eval_loss"])

        trainer = LoggingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        log("  Starting training...")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        result["status"] = "success"
        result["training_time_seconds"] = round(training_time, 1)
        result["train_losses"] = trainer.train_losses
        result["val_losses"] = trainer.val_losses
        result["final_train_loss"] = trainer.train_losses[-1] if trainer.train_losses else None

        # Save adapter
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        result["adapter_saved"] = True

        log(f"  Training complete: {training_time:.0f}s")
        if trainer.train_losses:
            log(f"  Initial loss: {trainer.train_losses[0]:.4f}")
            log(f"  Final loss: {trainer.train_losses[-1]:.4f}")

        del model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log(f"  Training error: {str(e)[:100]}")

    return result


def _evaluate_v4_full(model_path: str, adapter_path: str, args, log) -> dict:
    """Evaluate v4 full adapter on canonical 150 items."""
    import torch

    eval_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "canonical_eval_manifest_150.jsonl"
    if not eval_path.exists():
        return {"status": "skipped", "reason": "eval_manifest_not_found"}

    eval_items = load_jsonl(eval_path)

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, adapter_path)

        log(f"  Evaluating {len(eval_items)} items...")

        correct = 0
        scored = 0
        outputs = []

        for item in eval_items:
            question = item.get("question", "")
            options = item.get("options", [])
            gold = str(item.get("answer", "")).strip()
            expected_format = item.get("expected_answer_format", "open_ended")

            # Build prompt
            if expected_format == "multiple_choice_letter" and options:
                opt_text = "\n".join(str(o) for o in options)
                prompt = f"{question}\n\n选项：\n{opt_text}\n\n请只输出选项字母（如A、B、C、D）。"
            else:
                prompt = f"{question}\n\n请直接回答。"

            try:
                messages = [{"role": "user", "content": prompt}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    gen_outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

                response = processor.decode(gen_outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

                # Score
                is_correct = _check_answer(response, gold, options, expected_format)
                if is_correct:
                    correct += 1
                scored += 1

                outputs.append({
                    "eval_index": item.get("eval_index"),
                    "benchmark_id": item.get("benchmark_id", ""),
                    "task_type": item.get("task_type", ""),
                    "expected_format": expected_format,
                    "gold": gold[:100],
                    "predicted": response[:100],
                    "correct": is_correct,
                    "status": "scored",
                })

            except Exception as e:
                outputs.append({
                    "eval_index": item.get("eval_index"),
                    "benchmark_id": item.get("benchmark_id", ""),
                    "status": "failed",
                    "error": str(e)[:100],
                })

        accuracy = correct / max(scored, 1)

        # Save outputs
        save_jsonl(outputs, PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_v4_full_lora" / "v4_full_outputs_150.jsonl")

        result = {
            "status": "success",
            "total": len(eval_items),
            "scored": scored,
            "correct": correct,
            "accuracy": round(accuracy, 4),
        }

        log(f"  V4 Full: {correct}/{scored} = {accuracy:.1%}")

        del model
        del processor
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        log(f"  Evaluation error: {str(e)[:100]}")
        return {"status": "error", "error": str(e)[:300]}


def _check_answer(response: str, gold: str, options: list, expected_format: str) -> bool:
    """Check if response matches gold answer."""
    if expected_format == "multiple_choice_letter" and options:
        pred_match = re.search(r'([A-H])', response.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            return pred_match.group(1) == gold_match.group(1)

    # Keyword match
    gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold[:200]))
    resp_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', response[:200]))
    if gold_keywords and len(gold_keywords & resp_keywords) / len(gold_keywords) > 0.3:
        return True

    return False


def _unified_comparison(baseline: dict, v4_results: dict, log) -> dict:
    """Unified comparison of all adapters."""
    comparison = {
        "models": dict(baseline),
    }

    if v4_results.get("status") == "success":
        comparison["models"]["v4_full"] = {
            "total": v4_results.get("total", 0),
            "scored": v4_results.get("scored", 0),
            "correct": v4_results.get("correct", 0),
            "accuracy": v4_results.get("accuracy", 0),
        }

    # Calculate vs_base for v4_full
    base_acc = baseline.get("base", {}).get("accuracy", 0)
    if "v4_full" in comparison["models"]:
        comparison["models"]["v4_full"]["vs_base"] = comparison["models"]["v4_full"]["accuracy"] - base_acc

    # Find best adapter
    best_name = "base"
    best_acc = base_acc
    for name, data in comparison["models"].items():
        if data.get("accuracy", 0) > best_acc:
            best_acc = data["accuracy"]
            best_name = name

    comparison["best_adapter"] = best_name
    comparison["best_accuracy"] = best_acc

    # Save CSV
    csv_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_v4_full_lora" / "base_vs_all_adapters_with_v4.csv"
    with open(csv_path, "w") as f:
        f.write("Model,Total,Scored,Correct,Accuracy,vs_Base\n")
        for name, data in comparison["models"].items():
            vs_base = data.get("accuracy", 0) - base_acc
            f.write(f"{name},{data.get('total',0)},{data.get('scored',0)},{data.get('correct',0)},{data.get('accuracy',0):.4f},{vs_base:+.4f}\n")

    log(f"  Best adapter: {best_name} ({best_acc:.1%})")
    return comparison


def _generate_report(preflight, baseline, training_results, v4_results, comparison, report_dir, log):
    """Generate Phase 8.3 report."""
    md = "# Phase 8.3: V4 Full LoRA Report\n\n"

    md += "## 1. Preflight\n\n"
    md += f"- Status: {preflight.get('status', 'unknown')}\n\n"

    md += "## 2. Frozen Baseline Scores\n\n"
    md += "| Model | Accuracy | vs Base |\n|-------|----------|--------|\n"
    for name, data in baseline.items():
        md += f"| {name} | {data.get('accuracy', 0):.1%} | {data.get('vs_base', 0):+.1%} |\n"
    md += "\n"

    md += "## 3. V4 Full Training\n\n"
    md += f"- Status: {training_results.get('status', 'unknown')}\n"
    md += f"- Train samples: {training_results.get('train_count', 0)}\n"
    md += f"- Val samples: {training_results.get('val_count', 0)}\n"
    if training_results.get("train_losses"):
        md += f"- Initial loss: {training_results['train_losses'][0]:.4f}\n"
        md += f"- Final loss: {training_results['train_losses'][-1]:.4f}\n"
    if training_results.get("training_time_seconds"):
        md += f"- Training time: {training_results['training_time_seconds']:.0f}s\n"
    md += "\n"

    md += "## 4. V4 Full Evaluation\n\n"
    md += f"- Status: {v4_results.get('status', 'unknown')}\n"
    md += f"- Total: {v4_results.get('total', 0)}\n"
    md += f"- Correct: {v4_results.get('correct', 0)}\n"
    md += f"- Accuracy: {v4_results.get('accuracy', 0):.1%}\n\n"

    md += "## 5. Unified Comparison\n\n"
    md += "| Model | Accuracy | vs Base |\n|-------|----------|--------|\n"
    for name, data in comparison.get("models", {}).items():
        vs_base = data.get("accuracy", 0) - baseline.get("base", {}).get("accuracy", 0)
        md += f"| {name} | {data.get('accuracy', 0):.1%} | {vs_base:+.1%} |\n"
    md += f"\n**Best adapter**: {comparison.get('best_adapter', 'unknown')} ({comparison.get('best_accuracy', 0):.1%})\n"

    with open(report_dir / "PHASE_8_3_REPORT.md", "w") as f:
        f.write(md)

    log("  Report generated")


if __name__ == "__main__":
    main()
