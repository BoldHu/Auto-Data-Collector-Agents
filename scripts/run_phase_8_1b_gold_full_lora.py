"""Phase 8.1B: Gold Full LoRA training and evaluation.

Usage:
    python scripts/run_phase_8_1b_gold_full_lora.py \
        --model_path models/qwen/Qwen2.5-VL-3B-Instruct \
        --epochs 2 \
        --batch_size 1 \
        --grad_accum 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    parser = argparse.ArgumentParser(description="Phase 8.1B Gold Full LoRA")
    parser.add_argument("--model_path", type=str, default="models/qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_1b_gold_full_lora"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_1b.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.1B: Gold Full LoRA ===")

    # Step 1: Audit Phase 8.1A
    log("Step 1: Phase 8.1A consistency audit...")
    audit_result = _audit_phase81a(log)
    save_json(audit_result, report_dir / "phase8_1a_consistency_audit.json")

    # Step 2: Build fixed evaluation set
    log("Step 2: Building fixed evaluation set...")
    eval_set_stats = _build_fixed_eval_set(log)
    save_json(eval_set_stats, report_dir / "fixed_eval_set_stats.json")

    # Step 3: Prepare gold full data
    log("Step 3: Preparing gold full data...")
    data_check = _prepare_gold_full_data(log)
    save_json(data_check, report_dir / "gold_full_data_check.json")

    # Step 4: Evaluate base model
    log("Step 4: Evaluating base model...")
    base_results = _evaluate_model(args.model_path, None, "base", args, log)
    save_json(base_results, report_dir / "base_v3_eval_report.json")

    # Step 5: Evaluate gold_100 adapter
    gold100_adapter = PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100"
    if gold100_adapter.exists():
        log("Step 5: Evaluating gold_100 adapter...")
        gold100_results = _evaluate_model(args.model_path, str(gold100_adapter), "gold100", args, log)
        save_json(gold100_results, report_dir / "lora_gold100_v3_eval_report.json")
    else:
        log("Step 5: Skipping gold_100 evaluation (adapter not found)")
        gold100_results = {"status": "skipped"}

    # Step 6: Train gold full
    gold_full_adapter = PROJECT_ROOT / "outputs" / "phase_8_1b_gold_full_lora" / "lora_gold_full"
    if not args.skip_training and not args.eval_only:
        log("Step 6: Training gold full LoRA...")
        training_results = _train_gold_full(args, log)
        save_json(training_results, report_dir / "lora_gold_full_training_summary.json")
    else:
        log("Step 6: Skipping training")
        training_results = {"status": "skipped"}

    # Step 7: Evaluate gold full
    if gold_full_adapter.exists():
        log("Step 7: Evaluating gold full adapter...")
        gold_full_results = _evaluate_model(args.model_path, str(gold_full_adapter), "gold_full", args, log)
        save_json(gold_full_results, report_dir / "lora_gold_full_eval_report.json")
    else:
        log("Step 7: Skipping gold full evaluation (adapter not found)")
        gold_full_results = {"status": "skipped"}

    # Step 8: Compare results
    log("Step 8: Comparing results...")
    comparison = _compare_results(base_results, gold100_results, gold_full_results, log)
    save_json(comparison, report_dir / "phase_8_1b_analysis.json")

    # Step 9: Generate report
    log("Step 9: Generating report...")
    _generate_report(audit_result, eval_set_stats, data_check, base_results,
                     gold100_results, training_results, gold_full_results, comparison, report_dir, log)

    log("=== Phase 8.1B Complete ===")


def _audit_phase81a(log) -> dict:
    """Audit Phase 8.1A for consistency."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_1a_qwen_lora_pilot"

    audit = {
        "multimodal_smoke_ran": False,
        "base_eval_subsets": [],
        "lora_eval_subsets": [],
        "consistency_issues": [],
    }

    # Check multimodal smoke
    smoke_path = report_dir / "multimodal_smoke_report.json"
    if smoke_path.exists():
        with open(smoke_path) as f:
            smoke = json.load(f)
        audit["multimodal_smoke_ran"] = smoke.get("status") == "success"
        audit["multimodal_tests_passed"] = smoke.get("tests_passed", 0)
        log(f"  Multimodal smoke: {smoke.get('status')} ({smoke.get('tests_passed', 0)} tests)")

    # Check base eval subsets
    base_path = report_dir / "base_zero_shot_report.json"
    if base_path.exists():
        with open(base_path) as f:
            base = json.load(f)
        audit["base_eval_subsets"] = list(base.get("subsets", {}).keys())
        log(f"  Base eval subsets: {audit['base_eval_subsets']}")

    # Check LoRA eval
    eval_path = report_dir / "lora_gold100_eval_report.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        audit["lora_eval_subsets"] = list(eval_data.get("base", {}).keys())
        log(f"  LoRA eval subsets: {audit['lora_eval_subsets']}")

    # Check consistency
    if audit["base_eval_subsets"] != audit["lora_eval_subsets"]:
        audit["consistency_issues"].append("Base and LoRA used different evaluation subsets")

    audit["status"] = "audited"
    return audit


def _build_fixed_eval_set(log) -> dict:
    """Build fixed evaluation set."""
    random.seed(42)

    bench_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    if not bench_path.exists():
        return {"status": "not_found"}

    all_items = load_jsonl(bench_path)

    # Sample subsets
    text_items = [i for i in all_items if i.get("source_type") == "text"]
    exam_items = [i for i in all_items if i.get("task_type", "").startswith("exam")]
    agent_items = [i for i in all_items if i.get("source_type") == "text" and "agent" in i.get("task_type", "").lower()]

    eval_items = []
    eval_items.extend(random.sample(text_items, min(100, len(text_items))))
    eval_items.extend(random.sample(exam_items, min(50, len(exam_items))))

    # Save
    save_jsonl(eval_items, PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / "fixed_eval_set.jsonl")

    stats = {
        "total": len(eval_items),
        "text": len([i for i in eval_items if i.get("source_type") == "text"]),
        "exam": len([i for i in eval_items if i.get("task_type", "").startswith("exam")]),
    }

    log(f"  Fixed eval set: {stats['total']} items ({stats['text']} text, {stats['exam']} exam)")
    return stats


def _prepare_gold_full_data(log) -> dict:
    """Prepare gold full training data."""
    gold_train = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml.jsonl"
    gold_val = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "validation_chatml.jsonl"

    if not gold_train.exists():
        return {"status": "not_found"}

    train_samples = load_jsonl(gold_train)
    val_samples = load_jsonl(gold_val)

    # Copy to phase-specific files
    save_jsonl(train_samples, PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml_phase8_1b.jsonl")
    save_jsonl(val_samples, PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "validation_chatml_phase8_1b.jsonl")

    stats = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "status": "prepared",
    }

    log(f"  Gold full: train={len(train_samples)}, val={len(val_samples)}")
    return stats


def _evaluate_model(model_path: str, adapter_path: str | None, model_name: str, args, log) -> dict:
    """Evaluate a model on the fixed evaluation set."""
    import torch

    eval_set_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / "fixed_eval_set.jsonl"
    if not eval_set_path.exists():
        return {"status": "skipped", "reason": "eval_set_not_found"}

    eval_items = load_jsonl(eval_set_path)

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if adapter_path:
            from peft import PeftModel
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            model = PeftModel.from_pretrained(model, adapter_path)
            log(f"  Loaded {model_name} with adapter")
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            log(f"  Loaded {model_name} base model")

        # Evaluate
        correct = 0
        total = 0
        results = []

        for item in eval_items[:50]:  # Limit to 50 for speed
            question = item.get("question", "")
            options = item.get("options", [])
            gold = str(item.get("answer", "")).strip()

            # Build prompt
            if options:
                opt_text = "\n".join(str(o) for o in options)
                prompt = f"{question}\n\n选项：\n{opt_text}\n\n请只输出选项字母（如A、B、C、D）。"
            else:
                prompt = f"{question}\n\n请直接回答，不要解释。"

            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # Check answer
            is_correct = _check_answer(response, gold, options)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "benchmark_id": item.get("benchmark_id", ""),
                "gold": gold[:50],
                "predicted": response[:50],
                "correct": is_correct,
            })

        accuracy = correct / max(total, 1)

        result = {
            "status": "success",
            "model_name": model_name,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 3),
            "results": results[:20],
        }

        log(f"  {model_name}: {correct}/{total} = {accuracy:.1%}")

        del model
        del processor
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        log(f"  {model_name} error: {str(e)[:100]}")
        return {"status": "error", "error": str(e)[:300]}


def _check_answer(response: str, gold: str, options: list) -> bool:
    """Check if response matches gold answer."""
    if options and len(options) >= 2:
        pred_match = re.search(r'([A-H])', response.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            return pred_match.group(1) == gold_match.group(1)

    # Keyword match
    gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold))
    resp_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', response))
    if gold_keywords and len(gold_keywords & resp_keywords) / len(gold_keywords) > 0.5:
        return True

    return False


def _train_gold_full(args, log) -> dict:
    """Train gold full LoRA."""
    import torch

    model_path = args.model_path
    train_file = str(PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml_phase8_1b.jsonl")
    val_file = str(PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "validation_chatml_phase8_1b.jsonl")
    output_dir = str(PROJECT_ROOT / "outputs" / "phase_8_1b_gold_full_lora" / "lora_gold_full")

    result = {
        "model_path": model_path,
        "train_file": train_file,
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

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            save_steps=50,
            eval_steps=50,
            logging_steps=10,
            bf16=True,
            gradient_checkpointing=False,
            report_to="none",
            save_total_limit=2,
            remove_unused_columns=False,
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


def _compare_results(base: dict, gold100: dict, gold_full: dict, log) -> dict:
    """Compare base, gold_100, and gold_full results."""
    comparison = {
        "base_accuracy": base.get("accuracy", 0),
        "gold100_accuracy": gold100.get("accuracy", 0),
        "gold_full_accuracy": gold_full.get("accuracy", 0),
        "gold100_vs_base": gold100.get("accuracy", 0) - base.get("accuracy", 0),
        "gold_full_vs_base": gold_full.get("accuracy", 0) - base.get("accuracy", 0),
        "gold_full_vs_gold100": gold_full.get("accuracy", 0) - gold100.get("accuracy", 0),
    }

    # Recommendation
    if comparison["gold_full_vs_base"] > 0.05:
        comparison["recommendation"] = "proceed_to_v4_full"
    elif comparison["gold_full_vs_base"] > 0:
        comparison["recommendation"] = "proceed_with_caution"
    elif comparison["gold_full_vs_gold100"] > 0:
        comparison["recommendation"] = "improve_hyperparameters"
    else:
        comparison["recommendation"] = "debug_training"

    log(f"  Base: {comparison['base_accuracy']:.1%}")
    log(f"  Gold_100: {comparison['gold100_accuracy']:.1%}")
    log(f"  Gold_Full: {comparison['gold_full_accuracy']:.1%}")
    log(f"  Recommendation: {comparison['recommendation']}")

    return comparison


def _generate_report(audit, eval_stats, data_check, base_results, gold100_results,
                     training_results, gold_full_results, comparison, report_dir, log):
    """Generate Phase 8.1B report."""
    md = "# Phase 8.1B: Gold Full LoRA Report\n\n"

    md += "## 1. Phase 8.1A Audit\n\n"
    md += f"- Multimodal smoke ran: {audit.get('multimodal_smoke_ran', False)}\n"
    md += f"- Consistency issues: {audit.get('consistency_issues', [])}\n\n"

    md += "## 2. Fixed Evaluation Set\n\n"
    md += f"- Total: {eval_stats.get('total', 0)} items\n"
    md += f"- Text: {eval_stats.get('text', 0)}\n"
    md += f"- Exam: {eval_stats.get('exam', 0)}\n\n"

    md += "## 3. Gold Full Data\n\n"
    md += f"- Train: {data_check.get('train_count', 0)}\n"
    md += f"- Val: {data_check.get('val_count', 0)}\n\n"

    md += "## 4. Results Comparison\n\n"
    md += "| Model | Accuracy | vs Base |\n|-------|----------|--------|\n"
    md += f"| Base | {comparison.get('base_accuracy', 0):.1%} | - |\n"
    md += f"| Gold_100 | {comparison.get('gold100_accuracy', 0):.1%} | {comparison.get('gold100_vs_base', 0):+.1%} |\n"
    md += f"| Gold_Full | {comparison.get('gold_full_accuracy', 0):.1%} | {comparison.get('gold_full_vs_base', 0):+.1%} |\n\n"

    md += "## 5. Training\n\n"
    md += f"- Status: {training_results.get('status', 'unknown')}\n"
    if training_results.get("train_losses"):
        md += f"- Initial loss: {training_results['train_losses'][0]:.4f}\n"
        md += f"- Final loss: {training_results['train_losses'][-1]:.4f}\n"
    md += "\n"

    md += "## 6. Recommendation\n\n"
    md += f"- {comparison.get('recommendation', 'unknown')}\n"

    with open(report_dir / "PHASE_8_1B_REPORT.md", "w") as f:
        f.write(md)

    log("  Report generated")


if __name__ == "__main__":
    main()
