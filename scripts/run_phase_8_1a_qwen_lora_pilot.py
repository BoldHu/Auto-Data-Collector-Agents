"""Phase 8.1A: Qwen-VL LoRA pilot training.

Usage:
    python scripts/run_phase_8_1a_qwen_lora_pilot.py \
        --model_path models/qwen/Qwen2.5-VL-3B-Instruct \
        --max_train 100 \
        --max_eval 50 \
        --epochs 2
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
    parser = argparse.ArgumentParser(description="Phase 8.1A Qwen-VL LoRA pilot")
    parser.add_argument("--model_path", type=str, default="models/qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_train", type=int, default=100)
    parser.add_argument("--max_eval", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_multimodal", action="store_true")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_1a_qwen_lora_pilot"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1a_qwen_lora_pilot"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_1a.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.1A: Qwen-VL LoRA Pilot ===")

    # Step 1: Verify qwen-vl-utils
    log("Step 1: Verifying qwen-vl-utils...")
    vl_status = _verify_qwen_vl_utils(log)
    save_json(vl_status, report_dir / "qwen_vl_utils_status.json")

    # Step 2: Multimodal smoke test
    if not args.skip_multimodal and vl_status.get("status") == "ok":
        log("Step 2: Multimodal smoke test...")
        smoke_results = _multimodal_smoke_test(args.model_path, log)
        save_json(smoke_results, report_dir / "multimodal_smoke_report.json")
    else:
        log("Step 2: Skipping multimodal smoke test")
        smoke_results = {"status": "skipped"}

    # Step 3: Build gold train_100 subset
    log("Step 3: Building gold train_100 subset...")
    subset_stats = _build_gold_subset(args.max_train, args.max_eval, log)
    save_json(subset_stats, report_dir / "gold_train_100_stats.json")

    # Step 4: Prompt-corrected base zero-shot
    log("Step 4: Prompt-corrected base zero-shot evaluation...")
    base_results = _base_zero_shot(args, log)
    save_json(base_results, report_dir / "base_zero_shot_report.json")

    # Step 5: LoRA training
    if not args.skip_training:
        log("Step 5: LoRA gold_100 training...")
        training_results = _lora_training(args, log)
        save_json(training_results, report_dir / "lora_gold100_training_summary.json")
    else:
        log("Step 5: Skipping training")
        training_results = {"status": "skipped"}

    # Step 6: Evaluate LoRA adapter
    adapter_path = PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100"
    if adapter_path.exists() and not args.skip_training:
        log("Step 6: Evaluating LoRA adapter...")
        eval_results = _evaluate_lora(args.model_path, str(adapter_path), args, log)
        save_json(eval_results, report_dir / "lora_gold100_eval_report.json")
    else:
        log("Step 6: Skipping LoRA evaluation (no adapter)")
        eval_results = {"status": "skipped"}

    # Step 7: Analysis
    log("Step 7: Pilot analysis...")
    analysis = _analyze_pilot(base_results, training_results, eval_results, log)
    save_json(analysis, report_dir / "pilot_analysis.json")

    # Step 8: Generate report
    log("Step 8: Generating report...")
    _generate_report(vl_status, smoke_results, subset_stats, base_results,
                     training_results, eval_results, analysis, report_dir, log)

    log("=== Phase 8.1A Complete ===")


def _verify_qwen_vl_utils(log) -> dict:
    """Verify qwen-vl-utils installation."""
    try:
        from qwen_vl_utils import process_vision_info
        log("  qwen_vl_utils imported successfully")
        return {"status": "ok", "process_vision_info": True}
    except ImportError as e:
        log(f"  qwen_vl_utils import failed: {e}")
        return {"status": "failed", "error": str(e)}


def _multimodal_smoke_test(model_path: str, log) -> dict:
    """Run multimodal smoke test."""
    import torch

    results = {"tests": []}

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        # Find a carbon fiber image
        imgs_dir = PROJECT_ROOT / "imgs_raw_data" / "carbon_fiber_mm"
        test_images = []
        if imgs_dir.exists():
            for folder in list(imgs_dir.iterdir())[:3]:
                imgs = list(folder.glob("*.jpg"))[:1]
                test_images.extend(imgs)

        # Test 1: Image captioning
        if test_images:
            log("  Test 1: Image captioning")
            img_path = str(test_images[0])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": "请描述这张碳纤维相关图像的内容。"}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            log(f"    Response: {response[:100]}...")
            results["tests"].append({"test": "image_caption", "status": "success", "response": response[:200]})

        # Test 2: CFBench-MM
        bench_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
        if bench_path.exists():
            items = load_jsonl(bench_path)
            mm_items = [i for i in items if i.get("source_type") == "multimodal"][:3]
            for item in mm_items:
                source_refs = item.get("source_refs", [])
                img_path = source_refs[0] if source_refs else None
                if img_path and os.path.exists(img_path):
                    log(f"  Test: CFBench-MM {item.get('benchmark_id', '')[:20]}")
                    question = item.get("question", "")
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": f"{question}\n请直接回答。"}
                    ]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, _ = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                    response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    log(f"    Response: {response[:80]}...")
                    results["tests"].append({"test": "cfbench_mm", "status": "success", "response": response[:200]})

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        results["peak_gpu_memory_gb"] = round(peak_mem, 2)
        results["status"] = "success"
        results["tests_passed"] = len(results["tests"])

        del model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)[:300]
        log(f"  Error: {str(e)[:100]}")

    return results


def _build_gold_subset(max_train: int, max_eval: int, log) -> dict:
    """Build gold train_100 subset."""
    random.seed(42)

    # Load gold train
    gold_train_path = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml.jsonl"
    gold_val_path = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "validation_chatml.jsonl"

    if not gold_train_path.exists():
        log("  Gold train not found, building from full v4...")
        # Build from full v4
        full_train = load_jsonl(PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "train_chatml.jsonl")
        random.shuffle(full_train)
        train_subset = full_train[:max_train]
        val_subset = full_train[max_train:max_train + max_eval]
    else:
        train_all = load_jsonl(gold_train_path)
        val_all = load_jsonl(gold_val_path)
        random.shuffle(train_all)
        train_subset = train_all[:max_train]
        val_subset = val_all[:max_eval]

    # Save subsets
    subsets_dir = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "subsets"
    save_jsonl(train_subset, subsets_dir / "train_100_chatml.jsonl")
    save_jsonl(val_subset, subsets_dir / "validation_50_chatml.jsonl")

    stats = {
        "train_count": len(train_subset),
        "val_count": len(val_subset),
        "train_path": str(subsets_dir / "train_100_chatml.jsonl"),
        "val_path": str(subsets_dir / "validation_50_chatml.jsonl"),
    }

    log(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")
    return stats


def _base_zero_shot(args, log) -> dict:
    """Run prompt-corrected base zero-shot evaluation."""
    import torch

    model_path = args.model_path
    results = {"subsets": {}, "total_correct": 0, "total_items": 0}

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        # Load benchmark subsets
        bench_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
        if not bench_path.exists():
            return {"status": "skipped", "reason": "benchmark_not_found"}

        all_items = load_jsonl(bench_path)

        # Sample subsets
        text_items = [i for i in all_items if i.get("source_type") == "text"][:50]
        exam_items = [i for i in all_items if i.get("task_type", "").startswith("exam")][:30]

        subsets = {
            "cfbench_text": text_items,
            "cfbench_exam": exam_items,
        }

        for subset_name, items in subsets.items():
            if not items:
                continue

            log(f"  Evaluating {subset_name}: {len(items)} items")
            correct = 0
            total = 0
            subset_results = []

            for item in items:
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

                # Parse answer
                is_correct = _check_answer(response, gold, options)
                if is_correct:
                    correct += 1
                total += 1

                subset_results.append({
                    "benchmark_id": item.get("benchmark_id", ""),
                    "gold": gold[:50],
                    "predicted": response[:50],
                    "correct": is_correct,
                })

            accuracy = correct / max(total, 1)
            results["subsets"][subset_name] = {
                "total": total,
                "correct": correct,
                "accuracy": round(accuracy, 3),
                "results": subset_results[:10],
            }
            results["total_correct"] += correct
            results["total_items"] += total

            log(f"    {subset_name}: {correct}/{total} = {accuracy:.1%}")

        results["overall_accuracy"] = round(results["total_correct"] / max(results["total_items"], 1), 3)
        results["status"] = "success"

        del model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)[:300]
        log(f"  Error: {str(e)[:100]}")

    return results


def _check_answer(response: str, gold: str, options: list) -> bool:
    """Check if response matches gold answer."""
    # For multiple choice
    if options and len(options) >= 2:
        pred_match = re.search(r'([A-H])', response.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            return pred_match.group(1) == gold_match.group(1)

    # For true/false
    tf_map = {"对": "正确", "错": "错误", "true": "正确", "false": "错误", "yes": "正确", "no": "错误"}
    resp_norm = response.strip().lower()
    gold_norm = gold.strip().lower()
    for k, v in tf_map.items():
        if k in resp_norm:
            resp_norm = v
        if k in gold_norm:
            gold_norm = v
    if resp_norm == gold_norm:
        return True

    # For short answer - keyword match
    gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold))
    resp_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', response))
    if gold_keywords and len(gold_keywords & resp_keywords) / len(gold_keywords) > 0.5:
        return True

    return False


def _lora_training(args, log) -> dict:
    """Run LoRA training on gold_100."""
    import torch

    model_path = args.model_path
    train_file = str(PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "subsets" / "train_100_chatml.jsonl")
    val_file = str(PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "subsets" / "validation_50_chatml.jsonl")
    output_dir = str(PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100")

    if not os.path.exists(train_file):
        return {"status": "skipped", "reason": "train_file_not_found"}

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
        result["target_modules"] = target_modules

        log(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

        # Load dataset
        train_samples = load_jsonl(Path(train_file))
        val_samples = load_jsonl(Path(val_file))

        # Create simple dataset
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
            warmup_ratio=0.1,
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
        )

        # Custom trainer with loss logging
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
        train_result = trainer.train()
        training_time = time.time() - start_time

        result["status"] = "success"
        result["training_time_seconds"] = round(training_time, 1)
        result["train_loss"] = trainer.train_losses[-1] if trainer.train_losses else None
        result["train_losses"] = trainer.train_losses
        result["val_losses"] = trainer.val_losses

        # Save adapter
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        result["adapter_saved"] = True

        log(f"  Training complete: {training_time:.0f}s")
        if trainer.train_losses:
            log(f"  Final train loss: {trainer.train_losses[-1]:.4f}")
        if trainer.val_losses:
            log(f"  Final val loss: {trainer.val_losses[-1]:.4f}")

        del model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log(f"  Training error: {str(e)[:100]}")

    return result


def _evaluate_lora(model_path: str, adapter_path: str, args, log) -> dict:
    """Evaluate LoRA adapter vs base model."""
    import torch

    results = {"base": {}, "lora": {}}

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel

        # Load base model
        log("  Loading base model...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        # Evaluate base
        bench_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
        if bench_path.exists():
            all_items = load_jsonl(bench_path)
            text_items = [i for i in all_items if i.get("source_type") == "text"][:30]

            base_correct = 0
            for item in text_items:
                response = _run_inference(base_model, processor, item)
                if _check_answer(response, str(item.get("answer", "")), item.get("options", [])):
                    base_correct += 1

            results["base"] = {
                "total": len(text_items),
                "correct": base_correct,
                "accuracy": round(base_correct / max(len(text_items), 1), 3),
            }
            log(f"  Base: {base_correct}/{len(text_items)} = {results['base']['accuracy']:.1%}")

        del base_model
        torch.cuda.empty_cache()

        # Load LoRA model
        log("  Loading LoRA adapter...")
        lora_model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        lora_model = PeftModel.from_pretrained(lora_model, adapter_path)

        if bench_path.exists():
            lora_correct = 0
            for item in text_items:
                response = _run_inference(lora_model, processor, item)
                if _check_answer(response, str(item.get("answer", "")), item.get("options", [])):
                    lora_correct += 1

            results["lora"] = {
                "total": len(text_items),
                "correct": lora_correct,
                "accuracy": round(lora_correct / max(len(text_items), 1), 3),
            }
            log(f"  LoRA: {lora_correct}/{len(text_items)} = {results['lora']['accuracy']:.1%}")

        results["improvement"] = results.get("lora", {}).get("accuracy", 0) - results.get("base", {}).get("accuracy", 0)
        results["status"] = "success"

        del lora_model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)[:300]
        log(f"  Evaluation error: {str(e)[:100]}")

    return results


def _run_inference(model, processor, item: dict) -> str:
    """Run inference on a single item."""
    import torch

    question = item.get("question", "")
    options = item.get("options", [])

    if options:
        opt_text = "\n".join(str(o) for o in options)
        prompt = f"{question}\n\n选项：\n{opt_text}\n\n请只输出选项字母。"
    else:
        prompt = f"{question}\n\n请直接回答。"

    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    return processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _analyze_pilot(base_results: dict, training_results: dict, eval_results: dict, log) -> dict:
    """Analyze pilot results."""
    analysis = {
        "training_successful": training_results.get("status") == "success",
        "loss_decreased": False,
        "lora_improved": False,
        "recommendation": "unknown",
    }

    # Check if loss decreased
    if training_results.get("train_losses"):
        losses = training_results["train_losses"]
        if len(losses) >= 2:
            analysis["loss_decreased"] = losses[-1] < losses[0]
            analysis["initial_loss"] = losses[0]
            analysis["final_loss"] = losses[-1]

    # Check if LoRA improved
    if eval_results.get("improvement", 0) > 0:
        analysis["lora_improved"] = True

    # Recommendation
    if analysis["training_successful"] and analysis["lora_improved"]:
        analysis["recommendation"] = "proceed_to_gold_full"
    elif analysis["training_successful"] and analysis["loss_decreased"]:
        analysis["recommendation"] = "proceed_with_caution"
    elif analysis["training_successful"]:
        analysis["recommendation"] = "debug_data_or_prompt"
    else:
        analysis["recommendation"] = "fix_training_setup"

    log(f"  Recommendation: {analysis['recommendation']}")
    return analysis


def _generate_report(vl_status, smoke_results, subset_stats, base_results,
                     training_results, eval_results, analysis, report_dir, log):
    """Generate Phase 8.1A report."""
    md = "# Phase 8.1A: Qwen-VL LoRA Pilot Report\n\n"

    md += "## 1. qwen-vl-utils Status\n\n"
    md += f"- Status: {vl_status.get('status', 'unknown')}\n\n"

    md += "## 2. Multimodal Smoke Test\n\n"
    md += f"- Status: {smoke_results.get('status', 'skipped')}\n"
    if smoke_results.get("tests_passed"):
        md += f"- Tests passed: {smoke_results['tests_passed']}\n"
    md += "\n"

    md += "## 3. Gold Train_100 Subset\n\n"
    md += f"- Train: {subset_stats.get('train_count', 0)} samples\n"
    md += f"- Val: {subset_stats.get('val_count', 0)} samples\n\n"

    md += "## 4. Base Zero-Shot Results\n\n"
    if base_results.get("subsets"):
        md += "| Subset | Total | Correct | Accuracy |\n|--------|-------|---------|----------|\n"
        for name, data in base_results["subsets"].items():
            md += f"| {name} | {data['total']} | {data['correct']} | {data['accuracy']:.1%} |\n"
        md += f"\nOverall: {base_results.get('overall_accuracy', 0):.1%}\n\n"

    md += "## 5. LoRA Training\n\n"
    md += f"- Status: {training_results.get('status', 'unknown')}\n"
    if training_results.get("training_time_seconds"):
        md += f"- Training time: {training_results['training_time_seconds']:.0f}s\n"
    if training_results.get("train_losses"):
        md += f"- Initial loss: {training_results['train_losses'][0]:.4f}\n"
        md += f"- Final loss: {training_results['train_losses'][-1]:.4f}\n"
    if training_results.get("trainable_parameters"):
        md += f"- Trainable parameters: {training_results['trainable_parameters']:,}\n"
    md += "\n"

    md += "## 6. LoRA Evaluation\n\n"
    if eval_results.get("status") == "success":
        md += f"- Base accuracy: {eval_results.get('base', {}).get('accuracy', 0):.1%}\n"
        md += f"- LoRA accuracy: {eval_results.get('lora', {}).get('accuracy', 0):.1%}\n"
        md += f"- Improvement: {eval_results.get('improvement', 0):.1%}\n"
    else:
        md += f"- Status: {eval_results.get('status', 'skipped')}\n"
    md += "\n"

    md += "## 7. Analysis\n\n"
    md += f"- Training successful: {analysis.get('training_successful', False)}\n"
    md += f"- Loss decreased: {analysis.get('loss_decreased', False)}\n"
    md += f"- LoRA improved: {analysis.get('lora_improved', False)}\n"
    md += f"- Recommendation: {analysis.get('recommendation', 'unknown')}\n"

    with open(report_dir / "PHASE_8_1A_REPORT.md", "w") as f:
        f.write(md)

    log("  Report generated")


if __name__ == "__main__":
    main()
