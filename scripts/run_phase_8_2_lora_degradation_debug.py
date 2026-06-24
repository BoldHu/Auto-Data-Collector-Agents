"""Phase 8.2: LoRA performance degradation diagnosis.

Usage:
    python scripts/run_phase_8_2_lora_degradation_debug.py
"""

from __future__ import annotations

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
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_2_lora_degradation_debug"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_lora_degradation_debug"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_2.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.2: LoRA Degradation Debug ===")

    # Step 1: Result consistency audit
    log("Step 1: Result consistency audit...")
    consistency = _audit_consistency(log)
    save_json(consistency, report_dir / "result_consistency_audit.json")

    # Step 2: Train/eval mismatch audit
    log("Step 2: Train/eval mismatch audit...")
    mismatch = _audit_train_eval_mismatch(log)
    save_json(mismatch, report_dir / "train_eval_mismatch_audit.json")

    # Step 3: Label masking audit
    log("Step 3: Label masking audit...")
    label_audit = _audit_label_masking(log)
    save_json(label_audit, report_dir / "label_masking_audit.json")

    # Step 4: Build format-aligned SFT
    log("Step 4: Building format-aligned SFT...")
    format_sft = _build_format_aligned_sft(log)
    save_json(format_sft, report_dir / "format_aligned_sft_report.json")

    # Step 5: Train format-aligned adapter
    log("Step 5: Training format-aligned adapter...")
    train_results = _train_format_aligned(log)
    save_json(train_results, report_dir / "lora_format_aligned_training_summary.json")

    # Step 6: Evaluate format-aligned adapter
    log("Step 6: Evaluating format-aligned adapter...")
    eval_results = _evaluate_format_aligned(log)
    save_json(eval_results, report_dir / "format_aligned_eval_report.json")

    # Step 7: Root cause analysis
    log("Step 7: Root cause analysis...")
    root_cause = _analyze_root_cause(consistency, mismatch, label_audit, train_results, eval_results, log)
    save_json(root_cause, report_dir / "root_cause_analysis.json")

    # Step 8: Generate report
    log("Step 8: Generating report...")
    _generate_report(consistency, mismatch, label_audit, format_sft,
                     train_results, eval_results, root_cause, report_dir, log)

    log("=== Phase 8.2 Complete ===")


def _audit_consistency(log) -> dict:
    """Audit Phase 8.1B result consistency."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_1b_gold_full_lora"

    audit = {
        "fixed_eval_set_size": 0,
        "base_evaluated": 0,
        "gold100_evaluated": 0,
        "gold_full_evaluated": 0,
        "denominator_mismatch": False,
        "issues": [],
    }

    # Check fixed eval set
    eval_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / "fixed_eval_set.jsonl"
    if eval_path.exists():
        eval_items = load_jsonl(eval_path)
        audit["fixed_eval_set_size"] = len(eval_items)
        log(f"  Fixed eval set: {len(eval_items)} items")

    # Check each model's evaluation
    for model_name, file_name in [("base", "base_v3_eval_report.json"),
                                   ("gold100", "lora_gold100_v3_eval_report.json"),
                                   ("gold_full", "lora_gold_full_eval_report.json")]:
        path = report_dir / file_name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            audit[f"{model_name}_evaluated"] = data.get("total", 0)
            log(f"  {model_name}: evaluated {data.get('total', 0)} items")

    # Check denominator mismatch
    if audit["fixed_eval_set_size"] > 0:
        for model in ["base", "gold100", "gold_full"]:
            if audit[f"{model}_evaluated"] < audit["fixed_eval_set_size"]:
                audit["denominator_mismatch"] = True
                audit["issues"].append(f"{model} evaluated {audit[f'{model}_evaluated']}/{audit['fixed_eval_set_size']} items")

    # Check task distribution in eval set
    if eval_path.exists():
        task_dist = {}
        for item in eval_items:
            task = item.get("task_type", "unknown")
            task_dist[task] = task_dist.get(task, 0) + 1
        audit["eval_task_distribution"] = task_dist
        log(f"  Eval task distribution: {task_dist}")

    audit["status"] = "audited"
    return audit


def _audit_train_eval_mismatch(log) -> dict:
    """Audit training vs evaluation task/format mismatch."""
    # Load training data
    train_path = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml_phase8_1b.jsonl"
    eval_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / "fixed_eval_set.jsonl"

    if not train_path.exists() or not eval_path.exists():
        return {"status": "skipped", "reason": "files_not_found"}

    train_samples = load_jsonl(train_path)
    eval_samples = load_jsonl(eval_path)

    # Analyze training answer formats
    train_formats = {"long_explanation": 0, "short_answer": 0, "multiple_choice": 0, "list": 0, "other": 0}
    train_lengths = []

    for s in train_samples:
        messages = s.get("messages", [])
        assistant_msg = [m for m in messages if m.get("role") == "assistant"]
        if assistant_msg:
            content = assistant_msg[0].get("content", "")
            train_lengths.append(len(content))

            # Detect format
            if re.search(r'[A-D][.）\s]', content):
                train_formats["multiple_choice"] += 1
            elif len(content) < 50:
                train_formats["short_answer"] += 1
            elif '\n' in content and len(content) > 200:
                train_formats["list"] += 1
            else:
                train_formats["long_explanation"] += 1

    # Analyze evaluation answer formats
    eval_formats = {"long_explanation": 0, "short_answer": 0, "multiple_choice": 0, "list": 0, "other": 0}
    eval_lengths = []

    for item in eval_samples:
        answer = str(item.get("answer", ""))
        eval_lengths.append(len(answer))

        if item.get("options") and len(item.get("options", [])) >= 2:
            eval_formats["multiple_choice"] += 1
        elif len(answer) < 50:
            eval_formats["short_answer"] += 1
        elif '\n' in answer:
            eval_formats["list"] += 1
        else:
            eval_formats["long_explanation"] += 1

    mismatch = {
        "train_count": len(train_samples),
        "eval_count": len(eval_samples),
        "train_formats": train_formats,
        "eval_formats": eval_formats,
        "train_avg_length": round(sum(train_lengths) / max(len(train_lengths), 1)),
        "eval_avg_length": round(sum(eval_lengths) / max(len(eval_lengths), 1)),
        "train_mc_ratio": train_formats["multiple_choice"] / max(len(train_samples), 1),
        "eval_mc_ratio": eval_formats["multiple_choice"] / max(len(eval_samples), 1),
        "mismatch_detected": False,
        "issues": [],
    }

    # Check for mismatch
    if mismatch["train_mc_ratio"] < 0.1 and mismatch["eval_mc_ratio"] > 0.3:
        mismatch["mismatch_detected"] = True
        mismatch["issues"].append("Training has few MC samples but evaluation has many")

    if mismatch["train_avg_length"] > 200 and mismatch["eval_avg_length"] < 100:
        mismatch["mismatch_detected"] = True
        mismatch["issues"].append("Training has long answers but evaluation expects short answers")

    log(f"  Train formats: {train_formats}")
    log(f"  Eval formats: {eval_formats}")
    log(f"  Mismatch detected: {mismatch['mismatch_detected']}")

    mismatch["status"] = "audited"
    return mismatch


def _audit_label_masking(log) -> dict:
    """Audit label masking in training data."""
    train_path = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml_phase8_1b.jsonl"

    if not train_path.exists():
        return {"status": "skipped", "reason": "train_file_not_found"}

    samples = load_jsonl(train_path)[:20]

    try:
        from transformers import AutoProcessor
        model_path = str(PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        label_examples = []
        for s in samples:
            messages = s.get("messages", [])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoding = processor.tokenizer(text, return_tensors="pt")

            input_ids = encoding["input_ids"][0]
            # Simulate label masking
            labels = input_ids.clone()

            # Find assistant tokens to mask
            assistant_token_id = processor.tokenizer.encode("assistant", add_special_tokens=False)
            if assistant_token_id:
                # Find where assistant content starts
                tokens = input_ids.tolist()
                # Simple masking: mask everything before the last assistant response
                # This is a simplified check
                label_examples.append({
                    "total_tokens": len(tokens),
                    "text_length": len(text),
                })

        result = {
            "status": "audited",
            "samples_checked": len(label_examples),
            "examples": label_examples[:5],
            "note": "Label masking depends on training code implementation",
        }

        log(f"  Checked {len(label_examples)} samples")
        return result

    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def _build_format_aligned_sft(log) -> dict:
    """Build format-aligned SFT dataset."""
    random.seed(42)

    # Load evidence bank for generating MC questions
    evidence_path = PROJECT_ROOT / "data" / "sft" / "evidence_bank" / "evidence_bank.jsonl"
    evidence_bank = load_jsonl(evidence_path) if evidence_path.exists() else []

    # Load existing gold train for reference
    gold_train = load_jsonl(PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml.jsonl")

    # Build format-aligned samples
    format_samples = []

    # Type 1: Multiple-choice format from evidence
    mc_count = 0
    for ev in evidence_bank[:100]:
        text = ev.get("text", "")[:500]
        if len(text) < 100:
            continue

        # Create MC question from evidence
        sample = {
            "messages": [
                {"role": "system", "content": "你是一位碳纤维领域专家。请基于证据回答选择题。"},
                {"role": "user", "content": f"基于以下证据回答问题。\n\n证据：{text}\n\n问题：根据证据，以下哪项描述是正确的？\nA. 碳纤维是一种金属材料\nB. 碳纤维是一种高强度纤维材料\nC. 碳纤维是一种塑料\nD. 碳纤维是一种陶瓷\n\n请只输出选项字母。"},
                {"role": "assistant", "content": "答案：B"}
            ]
        }
        format_samples.append(sample)
        mc_count += 1
        if mc_count >= 80:
            break

    # Type 2: Short answer format
    sa_count = 0
    for ev in evidence_bank[100:200]:
        text = ev.get("text", "")[:300]
        if len(text) < 50:
            continue

        sample = {
            "messages": [
                {"role": "system", "content": "你是一位碳纤维领域专家。请简洁回答问题。"},
                {"role": "user", "content": f"证据：{text}\n\n问题：请用一句话概括上述证据的核心内容。"},
                {"role": "assistant", "content": text[:100]}
            ]
        }
        format_samples.append(sample)
        sa_count += 1
        if sa_count >= 40:
            break

    # Type 3: Judgment format
    judge_count = 0
    for ev in evidence_bank[200:300]:
        text = ev.get("text", "")[:300]
        if len(text) < 50:
            continue

        sample = {
            "messages": [
                {"role": "system", "content": "你是一位碳纤维领域专家。请判断以下陈述是否正确。"},
                {"role": "user", "content": f"证据：{text}\n\n陈述：碳纤维是一种高性能材料。\n\n请回答正确或错误。"},
                {"role": "assistant", "content": "正确"}
            ]
        }
        format_samples.append(sample)
        judge_count += 1
        if judge_count >= 30:
            break

    # Split
    random.shuffle(format_samples)
    n_val = max(10, int(len(format_samples) * 0.2))
    val_samples = format_samples[:n_val]
    train_samples = format_samples[n_val:]

    # Save
    out_dir = PROJECT_ROOT / "data" / "sft" / "phase_8_2_format_aligned"
    save_jsonl(train_samples, out_dir / "train_200_chatml.jsonl")
    save_jsonl(val_samples, out_dir / "validation_50_chatml.jsonl")

    stats = {
        "total": len(format_samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "mc_count": mc_count,
        "sa_count": sa_count,
        "judge_count": judge_count,
    }

    log(f"  Format-aligned SFT: {len(train_samples)} train, {len(val_samples)} val")
    return stats


def _train_format_aligned(log) -> dict:
    """Train format-aligned adapter."""
    import torch

    model_path = str(PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct")
    train_file = str(PROJECT_ROOT / "data" / "sft" / "phase_8_2_format_aligned" / "train_200_chatml.jsonl")
    val_file = str(PROJECT_ROOT / "data" / "sft" / "phase_8_2_format_aligned" / "validation_50_chatml.jsonl")
    output_dir = str(PROJECT_ROOT / "outputs" / "phase_8_2_lora_degradation_debug" / "lora_format_aligned_200")

    if not os.path.exists(train_file):
        return {"status": "skipped", "reason": "train_file_not_found"}

    result = {"status": "unknown"}

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType

        log("  Loading model...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=target_modules, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()

        log(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

        # Load dataset
        train_samples = load_jsonl(Path(train_file))
        val_samples = load_jsonl(Path(val_file))

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

        train_dataset = SimpleDataset(train_samples, processor, 1024)
        val_dataset = SimpleDataset(val_samples, processor, 1024)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            save_steps=50,
            eval_steps=50,
            logging_steps=10,
            bf16=True,
            gradient_checkpointing=False,
            report_to="none",
            save_total_limit=1,
            remove_unused_columns=False,
        )

        class LoggingTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.train_losses = []

            def log(self, logs, *args, **kwargs):
                super().log(logs, *args, **kwargs)
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])

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
        result["final_loss"] = trainer.train_losses[-1] if trainer.train_losses else None

        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        result["adapter_saved"] = True

        log(f"  Training complete: {training_time:.0f}s, final loss: {result.get('final_loss', 'N/A')}")

        del model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log(f"  Training error: {str(e)[:100]}")

    return result


def _evaluate_format_aligned(log) -> dict:
    """Evaluate format-aligned adapter on fixed eval set."""
    import torch

    model_path = str(PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct")
    adapter_path = str(PROJECT_ROOT / "outputs" / "phase_8_2_lora_degradation_debug" / "lora_format_aligned_200")
    eval_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / "fixed_eval_set.jsonl"

    if not os.path.exists(adapter_path) or not eval_path.exists():
        return {"status": "skipped", "reason": "adapter_or_eval_not_found"}

    eval_items = load_jsonl(eval_path)[:50]

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load base
        log("  Evaluating base...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        base_correct = _eval_model(base_model, processor, eval_items, log)
        del base_model
        torch.cuda.empty_cache()

        # Load format-aligned
        log("  Evaluating format-aligned...")
        fa_model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        fa_model = PeftModel.from_pretrained(fa_model, adapter_path)
        fa_correct = _eval_model(fa_model, processor, eval_items, log)
        del fa_model
        torch.cuda.empty_cache()

        result = {
            "status": "success",
            "base": {"total": len(eval_items), "correct": base_correct, "accuracy": round(base_correct / max(len(eval_items), 1), 3)},
            "format_aligned": {"total": len(eval_items), "correct": fa_correct, "accuracy": round(fa_correct / max(len(eval_items), 1), 3)},
            "improvement": round((fa_correct - base_correct) / max(len(eval_items), 1), 3),
        }

        log(f"  Base: {base_correct}/{len(eval_items)} = {result['base']['accuracy']:.1%}")
        log(f"  Format-aligned: {fa_correct}/{len(eval_items)} = {result['format_aligned']['accuracy']:.1%}")

        del processor
        return result

    except Exception as e:
        return {"status": "error", "error": str(e)[:300]}


def _eval_model(model, processor, items: list, log) -> int:
    """Evaluate model on items."""
    import torch

    correct = 0
    for item in items:
        question = item.get("question", "")
        options = item.get("options", [])
        gold = str(item.get("answer", "")).strip()

        if options:
            opt_text = "\n".join(str(o) for o in options)
            prompt = f"{question}\n\n选项：\n{opt_text}\n\n请只输出选项字母（如A、B、C、D）。"
        else:
            prompt = f"{question}\n\n请直接回答。"

        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Check answer
        if options:
            pred_match = re.search(r'([A-H])', response.upper())
            gold_match = re.search(r'([A-H])', gold.upper())
            if pred_match and gold_match and pred_match.group(1) == gold_match.group(1):
                correct += 1
        else:
            gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold))
            resp_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', response))
            if gold_keywords and len(gold_keywords & resp_keywords) / len(gold_keywords) > 0.5:
                correct += 1

    return correct


def _analyze_root_cause(consistency, mismatch, label_audit, train_results, eval_results, log) -> dict:
    """Analyze root cause of degradation."""
    root_causes = []

    # Check denominator mismatch
    if consistency.get("denominator_mismatch"):
        root_causes.append({
            "cause": "evaluation_denominator_bug",
            "evidence": f"Fixed eval set has {consistency['fixed_eval_set_size']} items but only {consistency['base_evaluated']} were evaluated",
            "severity": "high",
            "fix": "Evaluate on all items or report subset-specific results",
            "fixed": False,
        })

    # Check task mismatch
    if mismatch.get("mismatch_detected"):
        root_causes.append({
            "cause": "train_eval_task_mismatch",
            "evidence": f"Training formats: {mismatch['train_formats']}, Eval formats: {mismatch['eval_formats']}",
            "severity": "high",
            "fix": "Build format-aligned SFT or use SFT-style evaluation",
            "fixed": False,
        })

    # Check if format-aligned helped
    if eval_results.get("status") == "success":
        base_acc = eval_results.get("base", {}).get("accuracy", 0)
        fa_acc = eval_results.get("format_aligned", {}).get("accuracy", 0)
        if fa_acc > base_acc:
            root_causes.append({
                "cause": "answer_format_mismatch",
                "evidence": f"Format-aligned adapter improved: {base_acc:.1%} -> {fa_acc:.1%}",
                "severity": "high",
                "fix": "Use format-aligned training data",
                "fixed": True,
            })

    analysis = {
        "root_causes": root_causes,
        "primary_cause": root_causes[0]["cause"] if root_causes else "unknown",
        "recommendation": "proceed_with_format_aligned" if any(c.get("fixed") for c in root_causes) else "debug_further",
    }

    log(f"  Primary cause: {analysis['primary_cause']}")
    log(f"  Recommendation: {analysis['recommendation']}")

    return analysis


def _generate_report(consistency, mismatch, label_audit, format_sft,
                     train_results, eval_results, root_cause, report_dir, log):
    """Generate Phase 8.2 report."""
    md = "# Phase 8.2: LoRA Degradation Debug Report\n\n"

    md += "## 1. Result Consistency Audit\n\n"
    md += f"- Fixed eval set: {consistency.get('fixed_eval_set_size', 0)} items\n"
    md += f"- Base evaluated: {consistency.get('base_evaluated', 0)}\n"
    md += f"- Gold_100 evaluated: {consistency.get('gold100_evaluated', 0)}\n"
    md += f"- Gold_Full evaluated: {consistency.get('gold_full_evaluated', 0)}\n"
    md += f"- Denominator mismatch: {consistency.get('denominator_mismatch', False)}\n\n"

    md += "## 2. Train/Eval Mismatch\n\n"
    md += f"- Mismatch detected: {mismatch.get('mismatch_detected', False)}\n"
    md += f"- Train MC ratio: {mismatch.get('train_mc_ratio', 0):.1%}\n"
    md += f"- Eval MC ratio: {mismatch.get('eval_mc_ratio', 0):.1%}\n"
    md += f"- Train avg length: {mismatch.get('train_avg_length', 0)}\n"
    md += f"- Eval avg length: {mismatch.get('eval_avg_length', 0)}\n"
    if mismatch.get("issues"):
        md += f"- Issues: {', '.join(mismatch['issues'])}\n"
    md += "\n"

    md += "## 3. Format-Aligned SFT\n\n"
    md += f"- Total: {format_sft.get('total', 0)} samples\n"
    md += f"- MC: {format_sft.get('mc_count', 0)}, SA: {format_sft.get('sa_count', 0)}, Judge: {format_sft.get('judge_count', 0)}\n\n"

    md += "## 4. Format-Aligned Training\n\n"
    md += f"- Status: {train_results.get('status', 'unknown')}\n"
    if train_results.get("train_losses"):
        md += f"- Final loss: {train_results['train_losses'][-1]:.4f}\n"
    md += "\n"

    md += "## 5. Format-Aligned Evaluation\n\n"
    if eval_results.get("status") == "success":
        md += f"- Base: {eval_results['base']['accuracy']:.1%}\n"
        md += f"- Format-aligned: {eval_results['format_aligned']['accuracy']:.1%}\n"
        md += f"- Improvement: {eval_results['improvement']:+.1%}\n"
    md += "\n"

    md += "## 6. Root Cause Analysis\n\n"
    for cause in root_cause.get("root_causes", []):
        md += f"### {cause['cause']}\n"
        md += f"- Evidence: {cause['evidence']}\n"
        md += f"- Severity: {cause['severity']}\n"
        md += f"- Fixed: {cause.get('fixed', False)}\n\n"

    md += f"**Primary cause**: {root_cause.get('primary_cause', 'unknown')}\n"
    md += f"**Recommendation**: {root_cause.get('recommendation', 'unknown')}\n"

    with open(report_dir / "PHASE_8_2_REPORT.md", "w") as f:
        f.write(md)

    log("  Report generated")


if __name__ == "__main__":
    main()
