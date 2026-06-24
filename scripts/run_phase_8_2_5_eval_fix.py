"""Phase 8.2.5: Evaluation denominator fix and full re-evaluation.

Usage:
    python scripts/run_phase_8_2_5_eval_fix.py \
        --model_path models/qwen/Qwen2.5-VL-3B-Instruct
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
    parser = argparse.ArgumentParser(description="Phase 8.2.5 eval fix")
    parser.add_argument("--model_path", type=str, default="models/qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_2_5_eval_fix"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_2_5.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.2.5: Evaluation Fix ===")

    # Step 1: Audit denominator bug
    log("Step 1: Auditing denominator bug...")
    bug_audit = _audit_denominator_bug(log)
    save_json(bug_audit, report_dir / "denominator_bug_audit.json")

    # Step 2: Build canonical 150-item manifest
    log("Step 2: Building canonical manifest...")
    manifest_stats = _build_canonical_manifest(log)
    save_json(manifest_stats, report_dir / "canonical_eval_manifest_stats.json")

    # Step 3: Verify adapter activation
    log("Step 3: Verifying adapter activation...")
    adapter_check = _verify_adapters(args.model_path, log)
    save_json(adapter_check, report_dir / "adapter_activation_check.json")

    # Step 4: Re-evaluate all models on 150 items
    log("Step 4: Re-evaluating all models on 150 items...")
    eval_results = _reevaluate_all(args.model_path, log)
    save_json(eval_results, report_dir / "reevaluation_150_report.json")

    # Step 5: Label masking verification
    log("Step 5: Verifying label masking...")
    label_check = _verify_label_masking(log)
    save_json(label_check, report_dir / "label_masking_verification.json")

    # Step 6: Root cause update
    log("Step 6: Updating root cause analysis...")
    root_cause = _update_root_cause(bug_audit, eval_results, adapter_check, label_check, log)
    save_json(root_cause, report_dir / "root_cause_update.json")

    # Step 7: Generate report
    log("Step 7: Generating report...")
    _generate_report(bug_audit, manifest_stats, adapter_check, eval_results,
                     label_check, root_cause, report_dir, log)

    log("=== Phase 8.2.5 Complete ===")


def _audit_denominator_bug(log) -> dict:
    """Audit why only 50/150 items were evaluated."""
    # Check the evaluation code
    eval_code_path = PROJECT_ROOT / "scripts" / "run_phase_8_1b_gold_full_lora.py"

    audit = {
        "bug_found": False,
        "bug_location": "",
        "bug_code": "",
        "fix_applied": False,
    }

    if eval_code_path.exists():
        with open(eval_code_path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "[:50]" in line and "eval_items" in line:
                audit["bug_found"] = True
                audit["bug_location"] = f"line {i+1}"
                audit["bug_code"] = line.strip()
                log(f"  Bug found at {audit['bug_location']}: {audit['bug_code']}")
                break

    # Check eval set size
    eval_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / "fixed_eval_set.jsonl"
    if eval_path.exists():
        eval_items = load_jsonl(eval_path)
        audit["eval_set_size"] = len(eval_items)
        log(f"  Eval set size: {len(eval_items)}")

    # Check actual evaluated items
    for model in ["base", "gold100", "goldfull"]:
        output_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / f"{model}_v3_outputs.jsonl"
        if output_path.exists():
            outputs = load_jsonl(output_path)
            audit[f"{model}_outputs_count"] = len(outputs)
            log(f"  {model} outputs: {len(outputs)}")

    audit["status"] = "audited"
    return audit


def _build_canonical_manifest(log) -> dict:
    """Build canonical 150-item evaluation manifest."""
    eval_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_1b_gold_full_lora" / "fixed_eval_set.jsonl"

    if not eval_path.exists():
        return {"status": "not_found"}

    eval_items = load_jsonl(eval_path)

    # Add eval_index and expected format
    manifest = []
    for i, item in enumerate(eval_items):
        entry = dict(item)
        entry["eval_index"] = i

        # Determine expected answer format
        options = item.get("options", [])
        task_type = item.get("task_type", "")

        if options and len(options) >= 2:
            entry["expected_answer_format"] = "multiple_choice_letter"
            entry["scoring_method"] = "option_letter_match"
        elif task_type.startswith("exam"):
            entry["expected_answer_format"] = "short_answer"
            entry["scoring_method"] = "keyword_match"
        else:
            entry["expected_answer_format"] = "open_ended"
            entry["scoring_method"] = "f1_keyword"

        manifest.append(entry)

    # Save manifest
    save_jsonl(manifest, PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "canonical_eval_manifest_150.jsonl")

    # Stats
    from collections import Counter
    task_dist = Counter(m.get("task_type", "unknown") for m in manifest)
    format_dist = Counter(m.get("expected_answer_format", "unknown") for m in manifest)
    subset_dist = Counter(m.get("source_type", "unknown") for m in manifest)

    stats = {
        "total": len(manifest),
        "task_distribution": dict(task_dist),
        "format_distribution": dict(format_dist),
        "subset_distribution": dict(subset_dist),
    }

    log(f"  Manifest: {len(manifest)} items")
    log(f"  Formats: {dict(format_dist)}")

    return stats


def _verify_adapters(model_path: str, log) -> dict:
    """Verify all adapters load correctly."""
    import torch

    adapters = {
        "gold100": str(PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100"),
        "goldfull": str(PROJECT_ROOT / "outputs" / "phase_8_1b_gold_full_lora" / "lora_gold_full"),
        "formataligned": str(PROJECT_ROOT / "outputs" / "phase_8_2_lora_degradation_debug" / "lora_format_aligned_200"),
    }

    results = {}

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load base model
        log("  Loading base model...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

        # Test prompts
        test_prompts = [
            "什么是碳纤维？",
            "碳纤维的主要应用领域有哪些？",
            "请解释碳纤维的制造工艺。",
        ]

        # Get base outputs
        base_outputs = []
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                outputs = base_model.generate(**inputs, max_new_tokens=50, do_sample=False)
            response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            base_outputs.append(response[:100])

        del base_model
        torch.cuda.empty_cache()

        # Test each adapter
        for adapter_name, adapter_path in adapters.items():
            if not os.path.exists(adapter_path):
                results[adapter_name] = {"status": "not_found"}
                log(f"  {adapter_name}: not found")
                continue

            log(f"  Testing {adapter_name}...")
            try:
                from peft import PeftModel

                model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
                )
                model = PeftModel.from_pretrained(model, adapter_path)

                # Check adapter config
                adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
                has_config = os.path.exists(adapter_config_path)

                # Get adapter outputs
                adapter_outputs = []
                for prompt in test_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                    response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    adapter_outputs.append(response[:100])

                # Check if outputs differ from base
                differs = sum(1 for a, b in zip(adapter_outputs, base_outputs) if a != b)

                results[adapter_name] = {
                    "status": "active",
                    "has_config": has_config,
                    "outputs_differ_from_base": differs,
                    "total_tests": len(test_prompts),
                    "example_outputs": adapter_outputs[:2],
                }

                log(f"    {adapter_name}: active, {differs}/{len(test_prompts)} outputs differ")

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                results[adapter_name] = {"status": "error", "error": str(e)[:200]}
                log(f"    {adapter_name}: error - {str(e)[:100]}")

        del processor

    except Exception as e:
        log(f"  Error: {str(e)[:100]}")

    return results


def _reevaluate_all(model_path: str, log) -> dict:
    """Re-evaluate all models on full 150 items."""
    import torch

    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix"
    manifest_path = eval_dir / "canonical_eval_manifest_150.jsonl"
    if not manifest_path.exists():
        return {"status": "skipped", "reason": "manifest_not_found"}

    eval_items = load_jsonl(manifest_path)
    log(f"  Evaluating {len(eval_items)} items...")

    adapters = {
        "base": None,
        "gold100": str(PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100"),
        "goldfull": str(PROJECT_ROOT / "outputs" / "phase_8_1b_gold_full_lora" / "lora_gold_full"),
        "formataligned": str(PROJECT_ROOT / "outputs" / "phase_8_2_lora_degradation_debug" / "lora_format_aligned_200"),
    }

    all_results = {}

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        for model_name, adapter_path in adapters.items():
            log(f"  Evaluating {model_name}...")

            try:
                if adapter_path and os.path.exists(adapter_path):
                    from peft import PeftModel
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
                    )
                    model = PeftModel.from_pretrained(model, adapter_path)
                else:
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
                    )

                # Evaluate all items
                outputs = []
                correct = 0
                scored = 0
                skipped = 0

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
                        skipped += 1
                        outputs.append({
                            "eval_index": item.get("eval_index"),
                            "benchmark_id": item.get("benchmark_id", ""),
                            "status": "failed",
                            "error": str(e)[:100],
                        })

                accuracy = correct / max(scored, 1)

                all_results[model_name] = {
                    "total": len(eval_items),
                    "scored": scored,
                    "skipped": skipped,
                    "correct": correct,
                    "accuracy": round(accuracy, 4),
                    "accuracy_over_all": round(correct / max(len(eval_items), 1), 4),
                }

                # Save outputs
                save_jsonl(outputs, eval_dir / f"{model_name}_outputs_150.jsonl")

                log(f"    {model_name}: {correct}/{scored} = {accuracy:.1%} ({skipped} skipped)")

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                all_results[model_name] = {"status": "error", "error": str(e)[:200]}
                log(f"    {model_name}: error - {str(e)[:100]}")

        del processor

    except Exception as e:
        log(f"  Error: {str(e)[:100]}")

    # Generate comparison
    comparison = {
        "base_accuracy": all_results.get("base", {}).get("accuracy", 0),
        "gold100_vs_base": all_results.get("gold100", {}).get("accuracy", 0) - all_results.get("base", {}).get("accuracy", 0),
        "goldfull_vs_base": all_results.get("goldfull", {}).get("accuracy", 0) - all_results.get("base", {}).get("accuracy", 0),
        "formataligned_vs_base": all_results.get("formataligned", {}).get("accuracy", 0) - all_results.get("base", {}).get("accuracy", 0),
    }

    # Save comparison CSV
    with open(eval_dir / "base_vs_all_adapters_150.csv", "w") as f:
        f.write("Model,Total,Scored,Correct,Accuracy,vs_Base\n")
        for name, data in all_results.items():
            if isinstance(data, dict) and "accuracy" in data:
                vs_base = data.get("accuracy", 0) - all_results.get("base", {}).get("accuracy", 0)
                f.write(f"{name},{data.get('total',0)},{data.get('scored',0)},{data.get('correct',0)},{data.get('accuracy',0):.4f},{vs_base:+.4f}\n")

    result = {
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "outputs"} for k, v in all_results.items() if isinstance(v, dict)},
        "comparison": comparison,
        "status": "complete",
    }
    return result


def _check_answer(response: str, gold: str, options: list, expected_format: str) -> bool:
    """Check if response matches gold answer."""
    if expected_format == "multiple_choice_letter" and options:
        pred_match = re.search(r'([A-H])', response.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            return pred_match.group(1) == gold_match.group(1)

    # Keyword match for other formats
    gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold[:200]))
    resp_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', response[:200]))
    if gold_keywords and len(gold_keywords & resp_keywords) / len(gold_keywords) > 0.3:
        return True

    return False


def _verify_label_masking(log) -> dict:
    """Verify label masking in training code."""
    # Check training code
    train_code_paths = [
        PROJECT_ROOT / "scripts" / "run_phase_8_1b_lora_gold_full.py",
        PROJECT_ROOT / "scripts" / "run_phase_8_1a_qwen_lora_pilot.py",
    ]

    findings = []
    label_masking_correct = True

    for code_path in train_code_paths:
        if not code_path.exists():
            continue

        with open(code_path) as f:
            content = f.read()

        # Check for label masking
        if "labels" in content and "-100" in content:
            findings.append(f"{code_path.name}: has label masking with -100")
        else:
            findings.append(f"{code_path.name}: MISSING label masking")
            label_masking_correct = False

        # Check for SimpleDataset class
        if "class SimpleDataset" in content:
            findings.append(f"{code_path.name}: uses SimpleDataset class")

            # Check if labels are properly masked
            if "labels[attention_mask == 0] = -100" in content:
                findings.append(f"{code_path.name}: masks padding tokens")
            else:
                findings.append(f"{code_path.name}: does NOT mask padding tokens")
                label_masking_correct = False

    result = {
        "label_masking_correct": label_masking_correct,
        "findings": findings,
        "note": "Current implementation masks padding but may not mask user/system tokens",
    }

    log(f"  Label masking: {'correct' if label_masking_correct else 'needs review'}")
    return result


def _update_root_cause(bug_audit, eval_results, adapter_check, label_check, log) -> dict:
    """Update root cause analysis."""
    root_causes = []

    # Denominator bug
    if bug_audit.get("bug_found"):
        root_causes.append({
            "cause": "evaluation_denominator_bug",
            "evidence": bug_audit.get("bug_code", ""),
            "severity": "critical",
            "fixed": True,
            "impact": "Only 50/150 items were evaluated",
        })

    # Check if adapters are active
    for adapter_name, check in adapter_check.items():
        if isinstance(check, dict) and check.get("status") == "active":
            if check.get("outputs_differ_from_base", 0) == 0:
                root_causes.append({
                    "cause": f"adapter_{adapter_name}_no_effect",
                    "evidence": "Adapter outputs identical to base",
                    "severity": "high",
                    "fixed": False,
                })

    # Check corrected results
    if eval_results.get("comparison"):
        comp = eval_results["comparison"]
        if comp.get("goldfull_vs_base", 0) > 0:
            root_causes.append({
                "cause": "gold_full_actually_improves",
                "evidence": f"After fix: Gold_Full vs Base = {comp['goldfull_vs_base']:+.4f}",
                "severity": "info",
                "fixed": True,
            })
        elif comp.get("goldfull_vs_base", 0) < 0:
            root_causes.append({
                "cause": "gold_full_still_degrades",
                "evidence": f"After fix: Gold_Full vs Base = {comp['goldfull_vs_base']:+.4f}",
                "severity": "high",
                "fixed": False,
            })

    analysis = {
        "root_causes": root_causes,
        "denominator_bug_fixed": bug_audit.get("bug_found", False),
        "recommendation": _determine_recommendation(root_causes, eval_results),
    }

    log(f"  Recommendation: {analysis['recommendation']}")
    return analysis


def _determine_recommendation(root_causes: list, eval_results: dict) -> str:
    """Determine recommendation based on findings."""
    # Check if Gold_Full improves after fix
    if eval_results.get("comparison"):
        comp = eval_results["comparison"]
        if comp.get("goldfull_vs_base", 0) > 0.05:
            return "proceed_to_phase_8_3"
        elif comp.get("goldfull_vs_base", 0) > 0:
            return "proceed_with_caution"
        else:
            return "stop_finetuning_experiments"

    return "needs_more_analysis"


def _generate_report(bug_audit, manifest_stats, adapter_check, eval_results,
                     label_check, root_cause, report_dir, log):
    """Generate Phase 8.2.5 report."""
    md = "# Phase 8.2.5: Evaluation Fix Report\n\n"

    md += "## 1. Denominator Bug Audit\n\n"
    md += f"- Bug found: {bug_audit.get('bug_found', False)}\n"
    md += f"- Location: {bug_audit.get('bug_location', 'N/A')}\n"
    md += f"- Code: `{bug_audit.get('bug_code', 'N/A')}`\n"
    md += f"- Eval set size: {bug_audit.get('eval_set_size', 0)}\n\n"

    md += "## 2. Canonical Manifest\n\n"
    md += f"- Total items: {manifest_stats.get('total', 0)}\n"
    md += f"- Format distribution: {manifest_stats.get('format_distribution', {})}\n\n"

    md += "## 3. Adapter Activation\n\n"
    for name, check in adapter_check.items():
        if isinstance(check, dict):
            md += f"- {name}: {check.get('status', 'unknown')}\n"
    md += "\n"

    md += "## 4. Full 150-Item Re-evaluation\n\n"
    if eval_results.get("comparison"):
        comp = eval_results["comparison"]
        md += "| Model | Accuracy | vs Base |\n|-------|----------|--------|\n"
        for name, data in eval_results.items():
            if isinstance(data, dict) and "accuracy" in data:
                vs_base = data["accuracy"] - comp.get("base_accuracy", 0)
                md += f"| {name} | {data['accuracy']:.1%} | {vs_base:+.1%} |\n"
    md += "\n"

    md += "## 5. Label Masking\n\n"
    md += f"- Correct: {label_check.get('label_masking_correct', 'unknown')}\n"
    md += f"- Note: {label_check.get('note', '')}\n\n"

    md += "## 6. Root Cause\n\n"
    for cause in root_cause.get("root_causes", []):
        md += f"- **{cause['cause']}**: {cause.get('evidence', '')}\n"
    md += f"\n**Recommendation**: {root_cause.get('recommendation', 'unknown')}\n"

    with open(report_dir / "PHASE_8_2_5_REPORT.md", "w") as f:
        f.write(md)

    log("  Report generated")


if __name__ == "__main__":
    main()
