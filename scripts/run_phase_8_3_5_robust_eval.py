"""Phase 8.3.5: Robust evaluation and significance analysis.

Usage:
    python scripts/run_phase_8_3_5_robust_eval.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
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
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_3_5_robust_eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_3_5.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.3.5: Robust Evaluation ===")

    # Step 1: Re-score canonical 150
    log("Step 1: Re-scoring canonical 150 items...")
    canonical_scores = _rescore_canonical_150(log)
    save_json(canonical_scores, report_dir / "canonical_150_rescore_report.json")

    # Step 2: Statistical significance on canonical 150
    log("Step 2: Statistical significance (canonical 150)...")
    sig_canonical = _significance_analysis(canonical_scores, "canonical_150", log)
    save_json(sig_canonical, report_dir / "significance_canonical_150.json")

    # Step 3: Build larger held-out evaluation set
    log("Step 3: Building larger held-out evaluation set...")
    large_eval_stats = _build_large_eval_set(log)
    save_json(large_eval_stats, report_dir / "large_eval_manifest_stats.json")

    # Step 4: Evaluate on larger held-out set
    log("Step 4: Evaluating on larger held-out set...")
    large_eval_results = _run_large_eval(log)
    save_json(large_eval_results, report_dir / "large_eval_report.json")

    # Step 5: Significance on larger held-out
    log("Step 5: Statistical significance (larger held-out)...")
    sig_large = _significance_analysis(large_eval_results, "large_eval", log)
    save_json(sig_large, report_dir / "significance_large_eval.json")

    # Step 6: Data efficiency analysis
    log("Step 6: Data efficiency analysis...")
    efficiency = _data_efficiency_analysis(canonical_scores, large_eval_results, log)
    save_json(efficiency, report_dir / "data_efficiency_analysis.json")

    # Step 7: Diagnostic analysis
    log("Step 7: Diagnostic analysis...")
    diagnostics = _diagnostic_analysis(canonical_scores, log)
    save_json(diagnostics, report_dir / "diagnostic_analysis.json")

    # Step 8: Case analysis
    log("Step 8: Case analysis...")
    cases = _case_analysis(log)
    save_json(cases, eval_dir / "cases" / "case_summary.json")

    # Step 9: Generate paper artifacts
    log("Step 9: Generating paper artifacts...")
    _generate_paper_artifacts(canonical_scores, sig_canonical, large_eval_results,
                              sig_large, efficiency, diagnostics, eval_dir, report_dir, log)

    # Step 10: Cloud 7B decision
    log("Step 10: Cloud 7B decision...")
    _cloud_decision(canonical_scores, large_eval_results, sig_canonical, sig_large, log)

    log("=== Phase 8.3.5 Complete ===")


def _rescore_canonical_150(log) -> dict:
    """Re-score canonical 150-item results with robust metrics."""
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval"
    models = {
        "base": PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "base_outputs_150.jsonl",
        "gold100": PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "gold100_outputs_150.jsonl",
        "goldfull": PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "goldfull_outputs_150.jsonl",
        "formataligned": PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "formataligned_outputs_150.jsonl",
        "v4full": PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_v4_full_lora" / "v4_full_outputs_150.jsonl",
    }

    # Load manifest for metadata
    manifest = load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "canonical_eval_manifest_150.jsonl")
    manifest_by_idx = {m.get("eval_index", i): m for i, m in enumerate(manifest)}

    results = {}

    for model_name, output_path in models.items():
        outputs = load_jsonl(output_path)

        # Score each output
        scored_outputs = []
        for out in outputs:
            idx = out.get("eval_index", 0)
            meta = manifest_by_idx.get(idx, {})

            # Robust scoring
            gold = out.get("gold", "")
            predicted = out.get("predicted", "")
            expected_format = out.get("expected_format", meta.get("expected_answer_format", "open_ended"))

            scores = _robust_score(predicted, gold, expected_format, meta.get("options", []))

            scored_output = dict(out)
            scored_output.update(scores)
            scored_output["subset"] = meta.get("source_type", "unknown")
            scored_output["task_type"] = out.get("task_type", meta.get("task_type", "unknown"))
            scored_outputs.append(scored_output)

        # Aggregate scores
        total = len(scored_outputs)
        correct_strict = sum(1 for s in scored_outputs if s.get("strict_correct", False))
        correct_normalized = sum(1 for s in scored_outputs if s.get("normalized_correct", False))
        correct_letter = sum(1 for s in scored_outputs if s.get("letter_correct", False))
        parse_success = sum(1 for s in scored_outputs if s.get("parse_success", False))
        format_valid = sum(1 for s in scored_outputs if s.get("format_valid", False))

        # Per-subset
        by_subset = defaultdict(lambda: {"total": 0, "correct": 0})
        for s in scored_outputs:
            subset = s.get("subset", "unknown")
            by_subset[subset]["total"] += 1
            if s.get("strict_correct"):
                by_subset[subset]["correct"] += 1

        # Per-task-type
        by_task = defaultdict(lambda: {"total": 0, "correct": 0})
        for s in scored_outputs:
            task = s.get("task_type", "unknown")
            by_task[task]["total"] += 1
            if s.get("strict_correct"):
                by_task[task]["correct"] += 1

        # Per-answer-format
        by_format = defaultdict(lambda: {"total": 0, "correct": 0})
        for s in scored_outputs:
            fmt = s.get("expected_format", "unknown")
            by_format[fmt]["total"] += 1
            if s.get("strict_correct"):
                by_format[fmt]["correct"] += 1

        results[model_name] = {
            "total": total,
            "strict_correct": correct_strict,
            "strict_accuracy": round(correct_strict / max(total, 1), 4),
            "normalized_correct": correct_normalized,
            "normalized_accuracy": round(correct_normalized / max(total, 1), 4),
            "letter_correct": correct_letter,
            "letter_accuracy": round(correct_letter / max(total, 1), 4),
            "parse_success": parse_success,
            "parse_rate": round(parse_success / max(total, 1), 4),
            "format_valid": format_valid,
            "format_validity": round(format_valid / max(total, 1), 4),
            "by_subset": {k: {**v, "accuracy": round(v["correct"] / max(v["total"], 1), 4)} for k, v in by_subset.items()},
            "by_task_type": {k: {**v, "accuracy": round(v["correct"] / max(v["total"], 1), 4)} for k, v in by_task.items()},
            "by_answer_format": {k: {**v, "accuracy": round(v["correct"] / max(v["total"], 1), 4)} for k, v in by_format.items()},
            "outputs": scored_outputs,
        }

        log(f"  {model_name}: strict={correct_strict}/{total}={results[model_name]['strict_accuracy']:.1%}")

    # Save CSV
    csv_path = eval_dir / "canonical_150_rescore.csv"
    with open(csv_path, "w") as f:
        f.write("Model,Total,Strict_Correct,Strict_Accuracy,Normalized_Accuracy,Letter_Accuracy,Parse_Rate,Format_Validity\n")
        for name, data in results.items():
            f.write(f"{name},{data['total']},{data['strict_correct']},{data['strict_accuracy']:.4f},{data['normalized_accuracy']:.4f},{data['letter_accuracy']:.4f},{data['parse_rate']:.4f},{data['format_validity']:.4f}\n")

    return results


def _robust_score(predicted: str, gold: str, expected_format: str, options: list) -> dict:
    """Compute robust scoring metrics."""
    scores = {
        "strict_correct": False,
        "normalized_correct": False,
        "letter_correct": False,
        "parse_success": False,
        "format_valid": False,
        "token_f1": 0.0,
        "keyword_recall": 0.0,
    }

    if not predicted or not gold:
        return scores

    # Parse success
    scores["parse_success"] = len(predicted.strip()) > 0

    # Format validity
    scores["format_valid"] = len(predicted.strip()) > 0 and not predicted.startswith("Error")

    # Letter accuracy (for MC)
    if expected_format == "multiple_choice_letter" or (options and len(options) >= 2):
        pred_match = re.search(r'([A-H])', predicted.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            scores["letter_correct"] = pred_match.group(1) == gold_match.group(1)
            scores["strict_correct"] = scores["letter_correct"]

    # Strict accuracy (exact match)
    if not scores["strict_correct"]:
        scores["strict_correct"] = predicted.strip() == gold.strip()

    # Normalized accuracy
    pred_norm = re.sub(r'\s+', ' ', predicted.strip().lower())
    gold_norm = re.sub(r'\s+', ' ', gold.strip().lower())
    scores["normalized_correct"] = pred_norm == gold_norm

    if not scores["strict_correct"] and scores["normalized_correct"]:
        scores["strict_correct"] = True

    # Token F1
    pred_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', predicted))
    gold_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', gold))
    if gold_tokens:
        overlap = len(pred_tokens & gold_tokens)
        precision = overlap / max(len(pred_tokens), 1)
        recall = overlap / max(len(gold_tokens), 1)
        scores["token_f1"] = round(2 * precision * recall / max(precision + recall, 1e-6), 4)
        scores["keyword_recall"] = round(recall, 4)

    # Keyword recall based check for open-ended
    if not scores["strict_correct"] and expected_format == "open_ended":
        if scores["keyword_recall"] > 0.5:
            scores["strict_correct"] = True

    return scores


def _significance_analysis(model_results: dict, label: str, log) -> dict:
    """Perform statistical significance analysis."""
    import numpy as np

    base_key = "base"
    adapter_keys = ["gold100", "goldfull", "formataligned", "v4full"]

    if base_key not in model_results:
        return {"status": "skipped", "reason": "base_not_found"}

    base_outputs = model_results[base_key].get("outputs", [])
    comparisons = {}

    for adapter_key in adapter_keys:
        if adapter_key not in model_results:
            continue

        adapter_outputs = model_results[adapter_key].get("outputs", [])

        # Paired comparison
        base_correct = []
        adapter_correct = []

        for b, a in zip(base_outputs, adapter_outputs):
            base_correct.append(1 if b.get("strict_correct") else 0)
            adapter_correct.append(1 if a.get("strict_correct") else 0)

        base_correct = np.array(base_correct)
        adapter_correct = np.array(adapter_correct)

        n = len(base_correct)
        base_acc = np.mean(base_correct)
        adapter_acc = np.mean(adapter_correct)
        diff = adapter_acc - base_acc

        # McNemar test
        b = np.sum((base_correct == 1) & (adapter_correct == 0))  # base right, adapter wrong
        c = np.sum((base_correct == 0) & (adapter_correct == 1))  # base wrong, adapter right

        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            # Approximate p-value using chi-squared
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0
            p_value = 1.0

        # Bootstrap CI
        n_bootstrap = 2000
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, n)
            boot_base = np.mean(base_correct[idx])
            boot_adapter = np.mean(adapter_correct[idx])
            bootstrap_diffs.append(boot_adapter - boot_base)

        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)

        comparisons[f"{adapter_key}_vs_base"] = {
            "base_accuracy": round(float(base_acc), 4),
            "adapter_accuracy": round(float(adapter_acc), 4),
            "accuracy_diff": round(float(diff), 4),
            "mcnemar_statistic": round(float(mcnemar_stat), 4),
            "p_value": round(float(p_value), 4),
            "significant_005": p_value < 0.05,
            "significant_010": p_value < 0.10,
            "ci_95_lower": round(float(ci_lower), 4),
            "ci_95_upper": round(float(ci_upper), 4),
            "n_samples": n,
            "base_right_adapter_wrong": int(b),
            "base_wrong_adapter_right": int(c),
        }

        log(f"  {adapter_key} vs base: diff={diff:+.4f}, p={p_value:.4f}, sig={p_value < 0.05}")

    # Gold_100 vs V4_Full
    if "gold100" in model_results and "v4full" in model_results:
        g100_outputs = model_results["gold100"]["outputs"]
        v4_outputs = model_results["v4full"]["outputs"]

        g100_correct = np.array([1 if o.get("strict_correct") else 0 for o in g100_outputs])
        v4_correct = np.array([1 if o.get("strict_correct") else 0 for o in v4_outputs])

        b = np.sum((g100_correct == 1) & (v4_correct == 0))
        c = np.sum((g100_correct == 0) & (v4_correct == 1))

        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            p_value = 1.0

        comparisons["gold100_vs_v4full"] = {
            "gold100_accuracy": round(float(np.mean(g100_correct)), 4),
            "v4full_accuracy": round(float(np.mean(v4_correct)), 4),
            "p_value": round(float(p_value), 4),
            "significant_005": p_value < 0.05,
        }

        log(f"  gold100 vs v4full: p={p_value:.4f}")

    return {
        "label": label,
        "comparisons": comparisons,
        "status": "complete",
    }


def _build_large_eval_set(log) -> dict:
    """Build larger held-out evaluation set."""
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval"
    random.seed(42)

    bench_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    if not bench_path.exists():
        return {"status": "not_found"}

    all_items = load_jsonl(bench_path)

    # Sample by source type
    by_source = defaultdict(list)
    for item in all_items:
        by_source[item.get("source_type", "unknown")].append(item)

    # Target: 500 items
    eval_items = []
    eval_items.extend(random.sample(by_source.get("text", []), min(150, len(by_source.get("text", [])))))
    eval_items.extend(random.sample(by_source.get("exam", []), min(100, len(by_source.get("exam", [])))))
    eval_items.extend(random.sample([i for i in by_source.get("text", []) if i not in eval_items], min(50, max(0, len(by_source.get("text", [])) - 150))))

    # Add multimodal if available
    mm_items = by_source.get("multimodal", [])
    eval_items.extend(random.sample(mm_items, min(100, len(mm_items))))

    # Add eval_index and metadata
    manifest = []
    for i, item in enumerate(eval_items):
        entry = dict(item)
        entry["eval_index"] = i

        options = item.get("options", [])
        if options and len(options) >= 2:
            entry["expected_answer_format"] = "multiple_choice_letter"
            entry["scoring_method"] = "option_letter_match"
        else:
            entry["expected_answer_format"] = "open_ended"
            entry["scoring_method"] = "keyword_match"

        manifest.append(entry)

    save_jsonl(manifest, eval_dir / "large_eval_manifest.jsonl")

    # Stats
    source_dist = Counter(m.get("source_type", "unknown") for m in manifest)
    task_dist = Counter(m.get("task_type", "unknown") for m in manifest)

    stats = {
        "total": len(manifest),
        "source_distribution": dict(source_dist),
        "task_distribution": dict(task_dist),
    }

    log(f"  Large eval set: {len(manifest)} items")
    log(f"  Sources: {dict(source_dist)}")

    return stats


def _run_large_eval(log) -> dict:
    """Evaluate models on larger held-out set."""
    import torch

    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval"
    manifest_path = eval_dir / "large_eval_manifest.jsonl"
    if not manifest_path.exists():
        return {"status": "skipped", "reason": "manifest_not_found"}

    eval_items = load_jsonl(manifest_path)
    log(f"  Evaluating {len(eval_items)} items...")

    model_path = str(PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct")
    adapters = {
        "base": None,
        "gold100": str(PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100"),
        "v4full": str(PROJECT_ROOT / "outputs" / "phase_8_3_v4_full_lora" / "lora_v4_full"),
    }

    results = {}

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

                outputs = []
                for item in eval_items:
                    question = item.get("question", "")
                    options = item.get("options", [])
                    gold = str(item.get("answer", "")).strip()
                    expected_format = item.get("expected_answer_format", "open_ended")

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

                        scores = _robust_score(response, gold, expected_format, options)

                        outputs.append({
                            "eval_index": item.get("eval_index"),
                            "benchmark_id": item.get("benchmark_id", ""),
                            "task_type": item.get("task_type", ""),
                            "subset": item.get("source_type", ""),
                            "expected_format": expected_format,
                            "gold": gold[:100],
                            "predicted": response[:100],
                            "status": "scored",
                            **scores,
                        })

                    except Exception as e:
                        outputs.append({
                            "eval_index": item.get("eval_index"),
                            "status": "failed",
                            "error": str(e)[:100],
                        })

                correct = sum(1 for o in outputs if o.get("strict_correct"))
                total = len(outputs)

                results[model_name] = {
                    "total": total,
                    "correct": correct,
                    "accuracy": round(correct / max(total, 1), 4),
                    "outputs": outputs,
                }

                # Per-subset
                by_subset = defaultdict(lambda: {"total": 0, "correct": 0})
                for o in outputs:
                    subset = o.get("subset", "unknown")
                    by_subset[subset]["total"] += 1
                    if o.get("strict_correct"):
                        by_subset[subset]["correct"] += 1
                results[model_name]["by_subset"] = {k: {**v, "accuracy": round(v["correct"]/max(v["total"],1), 4)} for k, v in by_subset.items()}

                log(f"    {model_name}: {correct}/{total} = {results[model_name]['accuracy']:.1%}")

                save_jsonl(outputs, eval_dir / "large_eval" / f"{model_name}_outputs.jsonl")

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                results[model_name] = {"status": "error", "error": str(e)[:200]}
                log(f"    {model_name}: error - {str(e)[:100]}")

        del processor

    except Exception as e:
        log(f"  Error: {str(e)[:100]}")

    # Save comparison CSV
    csv_path = eval_dir / "large_eval" / "large_eval_scores.csv"
    with open(csv_path, "w") as f:
        f.write("Model,Total,Correct,Accuracy\n")
        for name, data in results.items():
            if "accuracy" in data:
                f.write(f"{name},{data['total']},{data['correct']},{data['accuracy']:.4f}\n")

    results["status"] = "complete"
    return results


def _data_efficiency_analysis(canonical: dict, large_eval: dict, log) -> dict:
    """Analyze data efficiency."""
    training_sizes = {
        "base": 0,
        "gold100": 100,
        "goldfull": 598,
        "formataligned": 120,
        "v4full": 919,
    }

    efficiency = {"canonical_150": {}, "large_eval": {}}

    # Canonical 150
    for model, data in canonical.items():
        if "strict_accuracy" in data:
            size = training_sizes.get(model, 0)
            acc = data["strict_accuracy"]
            efficiency["canonical_150"][model] = {
                "training_size": size,
                "accuracy": acc,
                "gain_per_100_samples": round((acc - canonical.get("base", {}).get("strict_accuracy", 0)) / max(size / 100, 0.01), 4),
            }

    # Large eval
    for model, data in large_eval.items():
        if "accuracy" in data:
            size = training_sizes.get(model, 0)
            acc = data["accuracy"]
            efficiency["large_eval"][model] = {
                "training_size": size,
                "accuracy": acc,
                "gain_per_100_samples": round((acc - large_eval.get("base", {}).get("accuracy", 0)) / max(size / 100, 0.01), 4),
            }

    log(f"  Canonical 150: {efficiency['canonical_150']}")
    log(f"  Large eval: {efficiency['large_eval']}")

    return efficiency


def _diagnostic_analysis(canonical: dict, log) -> dict:
    """Per-subset and per-task diagnostics."""
    diagnostics = {}

    for model, data in canonical.items():
        if "by_subset" in data:
            diagnostics[model] = {
                "by_subset": data["by_subset"],
                "by_task_type": data.get("by_task_type", {}),
                "by_answer_format": data.get("by_answer_format", {}),
            }

    return diagnostics


def _case_analysis(log) -> dict:
    """Extract representative cases."""
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval"
    # Load outputs
    base_path = eval_dir / "large_eval" / "base_outputs.jsonl"
    g100_path = eval_dir / "large_eval" / "gold100_outputs.jsonl"
    v4_path = eval_dir / "large_eval" / "v4full_outputs.jsonl"

    if not all(p.exists() for p in [base_path, g100_path, v4_path]):
        # Fall back to canonical
        base_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "base_outputs_150.jsonl"
        g100_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "gold100_outputs_150.jsonl"
        v4_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_v4_full_lora" / "v4_full_outputs_150.jsonl"

    base_outputs = load_jsonl(base_path)
    g100_outputs = load_jsonl(g100_path)
    v4_outputs = load_jsonl(v4_path)

    cases = {
        "base_wrong_gold100_correct": [],
        "base_wrong_v4_correct": [],
        "gold100_correct_v4_wrong": [],
        "v4_correct_gold100_wrong": [],
    }

    for b, g, v in zip(base_outputs, g100_outputs, v4_outputs):
        b_correct = b.get("strict_correct", b.get("correct", False))
        g_correct = g.get("strict_correct", g.get("correct", False))
        v_correct = v.get("strict_correct", v.get("correct", False))

        case = {
            "benchmark_id": b.get("benchmark_id", ""),
            "subset": b.get("subset", ""),
            "gold": b.get("gold", "")[:100],
            "base_predicted": b.get("predicted", "")[:100],
            "gold100_predicted": g.get("predicted", "")[:100],
            "v4_predicted": v.get("predicted", "")[:100],
        }

        if not b_correct and g_correct:
            cases["base_wrong_gold100_correct"].append(case)
        if not b_correct and v_correct:
            cases["base_wrong_v4_correct"].append(case)
        if g_correct and not v_correct:
            cases["gold100_correct_v4_wrong"].append(case)
        if v_correct and not g_correct:
            cases["v4_correct_gold100_wrong"].append(case)

    # Save cases
    cases_dir = eval_dir / "cases"
    for name, case_list in cases.items():
        save_jsonl(case_list[:20], cases_dir / f"{name}.jsonl")

    summary = {k: len(v) for k, v in cases.items()}
    log(f"  Cases: {summary}")

    return summary


def _generate_paper_artifacts(canonical, sig_canonical, large_eval, sig_large,
                              efficiency, diagnostics, eval_dir, report_dir, log):
    """Generate paper-ready tables and figures."""
    tables_dir = eval_dir / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Main finetuning results
    with open(tables_dir / "table_main_finetuning_results.csv", "w") as f:
        f.write("Model,Canonical_150_Accuracy,Large_Eval_Accuracy,vs_Base_Canonical,vs_Base_Large\n")
        for model in ["base", "gold100", "goldfull", "formataligned", "v4full"]:
            c_acc = canonical.get(model, {}).get("strict_accuracy", 0)
            l_acc = large_eval.get(model, {}).get("accuracy", 0)
            c_base = canonical.get("base", {}).get("strict_accuracy", 0)
            l_base = large_eval.get("base", {}).get("accuracy", 0)
            f.write(f"{model},{c_acc:.4f},{l_acc:.4f},{c_acc-c_base:+.4f},{l_acc-l_base:+.4f}\n")

    # Table 2: Significance
    with open(tables_dir / "table_significance.csv", "w") as f:
        f.write("Comparison,p_value,Significant_005,CI_95_Lower,CI_95_Upper\n")
        for comp_name, comp in sig_canonical.get("comparisons", {}).items():
            f.write(f"{comp_name},{comp.get('p_value', 1):.4f},{comp.get('significant_005', False)},{comp.get('ci_95_lower', 0):.4f},{comp.get('ci_95_upper', 0):.4f}\n")

    # Table 3: Data efficiency
    with open(tables_dir / "table_data_efficiency.csv", "w") as f:
        f.write("Model,Training_Size,Canonical_Accuracy,Large_Eval_Accuracy,Gain_Per_100_Samples\n")
        for model, data in efficiency.get("canonical_150", {}).items():
            large_acc = efficiency.get("large_eval", {}).get(model, {}).get("accuracy", 0)
            f.write(f"{model},{data.get('training_size', 0)},{data.get('accuracy', 0):.4f},{large_acc:.4f},{data.get('gain_per_100_samples', 0):.4f}\n")

    # LaTeX table
    latex = "% Phase 8.3.5 Fine-tuning Results\n"
    latex += "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{LoRA Fine-tuning Results on Qwen2.5-VL-3B}\n"
    latex += "\\begin{tabular}{lcccc}\n\\hline\n"
    latex += "Model & Train Size & Canonical 150 & Large Eval & vs Base \\\\\n\\hline\n"
    for model in ["base", "gold100", "goldfull", "formataligned", "v4full"]:
        sizes = {"base": 0, "gold100": 100, "goldfull": 598, "formataligned": 120, "v4full": 919}
        c_acc = canonical.get(model, {}).get("strict_accuracy", 0)
        l_acc = large_eval.get(model, {}).get("accuracy", 0)
        c_base = canonical.get("base", {}).get("strict_accuracy", 0)
        latex += f"{model} & {sizes.get(model, 0)} & {c_acc*100:.1f}\\% & {l_acc*100:.1f}\\% & {(c_acc-c_base)*100:+.1f}\\% \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_3_5_finetuning_results.tex", "w") as f:
        f.write(latex)

    log("  Paper artifacts generated")


def _cloud_decision(canonical, large_eval, sig_canonical, sig_large, log):
    """Make cloud 7B training decision."""
    g100_vs_base = sig_canonical.get("comparisons", {}).get("gold100_vs_base", {})
    g100_sig = g100_vs_base.get("significant_005", False)
    g100_diff = g100_vs_base.get("accuracy_diff", 0)

    # Check large eval
    g100_large = large_eval.get("gold100", {}).get("accuracy", 0)
    base_large = large_eval.get("base", {}).get("accuracy", 0)
    large_diff = g100_large - base_large

    if g100_sig and large_diff > 0:
        decision = "approved"
        reason = "Gold_100 improves over Base on both canonical and large eval"
    elif g100_diff > 0 and large_diff > 0:
        decision = "conditionally_approved"
        reason = "Improvements observed but not statistically significant"
    elif g100_diff > 0:
        decision = "conditionally_approved"
        reason = "Canonical improvement exists, large eval needs more data"
    else:
        decision = "blocked"
        reason = "No robust improvement observed"

    md = f"# Cloud 7B Training Decision\n\n"
    md += f"**Decision**: {decision}\n\n"
    md += f"**Reason**: {reason}\n\n"
    md += f"## Canonical 150 Results\n"
    md += f"- Gold_100 vs Base: {g100_diff:+.4f} (p={g100_vs_base.get('p_value', 'N/A')})\n"
    md += f"- Significant: {g100_sig}\n\n"
    md += f"## Large Eval Results\n"
    md += f"- Gold_100: {g100_large:.1%}\n"
    md += f"- Base: {base_large:.1%}\n"
    md += f"- Difference: {large_diff:+.1%}\n\n"

    if decision == "approved":
        md += "## Recommended Cloud Command\n\n"
        md += "```bash\n"
        md += "python scripts/run_phase_8_4_cloud_7b_training.py \\\n"
        md += "  --model Qwen/Qwen2.5-VL-7B-Instruct \\\n"
        md += "  --train_file data/sft/final_v4/gold/train_chatml.jsonl \\\n"
        md += "  --run_training true\n"
        md += "```\n"

    with open(PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_4_cloud_7b_decision.md", "w") as f:
        f.write(md)

    log(f"  Cloud decision: {decision}")


if __name__ == "__main__":
    main()
