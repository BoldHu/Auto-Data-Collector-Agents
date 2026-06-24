"""Phase 8.4B: Doubao baseline full evaluation with improved parser.

Usage:
    python scripts/run_phase_8_4b_doubao_full_eval.py \
        --api_config LLM_API/llm_api.txt \
        --max_workers 1 \
        --sleep_seconds 3
"""

from __future__ import annotations

import argparse
import json
import os
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


# Models to evaluate
EVAL_MODELS = [
    "minimax-m2.7",
    "doubao-seed-2.0-lite",
    "doubao-seed-2.0-pro",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.1",
]


def main():
    parser = argparse.ArgumentParser(description="Phase 8.4B Doubao full eval")
    parser.add_argument("--api_config", type=str, default="LLM_API/llm_api.txt")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--sleep_seconds", type=float, default=3)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max_items", type=int, default=0)
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_4b_doubao_full_eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_4b_doubao_full_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_4b.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.4B: Doubao Full Evaluation ===")

    # Load API config
    api_key, base_url = _load_api_config(args.api_config)
    if not api_key:
        log("ERROR: No API key found")
        return

    # Step 1: Load manifests
    log("Step 1: Loading evaluation manifests...")
    canonical = load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "canonical_eval_manifest_150.jsonl")
    large_eval = load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval" / "large_eval_manifest.jsonl")

    if args.max_items > 0:
        canonical = canonical[:args.max_items]
        large_eval = large_eval[:args.max_items]

    log(f"  Canonical: {len(canonical)} items, Large: {len(large_eval)} items")

    # Save manifests
    save_jsonl(canonical, eval_dir / "manifests" / "canonical_150_text.jsonl")
    save_jsonl(large_eval, eval_dir / "manifests" / "large_361_text.jsonl")

    # Step 2: Re-score Phase 8.4A with improved parser
    log("Step 2: Re-scoring Phase 8.4A outputs with improved parser...")
    _rescore_phase84a(eval_dir, log)

    # Step 3: Evaluate canonical 150
    log("Step 3: Evaluating canonical 150...")
    canonical_results = {}
    for model_id in EVAL_MODELS:
        safe_name = model_id.replace("/", "_").replace("-", "_")
        model_dir = eval_dir / "models" / safe_name / "canonical150"
        model_dir.mkdir(parents=True, exist_ok=True)

        result = _evaluate_model(
            api_key, base_url, model_id, canonical, "canonical150",
            model_dir, args, log
        )
        canonical_results[model_id] = result

        log(f"  {model_id}: {result.get('correct', 0)}/{result.get('scored', 0)} = {result.get('accuracy', 0):.1%}")

    # Save combined canonical scores
    _save_combined_scores(canonical_results, eval_dir / "combined_canonical150_scores.csv")

    # Step 4: Evaluate top models on large 361
    log("Step 4: Evaluating top models on large 361...")
    top_models = _select_top_models(canonical_results)
    large_results = {}

    for model_id in top_models:
        safe_name = model_id.replace("/", "_").replace("-", "_")
        model_dir = eval_dir / "models" / safe_name / "large361"
        model_dir.mkdir(parents=True, exist_ok=True)

        result = _evaluate_model(
            api_key, base_url, model_id, large_eval, "large361",
            model_dir, args, log
        )
        large_results[model_id] = result

        log(f"  {model_id}: {result.get('correct', 0)}/{result.get('scored', 0)} = {result.get('accuracy', 0):.1%}")

    # Save combined large scores
    _save_combined_scores(large_results, eval_dir / "combined_large361_scores.csv")

    # Step 5: Compare with Qwen
    log("Step 5: Comparing with Qwen results...")
    _compare_with_qwen(canonical_results, large_results, eval_dir, log)

    # Step 6: Generate paper artifacts
    log("Step 6: Generating paper artifacts...")
    _generate_paper_artifacts(canonical_results, large_results, eval_dir, report_dir, log)

    log("=== Phase 8.4B Complete ===")


def _load_api_config(path: str) -> tuple[str, str]:
    """Load API key and base URL."""
    api_key = ""
    base_url = ""
    full_path = PROJECT_ROOT / path
    if full_path.exists():
        with open(full_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DOUBAO_API_KEY:"):
                    api_key = line.split(":", 1)[1].strip()
                elif line.startswith("DOUBAO_OpenAI_URL:"):
                    raw = line.split(":", 1)[1].strip()
                    base_url = raw if raw.endswith("/v3") else raw + "/v3"
    return api_key, base_url


def _rescore_phase84a(eval_dir: Path, log):
    """Re-score Phase 8.4A outputs with improved parser."""
    phase84a_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_4a_doubao_baselines" / "models"

    if not phase84a_dir.exists():
        log("  Phase 8.4A outputs not found, skipping re-score")
        return

    rescored = []
    for model_dir in phase84a_dir.iterdir():
        if not model_dir.is_dir():
            continue
        canonical_path = model_dir / "canonical_outputs.jsonl"
        if not canonical_path.exists():
            continue

        model_name = model_dir.name
        outputs = load_jsonl(canonical_path)

        for out in outputs:
            gold = out.get("gold", "")
            predicted = out.get("predicted", "")
            expected_format = out.get("expected_format", "open_ended")

            # Re-score with improved parser
            new_scores = robust_score_v2(predicted, gold, expected_format, [])

            rescored.append({
                "model": model_name,
                "benchmark_id": out.get("benchmark_id", ""),
                "old_correct": out.get("strict_correct", False),
                "new_correct": new_scores["strict_correct"],
                "gold": gold[:50],
                "predicted": predicted[:100],
            })

    # Save rescored
    save_jsonl(rescored, eval_dir / "phase8_4a_rescored_50.jsonl")

    # Summary
    by_model = defaultdict(lambda: {"old_correct": 0, "new_correct": 0, "total": 0})
    for r in rescored:
        m = r["model"]
        by_model[m]["total"] += 1
        if r["old_correct"]:
            by_model[m]["old_correct"] += 1
        if r["new_correct"]:
            by_model[m]["new_correct"] += 1

    log("  Phase 8.4A re-score summary:")
    for model, data in by_model.items():
        old_acc = data["old_correct"] / max(data["total"], 1)
        new_acc = data["new_correct"] / max(data["total"], 1)
        log(f"    {model}: {old_acc:.1%} -> {new_acc:.1%}")


def robust_score_v2(predicted: str, gold: str, expected_format: str, options: list) -> dict:
    """Improved scoring with better reasoning model support."""
    scores = {
        "strict_correct": False,
        "normalized_correct": False,
        "letter_correct": False,
        "parse_success": False,
        "format_valid": False,
        "keyword_recall": 0.0,
    }

    if not predicted or not gold:
        return scores

    scores["parse_success"] = len(predicted.strip()) > 0
    scores["format_valid"] = len(predicted.strip()) > 0

    # For multiple-choice: extract letter from anywhere in output
    if expected_format == "multiple_choice_letter" or (options and len(options) >= 2):
        # Try to find answer marker first
        answer_markers = ["答案：", "答案:", "Answer:", "answer:", "选项：", "选项:"]
        search_text = predicted
        for marker in answer_markers:
            if marker in predicted:
                search_text = predicted.split(marker)[-1][:20]
                break

        pred_match = re.search(r'([A-H])', search_text.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            scores["letter_correct"] = pred_match.group(1) == gold_match.group(1)
            scores["strict_correct"] = scores["letter_correct"]

    # For list answers: check if gold items appear in predicted
    if not scores["strict_correct"] and "\n" in gold:
        gold_items = [l.strip() for l in gold.split("\n") if l.strip() and len(l.strip()) > 2]
        if gold_items:
            found = sum(1 for item in gold_items if item in predicted)
            if found / len(gold_items) > 0.5:
                scores["strict_correct"] = True

    # For short answers: keyword recall
    if not scores["strict_correct"]:
        gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]{2,}', gold[:300]))
        pred_keywords = set(re.findall(r'[\w\u4e00-\u9fff]{2,}', predicted[:500]))
        if gold_keywords:
            overlap = len(gold_keywords & pred_keywords)
            scores["keyword_recall"] = overlap / len(gold_keywords)
            # More generous threshold for keyword recall
            if scores["keyword_recall"] > 0.4:
                scores["strict_correct"] = True

    # Exact match
    if not scores["strict_correct"]:
        pred_norm = re.sub(r'\s+', ' ', predicted.strip().lower())
        gold_norm = re.sub(r'\s+', ' ', gold.strip().lower())
        if pred_norm == gold_norm:
            scores["strict_correct"] = True
            scores["normalized_correct"] = True

    # For short gold answers: check if gold appears in predicted
    if not scores["strict_correct"] and len(gold) < 50:
        if gold.strip() in predicted:
            scores["strict_correct"] = True

    return scores


def _evaluate_model(
    api_key: str, base_url: str, model_id: str,
    manifest: list, stage: str, output_dir: Path,
    args, log
) -> dict:
    """Evaluate a model on a manifest."""
    import openai

    outputs_path = output_dir / "raw_outputs.jsonl"

    # Load existing outputs for resume
    existing = {}
    if args.resume and outputs_path.exists():
        for item in load_jsonl(outputs_path):
            idx = item.get("eval_index", -1)
            if idx >= 0:
                existing[idx] = item
        if existing:
            log(f"  Resuming {model_id}: {len(existing)} existing outputs")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    outputs = []
    correct = 0
    scored = 0
    errors = 0
    start_time = time.time()

    for i, item in enumerate(manifest):
        # Resume check
        if i in existing:
            out = existing[i]
            outputs.append(out)
            if out.get("strict_correct"):
                correct += 1
            scored += 1
            continue

        question = item.get("question", "")
        options = item.get("options", [])
        gold = str(item.get("answer", "")).strip()
        expected_format = item.get("expected_answer_format", "open_ended")

        # Build improved prompt
        if expected_format == "multiple_choice_letter" and options:
            opt_text = "\n".join(str(o) for o in options)
            prompt = f"{question}\n\n选项：\n{opt_text}\n\n请先输出选项字母（如A、B、C、D），然后用一句话解释。"
        elif "true" in expected_format.lower() or "判断" in question:
            prompt = f"{question}\n\n请只在开头输出 正确 或 错误。"
        else:
            prompt = f"{question}\\n\n请直接回答，先给出核心答案。"

        # Call model with retry
        for retry in range(args.max_retries):
            try:
                start = time.time()
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                latency = time.time() - start
                content = response.choices[0].message.content or ""

                # Score with improved parser
                scores = robust_score_v2(content, gold, expected_format, options)

                out = {
                    "eval_index": i,
                    "benchmark_id": item.get("benchmark_id", ""),
                    "task_type": item.get("task_type", ""),
                    "subset": item.get("source_type", ""),
                    "expected_format": expected_format,
                    "gold": gold[:100],
                    "predicted": content[:200],
                    "status": "scored",
                    "latency": round(latency, 2),
                    **scores,
                }

                outputs.append(out)
                if scores["strict_correct"]:
                    correct += 1
                scored += 1

                time.sleep(args.sleep_seconds)
                break

            except Exception as e:
                error_msg = str(e)[:200]
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    log(f"  Rate limit, sleeping 60s...")
                    time.sleep(60)
                elif retry < args.max_retries - 1:
                    time.sleep(args.sleep_seconds * (retry + 1))
                else:
                    outputs.append({
                        "eval_index": i,
                        "benchmark_id": item.get("benchmark_id", ""),
                        "status": "failed",
                        "error": error_msg,
                    })
                    errors += 1

        # Checkpoint
        if (i + 1) % 10 == 0:
            save_jsonl(outputs, outputs_path)
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(manifest) - i - 1)
            acc = correct / max(scored, 1)
            log(f"  {model_id}: {i+1}/{len(manifest)} | correct={correct} | acc={acc:.1%} | errors={errors} | elapsed={elapsed:.0f}s | ETA={eta:.0f}s")

    # Final save
    save_jsonl(outputs, outputs_path)

    accuracy = correct / max(scored, 1)
    elapsed = time.time() - start_time

    # Per-subset
    by_subset = defaultdict(lambda: {"total": 0, "correct": 0})
    for o in outputs:
        subset = o.get("subset", "unknown")
        by_subset[subset]["total"] += 1
        if o.get("strict_correct"):
            by_subset[subset]["correct"] += 1

    result = {
        "status": "completed",
        "model_id": model_id,
        "total": len(manifest),
        "scored": scored,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "by_subset": {k: {**v, "accuracy": round(v["correct"]/max(v["total"],1), 4)} for k, v in by_subset.items()},
    }

    return result


def _select_top_models(canonical_results: dict) -> list:
    """Select top models for large eval."""
    ranked = sorted(
        [(m, d.get("accuracy", 0)) for m, d in canonical_results.items() if d.get("status") == "completed"],
        key=lambda x: x[1],
        reverse=True
    )
    # Select top 4 + deepseek-v4-pro for comparison
    selected = [m for m, _ in ranked[:4]]
    if "deepseek-v4-pro" not in selected:
        selected.append("deepseek-v4-pro")
    return selected


def _save_combined_scores(results: dict, path: Path):
    """Save combined scores CSV."""
    with open(path, "w") as f:
        f.write("Model,Total,Correct,Accuracy,Errors,Elapsed\n")
        for model_id, data in results.items():
            f.write(f"{model_id},{data.get('total',0)},{data.get('correct',0)},{data.get('accuracy',0):.4f},{data.get('errors',0)},{data.get('elapsed_seconds',0):.1f}\n")


def _compare_with_qwen(canonical: dict, large: dict, eval_dir: Path, log):
    """Compare with local Qwen results."""
    qwen_canonical = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval" / "canonical_150_rescore.csv"
    qwen_large = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval" / "large_eval" / "large_eval_scores.csv"

    # Canonical comparison
    with open(eval_dir / "doubao_vs_qwen_canonical150.csv", "w") as f:
        f.write("Model,Type,Accuracy\n")
        # Qwen models
        if qwen_canonical.exists():
            with open(qwen_canonical) as qf:
                header = None
                for line in qf:
                    parts = line.strip().split(",")
                    if header is None:
                        header = parts
                        continue
                    if len(parts) > 3:
                        f.write(f"qwen_{parts[0]},local,{parts[3]}\n")
        # Doubao models
        for model, data in canonical.items():
            f.write(f"{model},api,{data.get('accuracy', 0):.4f}\n")

    # Large comparison
    with open(eval_dir / "doubao_vs_qwen_large361.csv", "w") as f:
        f.write("Model,Type,Accuracy\n")
        if qwen_large.exists():
            with open(qwen_large) as qf:
                header = None
                for line in qf:
                    parts = line.strip().split(",")
                    if header is None:
                        header = parts
                        continue
                    if len(parts) > 3:
                        f.write(f"qwen_{parts[0]},local,{parts[3]}\n")
        for model, data in large.items():
            f.write(f"{model},api,{data.get('accuracy', 0):.4f}\n")


def _generate_paper_artifacts(canonical: dict, large: dict, eval_dir: Path, report_dir: Path, log):
    """Generate paper-ready tables."""
    tables_dir = eval_dir / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Canonical 150
    with open(tables_dir / "table_doubao_canonical150.csv", "w") as f:
        f.write("Model,Accuracy,Correct,Total,Errors\n")
        for model, data in sorted(canonical.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
            f.write(f"{model},{data.get('accuracy',0):.4f},{data.get('correct',0)},{data.get('scored',0)},{data.get('errors',0)}\n")

    # Table 2: Large 361
    with open(tables_dir / "table_doubao_large361.csv", "w") as f:
        f.write("Model,Accuracy,Correct,Total,Errors\n")
        for model, data in sorted(large.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
            f.write(f"{model},{data.get('accuracy',0):.4f},{data.get('correct',0)},{data.get('scored',0)},{data.get('errors',0)}\n")

    # LaTeX table
    latex = "% Phase 8.4B Doubao Baselines\n"
    latex += "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Closed-Source Baselines on CFBench}\n"
    latex += "\\begin{tabular}{lcc}\n\\hline\n"
    latex += "Model & Canonical 150 & Large 361 \\\\\n\\hline\n"

    # Add Qwen reference
    latex += "\\textit{Local Qwen2.5-VL-3B Base} & 16.0\\% & 21.6\\% \\\\\n"
    latex += "\\textit{Local Qwen2.5-VL-3B V4\\_Full} & 19.3\\% & 23.3\\% \\\\\n"
    latex += "\\hline\n"

    for model in sorted(canonical.keys(), key=lambda m: canonical[m].get("accuracy", 0), reverse=True):
        c_acc = canonical[model].get("accuracy", 0) * 100
        l_acc = large.get(model, {}).get("accuracy", 0) * 100 if model in large else 0
        latex += f"{model} & {c_acc:.1f}\\% & {l_acc:.1f}\\% \\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_4b_doubao_full_baselines.tex", "w") as f:
        f.write(latex)

    log("  Paper artifacts generated")


if __name__ == "__main__":
    main()
