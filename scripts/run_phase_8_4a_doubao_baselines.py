"""Phase 8.4A: Doubao baseline evaluation on CFBench.

Usage:
    python scripts/run_phase_8_4a_doubao_baselines.py \
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


# Doubao model registry
DOUBAO_MODELS = [
    {
        "model_id": "doubao-seed-2.0-pro",
        "display_name": "Doubao Seed 2.0 Pro",
        "provider": "ByteDance",
        "supports_text": True,
        "supports_vision": True,
        "enabled": True,
        "priority": 1,
    },
    {
        "model_id": "doubao-seed-2.0-lite",
        "display_name": "Doubao Seed 2.0 Lite",
        "provider": "ByteDance",
        "supports_text": True,
        "supports_vision": True,
        "enabled": True,
        "priority": 2,
    },
    {
        "model_id": "deepseek-v4-flash",
        "display_name": "DeepSeek V4 Flash",
        "provider": "DeepSeek",
        "supports_text": True,
        "supports_vision": False,
        "enabled": True,
        "priority": 3,
    },
    {
        "model_id": "deepseek-v4-pro",
        "display_name": "DeepSeek V4 Pro",
        "provider": "DeepSeek",
        "supports_text": True,
        "supports_vision": False,
        "enabled": True,
        "priority": 4,
    },
    {
        "model_id": "deepseek-v3.2",
        "display_name": "DeepSeek V3.2",
        "provider": "DeepSeek",
        "supports_text": True,
        "supports_vision": False,
        "enabled": True,
        "priority": 5,
    },
    {
        "model_id": "glm-5.1",
        "display_name": "GLM 5.1",
        "provider": "Zhipu AI",
        "supports_text": True,
        "supports_vision": False,
        "enabled": True,
        "priority": 6,
    },
    {
        "model_id": "kimi-k2.6",
        "display_name": "Kimi K2.6",
        "provider": "Moonshot",
        "supports_text": True,
        "supports_vision": True,
        "enabled": True,
        "priority": 7,
    },
    {
        "model_id": "doubao-seed-code",
        "display_name": "Doubao Seed Code",
        "provider": "ByteDance",
        "supports_text": True,
        "supports_vision": True,
        "enabled": False,
        "priority": 8,
    },
    {
        "model_id": "minimax-m2.7",
        "display_name": "MiniMax M2.7",
        "provider": "MiniMax",
        "supports_text": True,
        "supports_vision": False,
        "enabled": True,
        "priority": 9,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Phase 8.4A Doubao baselines")
    parser.add_argument("--api_config", type=str, default="LLM_API/llm_api.txt")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--sleep_seconds", type=float, default=3)
    parser.add_argument("--model_sleep_seconds", type=float, default=30)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--max_items", type=int, default=0)
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_4a_doubao_baselines"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_4a_doubao_baselines"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_4a.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.4A: Doubao Baselines ===")

    # Load API config
    api_key, base_url = _load_api_config(args.api_config)
    if not api_key:
        log("ERROR: No API key found")
        return

    # Save registry
    save_json({"models": DOUBAO_MODELS}, report_dir / "doubao_model_registry.json")
    log(f"Registry: {len(DOUBAO_MODELS)} models, {sum(1 for m in DOUBAO_MODELS if m['enabled'])} enabled")

    # Step 1: Smoke test
    log("Step 1: Smoke testing enabled models...")
    smoke_results = _run_smoke_tests(api_key, base_url, args, log)
    save_json(smoke_results, report_dir / "doubao_smoke_test_report.json")

    if args.smoke_only:
        log("Smoke only mode, stopping.")
        return

    # Step 2: Load evaluation manifests
    log("Step 2: Loading evaluation manifests...")
    canonical = load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix" / "canonical_eval_manifest_150.jsonl")
    large_eval = load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval" / "large_eval_manifest.jsonl")

    if args.max_items > 0:
        canonical = canonical[:args.max_items]
        large_eval = large_eval[:args.max_items]

    log(f"  Canonical: {len(canonical)} items, Large: {len(large_eval)} items")

    # Step 3: Evaluate each model sequentially
    log("Step 3: Evaluating models sequentially...")
    all_results = {}

    enabled_models = [m for m in DOUBAO_MODELS if m["enabled"]]
    smoke_passed = set()

    for model_info in enabled_models:
        model_id = model_info["model_id"]
        safe_name = model_id.replace("/", "_").replace("-", "_")

        # Check smoke test
        smoke_key = model_id
        if smoke_key in smoke_results.get("models", {}):
            if smoke_results["models"][smoke_key].get("status") != "passed":
                log(f"  Skipping {model_id} (smoke failed)")
                continue
        smoke_passed.add(model_id)

        log(f"\n--- Evaluating {model_id} ---")
        model_dir = eval_dir / "models" / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Canonical 150
        canonical_results = _evaluate_model_on_manifest(
            api_key, base_url, model_id, canonical, "canonical",
            model_dir, args, log
        )

        # Stage 2: Larger held-out (if canonical completed)
        large_results = {"status": "skipped"}
        if canonical_results.get("status") == "completed":
            large_results = _evaluate_model_on_manifest(
                api_key, base_url, model_id, large_eval, "large",
                model_dir, args, log
            )

        all_results[model_id] = {
            "canonical": canonical_results,
            "large": large_results,
        }

        # Sleep between models
        log(f"  Sleeping {args.model_sleep_seconds}s before next model...")
        time.sleep(args.model_sleep_seconds)

    # Step 4: Generate comparison tables
    log("Step 4: Generating comparison tables...")
    _generate_comparison(all_results, eval_dir, report_dir, log)

    # Step 5: Load local Qwen results and compare
    log("Step 5: Comparing with local Qwen results...")
    _compare_with_qwen(all_results, eval_dir, report_dir, log)

    log("=== Phase 8.4A Complete ===")


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
                    if not raw.endswith("/v3"):
                        base_url = raw + "/v3"
                    else:
                        base_url = raw

    return api_key, base_url


def _run_smoke_tests(api_key: str, base_url: str, args, log) -> dict:
    """Run smoke tests on enabled models."""
    import openai

    results = {"models": {}}

    test_prompts = [
        {"type": "factual", "prompt": "什么是碳纤维？请用一句话回答。", "expected_keywords": ["碳", "纤维"]},
        {"type": "multiple_choice", "prompt": "碳纤维的主要成分是什么？\nA. 碳化硅\nB. 碳原子\nC. 石墨\nD. 聚丙烯\n请只输出选项字母。", "expected_keywords": ["B"]},
    ]

    for model_info in DOUBAO_MODELS:
        if not model_info["enabled"]:
            continue

        model_id = model_info["model_id"]
        log(f"  Smoke testing {model_id}...")

        model_result = {"status": "unknown", "tests": []}

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)

            for test in test_prompts:
                try:
                    start = time.time()
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": test["prompt"]}],
                        max_tokens=100,
                        temperature=0.0,
                    )
                    latency = time.time() - start
                    content = response.choices[0].message.content or ""

                    # Check if response contains expected keywords
                    has_keywords = any(kw in content for kw in test["expected_keywords"])

                    model_result["tests"].append({
                        "type": test["type"],
                        "response": content[:100],
                        "latency": round(latency, 2),
                        "has_keywords": has_keywords,
                    })

                    time.sleep(args.sleep_seconds)

                except Exception as e:
                    model_result["tests"].append({
                        "type": test["type"],
                        "error": str(e)[:200],
                    })

            # Determine status
            passed_tests = sum(1 for t in model_result["tests"] if t.get("has_keywords"))
            model_result["status"] = "passed" if passed_tests >= 1 else "failed"
            model_result["passed_tests"] = passed_tests
            model_result["total_tests"] = len(test_prompts)

            log(f"    Status: {model_result['status']} ({passed_tests}/{len(test_prompts)})")

        except Exception as e:
            model_result["status"] = "error"
            model_result["error"] = str(e)[:200]
            log(f"    Error: {str(e)[:100]}")

        results["models"][model_id] = model_result

    return results


def _evaluate_model_on_manifest(
    api_key: str, base_url: str, model_id: str,
    manifest: list, stage: str, output_dir: Path,
    args, log
) -> dict:
    """Evaluate a model on a manifest."""
    import openai

    # Check for existing outputs
    outputs_path = output_dir / f"{stage}_outputs.jsonl"
    existing_outputs = {}
    if args.resume and outputs_path.exists():
        for item in load_jsonl(outputs_path):
            idx = item.get("eval_index", -1)
            if idx >= 0:
                existing_outputs[idx] = item
        log(f"  Resuming: {len(existing_outputs)} existing outputs")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    outputs = []
    correct = 0
    scored = 0
    errors = 0
    total_latency = 0

    for i, item in enumerate(manifest):
        # Check if already done
        if i in existing_outputs:
            out = existing_outputs[i]
            outputs.append(out)
            if out.get("strict_correct"):
                correct += 1
            scored += 1
            continue

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

        # Call model
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
                total_latency += latency

                # Score
                scores = _robust_score(content, gold, expected_format, options)

                out = {
                    "eval_index": i,
                    "benchmark_id": item.get("benchmark_id", ""),
                    "task_type": item.get("task_type", ""),
                    "subset": item.get("source_type", ""),
                    "expected_format": expected_format,
                    "gold": gold[:100],
                    "predicted": content[:100],
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
                    log(f"  Rate limit hit, sleeping 60s...")
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

        # Save checkpoint every 10 items
        if (i + 1) % 10 == 0:
            save_jsonl(outputs, outputs_path)
            acc = correct / max(scored, 1)
            log(f"  {stage}: {i+1}/{len(manifest)} | correct={correct} | acc={acc:.1%} | errors={errors}")

    # Final save
    save_jsonl(outputs, outputs_path)

    accuracy = correct / max(scored, 1)
    avg_latency = total_latency / max(scored, 1)

    result = {
        "status": "completed",
        "total": len(manifest),
        "scored": scored,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "errors": errors,
        "avg_latency": round(avg_latency, 2),
    }

    # Per-subset
    by_subset = defaultdict(lambda: {"total": 0, "correct": 0})
    for o in outputs:
        subset = o.get("subset", "unknown")
        by_subset[subset]["total"] += 1
        if o.get("strict_correct"):
            by_subset[subset]["correct"] += 1
    result["by_subset"] = {k: {**v, "accuracy": round(v["correct"]/max(v["total"],1), 4)} for k, v in by_subset.items()}

    log(f"  {stage} complete: {correct}/{scored} = {accuracy:.1%}")

    return result


def _robust_score(predicted: str, gold: str, expected_format: str, options: list) -> dict:
    """Compute robust scoring metrics."""
    scores = {
        "strict_correct": False,
        "normalized_correct": False,
        "letter_correct": False,
        "parse_success": False,
        "format_valid": False,
    }

    if not predicted or not gold:
        return scores

    scores["parse_success"] = len(predicted.strip()) > 0
    scores["format_valid"] = len(predicted.strip()) > 0

    # Letter accuracy
    if expected_format == "multiple_choice_letter" or (options and len(options) >= 2):
        pred_match = re.search(r'([A-H])', predicted.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            scores["letter_correct"] = pred_match.group(1) == gold_match.group(1)
            scores["strict_correct"] = scores["letter_correct"]

    # Strict accuracy
    if not scores["strict_correct"]:
        scores["strict_correct"] = predicted.strip() == gold.strip()

    # Normalized
    pred_norm = re.sub(r'\s+', ' ', predicted.strip().lower())
    gold_norm = re.sub(r'\s+', ' ', gold.strip().lower())
    scores["normalized_correct"] = pred_norm == gold_norm
    if scores["normalized_correct"]:
        scores["strict_correct"] = True

    # Keyword recall for open-ended
    if not scores["strict_correct"] and expected_format == "open_ended":
        gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold[:200]))
        pred_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', predicted[:200]))
        if gold_keywords and len(gold_keywords & pred_keywords) / len(gold_keywords) > 0.5:
            scores["strict_correct"] = True

    return scores


def _generate_comparison(all_results: dict, eval_dir: Path, report_dir: Path, log):
    """Generate comparison tables."""
    # Canonical scores
    with open(eval_dir / "combined_canonical_scores.csv", "w") as f:
        f.write("Model,Total,Correct,Accuracy,Avg_Latency,Errors\n")
        for model_id, data in all_results.items():
            c = data.get("canonical", {})
            f.write(f"{model_id},{c.get('total',0)},{c.get('correct',0)},{c.get('accuracy',0):.4f},{c.get('avg_latency',0):.2f},{c.get('errors',0)}\n")

    # Large scores
    with open(eval_dir / "combined_large_scores.csv", "w") as f:
        f.write("Model,Total,Correct,Accuracy,Avg_Latency,Errors\n")
        for model_id, data in all_results.items():
            l = data.get("large", {})
            if l.get("status") == "completed":
                f.write(f"{model_id},{l.get('total',0)},{l.get('correct',0)},{l.get('accuracy',0):.4f},{l.get('avg_latency',0):.2f},{l.get('errors',0)}\n")

    log("  Comparison tables generated")


def _compare_with_qwen(all_results: dict, eval_dir: Path, report_dir: Path, log):
    """Compare with local Qwen results."""
    # Load Qwen canonical scores
    qwen_path = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval" / "canonical_150_rescore.csv"

    qwen_scores = {}
    if qwen_path.exists():
        with open(qwen_path) as f:
            header = None
            for line in f:
                parts = line.strip().split(",")
                if header is None:
                    header = parts
                    continue
                qwen_scores[parts[0]] = {
                    "accuracy": float(parts[3]) if len(parts) > 3 else 0,
                }

    # Combined table
    with open(eval_dir / "doubao_vs_qwen_canonical.csv", "w") as f:
        f.write("Model,Type,Canonical_Accuracy\n")
        for model, data in qwen_scores.items():
            f.write(f"qwen_{model},local,{data['accuracy']:.4f}\n")
        for model_id, data in all_results.items():
            c = data.get("canonical", {})
            f.write(f"{model_id},api,{c.get('accuracy', 0):.4f}\n")

    log("  Qwen comparison generated")


if __name__ == "__main__":
    main()
