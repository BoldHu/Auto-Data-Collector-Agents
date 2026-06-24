"""Run Phase 6.5 Doubao baseline evaluation on CFBench.

Usage:
    python scripts/run_phase_6_5_doubao_evaluation.py \
        --run_id phase_6_5_doubao \
        --benchmark_root data/benchmark/subsets \
        --max_workers 64 \
        --judge_workers 32 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def create_doubao_caller(model_info: dict):
    """Create a callable for a Doubao model."""
    from src.autodata.utils.doubao_model_client import DoubaoModelClient

    client = DoubaoModelClient(
        default_model=model_info.get("model_id", "doubao-seed-2.0-lite"),
        timeout=60.0,  # 60 second timeout per call
    )

    def caller(prompt, system_prompt):
        response = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.content

    return caller


def create_xiaomi_caller(model_info: dict):
    """Create a callable for Xiaomi model."""
    from src.autodata.utils.model_pool import get_model_pool

    pool = get_model_pool(use_key2=False)

    def caller(prompt, system_prompt):
        response = pool.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model_info.get("model_id", "mimo-v2.5-pro"),
            max_completion_tokens=4096,
            temperature=0.7,
        )
        return response.content

    return caller


def evaluate_single_item_v2(
    model_name: str,
    model_caller,
    item: dict,
    subset: str,
    run_id: str,
    judge_pool=None,
):
    """Evaluate a single model on a single item with LLM judge for open-ended."""
    from src.autodata.evaluation.evaluation_schema import EvaluationResult
    from src.autodata.evaluation.prompt_builder import build_eval_prompt
    from src.autodata.evaluation.metric_calculator import compute_metrics
    from src.autodata.evaluation.answer_normalizer import normalize_answer
    from src.autodata.evaluation.open_answer_evaluator import (
        needs_llm_judge, judge_open_answer, is_judge_correct,
    )

    benchmark_id = item.get("benchmark_id", "")
    task_type = item.get("task_type", "")
    modality = item.get("modality", "text")
    gold_answer = item.get("answer", "")

    # Skip multimodal items for text-only models
    if modality == "multimodal" and item.get("image_refs"):
        # Check if model supports images
        # For now, skip if model is text-only
        return EvaluationResult(
            run_id=run_id,
            model_name=model_name,
            benchmark_id=benchmark_id,
            subset=subset,
            task_type=task_type,
            modality=modality,
            prompt="",
            raw_response="",
            parsed_answer="",
            gold_answer=gold_answer,
            error="skipped_multimodal",
        )

    # Build prompt
    prompt = build_eval_prompt(item)

    # Call model with retry for empty responses
    start_time = time.time()
    raw_response = ""
    latency = 0
    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw_response = model_caller(prompt, "你是一位碳纤维领域专家。请准确回答问题。")
            latency = time.time() - start_time
            if raw_response and raw_response.strip():
                break  # Got a valid response
            elif attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
        except Exception as e:
            if attempt == max_retries - 1:
                return EvaluationResult(
                    run_id=run_id,
                    model_name=model_name,
                    benchmark_id=benchmark_id,
                    subset=subset,
                    task_type=task_type,
                    modality=modality,
                    prompt=prompt,
                    raw_response="",
                    parsed_answer="",
                    gold_answer=gold_answer,
                    error=str(e)[:200],
                    latency_seconds=time.time() - start_time,
                )
            time.sleep(2)  # Wait before retry

    # Parse answer
    parsed_answer = normalize_answer(raw_response)

    # Compute automatic metrics
    metrics = compute_metrics(item, parsed_answer, raw_response)

    # LLM judge for open-ended questions
    judge_score = None
    judge_verdict = None
    if judge_pool and needs_llm_judge(item, metrics):
        try:
            judge_result = judge_open_answer(judge_pool, item, raw_response)
            judge_score = judge_result.final_score
            judge_verdict = judge_result.verdict

            # Override is_correct with judge verdict for open-ended
            if is_judge_correct(judge_result):
                metrics["is_correct"] = True
            elif judge_result.verdict == "partially_correct":
                metrics["is_correct"] = False  # Count as wrong but note partial
        except Exception:
            pass

    return EvaluationResult(
        run_id=run_id,
        model_name=model_name,
        benchmark_id=benchmark_id,
        subset=subset,
        task_type=task_type,
        modality=modality,
        prompt=prompt,
        raw_response=raw_response[:1000],
        parsed_answer=parsed_answer,
        gold_answer=gold_answer,
        is_correct=metrics.get("is_correct"),
        exact_match=metrics.get("exact_match"),
        f1=metrics.get("f1"),
        numeric_score=metrics.get("numeric_score"),
        llm_judge_score=judge_score,
        latency_seconds=latency,
    )


def run_evaluation_v2(
    models: list[dict],
    benchmark_items: list[dict],
    subset: str,
    run_id: str,
    output_dir: Path,
    max_workers: int = 32,
    judge_pool=None,
    checkpoint_path: Path = None,
) -> list:
    """Run evaluation with LLM judge support."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.autodata.evaluation.evaluation_schema import EvaluationResult

    results = []
    completed_pairs = set()

    # Load checkpoint
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
            completed_pairs = set(data.get("completed_pairs", []))

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_info in models:
        model_name = model_info["model_name"]
        if not model_info.get("enabled", True):
            continue

        # Create caller
        provider = model_info.get("provider", "")
        if provider == "doubao":
            model_caller = create_doubao_caller(model_info)
        else:
            model_caller = create_xiaomi_caller(model_info)

        # Filter items to evaluate
        items_to_eval = []
        for item in benchmark_items:
            pair_key = f"{model_name}:{item.get('benchmark_id', '')}"
            if pair_key not in completed_pairs:
                items_to_eval.append(item)

        if not items_to_eval:
            continue

        print(f"  Evaluating {model_name}: {len(items_to_eval)} items", flush=True)

        # Evaluate with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for item in items_to_eval:
                future = executor.submit(
                    evaluate_single_item_v2,
                    model_name,
                    model_caller,
                    item,
                    subset,
                    run_id,
                    judge_pool,
                )
                futures[future] = item

            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per item
                    results.append(result)

                    pair_key = f"{model_name}:{result.benchmark_id}"
                    completed_pairs.add(pair_key)

                    if len(completed_pairs) % 50 == 0 and checkpoint_path:
                        _save_checkpoint(checkpoint_path, completed_pairs)

                except Exception as e:
                    benchmark_id = item.get("benchmark_id", "")
                    results.append(EvaluationResult(
                        run_id=run_id,
                        model_name=model_name,
                        benchmark_id=benchmark_id,
                        subset=subset,
                        task_type=item.get("task_type", ""),
                        modality=item.get("modality", "text"),
                        prompt="",
                        raw_response="",
                        parsed_answer="",
                        gold_answer=item.get("answer", ""),
                        error=str(e)[:200],
                    ))

        # Save checkpoint after each model
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, completed_pairs)

    return results


def _save_checkpoint(path: Path, completed_pairs: set):
    """Save checkpoint atomically."""
    import os
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"completed_pairs": list(completed_pairs), "count": len(completed_pairs)}, f)
    os.replace(tmp_path, path)


def compute_summary_v2(results: list) -> dict:
    """Compute summary with strict and judge accuracy."""
    total = len(results)
    skipped = sum(1 for r in results if r.error == "skipped_multimodal")
    errors = sum(1 for r in results if r.error and r.error != "skipped_multimodal")
    evaluated = total - skipped - errors

    correct = sum(1 for r in results if r.is_correct is True)
    strict_accuracy = correct / evaluated if evaluated > 0 else 0

    # Judge accuracy (items with judge scores)
    judged = [r for r in results if r.llm_judge_score is not None]
    judge_correct = sum(1 for r in judged if r.llm_judge_score >= 0.75)
    judge_accuracy = judge_correct / len(judged) if judged else 0

    # Per-task-type
    task_results = {}
    for r in results:
        if r.error == "skipped_multimodal":
            continue
        tt = r.task_type
        if tt not in task_results:
            task_results[tt] = {"total": 0, "correct": 0, "judged": 0, "judge_correct": 0}
        task_results[tt]["total"] += 1
        if r.is_correct is True:
            task_results[tt]["correct"] += 1
        if r.llm_judge_score is not None:
            task_results[tt]["judged"] += 1
            if r.llm_judge_score >= 0.75:
                task_results[tt]["judge_correct"] += 1

    return {
        "total": total,
        "skipped": skipped,
        "errors": errors,
        "evaluated": evaluated,
        "correct": correct,
        "strict_accuracy": strict_accuracy,
        "judge_accuracy": judge_accuracy,
        "judged_count": len(judged),
        "task_accuracy": task_results,
    }


def save_results(results: list, output_path: Path):
    """Save results to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 6.5 Doubao evaluation")
    parser.add_argument("--run_id", type=str, default="phase_6_5_doubao")
    parser.add_argument("--benchmark_root", type=str, default="data/benchmark/subsets")
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--judge_workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    from src.autodata.evaluation.model_registry import load_model_registry_6_5, save_model_registry_6_5
    from src.autodata.utils.llm_api_loader import get_llm_config, get_sanitized_status
    from src.autodata.utils.model_pool import get_model_pool

    benchmark_root = Path(args.benchmark_root)
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_5_doubao_evaluation"
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_5"

    # Parse API config
    config = get_llm_config()
    status = get_sanitized_status(config)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "llm_api_config_status.json", "w") as f:
        json.dump(status, f, indent=2)
    print(f"API config: Xiaomi={status['xiaomi_config_found']}, Doubao={status['doubao_config_found']}")

    # Load models
    models = load_model_registry_6_5()
    save_model_registry_6_5(models)
    enabled = [m for m in models if m.get("enabled", True)]
    print(f"Loaded {len(models)} models, {len(enabled)} enabled")

    for m in models:
        status = "✓" if m.get("enabled") else "✗"
        print(f"  {status} {m.get('display_name', m['model_name'])} ({m['provider']})")

    # Judge pool - use Doubao Seed 2.0 Pro for judging (Xiaomi API key invalid)
    # Create a Doubao client for judging
    from src.autodata.utils.doubao_model_client import DoubaoModelClient
    judge_client = DoubaoModelClient(default_model="doubao-seed-2.0-pro")

    # Create a judge pool wrapper
    class JudgePoolWrapper:
        def chat_quality(self, messages, max_completion_tokens=2048, temperature=0.3):
            return judge_client.chat(
                messages=messages,
                max_tokens=max_completion_tokens,
                temperature=temperature,
            )

    judge_pool = JudgePoolWrapper()

    # Define subsets
    subsets = {
        "cfbench_text": benchmark_root / "cfbench_text_test.jsonl",
        "cfbench_exam": benchmark_root / "cfbench_exam_test.jsonl",
        "cfbench_core": benchmark_root / "cfbench_core_test.jsonl",
        "cfbench_hard": benchmark_root / "cfbench_hard_test.jsonl",
        "cfbench_agenttask": benchmark_root / "cfbench_agenttask_test.jsonl",
    }

    # Smoke test
    if args.smoke_test:
        print("\n=== Smoke Test ===")
        smoke_models = [m for m in enabled if m["provider"] == "xiaomi"][:1]
        smoke_models += [m for m in enabled if m["provider"] == "doubao"][:2]
        smoke_results = []

        for subset_name, subset_path in subsets.items():
            items = load_jsonl(subset_path)[:10]
            if not items:
                continue
            print(f"  {subset_name}: {len(items)} items x {len(smoke_models)} models")

            checkpoint_path = report_dir / f"smoke_checkpoint_{subset_name}.json"
            results = run_evaluation_v2(
                smoke_models, items, subset_name, args.run_id,
                eval_dir / "raw_outputs", max_workers=4,
                judge_pool=judge_pool, checkpoint_path=checkpoint_path,
            )
            smoke_results.extend(results)

        save_results(smoke_results, report_dir / "smoke_test_results.jsonl")
        summary = compute_summary_v2(smoke_results)
        with open(report_dir / "smoke_test_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Smoke test: strict={summary['strict_accuracy']:.2%}, judge={summary['judge_accuracy']:.2%}")
        return

    # Full evaluation
    print("\n=== Full Evaluation ===")
    all_results = []
    start_time = time.time()

    for subset_name, subset_path in subsets.items():
        items = load_jsonl(subset_path)
        if not items:
            continue

        print(f"\n--- {subset_name}: {len(items)} items x {len(enabled)} models ---")

        checkpoint_path = report_dir / f"checkpoint_{subset_name}.json"
        results = run_evaluation_v2(
            enabled, items, subset_name, args.run_id,
            eval_dir / "raw_outputs", max_workers=args.max_workers,
            judge_pool=judge_pool, checkpoint_path=checkpoint_path,
        )
        all_results.extend(results)

        # Save per-subset
        save_results(results, eval_dir / "parsed_predictions" / f"{subset_name}_predictions.jsonl")

        # Per-subset summary per model
        for model_info in enabled:
            model_name = model_info["model_name"]
            model_results = [r for r in results if r.model_name == model_name]
            if model_results:
                summary = compute_summary_v2(model_results)
                print(f"  {model_name}: strict={summary['strict_accuracy']:.2%}, judge={summary['judge_accuracy']:.2%} ({summary['correct']}/{summary['evaluated']})")

                with open(report_dir / f"{subset_name}_{model_name}_results.json", "w") as f:
                    json.dump(summary, f, indent=2)

    # Overall summary
    elapsed = time.time() - start_time
    overall = compute_summary_v2(all_results)

    with open(report_dir / "test_results_summary.json", "w") as f:
        json.dump({
            "run_id": args.run_id,
            "elapsed_seconds": elapsed,
            "total_results": len(all_results),
            "overall": overall,
            "models_evaluated": [m["model_name"] for m in enabled],
            "api_config": status,
        }, f, indent=2)

    print(f"\n=== Evaluation Complete ===")
    print(f"Total results: {len(all_results)}")
    print(f"Strict accuracy: {overall['strict_accuracy']:.2%}")
    print(f"Judge accuracy: {overall['judge_accuracy']:.2%}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
