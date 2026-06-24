"""Evaluation runner for Phase 6.

Main evaluation loop: load benchmark, iterate models, compute metrics.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.autodata.evaluation.evaluation_schema import EvaluationResult
from src.autodata.evaluation.prompt_builder import build_eval_prompt
from src.autodata.evaluation.metric_calculator import compute_metrics
from src.autodata.evaluation.answer_normalizer import extract_mc_answer, normalize_answer
from src.autodata.evaluation.llm_judge import judge_response

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


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


def load_checkpoint(checkpoint_path: Path) -> set:
    """Load completed model-item pairs from checkpoint."""
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path) as f:
        data = json.load(f)
        return set(data.get("completed_pairs", []))


def save_checkpoint(checkpoint_path: Path, completed_pairs: set) -> None:
    """Save checkpoint atomically."""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"completed_pairs": list(completed_pairs), "count": len(completed_pairs)}, f)
    os.replace(tmp_path, checkpoint_path)


def evaluate_single_item(
    model_name: str,
    model_caller,
    item: dict,
    subset: str,
    run_id: str,
    judge_pool=None,
) -> EvaluationResult:
    """Evaluate a single model on a single item.

    Args:
        model_name: Name of the model
        model_caller: Callable that takes (prompt, system_prompt) and returns response text
        item: Benchmark item dict
        subset: Subset name
        run_id: Run ID
        judge_pool: ModelPool for LLM judge (optional)

    Returns:
        EvaluationResult
    """
    benchmark_id = item.get("benchmark_id", "")
    task_type = item.get("task_type", "")
    modality = item.get("modality", "text")
    gold_answer = item.get("answer", "")

    # Skip multimodal items for text-only models
    if modality == "multimodal" and item.get("image_refs"):
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

    # Call model
    start_time = time.time()
    try:
        raw_response = model_caller(prompt, "你是一位碳纤维领域专家。请准确回答问题。")
        latency = time.time() - start_time
    except Exception as e:
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

    # Parse answer
    parsed_answer = normalize_answer(raw_response)

    # Compute metrics
    metrics = compute_metrics(item, parsed_answer, raw_response)

    # LLM judge for open-ended questions
    judge_score = None
    if judge_pool and _needs_judge(item, metrics):
        try:
            judge_result = judge_response(judge_pool, item, raw_response)
            judge_score = judge_result.final_score
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


def _needs_judge(item: dict, metrics: dict) -> bool:
    """Check if an item needs LLM judge evaluation."""
    # Use judge for open-ended questions where automatic metrics are insufficient
    task_type = item.get("task_type", "")
    if task_type in ("agent_task",):
        return True
    if task_type in ("explanation", "comparison", "domain_knowledge_qa"):
        return True
    if task_type in ("source_grounded_reasoning",):
        return True
    # Use judge when exact_match is False but answer might be partially correct
    if metrics.get("exact_match") is False and not item.get("options"):
        return True
    return False


def run_evaluation(
    models: list[dict],
    benchmark_items: list[dict],
    subset: str,
    run_id: str,
    output_dir: Path,
    max_workers: int = 32,
    judge_pool=None,
    checkpoint_path: Optional[Path] = None,
) -> list[EvaluationResult]:
    """Run evaluation of all models on a benchmark subset.

    Args:
        models: List of model info dicts
        benchmark_items: List of benchmark items
        subset: Subset name
        run_id: Run ID
        output_dir: Output directory
        max_workers: Max concurrent workers
        judge_pool: ModelPool for LLM judge
        checkpoint_path: Path for checkpoint file

    Returns:
        List of EvaluationResults
    """
    results = []
    completed_pairs = load_checkpoint(checkpoint_path) if checkpoint_path else set()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_info in models:
        model_name = model_info["model_name"]
        model_caller = _create_model_caller(model_info)

        # Filter items to evaluate
        items_to_eval = []
        for item in benchmark_items:
            pair_key = f"{model_name}:{item.get('benchmark_id', '')}"
            if pair_key not in completed_pairs:
                items_to_eval.append(item)

        if not items_to_eval:
            continue

        # Evaluate with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for item in items_to_eval:
                future = executor.submit(
                    evaluate_single_item,
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
                    result = future.result()
                    results.append(result)

                    # Update checkpoint
                    pair_key = f"{model_name}:{result.benchmark_id}"
                    completed_pairs.add(pair_key)

                    if checkpoint_path and len(completed_pairs) % 50 == 0:
                        save_checkpoint(checkpoint_path, completed_pairs)

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
            save_checkpoint(checkpoint_path, completed_pairs)

    return results


def _create_model_caller(model_info: dict):
    """Create a callable for a model."""
    model_name = model_info["model_name"]
    provider = model_info.get("provider", "")

    if provider == "xiaomi":
        # Use Xiaomi ModelPool
        from src.autodata.utils.model_pool import get_model_pool
        pool = get_model_pool(use_key2=False)

        def xiaomi_caller(prompt, system_prompt):
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

        return xiaomi_caller

    else:
        # Use BaselineModelRunner
        from src.autodata.utils.baseline_model_loader import load_baseline_configs, create_runners
        configs = load_baseline_configs()
        runners = create_runners(configs)
        # Find runner by model name
        runner = None
        for r in runners:
            if r.model_config.name == model_name:
                runner = r
                break
        if not runner:
            raise ValueError(f"Model runner not found: {model_name}")

        def baseline_caller(prompt, system_prompt):
            response = runner.invoke(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=4096,
                temperature=0.7,
                thinking=model_info.get("supports_thinking", False),
            )
            return response.content

        return baseline_caller


def save_results(results: list[EvaluationResult], output_path: Path) -> None:
    """Save evaluation results to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def compute_summary(results: list[EvaluationResult]) -> dict:
    """Compute summary statistics from evaluation results."""
    total = len(results)
    skipped = sum(1 for r in results if r.error == "skipped_multimodal")
    errors = sum(1 for r in results if r.error and r.error != "skipped_multimodal")
    evaluated = total - skipped - errors

    correct = sum(1 for r in results if r.is_correct is True)
    accuracy = correct / evaluated if evaluated > 0 else 0

    # Per-task-type
    task_results = {}
    for r in results:
        if r.error == "skipped_multimodal":
            continue
        tt = r.task_type
        if tt not in task_results:
            task_results[tt] = {"total": 0, "correct": 0}
        task_results[tt]["total"] += 1
        if r.is_correct is True:
            task_results[tt]["correct"] += 1

    task_accuracy = {
        tt: {"accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0, **d}
        for tt, d in task_results.items()
    }

    # Per-difficulty
    diff_results = {}
    for r in results:
        if r.error == "skipped_multimodal":
            continue
        diff = r.modality  # Use modality as proxy
        if diff not in diff_results:
            diff_results[diff] = {"total": 0, "correct": 0}
        diff_results[diff]["total"] += 1
        if r.is_correct is True:
            diff_results[diff]["correct"] += 1

    return {
        "total": total,
        "skipped": skipped,
        "errors": errors,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "task_accuracy": task_accuracy,
    }
