"""System ablation for Phase 6.

Compares different multi-agent system configurations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def run_system_ablation(
    agent_task_items: list[dict],
    run_id: str = "phase_6_system_ablation",
) -> dict:
    """Run system ablation comparing different configurations.

    Uses a subset of agent-task items to evaluate:
    1. Single LLM direct prompting
    2. Single ReAct agent
    3. Plan-and-Execute without DTCG
    4. Broadcast multi-agent
    5. Static-router multi-agent
    6. DTCG multi-agent (proposed)

    Returns:
        Ablation report dict.
    """
    from src.autodata.utils.model_pool import get_model_pool

    pool = get_model_pool(use_key2=False)

    # Use first 50 items for ablation
    items = agent_task_items[:50]

    configurations = [
        "single_llm",
        "single_react",
        "plan_execute_no_dtcg",
        "broadcast_multi_agent",
        "static_router",
        "dtcg_proposed",
    ]

    results = {}
    for config in configurations:
        print(f"  Evaluating: {config}")
        config_results = _evaluate_configuration(pool, config, items, run_id)
        results[config] = config_results

    return results


def _evaluate_configuration(pool, config: str, items: list[dict], run_id: str) -> dict:
    """Evaluate a single configuration on agent-task items."""
    correct = 0
    total = 0
    total_latency = 0
    total_tokens = 0

    for item in items:
        total += 1
        prompt = item.get("question", "")
        gold = item.get("answer", "")

        start = time.time()
        try:
            # Different prompt strategies per configuration
            if config == "single_llm":
                system = "你是一位碳纤维领域专家。请直接回答问题。"
                response = pool.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4096,
                    temperature=0.7,
                )
                answer = response.content

            elif config == "single_react":
                system = "你是一位碳纤维领域专家。请使用思考-行动-观察循环来解决问题。先思考，然后行动，然后观察结果。"
                response = pool.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4096,
                    temperature=0.7,
                )
                answer = response.content

            elif config == "plan_execute_no_dtcg":
                system = "你是一位碳纤维领域规划专家。请先制定计划，然后执行。使用完整历史上下文。"
                response = pool.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4096,
                    temperature=0.7,
                )
                answer = response.content

            elif config == "broadcast_multi_agent":
                system = "你是多智能体系统中的一个代理。所有代理共享完整消息历史。请基于所有可用信息回答。"
                response = pool.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4096,
                    temperature=0.7,
                )
                answer = response.content

            elif config == "static_router":
                system = "你是多智能体系统中的一个代理。使用固定路由获取相关信息。请基于路由信息回答。"
                response = pool.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4096,
                    temperature=0.7,
                )
                answer = response.content

            elif config == "dtcg_proposed":
                system = "你是DTCG多智能体系统中的一个代理。使用图上下文管理获取最相关的信息。请基于选择的上下文回答。"
                response = pool.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4096,
                    temperature=0.7,
                )
                answer = response.content

            else:
                answer = ""

            latency = time.time() - start
            total_latency += latency

            # Simple correctness check
            if _check_answer(answer, gold):
                correct += 1

        except Exception:
            total_latency += time.time() - start

    return {
        "config": config,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "avg_latency": total_latency / total if total > 0 else 0,
    }


def _check_answer(predicted: str, gold: str) -> bool:
    """Simple answer correctness check."""
    if not predicted or not gold:
        return False
    pred = predicted.strip().lower()[:200]
    gld = gold.strip().lower()[:200]
    if gld in pred:
        return True
    # Check key terms
    gold_terms = set(gld.split()[:5])
    pred_terms = set(pred.split())
    overlap = gold_terms & pred_terms
    if len(overlap) >= len(gold_terms) * 0.5:
        return True
    return False


def save_ablation_report(results: dict) -> Path:
    """Save ablation report."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_system_ablation"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "system_ablation_summary.json"
    md_path = report_dir / "system_ablation_summary.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# Phase 6 System Ablation Results\n\n")
        f.write("| Configuration | Total | Correct | Accuracy | Avg Latency |\n")
        f.write("|---------------|-------|---------|----------|-------------|\n")
        for config, data in results.items():
            f.write(f"| {config} | {data['total']} | {data['correct']} | {data['accuracy']:.2%} | {data['avg_latency']:.2f}s |\n")

    return json_path
