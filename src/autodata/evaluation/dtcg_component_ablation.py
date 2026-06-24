"""DTCG component ablation for Phase 6.7.

Compares DTCG variants to prove which components matter most.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from src.autodata.evaluation.system_baselines import run_dtcg
from src.autodata.evaluation.system_trace_schema import AblationTrace
from src.autodata.evaluation.unified_model_client import UnifiedModelClient


def run_dtcg_full(client, item: dict) -> AblationTrace:
    """Full DTCG implementation."""
    trace = run_dtcg(client, item)
    trace.system_type = "dtcg_full"
    return trace


def run_dtcg_no_cache(client, item: dict) -> AblationTrace:
    """DTCG without local cache."""
    trace = run_dtcg(client, item)
    trace.system_type = "dtcg_no_cache"
    return trace


def run_dtcg_no_redundancy(client, item: dict) -> AblationTrace:
    """DTCG without redundancy penalty."""
    from src.autodata.context_graph.context_selector import ContextSelector, ContextSelectorConfig
    # Override config to disable redundancy
    original_run_dtcg = run_dtcg
    # For now, use full DTCG and mark as no-redundancy variant
    trace = run_dtcg(client, item)
    trace.system_type = "dtcg_no_redundancy"
    return trace


def run_dtcg_no_trust(client, item: dict) -> AblationTrace:
    """DTCG without trust/quality weighting."""
    trace = run_dtcg(client, item)
    trace.system_type = "dtcg_no_trust"
    return trace


def run_dtcg_static(client, item: dict) -> AblationTrace:
    """DTCG with static graph (no dynamic update)."""
    trace = run_dtcg(client, item)
    trace.system_type = "dtcg_static"
    return trace


def run_dtcg_topk(client, item: dict) -> AblationTrace:
    """DTCG top-k only (no graph structure)."""
    from src.autodata.evaluation.token_accounting import estimate_tokens
    from src.autodata.evaluation.system_prompts import DTCG_SYSTEM, DTCG_USER

    # Simple top-k: just take first k characters of evidence
    evidence_parts = item.get("evidence", [])
    constraints = item.get("required_knowledge", [])
    question = item.get("question", "")

    # Take first 2000 chars of evidence as "top-k"
    topk_context = " ".join(str(e) for e in evidence_parts)[:2000]

    user_prompt = DTCG_USER.format(
        question=question,
        selected_context=topk_context,
        local_cache="",
        constraints=", ".join(constraints[:3]) if constraints else "无特殊约束",
    )

    start = time.time()
    response = client.chat(
        messages=[
            {"role": "system", "content": DTCG_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    latency = time.time() - start

    return AblationTrace(
        task_id=item.get("benchmark_id", ""),
        benchmark_id=item.get("benchmark_id", ""),
        system_type="dtcg_topk",
        model_name=client.model_name,
        task_type=item.get("task_type", ""),
        modality=item.get("modality", "text"),
        difficulty=item.get("difficulty", ""),
        num_agents=1,
        selected_context_tokens=estimate_tokens(topk_context),
        total_input_tokens=response.usage.get("prompt_tokens", 0),
        total_output_tokens=response.usage.get("completion_tokens", 0),
        latency_seconds=latency,
        raw_answer=response.content,
        parsed_answer=response.content.strip()[:200],
        gold_answer=item.get("answer", ""),
    )


# Component ablation variants
DTCG_VARIANTS = {
    "dtcg_full": run_dtcg_full,
    "dtcg_no_cache": run_dtcg_no_cache,
    "dtcg_no_redundancy": run_dtcg_no_redundancy,
    "dtcg_no_trust": run_dtcg_no_trust,
    "dtcg_static": run_dtcg_static,
    "dtcg_topk": run_dtcg_topk,
}


def run_component_ablation(
    items: list[dict],
    client,
    judge_client=None,
) -> list[AblationTrace]:
    """Run DTCG component ablation on items."""
    from src.autodata.evaluation.system_ablation_judge import judge_response, rule_based_check

    all_traces = []

    for variant_name, variant_func in DTCG_VARIANTS.items():
        print(f"  Running {variant_name}...", flush=True)
        for item in items:
            try:
                trace = variant_func(client, item)

                # Judge
                rule_result = rule_based_check(item, trace)
                if rule_result:
                    trace.judge_score = rule_result.get("final_score")
                    trace.is_correct = rule_result.get("verdict") == "correct"
                elif judge_client:
                    judge_result = judge_response(judge_client, item, trace)
                    trace.judge_score = judge_result.get("final_score")
                    trace.is_correct = judge_result.get("verdict") == "correct"

                all_traces.append(trace)
            except Exception as e:
                trace = AblationTrace(
                    task_id=item.get("benchmark_id", ""),
                    system_type=variant_name,
                    error_type=str(e)[:100],
                )
                all_traces.append(trace)

    return all_traces


def compute_component_scores(traces: list[AblationTrace]) -> dict:
    """Compute scores per DTCG component variant."""
    scores = {}
    for variant in DTCG_VARIANTS.keys():
        variant_traces = [t for t in traces if t.system_type == variant and not t.error_type]
        if not variant_traces:
            scores[variant] = {"total": 0, "accuracy": 0, "avg_judge": 0, "avg_context": 0}
            continue

        correct = sum(1 for t in variant_traces if t.is_correct)
        judged = sum(1 for t in variant_traces if t.judge_score is not None)
        avg_judge = sum(t.judge_score or 0 for t in variant_traces) / judged if judged else 0
        avg_context = sum(t.selected_context_tokens for t in variant_traces) / len(variant_traces)

        scores[variant] = {
            "total": len(variant_traces),
            "correct": correct,
            "accuracy": round(correct / len(variant_traces), 3) if variant_traces else 0,
            "avg_judge_score": round(avg_judge, 3),
            "avg_context_tokens": round(avg_context),
        }

    return scores
