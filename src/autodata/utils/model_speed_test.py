"""Model speed benchmark — tests all Xiaomi models on both APIs.

Sends short prompts to each model+API combination, measures latency,
and recommends the fastest model for pipeline use.

Usage:
    python -m src.autodata.utils.model_speed_test
    # Or:
    python scripts/model_speed_test.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.autodata.utils.model_pool import ModelPool, ModelEndpoint
from src.autodata.utils.model_client import XiaomiModelClient
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("model_speed_test")


# Models to test (all confirmed working on both APIs)
TEST_MODELS = [
    "mimo-v2-omni",
    "mimo-v2.5",
    "mimo-v2-pro",
    "mimo-v2.5-pro",
]

# Test prompts — short, realistic cleaning-style prompts
TEST_PROMPTS = [
    {
        "system": "You are a professional technical document cleaning specialist. Always respond with valid JSON.",
        "user": "请将以下OCR文本清洗为格式正确的中文技术文档。原文：碳纤维复合材料是由碳纤维和基体树脂复合而成的材料。其特点是比强度高、比模量高，在航空航天领域有广泛应用。请输出JSON格式：{\"cleaned_text\": \"...\", \"confidence\": 0.8}",
    },
    {
        "system": "You are a carbon fiber domain knowledge extraction expert. Always respond with valid JSON array.",
        "user": "从以下文本中提取碳纤维相关知识单元：PAN基碳纤维的拉伸强度可达3.5-7.0 GPa，拉伸模量为200-600 GPa。",
    },
]


@dataclass
class SpeedResult:
    """Result of a single model speed test."""
    model: str
    api_index: int  # 0 = API_KEY, 1 = API_KEY2
    prompt_idx: int
    latency_ms: float
    tokens_used: int
    success: bool
    error: str = ""


def run_speed_test(
    models: Optional[list[str]] = None,
    num_prompts: int = 2,
    num_rounds: int = 3,
    output_path: Optional[str] = None,
) -> dict:
    """Run speed benchmark across all models on both APIs.

    Args:
        models: List of model names to test (defaults to TEST_MODELS).
        num_prompts: Number of test prompts to use (1-2).
        num_rounds: Number of rounds per model+API (default 3).
        output_path: Optional path to save results JSON.

    Returns:
        Dict with results, recommendations, and stats.
    """
    models = models or TEST_MODELS
    prompts = TEST_PROMPTS[:num_prompts]

    results: list[SpeedResult] = []

    # Create clients for both APIs
    client_api0 = XiaomiModelClient(use_key2=False)
    client_api1 = XiaomiModelClient(use_key2=True)

    clients = {0: client_api0, 1: client_api1}

    logger.info(f"Starting speed test: {len(models)} models × 2 APIs × {num_prompts} prompts × {num_rounds} rounds")

    for api_idx, client in clients.items():
        for model in models:
            for prompt_idx, prompt in enumerate(prompts):
                for round_idx in range(num_rounds):
                    messages = [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ]

                    start = time.time()
                    try:
                        response = client.chat(
                            messages=messages,
                            model=model,
                            max_completion_tokens=1024,
                        )
                        latency_ms = (time.time() - start) * 1000
                        results.append(SpeedResult(
                            model=model,
                            api_index=api_idx,
                            prompt_idx=prompt_idx,
                            latency_ms=latency_ms,
                            tokens_used=response.total_tokens,
                            success=True,
                        ))
                        logger.info(
                            f"  {model}/API{api_idx}/p{prompt_idx}/r{round_idx}: "
                            f"{latency_ms:.0f}ms, {response.total_tokens} tokens"
                        )
                    except Exception as e:
                        latency_ms = (time.time() - start) * 1000
                        results.append(SpeedResult(
                            model=model,
                            api_index=api_idx,
                            prompt_idx=prompt_idx,
                            latency_ms=latency_ms,
                            tokens_used=0,
                            success=False,
                            error=str(e)[:100],
                        ))
                        logger.warning(
                            f"  {model}/API{api_idx}/p{prompt_idx}/r{round_idx}: "
                            f"FAILED - {str(e)[:80]}"
                        )

    # Aggregate results
    summary = {}
    for model in models:
        for api_idx in [0, 1]:
            key = f"{model}/API{api_idx}"
            model_results = [
                r for r in results
                if r.model == model and r.api_index == api_idx
            ]
            successes = [r for r in model_results if r.success]
            failures = [r for r in model_results if not r.success]

            if successes:
                avg_latency = sum(r.latency_ms for r in successes) / len(successes)
                min_latency = min(r.latency_ms for r in successes)
                max_latency = max(r.latency_ms for r in successes)
                avg_tokens = sum(r.tokens_used for r in successes) / len(successes)
            else:
                avg_latency = float("inf")
                min_latency = float("inf")
                max_latency = float("inf")
                avg_tokens = 0

            summary[key] = {
                "model": model,
                "api_index": api_idx,
                "success_count": len(successes),
                "failure_count": len(failures),
                "avg_latency_ms": round(avg_latency, 1),
                "min_latency_ms": round(min_latency, 1),
                "max_latency_ms": round(max_latency, 1),
                "avg_tokens": round(avg_tokens, 1),
            }

    # Sort by average latency (fastest first)
    sorted_summary = sorted(
        summary.items(),
        key=lambda x: x[1]["avg_latency_ms"],
    )

    # Recommendations
    fastest = sorted_summary[0] if sorted_summary else None
    recommendations = {
        "fastest_model": fastest[1] if fastest else None,
        "quality_fallback": summary.get("mimo-v2.5-pro/API0") or summary.get("mimo-v2.5-pro/API1"),
        "recommended_pool_order": [
            item[1]["model"] + f"/API{item[1]['api_index']}"
            for item in sorted_summary
            if item[1]["success_count"] > 0
        ],
    }

    # Full report
    report = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_models": len(models),
        "num_apis": 2,
        "num_prompts": num_prompts,
        "num_rounds": num_rounds,
        "total_calls": len(results),
        "success_rate": len([r for r in results if r.success]) / max(len(results), 1),
        "results": [r.__dict__ for r in results],
        "summary": summary,
        "sorted_by_speed": [(k, v) for k, v in sorted_summary],
        "recommendations": recommendations,
    }

    # Save to file
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Speed test results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL SPEED TEST RESULTS")
    print("=" * 60)
    print(f"Total calls: {len(results)}, Success rate: {report['success_rate']:.1%}")
    print()
    print("Ranking (fastest to slowest):")
    print("-" * 40)
    for rank, (key, info) in enumerate(sorted_summary, 1):
        status = "OK" if info["success_count"] > 0 else "FAIL"
        print(f"  {rank}. {key}: avg {info['avg_latency_ms']:.0f}ms [{status}]")
    print()
    if recommendations["fastest_model"]:
        print(f"Fastest model: {recommendations['fastest_model']['model']}/API{recommendations['fastest_model']['api_index']}")
        print(f"  Average latency: {recommendations['fastest_model']['avg_latency_ms']:.0f}ms")
    print(f"Quality fallback: mimo-v2.5-pro")
    print(f"Recommended pool order: {recommendations['recommended_pool_order']}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    import sys

    output = "data/reports/phase_2_7_restart_cleaning/model_speed_test.json"
    if len(sys.argv) > 1:
        output = sys.argv[1]

    report = run_speed_test(output_path=output)

    # Exit with error if no model succeeded
    if report["success_rate"] == 0:
        print("ERROR: All model speed test calls failed")
        sys.exit(1)