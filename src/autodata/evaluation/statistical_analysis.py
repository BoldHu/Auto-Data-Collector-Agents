"""Statistical significance analysis for Phase 6.7.

Bootstrap confidence intervals, McNemar test, Wilcoxon signed-rank.
"""

from __future__ import annotations

import json
import random
from typing import Optional


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval.

    Args:
        values: List of numeric values
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Dict with mean, ci_lower, ci_upper, std
    """
    if not values:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0}

    random.seed(seed)
    n = len(values)
    means = []

    for _ in range(n_resamples):
        sample = random.choices(values, k=n)
        means.append(sum(sample) / len(sample))

    means.sort()
    lower_idx = int((1 - confidence) / 2 * n_resamples)
    upper_idx = int((1 + confidence) / 2 * n_resamples)

    return {
        "mean": sum(values) / len(values),
        "ci_lower": means[lower_idx],
        "ci_upper": means[min(upper_idx, n_resamples - 1)],
        "std": (sum((m - sum(means)/len(means))**2 for m in means) / len(means)) ** 0.5,
        "n_samples": len(values),
        "n_resamples": n_resamples,
    }


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """McNemar test for paired binary correctness.

    Args:
        correct_a: List of booleans (correct/incorrect) for system A
        correct_b: List of booleans (correct/incorrect) for system B

    Returns:
        Dict with b (A correct, B wrong), c (A wrong, B correct), statistic, p_value
    """
    if len(correct_a) != len(correct_b):
        return {"error": "Lists must have same length"}

    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct
    for a, bb in zip(correct_a, correct_b):
        if a and not bb:
            b += 1
        elif not a and bb:
            c += 1

    # McNemar statistic with continuity correction
    if b + c == 0:
        return {"b": 0, "c": 0, "statistic": 0, "p_value": 1.0, "interpretation": "No discordant pairs"}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value using chi-squared distribution with 1 df
    # p-value = 1 - CDF of chi-squared(statistic, df=1)
    # Simple approximation:
    import math
    p_value = math.exp(-statistic / 2) if statistic > 0 else 1.0

    return {
        "b": b,
        "c": c,
        "statistic": round(statistic, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference",
    }


def wilcoxon_signed_rank(scores_a: list[float], scores_b: list[float]) -> dict:
    """Wilcoxon signed-rank test for paired judge scores.

    Args:
        scores_a: Judge scores for system A
        scores_b: Judge scores for system B

    Returns:
        Dict with statistic, p_value approximation
    """
    if len(scores_a) != len(scores_b):
        return {"error": "Lists must have same length"}

    # Compute differences
    diffs = [(a - b, i) for i, (a, b) in enumerate(zip(scores_a, scores_b)) if a != b]

    if not diffs:
        return {"n_pairs": len(scores_a), "n_nonzero": 0, "statistic": 0, "p_value": 1.0}

    # Rank absolute differences
    abs_diffs = sorted([(abs(d), i) for d, i in diffs])
    ranks = {}
    for rank, (abs_d, idx) in enumerate(abs_diffs, 1):
        ranks[idx] = rank

    # Sum of positive ranks (A > B)
    w_plus = sum(ranks[i] for d, i in diffs if d > 0)
    w_minus = sum(ranks[i] for d, i in diffs if d < 0)

    # Use the smaller statistic
    w = min(w_plus, w_minus)
    n = len(diffs)

    # Approximate p-value using normal approximation for n > 10
    if n > 10:
        mean_w = n * (n + 1) / 4
        std_w = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
        z = (w - mean_w) / std_w if std_w > 0 else 0
        import math
        p_value = 2 * (1 - abs(z) / (abs(z) + 1))  # rough approximation
    else:
        # For small n, use conservative estimate
        p_value = 0.5

    return {
        "n_pairs": len(scores_a),
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": w,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference",
    }


def run_significance_analysis(traces_by_system: dict[str, list[dict]]) -> dict:
    """Run full significance analysis between systems.

    Args:
        traces_by_system: Dict of system_name -> list of trace dicts

    Returns:
        Analysis results dict.
    """
    results = {"pairwise_comparisons": {}, "bootstrap_cis": {}}

    # Bootstrap CIs for accuracy
    for system, traces in traces_by_system.items():
        correct_values = [1.0 if t.get("is_correct") else 0.0 for t in traces]
        judge_values = [t.get("judge_score", 0) for t in traces if t.get("judge_score") is not None]

        results["bootstrap_cis"][system] = {
            "accuracy": bootstrap_ci(correct_values),
            "judge_score": bootstrap_ci(judge_values) if judge_values else None,
        }

    # Paired comparisons: DTCG vs others
    dtcg_traces = traces_by_system.get("dtcg", [])
    if not dtcg_traces:
        return results

    dtcg_correct = [t.get("is_correct", False) for t in dtcg_traces]
    dtcg_judges = [t.get("judge_score", 0) for t in dtcg_traces if t.get("judge_score") is not None]

    for system, traces in traces_by_system.items():
        if system == "dtcg":
            continue

        # Pad to same length if needed
        min_len = min(len(dtcg_traces), len(traces))
        if min_len < 10:
            continue

        other_correct = [t.get("is_correct", False) for t in traces[:min_len]]
        other_judges = [t.get("judge_score", 0) for t in traces[:min_len] if t.get("judge_score") is not None]

        comparison = {
            "mcnemar": mcnemar_test(dtcg_correct[:min_len], other_correct),
            "context_saving": {
                "dtcg_avg_context": sum(t.get("selected_context_tokens", 0) for t in dtcg_traces[:min_len]) / min_len,
                "other_avg_context": sum(t.get("selected_context_tokens", 0) for t in traces[:min_len]) / min_len,
            },
        }

        # Wilcoxon for judge scores
        min_judge_len = min(len(dtcg_judges), len(other_judges))
        if min_judge_len >= 5:
            comparison["wilcoxon"] = wilcoxon_signed_rank(dtcg_judges[:min_judge_len], other_judges[:min_judge_len])

        results["pairwise_comparisons"][f"dtcg_vs_{system}"] = comparison

    return results
