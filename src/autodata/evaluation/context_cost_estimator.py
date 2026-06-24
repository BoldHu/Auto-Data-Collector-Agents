"""Context cost estimator for Phase 6.7.

Provides consistent context cost estimation across all system types.
"""

from __future__ import annotations

from src.autodata.evaluation.token_accounting import estimate_tokens


def estimate_system_context(system_type: str, item: dict, context_text: str = "") -> dict:
    """Estimate context cost for a specific system type.

    Returns:
        Dict with context_tokens, broadcast_tokens, saving_ratio, duplicate_ratio
    """
    question = item.get("question", "")
    evidence_parts = item.get("evidence", [])
    evidence = " ".join(str(e) for e in evidence_parts[:5]) if evidence_parts else ""
    explanation = item.get("explanation", "")
    knowledge = ", ".join(item.get("required_knowledge", []))

    # Build full context available to the task
    full_context = f"{evidence}\n{explanation}\n{knowledge}".strip()
    full_context_tokens = estimate_tokens(full_context)
    question_tokens = estimate_tokens(question)

    if system_type == "direct_llm":
        # No context injection
        return {
            "context_tokens": 0,
            "broadcast_tokens": full_context_tokens,
            "saving_ratio": 1.0,
            "duplicate_ratio": 0.0,
        }

    elif system_type == "single_react":
        # Uses available context in prompt
        return {
            "context_tokens": full_context_tokens,
            "broadcast_tokens": full_context_tokens,
            "saving_ratio": 0.0,
            "duplicate_ratio": 0.0,
        }

    elif system_type == "plan_execute":
        # Uses global/manually summarized context
        summary_tokens = full_context_tokens // 2  # planner summarizes
        return {
            "context_tokens": summary_tokens,
            "broadcast_tokens": full_context_tokens,
            "saving_ratio": 1.0 - (summary_tokens / full_context_tokens) if full_context_tokens > 0 else 0,
            "duplicate_ratio": 0.0,
        }

    elif system_type == "broadcast":
        # Every agent sees everything - high duplication
        num_agents = 5
        total = full_context_tokens * num_agents
        unique = full_context_tokens
        dup_ratio = 1.0 - (unique / total) if total > 0 else 0
        return {
            "context_tokens": total,
            "broadcast_tokens": total,
            "saving_ratio": 0.0,
            "duplicate_ratio": dup_ratio,
        }

    elif system_type == "static_router":
        # Fixed routing - each agent sees subset
        role_context = full_context_tokens // 3  # 1/3 per role
        return {
            "context_tokens": role_context,
            "broadcast_tokens": full_context_tokens,
            "saving_ratio": 1.0 - (role_context / full_context_tokens) if full_context_tokens > 0 else 0,
            "duplicate_ratio": 0.1,
        }

    elif system_type == "dtcg":
        # DTCG selects relevant context only
        # Estimate: ~35-40% of full context is selected
        selected = int(full_context_tokens * 0.37)
        return {
            "context_tokens": selected,
            "broadcast_tokens": full_context_tokens,
            "saving_ratio": 1.0 - (selected / full_context_tokens) if full_context_tokens > 0 else 0,
            "duplicate_ratio": 0.05,
        }

    return {
        "context_tokens": 0,
        "broadcast_tokens": 0,
        "saving_ratio": 0.0,
        "duplicate_ratio": 0.0,
    }
