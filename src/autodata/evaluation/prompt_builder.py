"""Prompt builder for Phase 6 evaluation.

Builds evaluation prompts for different task types.
"""

from __future__ import annotations


def build_eval_prompt(item: dict) -> str:
    """Build evaluation prompt for a benchmark item."""
    task_type = item.get("task_type", "unknown")
    question = item.get("question", "")
    options = item.get("options", [])
    source_type = item.get("source_type", "")

    # Agent task: include scenario and constraints
    if source_type == "agent_task" or task_type == "agent_task":
        return _build_agent_task_prompt(item)

    # Multiple choice
    if options and len(options) >= 2:
        return _build_mc_prompt(question, options)

    # Calculation
    if task_type in ("exam_calculation",) or "计算" in question:
        return _build_calc_prompt(question)

    # True/false
    if task_type == "exam_true_false" or any(kw in question for kw in ["判断", "是否", "对错"]):
        return _build_tf_prompt(question)

    # Fill blank
    if task_type == "exam_fill_blank" or "填空" in question or "____" in question:
        return _build_fill_prompt(question)

    # Default: short answer
    return _build_short_answer_prompt(question)


def _build_mc_prompt(question: str, options: list) -> str:
    """Build multiple-choice prompt."""
    opt_text = "\n".join(
        f"{opt.get('key', '?')}. {opt.get('text', '')}"
        for opt in options
        if isinstance(opt, dict)
    )
    return f"""请回答以下选择题。只输出选项字母（如 A、B、C、D），不要输出其他内容。

题目：{question}

选项：
{opt_text}

答案："""


def _build_calc_prompt(question: str) -> str:
    """Build calculation prompt."""
    return f"""请解答以下计算题。先简要说明计算过程，然后给出最终数值答案。

题目：{question}

解答："""


def _build_tf_prompt(question: str) -> str:
    """Build true/false prompt."""
    return f"""请判断以下陈述是否正确。只输出"正确"或"错误"。

题目：{question}

答案："""


def _build_fill_prompt(question: str) -> str:
    """Build fill-in-the-blank prompt."""
    return f"""请填写以下填空题的答案。

题目：{question}

答案："""


def _build_short_answer_prompt(question: str) -> str:
    """Build short answer prompt."""
    return f"""请简洁地回答以下问题。

问题：{question}

答案："""


def _build_agent_task_prompt(item: dict) -> str:
    """Build agent task prompt."""
    scenario = item.get("task_scenario", item.get("question", ""))
    artifacts = item.get("input_artifacts", [])
    constraints = item.get("constraints", [])
    rubric = item.get("scoring_rubric", "")

    parts = [f"场景：{scenario}"]

    if artifacts:
        parts.append("输入数据：")
        for a in artifacts:
            parts.append(f"- {a}")

    if constraints:
        parts.append("约束条件：")
        for c in constraints:
            parts.append(f"- {c}")

    if rubric:
        parts.append(f"评分标准：{rubric}")

    parts.append("请给出你的决策或答案，并解释理由。")
    return "\n\n".join(parts)
