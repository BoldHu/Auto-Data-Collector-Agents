"""Judge prompts for Phase 6 LLM judge.

Separate from generation prompts to ensure independent evaluation.
"""

JUDGE_SYSTEM_PROMPT = """你是一位碳纤维领域的基准测试评审专家。你的任务是独立评估AI模型对基准题目的回答质量。

评估维度：
1. correctness: 答案正确性（0.0-1.0）
2. evidence_support: 证据支持度（0.0-1.0）
3. reasoning_quality: 推理质量（0.0-1.0）
4. hallucination: 幻觉程度（0.0=无幻觉, 1.0=严重幻觉）
5. format_validity: 格式有效性（0.0-1.0）

判定标准：
- 对于选择题和数值题，严格比较答案
- 对于简答题，接受语义等价的表述
- 不奖励无证据支持但看似合理的答案
- 检测幻觉：如果模型编造了不存在的事实，标记为幻觉

输出严格JSON格式。"""

JUDGE_USER_PROMPT = """请评估以下回答：

题目：{question}
参考答案：{gold_answer}
模型回答：{model_answer}
证据：{evidence}

请输出JSON：
{{
  "correctness": 0.0,
  "evidence_support": 0.0,
  "reasoning_quality": 0.0,
  "hallucination": 0.0,
  "format_validity": 0.0,
  "final_score": 0.0,
  "verdict": "correct|partially_correct|incorrect|invalid",
  "rationale": "评估理由"
}}"""

AGENT_TASK_JUDGE_PROMPT = """请评估以下代理任务回答：

场景：{scenario}
约束条件：{constraints}
评分标准：{rubric}
模型回答：{model_answer}

请输出JSON：
{{
  "correctness": 0.0,
  "evidence_support": 0.0,
  "reasoning_quality": 0.0,
  "hallucination": 0.0,
  "format_validity": 0.0,
  "final_score": 0.0,
  "verdict": "correct|partially_correct|incorrect|invalid",
  "rationale": "评估理由"
}}"""


def build_judge_prompt(item: dict, model_answer: str) -> str:
    """Build judge prompt for a benchmark item."""
    if item.get("source_type") == "agent_task" or item.get("task_type") == "agent_task":
        return AGENT_TASK_JUDGE_PROMPT.format(
            scenario=item.get("task_scenario", item.get("question", "")),
            constraints="\n".join(item.get("constraints", [])),
            rubric=item.get("scoring_rubric", ""),
            model_answer=model_answer,
        )

    evidence = "\n".join(item.get("evidence", [])[:3])
    return JUDGE_USER_PROMPT.format(
        question=item.get("question", ""),
        gold_answer=item.get("answer", ""),
        model_answer=model_answer,
        evidence=evidence[:2000],
    )
