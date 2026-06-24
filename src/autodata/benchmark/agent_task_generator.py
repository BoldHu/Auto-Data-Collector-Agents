"""Agent task generator for Phase 5.5.

Generates benchmark items that evaluate long-horizon domain agent capabilities.
Uses API_KEY1 only.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent

AGENT_TASK_SYSTEM_PROMPT = """你是一位多智能体系统和碳纤维领域专家。你的任务是生成评估AI代理在碳纤维领域数据构建能力的基准测试题目。

这些题目应评估代理是否能够：
1. 选择有用的领域数据源
2. 识别有噪声的OCR文本
3. 决定文本块是否应保留或丢弃
4. 分类碳纤维图像相关性
5. 检测重复或近重复数据
6. 验证生成的基准题目是否可回答
7. 从多个源片段中选择证据
8. 识别编造的标签或描述
9. 从源材料构建小型基准题目
10. 在约束下推理（如"保留公式"或"不使用无支持的事实"）
11. 规划多步数据构建工作流
12. 比较广播通信和DTCG选择上下文

每个题目应包含：
- 任务场景（task_scenario）
- 输入数据或摘要（input_artifacts）
- 约束条件（constraints）
- 期望决策或答案（expected_decision）
- 评分标准（scoring_rubric）"""

AGENT_TASK_USER_PROMPT = """请生成{count}个评估AI代理碳纤维领域数据构建能力的基准测试题目。

要求：
- 每个题目应模拟真实的数据构建场景
- 包含具体的输入数据和约束条件
- 答案应明确可评判
- 包含评分标准

输出JSON数组格式：
{output_schema}"""

AGENT_TASK_OUTPUT_SCHEMA = """[
  {
    "task_type": "agent_task",
    "task_scenario": "场景描述",
    "question": "完整题目文本（包含场景和问题）",
    "input_artifacts": ["输入数据描述1", "输入数据描述2"],
    "constraints": ["约束1", "约束2"],
    "answer": "期望答案或决策",
    "scoring_rubric": "评分标准",
    "explanation": "解释为什么这是正确答案",
    "reasoning_type": ["constraint_satisfaction", "evidence_selection"],
    "difficulty": "medium",
    "required_knowledge": ["碳纤维", "数据清洗"]
  }
]"""

AGENT_CRITIC_SYSTEM_PROMPT = """你是一位独立的代理任务基准评审专家。请评估以下代理任务题目的质量。

评估维度：
- scenario_realism: 场景真实度
- task_clarity: 任务清晰度
- answer_judgeability: 答案可评判性
- constraint_quality: 约束质量
- scoring_rubric_quality: 评分标准质量
- domain_relevance: 领域相关性
- benchmark_usefulness: 基准有用性"""

AGENT_CRITIC_USER_PROMPT = """请评估以下代理任务题目：

{question_json}

请输出JSON：
{{
  "quality_status": "keep|review|drop",
  "scenario_realism": 0.0-1.0,
  "task_clarity": 0.0-1.0,
  "answer_judgeability": 0.0-1.0,
  "constraint_quality": 0.0-1.0,
  "scoring_rubric_quality": 0.0-1.0,
  "domain_relevance": 0.0-1.0,
  "benchmark_usefulness": 0.0-1.0,
  "detected_issues": []
}}"""


def generate_agent_tasks(pool, count: int = 10) -> list[dict]:
    """Generate agent task benchmark items."""
    user_prompt = AGENT_TASK_USER_PROMPT.replace(
        "{count}", str(count)
    ).replace("{output_schema}", AGENT_TASK_OUTPUT_SCHEMA)

    try:
        response = pool.chat(
            messages=[
                {"role": "system", "content": AGENT_TASK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=8192,
            temperature=0.7,
        )
        response_text = response.content
    except Exception:
        return []

    json_start = response_text.find("[")
    json_end = response_text.rfind("]") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass
    return []


def validate_agent_task(pool, item: dict) -> dict:
    """Validate an agent task benchmark item."""
    question_json = json.dumps(item, ensure_ascii=False, indent=2)
    user_prompt = AGENT_CRITIC_USER_PROMPT.replace("{question_json}", question_json)

    try:
        response = pool.chat_quality(
            messages=[
                {"role": "system", "content": AGENT_CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2048,
            temperature=0.7,
        )
        response_text = response.content
    except Exception:
        return {"quality_status": "review", "detected_issues": ["validation_failed"]}

    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass
    return {"quality_status": "review", "detected_issues": ["parse_failed"]}
