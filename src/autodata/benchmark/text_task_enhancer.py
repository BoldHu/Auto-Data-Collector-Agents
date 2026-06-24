"""Text task enhancer for Phase 5.5.

Generates additional text-only benchmark items from pretraining corpus and knowledge units.
Uses API_KEY1 only.
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent

# Text generation prompt
TEXT_BENCHMARK_SYSTEM_PROMPT = """你是一位碳纤维领域专家和基准题目设计专家。你的任务是基于提供的文本源材料，生成高质量的碳纤维领域基准测试题目。

关键规则：
1. 每个题目必须有证据支持（evidence_text）
2. 答案必须由证据文本直接支持，不得编造
3. 生成多种推理类型的题目
4. 难度应有梯度（easy/medium/hard）
5. 保持中文
6. 题目应测试深度理解，而非表面记忆"""

TEXT_BENCHMARK_USER_PROMPT = """请基于以下碳纤维领域文本，生成{count}个高质量基准测试题目。

文本内容：
{text_content}

文本来源：{source_file}

请生成以下类型的题目（尽量覆盖多种类型）：
- source_grounded_reasoning: 基于文本的推理
- process_reasoning: 工艺流程推理
- causal_reasoning: 因果关系推理
- constraint_satisfaction: 约束满足
- error_diagnosis: 错误诊断
- information_extraction: 信息提取
- calculation: 数值计算（如果有数值数据）
- comparison: 材料/工艺比较
- parameter_interpretation: 参数解释
- mechanism_explanation: 机理说明

输出JSON数组格式：
{output_schema}"""

TEXT_BENCHMARK_OUTPUT_SCHEMA = """[
  {
    "task_type": "process_reasoning",
    "question": "题目文本",
    "options": [],
    "answer": "答案文本",
    "explanation": "解释",
    "evidence": ["证据文本片段"],
    "reasoning_type": ["process_reasoning"],
    "difficulty": "medium",
    "required_knowledge": ["碳纤维", "氧化工艺"]
  }
]"""

TEXT_CRITIC_SYSTEM_PROMPT = """你是一位独立的基准题目质量评审专家。请评估以下题目质量。

评估维度：
- clarity: 题目清晰度
- completeness: 完整性
- answerability: 可回答性
- evidence_support: 证据支持度
- domain_relevance: 领域相关性
- reasoning_depth: 推理深度

检测问题：
- 答案无证据支持
- 题目模糊或多义
- 答案出现在题目中
- 过于简单或琐碎
- 重复题目"""

TEXT_CRITIC_USER_PROMPT = """请评估以下基准题目：

{question_json}

请输出JSON：
{{
  "quality_status": "keep|review|drop",
  "clarity": 0.0-1.0,
  "completeness": 0.0-1.0,
  "answerability": 0.0-1.0,
  "evidence_support": 0.0-1.0,
  "domain_relevance": 0.0-1.0,
  "reasoning_depth": 0.0-1.0,
  "detected_issues": [],
  "revision_suggestion": ""
}}"""


def generate_text_items(
    pool,
    text_content: str,
    source_file: str,
    count: int = 5,
) -> list[dict]:
    """Generate text benchmark items from a text chunk."""
    user_prompt = TEXT_BENCHMARK_USER_PROMPT.replace(
        "{count}", str(count)
    ).replace(
        "{text_content}", text_content[:6000]
    ).replace(
        "{source_file}", source_file
    ).replace(
        "{output_schema}", TEXT_BENCHMARK_OUTPUT_SCHEMA
    )

    try:
        response = pool.chat(
            messages=[
                {"role": "system", "content": TEXT_BENCHMARK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=8192,
            temperature=0.7,
        )
        response_text = response.content
    except Exception:
        return []

    # Parse response
    json_start = response_text.find("[")
    json_end = response_text.rfind("]") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            items = json.loads(response_text[json_start:json_end])
            return items
        except json.JSONDecodeError:
            pass
    return []


def validate_text_item(pool, item: dict) -> dict:
    """Validate a text benchmark item using independent critic."""
    question_json = json.dumps(item, ensure_ascii=False, indent=2)
    user_prompt = TEXT_CRITIC_USER_PROMPT.replace("{question_json}", question_json)

    try:
        response = pool.chat_quality(
            messages=[
                {"role": "system", "content": TEXT_CRITIC_SYSTEM_PROMPT},
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
