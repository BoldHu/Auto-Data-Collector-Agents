"""SFT expander for Phase 7.5.

Generates new SFT samples from leakage-safe source pool using LLM.
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.autodata.finetuning.sft_schema import SFTSample, DEFAULT_SYSTEM_PROMPT


# Task type templates
TASK_TEMPLATES = {
    "domain_knowledge_qa": {
        "instruction": "基于以下碳纤维领域知识，回答问题。\n\n知识：{text}\n\n问题：{question}",
        "output_format": "基于提供的知识，{answer}",
    },
    "source_grounded_reasoning": {
        "instruction": "基于以下证据进行推理分析。\n\n证据：{text}\n\n问题：{question}",
        "output_format": "根据证据分析，{answer}",
    },
    "process_reasoning": {
        "instruction": "解释以下碳纤维相关工艺过程。\n\n背景：{text}\n\n问题：{question}",
        "output_format": "该工艺过程如下：{answer}",
    },
    "extraction": {
        "instruction": "从以下文本中提取关键信息。\n\n文本：{text}\n\n请提取：{question}",
        "output_format": "提取结果：{answer}",
    },
    "error_correction": {
        "instruction": "以下回答存在错误，请指出错误并给出正确答案。\n\n原问题：{question}\n错误回答：{wrong_answer}\n\n参考知识：{text}",
        "output_format": "错误分析：{analysis}\n\n正确答案：{answer}",
    },
    "comparison": {
        "instruction": "比较以下两种碳纤维相关概念或工艺。\n\n参考材料：{text}\n\n问题：{question}",
        "output_format": "比较分析：{answer}",
    },
}


def generate_qa_from_chunk(client, chunk_text: str, chunk_id: str, task_type: str = "domain_knowledge_qa") -> dict | None:
    """Generate a QA sample from a text chunk."""
    prompt = f"""你是一位碳纤维领域专家。基于以下文本，生成一个高质量的问答对。

文本：
{chunk_text[:1500]}

要求：
1. 问题应基于文本内容，不能脱离文本
2. 答案必须完全基于文本，不能编造
3. 问题应有实际意义，不能是trivial的
4. 难度为中等

请输出JSON格式：
{{"question": "...", "answer": "...", "difficulty": "medium", "reasoning_type": ["..."]}}"""

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        text = response.content.strip()
        # Parse JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return {
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "difficulty": data.get("difficulty", "medium"),
                "reasoning_type": data.get("reasoning_type", []),
            }
    except Exception:
        pass
    return None


def generate_reasoning_from_chunk(client, chunk_text: str, chunk_id: str) -> dict | None:
    """Generate a reasoning sample from a text chunk."""
    prompt = f"""你是一位碳纤维领域专家。基于以下文本，生成一个需要推理分析的问题。

文本：
{chunk_text[:1500]}

要求：
1. 问题需要基于文本进行推理，不能直接从文本中找到答案
2. 答案必须基于文本中的证据
3. 包含因果关系或对比分析

请输出JSON格式：
{{"question": "...", "answer": "...", "evidence": ["..."], "difficulty": "hard", "reasoning_type": ["causal_reasoning"]}}"""

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return {
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "evidence": data.get("evidence", []),
                "difficulty": data.get("difficulty", "hard"),
                "reasoning_type": data.get("reasoning_type", ["causal_reasoning"]),
            }
    except Exception:
        pass
    return None


def generate_extraction_from_chunk(client, chunk_text: str, chunk_id: str) -> dict | None:
    """Generate an information extraction sample."""
    prompt = f"""你是一位碳纤维领域专家。基于以下文本，生成一个信息提取任务。

文本：
{chunk_text[:1500]}

要求：
1. 提取任务应要求从文本中找出特定信息（如参数、性质、工艺步骤等）
2. 答案应为从文本中提取的结构化信息
3. 任务应有实际应用价值

请输出JSON格式：
{{"question": "提取...", "answer": "...", "difficulty": "medium", "reasoning_type": ["extraction"]}}"""

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return {
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "difficulty": data.get("difficulty", "medium"),
                "reasoning_type": data.get("reasoning_type", ["extraction"]),
            }
    except Exception:
        pass
    return None


def create_sft_sample(
    source_item: dict,
    generated: dict,
    task_type: str,
    generator_model: str = "deepseek-v4-flash",
) -> dict:
    """Create an SFT sample from source and generated content."""
    question = generated.get("question", "")
    answer = generated.get("answer", "")
    evidence_text = source_item.get("text", "")[:500]

    instruction = question
    input_text = ""
    if evidence_text:
        input_text = f"参考证据：\n{evidence_text}"

    sample = SFTSample(
        source_type=source_item.get("source_type", "text"),
        task_type=task_type,
        instruction=instruction,
        input=input_text,
        output=answer,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        evidence=[evidence_text] if evidence_text else [],
        source_refs=source_item.get("source_refs", []),
        difficulty=generated.get("difficulty", "medium"),
        reasoning_type=generated.get("reasoning_type", []),
        leakage_group_id=source_item.get("source_id", ""),
        metadata={
            "created_by": "phase_7_5_sft_expander",
            "generator_model": generator_model,
            "validated": False,
            "excluded_from_benchmark": True,
            "source_grounded": True,
        },
    )
    return sample.to_dict()


def expand_sft_from_source_pool(
    source_pool: list[dict],
    client,
    target_per_source: int = 2,
    max_workers: int = 4,
) -> list[dict]:
    """Generate SFT samples from source pool."""
    samples = []
    errors = 0

    # Task type distribution
    task_types = [
        ("domain_knowledge_qa", 0.30),
        ("source_grounded_reasoning", 0.25),
        ("process_reasoning", 0.15),
        ("extraction", 0.15),
        ("comparison", 0.15),
    ]

    def process_source(source_item):
        results = []
        text = source_item.get("text", "")
        if len(text) < 100:
            return results

        # Generate QA
        for i in range(target_per_source):
            try:
                if i == 0:
                    generated = generate_qa_from_chunk(client, text, source_item.get("source_id", ""))
                    task_type = "domain_knowledge_qa"
                elif i == 1:
                    generated = generate_reasoning_from_chunk(client, text, source_item.get("source_id", ""))
                    task_type = "source_grounded_reasoning"
                else:
                    generated = generate_extraction_from_chunk(client, text, source_item.get("source_id", ""))
                    task_type = "extraction"

                if generated and generated.get("question") and generated.get("answer"):
                    sample = create_sft_sample(source_item, generated, task_type)
                    results.append(sample)
            except Exception:
                pass
        return results

    # Process sources with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_source, s): s for s in source_pool}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                samples.extend(result)
            except Exception:
                errors += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(source_pool)} sources, {len(samples)} samples generated")

    return samples
