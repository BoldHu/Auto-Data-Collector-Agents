"""Unified SFT sample schema for Phase 7.

Supports Alpaca, ChatML, and ShareGPT formats.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class SFTSample:
    """Unified SFT sample."""
    sample_id: str = ""
    source_type: str = ""       # text|exam|knowledge_unit|agent_task|benchmark_candidate|dtcg_trace|error_repair
    task_type: str = ""
    instruction: str = ""
    input: str = ""
    output: str = ""
    system_prompt: str = ""
    evidence: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    reasoning_type: list[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    leakage_group_id: str = ""
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = f"sft_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_alpaca(self) -> dict:
        """Alpaca format: instruction/input/output."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_chatml(self) -> dict:
        """ChatML format: messages list."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = self.instruction
        if self.input:
            user_content += "\n\n" + self.input
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": self.output})
        return {"messages": messages}

    def to_sharegpt(self) -> dict:
        """ShareGPT format: conversations list."""
        conversations = []
        if self.system_prompt:
            conversations.append({"from": "system", "value": self.system_prompt})
        user_content = self.instruction
        if self.input:
            user_content += "\n\n" + self.input
        conversations.append({"from": "human", "value": user_content})
        conversations.append({"from": "gpt", "value": self.output})
        return {"conversations": conversations}


DEFAULT_SYSTEM_PROMPT = "你是一位碳纤维和复合材料领域专家。请基于提供的证据准确回答问题。"


def make_qa_sample(
    question: str,
    answer: str,
    evidence: list[str] = None,
    source_refs: list[str] = None,
    task_type: str = "qa",
    difficulty: str = "medium",
    source_type: str = "text",
    leakage_group_id: str = "",
) -> SFTSample:
    """Create a QA-style SFT sample."""
    instruction = question
    input_text = ""
    if evidence:
        input_text = "证据：\n" + "\n".join(f"- {e}" for e in evidence[:5])

    return SFTSample(
        source_type=source_type,
        task_type=task_type,
        instruction=instruction,
        input=input_text,
        output=answer,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        evidence=evidence or [],
        source_refs=source_refs or [],
        difficulty=difficulty,
        leakage_group_id=leakage_group_id,
    )


def make_exam_sample(
    question: str,
    options: list[str],
    answer: str,
    explanation: str = "",
    source_file: str = "",
    task_type: str = "exam_single_choice",
    difficulty: str = "medium",
) -> SFTSample:
    """Create an exam-style SFT sample."""
    instruction = question
    input_text = ""
    if options:
        opt_strs = []
        for o in options:
            if isinstance(o, dict):
                key = o.get("key", "")
                text = o.get("text", "")
                opt_strs.append(f"{key}. {text}" if key else text)
            else:
                opt_strs.append(str(o))
        input_text = "选项：\n" + "\n".join(opt_strs)

    output = answer
    if explanation:
        output += f"\n\n解析：{explanation}"

    return SFTSample(
        source_type="exam",
        task_type=task_type,
        instruction=instruction,
        input=input_text,
        output=output,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        source_refs=[source_file] if source_file else [],
        difficulty=difficulty,
        leakage_group_id=source_file,
    )


def make_agent_task_sample(
    scenario: str,
    question: str,
    constraints: list[str],
    answer: str,
    task_type: str = "agent_task",
    difficulty: str = "medium",
) -> SFTSample:
    """Create an agent-task SFT sample."""
    instruction = f"{scenario}\n\n{question}"
    input_text = ""
    if constraints:
        input_text = "约束条件：\n" + "\n".join(f"- {c}" for c in constraints)

    return SFTSample(
        source_type="agent_task",
        task_type=task_type,
        instruction=instruction,
        input=input_text,
        output=answer,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        difficulty=difficulty,
    )


def make_dtcg_sample(
    question: str,
    selected_context: str,
    answer: str,
    gold_answer: str = "",
    is_correct: bool = False,
    difficulty: str = "medium",
) -> SFTSample:
    """Create a DTCG reasoning SFT sample."""
    instruction = f"基于以下选择的上下文回答问题。\n\n问题：{question}"
    input_text = f"DTCG选择的上下文：\n{selected_context}"

    if is_correct:
        output = answer
    else:
        # For incorrect answers, create a correction sample
        output = f"正确答案应为：{gold_answer}\n\n之前的回答不够准确，因为选择的上下文可能未包含足够的关键信息。"

    return SFTSample(
        source_type="dtcg_trace",
        task_type="evidence_based_qa",
        instruction=instruction,
        input=input_text,
        output=output,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        difficulty=difficulty,
    )
