"""SFT data builder for Phase 7.

Constructs 5 SFT pools from existing processed data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.autodata.finetuning.sft_schema import (
    SFTSample, make_qa_sample, make_exam_sample,
    make_agent_task_sample, make_dtcg_sample, DEFAULT_SYSTEM_PROMPT,
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


class SFTDataBuilder:
    """Builds SFT pools from processed data sources."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.benchmark_dev_ids: set[str] = set()
        self.benchmark_test_ids: set[str] = set()
        self._load_benchmark_ids()

    def _load_benchmark_ids(self):
        """Load benchmark dev/test IDs for exclusion."""
        dev_path = self.project_root / "data" / "benchmark" / "carbon_fiber_benchmark_dev.jsonl"
        test_path = self.project_root / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
        for path, id_set in [(dev_path, self.benchmark_dev_ids), (test_path, self.benchmark_test_ids)]:
            if path.exists():
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            id_set.add(d.get("benchmark_id", ""))

    def _is_in_benchmark(self, item: dict) -> bool:
        """Check if an item is in benchmark dev/test."""
        bid = item.get("benchmark_id", "") or item.get("question_id", "") or item.get("sample_id", "")
        return bid in self.benchmark_dev_ids or bid in self.benchmark_test_ids

    def build_domain_knowledge_pool(self) -> list[dict]:
        """Pool 1: Domain knowledge SFT from text, knowledge units, text-enhanced."""
        samples = []

        # From SFT candidates (text cleaning outputs)
        sft_path = self.project_root / "data" / "processed" / "sft_candidates" / "sft_candidates_pilot.jsonl"
        for item in load_jsonl(sft_path):
            if self._is_in_benchmark(item):
                continue
            sample = make_qa_sample(
                question=item.get("instruction", ""),
                answer=item.get("output", ""),
                evidence=[item.get("evidence_text", "")] if item.get("evidence_text") else [],
                source_refs=item.get("source_refs", []),
                task_type=item.get("task_type", "qa"),
                difficulty=item.get("difficulty", "medium"),
                source_type="text",
                leakage_group_id=item.get("source_chunk_id", ""),
            )
            samples.append(sample.to_dict())

        # From knowledge units
        ku_path = self.project_root / "data" / "processed" / "knowledge_units" / "knowledge_units_pilot.jsonl"
        for item in load_jsonl(ku_path):
            topic = item.get("topic", "碳纤维")
            subtopic = item.get("subtopic", "")
            claim = item.get("claim", "")
            evidence = item.get("evidence_text", "")
            if not claim:
                continue

            # Create QA from knowledge unit
            question = f"关于{topic}的{subtopic}，请解释。"
            if item.get("knowledge_type") == "definition":
                question = f"什么是{topic}？"
            elif item.get("knowledge_type") == "property":
                question = f"{topic}有哪些{subtopic}特性？"

            sample = make_qa_sample(
                question=question,
                answer=claim,
                evidence=[evidence] if evidence else [],
                source_refs=item.get("source_refs", []),
                task_type="explanation",
                difficulty="medium",
                source_type="knowledge_unit",
                leakage_group_id=item.get("source_chunk_id", ""),
            )
            samples.append(sample.to_dict())

        # From text-enhanced candidates (not in benchmark)
        te_path = self.project_root / "data" / "benchmark_candidates" / "text_enhanced" / "text_enhanced_candidates_validated.jsonl"
        for item in load_jsonl(te_path):
            if self._is_in_benchmark(item):
                continue
            q = item.get("question", "")
            a = item.get("answer", "")
            if not q or not a:
                continue
            sample = make_qa_sample(
                question=q,
                answer=a,
                evidence=item.get("evidence", []),
                source_refs=[item.get("source_file", "")],
                task_type=item.get("task_type", "qa"),
                difficulty=item.get("difficulty", "medium"),
                source_type="text",
                leakage_group_id=item.get("source_file", ""),
            )
            samples.append(sample.to_dict())

        return samples

    def build_exam_pool(self) -> list[dict]:
        """Pool 2: Exam SFT from exam questions not in benchmark."""
        samples = []
        exam_path = self.project_root / "data" / "processed" / "exam_questions" / "exam_questions_unique.jsonl"

        for item in load_jsonl(exam_path):
            if self._is_in_benchmark(item):
                continue

            q = item.get("question_text", "") or item.get("question", "")
            options = item.get("options", [])
            answer = item.get("answer", "")
            explanation = item.get("explanation", "")
            qtype = item.get("question_type", "single_choice")

            if not q or not answer:
                continue

            task_type = f"exam_{qtype}"
            sample = make_exam_sample(
                question=q,
                options=options,
                answer=answer,
                explanation=explanation,
                source_file=item.get("source_file", ""),
                task_type=task_type,
                difficulty=item.get("difficulty", "medium"),
            )
            samples.append(sample.to_dict())

        return samples

    def build_agent_task_pool(self) -> list[dict]:
        """Pool 3: Agent task SFT."""
        samples = []

        # From agent task candidates
        at_path = self.project_root / "data" / "benchmark_candidates" / "agent_task" / "agent_task_candidates_validated.jsonl"
        for item in load_jsonl(at_path):
            if self._is_in_benchmark(item):
                continue

            scenario = item.get("task_scenario", "")
            question = item.get("question", "")
            constraints = item.get("constraints", [])
            answer = item.get("answer", "")

            if not question or not answer:
                continue

            sample = make_agent_task_sample(
                scenario=scenario,
                question=question,
                constraints=constraints,
                answer=answer,
                task_type="agent_task",
                difficulty=item.get("difficulty", "medium"),
            )
            samples.append(sample.to_dict())

        return samples

    def build_dtcg_reasoning_pool(self) -> list[dict]:
        """Pool 4: DTCG reasoning SFT from Phase 6.9 traces."""
        samples = []
        traces_path = self.project_root / "data" / "evaluation" / "phase_6_9" / "targeted_rerun_traces.jsonl"

        dtcg_traces = []
        for item in load_jsonl(traces_path):
            if item.get("system_type") == "dtcg":
                dtcg_traces.append(item)

        # Use both correct and incorrect DTCG traces
        for trace in dtcg_traces:
            benchmark_id = trace.get("benchmark_id", "")
            if benchmark_id in self.benchmark_test_ids:
                continue  # Skip test items

            question = trace.get("raw_answer", "")[:200]  # Use raw_answer as it contains the question context
            selected_context = trace.get("selected_context_text", "")
            gold_answer = trace.get("gold_answer", "")
            parsed_answer = trace.get("parsed_answer", "")
            is_correct = trace.get("is_correct", False)

            if not gold_answer:
                continue

            # For the instruction, we need the original question
            # Since traces don't store the original question, use gold_answer as proxy
            sample = make_dtcg_sample(
                question=f"请回答以下碳纤维领域问题。",
                selected_context=selected_context if selected_context else "无相关上下文",
                answer=parsed_answer if is_correct else gold_answer,
                gold_answer=gold_answer,
                is_correct=is_correct,
                difficulty=trace.get("difficulty", "medium"),
            )
            sample.leakage_group_id = benchmark_id
            samples.append(sample.to_dict())

        return samples

    def build_error_correction_pool(self) -> list[dict]:
        """Pool 5: Error correction SFT from model failures."""
        samples = []
        traces_path = self.project_root / "data" / "evaluation" / "phase_6_9" / "targeted_rerun_traces.jsonl"

        for trace in load_jsonl(traces_path):
            benchmark_id = trace.get("benchmark_id", "")
            if benchmark_id in self.benchmark_test_ids:
                continue

            is_correct = trace.get("is_correct", False)
            judge_score = trace.get("judge_score", 0) or 0
            gold_answer = trace.get("gold_answer", "")
            parsed_answer = trace.get("parsed_answer", "")

            # Only use incorrect answers for error correction
            if is_correct or judge_score > 0.5:
                continue
            if not gold_answer or not parsed_answer:
                continue

            # Create error correction sample
            sample = SFTSample(
                source_type="error_repair",
                task_type="error_correction",
                instruction=f"以下回答不正确，请给出正确答案并解释原因。\n\n原问题：碳纤维领域问题\n错误回答：{parsed_answer[:200]}",
                input="",
                output=f"正确答案：{gold_answer}\n\n之前的回答未能准确匹配参考答案。在回答碳纤维领域问题时，需要严格依据提供的证据和领域知识。",
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                difficulty=trace.get("difficulty", "medium"),
                leakage_group_id=benchmark_id,
            )
            samples.append(sample.to_dict())

        return samples

    def build_all_pools(self) -> dict[str, list[dict]]:
        """Build all 5 SFT pools."""
        pools = {
            "domain_knowledge": self.build_domain_knowledge_pool(),
            "exam": self.build_exam_pool(),
            "agent_task": self.build_agent_task_pool(),
            "dtcg_reasoning": self.build_dtcg_reasoning_pool(),
            "error_correction": self.build_error_correction_pool(),
        }
        return pools

    def save_pools(self, pools: dict[str, list[dict]], output_dir: Path) -> dict[str, int]:
        """Save pools to JSONL files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for name, samples in pools.items():
            path = output_dir / f"{name}_sft.jsonl"
            save_jsonl(samples, path)
            counts[name] = len(samples)
        return counts
