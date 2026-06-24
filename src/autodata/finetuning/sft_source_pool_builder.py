"""Source pool builder for Phase 7.5 SFT expansion.

Builds leakage-safe source pool from processed data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


class SourcePoolBuilder:
    """Builds a leakage-safe source pool for SFT expansion."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.benchmark_source_files: set[str] = set()
        self.benchmark_questions: set[str] = set()
        self._load_benchmark_refs()

    def _load_benchmark_refs(self):
        """Load benchmark source references for exclusion."""
        for fname in ["carbon_fiber_benchmark_dev.jsonl", "carbon_fiber_benchmark_test.jsonl"]:
            path = self.project_root / "data" / "benchmark" / fname
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    # Track source files
                    sf = d.get("source_file", "")
                    if sf:
                        self.benchmark_source_files.add(sf)
                    for ref in d.get("source_refs", []):
                        self.benchmark_source_files.add(ref)
                    # Track questions for dedup
                    q = d.get("question", "")
                    if q:
                        self.benchmark_questions.add(q[:100].strip().lower())

    def _is_safe_source(self, source_refs: list[str]) -> bool:
        """Check if source refs are safe (not in benchmark)."""
        for ref in source_refs:
            if ref in self.benchmark_source_files:
                return False
        return True

    def _is_safe_question(self, question: str) -> bool:
        """Check if question is safe (not near-duplicate of benchmark)."""
        q_norm = question[:100].strip().lower()
        return q_norm not in self.benchmark_questions

    def build_from_pretraining_corpus(self) -> list[dict]:
        """Build source pool from pretraining corpus."""
        pool = []
        corpus_path = self.project_root / "data" / "processed" / "pretraining_corpus" / "pretraining_corpus_reclean.jsonl"

        for item in load_jsonl(corpus_path):
            source_file = item.get("source_file", "")
            text = item.get("text", "")
            if not text or len(text) < 100:
                continue

            # Check safety
            safe = self._is_safe_source([source_file])

            pool.append({
                "source_id": f"corpus_{item.get('original_content_hash', '')[:12]}",
                "source_type": "cleaned_text",
                "text": text[:3000],
                "source_refs": [source_file] if source_file else [],
                "quality_score": None,
                "domain_relevance": None,
                "leakage_risk": "low" if safe else "high",
                "allowed_for_sft": safe,
                "reason": "" if safe else "source_file_in_benchmark",
            })

        return pool

    def build_from_knowledge_units(self) -> list[dict]:
        """Build source pool from knowledge units."""
        pool = []
        ku_path = self.project_root / "data" / "processed" / "knowledge_units" / "knowledge_units_pilot.jsonl"

        for item in load_jsonl(ku_path):
            claim = item.get("claim", "")
            evidence = item.get("evidence_text", "")
            source_refs = item.get("source_refs", [])

            if not claim:
                continue

            safe = self._is_safe_source(source_refs)

            pool.append({
                "source_id": item.get("unit_id", ""),
                "source_type": "knowledge_unit",
                "text": f"{claim}\n\n证据：{evidence}" if evidence else claim,
                "source_refs": source_refs,
                "quality_score": item.get("quality_score"),
                "domain_relevance": None,
                "leakage_risk": "low" if safe else "high",
                "allowed_for_sft": safe,
                "reason": "" if safe else "source_in_benchmark",
                "metadata": {
                    "topic": item.get("topic", ""),
                    "subtopic": item.get("subtopic", ""),
                    "knowledge_type": item.get("knowledge_type", ""),
                    "entities": item.get("entities", []),
                },
            })

        return pool

    def build_from_sft_candidates(self) -> list[dict]:
        """Build source pool from SFT candidates."""
        pool = []
        sft_path = self.project_root / "data" / "processed" / "sft_candidates" / "sft_candidates_pilot.jsonl"

        for item in load_jsonl(sft_path):
            source_refs = item.get("source_refs", [])
            safe = self._is_safe_source(source_refs)

            pool.append({
                "source_id": item.get("sample_id", ""),
                "source_type": "sft_candidate",
                "text": item.get("evidence_text", "") or item.get("output", ""),
                "source_refs": source_refs,
                "quality_score": item.get("quality_score"),
                "domain_relevance": None,
                "leakage_risk": "low" if safe else "high",
                "allowed_for_sft": safe,
                "reason": "" if safe else "source_in_benchmark",
            })

        return pool

    def build_from_text_enhanced(self) -> list[dict]:
        """Build source pool from text-enhanced candidates (not in benchmark)."""
        pool = []
        te_path = self.project_root / "data" / "benchmark_candidates" / "text_enhanced" / "text_enhanced_candidates_validated.jsonl"

        for item in load_jsonl(te_path):
            # Skip if in benchmark
            bid = item.get("benchmark_id", "")
            if bid and self._is_benchmark_id(bid):
                continue

            source_file = item.get("source_file", "")
            safe = self._is_safe_source([source_file])

            pool.append({
                "source_id": bid or f"te_{hash(item.get('question', '')) % 10000:04d}",
                "source_type": "text_enhanced",
                "text": item.get("evidence", [""])[0] if item.get("evidence") else "",
                "source_refs": [source_file] if source_file else [],
                "quality_score": None,
                "domain_relevance": None,
                "leakage_risk": "low" if safe else "high",
                "allowed_for_sft": safe,
                "reason": "" if safe else "source_in_benchmark",
                "metadata": {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "task_type": item.get("task_type", ""),
                },
            })

        return pool

    def build_from_agent_tasks(self) -> list[dict]:
        """Build source pool from agent task candidates."""
        pool = []
        at_path = self.project_root / "data" / "benchmark_candidates" / "agent_task" / "agent_task_candidates_validated.jsonl"

        for item in load_jsonl(at_path):
            scenario = item.get("task_scenario", "")
            question = item.get("question", "")
            answer = item.get("answer", "")

            if not question:
                continue

            pool.append({
                "source_id": f"at_{hash(question) % 10000:04d}",
                "source_type": "agent_task_source",
                "text": f"{scenario}\n\n{question}",
                "source_refs": [],
                "quality_score": None,
                "domain_relevance": None,
                "leakage_risk": "low",
                "allowed_for_sft": True,
                "reason": "",
                "metadata": {
                    "answer": answer,
                    "constraints": item.get("constraints", []),
                },
            })

        return pool

    def _is_benchmark_id(self, bid: str) -> bool:
        """Check if ID is in benchmark."""
        for fname in ["carbon_fiber_benchmark_dev.jsonl", "carbon_fiber_benchmark_test.jsonl"]:
            path = self.project_root / "data" / "benchmark" / fname
            if path.exists():
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if d.get("benchmark_id") == bid:
                                return True
        return False

    def build_all(self) -> tuple[list[dict], list[dict]]:
        """Build complete source pool. Returns (allowed, rejected)."""
        all_pool = []
        all_pool.extend(self.build_from_pretraining_corpus())
        all_pool.extend(self.build_from_knowledge_units())
        all_pool.extend(self.build_from_sft_candidates())
        all_pool.extend(self.build_from_text_enhanced())
        all_pool.extend(self.build_from_agent_tasks())

        allowed = [p for p in all_pool if p["allowed_for_sft"]]
        rejected = [p for p in all_pool if not p["allowed_for_sft"]]
        return allowed, rejected
