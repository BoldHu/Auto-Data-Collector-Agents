"""Leakage detection for SFT data.

Prevents benchmark dev/test items from leaking into training data.
"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def normalize_text(text) -> str:
    """Normalize text for comparison."""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text


def text_hash(text: str) -> str:
    """Compute hash of normalized text."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def fuzzy_similarity(a: str, b: str) -> float:
    """Compute fuzzy string similarity."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


class LeakageDetector:
    """Detects and prevents benchmark leakage in SFT data."""

    def __init__(self):
        self.benchmark_questions: dict[str, str] = {}  # hash -> original
        self.benchmark_answers: dict[str, str] = {}
        self.benchmark_ids: set[str] = set()
        self.benchmark_sources: set[str] = set()
        self.benchmark_questions_list: list[str] = []  # for fuzzy matching

    def load_benchmark(self, dev_path: Path, test_path: Path) -> None:
        """Load benchmark dev/test items for leakage checking."""
        for path in [dev_path, test_path]:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    q = item.get("question", "")
                    a = str(item.get("answer", ""))
                    bid = item.get("benchmark_id", "")

                    if q:
                        h = text_hash(q)
                        self.benchmark_questions[h] = q
                        self.benchmark_questions_list.append(q)
                    if a:
                        self.benchmark_answers[text_hash(a)] = a
                    if bid:
                        self.benchmark_ids.add(bid)

                    # Track source refs
                    for ref in item.get("source_refs", []):
                        self.benchmark_sources.add(ref)

                    # Track leakage group
                    lg = item.get("leakage_group_id", "")
                    if lg:
                        self.benchmark_sources.add(lg)

    def check_sample(self, sample: dict) -> dict:
        """Check a single SFT sample for leakage.

        Returns:
            dict with 'has_leakage' bool, 'reasons' list, 'similarity' float
        """
        reasons = []
        max_similarity = 0.0

        instruction = sample.get("instruction", "") or sample.get("question", "")
        output = sample.get("output", "") or sample.get("answer", "")
        if isinstance(output, list):
            output = " ".join(str(o) for o in output)
        output = str(output)
        sample_id = sample.get("benchmark_id", "") or sample.get("sample_id", "")
        source_refs = sample.get("source_refs", [])
        leakage_group = sample.get("leakage_group_id", "")

        # 1. Exact benchmark_id match
        if sample_id and sample_id in self.benchmark_ids:
            reasons.append(f"exact_benchmark_id: {sample_id}")

        # 2. Exact question match
        if instruction:
            q_hash = text_hash(instruction)
            if q_hash in self.benchmark_questions:
                reasons.append("exact_question_match")

        # 3. Fuzzy question match
        if instruction and self.benchmark_questions_list:
            for bq in self.benchmark_questions_list[:500]:  # limit for performance
                sim = fuzzy_similarity(instruction, bq)
                if sim > max_similarity:
                    max_similarity = sim
                if sim > 0.90:
                    reasons.append(f"fuzzy_question_match (sim={sim:.2f})")
                    break

        # 4. Source ref overlap
        for ref in source_refs:
            if ref in self.benchmark_sources:
                reasons.append(f"source_ref_overlap: {ref}")
                break

        # 5. Leakage group overlap
        if leakage_group and leakage_group in self.benchmark_sources:
            reasons.append(f"leakage_group_overlap: {leakage_group}")

        # 6. Check if output contains exact benchmark answer (only for long answers)
        if output and len(output) > 20:  # Skip short answers like "A", "B", "C"
            o_hash = text_hash(output[:200])
            if o_hash in self.benchmark_answers:
                reasons.append("exact_answer_match")

        return {
            "has_leakage": len(reasons) > 0,
            "reasons": reasons,
            "max_similarity": max_similarity,
        }

    def filter_samples(self, samples: list[dict]) -> tuple[list[dict], list[dict]]:
        """Filter samples, separating clean and leaked.

        Returns:
            (clean_samples, leaked_samples)
        """
        clean = []
        leaked = []
        for sample in samples:
            result = self.check_sample(sample)
            if result["has_leakage"]:
                sample["_leakage_result"] = result
                leaked.append(sample)
            else:
                clean.append(sample)
        return clean, leaked


def run_leakage_check(
    sft_samples: list[dict],
    dev_path: Path,
    test_path: Path,
) -> dict:
    """Run leakage check on SFT samples.

    Returns:
        Report dict with statistics and leaked samples.
    """
    detector = LeakageDetector()
    detector.load_benchmark(dev_path, test_path)

    clean, leaked = detector.filter_samples(sft_samples)

    return {
        "total_samples": len(sft_samples),
        "clean_samples": len(clean),
        "leaked_samples": len(leaked),
        "leakage_rate": len(leaked) / max(len(sft_samples), 1),
        "leakage_reasons": {},
        "clean_samples_list": clean,
        "leaked_samples_list": leaked,
    }
