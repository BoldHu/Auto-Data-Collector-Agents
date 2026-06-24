"""Exam question deduplication and normalization for Phase 4.

Deduplicates and normalizes extracted exam questions.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize question text for comparison."""
    # Remove whitespace variations
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove numbering prefixes
    text = re.sub(r'^[\d]+[.、．)\s]+', '', text)
    # Normalize punctuation
    text = text.replace('，', ',').replace('。', '.').replace('；', ';')
    text = text.replace('（', '(').replace('）', ')')
    return text.lower().strip()


def normalize_options(options: list[dict]) -> str:
    """Normalize options for comparison."""
    if not options:
        return ""
    parts = []
    for opt in sorted(options, key=lambda x: x.get("key", "")):
        key = opt.get("key", "").upper()
        text = normalize_text(opt.get("text", ""))
        parts.append(f"{key}:{text}")
    return "|".join(parts)


def text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    return SequenceMatcher(None, text1, text2).ratio()


def deduplicate_questions(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deduplicate questions using multiple strategies.

    Returns:
        (unique_questions, duplicate_groups)
    """
    if not questions:
        return [], []

    # Normalize all questions
    normalized = []
    for q in questions:
        norm_text = normalize_text(q.get("question_text", ""))
        norm_opts = normalize_options(q.get("options", []))
        normalized.append({
            "original": q,
            "norm_text": norm_text,
            "norm_opts": norm_opts,
            "text_hash": hashlib.md5(norm_text.encode()).hexdigest(),
        })

    # Group by exact text hash
    hash_groups = defaultdict(list)
    for item in normalized:
        hash_groups[item["text_hash"]].append(item)

    unique_questions = []
    duplicate_groups = []

    for text_hash, group in hash_groups.items():
        if len(group) == 1:
            # Unique question
            q = group[0]["original"].copy()
            q["dedup_status"] = "unique"
            unique_questions.append(q)
        else:
            # Potential duplicates - check fuzzy similarity
            best = group[0]
            for item in group[1:]:
                sim = text_similarity(best["norm_text"], item["norm_text"])
                if sim > 0.85:
                    # Check if options also match
                    opt_sim = text_similarity(best["norm_opts"], item["norm_opts"])
                    if opt_sim > 0.8:
                        # Confirmed duplicate
                        continue
                # Different enough to be separate
                best = item

            # Keep the best one
            q = best["original"].copy()
            q["dedup_status"] = "unique_after_dedup"
            unique_questions.append(q)

            # Record duplicate group
            dup_ids = [item["original"].get("question_id", "") for item in group]
            duplicate_groups.append({
                "canonical_id": q.get("question_id", ""),
                "duplicate_ids": dup_ids,
                "count": len(group),
            })

    return unique_questions, duplicate_groups


def normalize_question_format(question: dict) -> dict:
    """Normalize question format for consistency."""
    q = question.copy()

    # Standardize question type
    qt = q.get("question_type", "unknown").lower().strip()
    type_map = {
        "单选": "single_choice", "单选题": "single_choice",
        "多选": "multiple_choice", "多选题": "multiple_choice",
        "判断": "true_false", "判断题": "true_false",
        "填空": "fill_blank", "填空题": "fill_blank",
        "简答": "short_answer", "简答题": "short_answer",
        "计算": "calculation", "计算题": "calculation",
        "案例": "case_analysis", "案例分析": "case_analysis",
    }
    q["question_type"] = type_map.get(qt, qt)

    # Standardize difficulty
    diff = q.get("difficulty", "medium").lower().strip()
    diff_map = {"容易": "easy", "简单": "easy", "困难": "hard", "较难": "hard"}
    q["difficulty"] = diff_map.get(diff, diff)

    # Standardize answer source
    ans_src = q.get("answer_source", "missing").lower().strip()
    src_map = {
        "显式答案": "explicit_answer_key", "答案键": "explicit_answer_key",
        "内嵌": "inline_solution", "模型推断": "model_inferred",
    }
    q["answer_source"] = src_map.get(ans_src, ans_src)

    # Standardize options format
    options = q.get("options", [])
    if options:
        standardized = []
        for opt in options:
            if isinstance(opt, str):
                # Parse "A. text" format
                match = re.match(r'^([A-Z])[.、．)\s]+(.+)', opt)
                if match:
                    standardized.append({"key": match.group(1), "text": match.group(2).strip()})
                else:
                    standardized.append({"key": "?", "text": opt})
            elif isinstance(opt, dict):
                standardized.append({
                    "key": opt.get("key", "?").upper(),
                    "text": opt.get("text", "").strip(),
                })
        q["options"] = standardized

    return q


def filter_benchmark_ready(questions: list[dict]) -> list[dict]:
    """Filter questions that are ready for benchmark use.

    Criteria:
    - quality_status == "keep"
    - domain_relevance >= 0.7
    - answerability >= 0.7
    - answer_consistency >= 0.7
    - extraction_confidence >= 0.7
    - answer_source in ("explicit_answer_key", "inline_solution")
    """
    ready = []
    for q in questions:
        if q.get("quality_status") != "keep":
            continue
        if q.get("domain_relevance", 0) < 0.7:
            continue
        if q.get("answerability", 0) < 0.7:
            continue
        if q.get("answer_consistency", 0) < 0.7:
            continue
        if q.get("extraction_confidence", 0) < 0.7:
            continue
        # Prefer explicit answers
        ans_src = q.get("answer_source", "missing")
        if ans_src not in ("explicit_answer_key", "inline_solution"):
            continue

        ready.append(q)

    return ready
