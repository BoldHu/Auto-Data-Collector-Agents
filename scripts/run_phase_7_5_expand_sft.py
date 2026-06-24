"""Phase 7.5: SFT dataset expansion and enhancement.

Usage:
    python scripts/run_phase_7_5_expand_sft.py \
        --target_candidates 3000 \
        --max_workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 7.5 SFT expansion")
    parser.add_argument("--target_candidates", type=int, default=3000)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_sources", type=int, default=0)
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_5_sft_expansion"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = report_dir / "progress_phase_7_5.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 7.5: SFT Expansion ===")

    # Step 1: Statistics audit
    log("Step 1: Phase 7 statistics audit...")
    _run_statistics_audit(PROJECT_ROOT, report_dir, log)

    # Step 2: Build source pool
    log("Step 2: Building source pool...")
    from src.autodata.finetuning.sft_source_pool_builder import SourcePoolBuilder

    builder = SourcePoolBuilder(PROJECT_ROOT)
    allowed, rejected = builder.build_all()

    save_jsonl(allowed, PROJECT_ROOT / "data" / "sft" / "source_pool" / "sft_source_pool.jsonl")
    save_jsonl(rejected, PROJECT_ROOT / "data" / "sft" / "source_pool" / "sft_source_pool_rejected.jsonl")

    log(f"  Source pool: {len(allowed)} allowed, {len(rejected)} rejected")

    # Source type distribution
    source_types = Counter(s.get("source_type", "") for s in allowed)
    log(f"  Source types: {dict(source_types)}")

    # Step 3: TF-IDF embedding leakage check on existing SFT
    log("Step 3: TF-IDF leakage check on existing SFT...")
    _run_tfidf_leakage_check(PROJECT_ROOT, report_dir, log)

    # Step 4: Expand SFT samples
    log("Step 4: Expanding SFT samples...")
    from src.autodata.finetuning.sft_expander import expand_sft_from_source_pool
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    client = UnifiedModelClient(model_name="deepseek-v4-flash")

    # Limit sources if needed
    sources_to_use = allowed
    if args.max_sources > 0:
        sources_to_use = allowed[:args.max_sources]

    # Calculate how many samples per source
    target = args.target_candidates
    per_source = max(1, target // max(len(sources_to_use), 1))
    per_source = min(per_source, 3)  # Max 3 per source to avoid overfitting

    log(f"  Using {len(sources_to_use)} sources, {per_source} samples per source")

    expanded = expand_sft_from_source_pool(
        sources_to_use, client, target_per_source=per_source, max_workers=args.max_workers
    )

    save_jsonl(expanded, PROJECT_ROOT / "data" / "sft" / "expanded" / "sft_expanded_candidates.jsonl")
    log(f"  Generated {len(expanded)} candidates")

    # Step 5: Validate expanded samples
    log("Step 5: Validating expanded samples...")
    validated, dropped = _validate_expanded(expanded, PROJECT_ROOT, log)

    save_jsonl(validated, PROJECT_ROOT / "data" / "sft" / "expanded" / "sft_expanded_validated.jsonl")
    save_jsonl(dropped, PROJECT_ROOT / "data" / "sft" / "expanded" / "sft_expanded_rejected.jsonl")
    log(f"  Validated: {len(validated)}, Dropped: {len(dropped)}")

    # Step 6: Merge with original
    log("Step 6: Merging with original Phase 7 data...")
    _merge_and_split(validated, PROJECT_ROOT, log)

    # Step 7: Build scaling subsets
    log("Step 7: Building scaling subsets...")
    _build_scaling_subsets(PROJECT_ROOT, log)

    log("=== Phase 7.5 Complete ===")


def _run_statistics_audit(project_root: Path, report_dir: Path, log):
    """Audit Phase 7 statistics."""
    # Load counts
    train = load_jsonl(project_root / "data" / "sft" / "final" / "train.jsonl")
    val = load_jsonl(project_root / "data" / "sft" / "final" / "validation.jsonl")
    train_chatml = load_jsonl(project_root / "data" / "sft" / "final" / "train_chatml.jsonl")
    val_chatml = load_jsonl(project_root / "data" / "sft" / "final" / "validation_chatml.jsonl")
    validated = load_jsonl(project_root / "data" / "sft" / "validated" / "sft_validated_all.jsonl")

    audit = {
        "train_count": len(train),
        "val_count": len(val),
        "train_val_total": len(train) + len(val),
        "validated_count": len(validated),
        "train_chatml_count": len(train_chatml),
        "val_chatml_count": len(val_chatml),
        "train_val_equals_validated": len(train) + len(val) == len(validated),
        "chatml_matches_raw": len(train) == len(train_chatml) and len(val) == len(val_chatml),
    }

    # Check disjoint
    train_ids = set(s.get("sample_id", "") for s in train)
    val_ids = set(s.get("sample_id", "") for s in val)
    audit["train_val_disjoint"] = len(train_ids & val_ids) == 0

    # Source type distribution
    train_sources = Counter(s.get("source_type", "unknown") for s in train)
    val_sources = Counter(s.get("source_type", "unknown") for s in val)
    audit["train_source_dist"] = dict(train_sources)
    audit["val_source_dist"] = dict(val_sources)

    # Check metadata
    has_leakage_group = sum(1 for s in train if s.get("leakage_group_id"))
    has_source_refs = sum(1 for s in train if s.get("source_refs"))
    audit["train_with_leakage_group"] = has_leakage_group
    audit["train_with_source_refs"] = has_source_refs

    # Fix: report says 40 val but actual is 44
    audit["report_says_40_actual_44"] = len(val) == 44

    with open(report_dir / "phase7_statistics_audit.json", "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    log(f"  Train: {len(train)}, Val: {len(val)}, Total: {len(train)+len(val)}")
    log(f"  ChatML matches: {audit['chatml_matches_raw']}")
    log(f"  Train/val disjoint: {audit['train_val_disjoint']}")
    log(f"  Report says 40, actual 44: {audit['report_says_40_actual_44']}")


def _run_tfidf_leakage_check(project_root: Path, report_dir: Path, log):
    """Run TF-IDF based leakage check."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        log("  WARNING: sklearn not available, skipping TF-IDF check")
        return

    # Load SFT samples
    sft_samples = load_jsonl(project_root / "data" / "sft" / "final" / "train.jsonl")
    sft_texts = [f"{s.get('instruction', '')} {s.get('input', '')} {s.get('output', '')}" for s in sft_samples]

    # Load benchmark questions
    bench_questions = []
    bench_answers = []
    for fname in ["carbon_fiber_benchmark_dev.jsonl", "carbon_fiber_benchmark_test.jsonl"]:
        path = project_root / "data" / "benchmark" / fname
        for item in load_jsonl(path):
            bench_questions.append(item.get("question", ""))
            bench_answers.append(str(item.get("answer", "")))

    if not sft_texts or not bench_questions:
        log("  WARNING: No data for TF-IDF check")
        return

    # Compute TF-IDF
    all_texts = sft_texts + bench_questions + bench_answers
    vectorizer = TfidfVectorizer(max_features=10000, analyzer="char_wb", ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    sft_vectors = tfidf_matrix[:len(sft_texts)]
    bench_vectors = tfidf_matrix[len(sft_texts):]

    # Compute similarities
    sim_matrix = cosine_similarity(sft_vectors, bench_vectors)

    # Find high similarity pairs
    flagged = []
    threshold = 0.85
    for i in range(len(sft_texts)):
        max_sim = float(np.max(sim_matrix[i]))
        if max_sim >= threshold:
            max_idx = int(np.argmax(sim_matrix[i]))
            flagged.append({
                "sft_index": i,
                "sft_sample_id": sft_samples[i].get("sample_id", ""),
                "max_similarity": round(max_sim, 4),
                "bench_index": max_idx,
            })

    report = {
        "sft_samples_checked": len(sft_texts),
        "bench_questions_checked": len(bench_questions),
        "flagged_count": len(flagged),
        "flagged_rate": len(flagged) / max(len(sft_texts), 1),
        "threshold": threshold,
        "flagged_examples": flagged[:20],
    }

    with open(report_dir / "embedding_leakage_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if flagged:
        save_jsonl(
            [sft_samples[f["sft_index"]] for f in flagged],
            project_root / "data" / "sft" / "validated" / "embedding_leakage_flagged.jsonl",
        )

    log(f"  TF-IDF checked: {len(sft_texts)} SFT vs {len(bench_questions)} benchmark")
    log(f"  Flagged (>= {threshold}): {len(flagged)} ({len(flagged)/max(len(sft_texts),1):.1%})")


def _validate_expanded(samples: list[dict], project_root: Path, log) -> tuple[list[dict], list[dict]]:
    """Validate expanded SFT samples."""
    validated = []
    dropped = []

    # Load benchmark questions for dedup
    bench_questions = set()
    for fname in ["carbon_fiber_benchmark_dev.jsonl", "carbon_fiber_benchmark_test.jsonl"]:
        path = project_root / "data" / "benchmark" / fname
        for item in load_jsonl(path):
            q = item.get("question", "")
            if q:
                bench_questions.add(q[:100].strip().lower())

    for sample in samples:
        issues = []
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")

        # Rule-based checks
        if not instruction or len(instruction.strip()) < 10:
            issues.append("short_instruction")
        if not output or len(output.strip()) < 10:
            issues.append("short_output")
        if len(output) > 3000:
            issues.append("output_too_long")

        # Check for hallucination markers
        hallucination_markers = ["我不知道", "无法回答", "抱歉", "sorry"]
        for marker in hallucination_markers:
            if marker in output:
                issues.append(f"hallucination_marker: {marker}")

        # Check domain relevance
        domain_terms = ["碳纤维", "复合材料", "CFRP", "碳化", "纤维", "树脂", "基体",
                        "carbon", "fiber", "composite", "PAN", "预浸料"]
        text = instruction + output
        if not any(term in text for term in domain_terms):
            issues.append("no_domain_relevance")

        # Check benchmark dedup
        q_norm = instruction[:100].strip().lower()
        if q_norm in bench_questions:
            issues.append("benchmark_question_duplicate")

        # Quality score
        quality_score = 1.0 - len(issues) * 0.2
        quality_score = max(0.0, min(1.0, quality_score))

        sample["_quality_score"] = quality_score
        sample["_quality_issues"] = issues

        if quality_score >= 0.6 and len(issues) == 0:
            validated.append(sample)
        else:
            dropped.append(sample)

    return validated, dropped


def _merge_and_split(expanded: list[dict], project_root: Path, log):
    """Merge original and expanded SFT, create final splits."""
    import random
    random.seed(42)

    # Load original validated
    original = load_jsonl(project_root / "data" / "sft" / "validated" / "sft_validated_all.jsonl")

    # Merge
    all_samples = original + expanded

    # Deduplicate
    seen = set()
    unique = []
    for s in all_samples:
        key = (s.get("instruction", "")[:100] + "||" + s.get("output", "")[:100]).strip()
        h = hash(key)
        if h not in seen:
            seen.add(h)
            unique.append(s)

    log(f"  Merged: {len(original)} original + {len(expanded)} expanded = {len(all_samples)} total")
    log(f"  After dedup: {len(unique)}")

    # Balance source types
    source_counts = Counter(s.get("source_type", "unknown") for s in unique)
    log(f"  Source distribution: {dict(source_counts)}")

    # Split
    random.shuffle(unique)
    n_val = max(10, int(len(unique) * 0.1))
    val = unique[:n_val]
    train = unique[n_val:]

    # Save
    output_dir = project_root / "data" / "sft" / "final_v2"
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "validation.jsonl")

    # ChatML format
    def to_chatml(sample):
        messages = []
        if sample.get("system_prompt"):
            messages.append({"role": "system", "content": sample["system_prompt"]})
        user_content = sample.get("instruction", "")
        if sample.get("input"):
            user_content += "\n\n" + sample["input"]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": sample.get("output", "")})
        return {"messages": messages}

    with open(output_dir / "train_chatml.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")
    with open(output_dir / "validation_chatml.jsonl", "w") as f:
        for s in val:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")

    # Statistics
    stats = {
        "total": len(unique),
        "train": len(train),
        "validation": len(val),
        "source_distribution": dict(Counter(s.get("source_type", "unknown") for s in unique)),
        "task_distribution": dict(Counter(s.get("task_type", "unknown") for s in unique)),
        "difficulty_distribution": dict(Counter(s.get("difficulty", "unknown") for s in unique)),
    }
    with open(output_dir / "sft_dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log(f"  Final v2: train={len(train)}, val={len(val)}, total={len(unique)}")
    log(f"  Sources: {stats['source_distribution']}")


def _build_scaling_subsets(project_root: Path, log):
    """Build small training subsets for scaling experiments."""
    import random
    random.seed(42)

    train = load_jsonl(project_root / "data" / "sft" / "final_v2" / "train.jsonl")
    val = load_jsonl(project_root / "data" / "sft" / "final_v2" / "validation.jsonl")

    subsets_dir = project_root / "data" / "sft" / "final_v2" / "subsets"

    # Training subsets
    for n in [100, 500, 1000]:
        subset = train[:min(n, len(train))]
        save_jsonl(subset, subsets_dir / f"train_{n}.jsonl")
        log(f"  Subset train_{n}: {len(subset)} samples")

    # Validation subset
    val_subset = val[:min(100, len(val))]
    save_jsonl(val_subset, subsets_dir / "validation_100.jsonl")
    log(f"  Subset validation_100: {len(val_subset)} samples")


if __name__ == "__main__":
    main()
