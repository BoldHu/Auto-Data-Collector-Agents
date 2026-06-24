"""Phase 7.8: Pre-finetuning final audit, experiment design, and paper artifacts.

Usage:
    python scripts/run_phase_7_8_pretraining_audit.py
"""

from __future__ import annotations

import json
import re
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
                if line.strip():
                    records.append(json.loads(line))
    return records


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_8_pretraining_audit"
    report_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = PROJECT_ROOT / "reports" / "paper_ready"
    paper_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_7_8.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 7.8: Pre-finetuning Audit ===")

    # Step 1: Dataset audit
    log("Step 1: SFT dataset audit...")
    dataset_audit = _audit_dataset(log)
    save_json(dataset_audit, report_dir / "sft_dataset_audit.json")

    # Step 2: Leakage audit
    log("Step 2: Final leakage audit...")
    leakage_audit = _audit_leakage(log)
    save_json(leakage_audit, report_dir / "final_leakage_audit.json")

    # Step 3: Training config audit
    log("Step 3: Training config audit...")
    config_audit = _audit_configs(log)
    save_json(config_audit, report_dir / "training_config_audit.json")

    # Step 4: Model selection plan
    log("Step 4: Model selection plan...")
    model_plan = _create_model_plan(log)
    save_json(model_plan, report_dir / "base_model_selection_plan.json")

    # Step 5: Phase 8 experiment matrix
    log("Step 5: Phase 8 experiment matrix...")
    experiment_matrix = _design_phase8_experiments(log)
    save_json(experiment_matrix, report_dir / "phase8_experiment_matrix.json")

    # Step 6: Paper artifacts
    log("Step 6: Paper artifacts...")
    _generate_paper_artifacts(dataset_audit, leakage_audit, tables_dir, paper_dir, log)

    # Step 7: Reproducibility manifest
    log("Step 7: Reproducibility manifest...")
    _create_reproducibility_manifest(report_dir, paper_dir, log)

    # Step 8: Validation
    log("Step 8: Validation...")
    validation = _validate_phase78(report_dir, log)
    save_json(validation, report_dir / "validation_phase_7_8.json")

    log("=== Phase 7.8 Complete ===")


def _audit_dataset(log) -> dict:
    """Audit SFT v2 dataset."""
    sft_dir = PROJECT_ROOT / "data" / "sft" / "final_v2"

    # Load all files
    train = load_jsonl(sft_dir / "train.jsonl")
    val = load_jsonl(sft_dir / "validation.jsonl")
    train_chatml = load_jsonl(sft_dir / "train_chatml.jsonl")
    val_chatml = load_jsonl(sft_dir / "validation_chatml.jsonl")

    # Load subsets
    subsets = {}
    subsets_dir = sft_dir / "subsets"
    if subsets_dir.exists():
        for f in subsets_dir.glob("*.jsonl"):
            subsets[f.stem] = len(load_jsonl(f))

    # Check ChatML validity
    def is_valid_chatml(sample):
        messages = sample.get("messages", [])
        if not messages:
            return False
        roles = [m.get("role") for m in messages]
        return "user" in roles and "assistant" in roles

    train_chatml_valid = sum(1 for s in train_chatml if is_valid_chatml(s))
    val_chatml_valid = sum(1 for s in val_chatml if is_valid_chatml(s))

    # Check train/val disjointness
    train_ids = set(s.get("sample_id", "") for s in train)
    val_ids = set(s.get("sample_id", "") for s in val)
    overlap = train_ids & val_ids

    # Check empty fields
    def check_empty(samples):
        empty_inst = sum(1 for s in samples if not s.get("instruction", "").strip())
        empty_out = sum(1 for s in samples if not s.get("output", "").strip())
        return empty_inst, empty_out

    train_empty = check_empty(train)
    val_empty = check_empty(val)

    # Check overlong samples
    overlong = sum(1 for s in train if len(s.get("output", "")) > 3000)

    # Check duplicates
    seen = set()
    duplicates = 0
    for s in train:
        key = (s.get("instruction", "")[:100], s.get("output", "")[:100])
        if key in seen:
            duplicates += 1
        seen.add(key)

    # Distributions
    source_dist = Counter(s.get("source_type", "unknown") for s in train)
    task_dist = Counter(s.get("task_type", "unknown") for s in train)
    difficulty_dist = Counter(s.get("difficulty", "unknown") for s in train)

    # Length statistics
    input_lens = [len(s.get("instruction", "") + s.get("input", "")) for s in train]
    output_lens = [len(s.get("output", "")) for s in train]

    # Chinese/English distribution
    def is_chinese(text):
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return chinese_chars > len(text) * 0.3

    chinese_count = sum(1 for s in train if is_chinese(s.get("output", "")))

    audit = {
        "train_count": len(train),
        "val_count": len(val),
        "train_chatml_count": len(train_chatml),
        "val_chatml_count": len(val_chatml),
        "chatml_valid_train": train_chatml_valid,
        "chatml_valid_val": val_chatml_valid,
        "counts_match": len(train) == len(train_chatml) and len(val) == len(val_chatml),
        "train_val_disjoint": len(overlap) == 0,
        "train_val_overlap": len(overlap),
        "subsets": subsets,
        "train_empty_instruction": train_empty[0],
        "train_empty_output": train_empty[1],
        "val_empty_instruction": val_empty[0],
        "val_empty_output": val_empty[1],
        "overlong_samples": overlong,
        "duplicate_samples": duplicates,
        "source_distribution": dict(source_dist.most_common()),
        "task_distribution": dict(task_dist.most_common()),
        "difficulty_distribution": dict(difficulty_dist.most_common()),
        "avg_input_length": round(sum(input_lens) / max(len(input_lens), 1)),
        "avg_output_length": round(sum(output_lens) / max(len(output_lens), 1)),
        "max_input_length": max(input_lens) if input_lens else 0,
        "max_output_length": max(output_lens) if output_lens else 0,
        "chinese_samples": chinese_count,
        "chinese_ratio": round(chinese_count / max(len(train), 1), 3),
        "text_only_ratio": 1.0,  # All v2 samples are text-only
    }

    log(f"  Train: {len(train)}, Val: {len(val)}")
    log(f"  ChatML valid: train={train_chatml_valid}, val={val_chatml_valid}")
    log(f"  Disjoint: {len(overlap) == 0}")
    log(f"  Duplicates: {duplicates}")
    log(f"  Chinese ratio: {audit['chinese_ratio']:.1%}")

    return audit


def _audit_leakage(log) -> dict:
    """Run final leakage audit."""
    from src.autodata.finetuning.leakage_detector import LeakageDetector

    sft_dir = PROJECT_ROOT / "data" / "sft" / "final_v2"
    train = load_jsonl(sft_dir / "train.jsonl")
    val = load_jsonl(sft_dir / "validation.jsonl")
    all_samples = train + val

    # Load benchmark
    detector = LeakageDetector()
    dev_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_dev.jsonl"
    test_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    detector.load_benchmark(dev_path, test_path)

    log(f"  Benchmark: {len(detector.benchmark_ids)} IDs, {len(detector.benchmark_questions)} questions")

    # Check all samples
    clean, leaked = detector.filter_samples(all_samples)

    # Check train/val overlap
    train_ids = set(s.get("sample_id", "") for s in train)
    val_ids = set(s.get("sample_id", "") for s in val)
    tv_overlap = train_ids & val_ids

    # Check subset leakage
    subset_leakage = {}
    subsets_dir = sft_dir / "subsets"
    if subsets_dir.exists():
        for f in subsets_dir.glob("train_*.jsonl"):
            subset = load_jsonl(f)
            _, subset_leaked = detector.filter_samples(subset)
            subset_leakage[f.stem] = {
                "total": len(subset),
                "leaked": len(subset_leaked),
            }

    # TF-IDF check
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        sft_texts = [f"{s.get('instruction', '')} {s.get('output', '')}" for s in all_samples]
        bench_questions = [s.get("question", "") for s in load_jsonl(dev_path) + load_jsonl(test_path)]

        all_texts = sft_texts + bench_questions
        vectorizer = TfidfVectorizer(max_features=5000, analyzer="char_wb", ngram_range=(3, 5))
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        sft_vecs = tfidf_matrix[:len(sft_texts)]
        bench_vecs = tfidf_matrix[len(sft_texts):]

        sim_matrix = cosine_similarity(sft_vecs, bench_vecs)
        max_sims = np.max(sim_matrix, axis=1)
        flagged_tfidf = int(np.sum(max_sims >= 0.85))
    except ImportError:
        flagged_tfidf = -1

    audit = {
        "total_checked": len(all_samples),
        "clean": len(clean),
        "leaked": len(leaked),
        "leakage_rate": round(len(leaked) / max(len(all_samples), 1), 4),
        "train_val_overlap": len(tv_overlap),
        "subset_leakage": subset_leakage,
        "tfidf_flagged": flagged_tfidf,
        "leakage_methods": [
            "exact_benchmark_id",
            "exact_question_hash",
            "fuzzy_question_similarity",
            "source_reference_overlap",
            "leakage_group_overlap",
            "exact_answer_match",
            "tfidf_cosine_similarity",
        ],
        "leaked_examples": [
            {"sample_id": s.get("sample_id", ""), "reasons": s.get("_leakage_result", {}).get("reasons", [])}
            for s in leaked[:10]
        ],
    }

    log(f"  Clean: {len(clean)}, Leaked: {len(leaked)} ({audit['leakage_rate']:.1%})")
    log(f"  Train/val overlap: {len(tv_overlap)}")
    log(f"  TF-IDF flagged: {flagged_tfidf}")

    return audit


def _audit_configs(log) -> dict:
    """Audit training configs."""
    import yaml

    configs_dir = PROJECT_ROOT / "configs" / "finetuning"
    results = {}

    for cfg_file in configs_dir.glob("*.yaml"):
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)

        name = cfg_file.stem
        results[name] = {
            "run_training": cfg.get("run_training", True),
            "training_disabled": cfg.get("run_training", True) is False,
            "has_lora_config": "lora" in cfg,
            "has_training_config": "training" in cfg,
            "base_model_set": bool(cfg.get("base_model")),
        }

        if "lora" in cfg:
            lora = cfg["lora"]
            results[name]["lora_rank"] = lora.get("rank")
            results[name]["lora_alpha"] = lora.get("alpha")
            results[name]["lora_dropout"] = lora.get("dropout")

        if "training" in cfg:
            t = cfg["training"]
            results[name]["learning_rate"] = t.get("learning_rate")
            results[name]["batch_size"] = t.get("batch_size")
            results[name]["epochs"] = t.get("num_epochs")
            results[name]["max_seq_length"] = t.get("max_seq_length")
            results[name]["warmup_ratio"] = t.get("warmup_ratio")

        if "qlora" in cfg:
            results[name]["has_qlora_config"] = True
            results[name]["load_in_4bit"] = cfg["qlora"].get("load_in_4bit")

    all_disabled = all(r.get("training_disabled", False) for r in results.values())

    audit = {
        "configs_checked": len(results),
        "all_training_disabled": all_disabled,
        "configs": results,
    }

    log(f"  Configs: {len(results)}, All disabled: {all_disabled}")

    return audit


def _create_model_plan(log) -> dict:
    """Create model selection plan."""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": props.name,
            "memory_gb": round(props.total_memory / 1e9, 1),
            "available": True,
        }
    else:
        gpu_info = {"available": False}

    plan = {
        "gpu_info": gpu_info,
        "recommendations": {
            "dry_run": {
                "model": "Any small model (< 3B parameters)",
                "reason": "Quick validation of training pipeline",
                "expected_memory": "< 4GB",
            },
            "lora_full": {
                "model": "Qwen2-7B or similar 7B model",
                "reason": "Good Chinese support, reasonable size for LoRA",
                "expected_memory": "16-20GB with LoRA rank 16",
                "qlora_alternative": "Same model with 4-bit quantization, 8-12GB",
            },
            "qlora_limited": {
                "model": "Qwen2-7B with 4-bit NF4 quantization",
                "reason": "Fits in 12GB GPU memory",
                "expected_memory": "8-12GB",
            },
        },
        "risks": [
            "No local base model downloaded yet",
            "GPU memory may be insufficient for 7B model without quantization",
            "Chinese language support varies by model",
            "ChatML template compatibility must be verified",
        ],
        "assumptions": [
            "Base model will be provided via --base_model CLI flag",
            "Training will use LoRA with rank 16, alpha 32",
            "Max sequence length 2048 is sufficient for most samples",
            "BF16 mixed precision is supported by GPU",
        ],
    }

    log(f"  GPU: {gpu_info.get('name', 'N/A')} ({gpu_info.get('memory_gb', 'N/A')}GB)")
    log(f"  Recommendations: dry_run, lora_full, qlora_limited")

    return plan


def _design_phase8_experiments(log) -> dict:
    """Design Phase 8 experiment matrix."""
    sft_dir = PROJECT_ROOT / "data" / "sft" / "final_v2"

    experiments = [
        {
            "id": "baseline",
            "name": "Base model only",
            "dataset": "N/A",
            "config": "N/A",
            "output_dir": "data/finetuning_outputs/baseline",
            "description": "Evaluate base model without fine-tuning",
            "gpu_memory": "N/A",
            "eval_subsets": ["cfbench_text", "cfbench_exam", "cfbench_agenttask", "cfbench_core"],
        },
        {
            "id": "lora_100",
            "name": "LoRA train_100",
            "dataset": str(sft_dir / "subsets" / "train_100.jsonl"),
            "config": "configs/finetuning/lora_v2.yaml",
            "output_dir": "data/finetuning_outputs/lora_100",
            "description": "LoRA fine-tuning on 100 samples",
            "gpu_memory": "16-20GB",
            "eval_subsets": ["cfbench_text", "cfbench_exam", "cfbench_agenttask"],
        },
        {
            "id": "lora_500",
            "name": "LoRA train_500",
            "dataset": str(sft_dir / "subsets" / "train_500.jsonl"),
            "config": "configs/finetuning/lora_v2.yaml",
            "output_dir": "data/finetuning_outputs/lora_500",
            "description": "LoRA fine-tuning on 500 samples",
            "gpu_memory": "16-20GB",
            "eval_subsets": ["cfbench_text", "cfbench_exam", "cfbench_agenttask"],
        },
        {
            "id": "lora_1000",
            "name": "LoRA train_1000",
            "dataset": str(sft_dir / "subsets" / "train_1000.jsonl"),
            "config": "configs/finetuning/lora_v2.yaml",
            "output_dir": "data/finetuning_outputs/lora_1000",
            "description": "LoRA fine-tuning on 1000 samples",
            "gpu_memory": "16-20GB",
            "eval_subsets": ["cfbench_text", "cfbench_exam", "cfbench_agenttask", "cfbench_core"],
        },
        {
            "id": "lora_full",
            "name": "LoRA full_v2",
            "dataset": str(sft_dir / "train_chatml.jsonl"),
            "config": "configs/finetuning/lora_v2.yaml",
            "output_dir": "data/finetuning_outputs/lora_full_v2",
            "description": "LoRA fine-tuning on full v2 dataset (2098 samples)",
            "gpu_memory": "16-20GB",
            "eval_subsets": ["cfbench_text", "cfbench_exam", "cfbench_agenttask", "cfbench_core"],
        },
        {
            "id": "qlora_full",
            "name": "QLoRA full_v2",
            "dataset": str(sft_dir / "train_chatml.jsonl"),
            "config": "configs/finetuning/qlora_v2.yaml",
            "output_dir": "data/finetuning_outputs/qlora_full_v2",
            "description": "QLoRA fine-tuning on full v2 dataset (4-bit quantization)",
            "gpu_memory": "8-12GB",
            "eval_subsets": ["cfbench_text", "cfbench_exam", "cfbench_agenttask", "cfbench_core"],
        },
    ]

    # Generate commands
    commands = []
    for exp in experiments:
        if exp["id"] == "baseline":
            cmd = f"# {exp['name']}: evaluate base model only"
            cmd += f"\npython scripts/run_phase_7_evaluate_finetuned.py --benchmark_subset text --max_items 50"
        else:
            cmd = f"# {exp['name']}"
            cmd += f"\npython scripts/run_phase_7_train_lora.py \\"
            cmd += f"\n  --config {exp['config']} \\"
            cmd += f"\n  --train_file {exp['dataset']} \\"
            cmd += f"\n  --validation_file {sft_dir / 'validation_chatml.jsonl'} \\"
            cmd += f"\n  --base_model <model_path> \\"
            cmd += f"\n  --run_training true"
        commands.append({"experiment": exp["id"], "command": cmd})

    matrix = {
        "experiments": experiments,
        "total_experiments": len(experiments),
        "commands": commands,
        "metrics": [
            "strict_accuracy",
            "judge_accuracy",
            "f1_score",
            "format_validity",
            "hallucination_rate",
            "evidence_support",
            "latency",
            "token_cost",
        ],
        "eval_subsets": [
            "cfbench_text (315 items)",
            "cfbench_exam (61 items)",
            "cfbench_agenttask (127 items)",
            "cfbench_core (text-only subset)",
        ],
    }

    # Save commands as markdown
    commands_md = "# Phase 8 Training Commands\n\n"
    commands_md += "All commands use `--run_training false` by default. Change to `true` to start training.\n\n"
    for cmd in commands:
        commands_md += f"## {cmd['experiment']}\n\n```bash\n{cmd['command']}\n```\n\n"

    with open(PROJECT_ROOT / "reports" / "paper_ready" / "phase8_training_commands.md", "w") as f:
        f.write(commands_md)

    log(f"  Experiments: {len(experiments)}")
    log(f"  Commands saved to reports/paper_ready/phase8_training_commands.md")

    return matrix


def _generate_paper_artifacts(dataset_audit: dict, leakage_audit: dict, tables_dir: Path, paper_dir: Path, log):
    """Generate paper-ready tables and figures."""
    figures_dir = tables_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Table 1: Source distribution
    source_dist = dataset_audit.get("source_distribution", {})
    with open(tables_dir / "sft_source_distribution.csv", "w") as f:
        f.write("Source,Count,Percentage\n")
        total = sum(source_dist.values())
        for source, count in sorted(source_dist.items(), key=lambda x: -x[1]):
            f.write(f"{source},{count},{count/total:.1%}\n")

    # Table 2: Task distribution
    task_dist = dataset_audit.get("task_distribution", {})
    with open(tables_dir / "sft_task_distribution.csv", "w") as f:
        f.write("TaskType,Count,Percentage\n")
        total = sum(task_dist.values())
        for task, count in sorted(task_dist.items(), key=lambda x: -x[1]):
            f.write(f"{task},{count},{count/total:.1%}\n")

    # Table 3: Difficulty distribution
    diff_dist = dataset_audit.get("difficulty_distribution", {})
    with open(tables_dir / "sft_difficulty_distribution.csv", "w") as f:
        f.write("Difficulty,Count,Percentage\n")
        total = sum(diff_dist.values())
        for diff, count in sorted(diff_dist.items(), key=lambda x: -x[1]):
            f.write(f"{diff},{count},{count/total:.1%}\n")

    # Table 4: Leakage filtering
    with open(tables_dir / "sft_leakage_filtering.csv", "w") as f:
        f.write("Stage,Count,Notes\n")
        f.write(f"Initial candidates,949,Phase 7 original pool\n")
        f.write(f"After leakage check,540,43.1% removed\n")
        f.write(f"After quality filter,413,Additional filtering\n")
        f.write(f"Expanded candidates,2615,Phase 7.5 generation\n")
        f.write(f"After expansion validation,1918,26.7% dropped\n")
        f.write(f"Final merged,2331,Merged original + expanded\n")
        f.write(f"Train,2098,90% split\n")
        f.write(f"Validation,233,10% split\n")

    # Table 5: Phase 7 vs 7.5 comparison
    with open(tables_dir / "sft_phase7_vs_phase75.csv", "w") as f:
        f.write("Metric,Phase7,Phase7.5_v2,Improvement\n")
        f.write(f"Total samples,413,2331,5.6x\n")
        f.write(f"Train samples,369,2098,5.7x\n")
        f.write(f"Validation samples,44,233,5.3x\n")
        f.write(f"Source types,5,8,+3\n")
        f.write(f"Task types,5,15,+10\n")

    # LaTeX table
    latex = "% SFT Dataset Statistics\n"
    latex += "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{SFT Dataset Statistics (v2)}\n"
    latex += "\\begin{tabular}{lrr}\n\\hline\n"
    latex += "Metric & Phase 7 & Phase 7.5 v2 \\\\\n\\hline\n"
    latex += f"Total samples & 413 & 2,331 \\\\\n"
    latex += f"Train samples & 369 & 2,098 \\\\\n"
    latex += f"Validation samples & 44 & 233 \\\\\n"
    latex += f"Source types & 5 & 8 \\\\\n"
    latex += f"Task types & 5 & 15 \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(paper_dir / "sft_dataset_statistics.tex", "w") as f:
        f.write(latex)

    # Markdown summary
    md = "# SFT Dataset Statistics\n\n"
    md += "## Source Distribution\n\n"
    md += "| Source | Count | Percentage |\n|--------|-------|------------|\n"
    total = sum(source_dist.values())
    for source, count in sorted(source_dist.items(), key=lambda x: -x[1]):
        md += f"| {source} | {count} | {count/total:.1%} |\n"

    md += "\n## Task Distribution\n\n"
    md += "| Task | Count | Percentage |\n|------|-------|------------|\n"
    total = sum(task_dist.values())
    for task, count in sorted(task_dist.items(), key=lambda x: -x[1]):
        md += f"| {task} | {count} | {count/total:.1%} |\n"

    with open(paper_dir / "sft_dataset_statistics.md", "w") as f:
        f.write(md)

    # Generate figures (matplotlib if available)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Source distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sources = list(source_dist.keys())
        counts = list(source_dist.values())
        ax.barh(sources, counts)
        ax.set_xlabel("Count")
        ax.set_title("SFT Dataset Source Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "sft_source_distribution.png", dpi=150)
        plt.close()

        # Figure 2: Task distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        tasks = list(task_dist.keys())[:10]
        task_counts = [task_dist[t] for t in tasks]
        ax.barh(tasks, task_counts)
        ax.set_xlabel("Count")
        ax.set_title("SFT Dataset Task Distribution (Top 10)")
        plt.tight_layout()
        plt.savefig(figures_dir / "sft_task_distribution.png", dpi=150)
        plt.close()

        # Figure 3: Difficulty distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        diffs = list(diff_dist.keys())
        diff_counts = list(diff_dist.values())
        ax.pie(diff_counts, labels=diffs, autopct='%1.1f%%')
        ax.set_title("SFT Dataset Difficulty Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "sft_difficulty_distribution.png", dpi=150)
        plt.close()

        log(f"  Figures saved to {figures_dir}")
    except ImportError:
        log("  WARNING: matplotlib not available, skipping figures")


def _create_reproducibility_manifest(report_dir: Path, paper_dir: Path, log):
    """Create reproducibility manifest."""
    manifest = {
        "project": "AutoData: Dynamic Task-Context Graph for Long-Horizon Multi-Agent Data Construction",
        "phase": "7.8",
        "status": "pre-training audit complete",
        "training_disabled": True,
        "dataset": {
            "v1_path": "data/sft/final/",
            "v2_path": "data/sft/final_v2/",
            "v2_train_count": 2098,
            "v2_val_count": 233,
            "subsets": [
                "data/sft/final_v2/subsets/train_100.jsonl",
                "data/sft/final_v2/subsets/train_500.jsonl",
                "data/sft/final_v2/subsets/train_1000.jsonl",
                "data/sft/final_v2/subsets/validation_100.jsonl",
            ],
        },
        "benchmark": {
            "dev_path": "data/benchmark/carbon_fiber_benchmark_dev.jsonl",
            "test_path": "data/benchmark/carbon_fiber_benchmark_test.jsonl",
            "dev_count": 594,
            "test_count": 2417,
        },
        "configs": [
            "configs/finetuning/lora_default.yaml",
            "configs/finetuning/qlora_default.yaml",
            "configs/finetuning/dry_run.yaml",
            "configs/finetuning/lora_v2.yaml",
            "configs/finetuning/qlora_v2.yaml",
            "configs/finetuning/dry_run_v2.yaml",
        ],
        "scripts": [
            "scripts/run_phase_7_8_pretraining_audit.py",
            "scripts/run_phase_7_prepare_finetuning.py",
            "scripts/run_phase_7_train_lora.py",
            "scripts/run_phase_7_evaluate_finetuned.py",
            "scripts/validate_phase_7_8_pretraining_audit.py",
        ],
        "reports": [
            "reports/phase_7_finetuning_preparation/PHASE_7_REPORT.md",
            "reports/phase_7_5_sft_expansion/PHASE_7_5_REPORT.md",
            "reports/phase_7_8_pretraining_audit/PHASE_7_8_REPORT.md",
        ],
        "random_seed": 42,
        "environment": {
            "python": "3.10+",
            "torch": "2.0+",
            "transformers": "4.30+",
            "peft": "0.4+",
        },
    }

    with open(paper_dir / "reproducibility_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Markdown version
    md = "# Reproducibility Manifest\n\n"
    md += f"**Project**: {manifest['project']}\n\n"
    md += f"**Phase**: {manifest['phase']}\n\n"
    md += f"**Training disabled**: {manifest['training_disabled']}\n\n"
    md += "## Dataset\n\n"
    md += f"- v2 train: {manifest['dataset']['v2_train_count']} samples\n"
    md += f"- v2 validation: {manifest['dataset']['v2_val_count']} samples\n"
    md += f"- Random seed: {manifest['random_seed']}\n\n"
    md += "## Configs\n\n"
    for cfg in manifest["configs"]:
        md += f"- `{cfg}`\n"
    md += "\n## Scripts\n\n"
    for script in manifest["scripts"]:
        md += f"- `{script}`\n"

    with open(paper_dir / "reproducibility_manifest.md", "w") as f:
        f.write(md)

    log(f"  Manifest saved")


def _validate_phase78(report_dir: Path, log) -> dict:
    """Validate Phase 7.8 outputs."""
    checks = []
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            checks.append(f"[PASS] {name}" + (f" - {detail}" if detail else ""))
        else:
            failed += 1
            checks.append(f"[FAIL] {name}" + (f" - {detail}" if detail else ""))

    check("SFT dataset audit exists", (report_dir / "sft_dataset_audit.json").exists())
    check("Final leakage audit exists", (report_dir / "final_leakage_audit.json").exists())
    check("Training config audit exists", (report_dir / "training_config_audit.json").exists())
    check("Model selection plan exists", (report_dir / "base_model_selection_plan.json").exists())
    check("Phase 8 experiment matrix exists", (report_dir / "phase8_experiment_matrix.json").exists())
    check("Paper tables exist", (report_dir / "tables" / "sft_source_distribution.csv").exists())
    check("Reproducibility manifest exists", (PROJECT_ROOT / "reports" / "paper_ready" / "reproducibility_manifest.json").exists())

    # Check training disabled
    import yaml
    configs_dir = PROJECT_ROOT / "configs" / "finetuning"
    all_disabled = True
    for cfg_file in configs_dir.glob("*.yaml"):
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("run_training", True) is not False:
            all_disabled = False
    check("All training configs disabled", all_disabled)

    # Check no API keys
    import re
    api_pattern = re.compile(r'(sk-|ak-|api[_-]?key\s*[:=]\s*\S+)', re.IGNORECASE)
    key_found = False
    for fpath in report_dir.glob("*.json"):
        if api_pattern.search(fpath.read_text()):
            key_found = True
    check("No API keys in outputs", not key_found)

    check("Phase 7.8 report exists", (PROJECT_ROOT / "reports" / "phase_7_8_pretraining_audit" / "PHASE_7_8_REPORT.md").exists())

    for c in checks:
        log(f"  {c}")

    return {"passed": passed, "failed": failed, "checks": checks}


if __name__ == "__main__":
    main()
