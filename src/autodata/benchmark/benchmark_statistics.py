"""Benchmark statistics for Phase 5.

Generates comprehensive statistics for the final benchmark.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_statistics(items: list[dict]) -> dict:
    """Compute comprehensive benchmark statistics."""
    stats = {
        "total_items": len(items),
        "source_distribution": dict(Counter(i.get("source_type", "unknown") for i in items).most_common()),
        "modality_distribution": dict(Counter(i.get("modality", "unknown") for i in items).most_common()),
        "task_type_distribution": dict(Counter(i.get("task_type", "unknown") for i in items).most_common()),
        "difficulty_distribution": dict(Counter(i.get("difficulty", "unknown") for i in items).most_common()),
        "split_distribution": dict(Counter(i.get("split", "unknown") for i in items).most_common()),
        "validation_distribution": dict(Counter(i.get("validation_status", "unknown") for i in items).most_common()),
    }

    # Answer type distribution
    answer_types = []
    for item in items:
        answer = item.get("answer", "")
        if isinstance(answer, list):
            answer = str(answer)
        if item.get("options"):
            answer_types.append("multiple_choice")
        elif isinstance(answer, str) and answer.lower() in ["true", "false", "对", "错", "正确", "错误"]:
            answer_types.append("true_false")
        elif answer and str(answer).strip():
            answer_types.append("text_answer")
        else:
            answer_types.append("no_answer")
    stats["answer_type_distribution"] = dict(Counter(answer_types).most_common())

    # Has explanation
    has_explanation = sum(1 for i in items if i.get("explanation"))
    stats["has_explanation"] = has_explanation
    stats["has_explanation_pct"] = has_explanation / len(items) if items else 0

    # Has options
    has_options = sum(1 for i in items if i.get("options"))
    stats["has_options"] = has_options

    # Source diversity
    source_files = set()
    for item in items:
        for ref in item.get("source_refs", []):
            if ref:
                source_files.add(ref)
    stats["source_file_count"] = len(source_files)

    # Image diversity
    image_refs = set()
    for item in items:
        for ref in item.get("image_refs", []):
            if ref:
                image_refs.add(ref)
    stats["unique_images"] = len(image_refs)

    return stats


def save_statistics(stats: dict) -> tuple[Path, Path]:
    """Save statistics as JSON and MD."""
    benchmark_dir = PROJECT_ROOT / "data" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    json_path = benchmark_dir / "benchmark_statistics.json"
    md_path = benchmark_dir / "benchmark_statistics.md"

    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# 碳纤维领域基准统计\n\n")
        f.write(f"总项目数: {stats['total_items']}\n\n")

        for key in ["source_distribution", "modality_distribution", "task_type_distribution",
                     "difficulty_distribution", "split_distribution", "answer_type_distribution"]:
            if key in stats:
                f.write(f"## {key.replace('_', ' ').title()}\n\n")
                f.write("| 类别 | 数量 |\n|------|------|\n")
                for k, v in stats[key].items():
                    f.write(f"| {k} | {v} |\n")
                f.write("\n")

        f.write(f"## 其他统计\n\n")
        f.write(f"- 有解释: {stats.get('has_explanation', 0)} ({stats.get('has_explanation_pct', 0):.1%})\n")
        f.write(f"- 有选项: {stats.get('has_options', 0)}\n")
        f.write(f"- 源文件数: {stats.get('source_file_count', 0)}\n")
        f.write(f"- 唯一图像: {stats.get('unique_images', 0)}\n")

    return json_path, md_path


def generate_benchmark_card(stats: dict) -> Path:
    """Generate benchmark card."""
    benchmark_dir = PROJECT_ROOT / "data" / "benchmark"
    card_path = benchmark_dir / "CARBON_FIBER_BENCHMARK_CARD.md"

    with open(card_path, "w") as f:
        f.write("# CARBON FIBER BENCHMARK CARD\n\n")
        f.write("## 1. Benchmark Motivation\n\n")
        f.write("碳纤维领域AI代理能力评估基准，用于测试模型在碳纤维材料、复合材料、制造工艺等领域的知识和推理能力。\n\n")

        f.write("## 2. Data Sources\n\n")
        f.write("- 清洗后的碳纤维技术文献\n")
        f.write("- 考试试卷（含答案）\n")
        f.write("- 碳纤维相关图像及标注\n\n")

        f.write("## 3. Construction Pipeline\n\n")
        f.write("- Phase 1-2: 文本清洗\n")
        f.write("- Phase 3: 图像标注与多模态候选生成\n")
        f.write("- Phase 4: 考试题目提取\n")
        f.write("- Phase 5: 基准构建与验证\n\n")

        f.write("## 4. Task Taxonomy\n\n")
        f.write(f"任务类型数: {len(stats.get('task_type_distribution', {}))}\n")
        for task, count in stats.get("task_type_distribution", {}).items():
            f.write(f"- {task}: {count}\n")

        f.write("\n## 5. Data Statistics\n\n")
        f.write(f"- 总项目: {stats.get('total_items', 0)}\n")
        f.write(f"- 源文件: {stats.get('source_file_count', 0)}\n")
        f.write(f"- 唯一图像: {stats.get('unique_images', 0)}\n\n")

        f.write("## 6. Intended Use\n\n")
        f.write("用于评估碳纤维领域AI代理的知识问答、推理、图像理解等能力。\n\n")

        f.write("## 7. Limitations\n\n")
        f.write("- 考试题目数量有限（61题）\n")
        f.write("- 部分图像候选可能包含幻觉\n")
        f.write("- 主要为中文内容\n\n")

        f.write("## 8. Ethical Considerations\n\n")
        f.write("- 所有数据来自公开来源\n")
        f.write("- 不包含个人隐私信息\n")
        f.write("- API密钥已从输出中移除\n")

    return card_path
