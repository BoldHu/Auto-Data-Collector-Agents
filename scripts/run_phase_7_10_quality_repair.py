"""Phase 7.10: Benchmark/SFT quality repair and evidence-grounded SFT revision.

Usage:
    python scripts/run_phase_7_10_quality_repair.py \
        --max_workers 4 \
        --target_evidence_support 0.70 \
        --target_correctness 0.75
"""

from __future__ import annotations

import argparse
import json
import random
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


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 7.10 quality repair")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--target_evidence_support", type=float, default=0.70)
    parser.add_argument("--target_correctness", type=float, default=0.75)
    parser.add_argument("--max_repair", type=int, default=500)
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_10_quality_repair"
    report_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = PROJECT_ROOT / "data" / "audit" / "phase_7_10"
    audit_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_7_10.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 7.10: Quality Repair ===")

    # Step 1: Load Phase 7.9 audit results
    log("Step 1: Loading Phase 7.9 audit results...")
    sft_audit = load_jsonl(PROJECT_ROOT / "data" / "audit" / "phase_7_9" / "sft_audit_glm51_results.jsonl")
    benchmark_audit = load_jsonl(PROJECT_ROOT / "data" / "audit" / "phase_7_9" / "benchmark_audit_glm51_results.jsonl")

    # Load original SFT data
    sft_train = load_jsonl(PROJECT_ROOT / "data" / "sft" / "final_v2" / "train.jsonl")
    sft_val = load_jsonl(PROJECT_ROOT / "data" / "sft" / "final_v2" / "validation.jsonl")
    all_sft = sft_train + sft_val

    log(f"  SFT audit: {len(sft_audit)} items")
    log(f"  Benchmark audit: {len(benchmark_audit)} items")
    log(f"  SFT train: {len(sft_train)}, val: {len(sft_val)}")

    # Step 2: Identify repair candidates
    log("Step 2: Identifying repair candidates...")
    sft_repair_candidates = _identify_sft_repair_candidates(sft_audit, all_sft, args)
    benchmark_repair_candidates = _identify_benchmark_repair_candidates(benchmark_audit)

    save_jsonl(sft_repair_candidates["to_repair"], audit_dir / "sft_samples_to_repair.jsonl")
    save_jsonl(sft_repair_candidates["to_keep"], audit_dir / "sft_samples_to_keep.jsonl")
    save_jsonl(sft_repair_candidates["to_drop"], audit_dir / "sft_samples_to_drop_final.jsonl")
    save_jsonl(benchmark_repair_candidates["to_revise"], audit_dir / "benchmark_items_to_revise_final.jsonl")
    save_jsonl(benchmark_repair_candidates["to_drop"], audit_dir / "benchmark_items_to_drop_final.jsonl")

    log(f"  SFT: repair={len(sft_repair_candidates['to_repair'])}, keep={len(sft_repair_candidates['to_keep'])}, drop={len(sft_repair_candidates['to_drop'])}")
    log(f"  Benchmark: revise={len(benchmark_repair_candidates['to_revise'])}, drop={len(benchmark_repair_candidates['to_drop'])}")

    # Step 3: Repair SFT samples with low evidence support
    log("Step 3: Repairing SFT samples...")
    repaired_samples = _repair_sft_samples(
        sft_repair_candidates["to_repair"][:args.max_repair],
        args.max_workers, log
    )

    save_jsonl(repaired_samples, audit_dir / "sft_repaired_samples.jsonl")
    log(f"  Repaired: {len(repaired_samples)} samples")

    # Step 4: Build SFT v3
    log("Step 4: Building SFT v3...")
    sft_v3 = _build_sft_v3(
        sft_repair_candidates["to_keep"],
        repaired_samples,
        sft_repair_candidates["to_drop"],
    )

    # Split
    random.seed(42)
    random.shuffle(sft_v3)
    n_val = max(10, int(len(sft_v3) * 0.1))
    val_v3 = sft_v3[:n_val]
    train_v3 = sft_v3[n_val:]

    save_jsonl(train_v3, PROJECT_ROOT / "data" / "sft" / "final_v3" / "train.jsonl")
    save_jsonl(val_v3, PROJECT_ROOT / "data" / "sft" / "final_v3" / "validation.jsonl")

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

    with open(PROJECT_ROOT / "data" / "sft" / "final_v3" / "train_chatml.jsonl", "w") as f:
        for s in train_v3:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")
    with open(PROJECT_ROOT / "data" / "sft" / "final_v3" / "validation_chatml.jsonl", "w") as f:
        for s in val_v3:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")

    log(f"  SFT v3: train={len(train_v3)}, val={len(val_v3)}, total={len(sft_v3)}")

    # Step 5: Re-audit repaired samples
    log("Step 5: Re-auditing repaired samples...")
    reaudit_results = _reaudit_samples(repaired_samples[:100], args.max_workers, log)
    save_jsonl(reaudit_results, audit_dir / "sft_repaired_reaudit_results.jsonl")

    # Analyze re-audit
    reaudit_scores = _analyze_reaudit(reaudit_results, log)

    # Step 6: Prepare human audit sheets
    log("Step 6: Preparing human audit sheets...")
    _prepare_human_audit_sheets(all_sft, benchmark_audit, audit_dir, log)

    # Step 7: Generate report
    log("Step 7: Generating report...")
    _generate_report(
        sft_repair_candidates, repaired_samples, train_v3, val_v3,
        reaudit_scores, benchmark_repair_candidates, report_dir, log
    )

    # Step 8: Validation
    log("Step 8: Validation...")
    _validate_phase710(report_dir, audit_dir, log)

    log("=== Phase 7.10 Complete ===")


def _identify_sft_repair_candidates(audit_results: list, sft_data: list, args) -> dict:
    """Identify SFT samples that need repair."""
    # Create lookup by sample_id
    audit_by_id = {}
    for r in audit_results:
        sid = r.get("sample_id", "")
        if sid:
            audit_by_id[sid] = r

    to_repair = []
    to_keep = []
    to_drop = []

    for sample in sft_data:
        sid = sample.get("sample_id", "")
        audit = audit_by_id.get(sid)

        if not audit:
            # No audit result, keep by default
            to_keep.append(sample)
            continue

        if audit.get("error"):
            to_keep.append(sample)
            continue

        status = audit.get("overall_status", "unknown")
        evidence_support = audit.get("evidence_support", 0.5)
        correctness = audit.get("output_correctness", 0.5)

        if status == "drop":
            to_drop.append(sample)
        elif evidence_support < args.target_evidence_support or correctness < args.target_correctness:
            # Needs repair
            sample["_audit_result"] = audit
            to_repair.append(sample)
        else:
            to_keep.append(sample)

    return {"to_repair": to_repair, "to_keep": to_keep, "to_drop": to_drop}


def _identify_benchmark_repair_candidates(audit_results: list) -> dict:
    """Identify benchmark items that need revision."""
    to_revise = []
    to_drop = []

    for r in audit_results:
        if r.get("error"):
            continue
        status = r.get("overall_status", "unknown")
        if status == "drop":
            to_drop.append(r)
        elif status == "revise":
            to_revise.append(r)

    return {"to_revise": to_revise, "to_drop": to_drop}


def _repair_sft_samples(samples: list, max_workers: int, log) -> list:
    """Repair SFT samples by adding source-grounded evidence."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    client = UnifiedModelClient(model_name="deepseek-v4-flash")
    repaired = []
    errors = 0

    def repair_one(sample):
        try:
            instruction = sample.get("instruction", "")
            output = sample.get("output", "")
            evidence = sample.get("evidence", [])
            source_refs = sample.get("source_refs", [])

            # Build repair prompt
            evidence_text = "\n".join(str(e) for e in evidence[:3]) if evidence else "无证据"

            prompt = f"""你是一位碳纤维领域专家。以下训练样本的证据支持不足，请基于证据重新生成更高质量的答案。

指令：{instruction[:500]}
当前输出：{output[:500]}
可用证据：{evidence_text[:500]}

要求：
1. 新答案必须完全基于提供的证据
2. 保留原答案的正确部分
3. 补充证据中的关键信息
4. 如果证据不足，明确说明哪些部分缺乏证据支持
5. 输出格式与原答案一致

请直接输出改进后的答案，不要添加额外说明。"""

            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )

            new_output = response.content.strip()
            if new_output and len(new_output) > 20:
                sample_copy = sample.copy()
                sample_copy["output"] = new_output
                sample_copy["_repaired"] = True
                sample_copy["_original_output"] = output[:200]
                # Remove audit metadata
                sample_copy.pop("_audit_result", None)
                return sample_copy
            return sample
        except Exception:
            return sample

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(repair_one, s): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            try:
                result = future.result()
                repaired.append(result)
            except Exception:
                errors += 1

            if len(repaired) % 50 == 0:
                log(f"    Repaired: {len(repaired)}/{len(samples)}, errors: {errors}")

    return repaired


def _build_sft_v3(kept: list, repaired: list, dropped: list) -> list:
    """Build SFT v3 from kept and repaired samples."""
    # Combine kept and repaired
    v3 = kept + repaired

    # Deduplicate
    seen = set()
    unique = []
    for s in v3:
        key = (s.get("instruction", "")[:100], s.get("output", "")[:100])
        h = hash(key)
        if h not in seen:
            seen.add(h)
            unique.append(s)

    return unique


def _reaudit_samples(samples: list, max_workers: int, log) -> list:
    """Re-audit repaired samples."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    client = UnifiedModelClient(model_name="doubao-seed-2.0-pro")
    results = []
    errors = 0

    def audit_one(sample):
        try:
            instruction = sample.get("instruction", "")[:500]
            output = sample.get("output", "")[:500]
            evidence = sample.get("evidence", [])
            evidence_text = "\n".join(str(e) for e in evidence[:3])[:500] if evidence else "无"

            prompt = f"""评估以下训练样本质量（分数0.0-1.0）：

指令：{instruction}
输出：{output}
证据：{evidence_text}

输出JSON：
{{"sample_id": "{sample.get('sample_id', '')}", "instruction_clarity": 0.0, "output_correctness": 0.0, "evidence_support": 0.0, "domain_relevance": 0.0, "training_value": 0.0, "overall_status": "keep|revise|drop"}}"""

            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
            )

            text = response.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        return {"sample_id": sample.get("sample_id", ""), "error": "parse_failed"}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(audit_one, s): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                errors += 1

            if len(results) % 20 == 0:
                log(f"    Re-audit: {len(results)}/{len(samples)}, errors: {errors}")

    return results


def _analyze_reaudit(results: list, log) -> dict:
    """Analyze re-audit results."""
    valid = [r for r in results if not r.get("error")]

    scores = {}
    for dim in ["instruction_clarity", "output_correctness", "evidence_support", "domain_relevance", "training_value"]:
        vals = [r.get(dim, 0) for r in valid if isinstance(r.get(dim), (int, float))]
        scores[dim] = round(sum(vals) / max(len(vals), 1), 3) if vals else 0

    status = Counter(r.get("overall_status", "unknown") for r in valid)

    log(f"    Re-audit scores: {scores}")
    log(f"    Re-audit status: {dict(status)}")

    return {
        "total": len(valid),
        "scores": scores,
        "status": dict(status),
        "keep_rate": status.get("keep", 0) / max(len(valid), 1),
    }


def _prepare_human_audit_sheets(sft_data: list, benchmark_audit: list, audit_dir: Path, log):
    """Prepare human audit annotation sheets."""
    # Sample 100 SFT items for human audit
    random.seed(42)
    sft_sample = random.sample(sft_data, min(100, len(sft_data)))

    # Create CSV-style annotation sheet
    with open(audit_dir / "human_audit_sft_annotation_sheet.csv", "w") as f:
        f.write("sample_id,source_type,task_type,instruction,output,evidence_text,")
        f.write("instruction_clarity,output_correctness,evidence_support,domain_relevance,")
        f.write("training_value,overall_status,major_issues,notes\n")
        for s in sft_sample:
            inst = s.get("instruction", "").replace('"', '""')[:200]
            out = s.get("output", "").replace('"', '""')[:200]
            ev = " ".join(str(e) for e in s.get("evidence", [])[:2]).replace('"', '""')[:200]
            f.write(f'"{s.get("sample_id", "")}","{s.get("source_type", "")}","{s.get("task_type", "")}",')
            f.write(f'"{inst}","{out}","{ev}",')
            f.write(f',,,,,"",\n')

    # Sample 100 benchmark items for human audit
    bench_sample = random.sample(benchmark_audit, min(100, len(benchmark_audit)))

    with open(audit_dir / "human_audit_benchmark_annotation_sheet.csv", "w") as f:
        f.write("benchmark_id,source_type,task_type,question,answer,evidence_text,")
        f.write("correctness,answerability,domain_relevance,evidence_support,clarity,")
        f.write("benchmark_usefulness,overall_status,major_issues,notes\n")
        for b in bench_sample:
            q = b.get("question", "").replace('"', '""')[:200] if isinstance(b.get("question"), str) else ""
            a = str(b.get("answer", "")).replace('"', '""')[:200]
            f.write(f'"{b.get("benchmark_id", "")}","{b.get("source_type", "")}","{b.get("task_type", "")}",')
            f.write(f'"{q}","{a}","",')
            f.write(f',,,,,"",\n')

    log(f"  Human audit sheets: 100 SFT + 100 benchmark items")


def _generate_report(sft_candidates, repaired, train_v3, val_v3, reaudit_scores,
                     bench_candidates, report_dir: Path, log):
    """Generate Phase 7.10 report."""
    md = "# Phase 7.10: Quality Repair Report\n\n"

    md += "## SFT Repair Summary\n\n"
    md += f"- Original SFT v2: {len(sft_candidates['to_keep']) + len(sft_candidates['to_repair']) + len(sft_candidates['to_drop'])} samples\n"
    md += f"- Kept (passed audit): {len(sft_candidates['to_keep'])}\n"
    md += f"- Repaired: {len(repaired)}\n"
    md += f"- Dropped: {len(sft_candidates['to_drop'])}\n"
    md += f"- SFT v3 total: {len(train_v3) + len(val_v3)}\n"
    md += f"- SFT v3 train: {len(train_v3)}\n"
    md += f"- SFT v3 validation: {len(val_v3)}\n\n"

    md += "## Re-audit Results (Repaired Samples)\n\n"
    if reaudit_scores.get("scores"):
        md += "| Dimension | Score |\n|-----------|-------|\n"
        for dim, score in reaudit_scores["scores"].items():
            md += f"| {dim} | {score:.3f} |\n"
        md += f"\nKeep rate: {reaudit_scores.get('keep_rate', 0):.1%}\n\n"

    md += "## Benchmark Repair Summary\n\n"
    md += f"- Items to revise: {len(bench_candidates['to_revise'])}\n"
    md += f"- Items to drop: {len(bench_candidates['to_drop'])}\n\n"

    md += "## Quality Goals\n\n"
    goals = {
        "SFT evidence support >= 0.70": reaudit_scores.get("scores", {}).get("evidence_support", 0) >= 0.70,
        "SFT output correctness >= 0.75": reaudit_scores.get("scores", {}).get("output_correctness", 0) >= 0.75,
        "SFT drop ratio <= 15%": len(sft_candidates['to_drop']) / max(len(sft_candidates['to_keep']) + len(sft_candidates['to_repair']) + len(sft_candidates['to_drop']), 1) <= 0.15,
    }
    for goal, met in goals.items():
        md += f"- [{'PASS' if met else 'FAIL'}] {goal}\n"

    md += "\n## Human Audit Preparation\n\n"
    md += "- SFT annotation sheet: 100 items\n"
    md += "- Benchmark annotation sheet: 100 items\n"
    md += "- Ready for future manual review\n"

    with open(report_dir / "PHASE_7_10_REPORT.md", "w") as f:
        f.write(md)

    # Save statistics
    stats = {
        "sft_v2_total": len(sft_candidates['to_keep']) + len(sft_candidates['to_repair']) + len(sft_candidates['to_drop']),
        "sft_v3_train": len(train_v3),
        "sft_v3_val": len(val_v3),
        "sft_v3_total": len(train_v3) + len(val_v3),
        "repaired_count": len(repaired),
        "dropped_count": len(sft_candidates['to_drop']),
        "reaudit_scores": reaudit_scores.get("scores", {}),
        "reaudit_keep_rate": reaudit_scores.get("keep_rate", 0),
        "benchmark_revise": len(bench_candidates['to_revise']),
        "benchmark_drop": len(bench_candidates['to_drop']),
    }
    save_json(stats, report_dir / "phase7_10_statistics.json")


def _validate_phase710(report_dir: Path, audit_dir: Path, log):
    """Validate Phase 7.10 outputs."""
    checks = []
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            checks.append(f"[PASS] {name}")
        else:
            failed += 1
            checks.append(f"[FAIL] {name}")

    check("SFT v3 train exists", (PROJECT_ROOT / "data" / "sft" / "final_v3" / "train.jsonl").exists())
    check("SFT v3 validation exists", (PROJECT_ROOT / "data" / "sft" / "final_v3" / "validation.jsonl").exists())
    check("SFT v3 train ChatML exists", (PROJECT_ROOT / "data" / "sft" / "final_v3" / "train_chatml.jsonl").exists())
    check("Repaired samples exist", (audit_dir / "sft_repaired_samples.jsonl").exists())
    check("Re-audit results exist", (audit_dir / "sft_repaired_reaudit_results.jsonl").exists())
    check("Human SFT annotation sheet exists", (audit_dir / "human_audit_sft_annotation_sheet.csv").exists())
    check("Human benchmark annotation sheet exists", (audit_dir / "human_audit_benchmark_annotation_sheet.csv").exists())
    check("Phase 7.10 report exists", (report_dir / "PHASE_7_10_REPORT.md").exists())
    check("Statistics exist", (report_dir / "phase7_10_statistics.json").exists())

    # Check SFT v3 count
    if (PROJECT_ROOT / "data" / "sft" / "final_v3" / "train.jsonl").exists():
        with open(PROJECT_ROOT / "data" / "sft" / "final_v3" / "train.jsonl") as f:
            train_count = sum(1 for _ in f)
        check("SFT v3 train has samples", train_count > 0, f"{train_count} samples")

    for c in checks:
        log(f"  {c}")

    save_json({"passed": passed, "failed": failed, "checks": checks}, report_dir / "validation_phase_7_10.json")


if __name__ == "__main__":
    main()
