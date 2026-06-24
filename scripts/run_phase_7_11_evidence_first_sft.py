"""Phase 7.11: Evidence-first SFT reconstruction.

Usage:
    python scripts/run_phase_7_11_evidence_first_sft.py \
        --max_workers 4 \
        --target_candidates 2000
"""

from __future__ import annotations

import argparse
import json
import random
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
    parser = argparse.ArgumentParser(description="Phase 7.11 evidence-first SFT")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--target_candidates", type=int, default=2000)
    parser.add_argument("--max_per_evidence", type=int, default=2)
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_11_evidence_first_sft"
    report_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_7_11.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 7.11: Evidence-First SFT Reconstruction ===")

    # Step 1: Analyze Phase 7.10 failure
    log("Step 1: Analyzing Phase 7.10 repair failure...")
    failure_analysis = _analyze_repair_failure(log)
    save_json(failure_analysis, report_dir / "repair_failure_analysis.json")

    # Step 2: Build evidence bank
    log("Step 2: Building evidence bank...")
    evidence_bank = _build_evidence_bank(log)
    save_jsonl(evidence_bank["units"], PROJECT_ROOT / "data" / "sft" / "evidence_bank" / "evidence_bank.jsonl")
    save_jsonl(evidence_bank["rejected"], PROJECT_ROOT / "data" / "sft" / "evidence_bank" / "evidence_bank_rejected.jsonl")
    save_json(evidence_bank["summary"], report_dir / "evidence_bank_summary.json")
    log(f"  Evidence bank: {len(evidence_bank['units'])} units, {len(evidence_bank['rejected'])} rejected")

    # Step 3: Build task evidence pools
    log("Step 3: Building task evidence pools...")
    task_pools = _build_task_pools(evidence_bank["units"], log)
    for pool_name, pool_items in task_pools.items():
        save_jsonl(pool_items, PROJECT_ROOT / "data" / "sft" / "evidence_bank" / "task_pools" / f"{pool_name}.jsonl")
    save_json({k: len(v) for k, v in task_pools.items()}, report_dir / "task_evidence_pool_summary.json")
    log(f"  Task pools: {', '.join(f'{k}={len(v)}' for k, v in task_pools.items())}")

    # Step 4: Generate evidence-first SFT
    log("Step 4: Generating evidence-first SFT...")
    candidates = _generate_evidence_first_sft(evidence_bank["units"], task_pools, args, log)
    save_jsonl(candidates, PROJECT_ROOT / "data" / "sft" / "evidence_first" / "evidence_first_candidates.jsonl")
    log(f"  Generated: {len(candidates)} candidates")

    # Step 5: Validate and split into gold/silver
    log("Step 5: Validating evidence-first SFT...")
    gold, silver, rejected = _validate_and_split(candidates, args.max_workers, log)
    save_jsonl(gold, PROJECT_ROOT / "data" / "sft" / "evidence_first" / "sft_gold_candidates.jsonl")
    save_jsonl(silver, PROJECT_ROOT / "data" / "sft" / "evidence_first" / "sft_silver_candidates.jsonl")
    save_jsonl(rejected, PROJECT_ROOT / "data" / "sft" / "evidence_first" / "sft_rejected_candidates.jsonl")
    log(f"  Gold: {len(gold)}, Silver: {len(silver)}, Rejected: {len(rejected)}")

    # Step 6: Build SFT v4
    log("Step 6: Building SFT v4...")
    v4_stats = _build_sft_v4(gold, silver, log)
    save_json(v4_stats, report_dir / "sft_v4_build_report.json")

    # Step 7: Audit v4 sample
    log("Step 7: Auditing SFT v4 sample...")
    audit_results = _audit_v4_sample(gold, silver, args.max_workers, log)
    save_json(audit_results, report_dir / "sft_v4_audit_summary.json")

    # Step 8: Generate paper artifacts
    log("Step 8: Generating paper artifacts...")
    _generate_paper_artifacts(evidence_bank, task_pools, gold, silver, v4_stats, audit_results, report_dir, log)

    # Step 9: Prepare human audit sheets
    log("Step 9: Preparing human audit sheets...")
    _prepare_human_audit_sheets(gold, silver, log)

    # Step 10: Validation
    log("Step 10: Validation...")
    _validate_phase711(report_dir, log)

    log("=== Phase 7.11 Complete ===")


def _analyze_repair_failure(log) -> dict:
    """Analyze why Phase 7.10 repair failed."""
    # Load reaudit results
    reaudit_path = PROJECT_ROOT / "data" / "audit" / "phase_7_10" / "sft_repaired_reaudit_results.jsonl"
    reaudit = load_jsonl(reaudit_path)

    errors = sum(1 for r in reaudit if r.get("error"))
    valid = [r for r in reaudit if not r.get("error")]

    analysis = {
        "total_repaired": len(reaudit),
        "audit_errors": errors,
        "error_rate": errors / max(len(reaudit), 1),
        "valid_results": len(valid),
        "scores": {},
        "failure_reasons": [
            "Repair prompt changed correct answers incorrectly",
            "Evidence was generic and not specific to the question",
            "Generator introduced unsupported facts",
            "Repair over-compressed answers",
            "domain_knowledge_qa samples were too generic",
            "source_refs insufficient for evidence grounding",
            "Auditor judged harshly because evidence not included in audit prompt",
        ],
    }

    if valid:
        for dim in ["evidence_support", "output_correctness", "domain_relevance", "training_value"]:
            vals = [r.get(dim, 0) for r in valid if isinstance(r.get(dim), (int, float))]
            analysis["scores"][dim] = round(sum(vals) / max(len(vals), 1), 3) if vals else 0

    log(f"  Repair failure: {errors}/{len(reaudit)} audit errors")
    log(f"  Scores: {analysis['scores']}")

    return analysis


def _build_evidence_bank(log) -> dict:
    """Build high-quality evidence bank."""
    units = []
    rejected = []

    # Load pretraining corpus
    corpus_path = PROJECT_ROOT / "data" / "processed" / "pretraining_corpus" / "pretraining_corpus_reclean.jsonl"
    corpus = load_jsonl(corpus_path)
    log(f"  Pretraining corpus: {len(corpus)} chunks")

    # Load knowledge units
    ku_path = PROJECT_ROOT / "data" / "processed" / "knowledge_units" / "knowledge_units_pilot.jsonl"
    knowledge_units = load_jsonl(ku_path)
    log(f"  Knowledge units: {len(knowledge_units)}")

    # Load source pool
    pool_path = PROJECT_ROOT / "data" / "sft" / "source_pool" / "sft_source_pool.jsonl"
    source_pool = load_jsonl(pool_path)
    log(f"  Source pool: {len(source_pool)}")

    # Filter corpus for quality
    for i, chunk in enumerate(corpus):
        text = chunk.get("text", "")
        source_file = chunk.get("source_file", "")

        # Quality checks
        if len(text) < 200:
            rejected.append({"source_id": f"corpus_{i}", "reason": "too_short"})
            continue

        # Check for carbon fiber relevance
        cf_terms = ["碳纤维", "碳化", "纤维", "复合材料", "CFRP", "PAN", "预浸料", "层压", "基体", "树脂"]
        if not any(term in text for term in cf_terms):
            rejected.append({"source_id": f"corpus_{i}", "reason": "no_cf_relevance"})
            continue

        # Check for OCR noise
        noise_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in "，。、；：""''（）《》【】…—")
        if noise_chars > len(text) * 0.1:
            rejected.append({"source_id": f"corpus_{i}", "reason": "high_ocr_noise"})
            continue

        # Check for boilerplate
        boilerplate = ["版权所有", "出版社", "作者简介", "目录", "前言", "参考文献"]
        if any(bp in text for bp in boilerplate):
            rejected.append({"source_id": f"corpus_{i}", "reason": "boilerplate"})
            continue

        # Extract domain terms
        domain_terms = [term for term in cf_terms if term in text]

        # Compute evidence density (ratio of technical terms)
        tech_terms = ["强度", "模量", "密度", "拉伸", "压缩", "弯曲", "剪切", "疲劳", "韧性", "刚度",
                      "碳化温度", "石墨化", "预氧化", "原丝", "前驱体", "固化", "固化剂", "固化温度"]
        tech_count = sum(1 for t in tech_terms if t in text)
        evidence_density = tech_count / max(len(tech_terms), 1)

        units.append({
            "evidence_id": f"corpus_{chunk.get('original_content_hash', '')[:12]}",
            "source_type": "cleaned_text",
            "text": text[:2000],
            "source_refs": [source_file] if source_file else [],
            "domain_terms": domain_terms,
            "evidence_density": round(evidence_density, 3),
            "length_chars": len(text),
            "allowed_task_types": ["domain_knowledge_qa", "source_grounded_reasoning", "process_reasoning",
                                   "mechanism_explanation", "information_extraction", "parameter_interpretation"],
            "leakage_risk": "low",
        })

    # Add knowledge units
    for ku in knowledge_units:
        claim = ku.get("claim", "")
        evidence = ku.get("evidence_text", "")
        if not claim or len(claim) < 20:
            continue

        units.append({
            "evidence_id": ku.get("unit_id", ""),
            "source_type": "knowledge_unit",
            "text": f"{claim}\n\n{evidence}" if evidence else claim,
            "source_refs": ku.get("source_refs", []),
            "domain_terms": ku.get("entities", []),
            "evidence_density": 0.5,
            "length_chars": len(claim) + len(evidence),
            "allowed_task_types": ["domain_knowledge_qa", "information_extraction", "comparison"],
            "leakage_risk": "low",
        })

    summary = {
        "total_units": len(units),
        "total_rejected": len(rejected),
        "source_types": dict(Counter(u["source_type"] for u in units)),
        "avg_length": round(sum(u["length_chars"] for u in units) / max(len(units), 1)),
        "avg_evidence_density": round(sum(u.get("evidence_density", 0) for u in units) / max(len(units), 1), 3),
    }

    return {"units": units, "rejected": rejected, "summary": summary}


def _build_task_pools(evidence_units: list, log) -> dict:
    """Build task-specific evidence pools."""
    pools = {
        "domain_knowledge_qa": [],
        "source_grounded_reasoning": [],
        "process_reasoning": [],
        "mechanism_causal": [],
        "agent_task": [],
        "dtcg_evidence_selection": [],
        "error_correction": [],
    }

    for unit in evidence_units:
        allowed_tasks = unit.get("allowed_task_types", [])
        for task in allowed_tasks:
            if task in pools:
                pools[task].append(unit)
            elif "reasoning" in task:
                pools["source_grounded_reasoning"].append(unit)
            elif "extraction" in task or "interpretation" in task or "comparison" in task:
                pools["domain_knowledge_qa"].append(unit)

    return pools


def _generate_evidence_first_sft(evidence_units: list, task_pools: dict, args, log) -> list:
    """Generate evidence-first SFT samples."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    client = UnifiedModelClient(model_name="deepseek-v4-flash")
    candidates = []
    errors = 0

    # Sample evidence units for generation
    random.seed(42)
    sampled_units = random.sample(evidence_units, min(len(evidence_units), args.target_candidates // args.max_per_evidence))

    def generate_from_evidence(unit):
        results = []
        text = unit.get("text", "")[:1500]
        source_type = unit.get("source_type", "cleaned_text")
        allowed_tasks = unit.get("allowed_task_types", ["domain_knowledge_qa"])

        for i in range(min(args.max_per_evidence, 2)):
            try:
                # Select task type
                task_type = random.choice(allowed_tasks) if allowed_tasks else "domain_knowledge_qa"

                # Build generation prompt
                if task_type == "domain_knowledge_qa":
                    prompt = f"""基于以下碳纤维领域证据，生成一个高质量问答对。

证据：
{text[:1000]}

要求：
1. 问题必须基于证据内容
2. 答案必须完全基于证据，不能添加证据中没有的信息
3. 问题应有实际意义，不能是简单的定义问题
4. 答案应引用或转述证据中的关键信息

输出JSON格式：
{{"question": "...", "answer": "...", "difficulty": "medium", "reasoning_type": ["..."]}}"""

                elif task_type == "source_grounded_reasoning":
                    prompt = f"""基于以下碳纤维领域证据，生成一个需要推理分析的问题。

证据：
{text[:1000]}

要求：
1. 问题需要基于证据进行推理
2. 答案必须基于证据中的信息
3. 包含因果关系或对比分析

输出JSON格式：
{{"question": "...", "answer": "...", "difficulty": "hard", "reasoning_type": ["causal_reasoning"]}}"""

                elif task_type == "process_reasoning":
                    prompt = f"""基于以下碳纤维领域证据，生成一个关于工艺过程的问题。

证据：
{text[:1000]}

要求：
1. 问题涉及工艺步骤、参数或流程
2. 答案必须基于证据中的工艺描述

输出JSON格式：
{{"question": "...", "answer": "...", "difficulty": "medium", "reasoning_type": ["process_reasoning"]}}"""

                else:
                    prompt = f"""基于以下碳纤维领域证据，生成一个高质量问答对。

证据：
{text[:1000]}

要求：
1. 问题类型：{task_type}
2. 答案必须完全基于证据

输出JSON格式：
{{"question": "...", "answer": "...", "difficulty": "medium", "reasoning_type": ["..."]}}"""

                response = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.7,
                )

                resp_text = response.content.strip()
                start = resp_text.find("{")
                end = resp_text.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(resp_text[start:end])
                    question = data.get("question", "")
                    answer = data.get("answer", "")

                    if question and answer and len(answer) > 20:
                        sample = {
                            "sample_id": f"ef_{unit.get('evidence_id', '')[:12]}_{i}",
                            "source_type": "evidence_first",
                            "task_type": task_type,
                            "instruction": question,
                            "input": f"证据：\n{text[:500]}",
                            "output": answer,
                            "evidence": [text[:500]],
                            "source_refs": unit.get("source_refs", []),
                            "evidence_ids": [unit.get("evidence_id", "")],
                            "difficulty": data.get("difficulty", "medium"),
                            "reasoning_type": data.get("reasoning_type", []),
                            "leakage_group_id": unit.get("source_refs", [""])[0] if unit.get("source_refs") else "",
                            "system_prompt": "你是一位碳纤维领域专家。请基于提供的证据准确回答问题。",
                            "metadata": {
                                "created_by": "phase_7_11_evidence_first_generator",
                                "generator_model": "deepseek-v4-flash",
                                "source_grounded": True,
                                "evidence_first": True,
                                "excluded_from_benchmark": True,
                            },
                        }
                        results.append(sample)
            except Exception:
                pass
        return results

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(generate_from_evidence, u): u for u in sampled_units}
        for future in as_completed(futures):
            try:
                results = future.result()
                candidates.extend(results)
            except Exception:
                errors += 1

            if len(candidates) % 100 == 0 and len(candidates) > 0:
                log(f"    Generated: {len(candidates)} candidates, {errors} errors")

    log(f"    Total: {len(candidates)} candidates, {errors} errors")
    return candidates


def _validate_and_split(candidates: list, max_workers: int, log) -> tuple[list, list, list]:
    """Validate candidates and split into gold/silver/rejected."""
    random.seed(42)

    gold = []
    silver = []
    rejected = []

    for sample in candidates:
        issues = []
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        evidence = sample.get("evidence", [])

        # Rule-based validation
        if not instruction or len(instruction.strip()) < 10:
            issues.append("short_instruction")
        if not output or len(output.strip()) < 20:
            issues.append("short_output")
        if not evidence:
            issues.append("no_evidence")

        # Check evidence support - does output share content with evidence?
        evidence_text = " ".join(str(e) for e in evidence[:3])
        output_words = set(re.findall(r'[\w\u4e00-\u9fff]+', output[:300]))
        evidence_words = set(re.findall(r'[\w\u4e00-\u9fff]+', evidence_text[:1000]))
        overlap = len(output_words & evidence_words)
        evidence_support_ratio = overlap / max(len(output_words), 1)

        if evidence_support_ratio < 0.15:
            issues.append("low_evidence_support")

        # Check for hallucination markers
        hallucination_markers = ["我不知道", "无法回答", "抱歉"]
        for marker in hallucination_markers:
            if marker in output:
                issues.append(f"hallucination_marker:{marker}")

        # Check domain relevance
        domain_terms = ["碳纤维", "复合材料", "CFRP", "碳化", "纤维", "树脂", "基体", "PAN"]
        if not any(term in instruction + output for term in domain_terms):
            issues.append("no_domain_relevance")

        # Split based on quality
        if len(issues) == 0 and evidence_support_ratio >= 0.25:
            gold.append(sample)
        elif len(issues) <= 1 and evidence_support_ratio >= 0.15:
            silver.append(sample)
        else:
            sample["_rejection_reasons"] = issues
            rejected.append(sample)

    log(f"    Gold: {len(gold)}, Silver: {len(silver)}, Rejected: {len(rejected)}")
    return gold, silver, rejected


def _build_sft_v4(gold: list, silver: list, log) -> dict:
    """Build SFT v4 datasets."""
    random.seed(42)

    # Gold dataset
    gold_all = gold.copy()
    random.shuffle(gold_all)
    gold_n_val = max(5, int(len(gold_all) * 0.1))
    gold_val = gold_all[:gold_n_val]
    gold_train = gold_all[gold_n_val:]

    # Silver dataset (gold + silver)
    silver_all = gold + silver
    random.shuffle(silver_all)
    silver_n_val = max(10, int(len(silver_all) * 0.1))
    silver_val = silver_all[:silver_n_val]
    silver_train = silver_all[silver_n_val:]

    # Full v4 dataset
    v4_all = gold + silver
    random.shuffle(v4_all)
    v4_n_val = max(10, int(len(v4_all) * 0.1))
    v4_val = v4_all[:v4_n_val]
    v4_train = v4_all[v4_n_val:]

    # Save datasets
    def save_split(train, val, dir_name):
        out_dir = PROJECT_ROOT / "data" / "sft" / "final_v4" / dir_name
        save_jsonl(train, out_dir / "train.jsonl")
        save_jsonl(val, out_dir / "validation.jsonl")

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

        with open(out_dir / "train_chatml.jsonl", "w") as f:
            for s in train:
                f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")
        with open(out_dir / "validation_chatml.jsonl", "w") as f:
            for s in val:
                f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")

    save_split(gold_train, gold_val, "gold")
    save_split(silver_train, silver_val, "silver")
    save_split(v4_train, v4_val, "full")

    # Build subsets for full v4
    subsets_dir = PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "subsets"
    for n in [100, 500, 1000]:
        subset = v4_train[:min(n, len(v4_train))]
        save_jsonl(subset, subsets_dir / f"train_{n}.jsonl")
    save_jsonl(v4_val[:min(100, len(v4_val))], subsets_dir / "validation_100.jsonl")

    # Statistics
    stats = {
        "gold": {"train": len(gold_train), "val": len(gold_val), "total": len(gold_all)},
        "silver": {"train": len(silver_train), "val": len(silver_val), "total": len(silver_all)},
        "v4_full": {"train": len(v4_train), "val": len(v4_val), "total": len(v4_all)},
        "task_distribution_gold": dict(Counter(s.get("task_type", "unknown") for s in gold)),
        "task_distribution_v4": dict(Counter(s.get("task_type", "unknown") for s in v4_all)),
        "source_distribution_v4": dict(Counter(s.get("source_type", "unknown") for s in v4_all)),
    }

    # Save statistics
    with open(PROJECT_ROOT / "data" / "sft" / "final_v4" / "sft_v4_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log(f"  Gold: train={len(gold_train)}, val={len(gold_val)}")
    log(f"  Silver: train={len(silver_train)}, val={len(silver_val)}")
    log(f"  V4 full: train={len(v4_train)}, val={len(v4_val)}")

    return stats


def _audit_v4_sample(gold: list, silver: list, max_workers: int, log) -> dict:
    """Audit a sample of v4 gold and silver."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    client = UnifiedModelClient(model_name="doubao-seed-2.0-pro")

    # Sample gold
    random.seed(42)
    gold_sample = random.sample(gold, min(100, len(gold)))
    silver_sample = random.sample(silver, min(100, len(silver)))

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

    # Audit gold
    gold_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(audit_one, s): s for s in gold_sample}
        for future in as_completed(futures):
            gold_results.append(future.result())
            if len(gold_results) % 20 == 0:
                log(f"    Gold audit: {len(gold_results)}/{len(gold_sample)}")

    # Audit silver
    silver_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(audit_one, s): s for s in silver_sample}
        for future in as_completed(futures):
            silver_results.append(future.result())
            if len(silver_results) % 20 == 0:
                log(f"    Silver audit: {len(silver_results)}/{len(silver_sample)}")

    # Analyze
    def analyze(results):
        valid = [r for r in results if not r.get("error")]
        scores = {}
        for dim in ["instruction_clarity", "output_correctness", "evidence_support", "domain_relevance", "training_value"]:
            vals = [r.get(dim, 0) for r in valid if isinstance(r.get(dim), (int, float))]
            scores[dim] = round(sum(vals) / max(len(vals), 1), 3) if vals else 0
        status = Counter(r.get("overall_status", "unknown") for r in valid)
        return {"scores": scores, "status": dict(status), "total": len(valid), "keep_rate": status.get("keep", 0) / max(len(valid), 1)}

    gold_analysis = analyze(gold_results)
    silver_analysis = analyze(silver_results)

    log(f"    Gold: {gold_analysis['scores']}, keep_rate={gold_analysis['keep_rate']:.1%}")
    log(f"    Silver: {silver_analysis['scores']}, keep_rate={silver_analysis['keep_rate']:.1%}")

    # Save results
    save_jsonl(gold_results, PROJECT_ROOT / "data" / "audit" / "phase_7_11" / "sft_gold_audit_results.jsonl")
    save_jsonl(silver_results, PROJECT_ROOT / "data" / "audit" / "phase_7_11" / "sft_v4_audit_results.jsonl")

    return {
        "gold": gold_analysis,
        "silver": silver_analysis,
        "comparison": {
            "evidence_support_improvement": silver_analysis["scores"].get("evidence_support", 0) - 0.499,
            "output_correctness_improvement": silver_analysis["scores"].get("output_correctness", 0) - 0.640,
        },
    }


def _generate_paper_artifacts(evidence_bank, task_pools, gold, silver, v4_stats, audit_results, report_dir, log):
    """Generate paper-ready artifacts."""
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Evidence bank statistics
    with open(tables_dir / "evidence_bank_statistics.csv", "w") as f:
        f.write("Metric,Value\n")
        f.write(f"Total evidence units,{evidence_bank['summary']['total_units']}\n")
        f.write(f"Rejected units,{evidence_bank['summary']['total_rejected']}\n")
        f.write(f"Avg length (chars),{evidence_bank['summary']['avg_length']}\n")
        f.write(f"Avg evidence density,{evidence_bank['summary']['avg_evidence_density']}\n")

    # V2/V3/V4 comparison
    with open(tables_dir / "sft_v2_v3_v4_comparison.csv", "w") as f:
        f.write("Metric,V2,V3,V4_Gold,V4_Full\n")
        f.write(f"Total samples,2331,2270,{v4_stats['gold']['total']},{v4_stats['v4_full']['total']}\n")
        f.write(f"Train samples,2098,2043,{v4_stats['gold']['train']},{v4_stats['v4_full']['train']}\n")
        f.write(f"Validation samples,233,227,{v4_stats['gold']['val']},{v4_stats['v4_full']['val']}\n")
        f.write(f"Evidence support,0.499,0.602,{audit_results['gold']['scores'].get('evidence_support', 'N/A')},{audit_results['silver']['scores'].get('evidence_support', 'N/A')}\n")
        f.write(f"Output correctness,0.640,0.599,{audit_results['gold']['scores'].get('output_correctness', 'N/A')},{audit_results['silver']['scores'].get('output_correctness', 'N/A')}\n")

    # Task distribution
    with open(tables_dir / "sft_gold_silver_statistics.csv", "w") as f:
        f.write("TaskType,Gold,Silver,Full_V4\n")
        all_tasks = set()
        for d in [v4_stats.get("task_distribution_gold", {}), v4_stats.get("task_distribution_v4", {})]:
            all_tasks.update(d.keys())
        for task in sorted(all_tasks):
            g = v4_stats.get("task_distribution_gold", {}).get(task, 0)
            s = v4_stats.get("task_distribution_v4", {}).get(task, 0)
            f.write(f"{task},{g},{s-g},{s}\n")

    # LaTeX tables
    latex = "% SFT Evidence-First Reconstruction\n"
    latex += "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{SFT Dataset Versions Comparison}\n"
    latex += "\\begin{tabular}{lcccc}\n\\hline\n"
    latex += "Metric & V2 & V3 & V4 Gold & V4 Full \\\\\n\\hline\n"
    latex += f"Total & 2,331 & 2,270 & {v4_stats['gold']['total']} & {v4_stats['v4_full']['total']} \\\\\n"
    latex += f"Train & 2,098 & 2,043 & {v4_stats['gold']['train']} & {v4_stats['v4_full']['train']} \\\\\n"
    latex += f"Evidence Support & 0.499 & 0.602 & {audit_results['gold']['scores'].get('evidence_support', 'N/A')} & {audit_results['silver']['scores'].get('evidence_support', 'N/A')} \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(PROJECT_ROOT / "reports" / "paper_ready" / "sft_v2_v3_v4_comparison.tex", "w") as f:
        f.write(latex)

    # Save report
    md = "# Phase 7.11: Evidence-First SFT Reconstruction\n\n"
    md += "## Results\n\n"
    md += f"- Evidence bank: {evidence_bank['summary']['total_units']} units\n"
    md += f"- Gold: {v4_stats['gold']['total']} samples\n"
    md += f"- Silver: {v4_stats['silver']['total']} samples\n"
    md += f"- V4 Full: {v4_stats['v4_full']['total']} samples\n\n"
    md += "## Audit Results\n\n"
    md += f"- Gold evidence support: {audit_results['gold']['scores'].get('evidence_support', 'N/A')}\n"
    md += f"- Gold output correctness: {audit_results['gold']['scores'].get('output_correctness', 'N/A')}\n"
    md += f"- Silver evidence support: {audit_results['silver']['scores'].get('evidence_support', 'N/A')}\n"
    md += f"- Silver output correctness: {audit_results['silver']['scores'].get('output_correctness', 'N/A')}\n"
    md += f"- Evidence support improvement: +{audit_results['comparison']['evidence_support_improvement']:.3f}\n"

    with open(report_dir / "PHASE_7_11_REPORT.md", "w") as f:
        f.write(md)

    log("  Paper artifacts generated")


def _prepare_human_audit_sheets(gold: list, silver: list, log):
    """Prepare human audit annotation sheets."""
    random.seed(42)

    # Sample for human audit
    gold_sample = random.sample(gold, min(100, len(gold)))
    silver_sample = random.sample(silver, min(100, len(silver)))
    all_sample = gold_sample + silver_sample

    # SFT annotation sheet
    with open(PROJECT_ROOT / "reports" / "paper_ready" / "human_audit_annotation_sheet_sft_v4.csv", "w") as f:
        f.write("sample_id,source_type,task_type,instruction,output,evidence_text,")
        f.write("instruction_clarity,output_correctness,evidence_support,domain_relevance,")
        f.write("training_value,overall_status,major_issues,notes\n")
        for s in all_sample:
            inst = s.get("instruction", "").replace('"', '""')[:200]
            out = s.get("output", "").replace('"', '""')[:200]
            ev = " ".join(str(e) for e in s.get("evidence", [])[:2]).replace('"', '""')[:200]
            f.write(f'"{s.get("sample_id", "")}","{s.get("source_type", "")}","{s.get("task_type", "")}",')
            f.write(f'"{inst}","{out}","{ev}",')
            f.write(f',,,,,"",\n')

    # Instructions
    md = "# Human Audit Annotation Instructions (SFT v4)\n\n"
    md += "## Purpose\nThis is for future manual review by domain experts.\n\n"
    md += "## Sample Size\n200 samples (100 gold + 100 silver)\n\n"
    md += "## Scoring (0.0-1.0)\n"
    md += "- instruction_clarity: Is the instruction clear?\n"
    md += "- output_correctness: Is the output correct?\n"
    md += "- evidence_support: Is the output supported by evidence?\n"
    md += "- domain_relevance: Is it relevant to carbon fiber?\n"
    md += "- training_value: Is it useful for training?\n\n"
    md += "## Overall Status\n- keep: Good quality\n- revise: Needs improvement\n- drop: Should be removed\n"

    with open(PROJECT_ROOT / "reports" / "paper_ready" / "human_audit_annotation_instructions_v4.md", "w") as f:
        f.write(md)

    log(f"  Human audit sheets: {len(all_sample)} samples")


def _validate_phase711(report_dir: Path, log):
    """Validate Phase 7.11 outputs."""
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

    check("Evidence bank exists", (PROJECT_ROOT / "data" / "sft" / "evidence_bank" / "evidence_bank.jsonl").exists())
    check("Gold candidates exist", (PROJECT_ROOT / "data" / "sft" / "evidence_first" / "sft_gold_candidates.jsonl").exists())
    check("Silver candidates exist", (PROJECT_ROOT / "data" / "sft" / "evidence_first" / "sft_silver_candidates.jsonl").exists())
    check("SFT v4 gold train exists", (PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train.jsonl").exists())
    check("SFT v4 full train exists", (PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "train.jsonl").exists())
    check("V4 audit summary exists", (report_dir / "sft_v4_audit_summary.json").exists())
    check("Paper tables exist", (report_dir / "tables" / "sft_v2_v3_v4_comparison.csv").exists())
    check("Human audit sheets exist", (PROJECT_ROOT / "reports" / "paper_ready" / "human_audit_annotation_sheet_sft_v4.csv").exists())
    check("Phase 7.11 report exists", (report_dir / "PHASE_7_11_REPORT.md").exists())

    # Check v4 counts
    if (PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "train.jsonl").exists():
        with open(PROJECT_ROOT / "data" / "sft" / "final_v4" / "full" / "train.jsonl") as f:
            count = sum(1 for _ in f)
        check("V4 full has samples", count > 0, f"{count} samples")

    for c in checks:
        log(f"  {c}")

    save_json({"passed": passed, "failed": failed, "checks": checks}, report_dir / "validation_phase_7_11.json")


if __name__ == "__main__":
    main()
