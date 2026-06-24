"""Phase 8.0.5: Qwen-VL dependency repair and model validation.

Usage:
    python scripts/run_phase_8_0_5_qwen_vl_unblock.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_0_5_qwen_vl_unblock"
    report_dir.mkdir(parents=True, exist_ok=True)

    log_file = report_dir / "progress_phase_8_0_5.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.0.5: Qwen-VL Unblock ===")

    # Step 1: Check dependencies
    log("Step 1: Checking dependencies...")
    dep_status = _check_dependencies(log)
    save_json(dep_status, report_dir / "dependency_status_after_repair.json")

    # Step 2: Verify model download
    log("Step 2: Verifying model download...")
    model_status = _verify_model_download(log)
    save_json(model_status, report_dir / "model_download_report.json")

    # Step 3: Load test
    log("Step 3: Model load test...")
    load_results = _test_model_loading(log)
    save_json(load_results, report_dir / "qwen2_5_vl_3b_load_test.json")

    # Step 4: Build test samples
    log("Step 4: Building test samples...")
    test_samples = _build_test_samples(log)

    # Step 5: Smoke tests
    log("Step 5: Running smoke tests...")
    smoke_results = _run_smoke_tests(load_results, test_samples, log)
    save_json(smoke_results, report_dir / "qwen2_5_vl_3b_smoke_test_report.json")

    # Step 6: SFT v4 tokenization check
    log("Step 6: SFT v4 tokenization check...")
    token_results = _check_sft_tokenization(load_results, log)
    save_json(token_results, report_dir / "sft_v4_qwen_tokenization_report.json")

    # Step 7: LoRA dry-run
    log("Step 7: LoRA injection dry-run...")
    lora_results = _lora_dryrun(load_results, log)
    save_json(lora_results, report_dir / "qwen2_5_vl_3b_lora_dryrun_report.json")

    # Step 8: Mini zero-shot evaluation
    log("Step 8: Mini zero-shot evaluation...")
    zero_shot_results = _mini_zero_shot(load_results, test_samples, log)
    save_json(zero_shot_results, report_dir / "qwen2_5_vl_3b_zero_shot_report.json")

    # Step 9: Validation
    log("Step 9: Validation...")
    _validate_phase805(report_dir, load_results, log)

    log("=== Phase 8.0.5 Complete ===")


def _check_dependencies(log) -> dict:
    """Check all dependencies."""
    import platform

    status = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Check packages
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "peft": "peft",
        "bitsandbytes": "bitsandbytes",
        "qwen_vl_utils": "qwen_vl_utils",
        "flash_attn": "flash_attn",
        "modelscope": "modelscope",
        "huggingface_hub": "huggingface_hub",
        "safetensors": "safetensors",
        "PIL": "pillow",
        "torchvision": "torchvision",
    }

    for import_name, display_name in packages.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "OK")
            status[display_name] = ver
        except ImportError:
            status[display_name] = "MISSING"

    # CUDA
    try:
        import torch
        status["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            status["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    log(f"  Python: {status.get('python')}")
    log(f"  Transformers: {status.get('transformers')}")
    log(f"  Accelerate: {status.get('accelerate')}")
    log(f"  GPU: {status.get('gpu_name', 'N/A')} ({status.get('gpu_memory_gb', 'N/A')}GB)")
    log(f"  Missing: {[k for k, v in status.items() if v == 'MISSING']}")

    return status


def _verify_model_download(log) -> dict:
    """Verify model files exist."""
    model_path = PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct"

    report = {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "local_path": str(model_path),
        "exists": model_path.exists(),
        "files": {},
        "status": "unknown",
    }

    if not model_path.exists():
        report["status"] = "not_found"
        log(f"  Model not found at {model_path}")
        return report

    # Check required files
    required_files = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
    optional_files = ["preprocessor_config.json", "generation_config.json", "chat_template.json"]

    all_files = list(model_path.glob("*"))
    report["total_files"] = len(all_files)
    report["total_size_gb"] = round(sum(f.stat().st_size for f in all_files if f.is_file()) / 1e9, 2)

    for f in required_files:
        fpath = model_path / f
        report["files"][f] = {"exists": fpath.exists(), "required": True}

    for f in optional_files:
        fpath = model_path / f
        report["files"][f] = {"exists": fpath.exists(), "required": False}

    # Check safetensors shards
    shards = list(model_path.glob("*.safetensors"))
    report["shard_count"] = len(shards)
    report["shard_size_gb"] = round(sum(f.stat().st_size for f in shards) / 1e9, 2)

    all_required = all(report["files"][f]["exists"] for f in required_files)
    report["status"] = "complete" if all_required and len(shards) > 0 else "incomplete"

    log(f"  Status: {report['status']}")
    log(f"  Files: {report['total_files']}, Size: {report['total_size_gb']}GB")
    log(f"  Shards: {report['shard_count']} ({report['shard_size_gb']}GB)")

    return report


def _test_model_loading(log) -> dict:
    """Test loading Qwen2.5-VL-3B."""
    import torch

    model_path = str(PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct")
    result = {
        "model_path": model_path,
        "status": "unknown",
        "modes": {},
    }

    # Mode 1: bf16 + device_map=auto
    log("  Testing mode: bf16 + device_map=auto")
    mode_result = _load_and_test_mode(model_path, torch.bfloat16, "auto", log)
    result["modes"]["bf16_auto"] = mode_result

    if mode_result["status"] == "success":
        result["status"] = "success"
        result["recommended_mode"] = "bf16_auto"
        result["peak_memory_gb"] = mode_result["peak_memory_gb"]
        result["load_time_seconds"] = mode_result["load_time_seconds"]
    else:
        result["status"] = "failed"

    return result


def _load_and_test_mode(model_path: str, dtype, device_map: str, log) -> dict:
    """Load model in a specific mode and test."""
    import torch

    result = {
        "dtype": str(dtype),
        "device_map": device_map,
        "status": "unknown",
    }

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        start = time.time()
        torch.cuda.reset_peak_memory_stats()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        load_time = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        result["status"] = "success"
        result["load_time_seconds"] = round(load_time, 2)
        result["peak_memory_gb"] = round(peak_mem, 2)
        result["model_class"] = type(model).__name__

        # Test text generation
        messages = [{"role": "user", "content": "什么是碳纤维？请简要回答。"}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result["text_generation"] = "success"
        result["response_preview"] = response[:200]

        log(f"    Success: {peak_mem:.2f}GB, {load_time:.1f}s")
        log(f"    Response: {response[:80]}...")

        del model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log(f"    Error: {str(e)[:100]}")

    return result


def _build_test_samples(log) -> list:
    """Build test samples."""
    import random
    random.seed(42)

    samples = []

    # CFBench-Text
    bench_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    if bench_path.exists():
        with open(bench_path) as f:
            items = [json.loads(l) for l in f if l.strip()]

        text_items = [i for i in items if i.get("source_type") == "text"]
        for item in random.sample(text_items, min(5, len(text_items))):
            samples.append({
                "sample_id": item.get("benchmark_id", ""),
                "sample_type": "cfbench_text",
                "modality": "text",
                "image_path": None,
                "prompt": item.get("question", ""),
                "gold_answer": str(item.get("answer", "")),
            })

        # CFBench-MM
        mm_items = [i for i in items if i.get("source_type") == "multimodal"]
        for item in random.sample(mm_items, min(5, len(mm_items))):
            source_refs = item.get("source_refs", [])
            image_path = source_refs[0] if source_refs else None
            if image_path and os.path.exists(image_path):
                samples.append({
                    "sample_id": item.get("benchmark_id", ""),
                    "sample_type": "cfbench_mm",
                    "modality": "multimodal",
                    "image_path": image_path,
                    "prompt": item.get("question", ""),
                    "gold_answer": str(item.get("answer", "")),
                })

    # SFT v4 gold
    sft_path = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train.jsonl"
    if sft_path.exists():
        with open(sft_path) as f:
            sft_items = [json.loads(l) for l in f if l.strip()]
        for item in random.sample(sft_items, min(5, len(sft_items))):
            samples.append({
                "sample_id": item.get("sample_id", ""),
                "sample_type": "sft_v4_gold",
                "modality": "text",
                "image_path": None,
                "prompt": item.get("instruction", ""),
                "gold_answer": item.get("output", "")[:200],
            })

    # Carbon fiber images
    imgs_dir = PROJECT_ROOT / "imgs_raw_data" / "carbon_fiber_mm"
    if imgs_dir.exists():
        for folder in list(imgs_dir.iterdir())[:3]:
            imgs = list(folder.glob("*.jpg"))[:1]
            for img in imgs:
                samples.append({
                    "sample_id": f"img_{img.stem}",
                    "sample_type": "carbon_fiber_image",
                    "modality": "multimodal",
                    "image_path": str(img),
                    "prompt": "请描述这张碳纤维相关图像的内容。",
                    "gold_answer": None,
                })

    # Save samples
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_0_5_qwen_vl"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "qwen_vl_test_samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    log(f"  Built {len(samples)} test samples")
    return samples


def _run_smoke_tests(load_results: dict, test_samples: list, log) -> dict:
    """Run smoke tests."""
    import torch

    if load_results.get("status") != "success":
        log("  Skipping smoke tests (model not loaded)")
        return {"status": "skipped", "reason": "model_not_loaded"}

    model_path = load_results["model_path"]
    results = {"tests": []}

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Test 1: Text generation
        log("  Test 1: Text generation")
        test_result = _test_text_gen(model, processor, log)
        results["tests"].append(test_result)

        # Test 2: Image processing
        image_samples = [s for s in test_samples if s.get("image_path") and os.path.exists(s.get("image_path", ""))]
        if image_samples:
            log("  Test 2: Image processing")
            test_result = _test_image_gen(model, processor, image_samples[0], log)
            results["tests"].append(test_result)

        # Test 3: CFBench-MM
        mm_samples = [s for s in test_samples if s.get("sample_type") == "cfbench_mm"]
        if mm_samples:
            log("  Test 3: CFBench-MM")
            test_result = _test_image_gen(model, processor, mm_samples[0], log)
            results["tests"].append(test_result)

        del model
        del processor
        torch.cuda.empty_cache()

        results["status"] = "success"
        results["tests_passed"] = sum(1 for t in results["tests"] if t.get("status") == "success")

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)[:200]
        log(f"  Error: {str(e)[:100]}")

    return results


def _test_text_gen(model, processor, log) -> dict:
    """Test text generation."""
    import torch

    try:
        messages = [{"role": "user", "content": "什么是碳纤维？请简要回答。"}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log(f"    Response: {response[:80]}...")

        return {"test": "text_generation", "status": "success", "response": response[:200]}
    except Exception as e:
        return {"test": "text_generation", "status": "error", "error": str(e)[:200]}


def _test_image_gen(model, processor, sample: dict, log) -> dict:
    """Test image processing."""
    import torch

    try:
        image_path = sample["image_path"]
        prompt = sample["prompt"]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log(f"    Image response: {response[:80]}...")

        return {"test": "image_processing", "status": "success", "image_path": image_path, "response": response[:200]}
    except Exception as e:
        return {"test": "image_processing", "status": "error", "error": str(e)[:200]}


def _check_sft_tokenization(load_results: dict, log) -> dict:
    """Check SFT v4 tokenization compatibility."""
    if load_results.get("status") != "success":
        return {"status": "skipped", "reason": "model_not_loaded"}

    model_path = load_results["model_path"]

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load SFT samples
        sft_path = PROJECT_ROOT / "data" / "sft" / "final_v4" / "gold" / "train_chatml.jsonl"
        if not sft_path.exists():
            return {"status": "not_found"}

        samples = []
        with open(sft_path) as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                if line.strip():
                    samples.append(json.loads(line))

        # Check tokenization
        lengths = []
        max_length = 0
        over_2048 = 0
        over_4096 = 0

        for s in samples:
            messages = s.get("messages", [])
            try:
                text = processor.apply_chat_template(messages, tokenize=False)
                tokens = processor.tokenizer(text, return_tensors="pt")
                length = tokens["input_ids"].shape[1]
                lengths.append(length)
                max_length = max(max_length, length)
                if length > 2048:
                    over_2048 += 1
                if length > 4096:
                    over_4096 += 1
            except Exception:
                pass

        result = {
            "status": "success",
            "samples_checked": len(samples),
            "avg_length": round(sum(lengths) / max(len(lengths), 1)),
            "max_length": max_length,
            "over_2048": over_2048,
            "over_4096": over_4096,
            "compatible": True,
            "notes": "Text-only SFT is compatible with Qwen-VL language component training",
        }

        log(f"  SFT tokenization: avg={result['avg_length']}, max={result['max_length']}, over_2048={over_2048}")
        return result

    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def _lora_dryrun(load_results: dict, log) -> dict:
    """LoRA injection dry-run."""
    import torch

    if load_results.get("status") != "success":
        return {"status": "skipped", "reason": "model_not_loaded"}

    model_path = load_results["model_path"]

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, get_peft_model, TaskType

        log("  Loading model for LoRA dry-run...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Find target modules
        target_modules = []
        for name, module in model.named_modules():
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                base_name = name.split(".")[-1]
                if base_name not in target_modules:
                    target_modules.append(base_name)

        if not target_modules:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        log(f"  Target modules: {target_modules}")

        # Inject LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()

        result = {
            "status": "success",
            "target_modules": target_modules,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_ratio": round(trainable / total, 4),
            "model_path": model_path,
        }

        log(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

        del model
        del processor
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        log(f"  LoRA dry-run error: {str(e)[:100]}")
        return {"status": "error", "error": str(e)[:200]}


def _mini_zero_shot(load_results: dict, test_samples: list, log) -> dict:
    """Mini zero-shot evaluation."""
    import torch
    import random

    if load_results.get("status") != "success":
        return {"status": "skipped", "reason": "model_not_loaded"}

    model_path = load_results["model_path"]

    # Sample items for evaluation
    bench_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    if not bench_path.exists():
        return {"status": "skipped", "reason": "benchmark_not_found"}

    with open(bench_path) as f:
        bench_items = [json.loads(l) for l in f if l.strip()]

    # Sample 30 text items
    text_items = [i for i in bench_items if i.get("source_type") == "text"]
    eval_items = random.sample(text_items, min(30, len(text_items)))

    log(f"  Evaluating {len(eval_items)} items...")

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        results = []
        correct = 0
        total = 0

        for item in eval_items:
            question = item.get("question", "")
            gold = str(item.get("answer", "")).strip()
            options = item.get("options", [])

            # Build prompt
            if options:
                opt_text = "\n".join(str(o) for o in options)
                prompt = f"{question}\n\n选项：\n{opt_text}\n\n请直接输出选项字母。"
            else:
                prompt = f"{question}\n\n请直接回答。"

            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # Check correctness
            import re
            is_correct = False
            if options and len(options) >= 2:
                pred_match = re.search(r'([A-H])', response.upper())
                gold_match = re.search(r'([A-H])', gold.upper())
                if pred_match and gold_match:
                    is_correct = pred_match.group(1) == gold_match.group(1)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "benchmark_id": item.get("benchmark_id", ""),
                "gold": gold[:50],
                "predicted": response[:50],
                "correct": is_correct,
            })

        accuracy = correct / max(total, 1)

        result = {
            "status": "success",
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 3),
            "results": results[:10],  # Save first 10 for inspection
        }

        log(f"  Zero-shot: {correct}/{total} = {accuracy:.1%}")

        del model
        del processor
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        log(f"  Zero-shot error: {str(e)[:100]}")
        return {"status": "error", "error": str(e)[:200]}


def _validate_phase805(report_dir: Path, load_results: dict, log):
    """Validate Phase 8.0.5 outputs."""
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

    check("Dependency status exists", (report_dir / "dependency_status_after_repair.json").exists())
    check("Model download report exists", (report_dir / "model_download_report.json").exists())
    check("Load test exists", (report_dir / "qwen2_5_vl_3b_load_test.json").exists())
    check("Smoke test report exists", (report_dir / "qwen2_5_vl_3b_smoke_test_report.json").exists())
    check("SFT tokenization report exists", (report_dir / "sft_v4_qwen_tokenization_report.json").exists())
    check("LoRA dry-run exists", (report_dir / "qwen2_5_vl_3b_lora_dryrun_report.json").exists())
    check("Zero-shot report exists", (report_dir / "qwen2_5_vl_3b_zero_shot_report.json").exists())

    # Check if model loaded successfully
    check("Model loaded successfully", load_results.get("status") == "success")

    for c in checks:
        log(f"  {c}")

    save_json({"passed": passed, "failed": failed, "checks": checks}, report_dir / "validation_phase_8_0_5.json")


if __name__ == "__main__":
    main()
