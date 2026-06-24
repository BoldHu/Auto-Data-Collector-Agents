"""Phase 8.0: Qwen-VL dry-run validation.

Usage:
    python scripts/run_phase_8_0_qwen_vl_dryrun.py \
        --skip_download \
        --skip_7b
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Phase 8.0 Qwen-VL dry-run")
    parser.add_argument("--skip_download", action="store_true", help="Skip model download")
    parser.add_argument("--skip_7b", action="store_true", help="Skip 7B model tests")
    parser.add_argument("--max_pixels", type=int, default=1003520, help="Max pixels for image processing")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_0_qwen_vl_dryrun"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_0_qwen_vl"

    log_file = report_dir / "progress_phase_8_0.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 8.0: Qwen-VL Dry-Run ===")

    # Step 1: Environment check
    log("Step 1: Environment check...")
    env_check = _check_environment(log)
    save_json(env_check, report_dir / "environment_check.json")

    # Step 2: Model download
    if not args.skip_download:
        log("Step 2: Checking model availability...")
        download_report = _check_model_availability(log)
        save_json(download_report, report_dir / "model_download_report.json")
    else:
        log("Step 2: Skipping download")

    # Step 3: Load test
    log("Step 3: Model load test...")
    load_results = _test_model_loading(args, log)
    save_json(load_results, report_dir / "qwen_vl_load_test_results.json")

    # Step 4: Build test samples
    log("Step 4: Building test samples...")
    test_samples = _build_test_samples(log)
    save_json(test_samples["stats"], report_dir / "qwen_vl_test_sample_stats.json")

    # Step 5: Smoke tests
    log("Step 5: Running smoke tests...")
    smoke_results = _run_smoke_tests(load_results, test_samples["samples"], args, log)
    save_json(smoke_results, report_dir / "qwen_vl_smoke_test_report.json")

    # Step 6: SFT v4 compatibility
    log("Step 6: SFT v4 compatibility check...")
    compat_results = _check_sft_compatibility(log)
    save_json(compat_results, report_dir / "sft_v4_qwen_compatibility.json")

    # Step 7: LoRA dry-run
    log("Step 7: LoRA injection dry-run...")
    lora_results = _lora_dryrun(load_results, args, log)
    save_json(lora_results, report_dir / "qwen_vl_lora_dryrun_report.json")

    # Step 8: Cloud migration package
    log("Step 8: Cloud migration package...")
    _create_cloud_package(log)

    # Step 9: Phase 8.1 training plan
    log("Step 9: Phase 8.1 training plan...")
    _create_training_plan(load_results, log)

    # Step 10: Validation
    log("Step 10: Validation...")
    _validate_phase80(report_dir, load_results, log)

    log("=== Phase 8.0 Complete ===")


def _check_environment(log) -> dict:
    """Check environment dependencies."""
    import platform

    checks = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # CUDA
    try:
        import torch
        checks["torch_version"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            checks["cuda_version"] = torch.version.cuda
    except ImportError:
        checks["torch_version"] = "MISSING"

    # Transformers
    try:
        import transformers
        checks["transformers_version"] = transformers.__version__
    except ImportError:
        checks["transformers_version"] = "MISSING"

    # Other packages
    packages = {
        "accelerate": "accelerate",
        "peft": "peft",
        "bitsandbytes": "bitsandbytes",
        "qwen_vl_utils": "qwen_vl_utils",
        "PIL": "pillow",
        "torchvision": "torchvision",
        "huggingface_hub": "huggingface_hub",
        "safetensors": "safetensors",
    }

    for import_name, display_name in packages.items():
        try:
            mod = __import__(import_name)
            checks[display_name] = getattr(mod, "__version__", "OK")
        except ImportError:
            checks[display_name] = "MISSING"

    # Flash attention
    try:
        import flash_attn
        checks["flash_attn"] = flash_attn.__version__
    except ImportError:
        checks["flash_attn"] = "MISSING"

    # Disk space
    import shutil
    total, used, free = shutil.disk_usage(str(PROJECT_ROOT))
    checks["disk_free_gb"] = round(free / 1e9, 1)

    log(f"  Python: {checks.get('python_version')}")
    log(f"  GPU: {checks.get('gpu_name', 'N/A')} ({checks.get('gpu_memory_gb', 'N/A')}GB)")
    log(f"  PyTorch: {checks.get('torch_version')}")
    log(f"  Transformers: {checks.get('transformers_version')}")
    log(f"  Missing: {[k for k, v in checks.items() if v == 'MISSING']}")

    return checks


def _check_model_availability(log) -> dict:
    """Check if Qwen models are available locally."""
    models_dir = PROJECT_ROOT / "models" / "qwen"

    report = {
        "models_dir": str(models_dir),
        "models": {},
    }

    model_names = ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"]
    for name in model_names:
        model_path = models_dir / name
        if model_path.exists():
            # Check files
            files = list(model_path.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            report["models"][name] = {
                "exists": True,
                "path": str(model_path),
                "files": len(files),
                "size_gb": round(total_size / 1e9, 2),
            }
            log(f"  {name}: found ({len(files)} files, {total_size/1e9:.2f}GB)")
        else:
            report["models"][name] = {"exists": False, "path": str(model_path)}
            log(f"  {name}: not found at {model_path}")

    return report


def _test_model_loading(args, log) -> dict:
    """Test loading Qwen models."""
    results = {}

    # Test 3B
    model_path = PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct"
    if model_path.exists():
        log("  Testing Qwen2.5-VL-3B-Instruct...")
        results["3b"] = _load_single_model(str(model_path), "3b", log)
    else:
        log("  Qwen2.5-VL-3B-Instruct not found, skipping")
        results["3b"] = {"status": "not_found"}

    # Test 7B
    if not args.skip_7b:
        model_path = PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-7B-Instruct"
        if model_path.exists():
            log("  Testing Qwen2.5-VL-7B-Instruct...")
            results["7b"] = _load_single_model(str(model_path), "7b", log)
        else:
            log("  Qwen2.5-VL-7B-Instruct not found, skipping")
            results["7b"] = {"status": "not_found"}
    else:
        results["7b"] = {"status": "skipped"}

    return results


def _load_single_model(model_path: str, size: str, log) -> dict:
    """Load a single model and measure memory."""
    import torch

    result = {
        "model_path": model_path,
        "size": size,
        "status": "unknown",
        "peak_memory_gb": 0,
        "load_time_seconds": 0,
    }

    try:
        start = time.time()
        torch.cuda.reset_peak_memory_stats()

        from transformers import AutoProcessor, AutoModelForCausalLM

        # Try loading processor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        result["processor_loaded"] = True

        # Try loading model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        load_time = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        result["status"] = "success"
        result["load_time_seconds"] = round(load_time, 2)
        result["peak_memory_gb"] = round(peak_mem, 2)
        result["model_dtype"] = str(model.dtype)
        result["device_map"] = str(model.hf_device_map) if hasattr(model, "hf_device_map") else "N/A"

        log(f"    Status: success, Memory: {peak_mem:.2f}GB, Time: {load_time:.1f}s")

        # Cleanup
        del model
        del processor
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log(f"    Status: error - {str(e)[:100]}")

    return result


def _build_test_samples(log) -> dict:
    """Build test samples for smoke testing."""
    import random
    random.seed(42)

    samples = []

    # Text-only samples from CFBench
    benchmark_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            items = [json.loads(l) for l in f if l.strip()]

        # Sample 5 text items
        text_items = [i for i in items if i.get("source_type") == "text"]
        for item in random.sample(text_items, min(5, len(text_items))):
            samples.append({
                "sample_id": item.get("benchmark_id", ""),
                "sample_type": "cfbench_text",
                "modality": "text",
                "image_path": None,
                "prompt": item.get("question", ""),
                "gold_answer": str(item.get("answer", "")),
                "evidence": item.get("evidence", []),
            })

        # Sample 5 multimodal items
        mm_items = [i for i in items if i.get("source_type") == "multimodal"]
        for item in random.sample(mm_items, min(5, len(mm_items))):
            image_refs = item.get("image_refs", [])
            source_refs = item.get("source_refs", [])
            image_path = source_refs[0] if source_refs else None
            samples.append({
                "sample_id": item.get("benchmark_id", ""),
                "sample_type": "cfbench_mm",
                "modality": "multimodal",
                "image_path": image_path,
                "prompt": item.get("question", ""),
                "gold_answer": str(item.get("answer", "")),
                "evidence": item.get("evidence", []),
            })

    # SFT v4 samples
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
                "evidence": item.get("evidence", []),
            })

    # Carbon fiber images
    imgs_dir = PROJECT_ROOT / "imgs_raw_data" / "carbon_fiber_mm"
    if imgs_dir.exists():
        img_folders = list(imgs_dir.iterdir())[:3]
        for folder in img_folders:
            imgs = list(folder.glob("*.jpg"))[:1]
            for img in imgs:
                samples.append({
                    "sample_id": f"img_{img.stem}",
                    "sample_type": "carbon_fiber_image",
                    "modality": "multimodal",
                    "image_path": str(img),
                    "prompt": "请描述这张碳纤维相关图像的内容。",
                    "gold_answer": None,
                    "evidence": [],
                })

    stats = {
        "total_samples": len(samples),
        "by_type": {},
        "by_modality": {},
    }
    for s in samples:
        stats["by_type"][s["sample_type"]] = stats["by_type"].get(s["sample_type"], 0) + 1
        stats["by_modality"][s["modality"]] = stats["by_modality"].get(s["modality"], 0) + 1

    # Save samples
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_0_qwen_vl"
    with open(eval_dir / "qwen_vl_test_samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    log(f"  Built {len(samples)} test samples")
    return {"samples": samples, "stats": stats}


def _run_smoke_tests(load_results: dict, test_samples: list, args, log) -> dict:
    """Run smoke tests on loaded models."""
    import torch

    results = {"tests": []}

    # Check if any model loaded successfully
    for size in ["3b", "7b"]:
        if load_results.get(size, {}).get("status") != "success":
            log(f"  Skipping {size} smoke test (model not loaded)")
            continue

        model_path = load_results[size]["model_path"]
        log(f"  Running smoke tests on {size}...")

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Test 1: Text-only generation
            test_result = _test_text_generation(model, processor, size, log)
            results["tests"].append(test_result)

            # Test 2: Image processing (if available)
            image_samples = [s for s in test_samples if s.get("image_path") and os.path.exists(s.get("image_path", ""))]
            if image_samples:
                test_result = _test_image_processing(model, processor, image_samples[0], size, log)
                results["tests"].append(test_result)

            # Cleanup
            del model
            del processor
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"    Error: {str(e)[:100]}")
            results["tests"].append({"test": f"{size}_smoke", "status": "error", "error": str(e)[:200]})

    return results


def _test_text_generation(model, processor, size: str, log) -> dict:
    """Test text-only generation."""
    import torch

    try:
        messages = [{"role": "user", "content": "什么是碳纤维？请用中文简要回答。"}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log(f"    Text generation: {response[:80]}...")

        return {
            "test": f"{size}_text_generation",
            "status": "success",
            "response_length": len(response),
            "response_preview": response[:200],
        }
    except Exception as e:
        return {"test": f"{size}_text_generation", "status": "error", "error": str(e)[:200]}


def _test_image_processing(model, processor, sample: dict, size: str, log) -> dict:
    """Test image processing."""
    import torch

    try:
        image_path = sample["image_path"]
        prompt = sample["prompt"]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        response = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log(f"    Image processing: {response[:80]}...")

        return {
            "test": f"{size}_image_processing",
            "status": "success",
            "image_path": image_path,
            "response_length": len(response),
            "response_preview": response[:200],
        }
    except Exception as e:
        return {"test": f"{size}_image_processing", "status": "error", "error": str(e)[:200]}


def _check_sft_compatibility(log) -> dict:
    """Check SFT v4 compatibility with Qwen-VL."""
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

    # Check ChatML format
    valid = 0
    has_image = 0
    lengths = []

    for s in samples:
        messages = s.get("messages", [])
        if messages and any(m.get("role") == "user" for m in messages) and any(m.get("role") == "assistant" for m in messages):
            valid += 1
        if any("image" in str(m.get("content", "")) for m in messages):
            has_image += 1
        total_len = sum(len(m.get("content", "")) for m in messages)
        lengths.append(total_len)

    result = {
        "total_checked": len(samples),
        "valid_chatml": valid,
        "has_image": has_image,
        "avg_length": round(sum(lengths) / max(len(lengths), 1)),
        "max_length": max(lengths) if lengths else 0,
        "status": "compatible" if valid > 0 else "incompatible",
        "notes": "Text-only SFT is compatible with Qwen-VL language component training",
    }

    log(f"  SFT compatibility: {valid}/{len(samples)} valid ChatML, avg length={result['avg_length']}")
    return result


def _lora_dryrun(load_results: dict, args, log) -> dict:
    """LoRA injection dry-run."""
    import torch

    # Use 3B if available
    model_info = load_results.get("3b", {})
    if model_info.get("status") != "success":
        log("  Skipping LoRA dry-run (no model loaded)")
        return {"status": "skipped", "reason": "no_model_loaded"}

    model_path = model_info["model_path"]

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        log("  Loading model for LoRA dry-run...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Find target modules
        target_modules = []
        for name, module in model.named_modules():
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
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

        # Cleanup
        del model
        del processor
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        log(f"  LoRA dry-run error: {str(e)[:100]}")
        return {"status": "error", "error": str(e)[:200]}


def _create_cloud_package(log):
    """Create cloud migration package."""
    manifest = {
        "code_paths": [
            "src/autodata/finetuning/",
            "scripts/run_phase_8_0_*.py",
            "configs/finetuning/",
        ],
        "data_paths": [
            "data/sft/final_v4/",
            "data/benchmark/carbon_fiber_benchmark_dev.jsonl",
            "data/benchmark/carbon_fiber_benchmark_test.jsonl",
            "data/benchmark/subsets/",
        ],
        "model_download_commands": [
            "python scripts/download_phase_8_0_qwen_vl_models.py --models Qwen/Qwen2.5-VL-7B-Instruct --output_dir models/qwen",
        ],
        "training_commands": [
            "python scripts/run_phase_8_0_qwen_vl_lora_dryrun.py --model_path models/qwen/Qwen2.5-VL-7B-Instruct --run_training false",
        ],
        "exclude": [
            "LLM_API/",
            "*.key",
            "*.env",
            "__pycache__/",
            ".cache/",
            "models/",
            "imgs_raw_data/",
        ],
    }

    save_json(manifest, PROJECT_ROOT / "data" / "reports" / "phase_8_0_qwen_vl_dryrun" / "cloud_migration_manifest.json")

    # Create .cloudignore
    cloudignore = """LLM_API/
*.key
*.env
__pycache__/
.cache/
models/
imgs_raw_data/
data/raw/
outputs/*/checkpoint-*
*.pyc
"""
    with open(PROJECT_ROOT / ".cloudignore", "w") as f:
        f.write(cloudignore)

    log("  Cloud migration package created")


def _create_training_plan(load_results: dict, log):
    """Create Phase 8.1 training plan."""
    plan = """# Phase 8.1 Qwen-VL Training Plan

## Recommended Base Model

Based on Phase 8.0 dry-run:
- **Primary**: Qwen2.5-VL-3B-Instruct (validated locally)
- **Secondary**: Qwen2.5-VL-7B-Instruct (cloud training)

## Training Sequence

1. **Base zero-shot evaluation**
   - CFBench-Text (50 items)
   - CFBench-AgentTask (30 items)
   - CFBench-Core (50 items)
   - CFBench-MM (30 items)

2. **LoRA on sft_gold train_100**
   - Quick validation of training pipeline
   - Expected: ~30 minutes on cloud

3. **LoRA on sft_gold full**
   - High-quality evidence-first training
   - Expected: ~2 hours on cloud

4. **LoRA on sft_v4 full**
   - Full dataset training
   - Expected: ~4 hours on cloud

5. **Optional QLoRA if memory constrained**
   - 4-bit quantization for 7B

## Evaluation

- CFBench-Text
- CFBench-AgentTask
- CFBench-Core text-only
- CFBench-MM (if multimodal inference stable)

## Expected Claims

- Evidence-first SFT improves domain answer quality
- Gold vs full v4 comparison tests quality-vs-quantity
- Multimodal model can serve as base for future image-text SFT

## Risks

- Text-only SFT may not improve visual reasoning
- Small SFT may overfit
- Benchmark coverage is limited
- Local 4090 may be insufficient for full 7B training
"""

    with open(PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_1_qwen_training_plan.md", "w") as f:
        f.write(plan)

    log("  Training plan created")


def _validate_phase80(report_dir: Path, load_results: dict, log):
    """Validate Phase 8.0 outputs."""
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

    check("Environment check exists", (report_dir / "environment_check.json").exists())
    check("Load test results exist", (report_dir / "qwen_vl_load_test_results.json").exists())
    check("Test samples exist", (PROJECT_ROOT / "data" / "evaluation" / "phase_8_0_qwen_vl" / "qwen_vl_test_samples.jsonl").exists())
    check("Smoke test report exists", (report_dir / "qwen_vl_smoke_test_report.json").exists())
    check("SFT compatibility exists", (report_dir / "sft_v4_qwen_compatibility.json").exists())
    check("LoRA dry-run exists", (report_dir / "qwen_vl_lora_dryrun_report.json").exists())
    check("Cloud migration exists", (report_dir / "cloud_migration_manifest.json").exists())
    check("Training plan exists", (PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_1_qwen_training_plan.md").exists())

    # Check if any model loaded
    any_loaded = any(load_results.get(s, {}).get("status") == "success" for s in ["3b", "7b"])
    check("At least one model loaded", any_loaded)

    for c in checks:
        log(f"  {c}")

    save_json({"passed": passed, "failed": failed, "checks": checks}, report_dir / "validation_phase_8_0.json")


if __name__ == "__main__":
    main()
