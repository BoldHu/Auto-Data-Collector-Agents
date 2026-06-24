"""Doubao model registry for Phase 6.5.

Builds model registry from doubao_llm.md documentation.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent

# Doubao Token Plan models (from doubao_llm.md)
DOUBAO_MODELS = [
    {
        "model_name": "doubao-seed-2.0-pro",
        "display_name": "Doubao Seed 2.0 Pro",
        "model_id": "doubao-seed-2.0-pro",
        "provider": "doubao",
        "capability": "multimodal",
        "supports_image": True,
        "supports_thinking": False,
        "supports_json_output": True,
        "context_length": 200000,
        "max_output": 128000,
        "recommended_use": "multimodal_baseline",
        "notes": "Flagship model, complex reasoning + multimodal",
    },
    {
        "model_name": "doubao-seed-2.0-code",
        "display_name": "Doubao Seed 2.0 Code",
        "model_id": "doubao-seed-2.0-code",
        "provider": "doubao",
        "capability": "multimodal",
        "supports_image": True,
        "supports_thinking": False,
        "supports_json_output": True,
        "context_length": 200000,
        "max_output": 128000,
        "recommended_use": "multimodal_baseline",
        "notes": "Code-focused with multimodal vision",
    },
    {
        "model_name": "doubao-seed-2.0-lite",
        "display_name": "Doubao Seed 2.0 Lite",
        "model_id": "doubao-seed-2.0-lite",
        "provider": "doubao",
        "capability": "multimodal",
        "supports_image": True,
        "supports_thinking": False,
        "supports_json_output": True,
        "context_length": 200000,
        "max_output": 128000,
        "recommended_use": "baseline",
        "notes": "Fast + balanced, good for general baseline",
    },
    {
        "model_name": "doubao-seed-code",
        "display_name": "Doubao Seed Code",
        "model_id": "doubao-seed-code",
        "provider": "doubao",
        "capability": "multimodal",
        "supports_image": True,
        "supports_thinking": False,
        "supports_json_output": True,
        "context_length": 200000,
        "max_output": 128000,
        "recommended_use": "baseline",
        "notes": "Code-focused model with vision",
    },
    {
        "model_name": "deepseek-v3.2-doubao",
        "display_name": "DeepSeek V3.2 (Doubao)",
        "model_id": "deepseek-v3.2",
        "provider": "doubao",
        "capability": "text",
        "supports_image": False,
        "supports_thinking": False,
        "supports_json_output": True,
        "context_length": 128000,
        "max_output": 8192,
        "recommended_use": "baseline",
        "notes": "Balanced reasoning model",
    },
    {
        "model_name": "deepseek-v4-flash",
        "display_name": "DeepSeek V4 Flash",
        "model_id": "deepseek-v4-flash",
        "provider": "doubao",
        "capability": "reasoning",
        "supports_image": False,
        "supports_thinking": True,
        "supports_json_output": True,
        "context_length": 1024000,
        "max_output": 384000,
        "recommended_use": "reasoning_baseline",
        "notes": "Fast reasoning with thinking enabled by default",
    },
    {
        "model_name": "deepseek-v4-pro",
        "display_name": "DeepSeek V4 Pro",
        "model_id": "deepseek-v4-pro",
        "provider": "doubao",
        "capability": "reasoning",
        "supports_image": False,
        "supports_thinking": True,
        "supports_json_output": True,
        "context_length": 1024000,
        "max_output": 384000,
        "recommended_use": "reasoning_baseline",
        "notes": "Strong reasoning + agent capabilities",
    },
    {
        "model_name": "glm-5.1",
        "display_name": "GLM 5.1",
        "model_id": "glm-5.1",
        "provider": "doubao",
        "capability": "reasoning",
        "supports_image": False,
        "supports_thinking": True,
        "supports_json_output": True,
        "context_length": 200000,
        "max_output": 128000,
        "recommended_use": "reasoning_baseline",
        "notes": "Flagship GLM model, complex reasoning",
    },
    {
        "model_name": "kimi-k2.6",
        "display_name": "Kimi K2.6",
        "model_id": "kimi-k2.6",
        "provider": "doubao",
        "capability": "multimodal",
        "supports_image": True,
        "supports_thinking": True,
        "supports_json_output": True,
        "context_length": 256000,
        "max_output": 32000,
        "recommended_use": "reasoning_baseline",
        "notes": "Strong thinking + multimodal vision",
    },
    {
        "model_name": "minimax-m2.7",
        "display_name": "MiniMax M2.7",
        "model_id": "minimax-m2.7",
        "provider": "doubao",
        "capability": "text",
        "supports_image": False,
        "supports_thinking": False,
        "supports_json_output": True,
        "context_length": 200000,
        "max_output": 128000,
        "recommended_use": "baseline",
        "notes": "Complex agent tasks, high token cost",
    },
]

# Selected models for evaluation (only 2 for speed)
SELECTED_MODELS = [
    "doubao-seed-2.0-pro",      # flagship multimodal (strongest)
    "deepseek-v4-flash",        # fast reasoning
]


def get_doubao_models() -> list[dict]:
    """Get all available Doubao models."""
    return DOUBAO_MODELS


def get_selected_models() -> list[dict]:
    """Get selected models for evaluation."""
    return [m for m in DOUBAO_MODELS if m["model_name"] in SELECTED_MODELS]


def get_vision_models() -> list[dict]:
    """Get models that support image input."""
    return [m for m in DOUBAO_MODELS if m["supports_image"]]


def get_reasoning_models() -> list[dict]:
    """Get models that support thinking/reasoning."""
    return [m for m in DOUBAO_MODELS if m["supports_thinking"]]


def save_model_registry(output_dir: Path = None) -> tuple[Path, Path]:
    """Save model registry as JSON and MD."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_5_doubao_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "doubao_model_registry.json"
    md_path = output_dir / "doubao_model_registry.md"

    # Sanitize: remove API key references
    sanitized = []
    for m in DOUBAO_MODELS:
        s = {k: v for k, v in m.items() if "key" not in k.lower()}
        sanitized.append(s)

    with open(json_path, "w") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# Doubao Token Plan Model Registry\n\n")
        f.write(f"Total models: {len(DOUBAO_MODELS)}\n\n")
        f.write("| Model | Capability | Image | Thinking | Recommended Use |\n")
        f.write("|-------|-----------|-------|----------|------------------|\n")
        for m in DOUBAO_MODELS:
            f.write(f"| {m['display_name']} | {m['capability']} | {m['supports_image']} | {m['supports_thinking']} | {m['recommended_use']} |\n")

    return json_path, md_path


def save_selected_models(output_dir: Path = None) -> tuple[Path, Path]:
    """Save selected models list."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_5_doubao_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "selected_models.json"
    md_path = output_dir / "selected_models.md"

    selected = get_selected_models()

    with open(json_path, "w") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# Selected Models for Phase 6.5 Evaluation\n\n")
        f.write(f"Total selected: {len(selected)}\n\n")
        for m in selected:
            f.write(f"## {m['display_name']}\n")
            f.write(f"- Model ID: `{m['model_id']}`\n")
            f.write(f"- Capability: {m['capability']}\n")
            f.write(f"- Image support: {m['supports_image']}\n")
            f.write(f"- Thinking: {m['supports_thinking']}\n")
            f.write(f"- Notes: {m['notes']}\n\n")

    return json_path, md_path
