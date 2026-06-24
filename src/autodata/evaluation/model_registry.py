"""Updated model registry for Phase 6.5.

Includes both Xiaomi and Doubao models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def load_model_registry_6_5() -> list[dict]:
    """Load all models for Phase 6.5 evaluation.

    Returns:
        List of model info dicts.
    """
    from src.autodata.evaluation.doubao_model_registry import get_selected_models
    from src.autodata.utils.llm_api_loader import get_llm_config

    models = []

    # Add Xiaomi mimo-v2.5-pro (disabled - API key invalid)
    models.append({
        "model_name": "xiaomi-mimo-v2.5-pro",
        "display_name": "Xiaomi MiMo V2.5 Pro",
        "model_id": "mimo-v2.5-pro",
        "provider": "xiaomi",
        "capability": "text",
        "supports_image": False,
        "supports_thinking": False,
        "context_length": 1000000,
        "max_output": 128000,
        "api_key_alias": "API_KEY1",
        "max_workers": 0,
        "enabled": False,  # Disabled - API key invalid
    })

    # Add selected Doubao models
    config = get_llm_config()
    if config.doubao_found:
        for dm in get_selected_models():
            dm["enabled"] = True
            dm["api_key_alias"] = "doubao_token_plan"
            dm["max_workers"] = 8
            models.append(dm)
    else:
        # Add Doubao models as disabled
        for dm in get_selected_models():
            dm["enabled"] = False
            dm["api_key_alias"] = "doubao_token_plan"
            dm["max_workers"] = 0
            models.append(dm)

    return models


def save_model_registry_6_5(models: list[dict] = None) -> tuple[Path, Path]:
    """Save Phase 6.5 model registry."""
    if models is None:
        models = load_model_registry_6_5()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_5_doubao_evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "model_registry.json"
    md_path = report_dir / "model_registry.md"

    # Sanitize
    sanitized = []
    for m in models:
        s = {k: v for k, v in m.items() if "key" not in k.lower() or k == "api_key_alias"}
        sanitized.append(s)

    with open(json_path, "w") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# Phase 6.5 Model Registry\n\n")
        f.write(f"Total models: {len(models)}\n")
        enabled = [m for m in models if m.get("enabled")]
        f.write(f"Enabled: {len(enabled)}\n\n")
        f.write("| Model | Provider | Capability | Image | Thinking | Enabled |\n")
        f.write("|-------|----------|-----------|-------|----------|--------|\n")
        for m in models:
            f.write(f"| {m.get('display_name', m['model_name'])} | {m['provider']} | {m.get('capability', 'text')} | {m.get('supports_image', False)} | {m.get('supports_thinking', False)} | {m.get('enabled', True)} |\n")

    return json_path, md_path
