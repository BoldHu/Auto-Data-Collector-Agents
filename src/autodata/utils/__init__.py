"""Shared utility functions for AutoData."""

from src.autodata.utils.api_loader import (
    XiaomiConfig,
    BaselineModelConfig,
    load_xiaomi_config,
    load_baseline_configs,
    create_xiaomi_openai_client,
    create_baseline_openai_client,
)
from src.autodata.utils.model_client import (
    XiaomiModelClient,
    ChatResponse,
    get_default_client,
)
from src.autodata.utils.baseline_model_loader import (
    BaselineModelRunner,
    BaselineResponse,
    load_baseline_models,
    create_runners,
)
from src.autodata.utils.logging_utils import (
    setup_logging,
    get_logger,
    safe_serialize,
)
from src.autodata.utils.io_utils import (
    atomic_write_text,
    atomic_write_json,
    atomic_write_jsonl,
    safe_read_json,
    safe_read_jsonl,
    safe_read_yaml,
    append_jsonl_record,
    ensure_dir,
    get_project_root,
)

__all__ = [
    # API loader
    "XiaomiConfig",
    "BaselineModelConfig",
    "load_xiaomi_config",
    "load_baseline_configs",
    "create_xiaomi_openai_client",
    "create_baseline_openai_client",
    # Model client
    "XiaomiModelClient",
    "ChatResponse",
    "get_default_client",
    # Baseline loader
    "BaselineModelRunner",
    "BaselineResponse",
    "load_baseline_models",
    "create_runners",
    # Logging
    "setup_logging",
    "get_logger",
    "safe_serialize",
    # I/O
    "atomic_write_text",
    "atomic_write_json",
    "atomic_write_jsonl",
    "safe_read_json",
    "safe_read_jsonl",
    "safe_read_yaml",
    "append_jsonl_record",
    "ensure_dir",
    "get_project_root",
]