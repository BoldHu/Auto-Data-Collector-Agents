"""Fine-Tuning Agent — manages model fine-tuning and evaluation.

Wraps LoRA/QLoRA training scripts with safe dry-run modes,
logs configurations, and manages adapter artifacts.
Inherits from ReActAgent for DTCG integration.

Note: Actual training execution requires GPU resources and local model weights.
This agent provides the interface and artifact management; training is
controlled by the --skip_training flag in orchestration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from src.autodata.agents.react_agent import ReActAgent
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    EdgeType,
    Node,
    NodeType,
)
from src.autodata.context_graph.local_cache import CacheEntryType
from src.autodata.context_graph.message_store import MessageType, MessageStore, Visibility
from src.autodata.utils.logging_utils import get_logger
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client

logger = get_logger("finetuning_agent")


class FineTuningAgent(ReActAgent):
    """Agent for managing model fine-tuning workflows.

    Capabilities:
    - Prepare training data manifests (SFT ChatML format)
    - Configure LoRA/QLoRA training runs
    - Execute training in dry-run or real mode
    - Evaluate trained adapters on benchmark subsets
    - Track adapter artifacts and training configs

    This agent wraps existing training scripts (train_lora.py) and
    evaluation scripts with proper artifact management and DTCG integration.
    """

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store: Optional[MessageStore] = None,
        run_id: str = "finetuning",
        output_path: Optional[str] = None,
        skip_training: bool = True,
    ) -> None:
        super().__init__(
            name="FineTuningAgent",
            model_client=model_client,
            message_store=message_store,
            max_iterations=10,
        )
        self.graph = graph or DynamicTaskContextGraph()
        self.run_id = run_id
        self.output_path = output_path
        self.skip_training = skip_training
        self._configs_prepared = 0
        self._adapters_evaluated = 0

        # Register tools
        self.tool_registry.register(
            "prepare_training_data",
            "Prepare SFT training data in ChatML format",
            self._prepare_data_tool,
        )
        self.tool_registry.register(
            "configure_training",
            "Configure a LoRA/QLoRA training run",
            self._configure_tool,
        )
        self.tool_registry.register(
            "run_training",
            "Execute training (dry-run if skip_training=True)",
            self._run_training_tool,
        )
        self.tool_registry.register(
            "evaluate_adapter",
            "Evaluate a trained adapter on benchmark items",
            self._evaluate_adapter_tool,
        )
        self.tool_registry.register(
            "list_adapters",
            "List available trained adapters",
            self._list_adapters_tool,
        )
        self.tool_registry.register(
            "finish",
            "Mark fine-tuning task as complete",
            self._finish_tool,
        )

        # Register agent node in DTCG
        self._register_in_graph()

    def _register_in_graph(self) -> None:
        """Register this agent as a node in the DTCG."""
        node = Node(
            node_id=self.graph_node_id,
            node_type=NodeType.AGENT,
            name=self.name,
            properties={
                "framework": "react",
                "model": self.model,
                "role": "finetuning",
                "skip_training": self.skip_training,
            },
        )
        self.graph.add_node(node)

    def _prepare_data_tool(self, params: str) -> str:
        """Prepare SFT training data in ChatML format."""
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            config = {"sft_path": params}

        sft_path = config.get("sft_path", "")
        path = Path(sft_path)
        if not path.exists():
            return f"Error: file not found at {sft_path}"

        # Count records and check format
        total = 0
        valid = 0
        task_types = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        record = json.loads(line)
                        if record.get("instruction") and record.get("output"):
                            valid += 1
                        tt = record.get("task_type", "unknown")
                        task_types[tt] = task_types.get(tt, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            return f"Error reading SFT data: {str(e)[:100]}"

        return json.dumps({
            "source": str(path),
            "total_records": total,
            "valid_records": valid,
            "task_type_distribution": dict(sorted(task_types.items(), key=lambda x: -x[1])[:10]),
            "format": "chatml_ready" if valid == total else "needs_conversion",
        }, ensure_ascii=False)

    def _configure_tool(self, params: str) -> str:
        """Configure a LoRA/QLoRA training run."""
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            return f"Error: expected JSON config, got: {params[:100]}"

        # Validate required fields
        required = ["model_name", "sft_data_path", "output_dir"]
        missing = [f for f in required if not config.get(f)]
        if missing:
            return f"Error: missing required fields: {', '.join(missing)}"

        # Add defaults
        config.setdefault("lora_r", 16)
        config.setdefault("lora_alpha", 32)
        config.setdefault("lora_dropout", 0.05)
        config.setdefault("learning_rate", 2e-4)
        config.setdefault("num_epochs", 3)
        config.setdefault("batch_size", 4)
        config.setdefault("max_seq_length", 2048)
        config.setdefault("seed", 42)
        config.setdefault("quantization", "none")

        self._configs_prepared += 1

        # Register config artifact
        artifact_node = Node(
            node_id=f"art_training_config_{self._configs_prepared}",
            node_type=NodeType.ARTIFACT,
            name=f"Training config {self._configs_prepared}",
            properties={
                "model_name": config.get("model_name"),
                "sft_data_path": config.get("sft_data_path"),
                "output_dir": config.get("output_dir"),
                "quantization": config.get("quantization"),
                "source_type": "training_config",
            },
        )
        self.graph.add_node(artifact_node)

        return json.dumps({
            "status": "configured",
            "config": config,
            "skip_training": self.skip_training,
        }, ensure_ascii=False)

    def _run_training_tool(self, params: str) -> str:
        """Execute training (dry-run if skip_training=True)."""
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            return f"Error: expected JSON config, got: {params[:100]}"

        model_name = config.get("model_name", "unknown")
        output_dir = config.get("output_dir", "")

        if self.skip_training:
            # Dry-run mode: log config but don't train
            logger.info(f"DRY RUN: Would train {model_name} -> {output_dir}")
            return json.dumps({
                "status": "dry_run",
                "model_name": model_name,
                "output_dir": output_dir,
                "message": "Training skipped (--skip_training mode). Config logged for reproducibility.",
                "config": config,
            })

        # Real training would call train_lora.py here
        # For now, this is a placeholder that marks the intent
        return json.dumps({
            "status": "training_requested",
            "model_name": model_name,
            "output_dir": output_dir,
            "message": "Training execution requires GPU resources and local model weights.",
        })

    def _evaluate_adapter_tool(self, params: str) -> str:
        """Evaluate a trained adapter on benchmark items."""
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            return f"Error: expected JSON config, got: {params[:100]}"

        adapter_path = config.get("adapter_path", "")
        benchmark_path = config.get("benchmark_path", "")

        # Check adapter exists
        if adapter_path and not Path(adapter_path).exists():
            return f"Error: adapter not found at {adapter_path}"

        # Check benchmark exists
        if benchmark_path and not Path(benchmark_path).exists():
            return f"Error: benchmark not found at {benchmark_path}"

        self._adapters_evaluated += 1

        # Register evaluation artifact
        artifact_node = Node(
            node_id=f"art_eval_{self._adapters_evaluated}",
            node_type=NodeType.ARTIFACT,
            name=f"Adapter evaluation {self._adapters_evaluated}",
            properties={
                "adapter_path": adapter_path,
                "benchmark_path": benchmark_path,
                "source_type": "adapter_evaluation",
            },
        )
        self.graph.add_node(artifact_node)

        return json.dumps({
            "status": "evaluation_configured",
            "adapter_path": adapter_path,
            "benchmark_path": benchmark_path,
            "message": "Evaluation requires model inference. Use run_final_qwen_eval.py for execution.",
        })

    def _list_adapters_tool(self, directory: str) -> str:
        """List available trained adapters."""
        path = Path(directory)
        if not path.exists():
            return f"Directory not found: {directory}"

        adapters = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                # Check for adapter files
                has_adapter = (item / "adapter_model.safetensors").exists()
                has_config = (item / "adapter_config.json").exists()
                status = "complete" if has_adapter and has_config else "incomplete"
                adapters.append({
                    "name": item.name,
                    "path": str(item),
                    "status": status,
                    "has_adapter": has_adapter,
                    "has_config": has_config,
                })

        if not adapters:
            return f"No adapters found in {directory}"

        return json.dumps({
            "directory": directory,
            "adapters": adapters,
            "total": len(adapters),
            "complete": sum(1 for a in adapters if a["status"] == "complete"),
        }, ensure_ascii=False, indent=2)

    def _finish_tool(self, _: str) -> str:
        """Mark fine-tuning task as complete."""
        return (
            f"TASK_COMPLETE: Fine-tuning finished. "
            f"{self._configs_prepared} configs prepared, "
            f"{self._adapters_evaluated} adapters evaluated."
        )

    def run(self, task: str, context: Optional[dict] = None) -> list:
        """Execute fine-tuning task using the ReAct loop."""
        return super().run(task=task, context=context)
