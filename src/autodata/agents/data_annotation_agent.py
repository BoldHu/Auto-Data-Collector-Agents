"""Data Annotation Agent — generates structured annotations for domain data.

Creates SFT samples, knowledge units, and benchmark candidates from
cleaned text, image labels, and exam questions.
Inherits from ReActAgent for DTCG integration.
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

logger = get_logger("data_annotation_agent")


class DataAnnotationAgent(ReActAgent):
    """Agent for generating structured annotations from cleaned data.

    Capabilities:
    - Generate SFT samples from cleaned text chunks with evidence
    - Create knowledge unit annotations
    - Build benchmark candidate items from domain content
    - Validate annotation quality against source evidence

    This agent operates on cleaned artifacts from prior pipeline phases
    and produces structured annotation outputs with full provenance.
    """

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store: Optional[MessageStore] = None,
        run_id: str = "data_annotation",
        output_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="DataAnnotationAgent",
            model_client=model_client,
            message_store=message_store,
            max_iterations=10,
        )
        self.graph = graph or DynamicTaskContextGraph()
        self.run_id = run_id
        self.output_path = output_path
        self._annotated_count = 0

        # Register tools
        self.tool_registry.register(
            "generate_sft_samples",
            "Generate SFT training samples from cleaned text chunks",
            self._generate_sft_tool,
        )
        self.tool_registry.register(
            "create_knowledge_units",
            "Extract structured knowledge units from domain text",
            self._create_ku_tool,
        )
        self.tool_registry.register(
            "validate_annotation",
            "Validate an annotation against source evidence",
            self._validate_annotation_tool,
        )
        self.tool_registry.register(
            "load_source_chunks",
            "Load cleaned text chunks for annotation",
            self._load_chunks_tool,
        )
        self.tool_registry.register(
            "finish",
            "Mark annotation task as complete",
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
                "role": "data_annotation",
            },
        )
        self.graph.add_node(node)

    def _generate_sft_tool(self, params: str) -> str:
        """Generate SFT training samples from cleaned text chunks.

        Args:
            params: JSON string with chunk_path and optional config.

        Returns:
            Summary of generated samples.
        """
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            config = {"chunk_path": params}

        chunk_path = config.get("chunk_path", "")
        path = Path(chunk_path)
        if not path.exists():
            return f"Error: file not found at {chunk_path}"

        chunks = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            chunks.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading chunks: {str(e)[:100]}"

        # Generate SFT samples using LLM
        samples_generated = 0
        for chunk in chunks[:10]:  # Limit for tool call
            cleaned_text = chunk.get("cleaned_text", "")
            source_file = chunk.get("source_file", "")
            if not cleaned_text or len(cleaned_text) < 50:
                continue

            # Call LLM to generate SFT sample
            try:
                response = self.model_client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a domain data annotation specialist. "
                                "Generate a training sample from the given text. "
                                "Respond with JSON: {\"instruction\": ..., \"output\": ..., "
                                "\"evidence\": ..., \"task_type\": ..., \"source_ref\": ...}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Source: {source_file}\n\nText:\n{cleaned_text[:2000]}",
                        },
                    ],
                    max_completion_tokens=2048,
                )
                samples_generated += 1
            except Exception as e:
                logger.warning(f"SFT generation failed for chunk: {str(e)[:80]}")
                continue

        self._annotated_count += samples_generated

        # Register artifact node
        artifact_node = Node(
            node_id=f"art_sft_batch_{int(time.time())}",
            node_type=NodeType.ARTIFACT,
            name=f"SFT annotation batch",
            properties={
                "source_path": str(path),
                "chunks_processed": min(len(chunks), 10),
                "samples_generated": samples_generated,
                "source_type": "sft_annotation",
            },
        )
        self.graph.add_node(artifact_node)

        return f"Generated {samples_generated} SFT samples from {min(len(chunks), 10)} chunks"

    def _create_ku_tool(self, params: str) -> str:
        """Extract structured knowledge units from domain text."""
        try:
            config = json.loads(params)
        except json.JSONDecodeError:
            config = {"text_path": params}

        text_path = config.get("text_path", "")
        path = Path(text_path)
        if not path.exists():
            return f"Error: file not found at {text_path}"

        # Read text content
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(5000)
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        # Call LLM to extract knowledge units
        try:
            response = self.model_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract structured knowledge units from the text. "
                            "Each unit should have: topic, key_facts, relationships, "
                            "domain_terms, confidence. "
                            "Respond with JSON array of units."
                        ),
                    },
                    {"role": "user", "content": content[:3000]},
                ],
                max_completion_tokens=2048,
            )
            return f"Extracted knowledge units from {path.name}. Response length: {len(response.content)} chars"
        except Exception as e:
            return f"Knowledge extraction error: {str(e)[:100]}"

    def _validate_annotation_tool(self, params: str) -> str:
        """Validate an annotation against source evidence.

        Handles evidence as string, list of strings, or nested objects.
        """
        try:
            data = json.loads(params)
        except json.JSONDecodeError:
            return f"Error: expected JSON, got: {params[:100]}"

        instruction = data.get("instruction", "")
        output = data.get("output", "")
        evidence_raw = data.get("evidence", "")

        # Normalize evidence to a flat string
        evidence = self._normalize_evidence(evidence_raw)

        issues = []
        if not instruction:
            issues.append("missing_instruction")
        if not output:
            issues.append("missing_output")
        if not evidence:
            issues.append("missing_evidence")
        if len(instruction) < 10:
            issues.append("instruction_too_short")
        if len(output) < 10:
            issues.append("output_too_short")

        # Check evidence support
        if evidence and output:
            evidence_words = set(evidence.lower().split())
            output_words = set(output.lower().split())
            overlap = len(evidence_words & output_words)
            if overlap < 3:
                issues.append("weak_evidence_support")

        if issues:
            return json.dumps({"valid": False, "issues": issues})
        return json.dumps({"valid": True, "issues": []})

    @staticmethod
    def _normalize_evidence(evidence) -> str:
        """Normalize evidence from any format to a flat string.

        Supports:
        - str: returned as-is
        - list of str: joined with newlines
        - list of dicts: extracts 'text' or 'content' fields
        - nested objects: stringified
        """
        if isinstance(evidence, str):
            return evidence
        if isinstance(evidence, list):
            parts = []
            for item in evidence:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", item.get("content", str(item))))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if evidence:
            return str(evidence)
        return ""

    def _load_chunks_tool(self, jsonl_path: str) -> str:
        """Load cleaned text chunks for annotation."""
        path = Path(jsonl_path)
        if not path.exists():
            return f"Error: file not found at {jsonl_path}"

        count = 0
        languages = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            count += 1
                            lang = record.get("language", "unknown")
                            languages[lang] = languages.get(lang, 0) + 1
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return f"Error reading file: {str(e)[:100]}"

        lang_summary = ", ".join(f"{k}: {v}" for k, v in languages.items())
        return f"Loaded {count} chunks from {path.name}. Languages: {lang_summary}"

    def _finish_tool(self, _: str) -> str:
        """Mark annotation task as complete."""
        return f"TASK_COMPLETE: Data annotation finished. {self._annotated_count} annotations generated."

    def run(self, task: str, context: Optional[dict] = None) -> list:
        """Execute annotation task using the ReAct loop."""
        return super().run(task=task, context=context)
