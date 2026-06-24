"""Data Cleaning Agent — cleans OCR/book text using Xiaomi LLM.

Inherits from ReActAgent, uses mimo-v2.5-pro by default,
receives graph-selected context, and writes cleaned chunks
with full provenance to JSONL files.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.autodata.agents.base_agent import (
    AgentFramework,
    AgentObservation,
    BaseAgent,
)
from src.autodata.agents.react_agent import ReActAgent, ToolRegistry
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    EdgeType,
    Node,
    NodeType,
    TaskStatus,
)
from src.autodata.context_graph.local_cache import CacheEntryType
from src.autodata.context_graph.message_store import MessageType, MessageStore, Visibility
from src.autodata.pipelines.prompts.text_cleaning_prompts import (
    PROMPT_VERSION,
    get_cleaning_prompt,
)
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    Language,
    QualityScore,
    QualityVerdict,
    content_hash,
)
from src.autodata.utils.logging_utils import get_logger, safe_serialize
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client
from src.autodata.utils.io_utils import append_jsonl_record

logger = get_logger("data_cleaning_agent")


class DataCleaningAgent(ReActAgent):
    """Agent that cleans OCR/book text chunks using Xiaomi LLM.

    Workflow:
    1. Receive a chunk to clean (via context package or direct call)
    2. Format the cleaning prompt based on language
    3. Call Xiaomi LLM (mimo-v2.5-pro)
    4. Parse the JSON response
    5. Create a CleanedChunk with full provenance
    6. Write to JSONL output file
    7. Update local cache and DTCG
    8. Send observation to MessageStore
    """

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        message_store: Optional[MessageStore] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        output_path: Optional[str] = None,
        run_id: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            name="DataCleaningAgent",
            model_client=model_client,
            message_store=message_store,
            **kwargs,
        )
        self.graph = graph or DynamicTaskContextGraph()
        self.output_path = output_path
        self.run_id = run_id
        self._cleaned_count = 0
        self._total_tokens = 0

        # Register cleaning tool
        self.tool_registry.register(
            "text_cleaner",
            "Clean OCR/book text chunks using Xiaomi LLM",
            func=self._clean_chunk_tool,
        )
        self.tool_registry.register(
            "finish",
            "Signal task completion",
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
                "role": "text_cleaning",
            },
        )
        self.graph.add_node(node)

    def clean_chunk(
        self,
        chunk_data: dict[str, Any],
    ) -> Optional[CleanedChunk]:
        """Clean a single text chunk using Xiaomi LLM.

        Args:
            chunk_data: Dict with chunk_text, page_number, source_file,
                        source_folder, content_hash, language, chunk_type

        Returns:
            CleanedChunk or None if cleaning failed
        """
        raw_text = chunk_data.get("chunk_text", "")
        language = chunk_data.get("language", "zh")
        source_file = chunk_data.get("source_file", "")
        source_folder = chunk_data.get("source_folder", "")
        page_number = chunk_data.get("page_number", 0)
        chunk_type = chunk_data.get("chunk_type", "body")

        # Skip empty and header/footer chunks
        if chunk_type in ("empty", "header_footer"):
            return CleanedChunk(
                chunk_id=f"chunk_{content_hash(raw_text)[:8]}",
                source_file=source_file,
                source_folder=source_folder,
                page_numbers=[page_number],
                language=Language(language),
                original_text=raw_text,
                cleaned_text=raw_text,
                original_content_hash=content_hash(raw_text),
                cleaned_content_hash=content_hash(raw_text),
                cleaning_model=self.model,
                cleaning_prompt_version=PROMPT_VERSION,
                run_id=self.run_id,
                chunk_type=chunk_type,
            )

        # Format cleaning prompt (route based on chunk_type)
        prompt = get_cleaning_prompt(language, raw_text, chunk_type=chunk_type)

        # Call Xiaomi LLM
        start_time = time.time()
        try:
            response = self.model_client.chat(
                messages=[
                    {"role": "system", "content": "You are a professional technical document cleaning specialist. Always respond with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model="mimo-v2.5-pro",
                max_completion_tokens=4096,
            )
            latency_ms = (time.time() - start_time) * 1000
            self._total_tokens += response.total_tokens
        except Exception as e:
            logger.error(f"Cleaning API call failed: {str(e)[:100]}")
            return None

        # Parse JSON response
        cleaned_text, actions, confidence = self._parse_cleaning_response(response.content)

        # If parsing failed, use raw response as cleaned text
        if not cleaned_text:
            cleaned_text = response.content
            confidence = 0.3

        # Create CleanedChunk
        chunk = CleanedChunk(
            chunk_id=f"chunk_{content_hash(raw_text)[:8]}_{self._cleaned_count}",
            source_file=source_file,
            source_folder=source_folder,
            page_numbers=[page_number],
            language=Language(language),
            original_text=raw_text,
            cleaned_text=cleaned_text,
            original_content_hash=content_hash(raw_text),
            cleaned_content_hash=content_hash(cleaned_text),
            cleaning_model=self.model,
            cleaning_prompt_version=PROMPT_VERSION,
            cleaning_timestamp=time.time(),
            run_id=self.run_id,
            chunk_type=chunk_type,
            metadata={
                "confidence": confidence,
                "cleaning_actions": actions[:5],
                "latency_ms": latency_ms,
                "api_tokens": response.total_tokens,
            },
        )

        # Add to local cache
        self.add_to_cache(
            CacheEntryType.OBSERVATION,
            content=f"Cleaned chunk from {source_file} page {page_number}: {cleaned_text[:100]}",
            relevance_tags=["cleaning", language, source_folder],
            importance=0.7,
        )

        # Write to JSONL output
        if self.output_path:
            append_jsonl_record(self.output_path, chunk.to_dict())

        # Create artifact node in DTCG
        artifact_node = Node(
            node_id=f"art_{chunk.chunk_id}",
            node_type=NodeType.ARTIFACT,
            name=f"Cleaned chunk {chunk.chunk_id}",
            properties={
                "source_file": source_file,
                "page_number": page_number,
                "language": language,
                "chunk_type": chunk_type,
            },
        )
        self.graph.add_node(artifact_node)

        self._cleaned_count += 1

        # Send message
        self.send_message(
            receiver="CentralPlanningAgent",
            content=f"Cleaned chunk from {source_file} p.{page_number}, lang={language}, confidence={confidence}",
            task_id="clean_text",
            message_type=MessageType.OBSERVATION,
            visibility=Visibility.LOCAL,
        )

        return chunk

    def _parse_cleaning_response(
        self, response_text: str
    ) -> tuple[str, list[dict], float]:
        """Parse the JSON cleaning response from the LLM."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                cleaned = data.get("cleaned_text", "")
                actions = data.get("cleaning_actions", [])
                confidence = float(data.get("confidence", 0.5))
                return cleaned, actions, confidence
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: treat entire response as cleaned text
        return response_text, [], 0.3

    def _clean_chunk_tool(self, action_input: str) -> str:
        """Tool function for ReAct framework."""
        # action_input is expected to be a JSON string of chunk data
        try:
            chunk_data = json.loads(action_input)
            result = self.clean_chunk(chunk_data)
            if result:
                return f"Cleaned chunk {result.chunk_id}, confidence={result.metadata.get('confidence', 0)}"
            return "Cleaning failed"
        except json.JSONDecodeError:
            return f"Invalid input format: {action_input[:50]}"