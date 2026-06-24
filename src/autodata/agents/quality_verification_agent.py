"""Quality Verification Agent — independently reviews cleaned text.

Uses mimo-v2.5-pro with a separate verification prompt to score
clarity, completeness, consistency, feasibility, complexity, and
domain relevance. Detects over-cleaning, hallucination, formula loss,
duplication, and non-domain content.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.autodata.agents.react_agent import ReActAgent, ToolRegistry
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    EdgeType,
    Node,
    NodeType,
)
from src.autodata.context_graph.local_cache import CacheEntryType
from src.autodata.context_graph.message_store import MessageType, MessageStore, Visibility
from src.autodata.pipelines.prompts.text_cleaning_prompts import (
    PROMPT_VERSION,
    get_quality_verification_prompt,
)
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    QualityScore,
    QualityVerdict,
)
from src.autodata.utils.logging_utils import get_logger
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client
from src.autodata.utils.io_utils import append_jsonl_record

logger = get_logger("quality_verification_agent")


class QualityVerificationAgent(ReActAgent):
    """Agent that independently verifies quality of cleaned text chunks.

    Uses mimo-v2.5-pro with a SEPARATE verification prompt (never
    the same prompt used for cleaning) to ensure independent critique.

    Scoring dimensions:
    - clarity: readability and clarity
    - completeness: source content preservation
    - consistency: internal consistency
    - feasibility: whether usable for pretraining/SFT
    - complexity: technical depth
    - domain_relevance: carbon fiber domain relevance

    Verdicts: passed, needs_revision, failed
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
            name="QualityVerificationAgent",
            model_client=model_client,
            message_store=message_store,
            **kwargs,
        )
        self.graph = graph or DynamicTaskContextGraph()
        self.output_path = output_path
        self.run_id = run_id
        self._verified_count = 0
        self._total_tokens = 0

        self.tool_registry.register(
            "quality_verifier",
            "Verify quality of cleaned text chunks",
            func=self._verify_chunk_tool,
        )
        self.tool_registry.register("finish", "Signal task completion")

        self._register_in_graph()

    def _register_in_graph(self) -> None:
        node = Node(
            node_id=self.graph_node_id,
            node_type=NodeType.AGENT,
            name=self.name,
            properties={
                "framework": "react",
                "model": self.model,
                "role": "quality_verification",
            },
        )
        self.graph.add_node(node)

    def verify_chunk(
        self,
        cleaned_chunk: CleanedChunk,
    ) -> QualityScore:
        """Verify a cleaned chunk's quality using an independent prompt.

        Args:
            cleaned_chunk: The CleanedChunk to verify.

        Returns:
            QualityScore with all dimensions scored and a verdict.
        """
        # Format verification prompt (separate from cleaning prompt)
        prompt = get_quality_verification_prompt(
            cleaned_text=cleaned_chunk.cleaned_text,
            original_text=cleaned_chunk.original_text,
        )

        start_time = time.time()
        try:
            response = self.model_client.chat(
                messages=[
                    {"role": "system", "content": "You are an independent text quality verification expert. Always respond with valid JSON. You must be critical and objective."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                max_completion_tokens=2048,
            )
            latency_ms = (time.time() - start_time) * 1000
            self._total_tokens += response.total_tokens
        except Exception as e:
            logger.error(f"Verification API call failed: {str(e)[:100]}")
            return QualityScore(
                verdict=QualityVerdict.NEEDS_REVISION,
                issues=["verification_api_failed"],
            )

        # Parse JSON response
        quality = self._parse_verification_response(response.content)
        quality.verification_model = self.model
        quality.verification_timestamp = time.time()

        # Update chunk's quality score
        cleaned_chunk.quality_score = quality

        # Write quality-score record to JSONL output (always write, never skip)
        if self.output_path:
            record = {
                "chunk_id": cleaned_chunk.chunk_id,
                "source_file": cleaned_chunk.source_file,
                "source_folder": cleaned_chunk.source_folder,
                "page_numbers": cleaned_chunk.page_numbers,
                "language": cleaned_chunk.language.value,
                "clarity": quality.clarity,
                "completeness": quality.completeness,
                "consistency": quality.consistency,
                "feasibility": quality.feasibility,
                "complexity": quality.complexity,
                "domain_relevance": quality.domain_relevance,
                "average_score": quality.average,
                "final_status": quality.verdict.value,
                "detected_issues": quality.issues,
                "verifier_model": quality.verification_model,
                "prompt_version": PROMPT_VERSION,
                "run_id": self.run_id,
                "timestamp": quality.verification_timestamp,
            }
            append_jsonl_record(self.output_path, record)

        # Add quality feedback edge in DTCG
        chunk_artifact_id = f"art_{cleaned_chunk.chunk_id}"
        quality_node = Node(
            node_id=f"quality_{cleaned_chunk.chunk_id}",
            node_type=NodeType.MEMORY,
            name=f"Quality score for {cleaned_chunk.chunk_id}",
            properties={
                "verdict": quality.verdict.value,
                "average_score": quality.average,
                "issues": quality.issues,
            },
        )
        self.graph.add_node(quality_node)

        # Add to local cache
        self.add_to_cache(
            CacheEntryType.VERIFIED_FACT,
            content=f"Verified {cleaned_chunk.chunk_id}: verdict={quality.verdict.value}, avg={quality.average:.2f}",
            relevance_tags=["quality", "verification"],
            importance=0.8,
        )

        # Send quality feedback message
        self.send_message(
            receiver="DataCleaningAgent",
            content=f"Quality verdict for {cleaned_chunk.chunk_id}: {quality.verdict.value}, avg={quality.average:.2f}, issues={quality.issues}",
            task_id="verify_quality",
            message_type=MessageType.CRITIQUE,
            visibility=Visibility.LOCAL,
        )

        self._verified_count += 1
        return quality

    def _parse_verification_response(self, response_text: str) -> QualityScore:
        """Parse the JSON verification response from the LLM."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return QualityScore(
                    clarity=float(data.get("clarity", 0.5)),
                    completeness=float(data.get("completeness", 0.5)),
                    consistency=float(data.get("consistency", 0.5)),
                    feasibility=float(data.get("feasibility", 0.5)),
                    complexity=float(data.get("complexity", 0.5)),
                    domain_relevance=float(data.get("domain_relevance", 0.5)),
                    verdict=QualityVerdict(data.get("verdict", "needs_revision")),
                    issues=data.get("issues", []),
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: heuristic scoring
        return QualityScore(
            clarity=0.5,
            completeness=0.5,
            consistency=0.5,
            feasibility=0.5,
            complexity=0.5,
            domain_relevance=0.5,
            verdict=QualityVerdict.NEEDS_REVISION,
            issues=["json_parse_failed"],
        )

    def _verify_chunk_tool(self, action_input: str) -> str:
        """Tool function for ReAct framework."""
        try:
            data = json.loads(action_input)
            chunk = CleanedChunk(**data)
            result = self.verify_chunk(chunk)
            return f"Verified: verdict={result.verdict.value}, avg={result.average:.2f}"
        except Exception as e:
            return f"Verification tool error: {str(e)[:100]}"