"""Knowledge-unit extraction from cleaned text.

Extracts atomic carbon-fiber knowledge units from cleaned chunks
using Xiaomi LLM. Each unit includes topic, knowledge_type, claim,
evidence_text, entities, relations, conditions, numeric_values,
and full provenance.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.autodata.pipelines.prompts.text_cleaning_prompts import (
    PROMPT_VERSION,
    get_knowledge_extraction_prompt,
)
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    KnowledgeUnit,
    KnowledgeType,
    Language,
    QualityScore,
    QualityVerdict,
)
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client
from src.autodata.utils.io_utils import append_jsonl_record
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("knowledge_extractor")


def extract_knowledge_units(
    chunk: CleanedChunk,
    model_client: Optional[XiaomiModelClient] = None,
    output_path: Optional[str] = None,
    run_id: str = "",
) -> list[KnowledgeUnit]:
    """Extract knowledge units from a single cleaned chunk.

    Args:
        chunk: The CleanedChunk to extract knowledge from.
        model_client: Xiaomi LLM client (uses default if None).
        output_path: Optional JSONL output file path.
        run_id: Current run ID for provenance.

    Returns:
        List of KnowledgeUnit objects.
    """
    client = model_client or get_default_client()

    # Skip empty/header_footer chunks
    if chunk.chunk_type in ("empty", "header_footer"):
        return []

    # Format extraction prompt
    prompt = get_knowledge_extraction_prompt(chunk.cleaned_text)

    start_time = time.time()
    try:
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a carbon fiber domain knowledge extraction expert. Always respond with valid JSON array. Only extract knowledge directly supported by the source text."},
                {"role": "user", "content": prompt},
            ],
            model=client.default_model,
            max_completion_tokens=4096,
        )
        latency_ms = (time.time() - start_time) * 1000
    except Exception as e:
        logger.error(f"Knowledge extraction API call failed: {str(e)[:100]}")
        return []

    # Parse JSON response
    units = _parse_knowledge_units(response.content, chunk, run_id, client)

    # Write to JSONL output
    if output_path and units:
        for unit in units:
            append_jsonl_record(output_path, unit.to_dict())

    return units


def _parse_knowledge_units(
    response_text: str,
    chunk: CleanedChunk,
    run_id: str,
    client: XiaomiModelClient,
) -> list[KnowledgeUnit]:
    """Parse the JSON knowledge extraction response."""
    # Try to extract JSON array
    try:
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            entries = json.loads(response_text[json_start:json_end])
            if isinstance(entries, list):
                units = []
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    kt_str = e.get("knowledge_type", "other")
                    try:
                        kt = KnowledgeType(kt_str)
                    except ValueError:
                        kt = KnowledgeType.OTHER

                    unit = KnowledgeUnit(
                        unit_id=f"ku_{chunk.chunk_id}_{len(units)}",
                        source_chunk_id=chunk.chunk_id,
                        language=chunk.language,
                        topic=e.get("topic", ""),
                        subtopic=e.get("subtopic", ""),
                        knowledge_type=kt,
                        claim=e.get("claim", ""),
                        evidence_text=e.get("evidence_text", ""),
                        entities=e.get("entities", []),
                        relations=e.get("relations", []),
                        conditions=e.get("conditions", []),
                        numeric_values=e.get("numeric_values", []),
                        source_refs=[
                            chunk.source_file,
                            chunk.source_folder,
                            str(chunk.page_numbers),
                        ],
                        extraction_model=client.default_model,
                        extraction_timestamp=time.time(),
                        run_id=run_id,
                    )
                    units.append(unit)
                return units
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: create a single generic unit from the whole text
    if chunk.cleaned_text.strip():
        return [KnowledgeUnit(
            unit_id=f"ku_{chunk.chunk_id}_0",
            source_chunk_id=chunk.chunk_id,
            language=chunk.language,
            topic="unparsed",
            knowledge_type=KnowledgeType.OTHER,
            claim=chunk.cleaned_text[:200],
            evidence_text=chunk.cleaned_text[:500],
            source_refs=[chunk.source_file, chunk.source_folder],
            extraction_model=client.default_model,
            run_id=run_id,
        )]
    return []