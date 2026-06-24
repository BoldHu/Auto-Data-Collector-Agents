"""SFT candidate generation from cleaned text and knowledge units.

Generates candidate supervised fine-tuning samples from cleaned chunks
using Xiaomi LLM. Each candidate is only a candidate — not final
training data. Full provenance is preserved.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.autodata.pipelines.prompts.text_cleaning_prompts import (
    PROMPT_VERSION,
    get_sft_generation_prompt,
)
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    Difficulty,
    SFTCandidate,
    SFTTaskType,
    Language,
    QualityScore,
    QualityVerdict,
)
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client
from src.autodata.utils.io_utils import append_jsonl_record
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("sft_candidate_generator")


def generate_sft_candidates(
    chunk: CleanedChunk,
    model_client: Optional[XiaomiModelClient] = None,
    output_path: Optional[str] = None,
    run_id: str = "",
) -> list[SFTCandidate]:
    """Generate SFT candidates from a single cleaned chunk.

    Args:
        chunk: The CleanedChunk to generate SFT samples from.
        model_client: Xiaomi LLM client (uses default if None).
        output_path: Optional JSONL output file path.
        run_id: Current run ID for provenance.

    Returns:
        List of SFTCandidate objects.
    """
    client = model_client or get_default_client()

    # Skip empty/header_footer chunks and short chunks
    if chunk.chunk_type in ("empty", "header_footer"):
        return []
    if len(chunk.cleaned_text.strip()) < 50:
        return []

    # Format SFT generation prompt
    prompt = get_sft_generation_prompt(chunk.cleaned_text)

    start_time = time.time()
    try:
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a carbon fiber domain SFT sample generation expert. Always respond with valid JSON array. Only generate samples supported by the source text. No hallucination."},
                {"role": "user", "content": prompt},
            ],
            model=client.default_model,
            max_completion_tokens=4096,
        )
        latency_ms = (time.time() - start_time) * 1000
    except Exception as e:
        logger.error(f"SFT generation API call failed: {str(e)[:100]}")
        return []

    # Parse JSON response
    candidates = _parse_sft_candidates(response.content, chunk, run_id, client)

    # Write to JSONL output
    if output_path and candidates:
        for c in candidates:
            append_jsonl_record(output_path, c.to_dict())

    return candidates


def _parse_sft_candidates(
    response_text: str,
    chunk: CleanedChunk,
    run_id: str,
    client: XiaomiModelClient,
) -> list[SFTCandidate]:
    """Parse the JSON SFT generation response."""
    try:
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            entries = json.loads(response_text[json_start:json_end])
            if isinstance(entries, list):
                candidates = []
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    tt_str = e.get("task_type", "qa")
                    try:
                        tt = SFTTaskType(tt_str)
                    except ValueError:
                        tt = SFTTaskType.QA

                    diff_str = e.get("difficulty", "medium")
                    try:
                        diff = Difficulty(diff_str)
                    except ValueError:
                        diff = Difficulty.MEDIUM

                    candidate = SFTCandidate(
                        sample_id=f"sft_{chunk.chunk_id}_{len(candidates)}",
                        source_chunk_id=chunk.chunk_id,
                        task_type=tt,
                        instruction=e.get("instruction", ""),
                        input=e.get("input", ""),
                        output=e.get("output", ""),
                        evidence_text=e.get("evidence_text", ""),
                        difficulty=diff,
                        source_refs=[
                            chunk.source_file,
                            chunk.source_folder,
                            str(chunk.page_numbers),
                        ],
                        generation_model=client.default_model,
                        generation_timestamp=time.time(),
                        run_id=run_id,
                    )
                    candidates.append(candidate)
                return candidates
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: create a single QA candidate from the text
    if len(chunk.cleaned_text.strip()) > 50:
        return [SFTCandidate(
            sample_id=f"sft_{chunk.chunk_id}_0",
            source_chunk_id=chunk.chunk_id,
            task_type=SFTTaskType.QA,
            instruction=f"请解释以下碳纤维相关内容（请根据原文回答，不要编造）：",
            input=chunk.cleaned_text[:200],
            output=chunk.cleaned_text[:500],
            evidence_text=chunk.cleaned_text[:300],
            difficulty=Difficulty.MEDIUM,
            source_refs=[chunk.source_file, chunk.source_folder],
            generation_model=client.default_model,
            run_id=run_id,
        )]
    return []