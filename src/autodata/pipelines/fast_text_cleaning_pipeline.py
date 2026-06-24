"""Fast text cleaning pipeline with ModelPool, concurrent workers, v2.0 prompts.

Key improvements over v1.0 pipeline:
- ModelPool: dual-API round-robin, failover, automatic model selection
- Concurrent workers: ThreadPoolExecutor with configurable num_workers
- v2.0 prompts: domain filtering, OCR repair rules, boilerplate removal
- v2.0 schema: keep_for_corpus, enriched_notes, uncertainty_notes, etc.
- Thread-safe output: all JSONL writes go through a shared lock
- Checkpointing: resume from last checkpoint if interrupted
- Progress tracking: enhanced with per-API stats
- Quality gate: skip KU/SFT generation for non-corpus chunks
- Graceful shutdown: finish current chunks, then stop

Default config:
- num_workers=4 (parallel chunk processing)
- Fast model: mimo-v2-omni (for cleaning)
- Quality model: mimo-v2.5-pro (for verification, KU, SFT)
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from src.autodata.pipelines.knowledge_extractor import extract_knowledge_units
from src.autodata.pipelines.prompts.text_cleaning_prompts import (
    PROMPT_VERSION,
    get_cleaning_prompt,
    get_knowledge_extraction_prompt,
    get_quality_verification_prompt,
    get_sft_generation_prompt,
)
from src.autodata.pipelines.sft_candidate_generator import generate_sft_candidates
from src.autodata.pipelines.text_preprocessor import (
    load_raw_document,
    preprocess_document,
    generate_noise_report,
)
from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    CleaningRunMetadata,
    Language,
    QualityScore,
    QualityVerdict,
    content_hash,
)
from src.autodata.utils.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    append_jsonl_record,
    ensure_dir,
    safe_read_json,
)
from src.autodata.utils.logging_utils import get_logger, setup_logging
from src.autodata.utils.model_pool import ModelPool, get_model_pool

logger = get_logger("fast_text_cleaning_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Thread-safe output lock
_output_lock = threading.Lock()


class FastTextCleaningPipeline:
    """Fast text cleaning pipeline with ModelPool and concurrent workers.

    Usage:
        pipeline = FastTextCleaningPipeline(
            run_id="phase_2_7_reclean_fast",
            language="zh",
            num_workers=4,
        )
        metadata = pipeline.run()
    """

    def __init__(
        self,
        run_id: str = "",
        language: str = "zh",
        num_workers: int = 4,
        fast_model: str = "mimo-v2-omni",
        quality_model: str = "mimo-v2.5-pro",
        max_files: Optional[int] = None,
        max_pages_per_file: Optional[int] = None,
        max_chunks: Optional[int] = None,
        resume: bool = False,
        skip_ku_sft: bool = False,
        skip_quality: bool = False,
        output_suffix: str = "_reclean",
    ) -> None:
        self.run_id = run_id or f"phase_2_7_reclean_fast_{int(time.time())}"
        self.language = language
        self.num_workers = num_workers
        self.fast_model = fast_model
        self.quality_model = quality_model
        self.max_files = max_files
        self.max_pages_per_file = max_pages_per_file
        self.max_chunks = max_chunks
        self.resume = resume
        self.skip_ku_sft = skip_ku_sft
        self.skip_quality = skip_quality
        self.output_suffix = output_suffix

        # ModelPool for dual-API round-robin
        self.pool = get_model_pool()

        # Output paths
        self._setup_output_paths()

        # Progress tracking
        self._start_time = time.time()
        self._completed_chunks = 0
        self._failed_chunks = 0
        self._skipped_chunks = 0
        self._total_llm_calls = 0
        self._total_tokens = 0
        self._corpus_chunks = 0  # keep_for_corpus=True count
        self._dropped_chunks = 0  # keep_for_corpus=False count
        self._errors: list[dict] = []
        self._progress_lock = threading.Lock()

        # Checkpoint
        self._processed_chunk_ids: set[str] = set()
        if self.resume:
            self._load_checkpoint()

        # Shutdown flag
        self._shutdown_requested = False

        # Register graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame) -> None:
        """Graceful shutdown: finish current chunks, then stop."""
        logger.info(f"Shutdown signal received ({signum}), finishing current chunks...")
        self._shutdown_requested = True

    def _setup_output_paths(self) -> None:
        """Create output directories for Phase 2.7 recleaning."""
        rid = self.run_id
        suffix = self.output_suffix
        base = PROJECT_ROOT

        report_dir = ensure_dir(base / "data" / "reports" / "phase_2_7_restart_cleaning")

        self.output_dirs = {
            "cleaned": ensure_dir(base / "data" / "interim" / "text_cleaned"),
            "pretraining": ensure_dir(base / "data" / "processed" / "pretraining_corpus"),
            "knowledge": ensure_dir(base / "data" / "processed" / "knowledge_units"),
            "sft": ensure_dir(base / "data" / "processed" / "sft_candidates"),
            "quality": ensure_dir(base / "data" / "processed" / "text_quality"),
            "reports": report_dir,
        }

        self.output_files = {
            "cleaned": self.output_dirs["cleaned"] / f"cleaned_chunks{suffix}.jsonl",
            "pretraining": self.output_dirs["pretraining"] / f"pretraining_corpus{suffix}.jsonl",
            "knowledge": self.output_dirs["knowledge"] / f"knowledge_units{suffix}.jsonl",
            "sft": self.output_dirs["sft"] / f"sft_candidates{suffix}.jsonl",
            "quality": self.output_dirs["quality"] / f"text_quality_scores{suffix}.jsonl",
            "checkpoint": report_dir / f"{rid}_checkpoint.json",
            "progress_json": report_dir / f"{rid}_progress.json",
            "progress_log": report_dir / f"{rid}_progress.log",
            "metadata": report_dir / f"{rid}_run_metadata.json",
            "errors": report_dir / f"{rid}_errors.jsonl",
            "noise": report_dir / f"ocr_noise_analysis{suffix}.json",
        }

    def _load_checkpoint(self) -> None:
        """Load checkpoint for resume."""
        cp_path = self.output_files["checkpoint"]
        data = safe_read_json(str(cp_path))
        if data and isinstance(data, dict):
            self._processed_chunk_ids = set(data.get("processed_chunk_ids", []))
            logger.info(f"Resumed from checkpoint: {len(self._processed_chunk_ids)} chunks already processed")

    def _save_checkpoint(self) -> None:
        """Save checkpoint state."""
        atomic_write_json(
            str(self.output_files["checkpoint"]),
            {
                "run_id": self.run_id,
                "processed_chunk_ids": list(self._processed_chunk_ids),
                "completed_chunks": self._completed_chunks,
                "failed_chunks": self._failed_chunks,
                "timestamp": time.time(),
                "pool_stats": self.pool.stats(),
            },
        )

    def _write_progress(self) -> None:
        """Write progress to JSON and log file."""
        elapsed = time.time() - self._start_time
        avg_per_chunk = elapsed / max(self._completed_chunks, 1)
        remaining = avg_per_chunk * (self._total_chunks_est - self._completed_chunks) if self._total_chunks_est > 0 else 0

        progress = {
            "run_id": self.run_id,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_formatted": f"{int(elapsed//3600):02d}:{int(elapsed%3600//60):02d}:{int(elapsed%60):02d}",
            "total_chunks_est": self._total_chunks_est,
            "completed_chunks": self._completed_chunks,
            "failed_chunks": self._failed_chunks,
            "skipped_chunks": self._skipped_chunks,
            "corpus_chunks": self._corpus_chunks,
            "dropped_chunks": self._dropped_chunks,
            "avg_seconds_per_chunk": round(avg_per_chunk, 2),
            "chunks_per_hour": round(self._completed_chunks / max(elapsed / 3600, 0.01), 1),
            "estimated_remaining_seconds": round(remaining, 1),
            "estimated_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining)),
            "total_llm_calls": self._total_llm_calls,
            "total_tokens": self._total_tokens,
            "num_workers": self.num_workers,
            "pool_stats": self.pool.stats(),
            "errors_last5": [e["error"] for e in self._errors[-5:]],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        atomic_write_json(str(self.output_files["progress_json"]), progress)

        # Log line
        log_line = (
            f"[{progress['timestamp']}] "
            f"Chunks: {self._completed_chunks}/{self._total_chunks_est} "
            f"(corpus={self._corpus_chunks}, dropped={self._dropped_chunks}, "
            f"failed={self._failed_chunks}) "
            f"LLM calls: {self._total_llm_calls} "
            f"ETA: {progress['estimated_finish_time']} "
            f"Pool: {self.pool.stats()['total_calls']} calls, "
            f"{self.pool.stats()['total_tokens']} tokens\n"
        )
        with open(str(self.output_files["progress_log"]), "a", encoding="utf-8") as f:
            f.write(log_line)

        # Also print to stdout for terminal monitoring
        print(log_line.strip(), flush=True)

    def _threadsafe_append(self, path: str, record: dict) -> None:
        """Thread-safe JSONL append."""
        with _output_lock:
            append_jsonl_record(path, record)

    def _record_error(self, error_msg: str, context: dict | None = None) -> None:
        """Record an error."""
        error_record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "error": error_msg,
            "context": context or {},
        }
        self._errors.append(error_record)
        self._threadsafe_append(str(self.output_files["errors"]), error_record)

    def _parse_v2_cleaning_response(
        self, response_text: str, raw_text: str
    ) -> dict[str, Any]:
        """Parse v2.0 JSON cleaning response with new fields."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])

                # Extract v2.0 fields
                cleaned_text = data.get("cleaned_text", "")
                enriched_notes = data.get("enriched_notes", "")
                keep_for_corpus = data.get("keep_for_corpus", True)
                drop_reason = data.get("drop_reason", "")
                removed_noise_types = data.get("removed_noise_types", [])
                ocr_repairs = data.get("ocr_repairs", [])
                technical_content_types = data.get("technical_content_types", [])
                uncertainty_notes = data.get("uncertainty_notes", "")
                confidence = float(data.get("confidence", 0.5))

                # Extract cleaning_actions (v1.0 compat)
                actions = data.get("cleaning_actions", [])

                return {
                    "cleaned_text": cleaned_text,
                    "enriched_notes": enriched_notes,
                    "keep_for_corpus": keep_for_corpus,
                    "drop_reason": drop_reason,
                    "removed_noise_types": removed_noise_types,
                    "ocr_repairs": ocr_repairs,
                    "technical_content_types": technical_content_types,
                    "uncertainty_notes": uncertainty_notes,
                    "cleaning_actions": actions,
                    "confidence": confidence,
                    "parse_success": True,
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed for cleaning response: {str(e)[:80]}")

        # Fallback: treat entire response as cleaned text, mark as uncertain
        return {
            "cleaned_text": response_text,
            "enriched_notes": "",
            "keep_for_corpus": True,  # keep by default, verify later
            "drop_reason": "",
            "removed_noise_types": [],
            "ocr_repairs": [],
            "technical_content_types": [],
            "uncertainty_notes": "JSON parse failed, raw response used as cleaned_text",
            "cleaning_actions": [],
            "confidence": 0.3,
            "parse_success": False,
        }

    def _process_chunk(self, chunk_data: dict, doc_language: str) -> Optional[CleanedChunk]:
        """Process a single chunk: clean, verify, extract KU/SFT.

        This is the main worker function, called by each thread.
        Uses ModelPool for all LLM calls.
        """
        source_file = chunk_data.get("source_file", "")
        page_number = chunk_data.get("page_number", 0)
        chunk_type = chunk_data.get("chunk_type", "body")
        raw_text = chunk_data.get("chunk_text", "")
        content_hash_val = chunk_data.get("content_hash", "")[:8]

        # Check checkpoint skip
        chunk_id_key = f"{source_file}_p{page_number}_{content_hash_val}"
        if chunk_id_key in self._processed_chunk_ids:
            logger.info(f"Skipping already processed: {chunk_id_key}")
            with self._progress_lock:
                self._skipped_chunks += 1
            return None

        # Check shutdown
        if self._shutdown_requested:
            return None

        # Handle empty/header_footer chunks (no LLM call needed)
        if chunk_type in ("empty",):
            with self._progress_lock:
                self._completed_chunks += 1
                self._skipped_chunks += 1
                self._dropped_chunks += 1
                self._processed_chunk_ids.add(chunk_id_key)
            return CleanedChunk(
                chunk_id=f"chunk_{content_hash_val}_skip",
                source_file=source_file,
                source_folder=chunk_data.get("source_folder", ""),
                page_numbers=[page_number],
                language=Language(doc_language),
                original_text=raw_text,
                cleaned_text="",
                original_content_hash=content_hash(raw_text),
                cleaned_content_hash=content_hash(""),
                cleaning_model="skip",
                cleaning_prompt_version=PROMPT_VERSION,
                run_id=self.run_id,
                chunk_type="empty",
                keep_for_corpus=False,
                drop_reason="空白内容",
                metadata={"confidence": 1.0, "skip_reason": "empty"},
            )

        if chunk_type == "header_footer":
            # Use LLM for header/footer to extract any useful info
            prompt = get_cleaning_prompt(doc_language, raw_text, chunk_type="header_footer")
            try:
                response = self.pool.chat(
                    messages=[
                        {"role": "system", "content": "You are a professional technical document cleaning specialist. Always respond with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=1024,
                )
                parsed = self._parse_v2_cleaning_response(response.content, raw_text)

                chunk = CleanedChunk(
                    chunk_id=f"chunk_{content_hash_val}_{self._completed_chunks}",
                    source_file=source_file,
                    source_folder=chunk_data.get("source_folder", ""),
                    page_numbers=[page_number],
                    language=Language(doc_language),
                    original_text=raw_text,
                    cleaned_text=parsed["cleaned_text"],
                    original_content_hash=content_hash(raw_text),
                    cleaned_content_hash=content_hash(parsed["cleaned_text"]),
                    cleaning_model=response.model,
                    cleaning_prompt_version=PROMPT_VERSION,
                    run_id=self.run_id,
                    chunk_type="header_footer",
                    keep_for_corpus=parsed["keep_for_corpus"],
                    drop_reason=parsed["drop_reason"],
                    enriched_notes=parsed["enriched_notes"],
                    uncertainty_notes=parsed["uncertainty_notes"],
                    metadata={"confidence": parsed["confidence"], "parse_success": parsed["parse_success"]},
                )

                with self._progress_lock:
                    self._completed_chunks += 1
                    self._total_llm_calls += 1
                    self._total_tokens += response.total_tokens
                    self._processed_chunk_ids.add(chunk_id_key)
                    if parsed["keep_for_corpus"]:
                        self._corpus_chunks += 1
                    else:
                        self._dropped_chunks += 1

                # Write cleaned chunk
                self._threadsafe_append(str(self.output_files["cleaned"]), chunk.to_dict())

                # Write quality record for header/footer
                quality_record = {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "language": chunk.language.value,
                    "chunk_type": chunk.chunk_type,
                    "keep_for_corpus": chunk.keep_for_corpus,
                    "drop_reason": chunk.drop_reason,
                    "final_status": "passed" if chunk.keep_for_corpus else "dropped",
                    "verifier_model": response.model,
                    "prompt_version": PROMPT_VERSION,
                    "run_id": self.run_id,
                    "timestamp": time.time(),
                }
                self._threadsafe_append(str(self.output_files["quality"]), quality_record)

                return chunk
            except Exception as e:
                self._record_error(f"header_footer clean {source_file} p{page_number}: {e}")
                with self._progress_lock:
                    self._failed_chunks += 1
                return None

        # ── Body/formula/table/mixed chunk: full processing ──

        # Step 1: Clean chunk using fast model
        prompt = get_cleaning_prompt(doc_language, raw_text, chunk_type=chunk_type)
        try:
            response = self.pool.chat(
                messages=[
                    {"role": "system", "content": "You are a professional technical document cleaning specialist. Always respond with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=4096,
            )
            parsed = self._parse_v2_cleaning_response(response.content, raw_text)
            cleaning_model = response.model
        except Exception as e:
            self._record_error(f"clean {source_file} p{page_number}: {e}", chunk_data)
            with self._progress_lock:
                self._failed_chunks += 1
            return None

        # Check shutdown after cleaning
        if self._shutdown_requested:
            # Save what we have so far
            self._save_checkpoint()
            return None

        # Create CleanedChunk with v2.0 fields
        cleaned_text = parsed["cleaned_text"]
        if not cleaned_text.strip():
            cleaned_text = raw_text
            parsed["confidence"] = 0.3
            parsed["uncertainty_notes"] += " [fallback: empty cleaned_text, using raw text]"

        chunk = CleanedChunk(
            chunk_id=f"chunk_{content_hash_val}_{self._completed_chunks}",
            source_file=source_file,
            source_folder=chunk_data.get("source_folder", ""),
            page_numbers=[page_number],
            language=Language(doc_language),
            original_text=raw_text,
            cleaned_text=cleaned_text,
            original_content_hash=content_hash(raw_text),
            cleaned_content_hash=content_hash(cleaned_text),
            cleaning_model=cleaning_model,
            cleaning_prompt_version=PROMPT_VERSION,
            run_id=self.run_id,
            chunk_type=chunk_type,
            keep_for_corpus=parsed["keep_for_corpus"],
            removed_noise_types=parsed["removed_noise_types"],
            ocr_repairs=parsed["ocr_repairs"],
            technical_content_types=parsed["technical_content_types"],
            uncertainty_notes=parsed["uncertainty_notes"],
            drop_reason=parsed["drop_reason"],
            enriched_notes=parsed["enriched_notes"],
            metadata={
                "confidence": parsed["confidence"],
                "cleaning_actions": parsed["cleaning_actions"][:5],
                "parse_success": parsed["parse_success"],
            },
        )

        # Write cleaned chunk immediately
        self._threadsafe_append(str(self.output_files["cleaned"]), chunk.to_dict())

        # Step 2: Verify quality (only for keep_for_corpus body chunks)
        quality = None
        if not self.skip_quality and chunk.keep_for_corpus and chunk_type == "body":
            try:
                verify_prompt = get_quality_verification_prompt(
                    cleaned_text=chunk.cleaned_text,
                    original_text=chunk.original_text,
                )
                verify_response = self.pool.chat_quality(
                    messages=[
                        {"role": "system", "content": "You are an independent text quality verification expert. Always respond with valid JSON. You must be critical and objective."},
                        {"role": "user", "content": verify_prompt},
                    ],
                    max_completion_tokens=2048,
                )
                # Parse quality response
                quality = self._parse_quality_response(verify_response.content)
                quality.verification_model = verify_response.model
                quality.verification_timestamp = time.time()
                chunk.quality_score = quality

                # Write quality record
                quality_record = {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "source_folder": chunk.source_folder,
                    "page_numbers": chunk.page_numbers,
                    "language": chunk.language.value,
                    "clarity": quality.clarity,
                    "completeness": quality.completeness,
                    "consistency": quality.consistency,
                    "feasibility": quality.feasibility,
                    "complexity": quality.complexity,
                    "domain_relevance": quality.domain_relevance,
                    "average_score": quality.average,
                    "final_status": quality.verdict.value,
                    "keep_for_corpus": chunk.keep_for_corpus,
                    "enrichment_leakage": quality.issues if hasattr(quality, 'issues') else [],
                    "detected_issues": quality.issues,
                    "verifier_model": quality.verification_model,
                    "prompt_version": PROMPT_VERSION,
                    "run_id": self.run_id,
                    "timestamp": quality.verification_timestamp,
                }
                self._threadsafe_append(str(self.output_files["quality"]), quality_record)

                # Update corpus/dropped based on quality verdict
                if quality.verdict == QualityVerdict.FAILED:
                    chunk.keep_for_corpus = False
                    chunk.drop_reason = f"quality_failed: {','.join(quality.issues[:3])}"

            except Exception as e:
                self._record_error(f"verify {chunk.chunk_id}: {e}")
                quality = None

        # Step 3: Extract knowledge units (only for keep_for_corpus body chunks with decent quality)
        units = []
        if not self.skip_ku_sft and chunk.keep_for_corpus and chunk_type == "body":
            if quality and quality.verdict != QualityVerdict.FAILED:
                try:
                    # Use pool for KU extraction (any model)
                    ku_prompt = get_knowledge_extraction_prompt(chunk.cleaned_text)
                    ku_response = self.pool.chat(
                        messages=[
                            {"role": "system", "content": "You are a carbon fiber domain knowledge extraction expert. Always respond with valid JSON array. Only extract knowledge directly supported by the source text."},
                            {"role": "user", "content": ku_prompt},
                        ],
                        max_completion_tokens=4096,
                    )
                    # Parse and create knowledge units
                    units = self._parse_knowledge_units(ku_response.content, chunk)
                    for unit in units:
                        self._threadsafe_append(str(self.output_files["knowledge"]), unit.to_dict())
                except Exception as e:
                    self._record_error(f"knowledge_extract {chunk.chunk_id}: {e}")
                    units = []

        # Step 4: Generate SFT candidates (only for keep_for_corpus body chunks)
        candidates = []
        if not self.skip_ku_sft and chunk.keep_for_corpus and chunk_type == "body":
            if quality and quality.verdict != QualityVerdict.FAILED:
                try:
                    sft_prompt = get_sft_generation_prompt(chunk.cleaned_text)
                    sft_response = self.pool.chat(
                        messages=[
                            {"role": "system", "content": "You are a carbon fiber domain SFT sample generation expert. Always respond with valid JSON array. Only generate samples supported by the source text. No hallucination."},
                            {"role": "user", "content": sft_prompt},
                        ],
                        max_completion_tokens=4096,
                    )
                    candidates = self._parse_sft_candidates(sft_response.content, chunk)
                    for cand in candidates:
                        self._threadsafe_append(str(self.output_files["sft"]), cand.to_dict())
                except Exception as e:
                    self._record_error(f"sft_generate {chunk.chunk_id}: {e}")
                    candidates = []

        # Update progress counters
        with self._progress_lock:
            self._completed_chunks += 1
            llm_calls = 1  # cleaning
            if quality:
                llm_calls += 1  # verification
            if units:
                llm_calls += 1  # KU extraction
            if candidates:
                llm_calls += 1  # SFT generation
            self._total_llm_calls += llm_calls
            self._processed_chunk_ids.add(chunk_id_key)
            if chunk.keep_for_corpus:
                self._corpus_chunks += 1
            else:
                self._dropped_chunks += 1

        # Periodic checkpoint and progress report
        if self._completed_chunks % 10 == 0 or self._completed_chunks % 50 == 0:
            self._save_checkpoint()
            self._write_progress()

        return chunk

    def _parse_quality_response(self, response_text: str) -> QualityScore:
        """Parse quality verification response."""
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

        return QualityScore(
            clarity=0.5, completeness=0.5, consistency=0.5,
            feasibility=0.5, complexity=0.5, domain_relevance=0.5,
            verdict=QualityVerdict.NEEDS_REVISION,
            issues=["json_parse_failed"],
        )

    def _parse_knowledge_units(self, response_text: str, chunk: CleanedChunk) -> list:
        """Parse knowledge extraction response."""
        from src.autodata.pipelines.text_schema import KnowledgeUnit, KnowledgeType

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
                            source_refs=[chunk.source_file, chunk.source_folder],
                            extraction_model="pool",
                            run_id=self.run_id,
                        )
                        units.append(unit)
                    return units
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def _parse_sft_candidates(self, response_text: str, chunk: CleanedChunk) -> list:
        """Parse SFT candidate generation response."""
        from src.autodata.pipelines.text_schema import SFTCandidate, SFTTaskType, Difficulty

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
                            source_refs=[chunk.source_file, chunk.source_folder],
                            generation_model="pool",
                            run_id=self.run_id,
                        )
                        candidates.append(candidate)
                    return candidates
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def run(self) -> CleaningRunMetadata:
        """Execute the fast cleaning pipeline with concurrent workers.

        Steps:
        1. Load documents and generate noise report
        2. Create chunk work items from all documents
        3. Process chunks concurrently with ThreadPoolExecutor
        4. Build pretraining corpus from keep_for_corpus chunks
        5. Save metadata, checkpoint, and progress
        """
        logger.info(f"Starting fast pipeline: run_id={self.run_id}, workers={self.num_workers}, lang={self.language}")

        # Step 1: Load documents
        documents = self._load_documents()

        # Generate noise report
        noise_report = generate_noise_report(documents)
        atomic_write_json(str(self.output_files["noise"]), noise_report)

        # Step 2: Create chunk work items
        all_work_items = []
        total_chunks = 0
        for doc in documents:
            chunks = preprocess_document(doc)
            for chunk_data in chunks:
                chunk_data["language"] = doc.language.value
                all_work_items.append(chunk_data)
            total_chunks += len(chunks)
            logger.info(f"Document {doc.file_name}: {len(chunks)} chunks")

        self._total_chunks_est = min(total_chunks, self.max_chunks or total_chunks)
        logger.info(f"Total chunks to process: {self._total_chunks_est} (workers={self.num_workers})")

        # Limit work items if max_chunks specified
        if self.max_chunks and len(all_work_items) > self.max_chunks:
            all_work_items = all_work_items[:self.max_chunks]

        # Step 3: Process chunks concurrently
        all_cleaned = []
        completed_futures = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks
            future_to_chunk = {}
            for chunk_data in all_work_items:
                doc_lang = chunk_data.get("language", self.language)
                future = executor.submit(self._process_chunk, chunk_data, doc_lang)
                future_to_chunk[future] = chunk_data

            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_data = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        all_cleaned.append(result)
                    completed_futures += 1
                except Exception as e:
                    self._record_error(f"future exception: {e}", chunk_data)
                    with self._progress_lock:
                        self._failed_chunks += 1
                    completed_futures += 1

                # Progress update every 5 completed futures
                if completed_futures % 5 == 0:
                    self._write_progress()

        # Step 4: Build pretraining corpus
        self._build_pretraining_corpus(all_cleaned)

        # Step 5: Save final metadata
        pool_stats = self.pool.stats()
        metadata = CleaningRunMetadata(
            run_id=self.run_id,
            mode="full",
            model_name=f"ModelPool(fast={self.fast_model}, quality={self.quality_model})",
            model=f"ModelPool({self.num_workers} workers)",
            prompt_version=PROMPT_VERSION,
            language_filter=self.language,
            max_files=self.max_files,
            max_pages_per_file=self.max_pages_per_file,
            total_raw_files_seen=len(documents),
            total_files_processed=len(documents),
            total_raw_chunks=len(all_work_items),
            total_cleaned_chunks=self._corpus_chunks,
            total_chunks_created=self._completed_chunks,
            total_chunks_passed=self._corpus_chunks,
            total_chunks_failed=self._failed_chunks,
            total_knowledge_units=pool_stats["total_calls"],  # approximate
            total_llm_calls=self._total_llm_calls,
            total_tokens_used=pool_stats["total_tokens"],
            total_api_calls=pool_stats["total_calls"],
            end_time=time.time(),
            errors=[e["error"] for e in self._errors],
        )
        atomic_write_json(str(self.output_files["metadata"]), metadata.to_dict())

        # Final checkpoint and progress
        self._save_checkpoint()
        self._write_progress()

        logger.info(
            f"Pipeline complete: {self._completed_chunks} chunks processed, "
            f"{self._corpus_chunks} corpus, {self._dropped_chunks} dropped, "
            f"{self._failed_chunks} failed, {self._total_llm_calls} LLM calls"
        )

        return metadata

    def _load_documents(self) -> list:
        """Load raw documents based on language filter."""
        documents = []
        zh_dir = PROJECT_ROOT / "text_raw_data" / "books"
        en_dir = PROJECT_ROOT / "text_raw_data" / "en_books"

        zh_files = sorted(os.listdir(zh_dir))
        en_files = sorted(os.listdir(en_dir))

        zh_count = len(zh_files) if self.language != "en" else 0
        en_count = len(en_files) if self.language != "zh" else 0

        if self.max_files:
            zh_count = min(zh_count, self.max_files)
            en_count = min(en_count, self.max_files)

        for fname in zh_files[:zh_count]:
            path = str(zh_dir / fname)
            doc = load_raw_document(path, source_folder="books", max_pages=self.max_pages_per_file)
            documents.append(doc)

        for fname in en_files[:en_count]:
            path = str(en_dir / fname)
            doc = load_raw_document(path, source_folder="en_books", max_pages=self.max_pages_per_file)
            documents.append(doc)

        return documents

    def _build_pretraining_corpus(self, chunks: list[CleanedChunk]) -> None:
        """Build pretraining corpus from keep_for_corpus chunks only."""
        for chunk in chunks:
            if not chunk.keep_for_corpus:
                continue
            if chunk.chunk_type in ("empty",):
                continue
            if not chunk.cleaned_text.strip():
                continue

            record = {
                "text": chunk.cleaned_text,
                "enriched_notes": chunk.enriched_notes,
                "source_file": chunk.source_file,
                "source_folder": chunk.source_folder,
                "page_numbers": chunk.page_numbers,
                "language": chunk.language.value,
                "chunk_type": chunk.chunk_type,
                "keep_for_corpus": chunk.keep_for_corpus,
                "cleaning_model": chunk.cleaning_model,
                "prompt_version": chunk.cleaning_prompt_version,
                "run_id": chunk.run_id,
                "original_content_hash": chunk.original_content_hash,
                "cleaned_content_hash": chunk.cleaned_content_hash,
            }
            self._threadsafe_append(str(self.output_files["pretraining"]), record)