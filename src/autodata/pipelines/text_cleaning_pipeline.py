"""Resumable text cleaning pipeline with DTCG coordination.

Supports pilot/full mode, checkpointing, atomic writes,
rate-limit handling, and full provenance preservation.

Default behavior: pilot mode with 2 zh + 2 en files, 20 pages each.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from src.autodata.agents.data_cleaning_agent import DataCleaningAgent
from src.autodata.agents.planning_agent import CentralPlanningAgent
from src.autodata.agents.quality_verification_agent import QualityVerificationAgent
from src.autodata.context_graph.context_selector import ContextSelector
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
    TaskStatus,
)
from src.autodata.context_graph.message_store import MessageStore
from src.autodata.pipelines.knowledge_extractor import extract_knowledge_units
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
)
from src.autodata.utils.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    append_jsonl_record,
    ensure_dir,
    safe_read_jsonl,
)
from src.autodata.utils.logging_utils import get_logger, setup_logging, safe_serialize
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client, get_key2_client
from src.autodata.utils.progress_tracker import ProgressTracker

logger = get_logger("text_cleaning_pipeline")

# ── Project paths ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── Pipeline class ─────────────────────────────────────────────────

class TextCleaningPipeline:
    """Resumable text cleaning pipeline with DTCG coordination.

    Modes:
    - pilot: Small representative sample (2 zh + 2 en, 20 pages)
    - full: All files (requires explicit instruction)
    """

    def __init__(
        self,
        mode: str = "pilot",
        max_files: Optional[int] = None,
        max_pages_per_file: Optional[int] = None,
        language_filter: str = "all",
        skip_llm: bool = False,
        resume: bool = False,
        run_id: Optional[str] = None,
        enable_dtcg_trace: bool = True,
        enable_quality_persistence: bool = True,
        enable_context_package_dump: bool = True,
        enable_progress_monitor: bool = False,
        use_key2: bool = False,
        model_name: Optional[str] = None,
        skip_file_indices: Optional[list[int]] = None,
    ) -> None:
        self.mode = mode
        self.max_files = max_files or (4 if mode == "pilot" else None)
        self.max_pages_per_file = max_pages_per_file or (20 if mode == "pilot" else None)
        self.language_filter = language_filter
        self.skip_llm = skip_llm
        self.resume = resume
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        self.enable_dtcg_trace = enable_dtcg_trace
        self.enable_quality_persistence = enable_quality_persistence
        self.enable_context_package_dump = enable_context_package_dump
        self.enable_progress_monitor = enable_progress_monitor
        self.use_key2 = use_key2
        self.model_name_override = model_name
        self.skip_file_indices = skip_file_indices or []

        # Determine effective model name
        self.effective_model = model_name or ("mimo-v2.5" if use_key2 else "mimo-v2.5-pro")

        # Model client — use key2 if requested
        if use_key2:
            self.model_client = get_key2_client(model=model_name)
        else:
            if model_name:
                self.model_client = XiaomiModelClient(default_model=model_name)
            else:
                self.model_client = get_default_client()

        # DTCG components
        self.graph = DynamicTaskContextGraph()
        self.message_store = MessageStore()
        self.context_selector = ContextSelector()

        # Output paths (must be set before agents that need them)
        self._setup_output_paths()

        # Context package tracking
        self._context_packages: list[dict] = []

        # Agents — use effective model name
        self.planner = CentralPlanningAgent(
            model_client=self.model_client,
            model=self.effective_model,
            graph=self.graph,
            message_store=self.message_store,
            context_selector=self.context_selector,
        )
        self.cleaning_agent = DataCleaningAgent(
            model_client=self.model_client,
            model=self.effective_model,
            message_store=self.message_store,
            graph=self.graph,
            run_id=self.run_id,
        )
        self.quality_agent = QualityVerificationAgent(
            model_client=self.model_client,
            model=self.effective_model,
            message_store=self.message_store,
            graph=self.graph,
            output_path=str(self.output_files["quality"]),
            run_id=self.run_id,
        )

        # Run metadata
        self.metadata = CleaningRunMetadata(
            run_id=self.run_id,
            mode=mode,
            model_name=self.effective_model,
            model=self.effective_model,
            prompt_version="v1.0",
            language_filter=language_filter,
            max_files=self.max_files,
            max_pages_per_file=self.max_pages_per_file,
        )

        # Progress tracker
        self.progress_tracker: Optional[ProgressTracker] = None
        if self.enable_progress_monitor and "progress_json" in self.output_files:
            self.progress_tracker = ProgressTracker(
                run_id=self.run_id,
                json_path=str(self.output_files["progress_json"]),
                log_path=str(self.output_files["progress_log"]),
                language=language_filter,
                mode=mode,
            )

        # Error tracking for full runs
        self._errors: list[dict] = []

        # Checkpoint state
        self._processed_files: set[str] = set()
        if self.resume:
            self._load_checkpoint()

    def _setup_output_paths(self) -> None:
        """Create output directories and set file paths.

        For full runs with a specific run_id (e.g. phase_2_full_zh),
        use run_id-based paths under a dedicated report directory.
        For pilot runs, keep legacy pilot/full suffix paths.
        """
        base = PROJECT_ROOT

        # Determine if this is a run_id-based full run
        is_run_id_full = (
            self.mode == "full"
            and self.run_id
            and self.run_id.startswith("phase_2_full_")
        )

        if is_run_id_full:
            # Dedicated directory for this full run
            rid = self.run_id  # e.g. phase_2_full_zh
            self.output_dirs = {
                "cleaned": ensure_dir(base / "data" / "interim" / "text_cleaned"),
                "pretraining": ensure_dir(base / "data" / "processed" / "pretraining_corpus"),
                "knowledge": ensure_dir(base / "data" / "processed" / "knowledge_units"),
                "sft": ensure_dir(base / "data" / "processed" / "sft_candidates"),
                "quality": ensure_dir(base / "data" / "processed" / "text_quality"),
                "reports": ensure_dir(base / "data" / "reports" / "phase_2_full_text_cleaning"),
                "repair_reports": ensure_dir(base / "data" / "reports" / "phase_2_full_text_cleaning"),
            }
            self.output_files = {
                "cleaned": self.output_dirs["cleaned"] / f"cleaned_chunks_{rid}.jsonl",
                "pretraining": self.output_dirs["pretraining"] / f"pretraining_corpus_{rid}.jsonl",
                "knowledge": self.output_dirs["knowledge"] / f"knowledge_units_{rid}.jsonl",
                "sft": self.output_dirs["sft"] / f"sft_candidates_{rid}.jsonl",
                "quality": self.output_dirs["quality"] / f"text_quality_scores_{rid}.jsonl",
                "checkpoint": self.output_dirs["reports"] / f"{rid}_checkpoint.json",
                "context_packages": self.output_dirs["reports"] / f"{rid}_context_packages.jsonl",
                "errors": self.output_dirs["reports"] / f"{rid}_errors.jsonl",
                "progress_json": self.output_dirs["reports"] / f"{rid}_progress.json",
                "progress_log": self.output_dirs["reports"] / f"{rid}_progress.log",
                "metadata": self.output_dirs["reports"] / f"{rid}_run_metadata.json",
                "dtcg_trace": self.output_dirs["reports"] / f"{rid}_dtcg_trace.json",
            }
        else:
            # Legacy pilot/repair paths
            self.output_dirs = {
                "cleaned": ensure_dir(base / "data" / "interim" / "text_cleaned"),
                "pretraining": ensure_dir(base / "data" / "processed" / "pretraining_corpus"),
                "knowledge": ensure_dir(base / "data" / "processed" / "knowledge_units"),
                "sft": ensure_dir(base / "data" / "processed" / "sft_candidates"),
                "quality": ensure_dir(base / "data" / "processed" / "text_quality"),
                "reports": ensure_dir(base / "data" / "reports" / "phase_2_text_cleaning"),
                "repair_reports": ensure_dir(base / "data" / "reports" / "phase_2_text_cleaning_repair"),
            }
            suffix = "pilot" if self.mode == "pilot" else "full"
            self.output_files = {
                "cleaned": self.output_dirs["cleaned"] / f"cleaned_chunks_{suffix}.jsonl",
                "pretraining": self.output_dirs["pretraining"] / f"pretraining_corpus_{suffix}.jsonl",
                "knowledge": self.output_dirs["knowledge"] / f"knowledge_units_{suffix}.jsonl",
                "sft": self.output_dirs["sft"] / f"sft_candidates_{suffix}.jsonl",
                "quality": self.output_dirs["quality"] / f"text_quality_scores_{suffix}.jsonl",
                "checkpoint": self.output_dirs["reports"] / f"checkpoint_{suffix}.json",
                "context_packages": self.output_dirs["repair_reports"] / "context_packages_repaired.jsonl",
            }

    def _load_checkpoint(self) -> None:
        """Load checkpoint state for resume."""
        cp_path = self.output_files["checkpoint"]
        if cp_path.exists():
            data = safe_read_json(str(cp_path))
            if data and isinstance(data, dict):
                self._processed_files = set(data.get("processed_files", []))
                logger.info(f"Resumed from checkpoint: {len(self._processed_files)} files already processed")

    def _save_checkpoint(self) -> None:
        """Save checkpoint state."""
        atomic_write_json(
            str(self.output_files["checkpoint"]),
            {
                "run_id": self.run_id,
                "processed_files": list(self._processed_files),
                "timestamp": time.time(),
                "metadata": self.metadata.to_dict(),
            },
        )

    def _generate_context_package(
        self,
        agent_name: str,
        task_id: str,
        current_goal: str,
    ) -> dict[str, Any]:
        """Generate and record a context package for an agent invocation."""
        # Find agent node ID
        agent_node_id = None
        for node in self.graph.nodes.values():
            if node.node_type == NodeType.AGENT and node.name == agent_name:
                agent_node_id = node.node_id
                break

        if agent_node_id:
            ctx_pkg = self.context_selector.select_context(
                graph=self.graph,
                agent_node_id=agent_node_id,
                task_id=task_id,
                current_goal=current_goal,
                token_budget=8000,
            )
        else:
            ctx_pkg = ContextPackage(
                agent_name=agent_name,
                task_id=task_id,
                current_goal=current_goal,
            )

        # Compute broadcast vs DTCG token estimates
        all_node_ids = list(self.graph.nodes.keys())
        broadcast_tokens = sum(
            self.context_selector._estimate_token_cost(n)
            for n in self.graph.nodes.values()
        )
        dtcg_tokens = ctx_pkg.total_token_estimate
        saving_ratio = round(
            (1 - dtcg_tokens / max(broadcast_tokens, 1)) * 100, 1
        ) if broadcast_tokens > 0 else 0.0

        # Build record
        record = {
            "agent_name": ctx_pkg.agent_name,
            "task_id": ctx_pkg.task_id,
            "current_goal": ctx_pkg.current_goal,
            "selected_memory": ctx_pkg.selected_memory,
            "selected_artifacts": ctx_pkg.selected_artifacts,
            "selected_constraints": ctx_pkg.constraints,
            "selected_messages": [],
            "token_estimate": ctx_pkg.total_token_estimate,
            "selected_node_ids": [
                item.get("node_id", "") for item in ctx_pkg.selected_memory + ctx_pkg.selected_artifacts + ctx_pkg.constraints
            ],
            "omitted_node_count": len(all_node_ids) - len(ctx_pkg.selected_memory + ctx_pkg.selected_artifacts + ctx_pkg.constraints),
            "broadcast_token_estimate": broadcast_tokens,
            "dtcg_token_estimate": dtcg_tokens,
            "estimated_saving_ratio": saving_ratio,
            "timestamp": time.time(),
            "run_id": self.run_id,
        }
        self._context_packages.append(record)

        # Write to JSONL
        if self.enable_context_package_dump:
            append_jsonl_record(str(self.output_files["context_packages"]), record)

        return record

    def run(self) -> CleaningRunMetadata:
        """Execute the text cleaning pipeline.

        Steps:
        1. Create DTCG task nodes for the pipeline
        2. Load and preprocess raw documents
        3. Clean chunks using DataCleaningAgent
        4. Verify quality using QualityVerificationAgent
        5. Extract knowledge units
        6. Generate SFT candidates
        7. Build pretraining corpus
        8. Save checkpoint and metadata
        """
        logger.info(f"Starting pipeline: mode={self.mode}, run_id={self.run_id}")

        # Step 1: Create DTCG plan
        self._create_pipeline_tasks()

        # Step 2: Load raw documents
        documents = self._load_documents()
        self.metadata.total_raw_files_seen = len(documents)

        # Initialize progress tracker with totals
        if self.progress_tracker:
            total_pages_est = sum(d.page_count for d in documents)
            # Estimate chunks: ~1 chunk per 3000 chars, ~1500 chars per page
            total_chunks_est = int(total_pages_est * 1500 / 3000)
            self.progress_tracker.set_totals(
                total_files=len(documents),
                total_pages=total_pages_est,
                total_chunks=total_chunks_est,
            )
            self.progress_tracker.set_checkpoint_path(str(self.output_files.get("checkpoint", "")))
            self.progress_tracker.set_output_paths(
                {k: str(v) for k, v in self.output_files.items()}
            )
            self.progress_tracker.start()

        # Step 3: Generate noise report
        noise_report = generate_noise_report(documents)
        atomic_write_json(
            str(self.output_dirs["reports"] / "ocr_noise_analysis.json"),
            noise_report,
        )

        # Step 4: Process each document
        all_chunks = []
        for doc in documents:
            if doc.file_name in self._processed_files and self.resume:
                logger.info(f"Skipping already processed: {doc.file_name}")
                if self.progress_tracker:
                    self.progress_tracker.on_file_complete()
                    self.progress_tracker.on_pages_completed(doc.page_count)
                continue

            if self.progress_tracker:
                self.progress_tracker.on_file_start(doc.file_name)

            # Preprocess into chunks
            chunks = preprocess_document(doc)
            self.metadata.total_raw_chunks += len(chunks)
            logger.info(f"Document {doc.file_name}: {len(chunks)} chunks from {doc.page_count} pages")

            self.metadata.total_pages_processed += doc.page_count
            if self.progress_tracker:
                self.progress_tracker.on_pages_completed(doc.page_count)

            # Clean each chunk
            for chunk_data in chunks:
                chunk_data["language"] = doc.language.value

                # Create raw chunk artifact node in DTCG
                raw_artifact_id = f"art_raw_{chunk_data['content_hash'][:8]}"
                raw_node = Node(
                    node_id=raw_artifact_id,
                    node_type=NodeType.ARTIFACT,
                    name=f"Raw chunk from {chunk_data['source_file']} p.{chunk_data['page_number']}",
                    properties={
                        "source_file": chunk_data["source_file"],
                        "page_number": chunk_data["page_number"],
                        "chunk_type": chunk_data.get("chunk_type", "body"),
                        "content_hash": chunk_data["content_hash"],
                    },
                )
                self.graph.add_node(raw_node)

                if self.skip_llm:
                    # Dry-run: create chunk without LLM call
                    cleaned = CleanedChunk(
                        chunk_id=f"chunk_dry_{chunk_data['content_hash'][:8]}",
                        source_file=chunk_data["source_file"],
                        source_folder=chunk_data["source_folder"],
                        page_numbers=[chunk_data["page_number"]],
                        language=Language(chunk_data.get("language", "zh")),
                        original_text=chunk_data["chunk_text"],
                        cleaned_text=chunk_data["chunk_text"],  # unchanged in dry-run
                        original_content_hash=chunk_data["content_hash"],
                        cleaned_content_hash=chunk_data["content_hash"],
                        cleaning_model="dry-run",
                        cleaning_prompt_version="v1.0",
                        run_id=self.run_id,
                        chunk_type=chunk_data.get("chunk_type", "body"),
                    )
                else:
                    # Generate context package for cleaning agent
                    self._generate_context_package(
                        "DataCleaningAgent",
                        "task_clean",
                        f"Clean chunk from {chunk_data['source_file']} p.{chunk_data['page_number']}",
                    )
                    try:
                        cleaned = self.cleaning_agent.clean_chunk(chunk_data)
                    except Exception as e:
                        error_msg = f"clean_chunk {chunk_data['source_file']} p.{chunk_data['page_number']}: {e}"
                        logger.error(error_msg)
                        self._record_error(error_msg, chunk_data)
                        cleaned = None

                if cleaned:
                    all_chunks.append(cleaned)
                    self.metadata.total_chunks_created += 1
                    if self.progress_tracker:
                        self.progress_tracker.on_chunk_cleaned()
                        if not self.skip_llm:
                            self.progress_tracker.on_llm_call()

                    # Write cleaned chunk immediately for quality checking
                    append_jsonl_record(str(self.output_files["cleaned"]), cleaned.to_dict())

                    # Create cleaned chunk artifact node and derived-from edge
                    cleaned_artifact_id = f"art_cleaned_{cleaned.chunk_id}"
                    cleaned_node = Node(
                        node_id=cleaned_artifact_id,
                        node_type=NodeType.ARTIFACT,
                        name=f"Cleaned chunk {cleaned.chunk_id}",
                        properties={
                            "source_file": cleaned.source_file,
                            "page_numbers": cleaned.page_numbers,
                            "chunk_type": cleaned.chunk_type,
                            "language": cleaned.language.value,
                        },
                    )
                    self.graph.add_node(cleaned_node)
                    derived_edge = Edge.new(
                        source_id=cleaned_artifact_id,
                        target_id=raw_artifact_id,
                        edge_type=EdgeType.ARTIFACT_DERIVED_FROM,
                        relevance_score=0.9,
                        trust_score=0.7,
                    )
                    self.graph.add_edge(derived_edge)

                    # Step 5: Verify quality and persist quality score
                    if not self.skip_llm and cleaned.chunk_type == "body":
                        # Generate context package for quality verification agent
                        self._generate_context_package(
                            "QualityVerificationAgent",
                            "task_verify",
                            f"Verify quality of chunk {cleaned.chunk_id}",
                        )
                        try:
                            quality = self.quality_agent.verify_chunk(cleaned)
                        except Exception as e:
                            error_msg = f"verify_chunk {cleaned.chunk_id}: {e}"
                            logger.error(error_msg)
                            self._record_error(error_msg, {"chunk_id": cleaned.chunk_id})
                            quality = None

                        if quality:
                            self.metadata.total_quality_scores += 1
                            if self.progress_tracker:
                                self.progress_tracker.on_chunk_verified()
                                self.progress_tracker.on_quality_written()
                                self.progress_tracker.on_llm_call()
                            if quality.verdict.value == "passed":
                                self.metadata.total_chunks_passed += 1
                            elif quality.verdict.value == "needs_revision":
                                self.metadata.total_chunks_needs_revision += 1
                            else:
                                self.metadata.total_chunks_failed += 1
                            # Quality feedback edge: verifier → cleaned chunk
                            qf_edge = Edge.new(
                                source_id=self.quality_agent.graph_node_id,
                                target_id=cleaned_artifact_id,
                                edge_type=EdgeType.QUALITY_FEEDBACK,
                                trust_score=quality.average,
                                properties={
                                    "verdict": quality.verdict.value,
                                    "average_score": quality.average,
                                    "issues": quality.issues,
                                },
                            )
                            self.graph.add_edge(qf_edge)
                    else:
                        # Non-body or dry-run: write a default quality record
                        self.metadata.total_chunks_passed += 1
                        self.metadata.total_quality_scores += 1
                        if self.progress_tracker:
                            self.progress_tracker.on_chunk_verified()
                            self.progress_tracker.on_quality_written()
                        default_quality_record = {
                            "chunk_id": cleaned.chunk_id,
                            "source_file": cleaned.source_file,
                            "source_folder": cleaned.source_folder,
                            "page_numbers": cleaned.page_numbers,
                            "language": cleaned.language.value,
                            "clarity": 1.0 if cleaned.chunk_type == "header_footer" else 0.5,
                            "completeness": 1.0 if cleaned.chunk_type == "header_footer" else 0.5,
                            "consistency": 1.0,
                            "feasibility": 0.5,
                            "complexity": 0.1,
                            "domain_relevance": 0.5,
                            "average_score": 0.5,
                            "final_status": "passed",
                            "detected_issues": [],
                            "verifier_model": "dry-run" if self.skip_llm else self.effective_model,
                            "prompt_version": "v1.0",
                            "run_id": self.run_id,
                            "timestamp": time.time(),
                        }
                        append_jsonl_record(
                            str(self.output_files["quality"]),
                            default_quality_record,
                        )

                    # Step 6: Extract knowledge units
                    if not self.skip_llm and cleaned.chunk_type == "body":
                        try:
                            units = extract_knowledge_units(
                                cleaned,
                                model_client=self.model_client,
                                output_path=str(self.output_files["knowledge"]),
                                run_id=self.run_id,
                            )
                        except Exception as e:
                            error_msg = f"extract_knowledge {cleaned.chunk_id}: {e}"
                            logger.error(error_msg)
                            self._record_error(error_msg, {"chunk_id": cleaned.chunk_id})
                            units = []

                        self.metadata.total_knowledge_units += len(units)
                        if self.progress_tracker:
                            self.progress_tracker.on_knowledge_units(len(units))
                            if units:
                                self.progress_tracker.on_llm_call()
                        # Create KU artifact nodes and derived-from edges
                        for unit in units:
                            ku_artifact_id = f"art_ku_{unit.unit_id}"
                            ku_node = Node(
                                node_id=ku_artifact_id,
                                node_type=NodeType.ARTIFACT,
                                name=f"Knowledge unit {unit.unit_id}: {unit.topic}",
                                properties={
                                    "knowledge_type": unit.knowledge_type.value,
                                    "topic": unit.topic,
                                    "claim": unit.claim[:80],
                                },
                            )
                            self.graph.add_node(ku_node)
                            ku_derived = Edge.new(
                                source_id=ku_artifact_id,
                                target_id=cleaned_artifact_id,
                                edge_type=EdgeType.ARTIFACT_DERIVED_FROM,
                                relevance_score=0.8,
                            )
                            self.graph.add_edge(ku_derived)

                    # Step 7: Generate SFT candidates
                    if not self.skip_llm and cleaned.chunk_type == "body":
                        try:
                            candidates = generate_sft_candidates(
                                cleaned,
                                model_client=self.model_client,
                                output_path=str(self.output_files["sft"]),
                                run_id=self.run_id,
                            )
                        except Exception as e:
                            error_msg = f"generate_sft {cleaned.chunk_id}: {e}"
                            logger.error(error_msg)
                            self._record_error(error_msg, {"chunk_id": cleaned.chunk_id})
                            candidates = []

                        self.metadata.total_sft_candidates += len(candidates)
                        if self.progress_tracker:
                            self.progress_tracker.on_sft_candidates(len(candidates))
                            if candidates:
                                self.progress_tracker.on_llm_call()
                        # Create SFT artifact nodes and derived-from edges
                        for cand in candidates:
                            sft_artifact_id = f"art_sft_{cand.sample_id}"
                            sft_node = Node(
                                node_id=sft_artifact_id,
                                node_type=NodeType.ARTIFACT,
                                name=f"SFT candidate {cand.sample_id}: {cand.task_type.value}",
                                properties={
                                    "task_type": cand.task_type.value,
                                    "difficulty": cand.difficulty.value,
                                    "instruction": cand.instruction[:80],
                                },
                            )
                            self.graph.add_node(sft_node)
                            sft_derived = Edge.new(
                                source_id=sft_artifact_id,
                                target_id=cleaned_artifact_id,
                                edge_type=EdgeType.ARTIFACT_DERIVED_FROM,
                                relevance_score=0.7,
                            )
                            self.graph.add_edge(sft_derived)

                    # Progress report after each chunk
                    if self.progress_tracker:
                        self.progress_tracker.maybe_report()
                        # Print progress to terminal every report interval
                        if self.enable_progress_monitor:
                            print(self.progress_tracker.report(), flush=True)

            # Mark file as processed
            self._processed_files.add(doc.file_name)
            self.metadata.total_files_processed += 1
            if self.progress_tracker:
                self.progress_tracker.on_file_complete()
            self._save_checkpoint()
            if self.enable_progress_monitor and self.progress_tracker:
                print(self.progress_tracker.report(), flush=True)

        # Step 8: Build pretraining corpus
        self.metadata.total_cleaned_chunks = len(all_chunks)
        self._build_pretraining_corpus(all_chunks)

        # Step 9: Save DTCG trace
        self._save_dtcg_trace()

        # Finalize metadata
        self.metadata.end_time = time.time()
        self.metadata.total_tokens_used = (
            self.model_client.total_tokens_used
        )
        self.metadata.total_api_calls = (
            self.model_client.call_count
        )
        self.metadata.total_llm_calls = (
            self.model_client.call_count
        )
        # Update progress tracker with final LLM/tokens
        if self.progress_tracker:
            # Sync final token count
            self.progress_tracker._tokens_used = self.metadata.total_tokens_used
            self.progress_tracker._llm_calls = self.metadata.total_llm_calls
            self.progress_tracker.finish()

        # Save metadata to appropriate paths
        self._save_metadata()

        logger.info(
            f"Pipeline complete: {self.metadata.total_files_processed} files, "
            f"{self.metadata.total_chunks_created} chunks, "
            f"{self.metadata.total_knowledge_units} knowledge units, "
            f"{self.metadata.total_sft_candidates} SFT candidates"
        )

        return self.metadata

    def _create_pipeline_tasks(self) -> None:
        """Create DTCG task nodes, edges, constraint/memory/tool nodes."""
        # Task nodes
        tasks = [
            ("task_preprocess", "Preprocess raw text pages"),
            ("task_clean", "Clean text chunks using Xiaomi LLM"),
            ("task_verify", "Verify quality of cleaned chunks"),
            ("task_extract_ku", "Extract knowledge units"),
            ("task_generate_sft", "Generate SFT candidates"),
            ("task_build_corpus", "Build pretraining corpus"),
            ("task_summarize", "Summarize results and generate report"),
        ]
        for task_id, desc in tasks:
            task_node = Node(
                node_id=task_id,
                node_type=NodeType.TASK,
                name=desc,
                properties={"status": "pending", "assigned_agent": ""},
            )
            self.graph.add_node(task_node)

        # Task dependency edges (sequential pipeline)
        task_ids = [t[0] for t in tasks]
        for i in range(len(task_ids) - 1):
            dep_edge = Edge.new(
                source_id=task_ids[i + 1],
                target_id=task_ids[i],
                edge_type=EdgeType.TASK_DEPENDENCY,
                dependency_score=0.8,
            )
            self.graph.add_edge(dep_edge)

        # Agent assignment edges
        agent_assignments = [
            (self.planner.graph_node_id, "task_preprocess", 0.9),
            (self.cleaning_agent.graph_node_id, "task_clean", 1.0),
            (self.quality_agent.graph_node_id, "task_verify", 1.0),
            (self.cleaning_agent.graph_node_id, "task_extract_ku", 0.7),
            (self.cleaning_agent.graph_node_id, "task_generate_sft", 0.7),
            (self.planner.graph_node_id, "task_build_corpus", 0.8),
            (self.planner.graph_node_id, "task_summarize", 0.9),
        ]
        for agent_id, task_id, strength in agent_assignments:
            assign_edge = Edge.new(
                source_id=agent_id,
                target_id=task_id,
                edge_type=EdgeType.AGENT_ASSIGNMENT,
                relevance_score=strength,
                dependency_score=strength,
            )
            self.graph.add_edge(assign_edge)

        # Constraint nodes
        constraint_nodes = [
            ("constraint_no_hallucination", "No hallucinated facts allowed in cleaned text", "quality"),
            ("constraint_preserve_provenance", "Every output must preserve source provenance", "provenance"),
            ("constraint_preserve_formula", "Formulas must not be simplified or removed", "domain"),
            ("constraint_preserve_table", "Tables must be preserved or marked table_uncertain", "domain"),
            ("constraint_preserve_units", "Units, numbers, equations must be preserved exactly", "domain"),
            ("constraint_no_raw_overwrite", "Raw data files must never be overwritten", "safety"),
            ("constraint_quality_threshold", "Passed chunks need average score >= 0.6", "quality"),
        ]
        for cid, desc, category in constraint_nodes:
            c_node = Node(
                node_id=cid,
                node_type=NodeType.CONSTRAINT,
                name=desc,
                properties={"category": category, "active": True},
            )
            self.graph.add_node(c_node)
            # Connect constraints to relevant tasks
            relevant_tasks = {
                "quality": ["task_clean", "task_verify"],
                "provenance": ["task_clean", "task_build_corpus"],
                "domain": ["task_clean"],
                "safety": ["task_preprocess"],
            }
            for tid in relevant_tasks.get(category, []):
                c_edge = Edge.new(
                    source_id=cid,
                    target_id=tid,
                    edge_type=EdgeType.CONTEXT_RELEVANCE,
                    relevance_score=0.9,
                )
                self.graph.add_edge(c_edge)

        # Memory nodes for phase summary
        mem_node = Node(
            node_id="mem_phase_2_inventory",
            node_type=NodeType.MEMORY,
            name="Phase 2 raw text inventory summary",
            properties={"phase": "phase_2", "type": "inventory"},
        )
        self.graph.add_node(mem_node)

        # Tool nodes
        tool_nodes = [
            ("tool_xiaomi_llm", "Xiaomi MiMo LLM API caller", "mimo-v2.5-pro"),
            ("tool_text_cleaner", "OCR text cleaning prompt formatter", "prompt"),
            ("tool_quality_verifier", "Quality verification prompt formatter", "prompt"),
            ("tool_knowledge_extractor", "Knowledge extraction prompt formatter", "prompt"),
            ("tool_sft_generator", "SFT sample generation prompt formatter", "prompt"),
            ("tool_jsonl_writer", "JSONL file writer", "io"),
            ("tool_noise_analyzer", "OCR noise pattern detector", "preprocess"),
        ]
        for tid, desc, tool_type in tool_nodes:
            t_node = Node(
                node_id=tid,
                node_type=NodeType.TOOL,
                name=desc,
                properties={"tool_type": tool_type},
            )
            self.graph.add_node(t_node)
            # Tool usage edges to agents
            agent_tid_map = {
                "tool_xiaomi_llm": [self.cleaning_agent.graph_node_id, self.quality_agent.graph_node_id, self.planner.graph_node_id],
                "tool_text_cleaner": [self.cleaning_agent.graph_node_id],
                "tool_quality_verifier": [self.quality_agent.graph_node_id],
                "tool_knowledge_extractor": [self.cleaning_agent.graph_node_id],
                "tool_sft_generator": [self.cleaning_agent.graph_node_id],
                "tool_jsonl_writer": [self.planner.graph_node_id],
                "tool_noise_analyzer": [self.planner.graph_node_id],
            }
            for a_id in agent_tid_map.get(tid, []):
                tu_edge = Edge.new(
                    source_id=a_id,
                    target_id=tid,
                    edge_type=EdgeType.TOOL_USAGE,
                    relevance_score=0.7,
                )
                self.graph.add_edge(tu_edge)

    def _load_documents(self) -> list:
        """Load raw documents based on mode and language filter.

        Skips files at indices specified in self.skip_file_indices.
        """
        documents = []
        zh_dir = PROJECT_ROOT / "text_raw_data" / "books"
        en_dir = PROJECT_ROOT / "text_raw_data" / "en_books"

        # Select files based on mode and language filter
        zh_files = sorted(os.listdir(zh_dir))
        en_files = sorted(os.listdir(en_dir))

        if self.mode == "pilot":
            zh_count = 2
            en_count = 2
        else:
            zh_count = len(zh_files)
            en_count = len(en_files)

        if self.language_filter == "zh":
            en_count = 0
        elif self.language_filter == "en":
            zh_count = 0

        if self.max_files:
            zh_count = min(zh_count, self.max_files)
            en_count = min(en_count, self.max_files)

        # Skip specified file indices (for parallel processing)
        skip_zh = set(self.skip_file_indices) if self.language_filter == "zh" else set()
        skip_en = set(self.skip_file_indices) if self.language_filter == "en" else set()

        # Load Chinese books (skip specified indices)
        for idx, fname in enumerate(zh_files[:zh_count]):
            if idx in skip_zh:
                logger.info(f"Skipping file index {idx}: {fname} (handled by other process)")
                continue
            path = str(zh_dir / fname)
            doc = load_raw_document(
                path,
                source_folder="books",
                max_pages=self.max_pages_per_file,
            )
            documents.append(doc)

        # Load English books (skip specified indices)
        for idx, fname in enumerate(en_files[:en_count]):
            if idx in skip_en:
                logger.info(f"Skipping file index {idx}: {fname} (handled by other process)")
                continue
            path = str(en_dir / fname)
            doc = load_raw_document(
                path,
                source_folder="en_books",
                max_pages=self.max_pages_per_file,
            )
            documents.append(doc)

        return documents

    def _record_error(self, error_msg: str, context: dict | None = None) -> None:
        """Record an error and append to error JSONL if available."""
        if self.progress_tracker:
            self.progress_tracker.on_error(error_msg)
        error_record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "error": error_msg,
            "context": context or {},
        }
        self._errors.append(error_record)
        if "errors" in self.output_files:
            append_jsonl_record(str(self.output_files["errors"]), error_record)

    def _save_metadata(self) -> None:
        """Save run metadata to all appropriate paths."""
        metadata_dict = self.metadata.to_dict()

        # Always save to standard reports dir
        atomic_write_json(
            str(self.output_dirs["reports"] / "phase_2_run_metadata.json"),
            metadata_dict,
        )
        # Save to repair reports if it's a different dir
        repair_meta = self.output_dirs.get("repair_reports")
        if repair_meta and repair_meta != self.output_dirs["reports"]:
            atomic_write_json(
                str(repair_meta / "phase_2_repair_run_metadata.json"),
                metadata_dict,
            )
        # Save to run_id-specific path if available
        if "metadata" in self.output_files:
            atomic_write_json(
                str(self.output_files["metadata"]),
                metadata_dict,
            )

    def _build_pretraining_corpus(self, chunks: list[CleanedChunk]) -> None:
        """Build pretraining corpus from cleaned chunks."""
        for chunk in chunks:
            # Only include passed/medium+ quality body chunks
            if chunk.chunk_type in ("empty", "header_footer"):
                continue

            record = {
                "text": chunk.cleaned_text,
                "source_file": chunk.source_file,
                "source_folder": chunk.source_folder,
                "page_numbers": chunk.page_numbers,
                "language": chunk.language.value,
                "original_content_hash": chunk.original_content_hash,
                "cleaned_content_hash": chunk.cleaned_content_hash,
                "cleaning_model": chunk.cleaning_model,
                "run_id": chunk.run_id,
            }
            append_jsonl_record(str(self.output_files["pretraining"]), record)

    def _save_dtcg_trace(self) -> None:
        """Save DTCG graph state and statistics."""
        # Save graph serialization
        atomic_write_json(
            str(self.output_dirs["reports"] / "dtcg_text_cleaning_trace.json"),
            self.graph.to_dict(),
        )
        # Also save to repair reports directory if different
        repair_dir = self.output_dirs.get("repair_reports")
        if repair_dir and repair_dir != self.output_dirs["reports"]:
            atomic_write_json(
                str(repair_dir / "dtcg_text_cleaning_trace_repaired.json"),
                self.graph.to_dict(),
            )
        # Save to run_id-specific DTCG trace path if available
        if "dtcg_trace" in self.output_files:
            atomic_write_json(
                str(self.output_files["dtcg_trace"]),
                self.graph.to_dict(),
            )

        # Compute context savings vs broadcast
        # Broadcast estimate: all agents see all messages
        total_msgs = self.message_store.count()
        avg_msg_tokens = 200  # estimate
        num_agents = 3  # planner + cleaning + quality

        broadcast_tokens = total_msgs * avg_msg_tokens * num_agents
        dtcg_tokens = 0
        for agent_name in ["CentralPlanningAgent", "DataCleaningAgent", "QualityVerificationAgent"]:
            msgs = self.message_store.get_for_agent(agent_name)
            dtcg_tokens += sum(m.token_estimate for m in msgs)

        savings = {
            "total_nodes": self.graph.node_count,
            "total_edges": self.graph.edge_count,
            "total_messages": total_msgs,
            "broadcast_token_estimate": broadcast_tokens,
            "dtcg_context_tokens": dtcg_tokens,
            "savings_ratio": round(broadcast_tokens / max(dtcg_tokens, 1), 1),
            "context_packages_generated": 0,
        }
        atomic_write_json(
            str(self.output_dirs["reports"] / "dtcg_savings_analysis.json"),
            savings,
        )