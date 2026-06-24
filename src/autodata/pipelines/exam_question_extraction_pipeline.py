"""Exam question extraction pipeline for Phase 4.

Multi-stage pipeline following FullImageLabelingPipeline pattern.
Uses API_KEY1 only via ModelPool.

Stages:
1. Document conversion + OCR
2. Text cleaning + question extraction (LLM)
3. Quality verification (LLM)
4. Deduplication + normalization
"""

from __future__ import annotations

import json
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from src.autodata.utils.logging_utils import get_logger

logger = get_logger("exam_pipeline")


class ExamQuestionExtractionPipeline:
    """Multi-stage exam question extraction pipeline.

    Uses API_KEY1 only. Follows FullImageLabelingPipeline pattern
    with checkpointing, WriterQueue, and progress tracking.
    """

    def __init__(
        self,
        exam_dir: Path,
        output_dir: Path,
        report_dir: Path,
        run_id: str = "phase_4_exam_extraction",
        max_workers: int = 16,
        extraction_workers: int = 8,
        quality_workers: int = 8,
        use_key2: bool = False,
    ) -> None:
        self.exam_dir = exam_dir
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.run_id = run_id
        self.max_workers = max_workers
        self.extraction_workers = extraction_workers
        self.quality_workers = quality_workers
        self.use_key2 = use_key2

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        self.blocks_path = output_dir / "exam_text_blocks.jsonl"
        self.raw_questions_path = output_dir / "exam_questions_raw.jsonl"
        self.validated_path = output_dir / "exam_questions_validated.jsonl"
        self.quality_path = output_dir / "exam_question_quality_scores.jsonl"
        self.duplicates_path = output_dir / "exam_question_duplicates.jsonl"
        self.unique_path = output_dir / "exam_questions_unique.jsonl"
        self.benchmark_ready_path = output_dir / "exam_questions_benchmark_ready_candidates.jsonl"
        self.failures_path = output_dir / "exam_extraction_failures.jsonl"

        # Checkpoint paths
        self.stage1_checkpoint = report_dir / "stage1_checkpoint.json"
        self.stage2_checkpoint = report_dir / "stage2_checkpoint.json"
        self.stage3_checkpoint = report_dir / "stage3_checkpoint.json"

        # Progress paths
        self.progress_json = report_dir / "progress_exam_extraction.json"
        self.progress_log = report_dir / "progress_exam_extraction.log"

        # Model pool (API_KEY1 only)
        self._pool = None

        # Shutdown flag
        self._shutdown = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown requested, saving checkpoint...")
        self._shutdown = True

    @property
    def pool(self):
        """Lazy-load ModelPool (API_KEY1 only)."""
        if self._pool is None:
            from src.autodata.utils.model_pool import get_model_pool
            self._pool = get_model_pool(use_key2=self.use_key2)
        return self._pool

    def load_checkpoint(self, checkpoint_path: Path) -> set:
        """Load processed IDs from checkpoint."""
        if not checkpoint_path.exists():
            return set()
        with open(checkpoint_path) as f:
            data = json.load(f)
            return set(data.get("processed_ids", []))

    def save_checkpoint(self, checkpoint_path: Path, processed_ids: set) -> None:
        """Save checkpoint atomically."""
        tmp_path = checkpoint_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump({
                "processed_ids": list(processed_ids),
                "timestamp": time.time(),
                "total_processed": len(processed_ids),
                "run_id": self.run_id,
            }, f)
        os.replace(tmp_path, checkpoint_path)

    def _update_progress(self, **kwargs) -> None:
        """Update progress JSON and log."""
        progress = {}
        if self.progress_json.exists():
            with open(self.progress_json) as f:
                progress = json.load(f)

        progress.update(kwargs)
        progress["timestamp"] = time.time()

        with open(self.progress_json, "w") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

        # Append to log
        log_line = (
            f"[{self.run_id}] "
            f"files {progress.get('completed_files', 0)}/{progress.get('total_files', 0)} | "
            f"text_blocks {progress.get('extracted_text_blocks', 0)} | "
            f"questions_raw {progress.get('raw_questions', 0)} | "
            f"validated {progress.get('validated_questions', 0)} | "
            f"keep {progress.get('keep_count', 0)} | "
            f"review {progress.get('review_count', 0)} | "
            f"drop {progress.get('drop_count', 0)} | "
            f"workers {progress.get('active_workers', 0)} | "
            f"elapsed {progress.get('elapsed_formatted', '00:00:00')} | "
            f"current_stage={progress.get('current_stage', 'unknown')}"
        )
        with open(self.progress_log, "a") as f:
            f.write(log_line + "\n")

    def run(self, start_stage: int = 1, max_files: Optional[int] = None) -> dict:
        """Run the full pipeline with DTCG integration."""
        from src.autodata.context_graph.pipeline_dtcg_integration import PipelineDTCG

        # Initialize DTCG runtime trace
        dtcg = PipelineDTCG("phase_4_exam_extraction", self.report_dir)
        agent_extract_id = dtcg.add_agent("ExamExtractionAgent", role="question_extraction")
        agent_quality_id = dtcg.add_agent("ExamQualityAgent", role="quality_verification")
        tool_pool_id = dtcg.add_tool("ModelPool", api="xiaomi0")
        tool_ocr_id = dtcg.add_tool("OCRTool", engine="tesseract")
        constraint_id = dtcg.add_constraint("no_hallucinate_answers")

        results = {"run_id": self.run_id, "start_stage": start_stage}
        start_time = time.time()

        if start_stage <= 1:
            logger.info("Stage 1: Document conversion + OCR")
            task1_id = dtcg.add_task("document_conversion", status="in_progress")
            dtcg.connect_tool_usage(tool_ocr_id, task1_id)
            results["stage_1"] = self.run_stage_1(max_files=max_files)
            art1_id = dtcg.add_artifact("exam_text_blocks.jsonl", path=str(self.blocks_path))
            dtcg.connect_artifact_derived(art1_id, task1_id)

        if start_stage <= 2 and not self._shutdown:
            logger.info("Stage 2: Question extraction")
            task2_id = dtcg.add_task("question_extraction", status="in_progress")
            dtcg.connect_agent_to_task(agent_extract_id, task2_id)
            dtcg.connect_tool_usage(tool_pool_id, task2_id)
            dtcg.connect_quality_feedback(constraint_id, task2_id)
            if start_stage <= 1:
                dtcg.connect_task_dependency(task1_id, task2_id)
            results["stage_2"] = self.run_stage_2()
            art2_id = dtcg.add_artifact("exam_questions_raw.jsonl", path=str(self.raw_questions_path))
            dtcg.connect_artifact_derived(art2_id, task2_id)

        if start_stage <= 3 and not self._shutdown:
            logger.info("Stage 3: Quality verification")
            task3_id = dtcg.add_task("quality_verification", status="in_progress")
            dtcg.connect_agent_to_task(agent_quality_id, task3_id)
            dtcg.connect_tool_usage(tool_pool_id, task3_id)
            if start_stage <= 2:
                dtcg.connect_task_dependency(task2_id, task3_id)
            results["stage_3"] = self.run_stage_3()
            art3_id = dtcg.add_artifact("exam_questions_validated.jsonl", path=str(self.validated_path))
            dtcg.connect_artifact_derived(art3_id, task3_id)

        if start_stage <= 4 and not self._shutdown:
            logger.info("Stage 4: Deduplication + normalization")
            task4_id = dtcg.add_task("dedup_normalization", status="in_progress")
            if start_stage <= 3:
                dtcg.connect_task_dependency(task3_id, task4_id)
            results["stage_4"] = self.run_stage_4()
            art4_id = dtcg.add_artifact("exam_questions_unique.jsonl", path=str(self.unique_path))
            dtcg.connect_artifact_derived(art4_id, task4_id)

        # Save DTCG runtime trace
        dtcg.save()

        results["total_elapsed_seconds"] = time.time() - start_time
        results["total_elapsed_formatted"] = self._format_time(results["total_elapsed_seconds"])

        # Save run metadata
        metadata_path = self.report_dir / "run_metadata_exam_extraction.json"
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def run_stage_1(self, max_files: Optional[int] = None) -> dict:
        """Stage 1: Document conversion + OCR."""
        from src.autodata.tools.document_converter import convert_document

        self._update_progress(current_stage="document_conversion", start_time=time.time())

        files = sorted([f for f in self.exam_dir.iterdir()
                       if f.is_file() and not f.name.startswith(".")])
        if max_files:
            files = files[:max_files]

        total_files = len(files)
        processed_ids = self.load_checkpoint(self.stage1_checkpoint)
        remaining = [f for f in files if f.name not in processed_ids]

        logger.info(f"Stage 1: {total_files} total, {len(processed_ids)} processed, {len(remaining)} remaining")

        total_blocks = 0
        total_errors = 0

        with open(self.blocks_path, "a") as blocks_f, \
             open(self.failures_path, "a") as failures_f:

            for i, file_path in enumerate(remaining):
                if self._shutdown:
                    break

                blocks = convert_document(file_path)

                for block in blocks:
                    blocks_f.write(json.dumps(block, ensure_ascii=False) + "\n")
                    total_blocks += 1

                    if block.get("extraction_method") == "error":
                        failures_f.write(json.dumps({
                            "file": file_path.name,
                            "error": block.get("text", ""),
                            "stage": "document_conversion",
                        }, ensure_ascii=False) + "\n")
                        total_errors += 1

                processed_ids.add(file_path.name)

                # Checkpoint periodically
                if (i + 1) % 5 == 0:
                    self.save_checkpoint(self.stage1_checkpoint, processed_ids)
                    self._update_progress(
                        completed_files=len(processed_ids),
                        total_files=total_files,
                        extracted_text_blocks=total_blocks,
                        failed_blocks=total_errors,
                    )

            # Final checkpoint
            self.save_checkpoint(self.stage1_checkpoint, processed_ids)

        self._update_progress(
            completed_files=len(processed_ids),
            total_files=total_files,
            extracted_text_blocks=total_blocks,
            failed_blocks=total_errors,
            current_stage="document_conversion_complete",
        )

        return {
            "total_files": total_files,
            "completed_files": len(processed_ids),
            "total_blocks": total_blocks,
            "total_errors": total_errors,
        }

    def run_stage_2(self) -> dict:
        """Stage 2: Question extraction using LLM."""
        from src.autodata.agents.exam_extraction_agent import ExamExtractionAgent
        from src.autodata.utils.single_key_concurrency_controller import SingleKeyConcurrencyController

        self._update_progress(current_stage="question_extraction", start_time=time.time())

        # Load text blocks grouped by source file
        file_blocks = {}
        if self.blocks_path.exists():
            with open(self.blocks_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    block = json.loads(line)
                    source = block.get("source_file", "unknown")
                    if source not in file_blocks:
                        file_blocks[source] = []
                    file_blocks[source].append(block)

        processed_ids = self.load_checkpoint(self.stage2_checkpoint)
        remaining_files = [f for f in file_blocks.keys() if f not in processed_ids]

        logger.info(f"Stage 2: {len(file_blocks)} files, {len(processed_ids)} processed, {len(remaining_files)} remaining")

        agent = ExamExtractionAgent(pool=self.pool, run_id=self.run_id)
        controller = SingleKeyConcurrencyController(
            initial_workers=self.extraction_workers,
            max_workers=self.max_workers,
        )
        controller.set_stage("question_extraction")

        total_questions = 0
        start_time = time.time()

        with open(self.raw_questions_path, "a") as raw_f, \
             open(self.failures_path, "a") as failures_f:

            with ThreadPoolExecutor(max_workers=controller.current_workers()) as executor:
                futures = {}
                for source_file in remaining_files:
                    if self._shutdown:
                        break
                    future = executor.submit(
                        agent.extract_questions,
                        file_blocks[source_file],
                        source_file,
                    )
                    futures[future] = source_file

                for future in as_completed(futures):
                    source_file = futures[future]
                    try:
                        questions = future.result()
                        for q in questions:
                            raw_f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
                            total_questions += 1
                        controller.record_success()
                        processed_ids.add(source_file)
                    except Exception as e:
                        controller.record_error()
                        failures_f.write(json.dumps({
                            "file": source_file,
                            "error": str(e)[:200],
                            "stage": "question_extraction",
                        }, ensure_ascii=False) + "\n")

                    controller.maybe_adjust()

                    # Checkpoint periodically
                    if len(processed_ids) % 5 == 0:
                        self.save_checkpoint(self.stage2_checkpoint, processed_ids)

            # Final checkpoint
            self.save_checkpoint(self.stage2_checkpoint, processed_ids)

        elapsed = time.time() - start_time
        self._update_progress(
            raw_questions=total_questions,
            active_workers=controller.current_workers(),
            current_stage="question_extraction_complete",
            elapsed_formatted=self._format_time(elapsed),
        )

        return {
            "total_files": len(file_blocks),
            "completed_files": len(processed_ids),
            "total_questions": total_questions,
            "elapsed_seconds": elapsed,
            "controller_stats": controller.report(),
        }

    def run_stage_3(self) -> dict:
        """Stage 3: Quality verification using LLM."""
        from src.autodata.agents.exam_quality_agent import ExamQualityAgent
        from src.autodata.utils.single_key_concurrency_controller import SingleKeyConcurrencyController

        self._update_progress(current_stage="quality_verification", start_time=time.time())

        # Load raw questions
        questions = []
        if self.raw_questions_path.exists():
            with open(self.raw_questions_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        questions.append(json.loads(line))

        processed_ids = self.load_checkpoint(self.stage3_checkpoint)
        remaining = [q for q in questions if q.get("question_id", "") not in processed_ids]

        logger.info(f"Stage 3: {len(questions)} questions, {len(processed_ids)} processed, {len(remaining)} remaining")

        agent = ExamQualityAgent(pool=self.pool, run_id=self.run_id)
        controller = SingleKeyConcurrencyController(
            initial_workers=self.quality_workers,
            max_workers=self.max_workers,
        )
        controller.set_stage("quality_verification")

        keep_count = 0
        review_count = 0
        drop_count = 0
        start_time = time.time()

        with open(self.validated_path, "a") as validated_f, \
             open(self.quality_path, "a") as quality_f, \
             open(self.failures_path, "a") as failures_f:

            with ThreadPoolExecutor(max_workers=controller.current_workers()) as executor:
                futures = {}
                for q in remaining:
                    if self._shutdown:
                        break
                    future = executor.submit(
                        agent.verify_question,
                        q,
                        q.get("raw_evidence", ""),
                    )
                    futures[future] = q

                for future in as_completed(futures):
                    q = futures[future]
                    qid = q.get("question_id", "")
                    try:
                        score = future.result()
                        quality_f.write(json.dumps(score.to_dict(), ensure_ascii=False) + "\n")

                        # Merge quality data into question
                        q["quality_status"] = score.quality_status
                        q["clarity"] = score.clarity
                        q["completeness"] = score.completeness
                        q["answerability"] = score.answerability
                        q["option_integrity"] = score.option_integrity
                        q["answer_consistency"] = score.answer_consistency
                        q["benchmark_usefulness"] = score.benchmark_usefulness
                        q["detected_issues"] = score.detected_issues
                        validated_f.write(json.dumps(q, ensure_ascii=False) + "\n")

                        if score.quality_status == "keep":
                            keep_count += 1
                        elif score.quality_status == "review":
                            review_count += 1
                        else:
                            drop_count += 1

                        controller.record_success()
                        processed_ids.add(qid)
                    except Exception as e:
                        controller.record_error()
                        failures_f.write(json.dumps({
                            "question_id": qid,
                            "error": str(e)[:200],
                            "stage": "quality_verification",
                        }, ensure_ascii=False) + "\n")

                    controller.maybe_adjust()

                    # Checkpoint periodically
                    if len(processed_ids) % 20 == 0:
                        self.save_checkpoint(self.stage3_checkpoint, processed_ids)

            # Final checkpoint
            self.save_checkpoint(self.stage3_checkpoint, processed_ids)

        elapsed = time.time() - start_time
        self._update_progress(
            validated_questions=keep_count + review_count + drop_count,
            keep_count=keep_count,
            review_count=review_count,
            drop_count=drop_count,
            active_workers=controller.current_workers(),
            current_stage="quality_verification_complete",
            elapsed_formatted=self._format_time(elapsed),
        )

        return {
            "total_questions": len(questions),
            "verified": len(processed_ids),
            "keep": keep_count,
            "review": review_count,
            "drop": drop_count,
            "elapsed_seconds": elapsed,
        }

    def run_stage_4(self) -> dict:
        """Stage 4: Deduplication + normalization."""
        from src.autodata.pipelines.exam_dedup_normalizer import (
            deduplicate_questions,
            normalize_question_format,
            filter_benchmark_ready,
        )

        self._update_progress(current_stage="deduplication", start_time=time.time())

        # Load validated questions
        questions = []
        if self.validated_path.exists():
            with open(self.validated_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        questions.append(json.loads(line))

        logger.info(f"Stage 4: {len(questions)} validated questions")

        # Normalize format
        normalized = [normalize_question_format(q) for q in questions]

        # Deduplicate
        unique, duplicate_groups = deduplicate_questions(normalized)

        # Save duplicates
        with open(self.duplicates_path, "w") as f:
            for group in duplicate_groups:
                f.write(json.dumps(group, ensure_ascii=False) + "\n")

        # Save unique questions
        with open(self.unique_path, "w") as f:
            for q in unique:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # Filter benchmark-ready
        ready = filter_benchmark_ready(unique)
        with open(self.benchmark_ready_path, "w") as f:
            for q in ready:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        self._update_progress(
            unique_questions=len(unique),
            benchmark_ready_candidates=len(ready),
            current_stage="complete",
        )

        return {
            "total_validated": len(questions),
            "unique_questions": len(unique),
            "duplicate_groups": len(duplicate_groups),
            "benchmark_ready": len(ready),
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
