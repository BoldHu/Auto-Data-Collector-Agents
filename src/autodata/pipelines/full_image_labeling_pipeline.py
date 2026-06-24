"""Full-scale image labeling, captioning, quality, and benchmark candidate pipeline.

Phase 3.9: Label, caption, and assess quality for all 11,624 unique images,
then generate balanced benchmark candidates and validate them.

Three stages with independent checkpoint/resume:
1. Label+Caption+Quality: all 11,624 unique images -> 3 JSONL files
2. Benchmark Candidate Generation: filtered high-quality images -> candidates
3. Critic Validation: candidates -> validation records

Features:
- WriterQueue for safe concurrent JSONL output (no lock contention)
- AdaptiveConcurrencyController for graduated worker scaling
- ImageProgressTracker with rolling-window ETA
- Category-aware benchmark candidate selection with balancing
- Separate checkpoints for each stage
- Graceful shutdown with checkpoint preservation
"""

from __future__ import annotations

import json
import os
import random
import signal
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from src.autodata.pipelines.image_schema import (
    BenchmarkTaskType,
    Difficulty,
    HallucinationRisk,
    AnswerabilityType,
    ImageCaptionRecord,
    ImageCategory,
    ImageLabelRecord,
    ImageModality,
    ImageQualityScore,
    MaterialForm,
    ProcessStage,
    ApplicationDomain,
    MultimodalBenchmarkCandidate,
    CandidateValidationRecord,
    QualityStatus,
    QualityVerdict,
    VisualTaskType,
)
from src.autodata.pipelines.prompts.image_labeling_prompts import (
    PROMPT_VERSION,
    BENCHMARK_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    COMBINED_OUTPUT_SCHEMA,
    CATEGORY_DEFINITIONS,
    MODALITY_DEFINITIONS,
    MATERIAL_FORM_DEFINITIONS,
    PROCESS_STAGE_DEFINITIONS,
    APPLICATION_DOMAIN_DEFINITIONS,
    LABELING_SYSTEM_PROMPT,
    BENCHMARK_OUTPUT_SCHEMA,
    CRITIC_OUTPUT_SCHEMA,
    BENCHMARK_USER_PROMPT_HEADER,
    build_benchmark_user_prompt,
    build_critic_user_prompt,
    get_combined_labeling_prompt,
)
from src.autodata.utils.adaptive_concurrency import AdaptiveConcurrencyController
from src.autodata.utils.image_progress_tracker import ImageProgressTracker
from src.autodata.utils.image_utils import (
    build_multimodal_message,
    resize_image_for_api,
    validate_image_file,
)
from src.autodata.utils.model_pool import get_model_pool
from src.autodata.utils.writer_queue import WriterQueue
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("full_image_labeling_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "image_corpus" / "image_unique_manifest.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "image_corpus"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_full_image_labeling"

RUN_ID = "phase_3_full_image_labeling"


class FullImageLabelingPipeline:
    """Full-scale 3-stage image labeling pipeline."""

    def __init__(
        self,
        manifest_path: Path = MANIFEST_PATH,
        output_dir: Path = OUTPUT_DIR,
        benchmark_dir: Path = BENCHMARK_DIR,
        report_dir: Path = REPORT_DIR,
        initial_workers: int = 8,
        max_workers: int = 16,
        run_id: str = RUN_ID,
        max_candidates_per_image: int = 3,
        max_candidates_per_category: int = 400,
        min_domain_relevance_for_candidates: float = 0.7,
    ) -> None:
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.benchmark_dir = benchmark_dir
        self.report_dir = report_dir
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        self.run_id = run_id
        self.max_candidates_per_image = max_candidates_per_image
        self.max_candidates_per_category = max_candidates_per_category
        self.min_domain_relevance_for_candidates = min_domain_relevance_for_candidates

        self.start_time = time.time()
        self.pool = get_model_pool()
        self._shutdown = False

        # Ensure dirs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Output file paths
        self.labels_file = self.output_dir / "image_labels_full.jsonl"
        self.captions_file = self.output_dir / "image_captions_full.jsonl"
        self.quality_file = self.output_dir / "image_quality_scores_full.jsonl"
        self.failures_file = self.output_dir / "image_labeling_failures_full.jsonl"
        self.candidates_file = self.benchmark_dir / "mm_benchmark_candidates_full.jsonl"
        self.validation_file = self.benchmark_dir / "mm_candidate_validation_full.jsonl"

        # Checkpoint paths
        self.stage1_checkpoint = self.report_dir / "labeling_checkpoint_full.json"
        self.stage2_checkpoint = self.report_dir / "benchmark_checkpoint_full.json"
        self.stage3_checkpoint = self.report_dir / "validation_checkpoint_full.json"

    # ── Stage 1: Label + Caption + Quality ──────────────────────

    def load_manifest(self) -> list[dict]:
        """Load unique image manifest."""
        logger.info(f"Loading manifest: {self.manifest_path}")
        items = []
        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        logger.info(f"Manifest loaded: {len(items)} unique images")
        return items

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

    def parse_labeling_response(self, response_text: str, image_id: str) -> Optional[dict]:
        """Parse combined labeling+captioning+quality JSON response."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed for {image_id}: {str(e)[:80]}")
        return None

    def build_records(self, parsed: dict, image_id: str, image_item: dict) -> dict:
        """Build label, caption, quality records from parsed JSON."""
        category_map = {e.value: e for e in ImageCategory}
        material_map = {e.value: e for e in MaterialForm}
        process_map = {e.value: e for e in ProcessStage}
        domain_map = {e.value: e for e in ApplicationDomain}
        status_map = {"keep": QualityStatus.KEEP, "review": QualityStatus.REVIEW, "drop": QualityStatus.DROP}

        label_record = ImageLabelRecord(
            image_id=image_id,
            primary_category=category_map.get(parsed.get("primary_category", ""), ImageCategory.UNKNOWN),
            secondary_categories=parsed.get("secondary_categories", []),
            material_form=material_map.get(parsed.get("material_form", ""), MaterialForm.UNKNOWN),
            process_stage=process_map.get(parsed.get("process_stage", ""), ProcessStage.UNKNOWN),
            application_domain=domain_map.get(parsed.get("application_domain", ""), ApplicationDomain.UNKNOWN),
            domain_relevance=float(parsed.get("domain_relevance", 0.0)),
            label_confidence=float(parsed.get("label_confidence", 0.5)),
            requires_human_review=bool(parsed.get("requires_human_review", False)),
            label_model=self.pool.endpoints[0].model_name if self.pool.endpoints else "mimo-v2-omni",
            label_prompt_version=PROMPT_VERSION,
            source_refs=[image_item.get("file_path", "")],
            timestamp=time.time(),
            run_id=self.run_id,
        )

        caption_record = ImageCaptionRecord(
            image_id=image_id,
            short_caption=parsed.get("short_caption", ""),
            technical_caption=parsed.get("technical_caption", ""),
            visible_objects=parsed.get("visible_objects", []),
            visible_materials=parsed.get("visible_materials", []),
            visible_processes=parsed.get("visible_processes", []),
            visible_equipment=parsed.get("visible_equipment", []),
            visible_text=parsed.get("visible_text", []),
            visual_evidence=parsed.get("visual_evidence", []),
            inferred_domain_context=parsed.get("inferred_domain_context", []),
            uncertainty_notes=[parsed.get("uncertainty_notes", "")] if parsed.get("uncertainty_notes") else [],
            caption_model=self.pool.endpoints[0].model_name if self.pool.endpoints else "mimo-v2-omni",
            caption_prompt_version=PROMPT_VERSION,
            caption_status=QualityVerdict.PASSED if parsed.get("label_confidence", 0.5) >= 0.5 else QualityVerdict.NEEDS_REVISION,
            source_refs=[image_item.get("file_path", "")],
            timestamp=time.time(),
            run_id=self.run_id,
        )

        quality_record = ImageQualityScore(
            image_id=image_id,
            clarity=float(parsed.get("clarity", 0.5)),
            domain_relevance=float(parsed.get("domain_relevance", 0.5)),
            visual_informativeness=float(parsed.get("visual_informativeness", 0.5)),
            captionability=float(parsed.get("captionability", 0.5)),
            reasoning_potential=float(parsed.get("reasoning_potential", 0.5)),
            metadata_completeness=0.7 if image_item.get("image_url") else 0.3,
            quality_status=status_map.get(parsed.get("quality_status", "review"), QualityStatus.REVIEW),
            drop_reason=parsed.get("drop_reason", ""),
            quality_model=self.pool.endpoints[0].model_name if self.pool.endpoints else "mimo-v2-omni",
            source_refs=[image_item.get("file_path", "")],
            timestamp=time.time(),
            run_id=self.run_id,
        )

        return {
            "label": label_record,
            "caption": caption_record,
            "quality": quality_record,
        }

    def process_image(self, image_item: dict, writer_queue: WriterQueue,
                      controller: AdaptiveConcurrencyController) -> tuple[bool, str, int]:
        """Process a single image: validate, prompt, LLM call, parse, write.

        Returns (success, image_id, tokens_used).
        """
        image_id = image_item["image_id"]
        image_path = image_item["file_path"]

        # Validate image
        validation = validate_image_file(image_path)
        if not validation["valid"]:
            logger.warning(f"Invalid image: {image_path} — {validation['reason']}")
            controller.record_error()
            writer_queue.put("failures", {
                "image_id": image_id, "reason": validation["reason"],
                "timestamp": time.time(), "run_id": self.run_id,
            })
            return False, image_id, 0

        # Build multimodal prompt
        try:
            messages = get_combined_labeling_prompt(image_path)
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_path}")
            controller.record_error()
            writer_queue.put("failures", {
                "image_id": image_id, "reason": "file_not_found",
                "timestamp": time.time(), "run_id": self.run_id,
            })
            return False, image_id, 0
        except Exception as e:
            logger.warning(f"Error building prompt for {image_id}: {str(e)[:80]}")
            controller.record_error()
            writer_queue.put("failures", {
                "image_id": image_id, "reason": f"prompt_error: {str(e)[:60]}",
                "timestamp": time.time(), "run_id": self.run_id,
            })
            return False, image_id, 0

        # Call LLM
        try:
            response = self.pool.chat_multimodal(
                messages=messages,
                max_completion_tokens=4096,
            )
            response_text = response.content
            tokens = response.total_tokens or 0
        except Exception as e:
            logger.warning(f"LLM call failed for {image_id}: {str(e)[:80]}")
            controller.record_error()
            writer_queue.put("failures", {
                "image_id": image_id, "reason": f"llm_error: {str(e)[:60]}",
                "timestamp": time.time(), "run_id": self.run_id,
            })
            return False, image_id, 0

        # Parse response
        parsed = self.parse_labeling_response(response_text, image_id)
        if parsed is None:
            logger.warning(f"Failed to parse response for {image_id}")
            controller.record_error()
            controller.record_json_parse_failure()
            writer_queue.put("failures", {
                "image_id": image_id, "reason": "json_parse_failure",
                "timestamp": time.time(), "run_id": self.run_id,
            })
            return False, image_id, 0

        # Build and write records
        records = self.build_records(parsed, image_id, image_item)
        writer_queue.put("labels", records["label"].to_dict())
        writer_queue.put("captions", records["caption"].to_dict())
        writer_queue.put("quality", records["quality"].to_dict())

        controller.record_success()
        qs = parsed.get("quality_status", "review")
        logger.info(f"Processed {image_id}: category={parsed.get('primary_category', '?')}, "
                    f"relevance={parsed.get('domain_relevance', '?')}, quality={qs}")

        return True, image_id, tokens

    def run_stage_1(self, max_images: Optional[int] = None) -> dict[str, Any]:
        """Stage 1: Label+Caption+Quality all unique images."""
        logger.info("=== Stage 1: Full Image Labeling ===")

        items = self.load_manifest()
        if max_images:
            items = items[:max_images]
        total = len(items)

        processed_ids = self.load_checkpoint(self.stage1_checkpoint)
        remaining = [i for i in items if i["image_id"] not in processed_ids]
        logger.info(f"Total: {total}, Already processed: {len(processed_ids)}, Remaining: {len(remaining)}")

        if not remaining:
            logger.info("All images already processed")
            return {"total_processed": len(processed_ids), "elapsed_seconds": 0}

        # Setup WriterQueue
        writer_queue = WriterQueue({
            "labels": self.labels_file,
            "captions": self.captions_file,
            "quality": self.quality_file,
            "failures": self.failures_file,
        })

        # Setup adaptive concurrency controller
        controller = AdaptiveConcurrencyController(
            initial_workers=self.initial_workers,
            max_workers=self.max_workers,
        )

        # Setup progress tracker
        tracker = ImageProgressTracker(
            run_id=self.run_id,
            json_path=self.report_dir / "labeling_progress_full.json",
            log_path=self.report_dir / "labeling_progress_full.log",
            total_images=len(remaining),
        )
        tracker.start()
        tracker.set_stage("labeling")

        # Setup shutdown handler
        self._setup_shutdown_handler()

        # Submit all remaining images, using executor max_workers for concurrency
        # The AdaptiveConcurrencyController monitors but doesn't gate submissions
        # because ThreadPoolExecutor's max_workers naturally limits concurrency
        completed = 0
        failed = 0
        total_tokens = 0

        with ThreadPoolExecutor(max_workers=controller.current_workers()) as executor:
            futures = {}
            for item in remaining:
                if self._shutdown:
                    break
                future = executor.submit(self.process_image, item, writer_queue, controller)
                futures[future] = item

            for future in as_completed(futures):
                if self._shutdown:
                    logger.info("Shutdown requested, saving checkpoint...")
                    break

                item = futures[future]
                try:
                    success, image_id, tokens = future.result()
                    total_tokens += tokens
                    if success:
                        completed += 1
                        tracker.on_image_completed(tokens=tokens, quality_status="")
                        processed_ids.add(image_id)
                    else:
                        failed += 1
                        tracker.on_image_failed()
                except Exception as e:
                    logger.warning(f"Future error for {item['image_id']}: {str(e)[:80]}")
                    failed += 1
                    tracker.on_image_failed()
                    controller.record_error()

                # Maybe adjust concurrency (monitoring only, executor size is fixed)
                controller.maybe_adjust()

                # Progress report and checkpoint
                tracker.maybe_report()
                if completed % 100 == 0 and completed > 0:
                    self.save_checkpoint(self.stage1_checkpoint, processed_ids)

        # Final save
        self.save_checkpoint(self.stage1_checkpoint, processed_ids)
        tracker.set_stage("labeling_complete")
        tracker.maybe_report(force=True)
        writer_queue.flush_and_close()

        elapsed = time.time() - self.start_time
        logger.info(f"=== Stage 1 complete: {completed} ok, {failed} fail, {elapsed:.1f}s ===")

        # Save model pool status
        pool_stats = self.pool.stats()
        with open(self.report_dir / "model_pool_status.json", "w") as f:
            json.dump(pool_stats, f, indent=2)

        return {
            "total_processed": completed,
            "total_failed": failed,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed,
            "pool_stats": pool_stats,
            "controller_stats": controller.report(),
        }

    # ── Stage 2: Benchmark Candidate Generation ──────────────────

    def load_labels(self) -> dict[str, dict]:
        """Load labels from full labeling output."""
        labels = {}
        if not self.labels_file.exists():
            logger.warning(f"Labels file not found: {self.labels_file}")
            return labels
        with open(self.labels_file) as f:
            for line in f:
                r = json.loads(line)
                labels[r["image_id"]] = r
        logger.info(f"Loaded {len(labels)} label records")
        return labels

    def load_quality(self) -> dict[str, dict]:
        """Load quality scores from full labeling output."""
        quality = {}
        if not self.quality_file.exists():
            logger.warning(f"Quality file not found: {self.quality_file}")
            return quality
        with open(self.quality_file) as f:
            for line in f:
                r = json.loads(line)
                quality[r["image_id"]] = r
        logger.info(f"Loaded {len(quality)} quality records")
        return quality

    def load_manifest_paths(self) -> dict[str, str]:
        """Load image_id -> file_path mapping from manifest."""
        paths = {}
        with open(self.manifest_path) as f:
            for line in f:
                r = json.loads(line)
                paths[r["image_id"]] = r["file_path"]
        return paths

    def select_balanced_images(
        self,
        labels: dict[str, dict],
        quality: dict[str, dict],
        manifest_paths: dict[str, str],
    ) -> list[dict]:
        """Select high-quality images for benchmark candidates with category balancing.

        Higher thresholds than pilot:
        - domain_relevance >= 0.7 (not 0.6)
        - quality_status == "keep" only (not "keep" or "review")
        - label_confidence >= 0.7
        - Per-category cap to prevent over-representation
        """
        eligible = []
        for image_id, label in labels.items():
            q = quality.get(image_id, {})
            fp = manifest_paths.get(image_id, "")
            if not fp:
                continue

            # Filter: high quality only
            if label.get("domain_relevance", 0) < self.min_domain_relevance_for_candidates:
                continue
            if q.get("quality_status") != "keep":
                continue
            if label.get("label_confidence", 0) < 0.7:
                continue
            if label.get("primary_category") == "irrelevant":
                continue

            eligible.append({
                "image_id": image_id,
                "file_path": fp,
                "primary_category": label.get("primary_category", "unknown"),
                "material_form": label.get("material_form", "unknown"),
                "process_stage": label.get("process_stage", "unknown"),
                "domain_relevance": label.get("domain_relevance", 0),
                "label_confidence": label.get("label_confidence", 0),
            })

        # Per-category cap
        by_category = defaultdict(list)
        for item in eligible:
            by_category[item["primary_category"]].append(item)

        selected = []
        category_stats = {}
        for category, items in by_category.items():
            take = min(self.max_candidates_per_category, len(items))
            if take < len(items):
                # Random sample down to cap
                items = random.sample(items, take)
            selected.extend(items)
            category_stats[category] = {"eligible": len(by_category[category]), "selected": take}

        logger.info(f"Balanced selection: {len(selected)} images from {len(by_category)} categories")
        for cat, stats in category_stats.items():
            logger.info(f"  {cat}: {stats['selected']}/{stats['eligible']}")

        return selected

    def parse_benchmark_response(self, response_text: str, image_id: str) -> list[dict]:
        """Parse benchmark candidate response JSON."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                candidates = data.get("candidates", [])
                return candidates
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Benchmark JSON parse failed for {image_id}: {str(e)[:80]}")
        return []

    def process_candidate_image(
        self,
        image_info: dict,
        writer_queue: WriterQueue,
    ) -> list[MultimodalBenchmarkCandidate]:
        """Generate benchmark candidates for one image."""
        image_id = image_info["image_id"]
        file_path = image_info["file_path"]

        # Build benchmark prompt
        try:
            b64_url = resize_image_for_api(file_path)
            user_prompt = build_benchmark_user_prompt(
                primary_category=image_info.get("primary_category", "unknown"),
                material_form=image_info.get("material_form", "unknown"),
                process_stage=image_info.get("process_stage", "unknown"),
                domain_relevance=image_info.get("domain_relevance", 0.7),
            )
            messages = [
                {"role": "system", "content": BENCHMARK_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": b64_url}},
                ]},
            ]
        except Exception as e:
            logger.warning(f"Benchmark prompt failed for {image_id}: {str(e)[:80]}")
            return []

        # Call LLM
        try:
            response = self.pool.chat_multimodal(messages=messages, max_completion_tokens=4096)
            response_text = response.content
        except Exception as e:
            logger.warning(f"Benchmark LLM call failed for {image_id}: {str(e)[:80]}")
            return []

        # Parse and build candidates
        raw_candidates = self.parse_benchmark_response(response_text, image_id)
        task_type_map = {e.value: e for e in BenchmarkTaskType}
        difficulty_map = {e.value: e for e in Difficulty}
        answerability_map = {e.value: e for e in AnswerabilityType}
        hallucination_map = {e.value: e for e in HallucinationRisk}

        candidates = []
        for raw in raw_candidates[:self.max_candidates_per_image]:
            cand = MultimodalBenchmarkCandidate(
                image_id=image_id,
                task_type=task_type_map.get(raw.get("task_type", ""), BenchmarkTaskType.VISUAL_QA),
                question=raw.get("question", ""),
                options=raw.get("options", []),
                answer=raw.get("answer", ""),
                explanation=raw.get("explanation", ""),
                visual_evidence=raw.get("visual_evidence", []),
                required_knowledge=raw.get("required_knowledge", []),
                reasoning_steps=raw.get("reasoning_steps", []),
                difficulty=difficulty_map.get(raw.get("difficulty", ""), Difficulty.MEDIUM),
                answerability=answerability_map.get(raw.get("answerability", ""), AnswerabilityType.IMAGE_ONLY),
                hallucination_risk=hallucination_map.get(raw.get("hallucination_risk", ""), HallucinationRisk.LOW),
                source_refs=[file_path],
                timestamp=time.time(),
                run_id=self.run_id,
            )
            writer_queue.put("candidates", cand.to_dict())
            candidates.append(cand)

        return candidates

    def run_stage_2(self) -> dict[str, Any]:
        """Stage 2: Generate benchmark candidates from labeled images."""
        logger.info("=== Stage 2: Benchmark Candidate Generation ===")

        labels = self.load_labels()
        quality = self.load_quality()
        manifest_paths = self.load_manifest_paths()

        selected = self.select_balanced_images(labels, quality, manifest_paths)
        if not selected:
            logger.warning("No eligible images for benchmark candidates")
            return {"total_candidates": 0}

        # Load checkpoint
        processed_ids = self.load_checkpoint(self.stage2_checkpoint)
        remaining = [i for i in selected if i["image_id"] not in processed_ids]
        logger.info(f"Selected {len(selected)}, already processed {len(processed_ids)}, remaining {len(remaining)}")

        writer_queue = WriterQueue({"candidates": self.candidates_file})

        tracker = ImageProgressTracker(
            run_id=self.run_id,
            json_path=self.report_dir / "benchmark_progress_full.json",
            log_path=self.report_dir / "benchmark_progress_full.log",
            total_images=len(remaining),
        )
        tracker.start()
        tracker.set_stage("benchmark_generation")

        total_candidates = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for item in remaining:
                future = executor.submit(self.process_candidate_image, item, writer_queue)
                futures[future] = item

            for future in as_completed(futures):
                item = futures[future]
                try:
                    candidates = future.result()
                    total_candidates += len(candidates)
                    completed += 1
                    tracker.on_image_completed()
                    for _ in candidates:
                        tracker.on_candidate_generated()
                    processed_ids.add(item["image_id"])
                except Exception as e:
                    logger.warning(f"Candidate generation failed for {item['image_id']}: {str(e)[:80]}")
                    completed += 1
                    tracker.on_image_failed()

                tracker.maybe_report()
                if completed % 50 == 0:
                    self.save_checkpoint(self.stage2_checkpoint, processed_ids)

        self.save_checkpoint(self.stage2_checkpoint, processed_ids)
        writer_queue.flush_and_close()
        tracker.finish()

        elapsed = time.time() - self.start_time
        logger.info(f"=== Stage 2 complete: {total_candidates} candidates in {elapsed:.1f}s ===")
        return {"total_candidates": total_candidates, "total_images_processed": completed}

    # ── Stage 3: Critic Validation ──────────────────────────

    def parse_critic_response(self, response_text: str, candidate_id: str) -> Optional[dict]:
        """Parse critic validation response."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Critic JSON parse failed for {candidate_id}: {str(e)[:80]}")
        return None

    def validate_candidate(
        self,
        candidate: dict,
        label: Optional[dict],
        writer_queue: WriterQueue,
    ) -> Optional[CandidateValidationRecord]:
        """Validate a benchmark candidate with independent critic."""
        candidate_id = candidate.get("candidate_id", "")
        image_id = candidate.get("image_id", "")

        # Build critic prompt (text-only, no image)
        try:
            critic_prompt = build_critic_user_prompt(
                task_type=candidate.get("task_type", ""),
                question=candidate.get("question", ""),
                options=str(candidate.get("options", [])),
                answer=candidate.get("answer", ""),
                explanation=candidate.get("explanation", ""),
                visual_evidence=str(candidate.get("visual_evidence", [])),
                primary_category=label.get("primary_category", "unknown") if label else "unknown",
                label_confidence=label.get("label_confidence", 0.5) if label else 0.5,
            )
            messages = [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": critic_prompt},
            ]
        except Exception as e:
            logger.warning(f"Critic prompt build failed for {candidate_id}: {str(e)[:80]}")
            return None

        # Call text-only quality model
        try:
            response = self.pool.chat_quality(messages=messages, max_completion_tokens=2048)
            response_text = response.content
        except Exception as e:
            logger.warning(f"Critic LLM call failed for {candidate_id}: {str(e)[:80]}")
            return None

        # Parse response
        parsed = self.parse_critic_response(response_text, candidate_id)
        if parsed is None:
            return None

        verdict_map = {e.value: e for e in QualityVerdict}
        verdict_map["review"] = QualityVerdict.NEEDS_REVISION
        hallucination_map = {e.value: e for e in HallucinationRisk}

        record = CandidateValidationRecord(
            candidate_id=candidate_id,
            image_id=image_id,
            validation_status=verdict_map.get(parsed.get("validation_status", ""), QualityVerdict.NEEDS_REVISION),
            answerability_score=float(parsed.get("answerability_score", 0)),
            visual_grounding_score=float(parsed.get("visual_grounding_score", 0)),
            domain_reasoning_score=float(parsed.get("domain_reasoning_score", 0)),
            hallucination_risk=hallucination_map.get(parsed.get("hallucination_risk", ""), HallucinationRisk.LOW),
            ambiguity_score=float(parsed.get("ambiguity_score", 0)),
            critic_notes=parsed.get("critic_notes", ""),
            revision_suggestion=parsed.get("revision_suggestion", ""),
            critic_model="mimo-v2.5-pro",
            timestamp=time.time(),
            run_id=self.run_id,
        )

        writer_queue.put("validation", record.to_dict())
        return record

    def run_stage_3(self) -> dict[str, Any]:
        """Stage 3: Independent critic validation of benchmark candidates."""
        logger.info("=== Stage 3: Critic Validation ===")

        # Load candidates
        candidates = []
        if not self.candidates_file.exists():
            logger.warning(f"Candidates file not found: {self.candidates_file}")
            return {"total_validated": 0}

        with open(self.candidates_file) as f:
            for line in f:
                candidates.append(json.loads(line))
        logger.info(f"Loaded {len(candidates)} candidates for validation")

        # Load labels for context
        labels = self.load_labels()

        # Load checkpoint
        processed_ids = self.load_checkpoint(self.stage3_checkpoint)
        remaining_candidates = [c for c in candidates if c.get("candidate_id", "") not in processed_ids]
        logger.info(f"Total: {len(candidates)}, Already validated: {len(processed_ids)}, Remaining: {len(remaining_candidates)}")

        writer_queue = WriterQueue({"validation": self.validation_file})

        tracker = ImageProgressTracker(
            run_id=self.run_id,
            json_path=self.report_dir / "validation_progress_full.json",
            log_path=self.report_dir / "validation_progress_full.log",
            total_images=len(remaining_candidates),
        )
        tracker.start()
        tracker.set_stage("critic_validation")

        total_validated = 0
        total_passed = 0
        total_needs_revision = 0
        total_failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for candidate in remaining_candidates:
                label = labels.get(candidate.get("image_id", ""))
                future = executor.submit(self.validate_candidate, candidate, label, writer_queue)
                futures[future] = candidate

            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    result = future.result()
                    if result:
                        total_validated += 1
                        tracker.on_image_completed()
                        tracker.on_candidate_validated()
                        processed_ids.add(candidate.get("candidate_id", ""))
                        if result.validation_status == QualityVerdict.PASSED:
                            total_passed += 1
                        elif result.validation_status == QualityVerdict.NEEDS_REVISION:
                            total_needs_revision += 1
                        else:
                            total_failed += 1
                    else:
                        total_failed += 1
                        tracker.on_image_failed()
                        processed_ids.add(candidate.get("candidate_id", ""))
                except Exception as e:
                    logger.warning(f"Validation failed for {candidate.get('candidate_id', '')}: {str(e)[:80]}")
                    tracker.on_image_failed()

                tracker.maybe_report()
                if total_validated % 50 == 0:
                    self.save_checkpoint(self.stage3_checkpoint, processed_ids)

        self.save_checkpoint(self.stage3_checkpoint, processed_ids)
        writer_queue.flush_and_close()
        tracker.finish()

        elapsed = time.time() - self.start_time
        logger.info(f"=== Stage 3 complete: {total_validated} validated, {elapsed:.1f}s ===")

        return {
            "total_validated": total_validated,
            "total_passed": total_passed,
            "total_needs_revision": total_needs_revision,
            "total_failed": total_failed,
            "elapsed_seconds": elapsed,
        }

    # ── Shutdown Handler ──────────────────────────────────

    def _setup_shutdown_handler(self):
        """Register SIGINT handler for graceful shutdown."""
        def handler(signum, frame):
            logger.info("Shutdown signal received, finishing current work...")
            self._shutdown = True
        signal.signal(signal.SIGINT, handler)

    # ── Full Run ──────────────────────────────────

    def run(self, start_stage: int = 1, max_images: Optional[int] = None) -> dict[str, Any]:
        """Execute the complete 3-stage pipeline with DTCG integration."""
        from src.autodata.context_graph.pipeline_dtcg_integration import PipelineDTCG

        # Initialize DTCG runtime trace
        dtcg = PipelineDTCG("phase_3_full_image_labeling", self.report_dir)
        agent_id = dtcg.add_agent("ImageLabelingAgent", role="multimodal_labeling")
        tool_pool_id = dtcg.add_tool("ModelPool", api="xiaomi0")
        constraint_id = dtcg.add_constraint("no_modify_raw_images")

        results = {}

        if start_stage <= 1:
            task1_id = dtcg.add_task("label_caption_quality", status="in_progress")
            dtcg.connect_agent_to_task(agent_id, task1_id)
            dtcg.connect_tool_usage(tool_pool_id, task1_id)
            dtcg.connect_quality_feedback(constraint_id, task1_id)
            results["stage_1"] = self.run_stage_1(max_images=max_images)
            # Mark task complete and add artifact
            art1_id = dtcg.add_artifact("image_labels_full.jsonl", path=str(self.labels_file))
            dtcg.connect_artifact_derived(art1_id, task1_id)

        if start_stage <= 2:
            task2_id = dtcg.add_task("benchmark_candidate_generation", status="in_progress")
            dtcg.connect_agent_to_task(agent_id, task2_id)
            dtcg.connect_task_dependency(task1_id if start_stage <= 1 else "task_labeling", task2_id)
            results["stage_2"] = self.run_stage_2()
            art2_id = dtcg.add_artifact("mm_benchmark_candidates_full.jsonl", path=str(self.candidates_file))
            dtcg.connect_artifact_derived(art2_id, task2_id)

        if start_stage <= 3:
            task3_id = dtcg.add_task("critic_validation", status="in_progress")
            dtcg.connect_agent_to_task(agent_id, task3_id)
            dtcg.connect_task_dependency(task2_id if start_stage <= 2 else "task_candidates", task3_id)
            results["stage_3"] = self.run_stage_3()
            art3_id = dtcg.add_artifact("mm_candidate_validation_full.jsonl", path=str(self.validation_file))
            dtcg.connect_artifact_derived(art3_id, task3_id)

        # Save DTCG runtime trace
        dtcg.save()

        return results