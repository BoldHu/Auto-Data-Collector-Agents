"""Image labeling pipeline — pilot 300 images with ModelPool, concurrent workers.

Phase 3.3: Label, caption, and assess quality for a pilot set of 300 images.

Pipeline flow per image:
1. Load image from dedup index (dedup_status=unique only)
2. Resize image for API (max 1024x1024, never modify raw)
3. Build multimodal message with combined labeling prompt
4. Call ModelPool.chat_multimodal() with mimo-v2-omni/mimo-v2.5
5. Parse JSON response into ImageLabelRecord + ImageCaptionRecord + ImageQualityScore
6. Write results to JSONL files

Features:
- ThreadPoolExecutor with configurable num_workers
- ModelPool: dual-API round-robin, failover, multimodal endpoints
- Combined labeling+captioning+quality prompt (single LLM call per image)
- Checkpointing: resume from last checkpoint if interrupted
- Stratified sampling: select diverse keyword folders
- Progress tracking: enhanced with per-model stats
- Graceful shutdown: finish current images, then stop
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

from src.autodata.pipelines.image_schema import (
    DedupStatus,
    ImageCaptionRecord,
    ImageCategory,
    ImageLabelRecord,
    ImageLabelingRunMetadata,
    ImageManifestItem,
    ImageModality,
    ImageQualityScore,
    MaterialForm,
    ProcessStage,
    ApplicationDomain,
    QualityStatus,
    QualityVerdict,
    VisualTaskType,
)
from src.autodata.pipelines.prompts.image_labeling_prompts import (
    PROMPT_VERSION,
    get_combined_labeling_prompt,
)
from src.autodata.utils.image_utils import (
    build_multimodal_message,
    resize_image_for_api,
    validate_image_file,
)
from src.autodata.utils.model_pool import get_model_pool
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("image_labeling_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEDUP_PATH = PROJECT_ROOT / "data" / "interim" / "image_dedup" / "image_dedup.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "image_labeled"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_image_labeling"


class ImageLabelingPipeline:
    """Image labeling, captioning, and quality assessment pipeline."""

    def __init__(
        self,
        dedup_path: Path = DEDUP_PATH,
        output_dir: Path = OUTPUT_DIR,
        report_dir: Path = REPORT_DIR,
        max_images: int = 300,
        num_workers: int = 10,
        stratified: bool = True,
    ) -> None:
        self.dedup_path = dedup_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.max_images = max_images
        self.num_workers = num_workers
        self.stratified = stratified
        self.start_time = time.time()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.pool = get_model_pool()
        self._output_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._shutdown = False

        # Output file paths
        self.labels_file = self.output_dir / "image_labels_pilot.jsonl"
        self.captions_file = self.output_dir / "image_captions_pilot.jsonl"
        self.quality_file = self.output_dir / "image_quality_scores_pilot.jsonl"
        self.checkpoint_file = self.output_dir / "image_labeling_checkpoint.json"
        self.progress_file = self.output_dir / "image_labeling_progress.json"

    def load_unique_images(self) -> list[dict]:
        """Load dedup index and filter to unique images only."""
        logger.info(f"Loading dedup index: {self.dedup_path}")
        items = []
        with open(self.dedup_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("dedup_status") == DedupStatus.UNIQUE.value:
                    items.append(record)
        logger.info(f"Unique images available: {len(items)}")
        return items

    def stratified_sample(self, items: list[dict]) -> list[dict]:
        """Select diverse images across different keyword folders.

        Strategy:
        1. Group images by keyword folder
        2. From each folder, randomly select 1-3 images
        3. This ensures diversity across keyword categories
        """
        import random

        # Group by folder_keyword
        folder_groups = {}
        for item in items:
            kw = item.get("folder_keyword", "unknown")
            if kw not in folder_groups:
                folder_groups[kw] = []
            folder_groups[kw].append(item)

        logger.info(f"Total keyword folders: {len(folder_groups)}")

        # Calculate images per folder to reach max_images
        images_per_folder = max(1, self.max_images // len(folder_groups))

        selected = []
        folder_list = list(folder_groups.keys())
        random.shuffle(folder_list)

        for folder in folder_list:
            folder_items = folder_groups[folder]
            # Select up to images_per_folder items, preferring those with metadata
            sorted_items = sorted(
                folder_items,
                key=lambda x: (x.get("source_status", "") != "metadata_missing", x.get("file_size", 0)),
                reverse=True,
            )
            take = min(images_per_folder, len(sorted_items))
            selected.extend(sorted_items[:take])

            if len(selected) >= self.max_images:
                break

        # Trim to exact max_images
        selected = selected[:self.max_images]
        logger.info(f"Stratified sample: {len(selected)} images from {len(set(i['folder_keyword'] for i in selected))} folders")
        return selected

    def load_checkpoint(self) -> set:
        """Load already-processed image IDs from checkpoint."""
        if not self.checkpoint_file.exists():
            return set()

        with open(self.checkpoint_file) as f:
            data = json.load(f)
        return set(data.get("processed_image_ids", []))

    def save_checkpoint(self, processed_ids: set) -> None:
        """Save checkpoint with processed image IDs."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "processed_image_ids": list(processed_ids),
                "timestamp": time.time(),
                "total_processed": len(processed_ids),
            }, f)

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
        """Build ImageLabelRecord, ImageCaptionRecord, ImageQualityScore from parsed JSON."""
        run_id = f"phase_3_3_pilot_{int(self.start_time)}"

        # Map string values to enums (with fallback to UNKNOWN)
        category_map = {e.value: e for e in ImageCategory}
        modality_map = {e.value: e for e in ImageModality}
        material_map = {e.value: e for e in MaterialForm}
        process_map = {e.value: e for e in ProcessStage}
        domain_map = {e.value: e for e in ApplicationDomain}
        status_map = {"keep": QualityStatus.KEEP, "review": QualityStatus.REVIEW, "drop": QualityStatus.DROP}

        # ImageLabelRecord
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
            run_id=run_id,
        )

        # ImageCaptionRecord
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
            caption_status=QualityVerdict.PASSED if parsed.get("label_confidence", 0.5) >= 0.5 else QualityVerdict.REVIEW,
            source_refs=[image_item.get("file_path", "")],
            timestamp=time.time(),
            run_id=run_id,
        )

        # ImageQualityScore
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
            run_id=run_id,
        )

        return {
            "label": label_record,
            "caption": caption_record,
            "quality": quality_record,
        }

    def process_image(self, image_item: dict, processed_ids: set) -> bool:
        """Process a single image: label, caption, quality assess."""
        image_id = image_item["image_id"]
        image_path = image_item["file_path"]

        # Skip already processed
        if image_id in processed_ids:
            return True

        # Validate image
        validation = validate_image_file(image_path)
        if not validation["valid"]:
            logger.warning(f"Invalid image: {image_path} — {validation['reason']}")
            return False

        # Build multimodal prompt
        try:
            messages = get_combined_labeling_prompt(image_path)
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_path}")
            return False
        except Exception as e:
            logger.warning(f"Error building prompt for {image_id}: {str(e)[:80]}")
            return False

        # Call LLM
        try:
            response = self.pool.chat_multimodal(
                messages=messages,
                max_completion_tokens=4096,
            )
            response_text = response.content
        except Exception as e:
            logger.warning(f"LLM call failed for {image_id}: {str(e)[:80]}")
            return False

        # Parse response
        parsed = self.parse_labeling_response(response_text, image_id)
        if parsed is None:
            logger.warning(f"Failed to parse response for {image_id}")
            return False

        # Build records
        records = self.build_records(parsed, image_id, image_item)

        # Write to JSONL files (thread-safe)
        with self._output_lock:
            with open(self.labels_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(records["label"].to_dict(), ensure_ascii=False) + "\n")
            with open(self.captions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(records["caption"].to_dict(), ensure_ascii=False) + "\n")
            with open(self.quality_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(records["quality"].to_dict(), ensure_ascii=False) + "\n")

        # Update progress
        with self._progress_lock:
            processed_ids.add(image_id)

        logger.info(f"Processed {image_id}: category={parsed.get('primary_category', '?')}, "
                     f"relevance={parsed.get('domain_relevance', '?')}, "
                     f"quality={parsed.get('quality_status', '?')}")

        return True

    def _setup_shutdown_handler(self):
        """Register SIGINT handler for graceful shutdown."""
        original_sigint = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            logger.info("Shutdown signal received, finishing current work...")
            self._shutdown = True

        signal.signal(signal.SIGINT, handler)

    def save_progress(self, progress: dict):
        """Save progress tracking file."""
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def run(self) -> dict[str, Any]:
        """Execute the complete pilot labeling pipeline."""
        logger.info(f"=== Phase 3.3: Image Labeling Pipeline (pilot {self.max_images}) ===")

        # Load unique images
        all_items = self.load_unique_images()

        # Stratified sample or simple sample
        if self.stratified:
            items_to_process = self.stratified_sample(all_items)
        else:
            items_to_process = all_items[:self.max_images]

        logger.info(f"Processing {len(items_to_process)} images with {self.num_workers} workers")

        # Load checkpoint
        processed_ids = self.load_checkpoint()
        logger.info(f"Already processed: {len(processed_ids)}")

        # Filter out already-processed
        items_to_process = [i for i in items_to_process if i["image_id"] not in processed_ids]
        logger.info(f"Remaining to process: {len(items_to_process)}")

        if not items_to_process:
            logger.info("All images already processed, nothing to do")
            return {
                "total_processed": len(processed_ids),
                "elapsed_seconds": 0,
            }

        # Setup shutdown handler
        self._setup_shutdown_handler()

        # Progress tracking
        progress = {
            "total_to_process": len(items_to_process),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "tokens_used": 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "model_stats": {},
        }

        # Process with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for item in items_to_process:
                if self._shutdown:
                    break
                future = executor.submit(self.process_image, item, processed_ids)
                futures[future] = item

            for future in as_completed(futures):
                if self._shutdown:
                    break
                item = futures[future]
                try:
                    success = future.result()
                    with self._progress_lock:
                        if success:
                            progress["completed"] += 1
                        else:
                            progress["failed"] += 1
                        progress["last_update"] = time.time()
                except Exception as e:
                    logger.warning(f"Future error for {item['image_id']}: {str(e)[:80]}")
                    with self._progress_lock:
                        progress["failed"] += 1

                # Periodic checkpoint save
                if progress["completed"] % 20 == 0 or progress["failed"] % 10 == 0:
                    self.save_checkpoint(processed_ids)
                    self.save_progress(progress)

        # Final checkpoint save
        self.save_checkpoint(processed_ids)
        self.save_progress(progress)

        # Get pool stats
        pool_stats = self.pool.stats()

        elapsed = time.time() - self.start_time
        logger.info(
            f"=== Phase 3.3 pilot complete: "
            f"{progress['completed']} completed, {progress['failed']} failed, "
            f"{elapsed:.1f}s ==="
        )

        return {
            "total_processed": progress["completed"],
            "total_failed": progress["failed"],
            "elapsed_seconds": elapsed,
            "pool_stats": pool_stats,
            "labels_file": str(self.labels_file),
            "captions_file": str(self.captions_file),
            "quality_file": str(self.quality_file),
        }