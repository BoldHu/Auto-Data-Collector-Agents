"""Multimodal benchmark candidate generator — generate benchmark items from labeled images.

Phase 3.4: Generate 1-3 benchmark candidate items per suitable image.

Pipeline:
1. Load labeled images with domain_relevance ≥ 0.6 and quality_status = "keep" or "review"
2. Build benchmark generation prompt for each image
3. Call multimodal LLM to generate candidates
4. Parse JSON response into MultimodalBenchmarkCandidate records
5. Write to JSONL output

Output:
    data/benchmark_candidates/multimodal/mm_benchmark_candidates_pilot.jsonl
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from src.autodata.pipelines.image_schema import (
    BenchmarkTaskType,
    Difficulty,
    HallucinationRisk,
    AnswerabilityType,
    MultimodalBenchmarkCandidate,
    QualityVerdict,
)
from src.autodata.pipelines.prompts.image_labeling_prompts import (
    PROMPT_VERSION,
    BENCHMARK_SYSTEM_PROMPT,
    build_benchmark_user_prompt,
)
from src.autodata.utils.model_pool import get_model_pool
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("benchmark_generator")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS_PATH = PROJECT_ROOT / "data" / "interim" / "image_labeled" / "image_labels_pilot.jsonl"
QUALITY_PATH = PROJECT_ROOT / "data" / "interim" / "image_labeled" / "image_quality_scores_pilot.jsonl"
DEDUP_PATH = PROJECT_ROOT / "data" / "interim" / "image_dedup" / "image_dedup.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_image_labeling"


class BenchmarkCandidateGenerator:
    """Generate multimodal benchmark candidates from labeled images."""

    def __init__(
        self,
        labels_path: Path = LABELS_PATH,
        quality_path: Path = QUALITY_PATH,
        dedup_path: Path = DEDUP_PATH,
        output_dir: Path = OUTPUT_DIR,
        num_workers: int = 10,
        min_domain_relevance: float = 0.6,
    ) -> None:
        self.labels_path = labels_path
        self.quality_path = quality_path
        self.dedup_path = dedup_path
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.min_domain_relevance = min_domain_relevance
        self.start_time = time.time()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pool = get_model_pool()
        self._output_lock = threading.Lock()

    def load_labels(self) -> dict[str, dict]:
        """Load label records, keyed by image_id."""
        if not self.labels_path.exists():
            logger.warning(f"Labels file not found: {self.labels_path}")
            return {}

        labels = {}
        with open(self.labels_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                labels[record["image_id"]] = record
        logger.info(f"Loaded {len(labels)} label records")
        return labels

    def load_quality(self) -> dict[str, dict]:
        """Load quality records, keyed by image_id."""
        if not self.quality_path.exists():
            logger.warning(f"Quality file not found: {self.quality_path}")
            return {}

        quality = {}
        with open(self.quality_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                quality[record["image_id"]] = record
        logger.info(f"Loaded {len(quality)} quality records")
        return quality

    def load_dedup_paths(self) -> dict[str, str]:
        """Load image file paths from dedup index."""
        paths = {}
        with open(self.dedup_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("dedup_status") == "unique":
                    paths[record["image_id"]] = record["file_path"]
        return paths

    def select_suitable_images(
        self,
        labels: dict,
        quality: dict,
        dedup_paths: dict,
    ) -> list[dict]:
        """Select images suitable for benchmark generation.

        Criteria:
        - domain_relevance ≥ min_domain_relevance
        - quality_status = "keep" or "review"
        - Has valid file path
        """
        suitable = []
        for image_id, label in labels.items():
            # Check domain relevance
            if label.get("domain_relevance", 0) < self.min_domain_relevance:
                continue

            # Check quality status
            q = quality.get(image_id, {})
            if q.get("quality_status", "drop") == "drop":
                continue

            # Check file path exists
            file_path = dedup_paths.get(image_id, "")
            if not file_path:
                continue

            suitable.append({
                "image_id": image_id,
                "file_path": file_path,
                "primary_category": label.get("primary_category", "unknown"),
                "material_form": label.get("material_form", "unknown"),
                "process_stage": label.get("process_stage", "unknown"),
                "domain_relevance": label.get("domain_relevance", 0),
                "label_confidence": label.get("label_confidence", 0),
            })

        logger.info(f"Suitable images for benchmark: {len(suitable)}")
        return suitable

    def build_benchmark_prompt(self, image_info: dict) -> list[dict]:
        """Build benchmark generation prompt with image."""
        from src.autodata.utils.image_utils import resize_image_for_api

        b64_url = resize_image_for_api(image_info["file_path"])

        user_text = build_benchmark_user_prompt(
            primary_category=image_info["primary_category"],
            material_form=image_info["material_form"],
            process_stage=image_info["process_stage"],
            domain_relevance=image_info["domain_relevance"],
        )

        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": b64_url}},
        ]

        return [
            {"role": "system", "content": BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def parse_benchmark_response(self, response_text: str, image_id: str) -> list[dict]:
        """Parse benchmark generation response into candidate dicts."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                candidates = data.get("candidates", [])
                if isinstance(candidates, list):
                    return candidates
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed for {image_id}: {str(e)[:80]}")

        return []

    def process_image(self, image_info: dict) -> list[MultimodalBenchmarkCandidate]:
        """Generate benchmark candidates for a single image."""
        image_id = image_info["image_id"]

        # Build prompt
        try:
            messages = self.build_benchmark_prompt(image_info)
        except Exception as e:
            logger.warning(f"Prompt build failed for {image_id}: {str(e)[:60]}")
            return []

        # Call LLM
        try:
            response = self.pool.chat_multimodal(
                messages=messages,
                max_completion_tokens=4096,
            )
        except Exception as e:
            logger.warning(f"LLM call failed for {image_id}: {str(e)[:80]}")
            return []

        # Parse response
        raw_candidates = self.parse_benchmark_response(response.content, image_id)

        # Build MultimodalBenchmarkCandidate records
        task_type_map = {e.value: e for e in BenchmarkTaskType}
        difficulty_map = {e.value: e for e in Difficulty}
        answerability_map = {e.value: e for e in AnswerabilityType}
        hallucination_map = {e.value: e for e in HallucinationRisk}

        candidates = []
        for raw in raw_candidates:
            candidate = MultimodalBenchmarkCandidate(
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
                source_refs=[image_info["file_path"]],
                timestamp=time.time(),
                run_id=f"phase_3_4_{int(self.start_time)}",
            )
            candidates.append(candidate)

        logger.info(f"Generated {len(candidates)} candidates for {image_id}")
        return candidates

    def run(self) -> dict[str, Any]:
        """Execute the benchmark candidate generation pipeline."""
        logger.info("=== Phase 3.4: Benchmark Candidate Generation ===")

        # Load data
        labels = self.load_labels()
        quality = self.load_quality()
        dedup_paths = self.load_dedup_paths()

        # Select suitable images
        suitable = self.select_suitable_images(labels, quality, dedup_paths)

        if not suitable:
            logger.info("No suitable images found, nothing to do")
            return {"total_candidates": 0}

        # Generate candidates
        output_path = self.output_dir / "mm_benchmark_candidates_pilot.jsonl"
        total_candidates = 0
        total_images_processed = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.process_image, img): img
                for img in suitable
            }

            for future in as_completed(futures):
                try:
                    candidates = future.result()
                    with self._output_lock:
                        for candidate in candidates:
                            with open(output_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
                            total_candidates += 1
                    total_images_processed += 1
                except Exception as e:
                    logger.warning(f"Future error: {str(e)[:80]}")

        elapsed = time.time() - self.start_time
        logger.info(
            f"=== Phase 3.4 complete: {total_candidates} candidates from "
            f"{total_images_processed} images, {elapsed:.1f}s ==="
        )

        return {
            "total_candidates": total_candidates,
            "total_images_processed": total_images_processed,
            "output_path": str(output_path),
            "elapsed_seconds": elapsed,
        }