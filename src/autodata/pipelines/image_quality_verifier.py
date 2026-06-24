"""Image quality verifier — independent critic validation of labels, captions, and benchmark candidates.

Phase 3.5: Validate labels and benchmark candidates using a text-only quality model (mimo-v2.5-pro).

Two validation tasks:
1. Label verification: check category confidence, caption speculation, domain overclaim, visual grounding
2. Benchmark candidate verification: check answer consistency, hallucination, difficulty appropriateness

Output:
    data/reports/phase_3_image_labeling/phase_3_5_validation_report.json
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from src.autodata.pipelines.image_schema import (
    CandidateValidationRecord,
    HallucinationRisk,
    QualityVerdict,
)
from src.autodata.pipelines.prompts.image_labeling_prompts import (
    PROMPT_VERSION,
    CRITIC_SYSTEM_PROMPT,
    build_critic_user_prompt,
)
from src.autodata.utils.model_pool import get_model_pool
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("quality_verifier")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_PATH = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal" / "mm_benchmark_candidates_pilot.jsonl"
LABELS_PATH = PROJECT_ROOT / "data" / "interim" / "image_labeled" / "image_labels_pilot.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark_candidates" / "multimodal"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_3_image_labeling"


class ImageQualityVerifier:
    """Verify labels and benchmark candidates using independent critic model."""

    def __init__(
        self,
        candidates_path: Path = CANDIDATES_PATH,
        labels_path: Path = LABELS_PATH,
        output_dir: Path = OUTPUT_DIR,
        report_dir: Path = REPORT_DIR,
        num_workers: int = 10,
    ) -> None:
        self.candidates_path = candidates_path
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.num_workers = num_workers
        self.start_time = time.time()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.pool = get_model_pool()
        self._output_lock = threading.Lock()

    def load_candidates(self) -> list[dict]:
        """Load benchmark candidate records."""
        if not self.candidates_path.exists():
            logger.warning(f"Candidates file not found: {self.candidates_path}")
            return []

        candidates = []
        with open(self.candidates_path, encoding="utf-8") as f:
            for line in f:
                candidates.append(json.loads(line))
        logger.info(f"Loaded {len(candidates)} benchmark candidates")
        return candidates

    def load_labels(self) -> dict[str, dict]:
        """Load label records keyed by image_id."""
        if not self.labels_path.exists():
            return {}

        labels = {}
        with open(self.labels_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                labels[record["image_id"]] = record
        return labels

    def build_validation_prompt(self, candidate: dict, label: dict) -> str:
        """Build critic validation prompt for a benchmark candidate.

        Uses text-only prompt — no image. The critic validates based on
        the candidate's claims and the label metadata, without seeing the image.
        This prevents the critic from simply agreeing with the generator.
        """
        prompt = build_critic_user_prompt(
            task_type=candidate.get("task_type", ""),
            question=candidate.get("question", ""),
            options=str(candidate.get("options", [])),
            answer=candidate.get("answer", ""),
            explanation=candidate.get("explanation", ""),
            visual_evidence=str(candidate.get("visual_evidence", [])),
            primary_category=label.get("primary_category", "unknown"),
            label_confidence=label.get("label_confidence", 0.5),
        )

        return prompt

    def validate_candidate(
        self,
        candidate: dict,
        label: dict,
    ) -> Optional[CandidateValidationRecord]:
        """Validate a single benchmark candidate using the critic model."""
        candidate_id = candidate.get("candidate_id", "")
        image_id = candidate.get("image_id", "")

        # Build validation prompt (text-only)
        prompt = self.build_validation_prompt(candidate, label)

        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Call quality model (text-only, mimo-v2.5-pro)
        try:
            response = self.pool.chat_quality(
                messages=messages,
                max_completion_tokens=2048,
            )
        except Exception as e:
            logger.warning(f"Quality model call failed for {candidate_id}: {str(e)[:80]}")
            return None

        # Parse response
        try:
            json_start = response.content.find("{")
            json_end = response.content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response.content[json_start:json_end])
            else:
                data = {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        verdict_map = {e.value: e for e in QualityVerdict}
        # Map "review" (which the LLM may return) to NEEDS_REVISION
        verdict_map["review"] = QualityVerdict.NEEDS_REVISION
        hallucination_map = {e.value: e for e in HallucinationRisk}

        record = CandidateValidationRecord(
            candidate_id=candidate_id,
            image_id=image_id,
            validation_status=verdict_map.get(data.get("validation_status", ""), QualityVerdict.NEEDS_REVISION),
            answerability_score=float(data.get("answerability_score", 0.5)),
            visual_grounding_score=float(data.get("visual_grounding_score", 0.5)),
            domain_reasoning_score=float(data.get("domain_reasoning_score", 0.5)),
            hallucination_risk=hallucination_map.get(data.get("hallucination_risk", ""), HallucinationRisk.MEDIUM),
            ambiguity_score=float(data.get("ambiguity_score", 0.5)),
            critic_notes=data.get("critic_notes", ""),
            revision_suggestion=data.get("revision_suggestion", ""),
            critic_model="mimo-v2.5-pro",
            timestamp=time.time(),
            run_id=f"phase_3_5_{int(self.start_time)}",
        )

        return record

    def run(self) -> dict[str, Any]:
        """Execute the validation pipeline."""
        logger.info("=== Phase 3.5: Quality Verification ===")

        # Load data
        candidates = self.load_candidates()
        labels = self.load_labels()

        if not candidates:
            logger.info("No candidates to validate")
            return {"total_validated": 0}

        # Validate candidates
        output_path = self.output_dir / "mm_candidate_validation_pilot.jsonl"
        total_validated = 0
        total_passed = 0
        total_needs_revision = 0
        total_failed = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for candidate in candidates:
                label = labels.get(candidate.get("image_id", ""), {})
                future = executor.submit(self.validate_candidate, candidate, label)
                futures[future] = candidate

            for future in as_completed(futures):
                try:
                    record = future.result()
                    if record:
                        with self._output_lock:
                            with open(output_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                        total_validated += 1
                        if record.validation_status == QualityVerdict.PASSED:
                            total_passed += 1
                        elif record.validation_status == QualityVerdict.NEEDS_REVISION:
                            total_needs_revision += 1
                        elif record.validation_status == QualityVerdict.FAILED:
                            total_failed += 1
                except Exception as e:
                    logger.warning(f"Validation future error: {str(e)[:80]}")

        # Write report
        elapsed = time.time() - self.start_time
        report = {
            "phase": "3.5",
            "run_id": f"phase_3_5_{int(self.start_time)}",
            "timestamp": time.time(),
            "total_candidates_validated": total_validated,
            "total_passed": total_passed,
            "total_needs_revision": total_needs_revision,
            "total_failed": total_failed,
            "pass_rate": total_passed / total_validated if total_validated else 0,
            "elapsed_seconds": elapsed,
            "errors": [],
        }

        report_path = self.report_dir / "phase_3_5_validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(
            f"=== Phase 3.5 complete: {total_validated} validated, "
            f"{total_passed} passed, {total_needs_revision} needs_revision, {total_failed} failed, {elapsed:.1f}s ==="
        )

        return {
            "total_validated": total_validated,
            "total_passed": total_passed,
            "total_needs_revision": total_needs_revision,
            "total_failed": total_failed,
            "elapsed_seconds": elapsed,
        }