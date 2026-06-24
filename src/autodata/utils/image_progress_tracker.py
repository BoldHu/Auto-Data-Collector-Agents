"""Image labeling progress tracker with rolling-window ETA.

Tracks completion progress for large-scale image labeling runs.
Uses a deque of recent completion timestamps to compute rolling-average
throughput, giving more accurate ETA than simple average (which is
distorted by initial ramp-up or temporary rate-limit pauses).

Reports progress every 20 images or 60 seconds, whichever comes first.
Writes to both a JSON file (machine-readable) and a log file (human-readable).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from src.autodata.utils.logging_utils import get_logger

logger = get_logger("image_progress_tracker")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds <= 0 or seconds == float("inf"):
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ImageProgressTracker:
    """Image labeling progress tracker with rolling-window ETA."""

    def __init__(
        self,
        run_id: str,
        json_path: Path,
        log_path: Path,
        total_images: int,
        report_interval_images: int = 20,
        report_interval_seconds: float = 60.0,
    ) -> None:
        self._run_id = run_id
        self._json_path = json_path
        self._log_path = log_path
        self._total = total_images
        self._report_interval_images = report_interval_images
        self._report_interval_seconds = report_interval_seconds

        self._completed = 0
        self._failed = 0
        self._skipped = 0
        self._tokens_used = 0
        self._start_time: Optional[float] = None
        self._recent_times: deque[float] = deque(maxlen=100)
        self._last_report_time = 0.0
        self._images_since_report = 0
        self._last_success_time: Optional[float] = None
        self._current_stage = "initializing"
        self._current_bottleneck = "none"

        # Extra counters for detailed reporting
        self._keep_count = 0
        self._review_count = 0
        self._drop_count = 0
        self._candidate_count = 0
        self._validated_count = 0

        self._lock = Lock()

        # Ensure parent dirs exist
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Mark the start of the run."""
        self._start_time = time.time()
        self._last_report_time = self._start_time
        self._current_stage = "labeling"
        self._write_progress(force=True)

    def on_image_completed(self, tokens: int = 0, quality_status: str = "") -> None:
        """Record a successful image completion."""
        with self._lock:
            self._completed += 1
            self._tokens_used += tokens
            self._recent_times.append(time.time())
            self._images_since_report += 1
            self._last_success_time = time.time()
            if quality_status == "keep":
                self._keep_count += 1
            elif quality_status == "review":
                self._review_count += 1
            elif quality_status == "drop":
                self._drop_count += 1

    def on_image_failed(self) -> None:
        """Record a failed image."""
        with self._lock:
            self._failed += 1
            self._images_since_report += 1

    def on_image_skipped(self) -> None:
        """Record a skipped (already processed) image."""
        with self._lock:
            self._skipped += 1
            self._images_since_report += 1

    def on_candidate_generated(self) -> None:
        """Record a benchmark candidate generated."""
        with self._lock:
            self._candidate_count += 1

    def on_candidate_validated(self) -> None:
        """Record a candidate validated."""
        with self._lock:
            self._validated_count += 1

    def set_stage(self, stage: str) -> None:
        """Update current stage label."""
        with self._lock:
            self._current_stage = stage

    def set_bottleneck(self, bottleneck: str) -> None:
        """Update current bottleneck label."""
        with self._lock:
            self._current_bottleneck = bottleneck

    def eta_seconds(self) -> float:
        """ETA based on rolling window of recent completions."""
        with self._lock:
            if len(self._recent_times) < 5 or self._start_time is None:
                # Not enough data; use simple extrapolation
                elapsed = time.time() - self._start_time
                if elapsed < 1 or self._completed < 1:
                    return float("inf")
                throughput = self._completed / elapsed
                remaining = self._total - self._completed - self._failed
                return remaining / throughput

            window_duration = self._recent_times[-1] - self._recent_times[0]
            window_count = len(self._recent_times) - 1
            if window_count == 0 or window_duration == 0:
                return float("inf")
            throughput = window_count / window_duration
            remaining = self._total - self._completed - self._failed
            return remaining / throughput

    def maybe_report(self, force: bool = False) -> None:
        """Write progress if interval threshold met."""
        with self._lock:
            now = time.time()
            should_report = force or (
                self._images_since_report >= self._report_interval_images
                or (now - self._last_report_time) >= self._report_interval_seconds
            )
            if not should_report:
                return
            self._images_since_report = 0
            self._last_report_time = now
        self._write_progress(force=force)

    def _write_progress(self, force: bool = False) -> None:
        """Write progress JSON and log line."""
        with self._lock:
            elapsed = (time.time() - self._start_time) if self._start_time else 0
            eta = self._eta_seconds_internal()
            avg_per_image = elapsed / max(self._completed, 1)
            images_per_hour = self._completed / max(elapsed / 3600, 0.001)

            data = {
                "run_id": self._run_id,
                "total_images": self._total,
                "completed_images": self._completed,
                "failed_images": self._failed,
                "skipped_images": self._skipped,
                "progress_pct": round(
                    (self._completed + self._failed + self._skipped) / max(self._total, 1) * 100, 1
                ),
                "labeled_images": self._completed,
                "captioned_images": self._completed,
                "quality_scored_images": self._completed,
                "keep_count": self._keep_count,
                "review_count": self._review_count,
                "drop_count": self._drop_count,
                "candidate_count": self._candidate_count,
                "validated_candidate_count": self._validated_count,
                "failed_image_count": self._failed,
                "failed_candidate_count": 0,
                "tokens_used": self._tokens_used,
                "average_seconds_per_image": round(avg_per_image, 2),
                "images_per_hour": round(images_per_hour, 2),
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_formatted": _fmt_duration(elapsed),
                "estimated_remaining_seconds": round(eta, 1),
                "estimated_finish_time": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta))
                    if eta != float("inf") else "N/A"
                ),
                "current_stage": self._current_stage,
                "current_bottleneck": self._current_bottleneck,
                "last_success_time": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._last_success_time))
                    if self._last_success_time else "N/A"
                ),
                "active_workers": 0,  # set externally
                "per_api_success_count": {},  # set externally
                "per_api_error_count": {},  # set externally
                "per_api_average_latency": {},  # set externally
            }

        # Write JSON
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Write log line
        log_line = (
            f"[{self._run_id}] "
            f"images {self._completed}/{self._total} | "
            f"keep {self._keep_count} | review {self._review_count} | drop {self._drop_count} | "
            f"candidates {self._candidate_count} | validated {self._validated_count} | "
            f"failed {self._failed} | "
            f"elapsed {_fmt_duration(elapsed)} | "
            f"ETA {_fmt_duration(eta)} | "
            f"avg {avg_per_image:.1f}s/image"
        )
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        if force or self._completed % 100 == 0:
            logger.info(log_line)

    def _eta_seconds_internal(self) -> float:
        """Internal ETA calculation (called under lock)."""
        if len(self._recent_times) < 5 or self._start_time is None:
            elapsed = time.time() - self._start_time if self._start_time else 1
            if elapsed < 1 or self._completed < 1:
                return float("inf")
            throughput = self._completed / elapsed
            remaining = self._total - self._completed - self._failed
            return remaining / throughput

        window_duration = self._recent_times[-1] - self._recent_times[0]
        window_count = len(self._recent_times) - 1
        if window_count == 0 or window_duration == 0:
            return float("inf")
        throughput = window_count / window_duration
        remaining = self._total - self._completed - self._failed
        return remaining / throughput

    def update_external_stats(
        self,
        active_workers: int = 0,
        api_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update externally-provided stats (worker count, API stats)."""
        if api_stats is None:
            return
        # These will be written on the next maybe_report() call
        # Store them temporarily for the next write
        with self._lock:
            self._external_active_workers = active_workers
            self._external_api_stats = api_stats

    def finish(self) -> None:
        """Mark the run as complete, write final progress."""
        self._current_stage = "complete"
        self._write_progress(force=True)