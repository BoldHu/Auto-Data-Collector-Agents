"""Adaptive concurrency controller for full-scale pipelines.

Monitors error rates and scales worker count dynamically:
- Start at initial_workers (default 8)
- Every evaluation_interval (5 min), check error rate
- If stable (<2% errors), scale up by step_size (4)
- If high errors (>5%), scale down by 50%
- Max workers capped at max_workers (16)
- Min workers capped at min_workers (4)

This is a pipeline-level controller. ModelPool itself is not modified.
The pipeline uses graduated submission: ThreadPoolExecutor is created
with max_workers capacity, but only current_workers tasks are submitted
initially. More tasks are submitted as the controller scales up.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from src.autodata.utils.logging_utils import get_logger

logger = get_logger("adaptive_concurrency")


class AdaptiveConcurrencyController:
    """Adaptive concurrency scaling based on error rate monitoring."""

    def __init__(
        self,
        initial_workers: int = 8,
        min_workers: int = 4,
        max_workers: int = 16,
        step_size: int = 4,
        evaluation_interval_seconds: float = 300.0,
        stable_threshold: float = 0.02,
        error_threshold: float = 0.05,
        json_parse_failure_threshold: float = 0.10,
    ) -> None:
        self._current_workers = initial_workers
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._step_size = step_size
        self._eval_interval = evaluation_interval_seconds
        self._stable_threshold = stable_threshold
        self._error_threshold = error_threshold
        self._json_parse_threshold = json_parse_failure_threshold

        # Counters for current evaluation window
        self._window_completed = 0
        self._window_errors = 0
        self._window_json_parse_failures = 0
        self._last_eval_time = time.time()

        # Cumulative counters for reporting
        self._total_completed = 0
        self._total_errors = 0
        self._total_scale_ups = 0
        self._total_scale_downs = 0

        self._lock = threading.Lock()
        logger.info(
            f"AdaptiveConcurrencyController initialized: "
            f"workers={initial_workers}, min={min_workers}, max={max_workers}, "
            f"step={step_size}, eval_interval={evaluation_interval_seconds}s"
        )

    def record_success(self) -> None:
        """Record a successful task completion."""
        with self._lock:
            self._window_completed += 1
            self._total_completed += 1

    def record_error(self) -> None:
        """Record a failed task."""
        with self._lock:
            self._window_errors += 1
            self._total_errors += 1

    def record_json_parse_failure(self) -> None:
        """Record a JSON parse failure (separate from API errors)."""
        with self._lock:
            self._window_json_parse_failures += 1

    def current_workers(self) -> int:
        """Get current target worker count."""
        with self._lock:
            return self._current_workers

    def maybe_adjust(self) -> int:
        """Evaluate error rate and potentially adjust worker count.

        Returns the new (or unchanged) worker count.
        Called periodically by the pipeline (e.g., after each task completes).
        """
        now = time.time()
        with self._lock:
            elapsed = now - self._last_eval_time
            if elapsed < self._eval_interval:
                return self._current_workers

            total_window = self._window_completed + self._window_errors
            if total_window < 20:
                # Need minimum sample size before adjusting
                return self._current_workers

            error_rate = self._window_errors / total_window
            json_failure_rate = self._window_json_parse_failures / total_window

            old_workers = self._current_workers

            if error_rate > self._error_threshold:
                # Scale down by 50%
                self._current_workers = max(
                    self._min_workers, self._current_workers // 2
                )
                self._total_scale_downs += 1
                logger.warning(
                    f"High error rate {error_rate:.2%} ({self._window_errors}/{total_window}), "
                    f"scaling down from {old_workers} to {self._current_workers} workers"
                )
            elif json_failure_rate > self._json_parse_threshold:
                # Reduce workers moderately for JSON quality issues
                self._current_workers = max(
                    self._min_workers, self._current_workers - self._step_size
                )
                self._total_scale_downs += 1
                logger.warning(
                    f"High JSON parse failure rate {json_failure_rate:.2%}, "
                    f"reducing from {old_workers} to {self._current_workers} workers"
                )
            elif error_rate < self._stable_threshold and json_failure_rate < self._json_parse_threshold / 2:
                # Stable: scale up
                self._current_workers = min(
                    self._max_workers, self._current_workers + self._step_size
                )
                self._total_scale_ups += 1
                if self._current_workers != old_workers:
                    logger.info(
                        f"Stable ({error_rate:.2%} errors), "
                        f"scaling up from {old_workers} to {self._current_workers} workers"
                    )

            # Reset window counters
            self._window_completed = 0
            self._window_errors = 0
            self._window_json_parse_failures = 0
            self._last_eval_time = now
            return self._current_workers

    def report(self) -> dict[str, Any]:
        """Return status report for progress tracking."""
        with self._lock:
            total_window = self._window_completed + self._window_errors
            error_rate = self._window_errors / max(total_window, 1)
            total_all = self._total_completed + self._total_errors
            overall_error_rate = self._total_errors / max(total_all, 1)

            return {
                "current_workers": self._current_workers,
                "min_workers": self._min_workers,
                "max_workers": self._max_workers,
                "window_completed": self._window_completed,
                "window_errors": self._window_errors,
                "window_error_rate": round(error_rate, 4),
                "total_completed": self._total_completed,
                "total_errors": self._total_errors,
                "overall_error_rate": round(overall_error_rate, 4),
                "total_scale_ups": self._total_scale_ups,
                "total_scale_downs": self._total_scale_downs,
                "last_eval_time": self._last_eval_time,
            }

    def reset_window(self) -> None:
        """Reset window counters (useful after a pause or resume)."""
        with self._lock:
            self._window_completed = 0
            self._window_errors = 0
            self._window_json_parse_failures = 0
            self._last_eval_time = time.time()