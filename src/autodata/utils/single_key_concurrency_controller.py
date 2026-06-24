"""Single-key concurrency controller for Phase 4.

Adaptive concurrency controller optimized for single API key operation.
Starts with 8 workers, scales up to 32 if stable, scales down if errors rise.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from threading import Lock


class SingleKeyConcurrencyController:
    """Adaptive concurrency controller for single API key mode.

    Features:
    - Start 8 workers, scale up by 4 every 5 min if error < 2%
    - Scale down 50% if error rate > 5%
    - Scale down if JSON parse failure > 10%
    - Max 32 workers, min 4 workers
    - Per-stage speed and error statistics
    - Graceful shutdown support
    """

    def __init__(
        self,
        initial_workers: int = 8,
        min_workers: int = 4,
        max_workers: int = 32,
        step_size: int = 4,
        evaluation_interval_seconds: int = 300,
        stable_threshold: float = 0.02,
        error_threshold: float = 0.05,
        json_parse_failure_threshold: float = 0.10,
    ) -> None:
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._step_size = step_size
        self._evaluation_interval = evaluation_interval_seconds
        self._stable_threshold = stable_threshold
        self._error_threshold = error_threshold
        self._json_parse_failure_threshold = json_parse_failure_threshold

        self._current_workers = initial_workers
        self._lock = Lock()

        # Sliding window stats
        self._window_start = time.monotonic()
        self._window_completed = 0
        self._window_errors = 0
        self._window_json_parse_failures = 0

        # Total stats
        self._total_completed = 0
        self._total_errors = 0
        self._total_json_parse_failures = 0
        self._total_scale_ups = 0
        self._total_scale_downs = 0

        # Per-stage stats
        self._stage_stats: dict[str, dict] = {}
        self._current_stage = "unknown"

    def set_stage(self, stage: str) -> None:
        """Set current processing stage for stats tracking."""
        self._current_stage = stage
        if stage not in self._stage_stats:
            self._stage_stats[stage] = {
                "completed": 0,
                "errors": 0,
                "json_parse_failures": 0,
                "start_time": time.monotonic(),
            }

    def record_success(self) -> None:
        """Record a successful processing."""
        with self._lock:
            self._window_completed += 1
            self._total_completed += 1
            if self._current_stage in self._stage_stats:
                self._stage_stats[self._current_stage]["completed"] += 1

    def record_error(self) -> None:
        """Record a processing error."""
        with self._lock:
            self._window_errors += 1
            self._total_errors += 1
            if self._current_stage in self._stage_stats:
                self._stage_stats[self._current_stage]["errors"] += 1

    def record_json_parse_failure(self) -> None:
        """Record a JSON parse failure."""
        with self._lock:
            self._window_json_parse_failures += 1
            self._total_json_parse_failures += 1
            if self._current_stage in self._stage_stats:
                self._stage_stats[self._current_stage]["json_parse_failures"] += 1

    def current_workers(self) -> int:
        """Get current target worker count."""
        with self._lock:
            return self._current_workers

    def maybe_adjust(self) -> int:
        """Evaluate and potentially adjust worker count.

        Returns:
            Current worker count after adjustment.
        """
        with self._lock:
            elapsed = time.monotonic() - self._window_start
            if elapsed < self._evaluation_interval:
                return self._current_workers

            total_in_window = self._window_completed + self._window_errors
            if total_in_window < 10:
                self._reset_window()
                return self._current_workers

            error_rate = self._window_errors / total_in_window
            json_failure_rate = (
                self._window_json_parse_failures / total_in_window
                if total_in_window > 0
                else 0
            )

            old_workers = self._current_workers

            # Scale down on high error rate
            if error_rate > self._error_threshold:
                self._current_workers = max(
                    self._min_workers, self._current_workers // 2
                )
                self._total_scale_downs += 1
            # Scale down on high JSON parse failure rate
            elif json_failure_rate > self._json_parse_failure_threshold:
                self._current_workers = max(
                    self._min_workers, self._current_workers - self._step_size
                )
                self._total_scale_downs += 1
            # Scale up if stable
            elif error_rate < self._stable_threshold:
                self._current_workers = min(
                    self._max_workers, self._current_workers + self._step_size
                )
                self._total_scale_ups += 1

            if self._current_workers != old_workers:
                pass  # Could log here

            self._reset_window()
            return self._current_workers

    def _reset_window(self) -> None:
        """Reset sliding window counters."""
        self._window_start = time.monotonic()
        self._window_completed = 0
        self._window_errors = 0
        self._window_json_parse_failures = 0

    def report(self) -> dict:
        """Get full status report."""
        with self._lock:
            elapsed = time.monotonic() - self._window_start
            total_in_window = self._window_completed + self._window_errors
            return {
                "current_workers": self._current_workers,
                "min_workers": self._min_workers,
                "max_workers": self._max_workers,
                "window_completed": self._window_completed,
                "window_errors": self._window_errors,
                "window_json_parse_failures": self._window_json_parse_failures,
                "window_error_rate": (
                    self._window_errors / total_in_window
                    if total_in_window > 0
                    else 0
                ),
                "window_elapsed_seconds": elapsed,
                "total_completed": self._total_completed,
                "total_errors": self._total_errors,
                "total_json_parse_failures": self._total_json_parse_failures,
                "total_scale_ups": self._total_scale_ups,
                "total_scale_downs": self._total_scale_downs,
                "stage_stats": self._stage_stats,
            }
