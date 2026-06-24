"""Phase 5 concurrency controller for high-throughput benchmark construction.

Starts with 16 workers, scales up to 48 if stable.
"""

from __future__ import annotations

import time
from threading import Lock


class Phase5ConcurrencyController:
    """Adaptive concurrency controller for Phase 5 benchmark construction.

    Features:
    - Start 16 workers, scale up by 8 every 5 min if error < 2%
    - Scale down 50% if error rate > 5%
    - Scale down if JSON parse failure > 8%
    - Max 48 workers, min 8 workers
    - Per-stage speed and error statistics
    """

    def __init__(
        self,
        initial_workers: int = 16,
        min_workers: int = 8,
        max_workers: int = 48,
        step_size: int = 8,
        evaluation_interval_seconds: int = 300,
        stable_threshold: float = 0.02,
        error_threshold: float = 0.05,
        json_parse_failure_threshold: float = 0.08,
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
        self._total_retries = 0

        # API stats
        self._api_success_count = 0
        self._api_error_count = 0
        self._latencies: list[float] = []

        # Per-stage stats
        self._stage_stats: dict[str, dict] = {}
        self._current_stage = "unknown"

    def set_stage(self, stage: str) -> None:
        self._current_stage = stage
        if stage not in self._stage_stats:
            self._stage_stats[stage] = {
                "completed": 0,
                "errors": 0,
                "json_parse_failures": 0,
                "start_time": time.monotonic(),
            }

    def record_success(self, latency: float = 0.0) -> None:
        with self._lock:
            self._window_completed += 1
            self._total_completed += 1
            self._api_success_count += 1
            if latency > 0:
                self._latencies.append(latency)
                if len(self._latencies) > 1000:
                    self._latencies = self._latencies[-500:]
            if self._current_stage in self._stage_stats:
                self._stage_stats[self._current_stage]["completed"] += 1

    def record_error(self) -> None:
        with self._lock:
            self._window_errors += 1
            self._total_errors += 1
            self._api_error_count += 1
            if self._current_stage in self._stage_stats:
                self._stage_stats[self._current_stage]["errors"] += 1

    def record_json_parse_failure(self) -> None:
        with self._lock:
            self._window_json_parse_failures += 1
            self._total_json_parse_failures += 1
            if self._current_stage in self._stage_stats:
                self._stage_stats[self._current_stage]["json_parse_failures"] += 1

    def record_retry(self) -> None:
        with self._lock:
            self._total_retries += 1

    def current_workers(self) -> int:
        with self._lock:
            return self._current_workers

    def maybe_adjust(self) -> int:
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

            if error_rate > self._error_threshold:
                self._current_workers = max(
                    self._min_workers, self._current_workers // 2
                )
                self._total_scale_downs += 1
            elif json_failure_rate > self._json_parse_failure_threshold:
                self._current_workers = max(
                    self._min_workers, self._current_workers - self._step_size
                )
                self._total_scale_downs += 1
            elif error_rate < self._stable_threshold:
                self._current_workers = min(
                    self._max_workers, self._current_workers + self._step_size
                )
                self._total_scale_ups += 1

            self._reset_window()
            return self._current_workers

    def _reset_window(self) -> None:
        self._window_start = time.monotonic()
        self._window_completed = 0
        self._window_errors = 0
        self._window_json_parse_failures = 0

    def average_latency(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    def report(self) -> dict:
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
                "total_completed": self._total_completed,
                "total_errors": self._total_errors,
                "total_json_parse_failures": self._total_json_parse_failures,
                "total_retries": self._total_retries,
                "total_scale_ups": self._total_scale_ups,
                "total_scale_downs": self._total_scale_downs,
                "api_success_count": self._api_success_count,
                "api_error_count": self._api_error_count,
                "average_latency": self.average_latency(),
                "stage_stats": self._stage_stats,
            }
