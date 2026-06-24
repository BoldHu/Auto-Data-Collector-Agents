"""Real-time progress tracker for long-running pipeline jobs.

Tracks files, pages, chunks, LLM calls, tokens, and estimated time.
Writes progress to JSON file and append-only log file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.autodata.utils.io_utils import atomic_write_json, ensure_dir


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class ProgressSnapshot:
    """Immutable snapshot of pipeline progress at a point in time."""
    run_id: str = ""
    language: str = ""
    mode: str = ""
    start_time: float = 0.0
    elapsed: float = 0.0
    total_files: int = 0
    completed_files: int = 0
    current_file: str = ""
    total_pages: int = 0
    completed_pages: int = 0
    total_chunks: int = 0
    completed_chunks: int = 0
    cleaned_chunks: int = 0
    verified_chunks: int = 0
    quality_scores_written: int = 0
    knowledge_units_generated: int = 0
    sft_candidates_generated: int = 0
    llm_calls_completed: int = 0
    tokens_used: int = 0
    avg_seconds_per_chunk: float = 0.0
    avg_seconds_per_llm_call: float = 0.0
    chunks_per_hour: float = 0.0
    llm_calls_per_hour: float = 0.0
    estimated_remaining_seconds: float = 0.0
    estimated_finish_time: float = 0.0
    current_stage: str = ""
    recent_errors: list[str] = field(default_factory=list)
    checkpoint_path: str = ""
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "language": self.language,
            "mode": self.mode,
            "start_time": self.start_time,
            "elapsed": round(self.elapsed, 1),
            "elapsed_formatted": _fmt_duration(self.elapsed),
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "current_file": self.current_file,
            "total_pages": self.total_pages,
            "completed_pages": self.completed_pages,
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "cleaned_chunks": self.cleaned_chunks,
            "verified_chunks": self.verified_chunks,
            "quality_scores_written": self.quality_scores_written,
            "knowledge_units_generated": self.knowledge_units_generated,
            "sft_candidates_generated": self.sft_candidates_generated,
            "llm_calls_completed": self.llm_calls_completed,
            "tokens_used": self.tokens_used,
            "avg_seconds_per_chunk": round(self.avg_seconds_per_chunk, 1),
            "avg_seconds_per_llm_call": round(self.avg_seconds_per_llm_call, 1),
            "chunks_per_hour": round(self.chunks_per_hour, 1),
            "llm_calls_per_hour": round(self.llm_calls_per_hour, 1),
            "estimated_remaining_seconds": round(self.estimated_remaining_seconds, 1),
            "estimated_remaining_formatted": _fmt_duration(self.estimated_remaining_seconds),
            "estimated_finish_time": self.estimated_finish_time,
            "estimated_finish_formatted": (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.estimated_finish_time))
                if self.estimated_finish_time > 0 else ""
            ),
            "current_stage": self.current_stage,
            "recent_errors": self.recent_errors[-5:],
            "checkpoint_path": self.checkpoint_path,
            "output_paths": self.output_paths,
        }


class ProgressTracker:
    """Real-time progress tracker with JSON and log output.

    Usage:
        tracker = ProgressTracker(
            run_id="phase_2_full_zh",
            json_path="data/reports/.../phase_2_full_zh_progress.json",
            log_path="data/reports/.../phase_2_full_zh_progress.log",
            language="zh",
            mode="full",
        )
        tracker.set_totals(total_files=38, total_pages=10066, total_chunks=3500)
        tracker.start()

        # ... during pipeline ...
        tracker.on_file_start("some_file.clean.json")
        tracker.on_pages_completed(15)
        tracker.on_chunk_cleaned()
        tracker.on_chunk_verified()
        tracker.on_quality_written()
        tracker.on_knowledge_units(5)
        tracker.on_sft_candidates(3)
        tracker.on_llm_call(tokens=1200)
        tracker.on_error("chunk_abc: JSON parse failed")

        # Periodically (every 5 chunks or 2 min):
        tracker.maybe_report(force=False)

        # At end:
        tracker.finish()
    """

    def __init__(
        self,
        run_id: str,
        json_path: str | Path,
        log_path: str | Path,
        language: str = "",
        mode: str = "",
        report_interval_chunks: int = 5,
        report_interval_seconds: float = 120.0,
    ) -> None:
        self.run_id = run_id
        self.language = language
        self.mode = mode
        self.json_path = Path(json_path)
        self.log_path = Path(log_path)
        self.report_interval_chunks = report_interval_chunks
        self.report_interval_seconds = report_interval_seconds

        ensure_dir(str(self.json_path.parent))
        ensure_dir(str(self.log_path.parent))

        # Internal counters
        self._start_time: float = 0.0
        self._total_files: int = 0
        self._total_pages: int = 0
        self._total_chunks: int = 0
        self._completed_files: int = 0
        self._completed_pages: int = 0
        self._completed_chunks: int = 0
        self._cleaned_chunks: int = 0
        self._verified_chunks: int = 0
        self._quality_scores_written: int = 0
        self._knowledge_units: int = 0
        self._sft_candidates: int = 0
        self._llm_calls: int = 0
        self._tokens_used: int = 0
        self._current_file: str = ""
        self._current_stage: str = "initialized"
        self._errors: list[str] = []
        self._checkpoint_path: str = ""
        self._output_paths: dict[str, str] = {}

        # Timing for report interval
        self._last_report_time: float = 0.0
        self._last_report_chunks: int = 0
        self._chunks_since_report: int = 0

    def set_totals(
        self,
        total_files: int = 0,
        total_pages: int = 0,
        total_chunks: int = 0,
    ) -> None:
        """Set total counts (may be estimates)."""
        self._total_files = total_files
        self._total_pages = total_pages
        self._total_chunks = total_chunks

    def set_checkpoint_path(self, path: str) -> None:
        self._checkpoint_path = path

    def set_output_paths(self, paths: dict[str, str]) -> None:
        self._output_paths = paths

    def start(self) -> None:
        """Mark the start of the run."""
        self._start_time = time.time()
        self._last_report_time = self._start_time
        self._current_stage = "running"
        self._write_progress()

    def on_file_start(self, filename: str) -> None:
        """Mark the start of processing a new file."""
        self._current_file = filename
        self._current_stage = f"processing: {filename}"

    def on_file_complete(self) -> None:
        """Mark a file as completed."""
        self._completed_files += 1
        self._current_file = ""

    def on_pages_completed(self, count: int) -> None:
        """Add completed pages."""
        self._completed_pages += count

    def on_chunk_cleaned(self) -> None:
        """Mark a chunk as cleaned."""
        self._completed_chunks += 1
        self._cleaned_chunks += 1
        self._chunks_since_report += 1

    def on_chunk_verified(self) -> None:
        """Mark a chunk as verified."""
        self._verified_chunks += 1

    def on_quality_written(self) -> None:
        """Mark a quality score written."""
        self._quality_scores_written += 1

    def on_knowledge_units(self, count: int) -> None:
        """Add knowledge units generated."""
        self._knowledge_units += count

    def on_sft_candidates(self, count: int) -> None:
        """Add SFT candidates generated."""
        self._sft_candidates += count

    def on_llm_call(self, tokens: int = 0) -> None:
        """Record an LLM call."""
        self._llm_calls += 1
        self._tokens_used += tokens

    def on_error(self, error_msg: str) -> None:
        """Record an error."""
        self._errors.append(error_msg)

    def snapshot(self) -> ProgressSnapshot:
        """Create a progress snapshot with computed fields."""
        now = time.time()
        elapsed = now - self._start_time if self._start_time > 0 else 0.0

        # Averages
        avg_s_per_chunk = elapsed / max(self._completed_chunks, 1)
        avg_s_per_llm = elapsed / max(self._llm_calls, 1)
        chunks_per_hr = self._completed_chunks / max(elapsed / 3600, 0.001)
        llm_per_hr = self._llm_calls / max(elapsed / 3600, 0.001)

        # ETA: based on remaining chunks × avg time per chunk
        remaining_chunks = max(self._total_chunks - self._completed_chunks, 0)
        est_remaining = remaining_chunks * avg_s_per_chunk
        est_finish = now + est_remaining

        return ProgressSnapshot(
            run_id=self.run_id,
            language=self.language,
            mode=self.mode,
            start_time=self._start_time,
            elapsed=elapsed,
            total_files=self._total_files,
            completed_files=self._completed_files,
            current_file=self._current_file,
            total_pages=self._total_pages,
            completed_pages=self._completed_pages,
            total_chunks=self._total_chunks,
            completed_chunks=self._completed_chunks,
            cleaned_chunks=self._cleaned_chunks,
            verified_chunks=self._verified_chunks,
            quality_scores_written=self._quality_scores_written,
            knowledge_units_generated=self._knowledge_units,
            sft_candidates_generated=self._sft_candidates,
            llm_calls_completed=self._llm_calls,
            tokens_used=self._tokens_used,
            avg_seconds_per_chunk=avg_s_per_chunk,
            avg_seconds_per_llm_call=avg_s_per_llm,
            chunks_per_hour=chunks_per_hr,
            llm_calls_per_hour=llm_per_hr,
            estimated_remaining_seconds=est_remaining,
            estimated_finish_time=est_finish,
            current_stage=self._current_stage,
            recent_errors=list(self._errors[-5:]),
            checkpoint_path=self._checkpoint_path,
            output_paths=dict(self._output_paths),
        )

    def _write_progress(self) -> None:
        """Write progress JSON and append to log."""
        snap = self.snapshot()
        data = snap.to_dict()

        # Atomic write JSON
        atomic_write_json(str(self.json_path), data)

        # Append to log
        log_line = (
            f"[{self.run_id}] "
            f"files {snap.completed_files}/{snap.total_files} | "
            f"pages {snap.completed_pages}/{snap.total_pages} | "
            f"chunks {snap.completed_chunks}/{snap.total_chunks} | "
            f"cleaned {snap.cleaned_chunks} | "
            f"verified {snap.verified_chunks} | "
            f"KU {snap.knowledge_units_generated} | "
            f"SFT {snap.sft_candidates_generated} | "
            f"elapsed {_fmt_duration(snap.elapsed)} | "
            f"ETA {_fmt_duration(snap.estimated_remaining_seconds)} | "
            f"avg {snap.avg_seconds_per_chunk:.1f}s/chunk | "
            f"LLM calls {snap.llm_calls_completed}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

    def maybe_report(self, force: bool = False) -> None:
        """Write progress if enough chunks or time has elapsed."""
        now = time.time()
        time_since = now - self._last_report_time
        if force or self._chunks_since_report >= self.report_interval_chunks or time_since >= self.report_interval_seconds:
            self._write_progress()
            self._last_report_time = now
            self._chunks_since_report = 0

    def report(self) -> str:
        """Return formatted progress line for terminal output."""
        snap = self.snapshot()
        return (
            f"[{self.run_id}] "
            f"files {snap.completed_files}/{snap.total_files} | "
            f"pages {snap.completed_pages}/{snap.total_pages} | "
            f"chunks {snap.completed_chunks}/{snap.total_chunks} | "
            f"cleaned {snap.cleaned_chunks} | "
            f"verified {snap.verified_chunks} | "
            f"KU {snap.knowledge_units_generated} | "
            f"SFT {snap.sft_candidates_generated} | "
            f"elapsed {_fmt_duration(snap.elapsed)} | "
            f"ETA {_fmt_duration(snap.estimated_remaining_seconds)} | "
            f"avg {snap.avg_seconds_per_chunk:.1f}s/chunk | "
            f"LLM calls {snap.llm_calls_completed}"
        )

    def finish(self) -> None:
        """Mark the run as finished and write final progress."""
        self._current_stage = "completed"
        self._write_progress()
