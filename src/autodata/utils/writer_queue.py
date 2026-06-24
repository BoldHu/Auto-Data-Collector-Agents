"""Thread-safe JSONL writer using Queue + single writer thread.

Eliminates file lock contention at high concurrency by using a
single dedicated writer thread that drains a Queue of records.
Worker threads put records on the queue; the writer thread
serializes and appends them to the appropriate JSONL file.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from src.autodata.utils.logging_utils import get_logger

logger = get_logger("writer_queue")


class WriterQueue:
    """Thread-safe multi-stream JSONL writer.

    Usage:
        wq = WriterQueue({
            "labels": Path("image_labels_full.jsonl"),
            "captions": Path("image_captions_full.jsonl"),
            "quality": Path("image_quality_scores_full.jsonl"),
        })
        # Worker threads:
        wq.put("labels", label_record_dict)
        wq.put("captions", caption_record_dict)
        wq.put("quality", quality_record_dict)
        # On shutdown:
        wq.flush_and_close()

    Features:
    - Bounded queue (maxsize=1000) provides backpressure
    - Per-stream file handles kept open for efficiency
    - Immediate flush after each write for checkpoint consistency
    - Thread-safe: no locks needed, single writer thread serializes
    """

    def __init__(self, output_files: dict[str, Path], maxsize: int = 1000) -> None:
        self._queue: Queue[tuple[str, dict[str, Any]]] = Queue(maxsize=maxsize)
        self._output_files: dict[str, Path] = output_files
        self._file_handles: dict[str, Any] = {}
        self._records_written: dict[str, int] = {}
        self._total_bytes: dict[str, int] = {}
        self._shutdown = False
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        # Open file handles
        for name, path in output_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handles[name] = open(path, "a", encoding="utf-8")
            self._records_written[name] = 0
            self._total_bytes[name] = 0

        # Start writer thread (daemon so it dies if main process crashes)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        logger.info(f"WriterQueue started with {len(output_files)} streams")

    def put(self, stream_name: str, record: dict[str, Any], timeout: float = 30.0) -> None:
        """Put a record on the queue for writing.

        Blocks if queue is full (backpressure). Raises queue.Full
        if timeout elapses without space.
        """
        self._queue.put((stream_name, record), timeout=timeout)

    def put_many(self, records: dict[str, dict[str, Any]], timeout: float = 30.0) -> None:
        """Put multiple records (one per stream) on the queue."""
        for stream_name, record in records.items():
            self.put(stream_name, record, timeout=timeout)

    def _writer_loop(self) -> None:
        """Single writer thread that processes the queue."""
        while not self._shutdown or not self._queue.empty():
            try:
                stream_name, record = self._queue.get(timeout=5.0)
                line = json.dumps(record, ensure_ascii=False) + "\n"
                fh = self._file_handles.get(stream_name)
                if fh is None:
                    logger.warning(f"Unknown stream: {stream_name}, skipping record")
                    self._queue.task_done()
                    continue
                fh.write(line)
                fh.flush()
                with self._stats_lock:
                    self._records_written[stream_name] += 1
                    self._total_bytes[stream_name] += len(line)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.warning(f"WriterQueue error: {e}")

    def flush_and_close(self, timeout: float = 120.0) -> None:
        """Signal shutdown, wait for queue to drain, close all files."""
        self._shutdown = True
        # Wait for queue to drain
        self._queue.join()
        # Wait for writer thread to finish
        self._writer_thread.join(timeout=timeout)
        # Close all file handles
        for name, fh in self._file_handles.items():
            fh.close()
            logger.info(f"Closed stream '{name}': {self._records_written[name]} records, "
                        f"{self._total_bytes[name]} bytes")
        logger.info("WriterQueue shutdown complete")

    def stats(self) -> dict[str, dict[str, int]]:
        """Return per-stream record counts and bytes written."""
        with self._stats_lock:
            return {
                name: {
                    "records_written": self._records_written[name],
                    "total_bytes": self._total_bytes[name],
                }
                for name in self._output_files
            }

    def queue_size(self) -> int:
        """Return approximate current queue size."""
        return self._queue.qsize()

    def is_shutdown(self) -> bool:
        """Return whether shutdown has been signaled."""
        return self._shutdown