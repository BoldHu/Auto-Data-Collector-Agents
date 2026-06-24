"""Exam parser orchestrator.

Orchestrates document conversion and text extraction for exam files.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def parse_exam_files(exam_dir: Path, output_dir: Path) -> dict:
    """Parse all exam files and extract text blocks.

    Args:
        exam_dir: Directory containing exam files
        output_dir: Directory for output JSONL files

    Returns:
        Parse results dict
    """
    from src.autodata.tools.document_converter import convert_document

    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = output_dir / "exam_text_blocks.jsonl"
    errors_path = output_dir / "exam_extraction_errors.jsonl"

    total_files = 0
    total_blocks = 0
    total_errors = 0

    with open(blocks_path, "w") as blocks_f, open(errors_path, "w") as errors_f:
        for file_path in sorted(exam_dir.iterdir()):
            if file_path.is_dir() or file_path.name.startswith("."):
                continue

            total_files += 1
            blocks = convert_document(file_path)

            for block in blocks:
                blocks_f.write(json.dumps(block, ensure_ascii=False) + "\n")
                total_blocks += 1

                if block.get("extraction_method") == "error":
                    errors_f.write(json.dumps(block, ensure_ascii=False) + "\n")
                    total_errors += 1

    return {
        "total_files": total_files,
        "total_blocks": total_blocks,
        "total_errors": total_errors,
        "blocks_path": str(blocks_path),
        "errors_path": str(errors_path),
    }
