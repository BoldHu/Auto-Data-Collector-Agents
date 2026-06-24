"""Retry failed chunks from phase_2_7 cleaning.

Extracts chunk data from error JSONL files, filters out already-processed
chunks, and re-runs the cleaning pipeline on them.

Usage:
    python scripts/retry_phase_2_7_failed_chunks.py --language zh --num_workers 20
    python scripts/retry_phase_2_7_failed_chunks.py --language en --num_workers 20
    python scripts/retry_phase_2_7_failed_chunks.py --language both --num_workers 20
"""

import argparse
import json
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.utils.model_pool import get_model_pool
from src.autodata.pipelines.prompts.text_cleaning_prompts import get_cleaning_prompt
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("retry_failed")

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_2_7_restart_cleaning"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "text_cleaned"


def parse_v2_cleaning_response(response_text: str, raw_text: str) -> dict:
    """Parse v2.0 JSON cleaning response."""
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(response_text[json_start:json_end])
            return {
                "cleaned_text": data.get("cleaned_text", ""),
                "enriched_notes": data.get("enriched_notes", ""),
                "keep_for_corpus": data.get("keep_for_corpus", True),
                "drop_reason": data.get("drop_reason", ""),
                "removed_noise_types": data.get("removed_noise_types", []),
                "ocr_repairs": data.get("ocr_repairs", []),
                "technical_content_types": data.get("technical_content_types", []),
                "uncertainty_notes": data.get("uncertainty_notes", ""),
                "cleaning_actions": data.get("cleaning_actions", []),
                "confidence": float(data.get("confidence", 0.5)),
                "parse_success": True,
            }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse failed: {str(e)[:80]}")

    return {
        "cleaned_text": response_text,
        "enriched_notes": "",
        "keep_for_corpus": True,
        "drop_reason": "",
        "removed_noise_types": [],
        "ocr_repairs": [],
        "technical_content_types": [],
        "uncertainty_notes": "JSON parse failed, raw response used",
        "cleaning_actions": [],
        "confidence": 0.3,
        "parse_success": False,
    }


def load_processed_ids(language: str) -> set:
    """Load already-processed chunk IDs from checkpoint."""
    if language == "zh":
        cp_path = REPORT_DIR / "phase_2_7_reclean_fast_checkpoint.json"
    else:
        cp_path = REPORT_DIR / "phase_2_7_en_reclean_checkpoint.json"

    if not cp_path.exists():
        return set()

    with open(cp_path) as f:
        data = json.load(f)
    return set(data.get("processed_chunk_ids", []))


def load_failed_chunks(language: str, processed_ids: set) -> list:
    """Extract retryable chunk data from error JSONL."""
    if language == "zh":
        error_path = REPORT_DIR / "phase_2_7_reclean_fast_errors.jsonl"
    else:
        error_path = REPORT_DIR / "phase_2_7_en_reclean_errors.jsonl"

    if not error_path.exists():
        return []

    chunks = []
    seen_ids = set()

    with open(error_path) as f:
        for line in f:
            record = json.loads(line)
            ctx = record.get("context", {})
            if not ctx or not ctx.get("chunk_text"):
                continue

            src = ctx.get("source_file", "")
            page = ctx.get("page_number", 0)
            hash_val = ctx.get("content_hash", "")
            chunk_id = f"{src}_p{page}_{hash_val}"

            if chunk_id in seen_ids or chunk_id in processed_ids:
                continue
            seen_ids.add(chunk_id)

            chunks.append({
                "chunk_id": chunk_id,
                "source_file": src,
                "page_number": page,
                "chunk_type": ctx.get("chunk_type", "body"),
                "chunk_text": ctx["chunk_text"],
                "source_folder": ctx.get("source_folder", ""),
                "content_hash": hash_val,
                "language": ctx.get("language", language),
            })

    return chunks


def retry_chunk(chunk_data: dict, pool, language: str, output_lock, output_file: Path,
                progress: dict, progress_lock) -> bool:
    """Retry cleaning a single failed chunk."""
    source_file = chunk_data["source_file"]
    page_number = chunk_data["page_number"]
    raw_text = chunk_data["chunk_text"]
    chunk_type = chunk_data.get("chunk_type", "body")
    doc_language = chunk_data.get("language", language)

    prompt = get_cleaning_prompt(doc_language, raw_text, chunk_type=chunk_type)
    try:
        response = pool.chat(
            messages=[
                {"role": "system", "content": "You are a professional technical document cleaning specialist. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=4096,
        )
        parsed = parse_v2_cleaning_response(response.content, raw_text)
    except Exception as e:
        logger.warning(f"Retry failed: {source_file} p{page_number}: {str(e)[:80]}")
        with progress_lock:
            progress["failed"] += 1
        return False

    # Build output record
    cleaned_text = parsed["cleaned_text"]
    if not cleaned_text.strip():
        cleaned_text = raw_text

    result = {
        "chunk_id": f"chunk_{chunk_data['content_hash'][:8]}_retry",
        "source_file": source_file,
        "source_folder": chunk_data.get("source_folder", ""),
        "page_numbers": [page_number],
        "language": doc_language,
        "original_text": raw_text,
        "cleaned_text": cleaned_text,
        "cleaning_model": response.model,
        "run_id": f"phase_2_7_retry_{language}",
        "chunk_type": chunk_type,
        "keep_for_corpus": parsed["keep_for_corpus"],
        "removed_noise_types": parsed.get("removed_noise_types", []),
        "ocr_repairs": parsed.get("ocr_repairs", []),
        "technical_content_types": parsed.get("technical_content_types", []),
        "uncertainty_notes": parsed.get("uncertainty_notes", ""),
        "drop_reason": parsed.get("drop_reason", ""),
        "enriched_notes": parsed.get("enriched_notes", ""),
        "confidence": parsed.get("confidence", 0.5),
        "retry": True,
    }

    # Write result
    with output_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with progress_lock:
        progress["completed"] += 1
        progress["tokens"] += response.total_tokens
        if parsed["keep_for_corpus"]:
            progress["corpus"] += 1
        else:
            progress["dropped"] += 1

    logger.info(f"Retry OK: {source_file} p{page_number} corpus={parsed['keep_for_corpus']}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Retry failed chunks from phase_2_7")
    parser.add_argument("--language", choices=["zh", "en", "both"], default="both")
    parser.add_argument("--num_workers", type=int, default=20)
    args = parser.parse_args()

    languages = ["zh", "en"] if args.language == "both" else [args.language]
    pool = get_model_pool()

    for lang in languages:
        logger.info(f"=== Retrying failed chunks for {lang} ===")

        processed_ids = load_processed_ids(lang)
        logger.info(f"Already processed: {len(processed_ids)}")

        failed_chunks = load_failed_chunks(lang, processed_ids)
        logger.info(f"Retryable failed chunks: {len(failed_chunks)}")

        if not failed_chunks:
            logger.info(f"No failed chunks for {lang}, skipping")
            continue

        suffix = "_reclean" if lang == "zh" else "_en_reclean"
        output_file = OUTPUT_DIR / f"cleaned_chunks{suffix}_retry.jsonl"

        progress = {"completed": 0, "corpus": 0, "dropped": 0, "failed": 0, "tokens": 0}
        output_lock = threading.Lock()
        progress_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for chunk_data in failed_chunks:
                future = executor.submit(
                    retry_chunk, chunk_data, pool, lang,
                    output_lock, output_file, progress, progress_lock,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Future error: {e}")

        logger.info(f"=== {lang} retry done: {progress['completed']} OK, "
                     f"{progress['failed']} fail, {progress['corpus']} corpus, "
                     f"{progress['dropped']} dropped ===")


if __name__ == "__main__":
    main()