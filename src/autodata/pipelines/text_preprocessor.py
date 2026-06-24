"""Text preprocessor — OCR noise analysis and page segmentation.

Loads .clean.json files, detects language, identifies noise patterns,
and splits pages into model-safe chunks while preserving provenance.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from src.autodata.pipelines.text_schema import (
    Language,
    RawDocument,
    RawPage,
    content_hash,
)


# ── Language detection ──────────────────────────────────────────

def detect_language(text: str) -> Language:
    """Detect whether text is primarily Chinese or English."""
    # Count CJK characters vs ASCII alphabetic characters
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    alpha_chars = len(re.findall(r'[a-zA-Z]', text))
    if cjk_chars > alpha_chars * 0.3:
        return Language.ZH
    elif alpha_chars > cjk_chars * 3:
        return Language.EN
    return Language.UNKNOWN


# ── Noise patterns ────────────────────────────────────────────

NOISE_PATTERNS = {
    "box_drawing": re.compile(r'[─━│┃┏┓┗┛├┤┬┴┼┄┅┆┇┈┉┊┋]'),
    "irregular_spacing": re.compile(r'(\S)\s{3,}(\S)'),
    "char_repetition": re.compile(r'(.)\1{4,}'),
    "page_number_header": re.compile(r'^第\d+页|^\s*Page\s*\d+\s*$'),
    "repeated_header": re.compile(r'^(碳纤维|Carbon Fiber|CFRP|第\d+章|Chapter\s+\d+)\s*$'),
    "garbled_unicode": re.compile(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,;:!?()\[\]{}\"\'/@#$%&*-+=<>~°μΩαβγδεηθλπσφψω℃±×÷≈≠≤≥∝∞∫∑∂√∇∫≡⊥∠∥∏√]'),
    "broken_line_hyphen": re.compile(r'-\s*\n\s*'),
    "empty_page": None,  # checked by length
}


def analyze_page_noise(text: str) -> list[str]:
    """Identify OCR noise patterns in a page."""
    issues = []
    if len(text.strip()) < 15:
        issues.append("empty_page")
        return issues

    for name, pattern in NOISE_PATTERNS.items():
        if pattern and pattern.search(text):
            issues.append(name)
    return issues


def is_header_footer(text: str, min_len: int = 3, max_len: int = 120) -> bool:
    """Check if text looks like a repeated header/footer."""
    stripped = text.strip()
    if len(stripped) < min_len or len(stripped) > max_len:
        return False
    # Common header/footer patterns
    patterns = [
        r'^第\d+页',
        r'^Page\s*\d+',
        r'^\d+$',
        r'^第\d+章\s*.*$',
        r'^Chapter\s+\d+',
        r'^目\s*录$',
        r'^Contents$',
        r'^碳纤维.*丛书$',
        r'^Carbon Fiber.*Series$',
    ]
    for p in patterns:
        if re.match(p, stripped, re.IGNORECASE):
            return True
    return False


# ── Chunk type classification ──────────────────────────────────

def classify_chunk_content(text: str, has_formula_guess: bool = False) -> str:
    """Classify chunk content type for specialized cleaning routing.

    Returns one of:
    - body: normal prose text
    - formula: formula-heavy content (many equations, symbols)
    - table: table-like content (structured rows/columns)
    - mixed: mixed formula/table/prose content
    - formula_table_uncertain: damaged table that cannot be reconstructed
    """
    stripped = text.strip()
    if len(stripped) < 15:
        return "empty"

    # Formula indicators
    formula_patterns = [
        r'[=≈≠≤≥±×÷∝∞∫∑∂√∇≡]',            # math symbols
        r'[αβγδεηθλπσφψωμΩ]',               # Greek letters used in formulas
        r'\b\d+\s*[°%℃]\b',                  # numerical values with units
        r'[Ee]\s*[=≈]\s*[\d.×]+',            # E = ... expressions
        r'σ|ρ|ε|η|ν|ξ|λ|μ|κ|τ',              # common Greek variable names
        r'\\frac|\\sqrt|\\int|\\sum|\\prod',  # LaTeX fragments
        r'[＋－×÷＝≠≈≤≥]',                    # CJK math symbols
    ]
    formula_hits = sum(1 for p in formula_patterns if re.search(p, stripped))
    formula_density = formula_hits / max(len(stripped) / 50, 1)  # hits per 50 chars

    # Table indicators
    table_patterns = [
        r'(\S+\s{2,}\S+\s{2,}\S+\s{2,}\S+)',  # columns separated by multiple spaces
        r'[│┃┤├┬┴┼─━┏┓┗┛]',                     # box-drawing characters
        r'^\s*\S+\t\S+',                          # tab-separated columns
        r'表\s*\d+',                              # "表X" (Table X in Chinese)
        r'Table\s+\d+',                           # "Table X" in English
        r'[┌┬┐├┼┤└┴┘]',                          # more box-drawing
    ]
    table_hits = sum(1 for p in table_patterns if re.search(p, stripped))
    table_density = table_hits / max(len(stripped) / 50, 1)

    # Use formula_guess from OCR metadata
    if has_formula_guess:
        formula_density += 2

    # Classification thresholds
    if formula_density >= 2 and table_density >= 2:
        return "mixed"
    elif formula_density >= 2:
        return "formula"
    elif table_density >= 2:
        # Check if table is too damaged (more box-drawing than content)
        box_chars = len(re.findall(r'[─━│┃┏┓┗┛├┤┬┴┼┄┅┆┇┈┉┊┋┌┬┐├┼┤└┴┘]', stripped))
        content_chars = len(re.sub(r'[─━│┃┏┓┗┛├┤┬┴┼┄┅┆┇┈┉┊┋┌┬┐├┼┤└┴┘\s]', '', stripped))
        if box_chars > content_chars * 0.5:
            return "table_uncertain"
        return "table"
    else:
        return "body"


# ── Chunk splitting ────────────────────────────────────────────

MAX_CHUNK_CHARS = 3000  # conservative limit for LLM input
OVERLAP_CHARS = 200     # overlap between chunks to avoid splitting mid-sentence


def split_page_into_chunks(
    text: str,
    page_number: int,
    source_file: str,
    source_folder: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    """Split a page's text into model-safe chunks with provenance.

    Returns a list of chunk dicts with:
      - chunk_text, page_number, source_file, source_folder, content_hash
      - chunk_type (body, header_footer, formula, table, empty)
    """
    stripped = text.strip()

    # Header/footer pages (check before empty, since headers like "第3页" are short)
    if is_header_footer(stripped):
        return [{
            "chunk_text": stripped,
            "page_number": page_number,
            "source_file": source_file,
            "source_folder": source_folder,
            "content_hash": content_hash(stripped),
            "chunk_type": "header_footer",
            "noise_issues": analyze_page_noise(text),
        }]

    # Empty pages (very short content that isn't a header/footer)
    if len(stripped) < 15:
        return [{
            "chunk_text": stripped,
            "page_number": page_number,
            "source_file": source_file,
            "source_folder": source_folder,
            "content_hash": content_hash(stripped),
            "chunk_type": "empty",
            "noise_issues": analyze_page_noise(text),
        }]

    # Short pages — single chunk
    if len(stripped) <= max_chars:
        noise = analyze_page_noise(text)
        chunk_type = classify_chunk_content(stripped, has_formula_guess=False)
        return [{
            "chunk_text": stripped,
            "page_number": page_number,
            "source_file": source_file,
            "source_folder": source_folder,
            "content_hash": content_hash(stripped),
            "chunk_type": chunk_type,
            "noise_issues": noise,
        }]

    # Long pages — split into overlapping chunks
    chunks = []
    start = 0
    while start < len(stripped):
        end = start + max_chars
        chunk_text = stripped[start:end]

        # If not at the end, try to find a sentence boundary
        if end < len(stripped):
            # Look for sentence-ending punctuation near the end
            boundary_chars = ['. ', '.\n', '。', '！', '？', '；', ';', '\n\n']
            best_boundary = -1
            for bc in boundary_chars:
                idx = chunk_text.rfind(bc)
                if idx > max_chars * 0.7:  # at least 70% of max_chars
                    best_boundary = max(best_boundary, idx + len(bc))
            if best_boundary > 0:
                chunk_text = stripped[start:start + best_boundary]
                end = start + best_boundary

        chunks.append({
            "chunk_text": chunk_text.strip(),
            "page_number": page_number,
            "source_file": source_file,
            "source_folder": source_folder,
            "content_hash": content_hash(chunk_text.strip()),
            "chunk_type": "body",
            "noise_issues": analyze_page_noise(chunk_text),
            "chunk_offset_start": start,
            "chunk_offset_end": end,
        })

        # Move to next chunk with overlap
        start = end - overlap if end < len(stripped) else end

    return chunks


# ── Document loader ────────────────────────────────────────────

def load_raw_document(
    file_path: str,
    source_folder: str,
    max_pages: Optional[int] = None,
) -> RawDocument:
    """Load a .clean.json file into a RawDocument with provenance."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = os.path.basename(file_path)
    pages_data = data.get("pages", [])
    if max_pages:
        pages_data = pages_data[:max_pages]

    # Detect document language from first non-empty page
    sample_text = ""
    for p in pages_data[:5]:
        sample_text += p.get("content", "") or p.get("clean_content", "") or ""
    language = detect_language(sample_text)

    # Build RawPage objects
    raw_pages = []
    for p in pages_data:
        content = p.get("content", "")
        clean_content = p.get("clean_content")
        raw_page = RawPage(
            page_number=p.get("page_number", 0),
            content=content,
            clean_content=clean_content if clean_content else None,
            has_formula_guess=p.get("has_formula_guess", False),
            source=p.get("source", "ocr"),
            content_hash=content_hash(content) if content else "",
            source_file=file_name,
            source_folder=source_folder,
        )
        raw_pages.append(raw_page)

    return RawDocument(
        file_name=file_name,
        source_folder=source_folder,
        page_count=data.get("page_count", len(pages_data)),
        file_size_bytes=data.get("file_size", 0),
        metadata=data.get("metadata", {}),
        pages=raw_pages,
        language=language,
    )


def preprocess_document(
    doc: RawDocument,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[dict[str, Any]]:
    """Preprocess a RawDocument into chunks for cleaning.

    For each page:
    1. Choose clean_content if available, otherwise content
    2. Identify noise patterns
    3. Split into model-safe chunks
    4. Preserve provenance
    """
    all_chunks = []

    for page in doc.pages:
        # Choose best available text
        text = page.clean_content if page.clean_content else page.content
        if not text:
            text = page.content

        page_chunks = split_page_into_chunks(
            text=text,
            page_number=page.page_number,
            source_file=page.source_file,
            source_folder=page.source_folder,
            max_chars=max_chunk_chars,
        )

        # Add formula hint to chunk metadata
        for chunk in page_chunks:
            chunk["has_formula_guess"] = page.has_formula_guess
            chunk["language"] = doc.language.value

        all_chunks.extend(page_chunks)

    return all_chunks


# ── OCR noise analysis report ──────────────────────────────────

def generate_noise_report(docs: list[RawDocument]) -> dict[str, Any]:
    """Generate a noise analysis report across all documents."""
    total_pages = 0
    empty_pages = 0
    header_footer_pages = 0
    formula_pages = 0
    noise_counts: dict[str, int] = {}
    total_chars_raw = 0
    total_chars_clean = 0

    for doc in docs:
        for page in doc.pages:
            total_pages += 1
            text = page.content or ""
            total_chars_raw += len(text)
            if page.clean_content:
                total_chars_clean += len(page.clean_content)

            if len(text.strip()) < 15:
                empty_pages += 1
                continue

            if is_header_footer(text):
                header_footer_pages += 1

            if page.has_formula_guess:
                formula_pages += 1

            issues = analyze_page_noise(text)
            for issue in issues:
                noise_counts[issue] = noise_counts.get(issue, 0) + 1

    return {
        "total_pages": total_pages,
        "empty_pages": empty_pages,
        "header_footer_pages": header_footer_pages,
        "formula_pages": formula_pages,
        "noise_counts": noise_counts,
        "total_chars_raw": total_chars_raw,
        "total_chars_clean": total_chars_clean,
        "avg_reduction_pct": round(
            (1 - total_chars_clean / max(total_chars_raw, 1)) * 100, 1
        ),
    }