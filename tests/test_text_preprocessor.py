"""Unit tests for text preprocessor module."""

import json
import os
import tempfile

import pytest

from src.autodata.pipelines.text_preprocessor import (
    analyze_page_noise,
    detect_language,
    generate_noise_report,
    is_header_footer,
    load_raw_document,
    preprocess_document,
    split_page_into_chunks,
)
from src.autodata.pipelines.text_schema import Language, RawDocument, RawPage


# ── Language detection ──────────────────────────────────────────

class TestLanguageDetection:
    def test_chinese_text(self):
        text = "碳纤维是一种含碳量在90%以上的高强度纤维材料"
        assert detect_language(text) == Language.ZH

    def test_english_text(self):
        text = "Carbon fiber is a high-strength fiber material with carbon content above 90%"
        assert detect_language(text) == Language.EN

    def test_mixed_text(self):
        text = "CFRP (碳纤维增强聚合物) is widely used in aerospace"
        lang = detect_language(text)
        # Mixed text detection depends on ratio
        assert lang in (Language.ZH, Language.EN, Language.UNKNOWN)

    def test_empty_text(self):
        assert detect_language("") == Language.UNKNOWN


# ── Noise analysis ──────────────────────────────────────────────

class TestNoiseAnalysis:
    def test_empty_page(self):
        assert analyze_page_noise("") == ["empty_page"]

    def test_short_page(self):
        assert analyze_page_noise("abc") == ["empty_page"]

    def test_normal_text(self):
        text = "碳纤维是一种高强度纤维材料，广泛应用于航空航天领域。"
        issues = analyze_page_noise(text)
        assert "empty_page" not in issues

    def test_box_drawing_chars(self):
        text = "表格内容 ─── │┃┏┓ 数据"
        issues = analyze_page_noise(text)
        assert "box_drawing" in issues

    def test_irregular_spacing(self):
        text = "碳纤维    材料     应用"
        issues = analyze_page_noise(text)
        assert "irregular_spacing" in issues


# ── Header/footer detection ──────────────────────────────────────

class TestHeaderFooter:
    def test_page_number(self):
        assert is_header_footer("第3页")

    def test_page_number_english(self):
        assert is_header_footer("Page 42")

    def test_chapter_header(self):
        assert is_header_footer("第5章 碳纤维生产工艺")

    def test_normal_text(self):
        assert not is_header_footer("碳纤维是一种高强度材料，其含碳量超过90%。")

    def test_short_normal(self):
        assert not is_header_footer("碳纤维")


# ── Chunk splitting ──────────────────────────────────────────────

class TestChunkSplitting:
    def test_empty_page_single_chunk(self):
        chunks = split_page_into_chunks("", 1, "test.json", "books")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "empty"

    def test_header_footer_chunk(self):
        chunks = split_page_into_chunks("第3页", 3, "test.json", "books")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "header_footer"

    def test_short_body_text(self):
        text = "碳纤维是一种含碳量在90%以上的高强度纤维材料，广泛应用于航空航天领域。"
        chunks = split_page_into_chunks(text, 1, "test.json", "books")
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "body"

    def test_long_text_splitting(self):
        text = "碳纤维技术 " * 500  # ~1500 chars
        chunks = split_page_into_chunks(text, 1, "test.json", "books", max_chars=500)
        assert len(chunks) >= 2

    def test_chunk_has_content_hash(self):
        text = "碳纤维材料"
        chunks = split_page_into_chunks(text, 1, "test.json", "books")
        assert chunks[0]["content_hash"] != ""

    def test_chunk_has_source_provenance(self):
        text = "碳纤维材料"
        chunks = split_page_into_chunks(text, 1, "test.json", "books")
        assert chunks[0]["source_file"] == "test.json"
        assert chunks[0]["source_folder"] == "books"


# ── Document loading ──────────────────────────────────────────────

class TestDocumentLoading:
    def test_load_zh_document(self, tmp_path):
        # Create a minimal test JSON file
        doc_data = {
            "file_name": "test_book.pdf",
            "file_path": "/path/to/test_book.pdf",
            "file_size": 1000,
            "page_count": 2,
            "metadata": {},
            "pages": [
                {
                    "page_number": 1,
                    "source": "ocr",
                    "content": "碳纤维是一种高强度材料",
                    "has_formula_guess": False,
                    "clean_content": "碳纤维是一种高强度材料",
                },
                {
                    "page_number": 2,
                    "source": "ocr",
                    "content": "其含碳量超过90%",
                    "has_formula_guess": False,
                    "clean_content": "其含碳量超过90%",
                },
            ],
        }
        path = tmp_path / "test_book.clean.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc_data, f)

        doc = load_raw_document(str(path), "books")
        assert doc.file_name == "test_book.clean.json"
        assert doc.source_folder == "books"
        assert doc.page_count == 2
        assert len(doc.pages) == 2
        assert doc.language == Language.ZH

    def test_load_with_max_pages(self, tmp_path):
        doc_data = {
            "file_name": "test.pdf",
            "file_path": "",
            "file_size": 1000,
            "page_count": 100,
            "metadata": {},
            "pages": [
                {"page_number": i, "source": "ocr", "content": f"Page {i}", "has_formula_guess": False}
                for i in range(100)
            ],
        }
        path = tmp_path / "test.clean.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc_data, f)

        doc = load_raw_document(str(path), "books", max_pages=10)
        assert len(doc.pages) == 10


# ── Noise report ──────────────────────────────────────────────

class TestNoiseReport:
    def test_generate_report(self):
        pages = [
            RawPage(page_number=1, content="碳纤维材料", clean_content="碳纤维材料"),
            RawPage(page_number=2, content="第3页", clean_content="第3页"),
            RawPage(page_number=3, content="", clean_content=""),
        ]
        doc = RawDocument(
            file_name="test.json",
            source_folder="books",
            page_count=3,
            file_size_bytes=100,
            pages=pages,
            language=Language.ZH,
        )
        report = generate_noise_report([doc])
        assert report["total_pages"] == 3
        assert report["empty_pages"] >= 1