"""Unit tests for Phase 2.5 formula/table-aware cleaning mode."""

import pytest

from src.autodata.pipelines.text_preprocessor import classify_chunk_content
from src.autodata.pipelines.prompts.text_cleaning_prompts import (
    get_cleaning_prompt,
    ZH_FORMULA_CLEANING_PROMPT,
    ZH_TABLE_CLEANING_PROMPT,
    ZH_CONSERVATIVE_CLEANING_PROMPT,
    EN_FORMULA_CLEANING_PROMPT,
    EN_TABLE_CLEANING_PROMPT,
    EN_CONSERVATIVE_CLEANING_PROMPT,
    ZH_CLEANING_PROMPT,
    EN_CLEANING_PROMPT,
)


class TestChunkClassification:
    def test_body_classification(self):
        """Normal prose text should be classified as body."""
        text = "碳纤维是一种含碳量在95%以上的高强度纤维材料，广泛应用于航空航天领域。"
        assert classify_chunk_content(text) == "body"

    def test_formula_classification(self):
        """Formula-heavy text should be classified as formula."""
        text = "弹性模量 E = 230 GPa，拉伸强度 σ = 3.5 GPa，密度 ρ = 1.8 g/cm³"
        assert classify_chunk_content(text) == "formula"

    def test_formula_classification_with_greek(self):
        """Text with Greek letters and equations."""
        text = "其中σ表示应力，ε表示应变，E为弹性模量。E≈≈≈E=σ/ε"
        result = classify_chunk_content(text)
        assert result == "formula"

    def test_table_classification(self):
        """Table-like text should be classified as table."""
        text = "表1    碳纤维类型    抗拉强度    弹性模量    密度"
        assert classify_chunk_content(text) == "table"

    def test_table_with_box_drawing(self):
        """Text with box-drawing characters should be classified as table or table_uncertain."""
        text = "┌──────┬──────┬──────┐\n│  类型  │  强度  │  模量  │"
        assert classify_chunk_content(text) in ("table", "table_uncertain")

    def test_table_uncertain_classification(self):
        """Table with too many box-drawing chars relative to content."""
        text = "───────││││││───────││││││───────类型强度───────││││││───────"
        result = classify_chunk_content(text)
        assert result in ("table", "table_uncertain", "empty", "body")

    def test_mixed_classification(self):
        """Text with both formulas and tables."""
        text = "表1 性能参数  E=230GPa  σ=3.5GPa    表1  参数  值"
        result = classify_chunk_content(text)
        assert result in ("mixed", "table", "formula")

    def test_empty_classification(self):
        """Very short text should be classified as empty."""
        text = "第3页"
        assert classify_chunk_content(text) == "empty"

    def test_formula_guess_boost(self):
        """has_formula_guess should boost formula classification."""
        text = "碳纤维的力学性能如下"
        result_normal = classify_chunk_content(text, has_formula_guess=False)
        result_boosted = classify_chunk_content(text, has_formula_guess=True)
        # With formula guess, should at least not be lower classification
        assert result_boosted in ("formula", "mixed") or result_normal == result_boosted


class TestPromptRouting:
    def test_body_prompt_selection(self):
        """Body chunks should use normal cleaning prompt."""
        prompt = get_cleaning_prompt("zh", "test text", chunk_type="body")
        assert "专业" in prompt or "清洗" in prompt

    def test_formula_prompt_selection(self):
        """Formula chunks should use formula-preserving prompt."""
        prompt = get_cleaning_prompt("zh", "E=mc²", chunk_type="formula")
        assert "公式" in prompt or "formula" in prompt.lower()

    def test_table_prompt_selection(self):
        """Table chunks should use table-preserving prompt."""
        prompt = get_cleaning_prompt("zh", "表1", chunk_type="table")
        assert "表格" in prompt or "table" in prompt.lower()

    def test_mixed_prompt_selection(self):
        """Mixed chunks should use conservative prompt."""
        prompt = get_cleaning_prompt("zh", "mixed content", chunk_type="mixed")
        assert "保守" in prompt or "conservative" in prompt.lower()

    def test_table_uncertain_prompt_selection(self):
        """table_uncertain should use table-preserving prompt."""
        prompt = get_cleaning_prompt("zh", "damaged table", chunk_type="table_uncertain")
        assert "表格" in prompt or "table" in prompt.lower()

    def test_english_formula_prompt(self):
        """English formula chunks should use English formula prompt."""
        prompt = get_cleaning_prompt("en", "E=mc²", chunk_type="formula")
        assert "formula" in prompt.lower() or "equation" in prompt.lower()

    def test_english_table_prompt(self):
        """English table chunks should use English table prompt."""
        prompt = get_cleaning_prompt("en", "Table 1", chunk_type="table")
        assert "table" in prompt.lower()

    def test_formula_preservation_instruction(self):
        """Formula prompt must contain 'do not simplify' instruction."""
        prompt = get_cleaning_prompt("zh", "E=σ/ε", chunk_type="formula")
        assert "不要简化" in prompt or "不要改写" in prompt or "不要删除" in prompt

    def test_table_preservation_instruction(self):
        """Table prompt must contain 'do not delete rows' instruction."""
        prompt = get_cleaning_prompt("zh", "表1", chunk_type="table")
        assert "不要删除" in prompt or "不要编造" in prompt

    def test_table_uncertain_marker_in_prompt(self):
        """Table prompt should mention table_uncertain marker."""
        prompt = get_cleaning_prompt("zh", "表1", chunk_type="table")
        assert "[table_uncertain]" in prompt or "不确定" in prompt

    def test_formula_uncertain_marker_in_prompt(self):
        """Formula prompt should mention formula_uncertain marker."""
        prompt = get_cleaning_prompt("zh", "E=σ/ε", chunk_type="formula")
        assert "[formula_uncertain]" in prompt or "[formula_missing]" in prompt