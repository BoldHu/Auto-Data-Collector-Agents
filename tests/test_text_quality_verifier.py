"""Unit tests for quality verification agent."""

import pytest

from src.autodata.pipelines.text_schema import (
    CleanedChunk,
    Language,
    QualityScore,
    QualityVerdict,
)
from src.autodata.agents.quality_verification_agent import (
    QualityVerificationAgent,
)


class TestQualityScore:
    def test_score_creation(self):
        score = QualityScore(
            clarity=0.8,
            completeness=0.7,
            consistency=0.9,
            feasibility=0.8,
            complexity=0.6,
            domain_relevance=0.9,
            verdict=QualityVerdict.PASSED,
        )
        assert score.clarity == 0.8
        assert score.verdict == QualityVerdict.PASSED

    def test_average_score(self):
        score = QualityScore(
            clarity=0.8,
            completeness=0.7,
            consistency=0.9,
            feasibility=0.8,
            domain_relevance=0.9,
        )
        avg = score.average
        assert 0.7 < avg < 0.9

    def test_score_serialization(self):
        score = QualityScore(
            clarity=0.8,
            completeness=0.7,
            verdict=QualityVerdict.PASSED,
            issues=["minor_typo"],
        )
        d = score.to_dict()
        assert "clarity" in d
        assert "average" in d
        assert "verdict" in d
        assert "issues" in d

    def test_verdict_values(self):
        assert QualityVerdict.PASSED.value == "passed"
        assert QualityVerdict.NEEDS_REVISION.value == "needs_revision"
        assert QualityVerdict.FAILED.value == "failed"


class TestQualityVerificationAgent:
    def test_agent_creation(self):
        agent = QualityVerificationAgent()
        assert agent.name == "QualityVerificationAgent"
        assert agent.model == "mimo-v2.5-pro"

    def test_parse_verification_response(self):
        agent = QualityVerificationAgent()
        response = """{
            "clarity": 0.85,
            "completeness": 0.9,
            "consistency": 0.8,
            "feasibility": 0.75,
            "complexity": 0.6,
            "domain_relevance": 0.95,
            "verdict": "passed",
            "issues": []
        }"""
        score = agent._parse_verification_response(response)
        assert score.clarity == 0.85
        assert score.verdict == QualityVerdict.PASSED

    def test_parse_invalid_response(self):
        agent = QualityVerificationAgent()
        response = "This is not JSON"
        score = agent._parse_verification_response(response)
        assert score.verdict == QualityVerdict.NEEDS_REVISION
        assert "json_parse_failed" in score.issues