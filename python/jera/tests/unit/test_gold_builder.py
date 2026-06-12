"""Unit tests for gold_builder operand-provenance guard and normalization.

All tests run fully offline — no API calls, no paid extras required.
Tests cover:
- ``_normalize_text``: thousands commas, full/half-width digits
- ``operand_in_chunk``: accept/reject cases incl. Korean units, commas, percent
- ``ClaudeGoldGenerator`` disabled-by-default guard
"""

from __future__ import annotations

import pytest

from jera.evaluation.gold_builder import (
    ClaudeGoldGenerator,
    _normalize_text,
    operand_in_chunk,
)

# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_strips_thousands_commas(self) -> None:
        assert _normalize_text("1,234,567") == "1234567"

    def test_fullwidth_digits(self) -> None:
        # Full-width digits (U+FF10–U+FF19) should become ASCII
        assert _normalize_text("１２３４") == "1234"

    def test_nfkc_normalization(self) -> None:
        # Half-width katakana → full-width (NFKC); here we just check no crash
        result = _normalize_text("3.5％")
        assert "3.5" in result

    def test_collapses_whitespace(self) -> None:
        assert _normalize_text("a  b\t c") == "a b c"

    def test_mixed(self) -> None:
        assert _normalize_text("총 1,980조　원") == "총 1980조 원"


# ---------------------------------------------------------------------------
# operand_in_chunk — accept cases
# ---------------------------------------------------------------------------


class TestOperandInChunkAccept:
    def test_plain_integer(self) -> None:
        assert operand_in_chunk(1234.0, "GDP는 1234조원이다")

    def test_comma_thousands_in_text(self) -> None:
        # text has comma-formatted number; normalization should strip it
        assert operand_in_chunk(1234.0, "총 1,234 억원 규모")

    def test_float_value(self) -> None:
        assert operand_in_chunk(3.5, "물가상승률은 3.5%로 전망")

    def test_korean_unit_eok(self) -> None:
        # 2억 = 200,000,000 = 2e8
        assert operand_in_chunk(2e8, "투자규모는 2억원")

    def test_korean_unit_jo(self) -> None:
        # 1조 = 1e12
        assert operand_in_chunk(1e12, "시가총액 1조원 돌파")

    def test_comma_in_chunk_text(self) -> None:
        # chunk text uses comma-thousands; cited_number is plain float
        assert operand_in_chunk(1980.3, "코스피 시가총액 1,980.3 조원")

    def test_zero_value(self) -> None:
        assert operand_in_chunk(0.0, "성장률 0% 기록")

    def test_large_number_with_comma(self) -> None:
        assert operand_in_chunk(12345678.0, "총액 12,345,678원")


# ---------------------------------------------------------------------------
# operand_in_chunk — reject cases (hallucinated operands)
# ---------------------------------------------------------------------------


class TestOperandInChunkReject:
    def test_number_not_in_text(self) -> None:
        assert not operand_in_chunk(9999.0, "GDP는 1234조원이다")

    def test_similar_but_different(self) -> None:
        # 1235 vs 1234 — should NOT match
        assert not operand_in_chunk(1235.0, "GDP는 1234조원이다")

    def test_empty_text(self) -> None:
        assert not operand_in_chunk(42.0, "")

    def test_float_not_matching_integer_in_text(self) -> None:
        # text has 3.0 but cited is 3.5
        assert not operand_in_chunk(3.5, "성장률은 3.0%")

    def test_partial_digit_not_enough(self) -> None:
        # text has "12" but cited is 123
        assert not operand_in_chunk(123.0, "총 12개 항목")


# ---------------------------------------------------------------------------
# ClaudeGoldGenerator disabled-by-default
# ---------------------------------------------------------------------------


class TestClaudeGoldGeneratorDisabled:
    def test_raises_when_not_enabled(self) -> None:
        with pytest.raises(RuntimeError, match="disabled by default"):
            ClaudeGoldGenerator()

    def test_raises_without_api_key(self) -> None:
        with pytest.raises(RuntimeError, match="api_key"):
            ClaudeGoldGenerator(enabled=True, api_key=None)

    def test_raises_without_api_key_empty_string(self) -> None:
        with pytest.raises(RuntimeError, match="api_key"):
            ClaudeGoldGenerator(enabled=True, api_key="")
