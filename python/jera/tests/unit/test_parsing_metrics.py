"""Parsing metrics (hand-constructed pairs) + parser_bench plumbing check + base-import guard."""

from __future__ import annotations

import json
import math
import pathlib

from jera.adapters.parsing.hwpx_parser import HwpxParser
from jera.domain.document import MediaType, SourceRef
from jera.evaluation.parser_bench import ParserBenchReport, grade
from jera.evaluation_contracts.parsing_metrics import (
    cer,
    element_type_accuracy,
    reading_order_score,
    table_f1,
)

_FIX = pathlib.Path(__file__).parents[1] / "fixtures" / "hwpx"


def test_cer() -> None:
    assert cer("hello", "hello") == 0.0
    assert cer("kitten", "sitting") == 3 / 6  # 3 edits / 6 ref chars
    assert cer("", "") == 0.0
    assert cer("abc", "") == 1.0


def test_table_f1_hand_constructed() -> None:
    gold = [(0, 0, "a"), (0, 1, "b"), (1, 0, "c")]
    assert table_f1(gold, gold) == 1.0
    assert table_f1([], []) == 1.0
    assert table_f1([(0, 0, "x")], gold) == 0.0  # no overlap
    # 1 of 3 overlap: precision 1/1, recall 1/3 → F1 = 0.5
    assert math.isclose(table_f1([(0, 0, "a")], gold), 0.5)


def test_element_type_accuracy() -> None:
    gold = {"0": "Title", "1": "NarrativeText", "2": "Table"}
    assert element_type_accuracy(gold, gold) == 1.0
    assert math.isclose(element_type_accuracy({"0": "Title"}, gold), 1 / 3)  # missing ids = miss
    assert element_type_accuracy({}, {}) == 1.0


def test_reading_order_score() -> None:
    assert reading_order_score(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert reading_order_score(["c", "b", "a"], ["a", "b", "c"]) == 0.0  # fully reversed
    assert reading_order_score(["a"], ["a", "b", "c"]) == 1.0  # ≤1 common → 1.0, no throw
    assert reading_order_score([], []) == 1.0


def test_parser_bench_grades_hwpx_against_independent_gold() -> None:
    # Plumbing/determinism check: HwpxParser output vs the SEPARATE gold.json (perfect on fixture).
    gold = json.loads((_FIX / "sample.gold.json").read_text(encoding="utf-8"))
    doc = HwpxParser().parse(
        SourceRef(
            source_id="f", media_type=MediaType.HWPX, content=(_FIX / "sample.hwpx").read_bytes()
        )
    )
    result = grade("hwpx", doc, gold)
    assert result.element_type_accuracy == 1.0
    assert result.reading_order_score == 1.0
    assert result.table_f1 == 1.0
    report = ParserBenchReport(results=[result])
    assert "table_f1" in report.to_markdown()


def test_base_only_imports() -> None:
    # AC9 guard: these must import with ZERO extras (stdlib + base deps only).
    import importlib

    for mod in (
        "jera.adapters.parsing.hwpx_parser",
        "jera.adapters.parsing.routing",
        "jera.adapters.parsing.routing_pdf_parser",
        "jera.evaluation_contracts.parsing_metrics",
        "jera.evaluation.parser_bench",
    ):
        assert importlib.import_module(mod) is not None
