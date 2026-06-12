"""Parser benchmark — grades a parser's output against INDEPENDENT gold labels.

In CI this is a deterministic plumbing/correctness check on the committed HWPX fixture + its
separate `sample.gold.json` (gold is graded *against*, never derived from, parser output). The
M5b `scripts/parser_bench.py` reuses this over real engines/corpora (the real "benchmark").
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from jera.domain.document import ParsedDocument
from jera.evaluation_contracts.parsing_metrics import (
    TableCell,
    element_type_accuracy,
    reading_order_score,
    table_f1,
)


class ParserBenchResult(BaseModel):
    parser: str
    element_type_accuracy: float
    reading_order_score: float
    table_f1: float


class ParserBenchReport(BaseModel):
    results: list[ParserBenchResult]

    def to_markdown(self) -> str:
        rows = ["parser           type_acc  read_order  table_f1"]
        for r in self.results:
            rows.append(
                f"{r.parser:<15}  {r.element_type_accuracy:>7.3f}  "
                f"{r.reading_order_score:>9.3f}  {r.table_f1:>7.3f}"
            )
        return "\n".join(rows)


def grade(parser_name: str, parsed: ParsedDocument, gold: dict[str, Any]) -> ParserBenchResult:
    """Grade one parser's ParsedDocument against a gold dict (keys are element order indices)."""
    predicted_types = {str(e.order): e.type.value for e in parsed.elements}
    predicted_order = [str(e.order) for e in parsed.elements]
    predicted_cells = _cells_from_tables(parsed)

    gold_cells: list[TableCell] = [
        (int(r), int(c), str(t)) for r, c, t in gold.get("table_cells", [])
    ]
    return ParserBenchResult(
        parser=parser_name,
        element_type_accuracy=element_type_accuracy(predicted_types, gold.get("element_types", {})),
        reading_order_score=reading_order_score(predicted_order, gold.get("reading_order", [])),
        table_f1=table_f1(predicted_cells, gold_cells),
    )


def _cells_from_tables(parsed: ParsedDocument) -> list[TableCell]:
    """Recover (row, col, text) triples from TABLE elements rendered as 'a | b\\nc | d'."""
    cells: list[TableCell] = []
    for el in parsed.elements:
        if el.type.value != "Table":
            continue
        for r, line in enumerate(el.text.splitlines()):
            for c, cell in enumerate(line.split("|")):
                text = cell.strip()
                if text:
                    cells.append((r, c, text))
    return cells
