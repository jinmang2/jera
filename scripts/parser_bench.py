"""Runnable parser benchmark — grades parsers against INDEPENDENT gold labels.

    uv run python scripts/parser_bench.py

Grades the CI-real `HwpxParser` against the committed HWPX fixture + its hand-authored
`sample.gold.json` (gold is graded *against*, never derived from, parser output), then prints a
comparison table. Opt-in engines (docling / opendataloader / camelot / pyhwp) are probed and
reported as available-or-skipped so the bench is honest about what actually ran in this
environment — heavy/Java/paid engines stay opt-in (no live run without their extras).
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

from jera.adapters.parsing import HwpxParser
from jera.domain.document import MediaType, SourceRef
from jera.evaluation.parser_bench import ParserBenchReport, grade

_REPO = Path(__file__).resolve().parents[1]
_HWPX_DIR = _REPO / "python/jera/tests/fixtures/hwpx"

# (import name, human label) for the opt-in engines added in M5b — probed, not required.
_OPT_IN = [
    ("docling", "docling"),
    ("opendataloader_pdf", "opendataloader"),
    ("camelot", "camelot-tables"),
    ("hwp5", "pyhwp(.hwp)"),
    ("pytesseract", "tesseract-ocr"),
    ("rapidocr_onnxruntime", "rapidocr"),
]


def _bench_hwpx() -> ParserBenchReport:
    sample = _HWPX_DIR / "sample.hwpx"
    gold_path = _HWPX_DIR / "sample.gold.json"
    if not (sample.exists() and gold_path.exists()):
        return ParserBenchReport(results=[])
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    parsed = HwpxParser().parse(
        SourceRef(source_id="hwpx-bench", media_type=MediaType.HWPX, content=sample.read_bytes())
    )
    return ParserBenchReport(results=[grade("hwpx(stdlib)", parsed, gold)])


def _engine_availability() -> list[str]:
    return [
        f"  {label:<16} {'available' if find_spec(mod) else 'skipped (extra not installed)'}"
        for mod, label in _OPT_IN
    ]


def main() -> None:
    report = _bench_hwpx()
    print("== parser benchmark (graded vs independent gold) ==\n")
    print(report.to_markdown() if report.results else "(no committed fixtures found)")
    print("\n== opt-in engine availability ==")
    print("\n".join(_engine_availability()))


if __name__ == "__main__":
    main()
