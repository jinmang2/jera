"""Build the Korean research-report eval dataset via Claude (opt-in, one-time run).

This script is NOT run by CI.  It requires:
  - ``uv sync --extra cloud``
  - ``ANTHROPIC_API_KEY`` environment variable
  - PDF files present under ``data/corpus/`` (verify with ``scripts/fetch_corpus.py``)

The output is written to ``data/eval/korean_research.json`` and committed to the
repo so CI reads the cached file without any paid API calls.

Usage::

    ANTHROPIC_API_KEY=sk-... uv run python scripts/build_eval_dataset.py \\
        [--manifest data/corpus/manifest.json] \\
        [--output data/eval/korean_research.json] \\
        [--cases-per-doc 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Korean eval dataset via Claude (opt-in, paid)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpus/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/korean_research.json"),
    )
    parser.add_argument(
        "--cases-per-doc",
        type=int,
        default=5,
        help="Number of eval cases to generate per document (default: 5).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not set.  This script makes paid API calls.",
            file=sys.stderr,
        )
        return 1

    # Local imports here so missing 'cloud' extra gives a clear message
    try:
        from jera.evaluation.gold_builder import ClaudeGoldGenerator
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    entries: list[dict[str, str]] = json.loads(args.manifest.read_text(encoding="utf-8"))
    corpus_dir = args.manifest.parent

    generator = ClaudeGoldGenerator(api_key=api_key, enabled=True)

    all_cases = []
    for entry in entries:
        filename = entry.get("filename", "")
        pdf_path = corpus_dir / filename if filename else None
        if not pdf_path or not pdf_path.exists():
            print(f"  [SKIP] PDF not present: {filename}")
            continue

        # Parse the PDF into chunks using jera's pipeline
        try:
            from jera.adapters.parsing.pymupdf_parser import PyMuPDFParser
            from jera.domain.document import MediaType, SourceRef
        except ImportError as exc:
            print(f"ERROR importing jera pipeline: {exc}", file=sys.stderr)
            return 1

        parser = PyMuPDFParser()
        source_ref = SourceRef(
            source_id=filename,
            media_type=MediaType.PDF,
            content=pdf_path.read_bytes(),
        )
        doc = parser.parse(source_ref)
        chunks = [
            {"chunk_id": str(i), "text": elem.text}
            for i, elem in enumerate(doc.elements)
            if elem.text.strip()
        ]
        if not chunks:
            print(f"  [SKIP] No text elements parsed from {filename}")
            continue

        print(f"  Generating {args.cases_per_doc} cases for {entry.get('title', filename)}...")
        cases = generator.generate_cases(
            chunks,
            source_inst=entry.get("inst", ""),
            source_url=entry.get("url", ""),
            license_=entry.get("license", ""),
            n_cases=args.cases_per_doc,
        )
        print(f"    → {len(cases)} cases accepted (after operand guard)")
        all_cases.extend(cases)

    if not all_cases:
        print("WARNING: no cases generated — output file will be empty.", file=sys.stderr)

    dataset = {"name": "korean_research", "cases": [c.model_dump() for c in all_cases]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_cases)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
