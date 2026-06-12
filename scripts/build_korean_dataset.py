"""Build the Korean research eval dataset from committed corpus fixtures (fully offline).

This script ingests the three synthetic Korean markdown reports under
``data/corpus/korean_research/`` through a TEST-profile RAG system (in-memory store,
hash embeddings, BM25 sparse — zero paid APIs), labels gold chunks by substring
containment, and writes the result to ``data/eval/korean_research.json``.

The output is deterministic: chunk IDs are stable_ids derived from content hashes, so
the JSON is reproducible across machines without any external service.

Usage::

    uv run --no-sync python scripts/build_korean_dataset.py [--output PATH]

Options:
    --output PATH   Where to write the JSON (default: data/eval/korean_research.json)

The script is idempotent — running it again overwrites the previous file with the same
content as long as the corpus fixtures are unchanged.  No ANTHROPIC_API_KEY required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running as a plain script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    """Entry point — build and write the dataset."""
    parser = argparse.ArgumentParser(
        description="Rebuild data/eval/korean_research.json from committed corpus fixtures."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/korean_research.json"),
        help="Output path for the eval dataset JSON (default: data/eval/korean_research.json)",
    )
    args = parser.parse_args()

    from jera.config.registry import build_system
    from jera.config.settings import Profile, Settings
    from jera.domain.document import MediaType
    from jera.evaluation.dataset_builder import CaseSpec, build_gold_dataset

    corpus_dir = _REPO_ROOT / "data" / "corpus" / "korean_research"
    if not corpus_dir.exists():
        print(f"ERROR: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Corpus: source_id → (MediaType, markdown text)
    # ------------------------------------------------------------------
    documents: dict[str, tuple[MediaType, str]] = {}
    for md_path in sorted(corpus_dir.glob("*.md")):
        documents[md_path.name] = (MediaType.MARKDOWN, md_path.read_text(encoding="utf-8"))

    if not documents:
        print(f"ERROR: no .md files found under {corpus_dir}", file=sys.stderr)
        return 1

    print(f"Loaded {len(documents)} corpus fixture(s): {list(documents)}")

    # ------------------------------------------------------------------
    # CaseSpecs: (case_id, natural-language query, answer-substring)
    # The answer_contains string is an exact substring that appears in exactly
    # one chunk produced by the heading-aware chunker on the corresponding doc.
    # ------------------------------------------------------------------
    case_specs: list[CaseSpec] = [
        # --- retrieval cases (bok_outlook.md) ---
        # Chunk: "요약" — contains the 1.4% forecast sentence
        CaseSpec(
            "kr-001",
            "한국은행이 2023년 경제성장률을 얼마로 전망했는가?",
            "**1.4%**로 전망하였다",
        ),
        # Chunk: "요약" — contains the 2024 2.3% forecast sentence
        CaseSpec(
            "kr-002",
            "2024년 국내 GDP 성장률 전망치는 얼마인가?",
            "**2.3%**로 개선될 것으로",
        ),
        # --- table cases (kdi_cpi.md) ---
        # Chunk: "개요" — contains 4.2% analysis sentence
        CaseSpec(
            "kr-003",
            "KDI가 발표한 2023년 상반기 소비자물가 상승률 전망치는?",
            "**4.2%**로 분석하였다",
        ),
        # Chunk: "물가 구성항목별 분석" — contains the markdown table with 7.3
        CaseSpec(
            "kr-004",
            "2023년 상반기 가공식품 물가 상승률과 전기·가스·수도 요금 상승률은 각각 얼마인가?",
            "가공식품 | 7.3 |",
        ),
        # --- computation cases (kcmi_market.md) ---
        # Chunk: "시장별 시가총액 현황" — contains both operands and the sum
        CaseSpec(
            "kr-005",
            "2023년 3분기 코스피와 코스닥 시가총액 합계는 얼마인가? (단위: 조원)",
            "**1,987.4조원**이며, 코스닥 시장의 시가총액은 **341.3조원**",
        ),
    ]

    # ------------------------------------------------------------------
    # Build the dataset against a fresh TEST-profile system
    # ------------------------------------------------------------------
    system = build_system(Settings(profile=Profile.TEST))
    dataset = build_gold_dataset(
        system,
        name="korean_research",
        documents=documents,
        cases=case_specs,
    )

    # ------------------------------------------------------------------
    # Attach extra metadata (kind, computation fields, attribution)
    # ------------------------------------------------------------------
    extra: dict[str, dict[str, object]] = {
        "kr-001": {
            "kind": "retrieval",
            "source_inst": "한국은행",
            "source_url": "https://www.bok.or.kr/",
            "license": "공공저작물 자유이용허락 (공공누리 제1유형)",
        },
        "kr-002": {
            "kind": "retrieval",
            "source_inst": "한국은행",
            "source_url": "https://www.bok.or.kr/",
            "license": "공공저작물 자유이용허락 (공공누리 제1유형)",
        },
        "kr-003": {
            "kind": "table",
            "source_inst": "KDI",
            "source_url": "https://www.kdi.re.kr/",
            "license": "공공저작물 자유이용허락 (공공누리 제1유형)",
        },
        "kr-004": {
            "kind": "table",
            "source_inst": "KDI",
            "source_url": "https://www.kdi.re.kr/",
            "license": "공공저작물 자유이용허락 (공공누리 제1유형)",
        },
        "kr-005": {
            "kind": "computation",
            "expected_value": 2328.7,
            "tolerance": 0.1,
            "formula": "1987.4 + 341.3",
            "cited_numbers": [1987.4, 341.3],
            "source_inst": "자본시장연구원",
            "source_url": "https://www.kcmi.re.kr/",
            "license": "출처 표시 후 자유이용 가능",
        },
    }

    output_cases = []
    for case in dataset.cases:
        d = case.model_dump()
        d.update(extra.get(case.case_id, {}))
        output_cases.append(d)

    output = {"name": "korean_research", "cases": output_cases}

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {len(output_cases)} cases to {out_path}")
    for c in output_cases:
        print(f"  {c['case_id']}  kind={c['kind']}  gold={[g['chunk_id'] for g in c['gold']]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
