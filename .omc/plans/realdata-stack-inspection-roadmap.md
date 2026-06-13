# Roadmap: Real-Data Stack Inspection & Effectiveness (analyst-report RAG)

**Date:** 2026-06-14
**Status:** PLAN ONLY — recorded for an upcoming deep-dive. No implementation in this pass.

## Why this exists (the reset)

The session drifted into **해골물**: after a solid correctness audit, the agent began *manufacturing*
problems (Korean tokenizer/entity-extractor "failures") from synthetic examples with no real data —
and "fixed" one with a brittle hand-rolled Korean josa suffix list (reverted). That is exactly the
"대충 눈대중으로 굴러만 가게" anti-pattern the audit was meant to kill, applied to a CI stub.

Reset to the user's real workflow: **analyst report → build/inspect the stack → 고도화 → papers →
check implementations → measure performance → branch out.** Reports arrive over time; the first real
one is a Mirae Asset *Earnings Revision* PDF (Korean, financial, 24pp, chart/table-heavy).

## Ground truth: what a real report actually demands

First corpus sample (Mirae Asset, 2026.6.15, 유명간) — three content modes in one document, each
with different parsing needs (measured, not assumed):

| Mode | Where (this report) | Survives pymupdf text? | What it really needs |
|------|---------------------|------------------------|----------------------|
| Prose / summary bullets | p1, p7 (e.g. "합산 2026 영업이익 888조원, +211%YoY"; 상위 업종 랭킹) | ✅ clean | Jera text path is sufficient |
| Data tables | p22–23 (Top-20 종목: code·name·country·sector·mktcap·12M-Fwd 1W/1M·return) | ⚠️ cells extract, row↔column structure breaks | real table extraction (camelot/docling — opt-in, **never run live**) |
| Charts / graphs | p3–6 (~30 per-sector trend-chart grid), p7 (그림32–36 bars) | ❌ bar labels extract but orphaned from category; trend lines are raster | visual RAG / chart understanding (ColPali) — text RAG **misses chart-encoded answers entirely** |

Implications: answers are **numeric/financial** (888조원, +571%YoY) → tool-use numeric QA +
FinQA `numeric_accuracy` + citation-must-resolve-to-source-table/chart. Language is **Korean** →
real multilingual models (bge-m3 / bge-reranker-v2-m3) carry quality; the deterministic CI analogues
are Latin-oriented and must be **honestly labeled, not hacked**.

## Validated baseline (correctness — DONE)

Two-round re-audit vs primary sources fixed **4 real bugs** + 1 honesty fix (see `.omc/audit/AUDIT.md`):
pricing ($15/$75→$5/$25), DBSF (min-max→Qdrant 3σ), qdrant `recreate_collection` deprecation, CRAG
answer-bypass. Every technique verified correct-vs-spec or honestly-labeled-as-analogue. **This is the
floor — correctness is established. The open question is now EFFECTIVENESS.**

## Workstreams (for the deep-dive — drive every one from MEASURED real-data failure)

### W1 — Stack effectiveness: "다른 기술들 정말 실효한지 전부 체크"
Correctness ≠ usefulness. For each shipped technique, measure whether it *actually improves* retrieval/
answer quality on real analyst reports, or is dead weight. Use the existing `AblationRunner` /
`run_profile` harness against a real-report corpus (built as reports arrive). Honest reporting: on
saturated/easy corpora techniques tie — surface that, don't hide it. Output: a per-technique
keep/cut/conditional verdict backed by numbers, not vibes.
- Candidates to scrutinize hardest (cheap to claim, hard to prove useful): multi-query/HyDE,
  late chunking, MMR vs listwise rerank, context compression, proposition chunking, HippoRAG graph.

### W2 — Parsing realism (the crux: "추가로 파싱 어떻게 좀 바꿔볼지")
Ingest real reports; measure `parsing_metrics` (CER, table_f1, element-type accuracy, reading-order)
on real data, not fixtures. Fix from real failure.
- Tables: actually run camelot/docling on p22–23-style tables → structured rows; measure table_f1.
- Charts: the frontier. Decide ColPali/visual-RAG vs chart-data-extraction (e.g. line-series OCR) —
  text RAG demonstrably misses p3–6. Scope a real visual path (opt-in, real model).
- Prose: confirmed working; keep as the cheap baseline.

### W3 — Semantic-unit chunking + structural robustness ("의미단위로 … 조항 들어왔을 때 버티는지")
Verify semantic/hierarchical/proposition chunkers produce *meaningful* units on real reports (not
mid-table or mid-chart splits). **Forward requirement (no such data yet — design for it, don't invent
it):** future corpora may include clause/article-structured docs (계약서/약관/법령: 제1조, ①②③, 1.,
가.나.다.). Need a structure-aware chunker that respects 조/항/호 boundaries and keeps a clause atomic
with its heading breadcrumb. Stress-test when such a document actually arrives.

### W4 — Numeric/financial QA fidelity
Ensure tool-use numeric QA + `numeric_accuracy` + citation resolution work on real report questions
("반도체 2026F 영업이익?", "기여도 상위 업종?"). A numeric answer must cite the exact table/chart it
came from; a number the source doesn't contain must not be emitted.

### W5 — Korean production path (confirmed, not a TODO)
`registry.py:282` → bge-m3 (1024-dim multilingual) + bge-reranker-v2-m3. Real Korean quality rides on
these. Action is **honesty + measurement**, not hacking analogues: label deterministic adapters as
Latin-oriented CI analogues; measure real Korean retrieval/rerank quality on the real-report corpus.

## Guardrails (lessons from this session — non-negotiable)
1. **No 해골물.** Never manufacture a "failure" from a synthetic premise. Improvements are driven by a
   *measured* failure on a *real* document.
2. **Deterministic adapters are CI analogues.** Never hack production capability (Korean NER, real
   tokenization) into them. The real path is the opt-in real model/adapter behind the existing port.
3. **Falsifiable verification only.** An "OK" verdict is worthless without a source link + a
   reproducible test. Re-checkable evidence, not trust.
4. **Measure before building; build from the measured gap.** Plan → ingest → measure → fix → re-measure.

## Immediate next step (when the deep-dive starts)
Ingest the first real report through Jera's pipeline and **measure what survives vs breaks across the
three content modes** — that measurement, not speculation, sets the W1–W5 priority order.

Local research scratch (gitignored, not redistributable): `.omc/research/analyst/`.
