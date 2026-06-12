# QA Report — Korean Research RAG / fastembed local / Tool-Use Numeric QA (M4)

Date: 2026-06-12
Plan: `.omc/plans/rag-korean-research-toolqa-plan.md` · Spec: `.omc/specs/deep-interview-rag-korean-research-toolqa.md`
Execution: `/team` (3 workers, staged pipeline plan→exec→verify→fix)

## Verdict: PASS (gates green; code-review REQUEST-CHANGES resolved)

## Gates (integrated, offline, no extras/keys required)
- ruff check — clean
- ruff format --check — clean
- mypy --strict (`-p jera -p app`) — clean (83 source files)
- pytest — **155 passed**

## Task completion (7 build + 2 fix)
| # | Task | Worker | Result |
|---|---|---|---|
| 1 | EvalCase contract + numeric_accuracy | w1 | done |
| 2 | tool-use runtime (turn-based) + calculator + README | w2 | done (33 tests) |
| 3 | ToolAugmentedGenerator (port + typed run()) | w2 | done |
| 4 | settings/registry multilingual + tooluse wiring | w1 | done |
| 5 | computation eval (numeric accuracy) | w2 | done (13 tests) |
| 6 | corpus manifest + gold builder + operand guard | w3 | done (16 tests) |
| 7 | matrix eval (dense/sparse/hybrid × chunking) | w3 | done (28 tests) |
| 8 | FIX: calculator %/Mod + contracts | w2 | done |
| 9 | FIX: build-script import + scaffold docs | w3 | done |

## Independent code review (separate lane, opus) — 8 consensus checks PASS
turn-based loop genuine (not linearized; multi-block fixture + tool_use_id correlation verified) · CalculatorTool AST-safe · ToolAugmentedGenerator port-conformant (chunk citations only; number only in typed RunResult) · numeric_accuracy relative-tolerance · operand-provenance guard normalizes Korean numbers · matrix re-ingests per strategy (no cross-strategy cache) · paid/extra isolation (cloud + fastembed disabled-by-default) · hexagonal boundary intact (no `app.*` domain imports).

Review findings resolved: BLOCKER (`build_eval_dataset.py` `PyMuPdfParser`→`PyMuPDFParser`), MAJOR (`ast.Mod` added to calculator), MINORs documented.

## Honest scope notes (not gated, by design)
- **AC3 (~20–30 real cases) NOT fully met:** committed `data/eval/korean_research.json` is a 3-case SCAFFOLD. Real population is the user's opt-in build (download freely-distributable 한국은행/KDI/자본시장연구원 PDFs → `scripts/build_eval_dataset.py` with a key). Copyright-avoidance keeps PDFs git-ignored.
- **Heavy/paid runs are user-triggered scripts**, not CI: S0 fastembed model verify, bge-m3 matrix (`scripts/eval_local_matrix.py` with `--extra local`), Claude gold-gen, real `ClaudeToolUseGenerator`. CI is 100% offline/deterministic.
- **Opt-in Claude path** (`ClaudeToolUseGenerator`) does not yet handle `pause_turn` (adaptive thinking on Opus 4.8) — documented; harden before real extended-thinking runs.
