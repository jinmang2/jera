"""LLM-assisted gold dataset builder for Korean research-report eval cases.

``ClaudeGoldGenerator`` is opt-in (requires the ``cloud`` extra and an API key).
It prompts Claude to propose case candidates, then:

1. Applies the **operand-provenance guard**: every ``cited_number`` in a
   computation case must appear (in normalized form) in the referenced
   supporting chunk text.  Cases that fail this guard are rejected so the
   final dataset never contains hallucinated operands.
2. Computes ``expected_value`` **deterministically** from ``cited_numbers``
   + ``operation`` using our own ``CalculatorTool`` — the LLM number is
   never trusted (Principle 3).
3. Carries ``source_inst`` / ``source_url`` / ``license`` from the corpus
   manifest entry into every emitted ``EvalCase`` (attribution propagation).

The operand-provenance guard and normalization helpers are public so they
can be unit-tested independently of the cloud dependency.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from jera.evaluation_contracts.dataset import CaseKind, EvalCase, GoldChunk

# ---------------------------------------------------------------------------
# Operand-provenance normalization helpers
# ---------------------------------------------------------------------------

# Korean numeric scale words mapped to multipliers
_KO_SCALE: dict[str, float] = {
    "조": 1e12,
    "억": 1e8,
    "만": 1e4,
}

# Full-width → half-width ASCII digit translation table
_FULLWIDTH_DIGITS = str.maketrans(
    "０１２３４５６７８９．",
    "0123456789.",
)


def _normalize_text(text: str) -> str:
    """Normalize a chunk text excerpt for operand-provenance matching.

    Steps applied:
    * Unicode NFKC normalization (full/half-width chars → ASCII).
    * Full-width digit → ASCII digit via explicit table.
    * Strip thousands-separator commas (``1,234`` → ``1234``).
    * Collapse whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_FULLWIDTH_DIGITS)
    # Remove comma used as thousands separator: digits,digits → digitsdigits
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _number_variants(value: float) -> list[str]:
    """Return candidate string representations of *value* to search for in text.

    Covers:
    * Plain decimal (``1234.5``, ``1234.0`` → ``1234``)
    * Integer form when the fractional part is zero (``1234``)
    * Comma-separated thousands (``1,234``)
    * Korean unit decompositions (``1억2000만`` style) — only for values that
      are exact multiples of a scale unit.

    Limitation — single-level decomposition only:
        Values like ``120,000,000`` (1억2000만) are decomposed as ``1억20000000``
        but *not* as ``1억2000만`` because the remainder after the highest-unit
        division is expressed as a plain integer, not further broken down into
        sub-units.  This means the guard may reject a computation case whose
        chunk text writes the number in multi-level Korean (e.g. ``1억2천만원``).
        Rejection is the **safe** direction — it prevents hallucinated operands
        from passing — so this limitation is acceptable for the current corpus.
        A future improvement could recurse into sub-units (만 → 천 → 백 → 십).

    All representations are returned as *normalized* strings (commas already
    stripped from the representation that uses them, so callers compare against
    ``_normalize_text`` output).
    """
    variants: list[str] = []

    # Plain float / int forms
    if value == int(value):
        variants.append(str(int(value)))
        # comma-thousands form — strip after adding so normalized comparison works
        int_str = f"{int(value):,}"
        variants.append(int_str.replace(",", ""))  # normalized (comma stripped)
    else:
        variants.append(str(value))

    # Korean scale decompositions
    for unit, mult in _KO_SCALE.items():
        if value >= mult and value % mult == 0:
            hi = int(value // mult)
            lo = int(value % mult)
            if lo == 0:
                variants.append(f"{hi}{unit}")
            else:
                # e.g. 12000 = 1만2000
                variants.append(f"{hi}{unit}{lo}")

    # Percentage / pp: if value looks like a percentage (0 < v < 100)
    # also try it as-is (the text may say "3.5%" or "3.5")
    # This is already covered by the plain-decimal form above.

    return variants


def operand_in_chunk(cited_number: float, chunk_text: str, *, tol: float = 1e-9) -> bool:
    """Return True if *cited_number* can be found in *chunk_text* after normalization.

    Normalization strips thousands commas, converts full/half-width digits,
    and checks Korean scale-unit decompositions.  A small *tol* is used when
    converting the float to a string to avoid floating-point representation
    artefacts (e.g. ``1234.0000000001``).

    This is the **operand-provenance guard** from the plan (S3).  It catches
    hallucinated operands: a number the LLM invented that never appears in the
    cited chunk.
    """
    norm = _normalize_text(chunk_text)
    for variant in _number_variants(cited_number):
        if variant in norm:
            return True
    # Fallback: try rounding to a few decimal places to absorb float noise
    for decimals in (0, 1, 2, 3):
        rounded = round(cited_number, decimals)
        for variant in _number_variants(rounded):
            if variant in norm:
                return True
    return False


# ---------------------------------------------------------------------------
# ClaudeGoldGenerator (opt-in / cloud extra)
# ---------------------------------------------------------------------------

_GOLD_PROMPT_TEMPLATE = """\
You are an evaluation-dataset annotator for a Korean RAG system.

Given the following document chunks from a research report, generate {n_cases} evaluation
cases. For each case, return a JSON object with these fields:
- "kind": one of "retrieval", "table", "computation"
- "query": a natural-language question in Korean (include a paraphrase variant)
- "supporting_chunk_ids": list of chunk_id strings that answer the query
- For "computation" kind only:
  - "cited_numbers": list of numeric values (floats) that appear literally in the chunks
  - "operation": an arithmetic expression using those numbers, e.g. "1234.5 + 678.9"

Return a JSON array of case objects. Do NOT compute expected_value yourself.

Document chunks:
{chunks_block}
"""


class ClaudeGoldGenerator:
    """Opt-in LLM-assisted gold generator (requires ``cloud`` extra + API key).

    Disabled by default — raises ``RuntimeError`` if instantiated without
    ``enabled=True``.  CI never instantiates this class.

    ``expected_value`` for computation cases is computed by our own
    ``CalculatorTool``, never taken from the LLM response (Principle 3).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-8",
        enabled: bool = False,
        max_tokens: int = 4096,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "ClaudeGoldGenerator is disabled by default. "
                "Pass enabled=True and api_key to use (paid; never in CI)."
            )
        if not api_key:
            raise RuntimeError("ClaudeGoldGenerator requires api_key when enabled.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ClaudeGoldGenerator requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate_cases(
        self,
        chunks: list[dict[str, Any]],
        *,
        source_inst: str,
        source_url: str,
        license_: str,
        n_cases: int = 5,
    ) -> list[EvalCase]:
        """Call Claude to propose eval cases, apply guards, and return valid ``EvalCase`` objects.

        *chunks* is a list of ``{"chunk_id": str, "text": str}`` dicts.
        Attribution fields are carried into every emitted case.
        Computation ``expected_value`` is computed deterministically by
        ``CalculatorTool``, not trusted from the LLM.
        """
        from jera.tooluse.tools import safe_eval  # local import — cloud path only

        chunks_block = "\n\n".join(f"[chunk_id={c['chunk_id']}]\n{c['text']}" for c in chunks)
        prompt = _GOLD_PROMPT_TEMPLATE.format(
            n_cases=n_cases,
            chunks_block=chunks_block,
        )

        resp = self._client.messages.create(  # pragma: no cover
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        import json

        raw_text = "".join(block.text for block in resp.content if block.type == "text")
        # Extract JSON array from the response (may be wrapped in markdown fences)
        json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON array found in Claude response:\n{raw_text}")
        raw_cases: list[dict[str, Any]] = json.loads(json_match.group())

        # Build a chunk_id → text lookup for provenance checks
        chunk_lookup: dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}

        results: list[EvalCase] = []
        for i, raw in enumerate(raw_cases):
            case_id = f"llm-{i:03d}"
            kind_str: str = raw.get("kind", "retrieval")
            try:
                kind = CaseKind(kind_str)
            except ValueError:
                continue  # skip unknown kinds

            gold = [
                GoldChunk(chunk_id=cid)
                for cid in raw.get("supporting_chunk_ids", [])
                if cid in chunk_lookup
            ]
            if not gold:
                continue  # no valid supporting chunks — skip

            expected_value: float | None = None
            formula: str | None = None
            cited_numbers: list[float] = []

            if kind == CaseKind.COMPUTATION:
                cited_numbers = [float(n) for n in raw.get("cited_numbers", [])]
                operation: str = raw.get("operation", "")

                # Operand-provenance guard: every cited_number must appear in
                # at least one of the supporting chunks.
                all_supporting_text = " ".join(chunk_lookup.get(g.chunk_id, "") for g in gold)
                rejected = [
                    n for n in cited_numbers if not operand_in_chunk(n, all_supporting_text)
                ]
                if rejected:
                    # Case rejected — hallucinated operands
                    continue

                # Compute expected_value deterministically (Principle 3)
                if operation:
                    try:
                        expected_value = safe_eval(operation)
                        formula = operation
                    except (ValueError, ZeroDivisionError):
                        continue  # invalid formula — skip

            results.append(
                EvalCase(
                    case_id=case_id,
                    query=raw.get("query", ""),
                    gold=gold,
                    kind=kind,
                    expected_value=expected_value,
                    formula=formula,
                    cited_numbers=cited_numbers,
                    source_inst=source_inst,
                    source_url=source_url,
                    license=license_,
                )
            )

        return results
