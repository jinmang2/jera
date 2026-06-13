"""ConnectiveDecomposer — deterministic multi-hop query decomposition, no LLM, CI-real.

Detects compound / multi-hop queries by matching a small, documented set of English connective
patterns and splits them into an *ordered* list of sub-questions.  Korean connectives are
intentionally deferred (// future: Korean 그리고/및 patterns).

Rule set (applied in order; first match wins):
  R1  "compare X with/and Y" / "difference between X and Y"
        → ["What is X?", "What is Y?"]
  R2  "how does X affect Y" / "what is the effect of X on Y"
        → ["What is X?", "How does X affect Y?"]
  R3  "both ... and ..."  — two balanced NPs
        → ["Tell me about <np1>", "Tell me about <np2>"]
  R4  Binary "and" or "as well as" joining two clauses > MIN_CLAUSE_CHARS each
        → [clause1, clause2]  (original order)
  R5  Questions joined by a comma + "and"
        → [q1, q2]
  Fallback: [original query]  (non-compound, safe no-op)

An LLM-based decomposer is a straightforward future extension — implement the QueryDecomposer
Protocol with a ``decompose`` method that calls the LLM and falls back to this adapter on
failure.
"""

from __future__ import annotations

import re

# Minimum character length for a clause to be treated as a meaningful sub-question.
_MIN_CLAUSE_CHARS = 5

# ── R1: compare / difference ────────────────────────────────────────────────
_COMPARE_WITH = re.compile(
    r"^\s*(?:compare|comparison of|difference between|differences between)\s+"
    r"(.+?)\s+(?:with|and|vs\.?|versus)\s+(.+?)\s*\??\s*$",
    re.IGNORECASE,
)

# ── R2: causal / effect ──────────────────────────────────────────────────────
_AFFECT = re.compile(
    r"^\s*(?:how does|how do|what is the effect of|what are the effects of)\s+"
    r"(.+?)\s+(?:affect|affect\s+the|on|impact)\s+(.+?)\s*\??\s*$",
    re.IGNORECASE,
)

# ── R3: "both X and Y" ───────────────────────────────────────────────────────
_BOTH_AND = re.compile(
    r"\bboth\s+(.+?)\s+and\s+(.+?)(?:\s*\?|$)",
    re.IGNORECASE,
)

# ── R4: binary "and" / "as well as" (not inside a "both…and") ───────────────
_BINARY_AND = re.compile(
    r"^(.+?)\s+(?:and also|as well as|and)\s+(.+?)\s*\??\s*$",
    re.IGNORECASE,
)

# ── R5: comma + "and" joining two interrogative clauses ─────────────────────
_COMMA_AND = re.compile(
    r"^(.+?)\s*,\s*and\s+(.+?)\s*\??\s*$",
    re.IGNORECASE,
)


def _clean(s: str) -> str:
    """Strip and ensure the sub-question ends with '?'."""
    s = s.strip().rstrip("?").strip()
    return s + "?" if s else s


def _noun_question(np: str) -> str:
    """Wrap a bare noun phrase as a simple 'What is <np>?' question."""
    np = np.strip().rstrip("?")
    if re.match(r"^\s*(?:what|where|when|who|how|why|which|is|are|does|do)\b", np, re.IGNORECASE):
        return _clean(np)
    return f"What is {np}?"


class ConnectiveDecomposer:
    """Deterministic multi-hop decomposer based on English connective patterns (no LLM).

    A simple (non-compound) query returns ``[query]`` unchanged — safe no-op that lets the
    :class:`~jera.pipeline.decompositional.DecompositionalQueryPipeline` degrade gracefully
    to single-step retrieval.
    """

    strategy = "connective"
    version = "1.0"

    def decompose(self, query: str) -> list[str]:
        """Return ordered sub-questions; non-compound queries return ``[query]``."""
        text = query.strip()
        if not text:
            return [query]

        # R1 — compare / difference between X and Y
        m = _COMPARE_WITH.match(text)
        if m:
            x, y = m.group(1).strip(), m.group(2).strip()
            if len(x) >= _MIN_CLAUSE_CHARS and len(y) >= _MIN_CLAUSE_CHARS:
                return [_noun_question(x), _noun_question(y)]

        # R2 — causal / effect
        m = _AFFECT.match(text)
        if m:
            cause, effect = m.group(1).strip(), m.group(2).strip()
            if len(cause) >= _MIN_CLAUSE_CHARS and len(effect) >= _MIN_CLAUSE_CHARS:
                return [_noun_question(cause), _clean(text)]

        # R3 — both X and Y
        m = _BOTH_AND.search(text)
        if m:
            x, y = m.group(1).strip(), m.group(2).strip()
            if len(x) >= _MIN_CLAUSE_CHARS and len(y) >= _MIN_CLAUSE_CHARS:
                return [_noun_question(x), _noun_question(y)]

        # R4 — binary "and" / "as well as" (applied only when NOT already matched above)
        m = _BINARY_AND.match(text)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            # Guard: don't split if either side is too short (noise from possessives, etc.)
            if len(left) >= _MIN_CLAUSE_CHARS and len(right) >= _MIN_CLAUSE_CHARS:
                # Avoid splitting on idiomatic "X and Y" that reads as a single concept:
                # a heuristic: if right side has no verb-like token, wrap as noun questions.
                has_verb = re.search(
                    r"\b(?:is|are|was|were|be|been|being|have|has|had|do|does|did"
                    r"|will|would|can|could|should|shall|may|might|must"
                    r"|located|found|known|called|named|used|made|built|situated)\b",
                    right,
                    re.IGNORECASE,
                )
                if has_verb:
                    return [_clean(left), _clean(right)]
                else:
                    return [_noun_question(left), _noun_question(right)]

        # R5 — comma + "and" joining clauses
        m = _COMMA_AND.match(text)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            if len(left) >= _MIN_CLAUSE_CHARS and len(right) >= _MIN_CLAUSE_CHARS:
                return [_clean(left), _clean(right)]

        # Fallback — not compound; safe no-op
        return [query]
