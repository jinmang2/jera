"""HeuristicQueryRouter — deterministic Adaptive-RAG complexity classifier, no LLM.

Routing logic (in priority order):

1. MULTI_STEP  — query contains any multi-hop cue token (configurable).  Cues cover
   comparative/relational phrasing that implies at least two retrieval hops:
   "compare", "difference between", "relationship", "how does … affect", "both", "versus", etc.

2. NO_RETRIEVAL — query is self-contained and needs no grounding: either
   (a) very short (≤ ``no_retrieval_max_tokens`` whitespace-delimited tokens, default 4), or
   (b) matches the pure-arithmetic pattern (digits / operators only).

3. SINGLE_STEP  — everything else.

All matching is case-insensitive.  The cue sets are passed as constructor arguments so
callers can extend them (e.g. add Korean multi-hop connectives) without subclassing.

Future opt-in
-------------
A TF-IDF + SVM variant trained on labelled complexity annotations (as described in Jeong et al.
NAACL 2024, and benchmarked at ~93 % in RAGRouter-Bench 2026) can satisfy the same
``QueryRouter`` protocol.  Replace this adapter in ``AdaptiveQueryPipeline`` to enable it —
no pipeline-layer changes required.
"""

from __future__ import annotations

import re

from jera.ports.query_router import QueryComplexity, QueryRouter

# ---------------------------------------------------------------------------
# Default multi-hop cue phrases (matched as substrings, case-insensitive).
# ---------------------------------------------------------------------------
_DEFAULT_MULTI_HOP_CUES: frozenset[str] = frozenset(
    {
        "compare",
        "comparison",
        "difference between",
        "differences between",
        "relationship",
        "relation between",
        "how does",
        "how do",
        "both",
        "and how",
        "versus",
        "vs.",
        "vs ",
        "contrast",
        "similarities",
        "similarity between",
        "distinguish",
        "while also",
        "as well as",
    }
)

# Pure arithmetic: optional leading sign, digits, operators (+−×÷*/^%), spaces, parens.
_ARITHMETIC_RE = re.compile(r"^[\d\s\+\-\*\/\^\(\)\.%±×÷]+$")


class HeuristicQueryRouter:
    """Deterministic three-tier Adaptive-RAG query complexity classifier.

    Parameters
    ----------
    multi_hop_cues:
        Iterable of lower-cased substring cues that indicate multi-hop intent.
        Defaults to ``_DEFAULT_MULTI_HOP_CUES``.
    no_retrieval_max_tokens:
        Queries whose whitespace-token count is ≤ this threshold are classified
        as NO_RETRIEVAL (too short to be a real retrieval query).  Default: 4.
    """

    def __init__(
        self,
        *,
        multi_hop_cues: frozenset[str] | None = None,
        no_retrieval_max_tokens: int = 4,
    ) -> None:
        self._cues: frozenset[str] = (
            multi_hop_cues if multi_hop_cues is not None else _DEFAULT_MULTI_HOP_CUES
        )
        self._no_retrieval_max_tokens = no_retrieval_max_tokens

    # ------------------------------------------------------------------
    # QueryRouter protocol
    # ------------------------------------------------------------------

    def route(self, query: str) -> QueryComplexity:
        """Classify *query* into NO_RETRIEVAL / SINGLE_STEP / MULTI_STEP.

        Priority: MULTI_STEP > NO_RETRIEVAL > SINGLE_STEP.
        """
        normalized = query.strip().lower()

        # --- MULTI_STEP: multi-hop cue present? ---
        if any(cue in normalized for cue in self._cues):
            return QueryComplexity.MULTI_STEP

        # --- NO_RETRIEVAL: trivially short or pure arithmetic? ---
        tokens = normalized.split()
        if len(tokens) <= self._no_retrieval_max_tokens:
            return QueryComplexity.NO_RETRIEVAL
        if _ARITHMETIC_RE.match(normalized):
            return QueryComplexity.NO_RETRIEVAL

        # --- Default: single focused retrieval pass ---
        return QueryComplexity.SINGLE_STEP


# Verify the class satisfies the protocol at import time (free runtime check).
assert isinstance(HeuristicQueryRouter(), QueryRouter)
