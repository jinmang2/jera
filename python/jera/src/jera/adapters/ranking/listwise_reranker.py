"""Listwise rerankers — deterministic offline variant + opt-in Claude (RankGPT-style).

References
----------
- RankLLM: Pradeep et al., SIGIR 2025. arXiv:2505.19284.
- Reranking survey: arXiv:2512.16236.
- RankGPT: Sun et al., 2023. Permutation-distillation listwise reranking.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from jera.domain.retrieval import ScoredChunk

_DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w+\b")


def _content_tokens(text: str) -> list[str]:
    """Return lowercased word tokens from *text*, excluding single-char tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


# ---------------------------------------------------------------------------
# ListwiseReranker — offline, deterministic, no LLM
# ---------------------------------------------------------------------------


class ListwiseReranker:
    """Listwise reranker using query-term coverage weighted by corpus rarity.

    A candidate's score is the mean IDF-weighted coverage of *distinct* query
    content-tokens present in the chunk text.  IDF is computed across the
    whole candidate set (document frequency = number of candidates whose text
    contains the token), making the score genuinely listwise: changing the
    candidate pool changes the per-token weights and therefore the ranking.

    Algorithm
    ---------
    1. Tokenise the query into a set of distinct content tokens Q.
    2. For each token t in Q compute df(t) = number of candidates whose text
       contains t.  Weight w(t) = 1 / df(t)  (rare tokens score higher).
    3. For each candidate c, compute
           score(c) = sum(w(t) for t in Q if t in tokens(c.text))
                      / sum(w(t) for t in Q)
       so the score is in [0, 1] with 1 meaning full coverage of all query
       tokens weighted by rarity.
    4. Sort descending by score, tie-break chunk_id ascending; return top_k.
    """

    model_id = "listwise-rerank-v1"

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Return up to *top_k* candidates reranked by listwise IDF-weighted coverage."""
        if not candidates:
            return []

        query_tokens = set(_content_tokens(query))
        if not query_tokens:
            # No scoreable query tokens — fall back to score-desc, chunk_id-asc ordering.
            return sorted(candidates, key=lambda c: (-c.score, c.chunk_id))[:top_k]

        # Build per-candidate token sets once.
        cand_token_sets: list[set[str]] = [
            set(_content_tokens(c.chunk.text if c.chunk is not None else "")) for c in candidates
        ]

        # Document frequency: how many candidates contain each query token.
        df: dict[str, int] = {}
        for t in query_tokens:
            df[t] = sum(1 for ts in cand_token_sets if t in ts)
            if df[t] == 0:
                df[t] = 1  # token absent from every candidate — weight still defined

        # IDF weights (inverse document frequency, unnormalised).
        weights = {t: 1.0 / df[t] for t in query_tokens}
        total_weight = sum(weights.values())

        # Score each candidate.
        scored: list[tuple[float, str, ScoredChunk]] = []
        for cand, tok_set in zip(candidates, cand_token_sets, strict=True):
            coverage = sum(weights[t] for t in query_tokens if t in tok_set)
            score = coverage / total_weight if total_weight > 0.0 else 0.0
            scored.append((score, cand.chunk_id, cand))

        # Sort: score desc, chunk_id asc (deterministic tie-break).
        scored.sort(key=lambda x: (-x[0], x[1]))

        return [
            sc.model_copy(update={"components": {"listwise": s}}) for s, _, sc in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# ClaudeListwiseReranker — opt-in, disabled-by-default (RankGPT-style)
# ---------------------------------------------------------------------------

_RANKGPT_PROMPT = (
    "I will provide you with {n} passages, each indicated by number identifier [].\n"
    "Rank the passages based on their relevance to the search query: {query}\n\n"
    "{passages}\n\n"
    "Search Query: {query}\n"
    "Rank the {n} passages above based on their relevance to the search query. "
    "The passages should be listed in descending order using identifiers. "
    "The most relevant passages should be listed first. "
    "The output format should be [] > [] > ..., e.g. [1] > [2] > [3]. "
    "Only respond with the ranking, no explanations."
)


def _build_passages_block(candidates: Sequence[ScoredChunk]) -> str:
    lines: list[str] = []
    for i, cand in enumerate(candidates, start=1):
        text = cand.chunk.text if cand.chunk is not None else ""
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _parse_permutation(response_text: str, n: int) -> list[int]:
    """Parse a RankGPT permutation string like '[2] > [1] > [3]' into 0-based indices.

    Returns a list of 0-based indices in ranked order.  Any index outside
    [1, n] is silently dropped; missing indices are appended in original order.
    """
    found = [int(m) for m in re.findall(r"\[(\d+)\]", response_text)]
    seen: set[int] = set()  # 0-based indices already added
    order: list[int] = []
    for rank in found:
        idx = rank - 1  # convert to 0-based
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    # Append any missing 0-based indices in original order.
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order


class ClaudeListwiseReranker:
    """RankGPT-style listwise reranker backed by an Anthropic Claude model.

    Disabled by default — live calls cost money and require a network.  Pass
    ``enabled=True`` and a valid ``api_key`` to activate.

    The model receives all candidates numbered ``[1]``..``[n]``, is asked to
    return a permutation (``[2] > [1] > [3]``), and the permutation is applied
    to reorder the candidates.
    """

    def __init__(
        self,
        model: str = _DEFAULT_CLAUDE_MODEL,
        api_key: str | None = None,
        enabled: bool = False,
        max_tokens: int = 256,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "ClaudeListwiseReranker is disabled by default. "
                "Pass enabled=True and an api_key "
                "(paid live calls; never enabled in automated tests)."
            )
        if not api_key:
            raise RuntimeError("ClaudeListwiseReranker requires an api_key when enabled.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ClaudeListwiseReranker requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.model_id = model
        self._max_tokens = max_tokens

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Rerank *candidates* using a Claude permutation response, return top_k."""
        if not candidates:
            return []

        n = len(candidates)
        passages_block = _build_passages_block(candidates)
        prompt = _RANKGPT_PROMPT.format(n=n, query=query, passages=passages_block)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        order = _parse_permutation(text, n)

        return [
            candidates[i].model_copy(update={"components": {"listwise_rank": float(rank + 1)}})
            for rank, i in enumerate(order[:top_k])
        ]
