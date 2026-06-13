"""HippoRAG-style multi-hop retrieval via Personalized PageRank.

Reference
---------
Gutiérrez et al., "HippoRAG: Neurologically Inspired Long-Term Memory for Large
Language Models", NeurIPS 2024, arXiv:2405.14831; HippoRAG 2 (2025).

Design
------
**Indexing** — for each chunk, extract entities with the injected
:class:`~jera.ports.entity_extractor.EntityExtractor`. Add all extracted
entities as graph nodes.  For every *pair* of entities that co-occur in the same
chunk, add (or increment) a directed edge in *both* directions so the undirected
co-occurrence graph is represented as a symmetric directed adjacency.

**Retrieval** — extract entities from the query text; seed a Personalized
PageRank (PPR) distribution uniformly over those seed nodes; run power iteration
for a fixed number of steps (default 50) with a restart probability equal to the
damping factor (``alpha ≈ 0.5``).  The final PPR vector assigns a score to each
entity node.  Score each chunk as the *sum* of PPR scores over all entities it
contains; return the top-k chunks as :class:`~jera.domain.retrieval.ScoredChunk`
objects, tie-broken by ``chunk_id`` ascending.

**Pure Python** — no NetworkX, no NumPy.  The adjacency is a
``dict[str, dict[str, float]]`` (source → {target: weight}).  Power iteration
operates on plain ``dict[str, float]`` probability vectors.

Grace cases
-----------
* Empty graph → ``retrieve`` returns ``[]``.
* Query entities not found in graph → ``retrieve`` returns ``[]``.
* Chunks not matched by any PPR-scored entity → score ``0.0``; excluded from
  results unless *all* chunks score zero (in which case still excluded).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from jera.domain.chunk import Chunk
from jera.domain.retrieval import ScoredChunk
from jera.ports.entity_extractor import EntityExtractor

# ---------------------------------------------------------------------------
# Personalized PageRank (pure-Python power iteration)
# ---------------------------------------------------------------------------

_DEFAULT_ALPHA = 0.5  # restart / damping probability
_DEFAULT_ITERATIONS = 50


def _personalized_pagerank(
    adj: dict[str, dict[str, float]],
    seeds: set[str],
    *,
    alpha: float = _DEFAULT_ALPHA,
    iterations: int = _DEFAULT_ITERATIONS,
) -> dict[str, float]:
    """Run PPR and return a probability distribution over all nodes in *adj*.

    Parameters
    ----------
    adj:
        Adjacency map ``{node: {neighbour: edge_weight}}``.  Edge weights need
        not be normalized — this function normalizes each row internally.
    seeds:
        Set of seed node names.  Only nodes already present in *adj* are used;
        unknown seeds are silently ignored.
    alpha:
        Restart probability.  At each step, with probability *alpha* the random
        walk teleports back to the seed distribution; with probability
        ``1 - alpha`` it follows a random out-edge.
    iterations:
        Number of power-iteration steps.  50 is enough for convergence on
        graphs with hundreds of nodes.

    Returns
    -------
    dict[str, float]
        Probability distribution over graph nodes.  Sums to ≈ 1.0.
        Returns an empty dict when *adj* is empty or no valid seeds exist.
    """
    nodes = list(adj.keys())
    if not nodes:
        return {}

    # Restrict seeds to nodes that exist in the graph.
    valid_seeds = seeds & adj.keys()
    if not valid_seeds:
        return {}

    # Build row-stochastic transition matrix as a dict of normalized rows.
    # Dangling nodes (no out-edges) get a uniform transition to all nodes.
    n = len(nodes)

    # Precompute normalized out-weights per node.
    trans: dict[str, dict[str, float]] = {}
    for src, neighbours in adj.items():
        total = sum(neighbours.values())
        if total > 0.0:
            trans[src] = {dst: w / total for dst, w in neighbours.items()}
        else:
            # Dangling: uniform over all nodes.
            uniform = 1.0 / n
            trans[src] = {node: uniform for node in nodes}

    # Seed distribution: uniform over valid seeds.
    seed_prob = 1.0 / len(valid_seeds)
    seed_dist: dict[str, float] = {s: seed_prob for s in valid_seeds}

    # Initial distribution = seed distribution.
    rank: dict[str, float] = dict(seed_dist)

    for _ in range(iterations):
        new_rank: dict[str, float] = {node: 0.0 for node in nodes}

        # Random-walk contribution: (1 - alpha) * M^T * rank
        for src in nodes:
            src_score = rank.get(src, 0.0)
            if src_score == 0.0:
                continue
            for dst, prob in trans[src].items():
                new_rank[dst] = new_rank[dst] + (1.0 - alpha) * src_score * prob

        # Restart contribution: alpha * seed_dist
        for s, p in seed_dist.items():
            new_rank[s] = new_rank[s] + alpha * p

        rank = new_rank

    return rank


# ---------------------------------------------------------------------------
# HippoGraphRetriever
# ---------------------------------------------------------------------------


class HippoGraphRetriever:
    """HippoRAG-style retriever: entity co-occurrence graph + Personalized PageRank.

    Parameters
    ----------
    extractor:
        Any :class:`~jera.ports.entity_extractor.EntityExtractor` implementation.
        Injected at construction time so the retriever is testable with a stub.
    alpha:
        PPR restart probability (default ``0.5``).
    iterations:
        Power-iteration steps for PPR (default ``50``).
    """

    def __init__(
        self,
        extractor: EntityExtractor,
        *,
        alpha: float = _DEFAULT_ALPHA,
        iterations: int = _DEFAULT_ITERATIONS,
    ) -> None:
        self._extractor = extractor
        self._alpha = alpha
        self._iterations = iterations

        # Adjacency: entity → {entity: co-occurrence count (float)}
        self._adj: dict[str, dict[str, float]] = {}

        # Reverse index: entity → set of chunk_ids that contain it
        self._entity_to_chunks: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # GraphRetriever protocol
    # ------------------------------------------------------------------

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Incorporate *chunks* into the entity co-occurrence graph.

        Can be called multiple times to add chunks incrementally.  Re-indexing
        the same chunk_id is idempotent only if the text (and therefore the
        extracted entities) did not change; otherwise edge weights will
        accumulate.
        """
        for chunk in chunks:
            entities = self._extractor.extract(chunk.text)
            if not entities:
                continue

            # Register entities in the adjacency and the reverse index.
            for ent in entities:
                if ent not in self._adj:
                    self._adj[ent] = {}
                self._entity_to_chunks.setdefault(ent, set()).add(chunk.chunk_id)

            # Add co-occurrence edges (undirected → both directions).
            for a, b in combinations(entities, 2):
                self._adj[a][b] = self._adj[a].get(b, 0.0) + 1.0
                self._adj[b][a] = self._adj[b].get(a, 0.0) + 1.0

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Rank chunks by PPR signal seeded on query entities.

        Returns an empty list when the graph is empty or the query yields no
        entities present in the graph.
        """
        if not self._adj:
            return []

        query_entities = set(self._extractor.extract(query))
        # Restrict to entities that actually exist in the graph.
        seeds = query_entities & self._adj.keys()
        if not seeds:
            return []

        ppr = _personalized_pagerank(
            self._adj,
            seeds,
            alpha=self._alpha,
            iterations=self._iterations,
        )

        # Score each chunk: sum PPR scores of all its entities.
        chunk_scores: dict[str, float] = {}
        for ent, ppr_score in ppr.items():
            if ppr_score <= 0.0:
                continue
            for chunk_id in self._entity_to_chunks.get(ent, set()):
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + ppr_score

        if not chunk_scores:
            return []

        # Sort: descending score, tie-break ascending chunk_id.
        ranked = sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))

        return [
            ScoredChunk(chunk_id=cid, score=score, components={"ppr": score})
            for cid, score in ranked[:top_k]
        ]
