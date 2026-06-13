"""Visual late-interaction (ColPali-style) — image↔image MaxSim retrieval (non-tautological).

A query image that shares an identical visual PATCH with one document page must out-rank a
document with no shared patch, scored by the existing MaxSimVectorStore. The shared patch yields a
cosine-1.0 per-patch match that MaxSim sums — a genuine consequence of the patch geometry, not a
mock. (Cross-modal text→image alignment needs the real VLM; that path is mechanism-only here.)
"""

from __future__ import annotations

from jera.adapters.embedding.visual_multivector import VisualMultiVectorEmbedding
from jera.adapters.vector_store.maxsim_store import MaxSimVectorStore

# 4 patches of 32 bytes each (n_patches=4). docA shares patch 0 (the "signature") with the query.
_SIG = b"S" * 32
_QUERY = _SIG + b"q" * 32 + b"r" * 32 + b"s" * 32  # patch0 == _SIG
_DOC_A = _SIG + b"a" * 32 + b"b" * 32 + b"c" * 32  # patch0 == _SIG (shares a visual patch)
_DOC_B = b"z" * 128  # no shared patch


def _emb() -> VisualMultiVectorEmbedding:
    return VisualMultiVectorEmbedding(dimensions=64, n_patches=4)


def test_visual_maxsim_ranks_patch_sharing_page_first() -> None:
    emb = _emb()
    store = MaxSimVectorStore()
    store.add(
        [("docA", emb.embed_image_patches(_DOC_A)), ("docB", emb.embed_image_patches(_DOC_B))]
    )

    ranked = store.search_maxsim(emb.embed_query_image(_QUERY), top_k=2)
    assert [c.chunk_id for c in ranked][0] == "docA"  # shared visual patch wins
    by_id = {c.chunk_id: c.score for c in ranked}
    assert by_id["docA"] > by_id["docB"]


def test_patch_count_and_determinism() -> None:
    emb = _emb()
    patches = emb.embed_image_patches(_DOC_A)
    assert len(patches) == 4  # n_patches
    assert all(len(v) == 64 for v in patches)  # dimensions
    # deterministic: same bytes → identical patch vectors
    assert emb.embed_image_patches(_DOC_A) == patches
    # the shared signature patch is identical across query and docA (cosine 1.0)
    assert emb.embed_image_patches(_QUERY)[0] == emb.embed_image_patches(_DOC_A)[0]


def test_short_and_empty_images_do_not_crash() -> None:
    emb = _emb()
    assert len(emb.embed_image_patches(b"")) == 4
    assert len(emb.embed_image_patches(b"ab")) == 4


def test_text_query_path_produces_token_vectors() -> None:
    qv = _emb().embed_query_multi("invoice total amount")
    assert len(qv) == 3 and all(len(v) == 64 for v in qv)
