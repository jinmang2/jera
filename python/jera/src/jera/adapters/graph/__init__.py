"""Graph-based retrieval adapters.

Currently provides:

* :class:`~jera.adapters.graph.regex_entity_extractor.RegexEntityExtractor` —
  deterministic, offline noun-phrase/proper-noun extraction via regex.
* :class:`~jera.adapters.graph.hippo_retriever.HippoGraphRetriever` —
  HippoRAG-style multi-hop retrieval using Personalized PageRank over an
  entity co-occurrence graph (Gutiérrez et al., NeurIPS 2024, arXiv:2405.14831;
  HippoRAG 2, 2025).
"""

from __future__ import annotations

from jera.adapters.graph.hippo_retriever import HippoGraphRetriever
from jera.adapters.graph.regex_entity_extractor import RegexEntityExtractor

__all__ = ["HippoGraphRetriever", "RegexEntityExtractor"]
