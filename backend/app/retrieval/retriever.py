"""Hybrid retrieval: dense + BM25 candidates -> fusion (RRF / weighted) -> cross-encoder rerank."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.core.config import RetrievalConfig
from app.indexing.index import HybridIndex, extract_identifiers
from app.ingestion.chunker import Chunk

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    collection: str | None = None
    boost: float = 0.0


@dataclass
class Candidates:
    vector: list[tuple[int, float]] = field(default_factory=list)
    keyword: list[tuple[int, float]] = field(default_factory=list)


def get_candidates(
    index: HybridIndex,
    query: str,
    embedder,
    k: int,
    mode: str = "hybrid",
    doc_ids: set[str] | None = None,
    query_vec: np.ndarray | None = None,
) -> Candidates:
    cands = Candidates()
    if mode in ("hybrid", "vector"):
        if query_vec is None:
            query_vec = embedder.encode([query])[0]
        cands.vector = index.vector_search(query_vec, k, doc_ids)
    if mode in ("hybrid", "keyword"):
        cands.keyword = index.keyword_search(query, k, doc_ids)
    return cands


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {i: 1.0 for i in values}
    return {i: (v - lo) / (hi - lo) for i, v in values.items()}


def fuse(cands: Candidates, cfg: RetrievalConfig, k: int | None = None) -> list[tuple[int, float]]:
    """Fuse candidate lists; returns [(chunk_idx, fused_score)] sorted desc."""
    k = k or cfg.candidate_k
    vec = cands.vector[:k] if cfg.mode != "keyword" else []
    kw = cands.keyword[:k] if cfg.mode != "vector" else []
    scores: dict[int, float] = {}
    if cfg.fusion == "rrf" or not (vec and kw):
        if vec and kw:
            for rank, (idx, _) in enumerate(vec):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
            for rank, (idx, _) in enumerate(kw):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        else:  # single retriever: keep its native scores
            scores = {idx: s for idx, s in (vec or kw)}
    else:
        nv, nk = _minmax(dict(vec)), _minmax(dict(kw))
        for idx in set(nv) | set(nk):
            scores[idx] = cfg.alpha * nv.get(idx, 0.0) + (1 - cfg.alpha) * nk.get(idx, 0.0)
    return sorted(scores.items(), key=lambda x: -x[1])


def identifier_boosts(index: HybridIndex, query: str, boost: float, doc_ids: set[str] | None = None) -> dict[int, float]:
    """Exact-identifier matching: generic rerankers ignore clause/policy numbers, users rely on them."""
    if boost <= 0:
        return {}
    ids = extract_identifiers(query)
    return {i: boost for i in index.identifier_lookup(ids, doc_ids)} if ids else {}


def apply_boosts(fused: list[tuple[int, float]], boosts: dict[int, float]) -> list[tuple[int, float]]:
    """Add boosted chunks to the candidate pool and lift them by ``boost`` x the best fused score."""
    if not boosts:
        return fused
    scores = dict(fused)
    top = max(scores.values(), default=1.0) or 1.0
    for idx, b in boosts.items():
        scores[idx] = scores.get(idx, 0.0) + b * top
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(
    index: HybridIndex,
    query: str,
    cfg: RetrievalConfig,
    embedder,
    reranker=None,
    doc_ids: set[str] | None = None,
    collection: str | None = None,
) -> tuple[list[Hit], dict[str, float]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    with index.lock:
        cands = get_candidates(index, query, embedder, cfg.candidate_k, cfg.mode, doc_ids)
        boosts = identifier_boosts(index, query, cfg.identifier_boost, doc_ids)
        fused = apply_boosts(fuse(cands, cfg), boosts)
        vec_scores, kw_scores = dict(cands.vector), dict(cands.keyword)
        hits = [
            Hit(
                chunk=index.chunks[idx],
                score=score,
                vector_score=vec_scores.get(idx),
                keyword_score=kw_scores.get(idx),
                collection=collection,
                boost=boosts.get(idx, 0.0),
            )
            for idx, score in fused
        ]
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    if cfg.use_reranker and reranker is not None and hits:
        t1 = time.perf_counter()
        hits = rerank_hits(query, hits, reranker)
        if cfg.min_rerank_score is not None:
            hits = [h for h in hits if (h.rerank_score or 0) >= cfg.min_rerank_score]
        timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
    return hits[: cfg.top_k], timings


def rerank_hits(query: str, hits: list[Hit], reranker) -> list[Hit]:
    probs = reranker.score(query, [h.chunk.embed_text for h in hits])
    for h, p in zip(hits, probs):
        h.rerank_score = float(p)
        h.score = float(p) + h.boost
    return sorted(hits, key=lambda h: -h.score)
