"""Retrieval and generation metrics."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from app.indexing.index import tokenize
from app.ingestion.chunker import Chunk

_CITE_RE = re.compile(r"\[\d+\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
OVERLAP_THRESHOLD = 0.5


# --------------------------------------------------------------------------- relevance


def has_gold(item: dict) -> bool:
    return bool(item.get("doc_id")) or bool(item.get("passage"))


def _span_overlap(chunk: Chunk, item: dict) -> int:
    if chunk.page != item.get("page") or chunk.metadata.get("section_idx") != item.get("section_idx"):
        return 0
    return max(0, min(chunk.end, item["end"]) - max(chunk.start, item["start"]))


def is_relevant(chunk: Chunk, item: dict) -> bool:
    if item.get("doc_id") and chunk.doc_id != item["doc_id"]:
        return False
    if item.get("start") is not None and item.get("end") is not None:
        overlap = _span_overlap(chunk, item)
        shorter = min(chunk.end - chunk.start, item["end"] - item["start"]) or 1
        return overlap / shorter >= OVERLAP_THRESHOLD
    if item.get("page") is not None:
        return chunk.page == item["page"]
    if item.get("passage"):
        gold = set(tokenize(item["passage"]))
        return bool(gold) and len(gold & set(tokenize(chunk.text))) / len(gold) >= 0.6
    return bool(item.get("doc_id"))


def gold_coverage(chunks: list[Chunk], item: dict) -> float:
    """Fraction of the gold span's characters covered by the given chunks (span-based recall)."""
    if item.get("start") is None or item.get("end") is None:
        return float(any(is_relevant(c, item) for c in chunks))
    length = item["end"] - item["start"]
    if length <= 0:
        return 0.0
    covered = np.zeros(length, dtype=bool)
    for c in chunks:
        if c.doc_id == item.get("doc_id") and _span_overlap(c, item):
            s, e = max(c.start, item["start"]) - item["start"], min(c.end, item["end"]) - item["start"]
            covered[s:e] = True
    return float(covered.mean())


# --------------------------------------------------------------------------- retrieval metrics


def retrieval_metrics(retrieved: list[Chunk], item: dict, n_relevant: int, ks=(1, 3, 5, 10)) -> dict:
    rel = [is_relevant(c, item) for c in retrieved]
    out: dict[str, float] = {}
    first = next((i for i, r in enumerate(rel) if r), None)
    for k in ks:
        top = rel[:k]
        out[f"hit@{k}"] = float(any(top))
        out[f"precision@{k}"] = sum(top) / k
        out[f"recall@{k}"] = gold_coverage(retrieved[:k], item)
        dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(top) if r)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, max(n_relevant, 1))))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    out["mrr"] = 1.0 / (first + 1) if first is not None and first < max(ks) else 0.0
    return out


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {}
    arr = np.asarray(latencies_ms)
    return {"latency_mean_ms": float(arr.mean()), "latency_p50_ms": float(np.percentile(arr, 50)), "latency_p95_ms": float(np.percentile(arr, 95))}


# --------------------------------------------------------------------------- generation metrics


def _content_tokens(text: str) -> list[str]:
    return tokenize(_CITE_RE.sub(" ", text))


def faithfulness(answer: str, context: str, threshold: float = 0.6) -> float:
    """Share of answer sentences whose content words are (mostly) present in the context."""
    ctx = set(tokenize(context))
    sentences = [s for s in _SENT_RE.split(_CITE_RE.sub(" ", answer)) if len(_content_tokens(s)) >= 3]
    if not sentences:
        return 1.0
    supported = 0
    for s in sentences:
        toks = _content_tokens(s)
        if sum(t in ctx for t in toks) / len(toks) >= threshold:
            supported += 1
    return supported / len(sentences)


def token_f1(prediction: str, reference: str) -> float:
    p, r = Counter(_content_tokens(prediction)), Counter(_content_tokens(reference))
    common = sum((p & r).values())
    if not common:
        return 0.0
    precision, recall = common / sum(p.values()), common / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0
