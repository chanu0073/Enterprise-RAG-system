"""Retrieval pipeline optimization.

Searches chunk size, retrieval mode, fusion method/weight, candidate depth and reranking for
the configuration that maximises a quality metric, optionally penalised by latency:

    objective = metric - latency_weight * (estimated_latency_ms / 1000)

Expensive work is shared across trials: one index per chunking config, one query embedding per
question, candidate lists fetched once at maximum depth, and rerank scores cached per pair.
Tuning runs on the ``train`` split; the winner is confirmed on the held-out ``test`` split.
"""

from __future__ import annotations

import itertools
import logging
import random
import time

from app.core.config import RetrievalConfig, get_retrieval_store
from app.core.db import get_db, new_id, now
from app.evaluation.metrics import aggregate, has_gold, is_relevant, retrieval_metrics
from app.evaluation.runner import EVAL_DEPTH, IndexProvider
from app.finetune.dataset import load_dataset, split_items
from app.indexing.models import get_registry, load_embedder, load_reranker
from app.retrieval.retriever import Candidates, apply_boosts, fuse, identifier_boosts

log = logging.getLogger(__name__)


def _trial_space(chunk_sizes, overlap_ratio, modes, fusions, alphas, candidate_ks, rerank) -> list[dict]:
    trials, seen = [], set()
    for cs, mode, fusion, alpha, ck, rr in itertools.product(chunk_sizes, modes, fusions, alphas, candidate_ks, rerank):
        if mode != "hybrid":
            fusion, alpha = "rrf", 0.5  # irrelevant for single retrievers
        elif fusion == "rrf":
            alpha = 0.5
        t = {"chunk_size": cs, "chunk_overlap": int(cs * overlap_ratio), "mode": mode, "fusion": fusion,
             "alpha": alpha, "candidate_k": ck, "use_reranker": rr}
        key = tuple(sorted(t.items()))
        if key not in seen:
            seen.add(key)
            trials.append(t)
    return trials


class _QueryCache:
    """Per (chunking, query) cache of candidates and rerank scores."""

    def __init__(self, index, items, embedder, reranker, max_k):
        self.index, self.items, self.reranker = index, items, reranker
        t0 = time.perf_counter()
        qvecs = embedder.encode([i["question"] for i in items]) if items else []
        self.embed_ms = (time.perf_counter() - t0) * 1000 / max(len(items), 1)
        self.cands: list[Candidates] = []
        vec_ms = kw_ms = 0.0
        for item, qv in zip(items, qvecs):
            t = time.perf_counter()
            vec = index.vector_search(qv, max_k)
            vec_ms += time.perf_counter() - t
            t = time.perf_counter()
            kw = index.keyword_search(item["question"], max_k)
            kw_ms += time.perf_counter() - t
            self.cands.append(Candidates(vector=vec, keyword=kw))
        n = max(len(items), 1)
        self.vec_ms, self.kw_ms = vec_ms * 1000 / n, kw_ms * 1000 / n
        self.n_relevant = [sum(1 for c in index.chunks if c.doc_id == it.get("doc_id") and is_relevant(c, it)) for it in items]
        self._rr: dict[tuple[int, int], float] = {}
        self.rerank_ms_per_pair = None

    def rerank_scores(self, qi: int, idxs: list[int]) -> dict[int, float]:
        missing = [i for i in idxs if (qi, i) not in self._rr]
        if missing:
            t = time.perf_counter()
            scores = self.reranker.score(self.items[qi]["question"], [self.index.chunks[i].embed_text for i in missing])
            per_pair = (time.perf_counter() - t) * 1000 / len(missing)
            self.rerank_ms_per_pair = per_pair if self.rerank_ms_per_pair is None else 0.9 * self.rerank_ms_per_pair + 0.1 * per_pair
            for i, s in zip(missing, scores):
                self._rr[(qi, i)] = float(s)
        return {i: self._rr[(qi, i)] for i in idxs}

    def evaluate(self, cfg: RetrievalConfig) -> tuple[dict, float]:
        rows, n_cands = [], []
        for qi, item in enumerate(self.items):
            boosts = identifier_boosts(self.index, item["question"], cfg.identifier_boost)
            fused = apply_boosts(fuse(self.cands[qi], cfg, cfg.candidate_k), boosts)
            idxs = [i for i, _ in fused]
            if cfg.use_reranker and self.reranker is not None and idxs:
                rr = self.rerank_scores(qi, idxs)
                idxs = sorted(idxs, key=lambda i: -(rr[i] + boosts.get(i, 0.0)))
            n_cands.append(len(fused))
            retrieved = [self.index.chunks[i] for i in idxs[:EVAL_DEPTH]]
            rows.append(retrieval_metrics(retrieved, item, self.n_relevant[qi]))
        latency = self.embed_ms if cfg.mode != "keyword" else 0.0
        latency += (self.vec_ms if cfg.mode != "keyword" else 0) + (self.kw_ms if cfg.mode != "vector" else 0)
        if cfg.use_reranker and self.reranker is not None and n_cands:
            latency += (self.rerank_ms_per_pair or 0) * (sum(n_cands) / len(n_cands))
        return aggregate(rows), latency


def run_optimization(
    dataset_id: str,
    collection: str | None = None,
    chunk_sizes: list[int] | None = None,
    overlap_ratio: float = 0.15,
    modes: list[str] | None = None,
    fusions: list[str] | None = None,
    alphas: list[float] | None = None,
    candidate_ks: list[int] | None = None,
    rerank_options: list[bool] | None = None,
    metric: str = "ndcg@10",
    latency_weight: float = 0.0,
    max_trials: int = 80,
    tune_split: str = "train",
    seed: int = 7,
    ctx=None,
) -> dict:
    dataset = load_dataset(dataset_id)
    collection = collection or dataset["collection"]
    tune_items = [i for i in split_items(dataset, tune_split) if has_gold(i)]
    test_items = [i for i in split_items(dataset, "test") if has_gold(i)]
    if not tune_items:
        tune_items = test_items  # tiny / manual datasets
    if not tune_items:
        raise ValueError("Dataset has no labelled items")
    current = get_retrieval_store().get()
    trials = _trial_space(
        chunk_sizes or [current.chunk_size],
        overlap_ratio,
        modes or ["hybrid", "vector", "keyword"],
        fusions or ["rrf", "weighted"],
        alphas or [0.3, 0.5, 0.7],
        candidate_ks or [10, 20, 40],
        rerank_options if rerank_options is not None else [True, False],
    )
    if len(trials) > max_trials:
        random.Random(seed).shuffle(trials)
        trials = trials[:max_trials]
    trials.sort(key=lambda t: t["chunk_size"])

    reg = get_registry()
    emb_path, rr_path = reg.active_path("embedding"), reg.active_path("reranker")
    embedder = load_embedder(emb_path)
    reranker = load_reranker(rr_path) if any(t["use_reranker"] for t in trials) else None
    provider = IndexProvider(collection)
    max_k = max(t["candidate_k"] for t in trials)
    caches: dict[tuple[int, int], _QueryCache] = {}
    leaderboard = []
    for n, t in enumerate(trials):
        if ctx:
            ctx.check()
            ctx.progress(0.9 * n / len(trials), f"trial {n + 1}/{len(trials)} (chunk {t['chunk_size']}, {t['mode']})")
        key = (t["chunk_size"], t["chunk_overlap"])
        if key not in caches:
            index = provider.index(*key, emb_path)
            caches[key] = _QueryCache(index, tune_items, embedder, reranker, max_k)
        cfg = current.model_copy(update=t)
        metrics, latency = caches[key].evaluate(cfg)
        score = metrics.get(metric, 0.0) - latency_weight * latency / 1000
        leaderboard.append({**t, "objective": round(score, 4), metric: round(metrics.get(metric, 0.0), 4),
                            "hit@1": round(metrics["hit@1"], 4), "hit@5": round(metrics["hit@5"], 4),
                            "mrr": round(metrics["mrr"], 4), "est_latency_ms": round(latency, 1)})
    leaderboard.sort(key=lambda r: (-r["objective"], r["est_latency_ms"]))
    best = leaderboard[0]
    best_params = {k: best[k] for k in ("chunk_size", "chunk_overlap", "mode", "fusion", "alpha", "candidate_k", "use_reranker")}

    # confirm on held-out test split: current vs best
    confirmation = {}
    if test_items and test_items is not tune_items:
        if ctx:
            ctx.progress(0.93, "confirming on held-out test split")
        for label, params in (("current", current.model_dump()), ("best", best_params)):
            cfg = current.model_copy(update=params)
            key = (cfg.chunk_size, cfg.chunk_overlap)
            qc = _QueryCache(provider.index(*key, emb_path), test_items, embedder,
                             reranker or (load_reranker(rr_path) if cfg.use_reranker else None), max(max_k, cfg.candidate_k))
            m, lat = qc.evaluate(cfg)
            confirmation[label] = {metric: round(m.get(metric, 0), 4), "hit@5": round(m["hit@5"], 4),
                                   "mrr": round(m["mrr"], 4), "est_latency_ms": round(lat, 1)}

    run = {
        "id": new_id("ev_"),
        "kind": "optimization",
        "name": f"optimize {metric} · {len(trials)} trials",
        "dataset_id": dataset_id,
        "collection": collection,
        "config": {"metric": metric, "latency_weight": latency_weight, "tune_split": tune_split,
                   "embedding_model": emb_path, "reranker_model": rr_path, "current": current.model_dump()},
        "metrics": {"best": best_params, "best_objective": best["objective"], "best_" + metric: best[metric],
                    "trials": len(trials), "confirmation": confirmation,
                    "requires_reindex": (best_params["chunk_size"], best_params["chunk_overlap"]) != current.chunking_key()},
        "details": leaderboard,
        "created_at": now(),
    }
    get_db().insert("eval_runs", run)
    return run
