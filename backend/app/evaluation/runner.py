"""Evaluation runners: retrieval quality, end-to-end answer quality, model comparison."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np

from app.core.config import RetrievalConfig, get_retrieval_store
from app.core.db import get_db, new_id, now
from app.evaluation.metrics import (
    aggregate,
    cosine,
    faithfulness,
    has_gold,
    is_relevant,
    latency_stats,
    retrieval_metrics,
    token_f1,
)
from app.finetune.dataset import load_dataset, split_items
from app.generation.llm import ExtractiveLLM, get_llm, parse_json
from app.generation.prompts import JUDGE_PROMPT
from app.indexing.index import HybridIndex
from app.indexing.models import get_registry, load_embedder, load_reranker, release_unless_active
from app.ingestion.chunker import Chunk
from app.retrieval.retriever import retrieve
from app.services.knowledge import get_knowledge

log = logging.getLogger(__name__)
EVAL_DEPTH = 10


class IndexProvider:
    """Builds (and caches) indexes for arbitrary (chunking, embedder) combinations of a collection."""

    def __init__(self, collection: str):
        self.collection = collection
        self.kb = get_knowledge()
        self._chunks: dict[tuple[int, int], list[Chunk]] = {}
        self._indexes: dict[tuple, HybridIndex] = {}
        self._live_keys: set[tuple[int, int]] = set()

    def chunks(self, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        key = (chunk_size, chunk_overlap)
        if key not in self._chunks:
            current = get_retrieval_store().get()
            live = self.kb.get_index(self.collection)
            if key == current.chunking_key() and self.kb._chunking_of(self.collection) in (None, key):
                self._chunks[key] = list(live.chunks)
                self._live_keys.add(key)
            else:
                cfg = current.model_copy(update={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
                out: list[Chunk] = []
                for doc in self.kb.list_documents(self.collection):
                    if doc["status"] == "indexed":
                        out.extend(self.kb._chunk_document(doc, cfg)[0])
                self._chunks[key] = out
        return self._chunks[key]

    def index(self, chunk_size: int, chunk_overlap: int, embedder_path: str) -> HybridIndex:
        key = (chunk_size, chunk_overlap, embedder_path)
        if key not in self._indexes:
            chunks = self.chunks(chunk_size, chunk_overlap)
            live = self.kb.get_index(self.collection)
            embedder = load_embedder(embedder_path)
            if (chunk_size, chunk_overlap) in self._live_keys and live.embedder_name == embedder.name:
                self._indexes[key] = live  # reuse the production index, no re-embedding
            else:
                self._indexes[key] = HybridIndex.build(chunks, embedder)
        return self._indexes[key]


def _n_relevant(index: HybridIndex, item: dict) -> int:
    return sum(1 for c in index.chunks if c.doc_id == item.get("doc_id") and is_relevant(c, item)) if item.get("doc_id") else 1


def evaluate_retrieval_on_index(
    index: HybridIndex,
    items: list[dict],
    cfg: RetrievalConfig,
    embedder,
    reranker=None,
    ctx=None,
    progress_span: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict, list[dict]]:
    eval_cfg = cfg.model_copy(update={"top_k": max(EVAL_DEPTH, cfg.top_k)})
    rows, details, latencies = [], [], []
    by_cat: dict[str, list[dict]] = defaultdict(list)
    lo, hi = progress_span
    for i, item in enumerate(items):
        if ctx:
            ctx.check()
            ctx.progress(lo + (hi - lo) * i / max(len(items), 1), f"evaluating {i + 1}/{len(items)}")
        t0 = time.perf_counter()
        hits, _ = retrieve(index, item["question"], eval_cfg, embedder, reranker if cfg.use_reranker else None)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = [h.chunk for h in hits]
        m = retrieval_metrics(retrieved, item, _n_relevant(index, item))
        rows.append(m)
        by_cat[item.get("category", "other")].append(m)
        first = next((r + 1 for r, c in enumerate(retrieved) if is_relevant(c, item)), None)
        details.append(
            {"qid": item["qid"], "question": item["question"], "category": item.get("category"),
             "first_relevant_rank": first, "hit@5": m["hit@5"], "mrr": m["mrr"],
             "top_sources": [f"{c.metadata.get('source')} p{c.page}" for c in retrieved[:3]]}
        )
    metrics = {**aggregate(rows), **latency_stats(latencies), "num_queries": len(rows)}
    metrics["by_category"] = {cat: {k: round(v, 4) for k, v in aggregate(r).items() if k in ("hit@5", "mrr", "ndcg@10")} | {"n": len(r)} for cat, r in by_cat.items()}
    return metrics, details


def run_retrieval_eval(
    dataset_id: str,
    collection: str | None = None,
    split: str = "test",
    config_overrides: dict | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    name: str | None = None,
    ctx=None,
    provider: IndexProvider | None = None,
    save: bool = True,
) -> dict:
    dataset = load_dataset(dataset_id)
    collection = collection or dataset["collection"]
    items = [i for i in split_items(dataset, split) if has_gold(i)]
    if not items:
        raise ValueError(f"Dataset {dataset_id} has no labelled items in split '{split}'")
    cfg = get_retrieval_store().get().model_copy(update=config_overrides or {})
    reg = get_registry()
    emb_path = embedding_model or reg.active_path("embedding")
    rr_path = reranker_model or reg.active_path("reranker")
    provider = provider or IndexProvider(collection)
    if ctx:
        ctx.progress(0.02, "building evaluation index")
    index = provider.index(cfg.chunk_size, cfg.chunk_overlap, emb_path)
    reranker = load_reranker(rr_path) if cfg.use_reranker else None
    try:
        metrics, details = evaluate_retrieval_on_index(index, items, cfg, load_embedder(emb_path), reranker, ctx, (0.1, 1.0))
    finally:
        release_unless_active(emb_path, rr_path)
    run = {
        "id": new_id("ev_"),
        "kind": "retrieval",
        "name": name or f"retrieval · {emb_path.split('/')[-1]}",
        "dataset_id": dataset_id,
        "collection": collection,
        "config": {**cfg.model_dump(), "embedding_model": emb_path, "reranker_model": rr_path if cfg.use_reranker else None, "split": split},
        "metrics": metrics,
        "details": details,
        "created_at": now(),
    }
    if save:
        get_db().insert("eval_runs", run)
    return run


def _judge(llm, item: dict, answer: str, context: str) -> dict | None:
    try:
        raw = llm.chat(
            "You are a strict, fair evaluator. Output JSON only.",
            JUDGE_PROMPT.format(question=item["question"], reference=item.get("answer") or "(none)",
                                context=context[:4000], answer=answer),
            json_mode=True,
            max_tokens=200,
        )
        data = parse_json(raw)
        if isinstance(data, dict):
            return {k: float(data[k]) for k in ("faithfulness", "correctness", "completeness") if k in data}
    except Exception as exc:
        log.debug("judge failed: %s", exc)
    return None


def run_generation_eval(
    dataset_id: str,
    collection: str | None = None,
    split: str = "test",
    max_items: int = 30,
    use_llm_judge: bool = False,
    name: str | None = None,
    ctx=None,
) -> dict:
    from app.services.rag import get_rag

    dataset = load_dataset(dataset_id)
    collection = collection or dataset["collection"]
    items = split_items(dataset, split)
    ood = [i for i in items if i.get("category") == "out_of_domain"]
    ind = [i for i in items if i.get("category") != "out_of_domain"][: max(1, max_items - len(ood))]
    items = ind + ood
    rag = get_rag()
    embedder = load_embedder(get_registry().active_path("embedding"))
    judge = None
    if use_llm_judge:
        llm = get_llm()
        if not isinstance(llm, ExtractiveLLM) and llm.available():
            judge = llm
    rows, ood_rows, details, latencies = [], [], [], []
    for i, item in enumerate(items):
        if ctx:
            ctx.check()
            ctx.progress(i / max(len(items), 1), f"answering {i + 1}/{len(items)}")
        res = rag.ask(item["question"], [collection], use_cache=False, log_query=False)
        latencies.append(res["latency"].get("total_ms", 0))
        context = "\n".join(s["text"] for s in res["sources"])
        detail = {"qid": item["qid"], "question": item["question"], "category": item.get("category"),
                  "answer": res["answer"], "refused": res["refused"], "model": res["model"]}
        if item.get("category") == "out_of_domain":
            ood_rows.append({"refusal_accuracy": float(res["refused"])})
        else:
            q_vec, a_vec = embedder.encode([item["question"], res["answer"]])
            m = {
                "faithfulness": faithfulness(res["answer"], context) if not res["refused"] else 1.0,
                "answer_relevance": cosine(q_vec, a_vec) if not res["refused"] else 0.0,
                "citation_rate": float(bool(res["cited_refs"])),
                "false_refusal_rate": float(res["refused"]),
                "context_hit": float(any(is_relevant(_src_chunk(s), item) for s in res["sources"])) if has_gold(item) else None,
            }
            if item.get("answer"):
                m["answer_f1"] = token_f1(res["answer"], item["answer"])
            if judge:
                if j := _judge(judge, item, res["answer"], context):
                    m.update({f"judge_{k}": v for k, v in j.items()})
            m = {k: v for k, v in m.items() if v is not None}
            rows.append(m)
            detail.update({k: round(v, 3) for k, v in m.items()})
        details.append(detail)

    metrics = {**_mean_dicts(rows), **_mean_dicts(ood_rows), **latency_stats(latencies),
               "num_queries": len(items), "num_out_of_domain": len(ood_rows)}
    llm_row = get_registry().active("llm")
    run = {
        "id": new_id("ev_"),
        "kind": "generation",
        "name": name or f"generation · {llm_row['name']}",
        "dataset_id": dataset_id,
        "collection": collection,
        "config": {**get_retrieval_store().get().model_dump(), "llm": llm_row["path"], "llm_judge": bool(judge), "split": split},
        "metrics": metrics,
        "details": details,
        "created_at": now(),
    }
    get_db().insert("eval_runs", run)
    return run


def _src_chunk(src: dict) -> Chunk:
    """Rebuild a Chunk for relevance checks from a RAG source dict (offsets are looked up in the index)."""
    kb = get_knowledge()
    for c in kb.get_index(src["collection"]).chunks if src.get("collection") else []:
        if c.id == src["chunk_id"]:
            return c
    return Chunk(id=src["chunk_id"], doc_id=src["doc_id"], text=src["text"], page=src["page"], start=-1, end=-1,
                 chunk_index=-1, metadata={})


def _mean_dicts(rows: list[dict]) -> dict:
    keys = {k for r in rows for k in r}
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def compare_embedding_models(dataset_id: str, model_paths: list[str], collection: str | None = None,
                             split: str = "test", ctx=None, use_reranker: bool = False) -> dict:
    """Evaluate several embedding models on the same data (retrieval only)."""
    dataset = load_dataset(dataset_id)
    provider = IndexProvider(collection or dataset["collection"])
    results = {}
    for i, path in enumerate(model_paths):
        if ctx:
            ctx.progress(i / len(model_paths), f"evaluating {path}")
        run = run_retrieval_eval(dataset_id, collection, split, {"use_reranker": use_reranker, "mode": "vector"},
                                 embedding_model=path, provider=provider, save=False)
        results[path] = {k: v for k, v in run["metrics"].items() if k != "by_category"}
    return results


def list_runs(kind: str | None = None, limit: int = 50) -> list[dict]:
    db = get_db()
    cols = "id, kind, name, dataset_id, collection, config, metrics, created_at"
    if kind:
        return db.fetchall(f"SELECT {cols} FROM eval_runs WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit))
    return db.fetchall(f"SELECT {cols} FROM eval_runs ORDER BY created_at DESC LIMIT ?", (limit,))


def get_run(run_id: str) -> dict | None:
    return get_db().fetchone("SELECT * FROM eval_runs WHERE id=?", (run_id,))
