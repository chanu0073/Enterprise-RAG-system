"""Question answering over the knowledge base: retrieve -> ground -> generate -> cite -> log."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Iterator

from app.core.config import RetrievalConfig, get_retrieval_store, get_settings
from app.core.db import get_db, new_id, now
from app.generation.llm import ExtractiveLLM, get_extractive, get_llm
from app.generation.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import Hit
from app.services.knowledge import get_knowledge

log = logging.getLogger(__name__)
_CITE_RE = re.compile(r"\[(\d+)\]")


class TTLCache:
    """Small LRU+TTL cache for repeated questions (a cheap but effective latency optimization)."""

    def __init__(self, size: int, ttl: float):
        self.size, self.ttl = size, ttl
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> dict | None:
        with self._lock:
            item = self._data.get(key)
            if item and time.time() - item[0] < self.ttl:
                self._data.move_to_end(key)
                self.hits += 1
                return item[1]
            self._data.pop(key, None)
            self.misses += 1
            return None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"size": len(self._data), "hits": self.hits, "misses": self.misses, "hit_rate": self.hits / total if total else 0.0}


def hit_to_source(h: Hit, n: int) -> dict:
    c = h.chunk
    return {
        "ref": n,
        "chunk_id": c.id,
        "doc_id": c.doc_id,
        "source": c.metadata.get("source"),
        "collection": h.collection or c.metadata.get("collection"),
        "page": c.page,
        "section": c.metadata.get("section"),
        "text": c.text,
        "score": round(h.score, 4),
        "vector_score": None if h.vector_score is None else round(h.vector_score, 4),
        "keyword_score": None if h.keyword_score is None else round(h.keyword_score, 4),
        "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 4),
    }


class RAGService:
    def __init__(self):
        s = get_settings()
        self.kb = get_knowledge()
        self.db = get_db()
        self.cache = TTLCache(s.query_cache_size, s.query_cache_ttl_s)

    def _cache_key(self, question: str, collections: list[str], doc_ids, cfg: RetrievalConfig, history) -> str:
        from app.indexing.models import get_registry

        reg = get_registry()
        raw = json.dumps(
            [
                question.strip().lower(),
                sorted(collections),
                sorted(doc_ids or []),
                cfg.model_dump(),
                [reg.active_path(k) for k in reg.KINDS],
                self.kb.version,
                history or [],
            ],
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def retrieve(self, question, collections, doc_ids=None, cfg=None):
        cfg = cfg or get_retrieval_store().get()
        hits, timings = self.kb.search(question, collections, cfg, doc_ids)
        return hits, timings, cfg

    @staticmethod
    def confidence(sources: list[dict]) -> float:
        if not sources:
            return 0.0
        top = sources[0]
        if top.get("rerank_score") is not None:
            return float(top["rerank_score"])
        return float(min(1.0, max(0.0, top.get("vector_score") or top["score"])))

    def _generate(self, question: str, sources: list[dict], history) -> tuple[str, str]:
        llm = get_llm()
        if isinstance(llm, ExtractiveLLM):
            return llm.answer(question, sources), llm.name
        try:
            return llm.chat(SYSTEM_PROMPT, build_user_prompt(question, sources, history)).strip(), llm.name
        except Exception as exc:
            log.warning("LLM %s failed (%s); falling back to extractive answer", llm.name, exc)
            return get_extractive().answer(question, sources), "extractive (fallback)"

    def ask(
        self,
        question: str,
        collections: list[str],
        doc_ids: list[str] | None = None,
        cfg: RetrievalConfig | None = None,
        history: list[dict] | None = None,
        use_cache: bool = True,
        log_query: bool = True,
    ) -> dict:
        cfg = cfg or get_retrieval_store().get()
        key = self._cache_key(question, collections, doc_ids, cfg, history)
        if use_cache and (cached := self.cache.get(key)):
            result = {**cached, "cached": True, "latency": {**cached["latency"], "total_ms": 0.0}}
            if log_query:
                result["query_id"] = self._log(question, collections, result)
            return result

        t0 = time.perf_counter()
        hits, timings, cfg = self.retrieve(question, collections, doc_ids, cfg)
        sources = [hit_to_source(h, i) for i, h in enumerate(hits, start=1)]
        t1 = time.perf_counter()
        if not sources:
            answer, model = REFUSAL, "none"
        else:
            answer, model = self._generate(question, sources, history)
        timings["generation_ms"] = (time.perf_counter() - t1) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        cited = sorted({int(n) for n in _CITE_RE.findall(answer) if 0 < int(n) <= len(sources)})
        result = {
            "answer": answer,
            "sources": sources,
            "cited_refs": cited,
            "confidence": round(self.confidence(sources), 4),
            "refused": answer.strip().startswith(REFUSAL[:30]),
            "model": model,
            "latency": {k: round(v, 1) for k, v in timings.items()},
            "cached": False,
        }
        if use_cache:
            self.cache.put(key, result)
        if log_query:
            result["query_id"] = self._log(question, collections, result)
        return result

    def ask_stream(
        self,
        question: str,
        collections: list[str],
        doc_ids: list[str] | None = None,
        cfg: RetrievalConfig | None = None,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Server-sent-event style generator: a `sources` event, `token` events, then `done`."""
        t0 = time.perf_counter()
        hits, timings, cfg = self.retrieve(question, collections, doc_ids, cfg)
        sources = [hit_to_source(h, i) for i, h in enumerate(hits, start=1)]
        yield _event("sources", {"sources": sources, "confidence": self.confidence(sources)})
        t1 = time.perf_counter()
        parts: list[str] = []
        model = "none"
        if not sources:
            parts.append(REFUSAL)
            yield _event("token", {"text": REFUSAL})
        else:
            llm = get_llm()
            model = llm.name
            try:
                if isinstance(llm, ExtractiveLLM):
                    raise RuntimeError("extractive")
                for piece in llm.stream(SYSTEM_PROMPT, build_user_prompt(question, sources, history)):
                    parts.append(piece)
                    yield _event("token", {"text": piece})
            except Exception as exc:
                if parts:  # partial generation then failure: report, keep what we have
                    yield _event("error", {"message": str(exc)})
                else:
                    text = get_extractive().answer(question, sources)
                    model = "extractive (fallback)" if not isinstance(llm, ExtractiveLLM) else "extractive"
                    parts.append(text)
                    yield _event("token", {"text": text})
        answer = "".join(parts).strip()
        timings["generation_ms"] = (time.perf_counter() - t1) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        result = {
            "answer": answer,
            "sources": sources,
            "cited_refs": sorted({int(n) for n in _CITE_RE.findall(answer) if 0 < int(n) <= len(sources)}),
            "confidence": round(self.confidence(sources), 4),
            "refused": answer.startswith(REFUSAL[:30]),
            "model": model,
            "latency": {k: round(v, 1) for k, v in timings.items()},
            "cached": False,
        }
        result["query_id"] = self._log(question, collections, result)
        yield _event("done", {k: v for k, v in result.items() if k != "sources"})

    def _log(self, question: str, collections: list[str], result: dict) -> str:
        qid = new_id("q_")
        lat = result.get("latency", {})
        try:
            self.db.insert(
                "query_log",
                {
                    "id": qid,
                    "collection": ",".join(collections),
                    "question": question,
                    "answer": result["answer"],
                    "num_sources": len(result["sources"]),
                    "top_score": result["sources"][0]["score"] if result["sources"] else None,
                    "latency_ms": lat.get("total_ms"),
                    "retrieval_ms": lat.get("retrieval_ms", 0) + lat.get("rerank_ms", 0),
                    "generation_ms": lat.get("generation_ms"),
                    "cached": int(result.get("cached", False)),
                    "sources": [{"chunk_id": s["chunk_id"], "doc_id": s["doc_id"], "score": s["score"]} for s in result["sources"]],
                    "created_at": now(),
                },
            )
        except Exception:
            log.exception("Failed to log query")
        return qid

    def feedback(self, query_id: str, rating: int, comment: str | None = None) -> None:
        if not self.db.fetchone("SELECT id FROM query_log WHERE id=?", (query_id,)):
            raise KeyError(query_id)
        self.db.update("query_log", "id", query_id, {"feedback": 1 if rating > 0 else -1, "feedback_comment": comment})


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"


@lru_cache
def get_rag() -> RAGService:
    return RAGService()
