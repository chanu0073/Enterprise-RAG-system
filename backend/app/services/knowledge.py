"""Knowledge base service: collections, document ingestion, indexing and search."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable

from app.core.config import RetrievalConfig, get_retrieval_store, get_settings
from app.core.db import get_db, new_id, now
from app.indexing.index import HybridIndex
from app.indexing.models import active_embedder, active_reranker, get_registry, load_embedder
from app.ingestion.chunker import Chunk, chunk_sections
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from app.retrieval.retriever import Hit, retrieve

log = logging.getLogger(__name__)

COLLECTION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
Progress = Callable[[float, str], None] | None


class KnowledgeService:
    def __init__(self):
        self.settings = get_settings()
        self.db = get_db()
        self._indexes: dict[str, HybridIndex] = {}
        self._lock = threading.RLock()
        self.version = 0  # bumped on every index mutation; part of the answer-cache key
        if not self.get_collection("default"):
            self.create_collection("default", "Default knowledge collection")

    # ------------------------------------------------------------------ collections
    def create_collection(self, name: str, description: str = "") -> dict:
        if not COLLECTION_RE.match(name):
            raise ValueError("Collection names may contain letters, digits, '-' and '_' (max 64 chars)")
        if self.get_collection(name):
            raise ValueError(f"Collection '{name}' already exists")
        self.db.insert("collections", {"name": name, "description": description, "created_at": now()})
        return self.get_collection(name)

    def get_collection(self, name: str) -> dict | None:
        row = self.db.fetchone("SELECT * FROM collections WHERE name=?", (name,))
        if row:
            stats = self.db.fetchone(
                "SELECT COUNT(*) AS documents, COALESCE(SUM(num_chunks),0) AS chunks FROM documents "
                "WHERE collection=? AND status='indexed'",
                (name,),
            )
            row.update(stats)
        return row

    def list_collections(self) -> list[dict]:
        names = [r["name"] for r in self.db.fetchall("SELECT name FROM collections ORDER BY created_at")]
        return [self.get_collection(n) for n in names]

    def delete_collection(self, name: str) -> None:
        if not self.get_collection(name):
            raise KeyError(name)
        with self._lock:
            for doc in self.list_documents(name):
                Path(doc["path"]).unlink(missing_ok=True)
            self.db.execute("DELETE FROM documents WHERE collection=?", (name,))
            self.db.execute("DELETE FROM collections WHERE name=?", (name,))
            self._indexes.pop(name, None)
            self.version += 1
            shutil.rmtree(self._index_dir(name), ignore_errors=True)

    def _index_dir(self, collection: str) -> Path:
        return self.settings.collections_dir / collection / "index"

    # ------------------------------------------------------------------ index access
    def get_index(self, collection: str) -> HybridIndex:
        with self._lock:
            if collection not in self._indexes:
                if not self.get_collection(collection):
                    raise KeyError(f"Unknown collection '{collection}'")
                embedder = active_embedder()
                index = HybridIndex.load(self._index_dir(collection))
                if index is None:
                    index = HybridIndex(embedder.name, embedder.get_dimension())
                elif index.embedder_name != embedder.name:
                    log.warning("Index for %s built with %s; re-embedding with %s", collection, index.embedder_name, embedder.name)
                    index = HybridIndex.build(index.chunks, embedder)
                    index.save(self._index_dir(collection))
                self._indexes[collection] = index
            return self._indexes[collection]

    def _bump(self) -> None:
        with self._lock:
            self.version += 1

    def invalidate(self) -> None:
        with self._lock:
            self._indexes.clear()
            self.version += 1

    def index_status(self, collection: str) -> dict:
        index = self.get_index(collection)
        cfg = get_retrieval_store().get()
        stale = self._chunking_of(collection) not in (None, cfg.chunking_key())
        return {"embedder": index.embedder_name, "num_chunks": len(index.chunks), "chunking_stale": stale}

    def _chunking_of(self, collection: str):
        row = self.db.fetchone(
            "SELECT metadata FROM documents WHERE collection=? AND status='indexed' LIMIT 1", (collection,)
        )
        if row and isinstance(row["metadata"], dict) and "chunk_size" in row["metadata"]:
            return (row["metadata"]["chunk_size"], row["metadata"]["chunk_overlap"])
        return None

    # ------------------------------------------------------------------ documents
    def list_documents(self, collection: str | None = None) -> list[dict]:
        if collection:
            return self.db.fetchall("SELECT * FROM documents WHERE collection=? ORDER BY created_at DESC", (collection,))
        return self.db.fetchall("SELECT * FROM documents ORDER BY created_at DESC")

    def get_document(self, doc_id: str) -> dict | None:
        return self.db.fetchone("SELECT * FROM documents WHERE id=?", (doc_id,))

    def save_upload(self, collection: str, filename: str, content: bytes) -> dict:
        """Persist the raw file and register it. Returns the document row (status 'pending' or 'duplicate')."""
        if not self.get_collection(collection):
            raise KeyError(f"Unknown collection '{collection}'")
        safe_name = Path(filename).name
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"File exceeds {self.settings.max_upload_mb} MB")
        sha = hashlib.sha256(content).hexdigest()
        dup = self.db.fetchone(
            "SELECT * FROM documents WHERE collection=? AND sha256=? AND status != 'failed'", (collection, sha)
        )
        if dup:
            return {**dup, "duplicate": True}
        doc_id = new_id("doc_")
        path = self.settings.uploads_dir / collection / f"{doc_id}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.db.insert(
            "documents",
            {
                "id": doc_id,
                "collection": collection,
                "filename": safe_name,
                "path": str(path),
                "sha256": sha,
                "size_bytes": len(content),
                "status": "pending",
                "metadata": {},
                "created_at": now(),
            },
        )
        return self.get_document(doc_id)

    def _chunk_document(self, doc: dict, cfg: RetrievalConfig) -> tuple[list[Chunk], int]:
        sections = load_document(doc["path"])
        chunks = chunk_sections(
            sections,
            doc_id=doc["id"],
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            base_metadata={"source": doc["filename"], "collection": doc["collection"]},
        )
        pages = len({s.page for s in sections if s.page is not None}) or len(sections)
        return chunks, pages

    def index_document(self, doc_id: str, progress: Progress = None) -> dict:
        doc = self.get_document(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        cfg = get_retrieval_store().get()
        self.db.update("documents", "id", doc_id, {"status": "indexing", "error": None})
        try:
            chunks, pages = self._chunk_document(doc, cfg)
            if not chunks:
                raise ValueError("No extractable text found (scanned PDF? try OCR first)")
            index = self.get_index(doc["collection"])
            embedder = active_embedder()
            with index.lock:
                index.remove_document(doc_id)
                index.add(chunks, embedder, progress=(lambda f: progress(f, "embedding")) if progress else None)
                index.save(self._index_dir(doc["collection"]))
            self._bump()
            self.db.update(
                "documents",
                "id",
                doc_id,
                {
                    "status": "indexed",
                    "num_chunks": len(chunks),
                    "num_pages": pages,
                    "metadata": {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap},
                },
            )
        except Exception as exc:
            log.exception("Indexing failed for %s", doc_id)
            self.db.update("documents", "id", doc_id, {"status": "failed", "error": str(exc)})
            raise
        return self.get_document(doc_id)

    def delete_document(self, doc_id: str) -> None:
        doc = self.get_document(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        index = self.get_index(doc["collection"])
        with index.lock:
            index.remove_document(doc_id)
            index.save(self._index_dir(doc["collection"]))
        self._bump()
        Path(doc["path"]).unlink(missing_ok=True)
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def document_chunks(self, doc_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        doc = self.get_document(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        chunks = [c for c in self.get_index(doc["collection"]).chunks if c.doc_id == doc_id]
        return [c.to_dict() for c in chunks[offset : offset + limit]]

    def reindex_collection(self, collection: str, progress: Progress = None) -> dict:
        """Re-chunk all documents with the current config and re-embed with the active model."""
        cfg = get_retrieval_store().get()
        embedder = active_embedder()
        docs = [d for d in self.list_documents(collection) if d["status"] in ("indexed", "failed", "pending")]
        all_chunks: list[Chunk] = []
        for i, doc in enumerate(docs):
            try:
                chunks, pages = self._chunk_document(doc, cfg)
                all_chunks.extend(chunks)
                self.db.update(
                    "documents",
                    "id",
                    doc["id"],
                    {
                        "status": "indexed" if chunks else "failed",
                        "error": None if chunks else "No extractable text",
                        "num_chunks": len(chunks),
                        "num_pages": pages,
                        "metadata": {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap},
                    },
                )
            except Exception as exc:
                self.db.update("documents", "id", doc["id"], {"status": "failed", "error": str(exc)})
            if progress:
                progress(0.3 * (i + 1) / max(len(docs), 1), f"chunked {i + 1}/{len(docs)}")
        index = HybridIndex.build(
            all_chunks, embedder, progress=(lambda f: progress(0.3 + 0.7 * f, "embedding")) if progress else None
        )
        index.save(self._index_dir(collection))
        with self._lock:
            self._indexes[collection] = index
            self.version += 1
        return {"collection": collection, "documents": len(docs), "chunks": len(all_chunks), "embedder": embedder.name}

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        collections: list[str],
        cfg: RetrievalConfig | None = None,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[Hit], dict[str, float]]:
        cfg = cfg or get_retrieval_store().get()
        embedder = active_embedder()
        reranker = active_reranker() if cfg.use_reranker else None
        all_hits: list[Hit] = []
        timings: dict[str, float] = {}
        # retrieve per collection without reranking, then rerank the merged pool once
        per_cfg = cfg.model_copy(update={"use_reranker": False, "top_k": cfg.candidate_k})
        for name in collections:
            hits, t = retrieve(self.get_index(name), query, per_cfg, embedder, None, set(doc_ids or []) or None, name)
            all_hits.extend(hits)
            timings["retrieval_ms"] = timings.get("retrieval_ms", 0.0) + t["retrieval_ms"]
        if len(collections) > 1:
            all_hits.sort(key=lambda h: -h.score)
        if reranker is not None and all_hits:
            import time

            from app.retrieval.retriever import rerank_hits

            t1 = time.perf_counter()
            all_hits = rerank_hits(query, all_hits, reranker)
            if cfg.min_rerank_score is not None:
                all_hits = [h for h in all_hits if (h.rerank_score or 0) >= cfg.min_rerank_score]
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
        return all_hits[: cfg.top_k], timings

    def all_chunks(self, collection: str) -> list[Chunk]:
        return list(self.get_index(collection).chunks)

    def warmup(self) -> None:
        active_embedder().encode(["warmup"])
        try:
            active_reranker().score("warmup", ["warmup"])
        except Exception as exc:  # reranker optional
            log.warning("Reranker warmup failed: %s", exc)


@lru_cache
def get_knowledge() -> KnowledgeService:
    return KnowledgeService()


def build_temp_index(chunks: list[Chunk], embedder_path: str) -> HybridIndex:
    """Index over given chunks with a specific (possibly non-active) embedder; used by eval/optimizer."""
    return HybridIndex.build(chunks, load_embedder(embedder_path))


__all__ = ["KnowledgeService", "get_knowledge", "build_temp_index", "get_registry"]
