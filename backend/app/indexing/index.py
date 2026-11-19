"""Per-collection hybrid index: FAISS (dense) + BM25 (sparse) over the same chunk list."""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.ingestion.chunker import Chunk

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who whom how why when where does do did shall should can could may might must".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# structured identifiers: clause/control/policy numbers such as 6.1.2, A.5.7, 8.22
IDENTIFIER_RE = re.compile(r"(?<![\w.])((?:[A-Za-z]\.)?\d{1,3}(?:\.\d{1,3})+)(?![\w.]*\d)")


def extract_identifiers(text: str) -> list[str]:
    return [m.upper() for m in IDENTIFIER_RE.findall(text)]


def _heading_identifier(heading: str | None) -> str | None:
    if not heading:
        return None
    first = heading.split()[0].rstrip(".")
    return first.upper() if IDENTIFIER_RE.fullmatch(first) or re.fullmatch(r"\d{1,3}", first) else None


class HybridIndex:
    """In-memory hybrid index with optional persistence. Not thread-safe by itself; callers lock."""

    def __init__(self, embedder_name: str, dim: int):
        self.embedder_name = embedder_name
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.faiss = faiss.IndexFlatIP(dim)
        self._bm25: BM25Okapi | None = None
        self._ids: dict[str, list[int]] | None = None
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ building
    @classmethod
    def build(cls, chunks: list[Chunk], embedder, batch_size: int = 64, progress=None) -> "HybridIndex":
        index = cls(embedder.name, embedder.get_dimension())
        index.add(chunks, embedder, batch_size=batch_size, progress=progress)
        return index

    def add(self, chunks: list[Chunk], embedder, batch_size: int = 64, progress=None) -> None:
        if not chunks:
            return
        parts = []
        step = max(batch_size * 4, 1)
        for i in range(0, len(chunks), step):
            parts.append(embedder.encode([c.embed_text for c in chunks[i : i + step]], batch_size=batch_size))
            if progress:
                progress(min(i + step, len(chunks)) / len(chunks))
        vectors = np.vstack(parts).astype(np.float32)
        self.add_with_embeddings(chunks, vectors)

    def add_with_embeddings(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.lock:
            self.chunks.extend(chunks)
            self.embeddings = np.vstack([self.embeddings, vectors]) if len(self.embeddings) else vectors
            self.faiss.add(vectors)
            self._bm25 = None
            self._ids = None

    def remove_document(self, doc_id: str) -> int:
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            removed = len(self.chunks) - len(keep)
            if removed:
                self.chunks = [self.chunks[i] for i in keep]
                self.embeddings = self.embeddings[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
                self.faiss = faiss.IndexFlatIP(self.dim)
                if len(self.embeddings):
                    self.faiss.add(self.embeddings)
                self._bm25 = None
                self._ids = None
            self._ids = None
            return removed

    # ------------------------------------------------------------------ search
    @property
    def bm25(self) -> BM25Okapi | None:
        if self._bm25 is None and self.chunks:
            self._bm25 = BM25Okapi([tokenize(c.embed_text) or ["_"] for c in self.chunks])
        return self._bm25

    def identifier_lookup(self, identifiers: list[str], doc_ids: set[str] | None = None) -> list[int]:
        """Chunks whose heading is exactly one of the identifiers (A.5.7 also matches a heading '5.7')."""
        if self._ids is None:
            ids: dict[str, list[int]] = {}
            for i, c in enumerate(self.chunks):
                if key := _heading_identifier(c.metadata.get("heading")):
                    ids.setdefault(key, []).append(i)
            self._ids = ids
        out: list[int] = []
        for ident in identifiers:
            for key in {ident, re.sub(r"^[A-Z]\.", "", ident)}:
                out.extend(i for i in self._ids.get(key, []) if not doc_ids or self.chunks[i].doc_id in doc_ids)
        return list(dict.fromkeys(out))

    def _allowed(self, doc_ids: set[str] | None) -> np.ndarray | None:
        if not doc_ids:
            return None
        return np.array([c.doc_id in doc_ids for c in self.chunks], dtype=bool)

    def vector_search(self, query_vec: np.ndarray, k: int, doc_ids: set[str] | None = None) -> list[tuple[int, float]]:
        n = len(self.chunks)
        if n == 0:
            return []
        allowed = self._allowed(doc_ids)
        fetch = n if allowed is not None else min(k, n)
        scores, ids = self.faiss.search(query_vec.reshape(1, -1).astype(np.float32), fetch)
        out = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0 or (allowed is not None and not allowed[idx]):
                continue
            out.append((int(idx), float(score)))
            if len(out) >= k:
                break
        return out

    def keyword_search(self, query: str, k: int, doc_ids: set[str] | None = None) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokens)
        allowed = self._allowed(doc_ids)
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        with self.lock:
            directory.mkdir(parents=True, exist_ok=True)
            tmp = directory / "chunks.jsonl.tmp"
            with tmp.open("w", encoding="utf-8") as f:
                for c in self.chunks:
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(directory / "chunks.jsonl")
            np.save(directory / "embeddings.npy", self.embeddings)
            (directory / "meta.json").write_text(
                json.dumps({"embedder": self.embedder_name, "dim": self.dim, "num_chunks": len(self.chunks)})
            )

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex | None":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        index = cls(meta["embedder"], meta["dim"])
        chunks_path = directory / "chunks.jsonl"
        if chunks_path.exists():
            with chunks_path.open(encoding="utf-8") as f:
                index.chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        emb_path = directory / "embeddings.npy"
        if emb_path.exists() and index.chunks:
            index.embeddings = np.load(emb_path).astype(np.float32)
            index.faiss.add(index.embeddings)
        return index
