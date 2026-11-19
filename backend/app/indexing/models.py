"""Model loading, caching and the model registry (base + fine-tuned models)."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from functools import lru_cache

import numpy as np

from app.core.config import get_settings
from app.core.db import get_db, new_id, now

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def resolve_device() -> str:
    device = get_settings().device
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class HashingEmbedder:
    """Dependency-free embedder (feature hashing of uni/bi-grams). Used for tests/offline mode."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        tokens = _TOKEN_RE.findall(text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str], batch_size: int = 64, **_) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def get_dimension(self) -> int:
        return self.dim


class STEmbedder:
    def __init__(self, path: str):
        from sentence_transformers import SentenceTransformer

        self.name = path
        self.model = SentenceTransformer(path, device=resolve_device())
        if self.model.device.type == "cuda":
            self.model.half()  # inference optimization: fp16 on GPU

    def encode(self, texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return emb.astype(np.float32)

    def get_dimension(self) -> int:
        fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(fn())


class Reranker:
    def __init__(self, path: str):
        from sentence_transformers import CrossEncoder

        self.name = path
        self.model = CrossEncoder(path, device=resolve_device())

    def score(self, query: str, passages: list[str], batch_size: int = 32) -> np.ndarray:
        """Return relevance probabilities in [0, 1]."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        logits = np.asarray(
            self.model.predict([(query, p) for p in passages], batch_size=batch_size, activation_fn=_identity),
            dtype=np.float32,
        ).reshape(len(passages), -1)[:, -1]
        return 1.0 / (1.0 + np.exp(-logits))


def _identity(x):
    return x


_cache: dict[tuple[str, str], object] = {}
_cache_lock = threading.Lock()


def load_embedder(path: str):
    key = ("embedding", path)
    with _cache_lock:
        if key not in _cache:
            log.info("Loading embedding model %s", path)
            _cache[key] = HashingEmbedder() if path.startswith("hash") else STEmbedder(path)
        return _cache[key]


def load_reranker(path: str) -> Reranker:
    key = ("reranker", path)
    with _cache_lock:
        if key not in _cache:
            log.info("Loading reranker %s", path)
            _cache[key] = Reranker(path)
        return _cache[key]


def evict(path: str) -> None:
    with _cache_lock:
        for key in [k for k in _cache if k[1] == path]:
            del _cache[key]
    _free_gpu()


def _free_gpu() -> None:
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def release_unless_active(*paths: str | None) -> None:
    """Drop models loaded only for evaluation/comparison so they don't pin (GPU) memory."""
    active = {get_registry().active_path(k) for k in ("embedding", "reranker")}
    for path in paths:
        if path and path not in active:
            evict(path)


# --------------------------------------------------------------------------- registry


class ModelRegistry:
    KINDS = ("embedding", "reranker", "llm")

    def __init__(self):
        self.db = get_db()
        self._ensure_base_models()

    def _ensure_base_models(self) -> None:
        s = get_settings()
        bases = {
            "embedding": s.embedding_model,
            "reranker": s.reranker_model,
            "llm": f"hf:{s.hf_base_model}" if s.llm_backend == "hf" else f"{s.llm_backend}:{s.llm_model}",
        }
        for kind, path in bases.items():
            existing = self.db.fetchone("SELECT * FROM models WHERE kind=? AND path=? AND base_model IS NULL", (kind, path))
            if existing is None:
                has_active = self.db.fetchone("SELECT id FROM models WHERE kind=? AND is_active=1", (kind,))
                self.db.insert(
                    "models",
                    {
                        "id": new_id(f"{kind[:3]}_"),
                        "kind": kind,
                        "name": f"{path} (base)",
                        "path": path,
                        "base_model": None,
                        "is_active": 0 if has_active else 1,
                        "metrics": {},
                        "params": {"backend": s.llm_backend} if kind == "llm" else {},
                        "created_at": now(),
                    },
                )

    def list(self, kind: str | None = None) -> list[dict]:
        if kind:
            return self.db.fetchall("SELECT * FROM models WHERE kind=? ORDER BY created_at DESC", (kind,))
        return self.db.fetchall("SELECT * FROM models ORDER BY kind, created_at DESC")

    def get(self, model_id: str) -> dict | None:
        return self.db.fetchone("SELECT * FROM models WHERE id=?", (model_id,))

    def active(self, kind: str) -> dict:
        row = self.db.fetchone("SELECT * FROM models WHERE kind=? AND is_active=1", (kind,))
        if row is None:  # registry was edited manually; fall back to newest base
            row = self.db.fetchone("SELECT * FROM models WHERE kind=? ORDER BY created_at LIMIT 1", (kind,))
        return row

    def active_path(self, kind: str) -> str:
        return self.active(kind)["path"]

    def register(self, kind: str, name: str, path: str, base_model: str, metrics: dict, params: dict) -> dict:
        model_id = new_id(f"{kind[:3]}_")
        self.db.insert(
            "models",
            {
                "id": model_id,
                "kind": kind,
                "name": name,
                "path": path,
                "base_model": base_model,
                "is_active": 0,
                "metrics": metrics,
                "params": params,
                "created_at": now(),
            },
        )
        return self.get(model_id)

    def activate(self, model_id: str) -> dict:
        row = self.get(model_id)
        if row is None:
            raise KeyError(model_id)
        self.db.execute("UPDATE models SET is_active=0 WHERE kind=?", (row["kind"],))
        self.db.execute("UPDATE models SET is_active=1 WHERE id=?", (model_id,))
        return self.get(model_id)

    def update_metrics(self, model_id: str, metrics: dict) -> None:
        row = self.get(model_id)
        merged = {**(row.get("metrics") or {}), **metrics}
        self.db.update("models", "id", model_id, {"metrics": merged})


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry()


def active_embedder():
    return load_embedder(get_registry().active_path("embedding"))


def active_reranker() -> Reranker:
    return load_reranker(get_registry().active_path("reranker"))
