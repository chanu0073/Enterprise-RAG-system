"""Platform configuration.

Two layers:
- ``Settings``: static deployment settings read from environment variables (prefix ``EKIP_``).
- ``RetrievalConfig``: runtime-tunable retrieval parameters persisted to disk, so the
  optimizer (or an admin) can change them without a redeploy.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EKIP_", env_file=".env", extra="ignore")

    app_name: str = "Enterprise Knowledge Intelligence Platform"
    data_dir: Path = Path("storage")
    api_key: str | None = None  # when set, every request must send X-API-Key
    max_upload_mb: int = 50
    cors_origins: list[str] = ["*"]

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # or "hash" for offline/tests
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "auto"  # auto | cpu | cuda

    # Generation
    llm_backend: Literal["ollama", "hf", "extractive"] = "ollama"
    llm_model: str = "qwen2.5:3b"
    ollama_url: str = "http://localhost:11434"
    llm_timeout: float = 180.0
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    hf_base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # base model for LoRA adaptation

    # Performance
    query_cache_size: int = 256
    query_cache_ttl_s: int = 900
    warmup_models: bool = True

    @property
    def collections_dir(self) -> Path:
        return self.data_dir / "collections"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "platform.db"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.collections_dir, self.uploads_dir, self.datasets_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


class RetrievalConfig(BaseModel):
    """Retrieval parameters. Chunking changes require a re-index."""

    chunk_size: int = Field(800, ge=100, le=4000)
    chunk_overlap: int = Field(120, ge=0, le=1000)
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "rrf"
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Vector weight for weighted fusion")
    candidate_k: int = Field(20, ge=1, le=200, description="Candidates per retriever before rerank")
    top_k: int = Field(5, ge=1, le=50, description="Chunks passed to the LLM")
    use_reranker: bool = True
    identifier_boost: float = Field(
        0.5, ge=0.0, le=2.0, description="Score boost for chunks whose heading matches an ID in the query (e.g. 6.1.2)"
    )
    min_rerank_score: float | None = Field(
        None, description="Drop chunks whose rerank probability is below this (None = keep all)"
    )

    def chunking_key(self) -> tuple[int, int]:
        return (self.chunk_size, self.chunk_overlap)


class RetrievalConfigStore:
    """Thread-safe persisted RetrievalConfig."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._config = self._load()

    def _load(self) -> RetrievalConfig:
        if self.path.exists():
            try:
                return RetrievalConfig(**json.loads(self.path.read_text()))
            except Exception:  # corrupted file -> defaults
                pass
        return RetrievalConfig()

    def get(self) -> RetrievalConfig:
        with self._lock:
            return self._config.model_copy()

    def set(self, config: RetrievalConfig) -> RetrievalConfig:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(config.model_dump_json(indent=2))
            self._config = config.model_copy()
            return self._config.model_copy()


@lru_cache
def get_retrieval_store() -> RetrievalConfigStore:
    return RetrievalConfigStore(get_settings().data_dir / "retrieval_config.json")
