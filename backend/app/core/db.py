"""Lightweight SQLite metadata store (documents, jobs, model registry, eval runs, query log)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from app.core.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    num_pages INTEGER DEFAULT 0,
    num_chunks INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    params TEXT DEFAULT '{}',
    result TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,          -- embedding | reranker | llm
    name TEXT NOT NULL,
    path TEXT NOT NULL,          -- hub id or local path
    base_model TEXT,
    is_active INTEGER DEFAULT 0,
    metrics TEXT DEFAULT '{}',
    params TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,          -- retrieval | generation | comparison
    name TEXT,
    dataset_id TEXT,
    collection TEXT,
    config TEXT DEFAULT '{}',
    metrics TEXT DEFAULT '{}',
    details TEXT DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
    id TEXT PRIMARY KEY,
    collection TEXT,
    question TEXT,
    answer TEXT,
    num_sources INTEGER,
    top_score REAL,
    latency_ms REAL,
    retrieval_ms REAL,
    generation_ms REAL,
    cached INTEGER DEFAULT 0,
    feedback INTEGER,            -- 1 / -1 / NULL
    feedback_comment TEXT,
    sources TEXT DEFAULT '[]',
    created_at REAL NOT NULL
);
"""

JSON_FIELDS = {"metadata", "params", "result", "metrics", "config", "details", "sources"}


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        with self._write_lock:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        for key in JSON_FIELDS & out.keys():
            if isinstance(out[key], str):
                try:
                    out[key] = json.loads(out[key])
                except json.JSONDecodeError:
                    pass
        return out

    @staticmethod
    def _encode(values: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (json.dumps(v, default=str) if k in JSON_FIELDS and not isinstance(v, str) and v is not None else v)
            for k, v in values.items()
        }

    def insert(self, table: str, values: dict[str, Any]) -> None:
        values = self._encode(values)
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self.connect() as conn:
            conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))

    def update(self, table: str, key: str, key_value: Any, values: dict[str, Any]) -> None:
        if not values:
            return
        values = self._encode(values)
        sets = ", ".join(f"{k} = ?" for k in values)
        with self.connect() as conn:
            conn.execute(f"UPDATE {table} SET {sets} WHERE {key} = ?", (*values.values(), key_value))

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        return self._decode(self._conn().execute(sql, params).fetchone())

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [self._decode(r) for r in self._conn().execute(sql, params).fetchall()]


@lru_cache
def get_db() -> Database:
    return Database(get_settings().db_path)


def now() -> float:
    return time.time()
