"""Background job runner for long tasks (indexing, dataset generation, training, eval, optimization).

Jobs run in a single worker thread by default so GPU-heavy jobs never compete for memory;
status/progress are persisted in SQLite so the UI can poll them.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable

from app.core.db import get_db, new_id, now

log = logging.getLogger(__name__)


class JobContext:
    def __init__(self, job_id: str, manager: "JobManager"):
        self.job_id = job_id
        self._manager = manager
        self._last = 0.0

    def progress(self, fraction: float, message: str = "") -> None:
        # throttle DB writes
        t = time.time()
        if t - self._last < 0.5 and fraction < 1.0:
            return
        self._last = t
        self._manager.db.update(
            "jobs", "id", self.job_id, {"progress": max(0.0, min(1.0, fraction)), "message": message, "updated_at": t}
        )

    def is_cancelled(self) -> bool:
        return self.job_id in self._manager._cancelled

    def check(self) -> None:
        """Call inside loops; raises if the job was cancelled."""
        if self.is_cancelled():
            raise JobCancelled(self.job_id)


class NullContext:
    """Stand-in when running job functions synchronously (tests, CLI)."""

    def progress(self, fraction: float, message: str = "") -> None:
        log.info("progress %.0f%% %s", fraction * 100, message)

    def check(self) -> None:
        pass


class JobManager:
    def __init__(self, workers: int = 1):
        self.db = get_db()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ekip-job")
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        # jobs interrupted by a restart can never finish
        self.db.execute(
            "UPDATE jobs SET status='failed', error='Interrupted by server restart' WHERE status IN ('queued','running')"
        )

    def submit(self, job_type: str, fn: Callable[[JobContext], Any], params: dict | None = None) -> dict:
        job_id = new_id("job_")
        t = now()
        self.db.insert(
            "jobs",
            {"id": job_id, "type": job_type, "status": "queued", "progress": 0.0, "message": "queued",
             "params": params or {}, "created_at": t, "updated_at": t},
        )

        def run():
            ctx = JobContext(job_id, self)
            if ctx.is_cancelled():
                return
            self.db.update("jobs", "id", job_id, {"status": "running", "message": "started", "updated_at": now()})
            try:
                result = fn(ctx)
                self.db.update(
                    "jobs", "id", job_id,
                    {"status": "succeeded", "progress": 1.0, "message": "done", "result": result, "updated_at": now()},
                )
            except JobCancelled:
                self.db.update("jobs", "id", job_id, {"status": "cancelled", "message": "cancelled", "updated_at": now()})
            except Exception as exc:
                log.error("Job %s (%s) failed: %s", job_id, job_type, traceback.format_exc())
                self.db.update(
                    "jobs", "id", job_id, {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "updated_at": now()}
                )

        self.pool.submit(run)
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        return self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))

    def list(self, limit: int = 50, job_type: str | None = None) -> list[dict]:
        if job_type:
            return self.db.fetchall("SELECT * FROM jobs WHERE type=? ORDER BY created_at DESC LIMIT ?", (job_type, limit))
        return self.db.fetchall("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    def cancel(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job and job["status"] in ("queued", "running"):
            with self._lock:
                self._cancelled.add(job_id)
            if job["status"] == "queued":
                self.db.update("jobs", "id", job_id, {"status": "cancelled", "updated_at": now()})
        return self.get(job_id)

    def wait(self, job_id: str, timeout: float = 600) -> dict:
        """Block until a job finishes (used by tests and CLI)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get(job_id)
            if job["status"] in ("succeeded", "failed", "cancelled"):
                return job
            time.sleep(0.2)
        raise TimeoutError(job_id)


class JobCancelled(Exception):
    pass


@lru_cache
def get_jobs() -> JobManager:
    return JobManager()
