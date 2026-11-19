"""Health, jobs and usage analytics."""

from __future__ import annotations

import platform

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.db import get_db, now
from app.core.jobs import get_jobs
from app.generation.llm import get_llm
from app.indexing.models import get_registry, resolve_device
from app.services.rag import get_rag

router = APIRouter(tags=["system"])
public_router = APIRouter(tags=["system"])  # unauthenticated: liveness probes


@public_router.get("/health")
def health():
    return {"status": "ok", "time": now()}


@router.get("/system/info")
def info():
    s = get_settings()
    reg = get_registry()
    llm = get_llm()
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(0),
                   "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}
    except ImportError:
        pass
    return {
        "app": s.app_name,
        "python": platform.python_version(),
        "device": resolve_device(),
        "gpu": gpu,
        "llm": {"name": llm.name, "available": llm.available()},
        "active_models": {k: reg.active(k)["name"] for k in reg.KINDS},
        "query_cache": get_rag().cache.stats(),
        "auth_enabled": bool(s.api_key),
    }


@router.get("/jobs")
def list_jobs(limit: int = 50, type: str | None = None):
    return get_jobs().list(limit, type)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = get_jobs().get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = get_jobs().cancel(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/analytics/queries")
def query_analytics(days: int = 30, limit: int = 100):
    db = get_db()
    since = now() - days * 86400
    summary = db.fetchone(
        "SELECT COUNT(*) AS total, AVG(latency_ms) AS avg_latency_ms, AVG(retrieval_ms) AS avg_retrieval_ms, "
        "AVG(generation_ms) AS avg_generation_ms, AVG(cached) AS cache_rate, "
        "SUM(CASE WHEN feedback=1 THEN 1 ELSE 0 END) AS positive, "
        "SUM(CASE WHEN feedback=-1 THEN 1 ELSE 0 END) AS negative, "
        "SUM(CASE WHEN num_sources=0 THEN 1 ELSE 0 END) AS no_result "
        "FROM query_log WHERE created_at >= ?",
        (since,),
    )
    daily = db.fetchall(
        "SELECT date(created_at, 'unixepoch') AS day, COUNT(*) AS queries, AVG(latency_ms) AS avg_latency_ms "
        "FROM query_log WHERE created_at >= ? GROUP BY day ORDER BY day",
        (since,),
    )
    recent = db.fetchall(
        "SELECT id, collection, question, num_sources, top_score, latency_ms, cached, feedback, created_at "
        "FROM query_log ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    low_confidence = db.fetchall(
        "SELECT question, top_score, created_at FROM query_log WHERE created_at >= ? AND (top_score IS NULL OR top_score < 0.3) "
        "ORDER BY created_at DESC LIMIT 20",
        (since,),
    )
    return {"summary": summary, "daily": daily, "recent": recent, "knowledge_gaps": low_confidence}
