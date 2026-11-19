"""FastAPI application entry point."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from app.api.routes import evaluation, knowledge, system, training
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ekip")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    expected = get_settings().api_key
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(401, "Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    from app.core.jobs import get_jobs
    from app.indexing.models import get_registry
    from app.services.knowledge import get_knowledge

    get_registry()
    get_jobs()
    kb = get_knowledge()
    if settings.warmup_models:
        # load models in the background so the first user query is fast without blocking startup
        threading.Thread(target=_safe_warmup, args=(kb,), daemon=True).start()
    log.info("%s ready (data dir: %s)", settings.app_name, settings.data_dir.resolve())
    yield


def _safe_warmup(kb) -> None:
    try:
        t = time.time()
        kb.warmup()
        log.info("Models warmed up in %.1fs", time.time() - t)
    except Exception as exc:
        log.warning("Warmup failed: %s", exc)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Turns an organisation's documents into a searchable, question-answerable knowledge system "
        "with domain fine-tuning, evaluation and optimization.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    deps = [Depends(require_api_key)]
    app.include_router(system.public_router, prefix="/api/v1")
    for r in (system.router, knowledge.router, training.router, evaluation.router):
        app.include_router(r, prefix="/api/v1", dependencies=deps)
    return app


app = create_app()
