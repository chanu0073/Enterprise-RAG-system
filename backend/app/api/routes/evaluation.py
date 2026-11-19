"""Evaluation, optimization and retrieval configuration."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import CompareRequest, GenerationEvalRequest, OptimizeRequest, RetrievalEvalRequest
from app.core.config import RetrievalConfig, get_retrieval_store
from app.core.jobs import get_jobs
from app.evaluation import runner
from app.finetune.dataset import load_dataset
from app.indexing.models import get_registry
from app.services.knowledge import get_knowledge
from app.services.rag import get_rag

router = APIRouter(tags=["evaluation"])


def _require_dataset(dataset_id: str) -> dict:
    try:
        return load_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, "Dataset not found")


def _model_path(model_id: str | None, kind: str) -> str | None:
    if not model_id:
        return None
    row = get_registry().get(model_id)
    if not row or row["kind"] != kind:
        raise HTTPException(404, f"{kind} model {model_id} not found")
    return row["path"]


@router.post("/eval/retrieval", status_code=202)
def eval_retrieval(body: RetrievalEvalRequest):
    _require_dataset(body.dataset_id)
    emb, rr = _model_path(body.embedding_model_id, "embedding"), _model_path(body.reranker_model_id, "reranker")
    overrides = body.overrides.as_dict() if body.overrides else None
    return get_jobs().submit(
        "eval_retrieval",
        lambda ctx: runner.run_retrieval_eval(body.dataset_id, body.collection, body.split, overrides, emb, rr, body.name, ctx),
        body.model_dump(),
    )


@router.post("/eval/generation", status_code=202)
def eval_generation(body: GenerationEvalRequest):
    _require_dataset(body.dataset_id)
    return get_jobs().submit(
        "eval_generation",
        lambda ctx: runner.run_generation_eval(body.dataset_id, body.collection, body.split, body.max_items,
                                               body.use_llm_judge, body.name, ctx),
        body.model_dump(),
    )


@router.post("/eval/compare", status_code=202)
def eval_compare(body: CompareRequest):
    _require_dataset(body.dataset_id)
    paths = [_model_path(mid, "embedding") for mid in body.model_ids]
    return get_jobs().submit(
        "eval_compare", lambda ctx: runner.compare_embedding_models(body.dataset_id, paths, split=body.split, ctx=ctx),
        body.model_dump(),
    )


@router.get("/eval/runs")
def list_runs(kind: str | None = None, limit: int = 50):
    return runner.list_runs(kind, limit)


@router.get("/eval/runs/{run_id}")
def get_run(run_id: str):
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


# ------------------------------------------------------------------ optimization
@router.post("/optimize", status_code=202)
def optimize(body: OptimizeRequest):
    _require_dataset(body.dataset_id)
    from app.optimization.tuner import run_optimization

    params = body.model_dump()
    return get_jobs().submit("optimize", lambda ctx: run_optimization(**params, ctx=ctx), params)


@router.post("/optimize/{run_id}/apply")
def apply_optimization(run_id: str):
    run = runner.get_run(run_id)
    if not run or run["kind"] != "optimization":
        raise HTTPException(404, "Optimization run not found")
    current = get_retrieval_store().get()
    return _set_config(current.model_copy(update=run["metrics"]["best"]))


# ------------------------------------------------------------------ retrieval config
@router.get("/config/retrieval")
def get_config():
    return get_retrieval_store().get()


@router.put("/config/retrieval")
def put_config(cfg: RetrievalConfig):
    return _set_config(cfg)


def _set_config(cfg: RetrievalConfig) -> dict:
    store = get_retrieval_store()
    previous = store.get()
    saved = store.set(cfg)
    get_rag().cache.clear()
    out: dict = {"config": saved, "reindex_jobs": []}
    if saved.chunking_key() != previous.chunking_key():
        kb = get_knowledge()
        for col in kb.list_collections():
            name = col["name"]
            job = get_jobs().submit("reindex", lambda ctx, n=name: kb.reindex_collection(n, ctx.progress), {"collection": name})
            out["reindex_jobs"].append(job["id"])
    return out
