"""Datasets, fine-tuning jobs and the model registry."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    DatasetGenerateRequest,
    EmbeddingTrainRequest,
    LLMTrainRequest,
    ManualDatasetRequest,
    RerankerTrainRequest,
)
from app.core.jobs import get_jobs
from app.finetune import dataset as ds
from app.indexing.models import evict, get_registry
from app.services.knowledge import get_knowledge
from app.services.rag import get_rag

router = APIRouter(tags=["training"])


# ------------------------------------------------------------------ datasets
@router.get("/datasets")
def list_datasets():
    return ds.list_datasets()


@router.post("/datasets/generate", status_code=202)
def generate_dataset(body: DatasetGenerateRequest):
    kb = get_knowledge()
    if not kb.get_collection(body.collection):
        raise HTTPException(404, "Collection not found")

    def run(ctx):
        return ds.generate_dataset(
            body.collection, kb.all_chunks(body.collection), body.method, body.max_chunks, body.questions_per_chunk,
            body.test_ratio, body.include_feedback, body.include_out_of_domain, body.seed, body.name, ctx,
        )

    return get_jobs().submit("generate_dataset", run, body.model_dump())


@router.post("/datasets/manual", status_code=201)
def import_manual(body: ManualDatasetRequest):
    kb = get_knowledge()
    if not kb.get_collection(body.collection):
        raise HTTPException(404, "Collection not found")
    return ds.import_manual_dataset(body.collection, body.items, kb.list_documents(body.collection), body.name)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, limit: int = 100, offset: int = 0, split: str = "all"):
    try:
        d = ds.load_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, "Dataset not found")
    items = ds.split_items(d, split)
    return {**{k: v for k, v in d.items() if k != "items"}, "items": items[offset : offset + limit], "total": len(items)}


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    try:
        ds.delete_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, "Dataset not found")


def _require_dataset(dataset_id: str) -> None:
    try:
        ds.load_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, "Dataset not found")


# ------------------------------------------------------------------ fine-tuning
@router.post("/finetune/embedding", status_code=202)
def finetune_embedding(body: EmbeddingTrainRequest):
    _require_dataset(body.dataset_id)
    from app.finetune.embedding_trainer import train_embedding_model

    params = body.model_dump()
    return get_jobs().submit("finetune_embedding", lambda ctx: train_embedding_model(**params, ctx=ctx), params)


@router.post("/finetune/reranker", status_code=202)
def finetune_reranker(body: RerankerTrainRequest):
    _require_dataset(body.dataset_id)
    from app.finetune.reranker_trainer import train_reranker

    params = body.model_dump()
    return get_jobs().submit("finetune_reranker", lambda ctx: train_reranker(**params, ctx=ctx), params)


@router.post("/finetune/llm", status_code=202)
def finetune_llm(body: LLMTrainRequest):
    _require_dataset(body.dataset_id)
    from app.finetune.llm_lora import train_llm_lora

    params = body.model_dump()
    return get_jobs().submit("finetune_llm", lambda ctx: train_llm_lora(**params, ctx=ctx), params)


# ------------------------------------------------------------------ model registry
@router.get("/models")
def list_models(kind: str | None = None):
    return get_registry().list(kind)


@router.get("/models/active")
def active_models():
    reg = get_registry()
    return {kind: reg.active(kind) for kind in reg.KINDS}


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str):
    reg = get_registry()
    try:
        previous = reg.active(reg.get(model_id)["kind"]) if reg.get(model_id) else None
        row = reg.activate(model_id)
    except KeyError:
        raise HTTPException(404, "Model not found")
    get_rag().cache.clear()
    result = {"model": row}
    if previous and previous["id"] != model_id:
        evict(previous["path"])
    if row["kind"] == "embedding":
        kb = get_knowledge()

        def reembed(ctx):
            kb.invalidate()
            out = {}
            cols = kb.list_collections()
            for i, col in enumerate(cols):
                ctx.progress(i / max(len(cols), 1), f"re-embedding {col['name']}")
                out[col["name"]] = len(kb.get_index(col["name"]).chunks)
            return {"reembedded": out}

        result["job"] = get_jobs().submit("reembed", reembed, {"model_id": model_id})
    elif row["kind"] == "llm":
        from app.generation import llm as llm_mod

        llm_mod._llm_cache.clear()
    return result
