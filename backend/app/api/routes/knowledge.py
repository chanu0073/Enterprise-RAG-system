"""Collections, documents, search, question answering and feedback."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import AskRequest, CollectionCreate, FeedbackRequest, SearchRequest
from app.core.config import get_retrieval_store
from app.core.jobs import get_jobs
from app.services.knowledge import get_knowledge
from app.services.rag import get_rag, hit_to_source

router = APIRouter(tags=["knowledge"])


def _check_collections(names: list[str]) -> None:
    kb = get_knowledge()
    missing = [n for n in names if not kb.get_collection(n)]
    if missing:
        raise HTTPException(404, f"Unknown collection(s): {', '.join(missing)}")


# ------------------------------------------------------------------ collections
@router.get("/collections")
def list_collections():
    return get_knowledge().list_collections()


@router.post("/collections", status_code=201)
def create_collection(body: CollectionCreate):
    try:
        return get_knowledge().create_collection(body.name, body.description)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/collections/{name}")
def get_collection(name: str):
    kb = get_knowledge()
    col = kb.get_collection(name)
    if not col:
        raise HTTPException(404, "Collection not found")
    return {**col, "index": kb.index_status(name)}


@router.delete("/collections/{name}", status_code=204)
def delete_collection(name: str):
    if name == "default":
        raise HTTPException(400, "The default collection cannot be deleted")
    try:
        get_knowledge().delete_collection(name)
    except KeyError:
        raise HTTPException(404, "Collection not found")


@router.post("/collections/{name}/reindex", status_code=202)
def reindex_collection(name: str):
    _check_collections([name])
    kb = get_knowledge()
    return get_jobs().submit("reindex", lambda ctx: kb.reindex_collection(name, ctx.progress), {"collection": name})


# ------------------------------------------------------------------ documents
@router.post("/collections/{name}/documents", status_code=202)
async def upload_documents(name: str, files: list[UploadFile] = File(...), sync: bool = Form(False)):
    """Upload one or more files. Indexing runs as a background job unless ``sync=true``."""
    _check_collections([name])
    kb = get_knowledge()
    results = []
    for f in files:
        content = await f.read()
        try:
            doc = kb.save_upload(name, f.filename or "upload", content)
        except ValueError as exc:
            results.append({"filename": f.filename, "error": str(exc)})
            continue
        if doc.get("duplicate"):
            results.append({"filename": f.filename, "document": doc, "duplicate": True})
            continue
        if sync:
            try:
                results.append({"filename": f.filename, "document": kb.index_document(doc["id"])})
            except Exception as exc:
                results.append({"filename": f.filename, "document": kb.get_document(doc["id"]), "error": str(exc)})
        else:
            job = get_jobs().submit(
                "index_document", lambda ctx, d=doc["id"]: kb.index_document(d, ctx.progress) and {"doc_id": d},
                {"doc_id": doc["id"], "filename": f.filename},
            )
            results.append({"filename": f.filename, "document": doc, "job_id": job["id"]})
    return {"results": results}


@router.get("/documents")
def list_documents(collection: str | None = None):
    return get_knowledge().list_documents(collection)


@router.get("/documents/{doc_id}")
def get_document(doc_id: str):
    doc = get_knowledge().get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, limit: int = Query(50, le=500), offset: int = 0):
    try:
        return get_knowledge().document_chunks(doc_id, limit, offset)
    except KeyError:
        raise HTTPException(404, "Document not found")


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    try:
        get_knowledge().delete_document(doc_id)
        get_rag().cache.clear()
    except KeyError:
        raise HTTPException(404, "Document not found")


# ------------------------------------------------------------------ search & ask
def _cfg(body: SearchRequest):
    cfg = get_retrieval_store().get()
    return cfg.model_copy(update=body.overrides.as_dict()) if body.overrides else cfg


@router.post("/search")
def search(body: SearchRequest):
    _check_collections(body.collections)
    hits, timings = get_knowledge().search(body.query, body.collections, _cfg(body), body.doc_ids)
    return {"results": [hit_to_source(h, i) for i, h in enumerate(hits, start=1)],
            "latency": {k: round(v, 1) for k, v in timings.items()}}


@router.post("/ask")
def ask(body: AskRequest):
    _check_collections(body.collections)
    rag = get_rag()
    history = [t.model_dump() for t in body.history] if body.history else None
    if body.stream:
        return StreamingResponse(
            rag.ask_stream(body.query, body.collections, body.doc_ids, _cfg(body), history),
            media_type="text/event-stream",
        )
    return rag.ask(body.query, body.collections, body.doc_ids, _cfg(body), history, use_cache=body.use_cache)


@router.post("/feedback", status_code=204)
def feedback(body: FeedbackRequest):
    try:
        get_rag().feedback(body.query_id, body.rating, body.comment)
    except KeyError:
        raise HTTPException(404, "Query not found")
