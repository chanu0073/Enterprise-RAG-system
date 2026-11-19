# Enterprise Knowledge Intelligence Platform

Turns an organisation's documents into a searchable, question-answerable knowledge system, and then
**adapts, evaluates and optimizes itself on that organisation's own content**.

- **Backend:** FastAPI (`backend/`)
- **UI:** Streamlit (`frontend/`)
- **Deployment:** Docker Compose (Ollama + backend + frontend, optional GPU)

## Capabilities

| Area | What it does |
|---|---|
| **Ingestion** | PDF, DOCX, Markdown, TXT, HTML, CSV · multi-file upload · SHA-256 de-duplication · background indexing jobs · per-department **collections** |
| **Chunking** | Offset-preserving recursive splitter · **heading-aware context** (clause/section headings such as `6.1.2 Risk assessment` are detected, carried across pages and prefixed to each chunk) |
| **Search** | FAISS dense + BM25 sparse · RRF or weighted fusion · cross-encoder reranking · **exact-identifier boosting** for clause/policy/control numbers · multi-collection and per-document filters |
| **Answers** | Grounded, numbered **citations** · refusal when context is insufficient · streaming (SSE) · chat history · confidence score · extractive fallback when the LLM is down |
| **Fine-tuning / model adaptation** | Synthetic QA dataset generation from your documents (LLM or offline heuristics, plus 👍 user feedback) · **embedding** fine-tuning (MNRL + mined hard negatives) · **reranker** fine-tuning · **LLM LoRA** adaptation for cited/abstaining answers · model registry with one-click activation |
| **Evaluation** | Retrieval: Hit@k, MRR, nDCG@k, span-Recall@k, per-category breakdown, latency p50/p95 · Generation: faithfulness, answer relevance, citation rate, refusal accuracy on out-of-domain questions, false-refusal rate, answer F1, optional LLM-as-judge · base vs fine-tuned comparison |
| **Optimization** | Search over chunk size, retrieval mode, fusion, weight, candidate depth and reranking, with a latency-aware objective · tuned on train split, confirmed on held-out test · one-click apply (auto re-index) · answer cache with index-version invalidation · fp16 inference on GPU · model warm-up |
| **Operations** | API-key auth · job queue with progress/cancel · query analytics, feedback and **knowledge-gap** report (low-confidence questions) · health checks |

## Architecture

```
               ┌──────────────── Streamlit UI (8501) ────────────────┐
               │ Ask · Knowledge Base · Fine-tuning · Eval · Optimize │
               └───────────────────────────┬──────────────────────────┘
                                           │ REST / SSE  (/api/v1)
┌──────────────────────────────── FastAPI backend (8000) ──────────────────────────────────┐
│ ingestion ─► heading-aware chunker ─► embedder ─► HybridIndex (FAISS + BM25, per collection)│
│                                                        │                                  │
│ query ─► dense + BM25 ─► fusion ─► id-boost ─► cross-encoder ─► prompt ─► LLM ─► cited answer
│                                                                                           │
│ dataset builder ─► trainers (embedding · reranker · LoRA) ─► model registry ─► activate   │
│ evaluator (retrieval · generation) · optimizer (config search) · job queue · SQLite meta  │
└───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                            │
                                   Ollama (qwen2.5:3b)  or  HF model + LoRA adapter
```

Storage (`EKIP_DATA_DIR`, default `storage/`): `platform.db` (SQLite metadata), `collections/<name>/index`
(chunks, embeddings), `uploads/`, `datasets/`, `models/` (fine-tuned weights), `retrieval_config.json`.

## Quick start (Docker)

```bash
cp .env.example .env            # optional
docker compose up -d --build    # CPU
# or, with NVIDIA Container Toolkit:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

The `ollama-pull` service downloads `qwen2.5:3b` on first start. Change it with `EKIP_LLM_MODEL`.

## Local development

```bash
python -m venv .venv && .venv/bin/python -m pip install torch   # CPU or CUDA build
make install                    # backend + frontend + test deps
make dev-backend                # http://localhost:8000
make dev-frontend               # http://localhost:8501
make test
```

Requires a local [Ollama](https://ollama.com) with `ollama pull qwen2.5:3b` (or set `EKIP_LLM_BACKEND=extractive`).
LLM LoRA training additionally needs `pip install -r requirements-finetune.txt`.

## The improvement loop

1. **Ingest:** create a collection and upload documents (Knowledge Base page).
2. **Build a dataset:** *Datasets & Fine-tuning → Generate*. Questions are written per chunk by the local LLM
   (or by heuristics offline), split by passage into train/test, and out-of-domain questions are added to test refusals.
   Hand-labelled sets can be imported as JSON (see `data/eval_questions.json`).
3. **Baseline:** *Evaluation → Retrieval* and *End-to-end answers*.
4. **Adapt:** fine-tune the embedding model and/or reranker. Each training job automatically evaluates base vs
   fine-tuned on the held-out split and stores the deltas in the model registry. Activate the better model;
   embedding activation re-embeds all collections in the background.
5. **Optimize:** *Optimization → Run* finds the best retrieval configuration for your data, then *Apply*.
6. **Operate:** watch latency, feedback and knowledge gaps on *System & Analytics*. 👍 answers flow into the next dataset.

### How relevance is judged

Every chunk stores `(doc_id, page, section, start, end)` character offsets. A retrieved chunk counts as relevant
when it overlaps the gold passage by ≥ 50 % of the shorter span. This keeps metrics valid across **different
chunk sizes and models**, which the chunking optimizer depends on.

## API overview (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET/POST/DELETE | `/collections`, `/collections/{name}` | manage collections |
| POST | `/collections/{name}/documents` | upload files (multipart `files`, optional `sync=true`) |
| POST | `/collections/{name}/reindex` | re-chunk + re-embed |
| GET/DELETE | `/documents`, `/documents/{id}`, `/documents/{id}/chunks` | inspect / remove documents |
| POST | `/search` | retrieval only |
| POST | `/ask` | grounded answer (`"stream": true` for SSE) |
| POST | `/feedback` | 👍 / 👎 on an answer |
| POST | `/datasets/generate`, `/datasets/manual` | build training/eval data |
| POST | `/finetune/embedding`, `/finetune/reranker`, `/finetune/llm` | training jobs |
| GET/POST | `/models`, `/models/{id}/activate` | model registry |
| POST | `/eval/retrieval`, `/eval/generation`, `/eval/compare` | evaluation jobs |
| POST | `/optimize`, `/optimize/{run_id}/apply` | configuration search |
| GET/PUT | `/config/retrieval` | runtime retrieval settings |
| GET | `/jobs`, `/jobs/{id}`, `/analytics/queries`, `/system/info`, `/health` | operations |

Long-running endpoints return a job (`202`). Poll `/jobs/{id}` for `progress`, `message` and `result`.

## Configuration

Environment variables (prefix `EKIP_`, see `backend/app/core/config.py` and `.env.example`):
`DATA_DIR`, `API_KEY`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `DEVICE`, `LLM_BACKEND` (`ollama`|`hf`|`extractive`),
`LLM_MODEL`, `OLLAMA_URL`, `HF_BASE_MODEL`, `MAX_UPLOAD_MB`, `QUERY_CACHE_SIZE`, `QUERY_CACHE_TTL_S`.

Retrieval parameters (chunking, mode, fusion, depth, reranking, identifier boost) are runtime settings, edited on the
Optimization page or via `PUT /config/retrieval`.

## Project layout

```
backend/app/
  core/          config, SQLite store, background jobs
  ingestion/     loaders (pdf/docx/md/txt/html/csv), heading-aware chunker
  indexing/      FAISS+BM25 hybrid index, model loading + registry
  retrieval/     candidate generation, fusion, id-boost, reranking
  generation/    prompts, Ollama / HF+LoRA / extractive backends
  services/      knowledge base service, RAG service (cache, streaming, logging)
  finetune/      dataset builder, embedding / reranker / LoRA trainers
  evaluation/    metrics, retrieval + generation runners, model comparison
  optimization/  configuration search
  api/           FastAPI routes + schemas
backend/tests/   offline test-suite (hashing embedder, extractive answers)
frontend/        Streamlit app (app.py + views/)
docker/          Dockerfiles · docker-compose*.yml at repo root
legacy/          original CLI prototype, kept for reference
```
