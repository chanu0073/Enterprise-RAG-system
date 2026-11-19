from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str
    description: str = ""


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RetrievalOverrides(BaseModel):
    mode: Literal["hybrid", "vector", "keyword"] | None = None
    fusion: Literal["rrf", "weighted"] | None = None
    alpha: float | None = Field(None, ge=0, le=1)
    candidate_k: int | None = Field(None, ge=1, le=200)
    top_k: int | None = Field(None, ge=1, le=50)
    use_reranker: bool | None = None
    min_rerank_score: float | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    collections: list[str] = ["default"]
    doc_ids: list[str] | None = None
    overrides: RetrievalOverrides | None = None


class AskRequest(SearchRequest):
    history: list[ChatTurn] | None = None
    stream: bool = False
    use_cache: bool = True


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal[1, -1]
    comment: str | None = None


class DatasetGenerateRequest(BaseModel):
    collection: str = "default"
    method: Literal["auto", "llm", "heuristic"] = "auto"
    max_chunks: int = Field(150, ge=5, le=5000)
    questions_per_chunk: int = Field(2, ge=1, le=5)
    test_ratio: float = Field(0.25, ge=0.05, le=0.9)
    include_feedback: bool = True
    include_out_of_domain: bool = True
    seed: int = 42
    name: str | None = None


class ManualDatasetRequest(BaseModel):
    collection: str = "default"
    name: str | None = None
    items: list[dict]


class EmbeddingTrainRequest(BaseModel):
    dataset_id: str
    base_model: str | None = None
    epochs: int = Field(2, ge=1, le=20)
    batch_size: int = Field(16, ge=2, le=256)
    learning_rate: float = Field(2e-5, gt=0, le=1e-2)
    hard_negatives: bool = True
    name: str | None = None


class RerankerTrainRequest(BaseModel):
    dataset_id: str
    base_model: str | None = None
    epochs: int = Field(2, ge=1, le=20)
    batch_size: int = Field(16, ge=2, le=256)
    learning_rate: float = Field(2e-5, gt=0, le=1e-2)
    negatives_per_query: int = Field(3, ge=1, le=10)
    name: str | None = None


class LLMTrainRequest(BaseModel):
    dataset_id: str
    base_model: str | None = None
    epochs: int = Field(2, ge=1, le=10)
    batch_size: int = Field(2, ge=1, le=32)
    grad_accum: int = Field(8, ge=1, le=64)
    learning_rate: float = Field(2e-4, gt=0, le=1e-2)
    lora_r: int = Field(16, ge=2, le=128)
    lora_alpha: int = Field(32, ge=2, le=256)
    refusal_ratio: float = Field(0.15, ge=0, le=0.5)
    name: str | None = None


class RetrievalEvalRequest(BaseModel):
    dataset_id: str
    collection: str | None = None
    split: Literal["train", "test", "all"] = "test"
    overrides: RetrievalOverrides | None = None
    embedding_model_id: str | None = None
    reranker_model_id: str | None = None
    name: str | None = None


class GenerationEvalRequest(BaseModel):
    dataset_id: str
    collection: str | None = None
    split: Literal["train", "test", "all"] = "test"
    max_items: int = Field(30, ge=1, le=500)
    use_llm_judge: bool = False
    name: str | None = None


class CompareRequest(BaseModel):
    dataset_id: str
    model_ids: list[str] = Field(..., min_length=1)
    split: Literal["train", "test", "all"] = "test"


class OptimizeRequest(BaseModel):
    dataset_id: str
    collection: str | None = None
    chunk_sizes: list[int] | None = None
    overlap_ratio: float = Field(0.15, ge=0, le=0.5)
    modes: list[Literal["hybrid", "vector", "keyword"]] | None = None
    fusions: list[Literal["rrf", "weighted"]] | None = None
    alphas: list[float] | None = None
    candidate_ks: list[int] | None = None
    rerank_options: list[bool] | None = None
    metric: Literal["ndcg@10", "mrr", "hit@1", "hit@3", "hit@5", "recall@5"] = "ndcg@10"
    latency_weight: float = Field(0.0, ge=0, le=10)
    max_trials: int = Field(80, ge=1, le=1000)
