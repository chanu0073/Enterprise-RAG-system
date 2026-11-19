from dataclasses import dataclass


@dataclass
class RAGConfig:

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k_vector: int = 5
    top_k_keyword: int = 5
    top_k_hybrid: int = 8

    # Reranking
    rerank_top_k: int = 3

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "qwen2.5:3b"

    # Storage
    index_path: str = "storage/faiss"