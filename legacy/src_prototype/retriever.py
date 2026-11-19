from config import RAGConfig
from hybrid_retriever import HybridRetriever
from reranker import Reranker


class RAGRetriever:

    def __init__(
        self,
        vector_store,
        documents,
        config: RAGConfig
    ):

        self.config = config

        self.hybrid = HybridRetriever(
            vector_store=vector_store,
            documents=documents,
            vector_k=config.top_k_vector,
            keyword_k=config.top_k_keyword
        )

        self.reranker = Reranker(
            config.reranker_model
        )

    def retrieve(
        self,
        query,
        source_filter=None
    ):

        documents = self.hybrid.retrieve(
            query
        )

        # Metadata-aware filtering
        if source_filter:

            documents = [
                document
                for document in documents
                if document.metadata.get("source")
                == source_filter
            ]

        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=self.config.rerank_top_k
        )

        return reranked