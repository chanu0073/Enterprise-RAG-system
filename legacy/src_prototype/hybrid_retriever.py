class HybridRetriever:

    def __init__(
        self,
        vector_store,
        documents,
        vector_k=5,
        keyword_k=5
    ):

        self.vector_retriever = (
            vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": vector_k
                }
            )
        )

        from keyword_retriever import KeywordRetriever

        self.keyword_retriever = KeywordRetriever(
            documents,
            top_k=keyword_k
        )

    def retrieve(self, query):

        vector_docs = (
            self.vector_retriever.invoke(query)
        )

        keyword_docs = (
            self.keyword_retriever.retrieve(query)
        )

        # Combine results while removing duplicates
        combined = []

        seen = set()

        for document in (
            vector_docs + keyword_docs
        ):

            key = (
                document.metadata.get("source"),
                document.metadata.get("chunk_id")
            )

            if key not in seen:

                seen.add(key)
                combined.append(document)

        return combined