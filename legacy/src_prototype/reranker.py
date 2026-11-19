from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):

        if not documents:
            return []

        pairs = [
            (
                query,
                document.page_content
            )
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = list(
            zip(documents, scores)
        )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for document, score in ranked[:top_k]:

            results.append(
                {
                    "document": document,
                    "score": float(score)
                }
            )

        return results