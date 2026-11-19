from langchain_community.retrievers import BM25Retriever


class KeywordRetriever:

    def __init__(self, documents, top_k=5):

        self.retriever = BM25Retriever.from_documents(
            documents
        )

        self.retriever.k = top_k

    def retrieve(self, query):

        return self.retriever.invoke(query)