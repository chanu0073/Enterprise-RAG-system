from pathlib import Path

from config import RAGConfig

from document_loader import load_document
from chunker import create_chunks
from embeddings import create_embeddings
from vector_store import (
    create_vector_store,
    save_vector_store
)

from retriever import RAGRetriever

from prompt import RAG_PROMPT
from llm import create_llm


class RAGApplication:

    def __init__(
        self,
        documents,
        vector_store,
        config
    ):

        self.documents = documents
        self.vector_store = vector_store
        self.config = config

        self.retriever = RAGRetriever(
            vector_store,
            documents,
            config
        )

        self.llm = create_llm(config)

    def answer(
        self,
        question,
        source_filter=None
    ):

        results = self.retriever.retrieve(
            question,
            source_filter=source_filter
        )

        if not results:

            return {
                "answer":
                    "I don't have enough information "
                    "in the provided documents.",
                "sources": []
            }

        context_parts = []

        sources = []

        for result in results:

            document = result["document"]

            context_parts.append(
                document.page_content
            )

            sources.append(
                {
                    "source":
                        document.metadata.get(
                            "source"
                        ),
                    "page":
                        document.metadata.get(
                            "page"
                        ),
                    "chunk_id":
                        document.metadata.get(
                            "chunk_id"
                        ),
                    "score":
                        result["score"]
                }
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        prompt_value = RAG_PROMPT.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = self.llm.invoke(
            prompt_value
        )

        return {
            "answer": response.content,
            "sources": sources
        }


def build_application(
    file_path,
    config
):

    print("Loading document...")

    documents = load_document(
        file_path
    )

    print(
        "Documents loaded:",
        len(documents)
    )

    print("Creating chunks...")

    chunks = create_chunks(
        documents,
        config
    )

    print(
        "Chunks created:",
        len(chunks)
    )

    print("Creating FAISS index...")

    vector_store = create_vector_store(
        chunks,
        config
    )

    save_vector_store(
        vector_store,
        config
    )

    return RAGApplication(
        chunks,
        vector_store,
        config
    )


if __name__ == "__main__":

    config = RAGConfig()

    application = build_application(
        "data/sample.pdf",
        config
    )

    print("\nRAG system ready.")

    while True:

        question = input(
            "\nAsk a question (or type 'exit'): "
        )

        if question.lower() == "exit":
            break

        result = application.answer(
            question
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")

        for source in result["sources"]:

            print(
                f"Source: {source['source']} | "
                f"Page: {source['page']} | "
                f"Chunk: {source['chunk_id']} | "
                f"Rerank Score: {source['score']:.4f}"
            )