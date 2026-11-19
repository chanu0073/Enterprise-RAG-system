from pathlib import Path

from langchain_community.vectorstores import FAISS

from config import RAGConfig
from embeddings import create_embeddings


def create_vector_store(chunks, config: RAGConfig):

    embeddings = create_embeddings(config)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


def save_vector_store(vector_store, config: RAGConfig):

    path = Path(config.index_path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(path)
    )

    print(
        f"FAISS index saved to {path}"
    )


def load_vector_store(config: RAGConfig):

    embeddings = create_embeddings(config)

    vector_store = FAISS.load_local(
        config.index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store