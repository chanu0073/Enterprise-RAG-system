from langchain_huggingface import HuggingFaceEmbeddings

from config import RAGConfig


def create_embeddings(config: RAGConfig):

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model
    )

    return embeddings


if __name__ == "__main__":

    config = RAGConfig()

    embeddings = create_embeddings(config)

    vector = embeddings.embed_query(
        "What is information security?"
    )

    print("Embedding dimension:", len(vector))
    print("First 10 values:", vector[:10])