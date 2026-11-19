from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import RAGConfig
from document_loader import load_document


def create_chunks(documents, config: RAGConfig):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    for chunk_id, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = chunk_id

    return chunks


if __name__ == "__main__":

    config = RAGConfig()

    documents = load_document("data/sample.pdf")

    chunks = create_chunks(
        documents,
        config
    )

    print("Number of chunks:", len(chunks))

    for chunk in chunks[:5]:

        print("\n--- CHUNK ---")

        print("Metadata:")
        print(chunk.metadata)

        print("\nText:")
        print(chunk.page_content)