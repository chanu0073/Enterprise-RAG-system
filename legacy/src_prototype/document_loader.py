from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader


def load_document(file_path: str):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension in [".md", ".markdown"]:
        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    # Add consistent source metadata
    for document in documents:
        document.metadata["source"] = str(path)

    return documents


if __name__ == "__main__":

    documents = load_document("data/sample.pdf")

    print("Number of documents:", len(documents))

    for document in documents[:2]:

        print("\n--- DOCUMENT ---")

        print("Metadata:")
        print(document.metadata)

        print("\nText:")
        print(document.page_content[:500])