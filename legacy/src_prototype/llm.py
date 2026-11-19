from langchain_ollama import ChatOllama

from config import RAGConfig


def create_llm(config: RAGConfig):

    llm = ChatOllama(
        model=config.llm_model,
        temperature=0
    )

    return llm