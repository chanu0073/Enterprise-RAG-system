from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Answer the question using ONLY the provided context.

If the context does not contain enough information
to answer the question, say:

"I don't have enough information in the provided documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)