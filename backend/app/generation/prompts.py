"""Prompt templates for grounded, cited answers."""

from __future__ import annotations

REFUSAL = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = f"""You are an enterprise knowledge assistant. Answer strictly from the numbered context passages.

Rules:
- Use ONLY facts stated in the context. Never use outside knowledge.
- Cite every claim with the number of the context passage it comes from, e.g. [1] or [2][3].
  Citations are passage numbers only; write clause or section numbers as plain text, never in square brackets.
- If the context does not contain the answer, reply exactly: "{REFUSAL}"
- Be concise and precise. Use bullet points for lists of requirements or steps.
- Preserve clause numbers, names and figures exactly as written."""


def format_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        loc = p.get("source") or "document"
        if p.get("page"):
            loc += f", page {p['page']}"
        if p.get("section"):
            loc += f", {p['section']}"
        blocks.append(f"[{i}] ({loc})\n{p['text']}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, passages: list[dict], history: list[dict] | None = None) -> str:
    convo = ""
    if history:
        turns = [f"{t['role'].capitalize()}: {t['content']}" for t in history[-6:]]
        convo = "Conversation so far:\n" + "\n".join(turns) + "\n\n"
    return f"{convo}Context:\n{format_context(passages)}\n\nQuestion: {question}\n\nAnswer (with citations):"


QUESTION_GEN_PROMPT = """You write evaluation questions for an enterprise search system.

Read the passage and write {n} distinct questions that a employee might ask and that the passage answers.
For each question also give a short answer taken from the passage.
Questions must be self-contained (mention the specific topic, not "this passage").

Return JSON only: {{"items": [{{"question": "...", "answer": "..."}}]}}

Passage:
\"\"\"{passage}\"\"\""""


JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}
Retrieved context:
\"\"\"{context}\"\"\"
System answer: {answer}

Score each from 1 (worst) to 5 (best):
- faithfulness: is every claim in the system answer supported by the context?
- correctness: does the system answer agree with the reference answer?
- completeness: does it cover the key points of the reference?

Return JSON only: {{"faithfulness": n, "correctness": n, "completeness": n, "rationale": "one sentence"}}"""
