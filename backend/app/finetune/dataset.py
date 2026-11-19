"""Training / evaluation dataset construction from the organisation's own documents.

Sources of (question -> gold passage) pairs:
- ``llm``: synthetic questions written by the local LLM for sampled chunks (best quality)
- ``heuristic``: offline rule-based questions (headings, definitions, salient key phrases)
- ``feedback``: real user questions that received positive feedback
- ``manual``: an uploaded, hand-labelled question set

Gold passages are stored as (doc_id, page, section_idx, start, end) spans, so the dataset stays
valid when the collection is re-chunked.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path

from app.core.config import get_settings
from app.core.db import get_db, new_id, now
from app.generation.llm import ExtractiveLLM, get_llm, parse_json
from app.generation.prompts import QUESTION_GEN_PROMPT
from app.indexing.index import STOPWORDS, tokenize
from app.ingestion.chunker import Chunk

log = logging.getLogger(__name__)

OUT_OF_DOMAIN = [
    "What is the capital of Japan?",
    "Who won the 2018 FIFA World Cup?",
    "What is the boiling point of water on Mars?",
    "How do I bake sourdough bread?",
    "What is the stock price of the company today?",
]

_HEADING_RE = re.compile(r"^\s*((?:\d+\.)*\d+|[A-Z]\.\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$", re.M)
_DEFN_RE = re.compile(
    r"(?:^|\n|\. )([A-Z][A-Za-z][\w\- ]{2,50}?)\s+(?:is|are|means|refers to|is defined as)\s+([^.\n]{15,200})"
)
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# --------------------------------------------------------------------------- generators


class HeuristicQuestionGenerator:
    def __init__(self, corpus: list[Chunk]):
        df: Counter = Counter()
        for c in corpus:
            df.update(set(tokenize(c.text)))
        self.n = max(len(corpus), 1)
        self.df = df

    def idf(self, t: str) -> float:
        return math.log((self.n + 1) / (self.df.get(t, 0) + 1)) + 1

    def generate(self, chunk: Chunk, n: int) -> list[dict]:
        text = chunk.text
        out: list[dict] = []
        sentences = [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 30]

        for m in _HEADING_RE.finditer(text):
            num, title = m.group(1), m.group(2).strip().rstrip(".")
            if len(title.split()) > 10:
                continue
            body = text[m.end() :].strip().split("\n")[0][:300]
            out.append({"question": f"What does section {num} {title} cover?", "answer": body, "category": "clause"})
            out.append({"question": f"What are the requirements for {title.lower()}?", "answer": body, "category": "topic"})
            break

        for m in _DEFN_RE.finditer(text):
            term = m.group(1).strip()
            if term.split()[0].lower() in STOPWORDS or len(term.split()) > 6:
                continue
            out.append({"question": f"What is {term[0].lower() + term[1:]}?", "answer": f"{term} is {m.group(2).strip()}", "category": "definition"})
            break

        # salient key phrase: the adjacent token pair with the highest idf mass
        tokens = tokenize(text)
        best, best_score = None, 0.0
        for a, b in zip(tokens, tokens[1:]):
            if a.isdigit() or b.isdigit() or len(a) < 3 or len(b) < 3:
                continue
            score = self.idf(a) + self.idf(b)
            if score > best_score:
                best, best_score = f"{a} {b}", score
        if best:
            ans = next((s for s in sentences if all(w in s.lower() for w in best.split())), sentences[0] if sentences else text[:200])
            templates = [
                "What does the documentation say about {}?",
                "How is {} handled?",
                "What is required regarding {}?",
            ]
            for tpl in templates[: max(1, n - len(out))]:
                out.append({"question": tpl.format(best), "answer": ans, "category": "keyphrase"})

        seen, unique = set(), []
        for q in out:
            if q["question"].lower() not in seen:
                seen.add(q["question"].lower())
                unique.append(q)
        return unique[:n]


class LLMQuestionGenerator:
    def __init__(self):
        self.llm = get_llm()
        if isinstance(self.llm, ExtractiveLLM) or not self.llm.available():
            raise RuntimeError(f"LLM {self.llm.name} unavailable for question generation")

    def generate(self, chunk: Chunk, n: int) -> list[dict]:
        raw = self.llm.chat(
            "You generate high-quality evaluation data. Output valid JSON only.",
            QUESTION_GEN_PROMPT.format(n=n, passage=chunk.text[:2500]),
            json_mode=True,
            max_tokens=600,
        )
        data = parse_json(raw) or {}
        items = data.get("items", data if isinstance(data, list) else [])
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            q = str(it.get("question", "")).strip()
            if 12 <= len(q) <= 300 and "passage" not in q.lower():
                out.append({"question": q if q.endswith("?") else q + "?", "answer": str(it.get("answer", "")).strip(), "category": "synthetic"})
        return out[:n]


# --------------------------------------------------------------------------- dataset ops


def _gold(chunk: Chunk) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "page": chunk.page,
        "section_idx": chunk.metadata.get("section_idx"),
        "start": chunk.start,
        "end": chunk.end,
        "source": chunk.metadata.get("source"),
        "passage": chunk.text,
    }


def generate_dataset(
    collection: str,
    chunks: list[Chunk],
    method: str = "auto",
    max_chunks: int = 150,
    questions_per_chunk: int = 2,
    test_ratio: float = 0.25,
    include_feedback: bool = True,
    include_out_of_domain: bool = True,
    seed: int = 42,
    name: str | None = None,
    ctx=None,
) -> dict:
    if not chunks:
        raise ValueError(f"Collection '{collection}' has no indexed chunks")
    rng = random.Random(seed)
    eligible = [c for c in chunks if len(c.text) >= 150]
    rng.shuffle(eligible)
    sample = eligible[:max_chunks]

    heuristic = HeuristicQuestionGenerator(chunks)
    generator, used_method = heuristic, "heuristic"
    if method in ("auto", "llm"):
        try:
            generator, used_method = LLMQuestionGenerator(), "llm"
        except Exception as exc:
            if method == "llm":
                raise
            log.warning("Falling back to heuristic question generation: %s", exc)

    test_chunks = {c.id for c in sample[: int(round(len(sample) * test_ratio))]}
    items: list[dict] = []
    llm_failures = 0
    for i, chunk in enumerate(sample):
        if ctx:
            ctx.check()
            ctx.progress(i / max(len(sample), 1), f"generating questions {i + 1}/{len(sample)}")
        try:
            qs = generator.generate(chunk, questions_per_chunk)
        except Exception as exc:
            llm_failures += 1
            log.debug("Generator failed on %s: %s", chunk.id, exc)
            qs = []
        origin = used_method
        if not qs and generator is not heuristic:
            qs, origin = heuristic.generate(chunk, questions_per_chunk), "heuristic"
        for q in qs:
            items.append(
                {
                    "qid": f"q{len(items):05d}",
                    **q,
                    **_gold(chunk),
                    "origin": origin,
                    "split": "test" if chunk.id in test_chunks else "train",
                }
            )

    if include_feedback:
        items.extend(_feedback_items(collection, chunks, start=len(items)))
    if include_out_of_domain:
        for q in OUT_OF_DOMAIN:
            items.append(
                {"qid": f"q{len(items):05d}", "question": q, "answer": "", "category": "out_of_domain",
                 "doc_id": None, "origin": "builtin", "split": "test"}
            )
    return save_dataset(collection, items, {"method": used_method, "seed": seed, "llm_failures": llm_failures}, name)


def _feedback_items(collection: str, chunks: list[Chunk], start: int) -> list[dict]:
    by_id = {c.id: c for c in chunks}
    rows = get_db().fetchall(
        "SELECT question, answer, sources FROM query_log WHERE feedback=1 AND collection LIKE ?", (f"%{collection}%",)
    )
    items = []
    for r in rows:
        srcs = r["sources"] if isinstance(r["sources"], list) else []
        if srcs and (chunk := by_id.get(srcs[0]["chunk_id"])):
            items.append(
                {"qid": f"q{start + len(items):05d}", "question": r["question"], "answer": r["answer"],
                 "category": "user_feedback", **_gold(chunk), "origin": "feedback", "split": "train"}
            )
    return items


def import_manual_dataset(collection: str, raw_items: list[dict], documents: list[dict], name: str | None = None) -> dict:
    """Import a hand-written eval set: [{question, answer?, category?, source?, page?, reference?}]."""
    by_name = {d["filename"]: d["id"] for d in documents}
    items = []
    for it in raw_items:
        if not it.get("question"):
            continue
        cat = it.get("category", "manual")
        ood = cat in ("out_of_document", "out_of_domain")
        items.append(
            {
                "qid": it.get("id") or f"q{len(items):05d}",
                "question": it["question"],
                "answer": it.get("answer", ""),
                "category": "out_of_domain" if ood else cat,
                "doc_id": None if ood else by_name.get(it.get("source"), it.get("doc_id")),
                "page": it.get("page"),
                "passage": it.get("reference"),
                "origin": "manual",
                "split": "test",
            }
        )
    return save_dataset(collection, items, {"method": "manual"}, name)


def _path(dataset_id: str) -> Path:
    if not re.fullmatch(r"ds_[a-f0-9]{12}", dataset_id):
        raise KeyError(dataset_id)
    return get_settings().datasets_dir / f"{dataset_id}.json"


def save_dataset(collection: str, items: list[dict], info: dict, name: str | None = None) -> dict:
    dataset_id = new_id("ds_")
    splits = Counter(i["split"] for i in items)
    meta = {
        "id": dataset_id,
        "name": name or f"{collection}-{info.get('method')}-{len(items)}",
        "collection": collection,
        "created_at": now(),
        "num_items": len(items),
        "splits": dict(splits),
        "categories": dict(Counter(i.get("category", "") for i in items)),
        **info,
    }
    _path(dataset_id).write_text(json.dumps({**meta, "items": items}, ensure_ascii=False, indent=1))
    return meta


def load_dataset(dataset_id: str) -> dict:
    path = _path(dataset_id)
    if not path.exists():
        raise KeyError(dataset_id)
    return json.loads(path.read_text())


def list_datasets() -> list[dict]:
    out = []
    for p in sorted(get_settings().datasets_dir.glob("ds_*.json"), key=lambda p: -p.stat().st_mtime):
        try:
            d = json.loads(p.read_text())
            d.pop("items", None)
            out.append(d)
        except Exception:
            continue
    return out


def delete_dataset(dataset_id: str) -> None:
    path = _path(dataset_id)
    if not path.exists():
        raise KeyError(dataset_id)
    path.unlink()


def split_items(dataset: dict, split: str) -> list[dict]:
    return [i for i in dataset["items"] if split == "all" or i.get("split") == split]
