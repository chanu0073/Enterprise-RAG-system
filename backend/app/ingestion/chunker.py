"""Offset-preserving recursive chunker.

Every chunk keeps (page, start, end) character offsets into its source section. Evaluation
uses these offsets to judge relevance by *span overlap*, which makes retrieval metrics
comparable across different chunk sizes (needed for chunking optimization).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.loaders import Section

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

# Numbered clause headings ("6.1.2  Information security risk assessment", "A.5.7 Threat intelligence")
# or markdown headings. Table-of-contents lines (dot leaders) are ignored.
HEADING_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+(?P<md>[^\n]{2,100})|(?P<num>(?:[A-Z]\.)?\d{1,2}(?:\.\d{1,2}){0,3})\.?[ \t]+(?P<title>[A-Z][^\n]{2,90}))[ \t]*$",
    re.M,
)


def find_headings(text: str) -> list[tuple[int, str]]:
    """Return (offset, heading) pairs found in ``text``."""
    out = []
    for m in HEADING_RE.finditer(text):
        line = m.group(0).strip()
        if "...." in line or "…" in line:
            continue
        if m.group("md"):
            out.append((m.start(), m.group("md").strip()))
        else:
            title = m.group("title").strip()
            if len(title.split()) <= 12 and not title.endswith((",", ";")):
                out.append((m.start(), f"{m.group('num')} {title}"))
    return out


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    page: int | None
    start: int
    end: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page": self.page,
            "start": self.start,
            "end": self.end,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)

    @property
    def embed_text(self) -> str:
        """Text used for embedding/BM25/reranking: the active heading gives each chunk its context."""
        heading = self.metadata.get("heading") or self.metadata.get("section")
        if heading and heading[:40] not in self.text[:200]:
            return f"{heading}\n{self.text}"
        return self.text


def _split_spans(text: str, start: int, end: int, size: int, seps: list[str]) -> list[tuple[int, int]]:
    if end - start <= size:
        return [(start, end)]
    for i, sep in enumerate(seps):
        if sep == "":
            return [(s, min(s + size, end)) for s in range(start, end, size)]
        segment = text[start:end]
        if sep not in segment:
            continue
        spans, pos = [], start
        for m in re.finditer(re.escape(sep), segment):
            cut = start + m.end()
            if cut > pos:
                spans.append((pos, cut))
                pos = cut
        if pos < end:
            spans.append((pos, end))
        out: list[tuple[int, int]] = []
        for s, e in spans:
            out.extend(_split_spans(text, s, e, size, seps[i + 1 :]) if e - s > size else [(s, e)])
        return out
    return [(start, end)]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) spans of chunks covering ``text``."""
    if not text.strip():
        return []
    chunk_overlap = min(chunk_overlap, chunk_size // 2)
    pieces = _split_spans(text, 0, len(text), chunk_size, DEFAULT_SEPARATORS)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(pieces):
        s, j = pieces[i][0], i
        while j < len(pieces) and pieces[j][1] - s <= chunk_size:
            j += 1
        j = max(j, i + 1)
        e = pieces[j - 1][1]
        spans.append((s, e))
        if j >= len(pieces):
            break
        k = j
        while k - 1 > i and e - pieces[k - 1][0] <= chunk_overlap:
            k -= 1
        i = k

    # trim surrounding whitespace while keeping offsets exact
    trimmed = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= 20 or (e > s and len(spans) == 1):
            trimmed.append((s, e))
    return trimmed


def chunk_sections(
    sections: list[Section],
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
    base_metadata: dict | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    carried: str | None = None  # heading active at the end of the previous page/section
    for section_idx, section in enumerate(sections):
        headings = find_headings(section.text)
        for start, end in split_text(section.text, chunk_size, chunk_overlap):
            # a heading in the first ~120 chars of the chunk (e.g. after a page header) governs it
            active = [h for off, h in headings if off <= min(start + 120, end)]
            heading = active[-1] if active else carried
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{idx}",
                    doc_id=doc_id,
                    text=section.text[start:end],
                    page=section.page,
                    start=start,
                    end=end,
                    chunk_index=idx,
                    metadata={
                        **(base_metadata or {}),
                        **section.metadata,
                        "section_idx": section_idx,
                        **({"heading": heading} if heading else {}),
                    },
                )
            )
        if headings:
            carried = headings[-1][1]
    return chunks
