"""Document loaders. Each loader returns a list of ``Section`` (one per page / logical unit)."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx", ".html", ".htm", ".csv"}


@dataclass
class Section:
    text: str
    page: int | None = None  # 1-based page for paged formats
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            sections.append(Section(text=text, page=i))
    return sections


def load_text(path: Path) -> list[Section]:
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    return [Section(text=text)] if text else []


def load_markdown(path: Path) -> list[Section]:
    """Split markdown on top-level headings so each section keeps its heading context."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"(?m)^(?=#{1,2} )", raw)
    sections = []
    for part in parts:
        text = _clean(part)
        if not text:
            continue
        heading = text.splitlines()[0].lstrip("#").strip() if text.startswith("#") else None
        sections.append(Section(text=text, metadata={"section": heading} if heading else {}))
    return sections


def load_docx(path: Path) -> list[Section]:
    import docx

    document = docx.Document(str(path))
    sections: list[Section] = []
    current_heading, buffer = None, []

    def flush():
        text = _clean("\n".join(buffer))
        if text:
            sections.append(Section(text=text, metadata={"section": current_heading} if current_heading else {}))

    for para in document.paragraphs:
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading") and para.text.strip():
            flush()
            current_heading, buffer = para.text.strip(), [para.text.strip()]
        else:
            buffer.append(para.text)
    flush()
    for t_index, table in enumerate(document.tables):
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        text = _clean("\n".join(rows))
        if text:
            sections.append(Section(text=text, metadata={"section": f"Table {t_index + 1}"}))
    return sections


def load_html(path: Path) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    text = _clean(soup.get_text("\n"))
    return [Section(text=text, metadata={"section": title} if title else {})] if text else []


def load_csv(path: Path, rows_per_section: int = 50) -> list[Section]:
    content = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    sections = []
    for start in range(0, len(body), rows_per_section):
        lines = []
        for row in body[start : start + rows_per_section]:
            lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, row) if v))
        text = _clean("\n".join(lines))
        if text:
            sections.append(Section(text=text, metadata={"rows": f"{start + 1}-{start + len(lines)}"}))
    return sections


LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".csv": load_csv,
}


def load_document(path: str | Path) -> list[Section]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return LOADERS[ext](path)
