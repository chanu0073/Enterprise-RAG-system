from app.ingestion.chunker import chunk_sections, split_text
from app.ingestion.loaders import Section


def test_spans_are_exact_offsets():
    text = ("Sentence number one is here. " * 40 + "\n\n") * 3
    spans = split_text(text, 300, 50)
    assert spans
    for s, e in spans:
        assert 0 <= s < e <= len(text)
        assert e - s <= 300
        assert not text[s].isspace() and not text[e - 1].isspace()


def test_overlap_and_coverage():
    text = " ".join(f"word{i}" for i in range(600))
    spans = split_text(text, 200, 40)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 <= e1 + 1  # contiguous or overlapping
        assert s2 > s1


def test_chunk_metadata():
    chunks = chunk_sections([Section("a " * 500, page=3, metadata={"section": "Intro"})], "doc_x", 200, 20, {"source": "f.pdf"})
    assert all(c.page == 3 and c.metadata["source"] == "f.pdf" and c.metadata["section_idx"] == 0 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].embed_text.startswith("Intro")


def test_headings_and_identifier_lookup():
    from app.indexing.index import HybridIndex
    from app.indexing.models import HashingEmbedder

    page1 = "Contents\n6.1.2  Risk assessment ........ 9\n\n6.1.1  General\nPlanning text " + "lorem ipsum " * 40
    page2 = "6.1.2  Information security risk assessment\nThe organization shall define a process. " + "dolor sit " * 60
    chunks = chunk_sections([Section(page1, page=1), Section(page2, page=2)], "d", 300, 30)
    assert chunks[0].metadata.get("heading") == "6.1.1 General"  # TOC dot-leader line ignored
    assert all(c.metadata["heading"].startswith("6.1.2") for c in chunks if c.page == 2)
    index = HybridIndex.build(chunks, HashingEmbedder())
    hits = index.identifier_lookup(["6.1.2"])
    assert hits and all(index.chunks[i].page == 2 for i in hits)
    assert index.identifier_lookup(["A.6.1.2"]) == hits
