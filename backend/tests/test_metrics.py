from app.evaluation.metrics import faithfulness, is_relevant, retrieval_metrics, token_f1
from app.ingestion.chunker import Chunk


def chunk(doc, start, end, page=1):
    return Chunk(id=f"{doc}:{start}", doc_id=doc, text="x", page=page, start=start, end=end, chunk_index=0,
                 metadata={"section_idx": 0})


GOLD = {"doc_id": "d1", "page": 1, "section_idx": 0, "start": 100, "end": 400}


def test_span_relevance_is_chunking_independent():
    assert is_relevant(chunk("d1", 150, 300), GOLD)       # smaller chunk inside gold
    assert is_relevant(chunk("d1", 0, 800), GOLD)         # larger chunk containing gold
    assert not is_relevant(chunk("d1", 380, 900), GOLD)   # marginal overlap
    assert not is_relevant(chunk("d2", 100, 400), GOLD)   # other document
    assert not is_relevant(chunk("d1", 100, 400, page=2), GOLD)


def test_ranking_metrics():
    retrieved = [chunk("d2", 0, 10), chunk("d1", 100, 400), chunk("d3", 0, 10)]
    m = retrieval_metrics(retrieved, GOLD, n_relevant=1)
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert abs(m["mrr"] - 0.5) < 1e-9
    assert 0 < m["ndcg@3"] < 1
    assert m["recall@3"] == 1.0


def test_generation_metrics():
    ctx = "Backups are taken nightly and retained for 35 days."
    assert faithfulness("Backups are taken nightly [1].", ctx) == 1.0
    assert faithfulness("Quantum servers orbit Jupiter weekly.", ctx) == 0.0
    assert token_f1("retained for 35 days", "Backups retained 35 days") > 0.5
