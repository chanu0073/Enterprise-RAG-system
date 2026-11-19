"""Shared helpers for fine-tuning: training pairs, hard-negative mining, batching."""

from __future__ import annotations

import random

from app.evaluation.metrics import is_relevant
from app.indexing.index import HybridIndex
from app.ingestion.chunker import Chunk


def training_items(dataset: dict) -> list[dict]:
    return [i for i in dataset["items"] if i.get("split") == "train" and i.get("passage") and i.get("doc_id")]


def mine_hard_negatives(index: HybridIndex, items: list[dict], per_item: int = 1, depth: int = 30,
                        embedder=None, seed: int = 13) -> dict[str, list[Chunk]]:
    """For each item, find highly ranked chunks that are NOT relevant (BM25 + optional dense)."""
    rng = random.Random(seed)
    out: dict[str, list[Chunk]] = {}
    qvecs = embedder.encode([i["question"] for i in items]) if embedder is not None else None
    for n, item in enumerate(items):
        ranked = [idx for idx, _ in index.keyword_search(item["question"], depth)]
        if qvecs is not None:
            ranked += [idx for idx, _ in index.vector_search(qvecs[n], depth)]
        seen, negs = set(), []
        for idx in ranked:
            if idx in seen:
                continue
            seen.add(idx)
            c = index.chunks[idx]
            if not is_relevant(c, item) and c.text != item["passage"]:
                negs.append(c)
            if len(negs) >= per_item * 3:
                break
        if len(negs) < per_item and index.chunks:  # fall back to random negatives
            negs += [c for c in rng.sample(index.chunks, min(len(index.chunks), per_item * 3)) if not is_relevant(c, item)]
        # skip the very top negative (often a near-duplicate / false negative) when we have spares
        out[item["qid"]] = (negs[1:] if len(negs) > per_item else negs)[:per_item]
    return out


def unique_passage_batches(items: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    """Batches where no passage repeats, so in-batch negatives are true negatives."""
    rng = random.Random(seed)
    pool = items[:]
    rng.shuffle(pool)
    batches: list[list[dict]] = []
    while pool:
        batch, used, rest = [], set(), []
        for it in pool:
            key = it["passage"][:200]
            if len(batch) < batch_size and key not in used:
                batch.append(it)
                used.add(key)
            else:
                rest.append(it)
        batches.append(batch)
        pool = rest
    return batches


def linear_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    import torch

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
