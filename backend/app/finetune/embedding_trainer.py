"""Domain adaptation of the embedding model.

Contrastive training with Multiple-Negatives-Ranking loss (in-batch negatives) plus mined
hard negatives, on (question, gold passage) pairs generated from the organisation's documents.
After training, base and fine-tuned models are evaluated on the held-out split and the new
model is registered (not activated) together with its metrics.
"""

from __future__ import annotations

import logging
import time

from app.core.config import get_settings
from app.core.db import new_id
from app.evaluation.runner import compare_embedding_models
from app.finetune.common import linear_warmup_schedule, mine_hard_negatives, training_items, unique_passage_batches
from app.finetune.dataset import load_dataset
from app.indexing.models import get_registry, resolve_device
from app.services.knowledge import get_knowledge

log = logging.getLogger(__name__)


def _features(model, texts: list[str], device: str):
    prep = getattr(model, "preprocess", None) or model.tokenize
    feats = prep(texts)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in feats.items()}


def train_embedding_model(
    dataset_id: str,
    base_model: str | None = None,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    hard_negatives: bool = True,
    max_seq_length: int = 256,
    scale: float = 20.0,
    seed: int = 42,
    name: str | None = None,
    ctx=None,
) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    dataset = load_dataset(dataset_id)
    items = training_items(dataset)
    if len(items) < 8:
        raise ValueError(f"Need at least 8 training pairs, dataset has {len(items)}")
    reg = get_registry()
    base_model = base_model or reg.active_path("embedding")
    if base_model.startswith("hash"):
        raise ValueError("The hashing embedder is not trainable; set a transformer embedding model")

    device = resolve_device()
    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = max_seq_length
    model.train()

    negs = {}
    if hard_negatives:
        if ctx:
            ctx.progress(0.02, "mining hard negatives")
        index = get_knowledge().get_index(dataset["collection"])
        negs = mine_hard_negatives(index, items, per_item=1)

    batches_per_epoch = len(unique_passage_batches(items, batch_size, seed))
    total = batches_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = linear_warmup_schedule(optimizer, int(total * warmup_ratio), total)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = torch.nn.CrossEntropyLoss()

    step, losses, t0 = 0, [], time.time()
    for epoch in range(epochs):
        for batch in unique_passage_batches(items, batch_size, seed + epoch):
            if ctx:
                ctx.check()
            queries = [b["question"] for b in batch]
            docs = [b["passage"] for b in batch]
            docs += [n.text for b in batch for n in negs.get(b["qid"], [])]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                q = model(_features(model, queries, device))["sentence_embedding"]
                d = model(_features(model, docs, device))["sentence_embedding"]
                q = torch.nn.functional.normalize(q, dim=-1)
                d = torch.nn.functional.normalize(d, dim=-1)
                scores = q @ d.T * scale
                loss = ce(scores.float(), torch.arange(len(batch), device=scores.device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss))
            step += 1
            if ctx:
                ctx.progress(0.05 + 0.7 * step / total, f"epoch {epoch + 1}/{epochs} step {step}/{total} loss {loss:.4f}")

    model_id = new_id("emb_")
    out_dir = get_settings().models_dir / "embedding" / model_id
    model.save(str(out_dir))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    train_secs = time.time() - t0

    if ctx:
        ctx.progress(0.8, "evaluating base vs fine-tuned")
    comparison = compare_embedding_models(dataset_id, [base_model, str(out_dir)])
    base_m, tuned_m = comparison[base_model], comparison[str(out_dir)]
    metrics = {
        "eval": tuned_m,
        "baseline": base_m,
        "improvement": {k: round(tuned_m[k] - base_m[k], 4) for k in ("hit@1", "hit@5", "mrr", "ndcg@10") if k in tuned_m},
        # activate only if ranking quality improved on held-out data without hurting top-1 accuracy
        "recommended": tuned_m.get("ndcg@10", 0) > base_m.get("ndcg@10", 0) and tuned_m.get("hit@1", 0) >= base_m.get("hit@1", 0) - 0.01,
        "train": {"pairs": len(items), "epochs": epochs, "steps": step, "final_loss": sum(losses[-10:]) / max(1, len(losses[-10:])),
                  "seconds": round(train_secs, 1), "device": device},
    }
    row = reg.register(
        kind="embedding",
        name=name or f"{base_model.split('/')[-1]}-ft-{dataset['collection']}",
        path=str(out_dir),
        base_model=base_model,
        metrics=metrics,
        params={"dataset_id": dataset_id, "epochs": epochs, "batch_size": batch_size, "lr": learning_rate,
                "hard_negatives": hard_negatives, "loss": "MultipleNegativesRanking"},
    )
    return {"model": row, "metrics": metrics}
