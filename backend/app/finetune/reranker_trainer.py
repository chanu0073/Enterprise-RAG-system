"""Domain adaptation of the cross-encoder reranker (binary relevance with hard negatives)."""

from __future__ import annotations

import logging
import random
import time

from app.core.config import get_settings
from app.core.db import new_id
from app.evaluation.runner import IndexProvider, run_retrieval_eval
from app.finetune.common import linear_warmup_schedule, mine_hard_negatives, training_items
from app.finetune.dataset import load_dataset
from app.indexing.models import active_embedder, get_registry, resolve_device
from app.services.knowledge import get_knowledge

log = logging.getLogger(__name__)


def train_reranker(
    dataset_id: str,
    base_model: str | None = None,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    negatives_per_query: int = 3,
    max_length: int = 384,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    name: str | None = None,
    ctx=None,
) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dataset = load_dataset(dataset_id)
    items = training_items(dataset)
    if len(items) < 8:
        raise ValueError(f"Need at least 8 training pairs, dataset has {len(items)}")
    reg = get_registry()
    base_model = base_model or reg.active_path("reranker")
    device = resolve_device()

    if ctx:
        ctx.progress(0.02, "mining hard negatives")
    index = get_knowledge().get_index(dataset["collection"])
    negs = mine_hard_negatives(index, items, per_item=negatives_per_query, embedder=active_embedder())
    examples = []
    for it in items:
        examples.append((it["question"], it["passage"], 1.0))
        examples += [(it["question"], n.text, 0.0) for n in negs.get(it["qid"], [])]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1).to(device)
    model.train()
    steps_per_epoch = (len(examples) + batch_size - 1) // batch_size
    total = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = linear_warmup_schedule(optimizer, int(total * warmup_ratio), total)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # positives are rarer than negatives: weight them up
    pos_weight = torch.tensor([float(negatives_per_query)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    step, losses, t0 = 0, [], time.time()
    for epoch in range(epochs):
        rng.shuffle(examples)
        for b in range(0, len(examples), batch_size):
            if ctx:
                ctx.check()
            batch = examples[b : b + batch_size]
            enc = tokenizer([x[0] for x in batch], [x[1] for x in batch], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([x[2] for x in batch], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.view(-1)
                loss = loss_fn(logits.float(), labels)
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
                ctx.progress(0.05 + 0.65 * step / total, f"epoch {epoch + 1}/{epochs} step {step}/{total} loss {loss:.4f}")

    model_id = new_id("rer_")
    out_dir = get_settings().models_dir / "reranker" / model_id
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    train_secs = time.time() - t0

    if ctx:
        ctx.progress(0.75, "evaluating base vs fine-tuned reranker")
    provider = IndexProvider(dataset["collection"])
    overrides = {"use_reranker": True, "mode": "hybrid"}
    base_run = run_retrieval_eval(dataset_id, config_overrides=overrides, reranker_model=base_model, provider=provider, save=False)
    tuned_run = run_retrieval_eval(dataset_id, config_overrides=overrides, reranker_model=str(out_dir), provider=provider, save=False)
    strip = lambda m: {k: v for k, v in m.items() if k != "by_category"}  # noqa: E731
    base_m, tuned_m = strip(base_run["metrics"]), strip(tuned_run["metrics"])
    metrics = {
        "eval": tuned_m,
        "baseline": base_m,
        "improvement": {k: round(tuned_m[k] - base_m[k], 4) for k in ("hit@1", "hit@5", "mrr", "ndcg@10") if k in tuned_m},
        # activate only if ranking quality improved on held-out data without hurting top-1 accuracy
        "recommended": tuned_m.get("ndcg@10", 0) > base_m.get("ndcg@10", 0) and tuned_m.get("hit@1", 0) >= base_m.get("hit@1", 0) - 0.01,
        "train": {"examples": len(examples), "epochs": epochs, "steps": step,
                  "final_loss": sum(losses[-10:]) / max(1, len(losses[-10:])), "seconds": round(train_secs, 1), "device": device},
    }
    row = reg.register(
        kind="reranker",
        name=name or f"{base_model.split('/')[-1]}-ft-{dataset['collection']}",
        path=str(out_dir),
        base_model=base_model,
        metrics=metrics,
        params={"dataset_id": dataset_id, "epochs": epochs, "batch_size": batch_size, "lr": learning_rate,
                "negatives_per_query": negatives_per_query, "loss": "BCE"},
    )
    return {"model": row, "metrics": metrics}
