"""LoRA adaptation of a small generator LLM for grounded, cited answering.

Each training example mimics the production prompt: the gold passage is shuffled among hard
negative passages and the target answer cites the gold passage's number. A share of examples
contain only negatives and target the refusal string, teaching the model to abstain instead of
hallucinating. Requires the optional `peft` dependency (see requirements-finetune.txt).
"""

from __future__ import annotations

import logging
import random
import time

from app.core.config import get_settings
from app.core.db import new_id
from app.finetune.common import linear_warmup_schedule, mine_hard_negatives, training_items
from app.finetune.dataset import load_dataset
from app.generation.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from app.indexing.models import get_registry, resolve_device
from app.services.knowledge import get_knowledge

log = logging.getLogger(__name__)


def build_sft_examples(items: list[dict], negs: dict, refusal_ratio: float, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    examples = []
    for it in items:
        if not it.get("answer"):
            continue
        negatives = [{"text": n.text, "source": n.metadata.get("source"), "page": n.page} for n in negs.get(it["qid"], [])]
        if negatives and rng.random() < refusal_ratio:
            examples.append((build_user_prompt(it["question"], negatives), REFUSAL))
            continue
        passages = negatives + [{"text": it["passage"], "source": it.get("source"), "page": it.get("page")}]
        rng.shuffle(passages)
        ref = next(i for i, p in enumerate(passages, start=1) if p["text"] == it["passage"])
        answer = it["answer"].strip().rstrip(".") + f". [{ref}]"
        examples.append((build_user_prompt(it["question"], passages), answer))
    return examples


def train_llm_lora(
    dataset_id: str,
    base_model: str | None = None,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 1536,
    negatives_per_query: int = 2,
    refusal_ratio: float = 0.15,
    seed: int = 42,
    name: str | None = None,
    ctx=None,
) -> dict:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("LLM fine-tuning needs `pip install -r requirements-finetune.txt` (peft)") from exc

    torch.manual_seed(seed)
    settings = get_settings()
    base_model = base_model or settings.hf_base_model
    dataset = load_dataset(dataset_id)
    items = training_items(dataset)
    device = resolve_device()

    if ctx:
        ctx.progress(0.02, "building SFT examples")
    index = get_knowledge().get_index(dataset["collection"])
    negs = mine_hard_negatives(index, items, per_item=negatives_per_query)
    examples = build_sft_examples(items, negs, refusal_ratio, seed)
    if len(examples) < 8:
        raise ValueError("Need at least 8 items with answers (generate the dataset with the LLM method)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    target = [m for m in ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn") if any(n.endswith(m) for n, _ in model.named_modules())]
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                             target_modules=target, task_type="CAUSAL_LM"))
    # train LoRA weights in fp32 for stability even when the base is fp16
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device).train()

    def encode(user: str, answer: str):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{user}\n"
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        p_ids = p_ids[-(max_length - len(a_ids)) :] if len(p_ids) + len(a_ids) > max_length else p_ids
        return p_ids + a_ids, [-100] * len(p_ids) + a_ids  # loss only on the answer tokens

    encoded = [encode(u, a) for u, a in examples]
    rng = random.Random(seed)
    steps_per_epoch = (len(encoded) + batch_size * grad_accum - 1) // (batch_size * grad_accum)
    total = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = linear_warmup_schedule(optimizer, max(1, total // 10), total)
    use_amp = device == "cuda"
    step, micro, losses, t0 = 0, 0, [], time.time()
    for epoch in range(epochs):
        rng.shuffle(encoded)
        for b in range(0, len(encoded), batch_size):
            if ctx:
                ctx.check()
            batch = encoded[b : b + batch_size]
            width = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), width), tokenizer.pad_token_id)
            labels = torch.full((len(batch), width), -100)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for i, (x, y) in enumerate(batch):
                ids[i, : len(x)], labels[i, : len(y)], mask[i, : len(x)] = torch.tensor(x), torch.tensor(y), 1
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
            (loss / grad_accum).backward()
            losses.append(float(loss))
            micro += 1
            if micro % grad_accum == 0 or b + batch_size >= len(encoded):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if ctx:
                    ctx.progress(0.05 + 0.9 * step / max(total, 1), f"epoch {epoch + 1}/{epochs} step {step}/{total} loss {loss:.4f}")

    model_id = new_id("llm_")
    out_dir = settings.models_dir / "llm" / model_id
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    metrics = {"train": {"examples": len(examples), "epochs": epochs, "optimizer_steps": step,
                         "final_loss": sum(losses[-20:]) / max(1, len(losses[-20:])), "seconds": round(time.time() - t0, 1),
                         "device": device}}
    row = get_registry().register(
        kind="llm",
        name=name or f"{base_model.split('/')[-1]}-lora-{dataset['collection']}",
        path=str(out_dir),
        base_model=base_model,
        metrics=metrics,
        params={"backend": "hf", "dataset_id": dataset_id, "lora_r": lora_r, "lora_alpha": lora_alpha,
                "epochs": epochs, "lr": learning_rate, "refusal_ratio": refusal_ratio},
    )
    return {"model": row, "metrics": metrics,
            "next_step": "Activate this model, then run a generation evaluation to compare against the base LLM."}
