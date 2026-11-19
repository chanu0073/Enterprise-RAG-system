"""LLM backends: Ollama (default), HuggingFace transformers (+ LoRA adapters), extractive fallback."""

from __future__ import annotations

import json
import logging
import re
import threading
from functools import lru_cache
from typing import Iterator, Protocol

import httpx

from app.core.config import get_settings
from app.generation.prompts import REFUSAL
from app.indexing.index import tokenize

log = logging.getLogger(__name__)


class LLM(Protocol):
    name: str

    def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int | None = None) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...

    def available(self) -> bool: ...


class OllamaLLM:
    def __init__(self, model: str, url: str, timeout: float, temperature: float, max_tokens: int):
        self.model, self.url = model, url.rstrip("/")
        self.name = f"ollama:{model}"
        self.timeout, self.temperature, self.max_tokens = timeout, temperature, max_tokens

    def _payload(self, system: str, user: str, stream: bool, json_mode: bool, max_tokens: int | None) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": stream,
            "options": {"temperature": self.temperature, "num_predict": max_tokens or self.max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int | None = None) -> str:
        r = httpx.post(
            f"{self.url}/api/chat", json=self._payload(system, user, False, json_mode, max_tokens), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def stream(self, system: str, user: str) -> Iterator[str]:
        with httpx.stream(
            "POST", f"{self.url}/api/chat", json=self._payload(system, user, True, False, None), timeout=self.timeout
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if piece := data.get("message", {}).get("content"):
                    yield piece
                if data.get("done"):
                    break

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=3)
            names = {m["name"] for m in r.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except Exception:
            return False


class HFLLM:
    """Local transformers model, optionally with a LoRA adapter produced by the fine-tuning module."""

    def __init__(self, base_model: str, adapter_path: str | None, temperature: float, max_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from app.indexing.models import resolve_device

        self.device = resolve_device()
        self.name = f"hf:{base_model}" + (f"+{adapter_path}" if adapter_path else "")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.temperature, self.max_tokens = temperature, max_tokens
        self._lock = threading.Lock()

    def _inputs(self, system: str, user: str):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system}\n\n{user}\n"
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def _gen_kwargs(self, max_tokens: int | None) -> dict:
        kw = {"max_new_tokens": max_tokens or self.max_tokens, "pad_token_id": self.tokenizer.eos_token_id}
        if self.temperature > 0:
            kw.update(do_sample=True, temperature=self.temperature)
        else:
            kw.update(do_sample=False)
        return kw

    def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int | None = None) -> str:
        import torch

        with self._lock, torch.no_grad():
            inputs = self._inputs(system, user)
            out = self.model.generate(**inputs, **self._gen_kwargs(max_tokens))
            return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def stream(self, system: str, user: str) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._inputs(system, user)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run():
            with self._lock:
                self.model.generate(**inputs, **self._gen_kwargs(None), streamer=streamer)

        threading.Thread(target=run, daemon=True).start()
        yield from streamer

    def available(self) -> bool:
        return True


_SENT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


class ExtractiveLLM:
    """No-LLM fallback: returns the most question-relevant sentences from the context with citations."""

    name = "extractive"

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def answer(self, question: str, passages: list[dict]) -> str:
        q = set(tokenize(question))
        scored = []
        for p_idx, p in enumerate(passages, start=1):
            for order, sent in enumerate(s.strip() for s in _SENT_RE.split(p["text"])):
                if len(sent) < 25:
                    continue
                toks = set(tokenize(sent))
                overlap = len(q & toks) / (len(q) or 1)
                scored.append((overlap - 0.01 * p_idx, p_idx, order, sent))
        scored = [s for s in scored if s[0] > 0.15]
        if not scored:
            return REFUSAL
        best = sorted(scored, key=lambda x: -x[0])[: self.max_sentences]
        best.sort(key=lambda x: (x[1], x[2]))
        return " ".join(f"{s} [{p}]" for _, p, _, s in best)

    # Protocol methods (used when extractive is the configured backend)
    def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int | None = None) -> str:
        raise RuntimeError("Extractive backend cannot follow free-form prompts")

    def stream(self, system: str, user: str) -> Iterator[str]:
        raise RuntimeError("Extractive backend cannot follow free-form prompts")

    def available(self) -> bool:
        return True


_llm_cache: dict[str, object] = {}
_llm_lock = threading.Lock()


def get_llm(model_row: dict | None = None):
    """Build the LLM for a registry row (defaults to the active LLM)."""
    from app.indexing.models import get_registry

    s = get_settings()
    row = model_row or get_registry().active("llm")
    key = row["id"]
    with _llm_lock:
        if key in _llm_cache:
            return _llm_cache[key]
        params = row.get("params") or {}
        backend = params.get("backend", s.llm_backend)
        if backend == "ollama":
            model = row["path"].split(":", 1)[1] if row["path"].startswith("ollama:") else row["path"]
            llm = OllamaLLM(model, s.ollama_url, s.llm_timeout, s.llm_temperature, s.llm_max_tokens)
        elif backend == "hf":
            base = row.get("base_model") or row["path"].removeprefix("hf:")
            adapter = row["path"] if row.get("base_model") else None
            llm = HFLLM(base, adapter, s.llm_temperature, s.llm_max_tokens)
        else:
            llm = ExtractiveLLM()
        _llm_cache[key] = llm
        return llm


@lru_cache
def get_extractive() -> ExtractiveLLM:
    return ExtractiveLLM()


def parse_json(text: str) -> dict | list | None:
    """Tolerant JSON extraction from LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
