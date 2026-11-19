"""Thin HTTP client for the EKIP backend."""

from __future__ import annotations

import json
import os
import time
from typing import Iterator

import requests
import streamlit as st

API_URL = os.getenv("EKIP_API_URL", "http://localhost:8000").rstrip("/") + "/api/v1"
API_KEY = os.getenv("EKIP_API_KEY")


class APIError(Exception):
    pass


def _headers() -> dict:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _handle(r: requests.Response):
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise APIError(f"{r.status_code}: {detail}")
    if r.status_code == 204 or not r.content:
        return None
    return r.json()


def get(path: str, **params):
    return _handle(requests.get(API_URL + path, params=params, headers=_headers(), timeout=60))


def post(path: str, json_body: dict | None = None, timeout: float = 600, **kwargs):
    return _handle(requests.post(API_URL + path, json=json_body, headers=_headers(), timeout=timeout, **kwargs))


def put(path: str, json_body: dict):
    return _handle(requests.put(API_URL + path, json=json_body, headers=_headers(), timeout=60))


def delete(path: str):
    return _handle(requests.delete(API_URL + path, headers=_headers(), timeout=60))


def upload(collection: str, files: list, sync: bool = False):
    payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
    return _handle(
        requests.post(f"{API_URL}/collections/{collection}/documents", files=payload, data={"sync": str(sync).lower()},
                      headers=_headers(), timeout=900)
    )


def ask_stream(body: dict) -> Iterator[tuple[str, dict]]:
    """Yield (event, data) tuples from the SSE answer stream."""
    with requests.post(API_URL + "/ask", json={**body, "stream": True}, headers=_headers(), stream=True, timeout=600) as r:
        if r.status_code >= 400:
            _handle(r)
        event = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])


def healthy() -> bool:
    try:
        return get("/health")["status"] == "ok"
    except Exception:
        return False


def track_job(job: dict, label: str = "Working") -> dict | None:
    """Show a live progress bar for a background job until it finishes."""
    bar = st.progress(0.0, text=f"{label}: queued")
    while True:
        job = get(f"/jobs/{job['id']}")
        bar.progress(float(job["progress"] or 0), text=f"{label}: {job['message'] or job['status']}")
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(1.0)
    if job["status"] == "succeeded":
        bar.progress(1.0, text=f"{label}: done")
        return job["result"]
    bar.empty()
    st.error(f"{label} {job['status']}: {job.get('error') or ''}")
    return None


def fmt_time(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else ""
