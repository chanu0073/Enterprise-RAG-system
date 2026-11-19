"""End-to-end API flow: ingest -> search -> ask -> dataset -> eval -> optimize -> apply."""

import time


def wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            assert job["status"] == "succeeded", job
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


def test_health(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_full_flow(client, policy_file):
    r = client.post("/api/v1/collections", json={"name": "policies", "description": "HR & security"})
    assert r.status_code == 201

    with policy_file.open("rb") as f:
        r = client.post("/api/v1/collections/policies/documents", files=[("files", ("policy.md", f, "text/markdown"))],
                        data={"sync": "true"})
    res = r.json()["results"][0]
    assert res["document"]["status"] == "indexed", res
    doc_id = res["document"]["id"]

    # duplicate upload is detected
    with policy_file.open("rb") as f:
        r = client.post("/api/v1/collections/policies/documents", files=[("files", ("policy.md", f, "text/markdown"))],
                        data={"sync": "true"})
    assert r.json()["results"][0]["duplicate"] is True

    # unsupported types are rejected per-file
    r = client.post("/api/v1/collections/policies/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert "error" in r.json()["results"][0]

    r = client.post("/api/v1/search", json={"query": "how long are backups retained", "collections": ["policies"]})
    top = r.json()["results"][0]
    assert "Backup" in top["text"]

    r = client.post("/api/v1/ask", json={"query": "How long are backups retained?", "collections": ["policies"]})
    body = r.json()
    assert "35 days" in body["answer"] and body["sources"] and body["query_id"]

    # cache
    r2 = client.post("/api/v1/ask", json={"query": "How long are backups retained?", "collections": ["policies"]})
    assert r2.json()["cached"] is True
    # any index change invalidates cached answers
    client.post("/api/v1/collections/policies/documents",
                files=[("files", ("extra.txt", b"Visitors must sign in at reception and wear a badge at all times.", "text/plain"))],
                data={"sync": "true"})
    r3 = client.post("/api/v1/ask", json={"query": "How long are backups retained?", "collections": ["policies"]})
    assert r3.json()["cached"] is False

    # streaming
    with client.stream("POST", "/api/v1/ask", json={"query": "When must incidents be reported?",
                                                     "collections": ["policies"], "stream": True}) as s:
        text = "".join(s.iter_text())
    assert "event: sources" in text and "event: done" in text and "24 hours" in text

    assert client.post("/api/v1/feedback", json={"query_id": body["query_id"], "rating": 1}).status_code == 204

    # dataset generation (heuristic, offline)
    job = client.post("/api/v1/datasets/generate", json={"collection": "policies", "method": "heuristic",
                                                          "max_chunks": 50, "test_ratio": 0.5}).json()
    ds = wait(client, job["id"])["result"]
    assert ds["num_items"] > 5
    ds_id = ds["id"]

    # retrieval evaluation
    run = wait(client, client.post("/api/v1/eval/retrieval", json={"dataset_id": ds_id, "split": "all"}).json()["id"])["result"]
    assert run["metrics"]["num_queries"] > 0
    assert 0 <= run["metrics"]["mrr"] <= 1

    # generation evaluation incl. out-of-domain refusals
    run = wait(client, client.post("/api/v1/eval/generation", json={"dataset_id": ds_id, "max_items": 10}).json()["id"])["result"]
    assert "refusal_accuracy" in run["metrics"]

    # optimization over chunk sizes, then apply
    opt = wait(client, client.post("/api/v1/optimize", json={
        "dataset_id": ds_id, "chunk_sizes": [250, 400], "rerank_options": [False], "candidate_ks": [10, 20],
    }).json()["id"])["result"]
    assert opt["details"] and opt["metrics"]["best"]
    applied = client.post(f"/api/v1/optimize/{opt['id']}/apply").json()
    for jid in applied["reindex_jobs"]:
        wait(client, jid)
    assert client.get("/api/v1/config/retrieval").json()["chunk_size"] == opt["metrics"]["best"]["chunk_size"]

    # manual dataset import + analytics + models
    r = client.post("/api/v1/datasets/manual", json={"collection": "policies", "items": [
        {"question": "What is the capital of Japan?", "category": "out_of_document"},
        {"question": "Is MFA required for remote access?", "source": "policy.md", "reference": "Multi-factor authentication is required for all remote access."},
    ]})
    assert r.status_code == 201
    a = client.get("/api/v1/analytics/queries").json()
    assert a["summary"]["total"] >= 3 and a["summary"]["positive"] >= 1
    assert {m["kind"] for m in client.get("/api/v1/models").json()} == {"embedding", "reranker", "llm"}

    # chunks & delete
    assert client.get(f"/api/v1/documents/{doc_id}/chunks").json()
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 204
    results = client.post("/api/v1/search", json={"query": "backups", "collections": ["policies"]}).json()["results"]
    assert all(r["doc_id"] != doc_id for r in results)


def test_api_key(monkeypatch, client):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.get("/api/v1/collections").status_code == 401
    assert client.get("/api/v1/collections", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/api/v1/health").status_code == 200
