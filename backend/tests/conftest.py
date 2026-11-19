"""Test configuration: offline, fast components (hashing embedder, extractive answers, no reranker)."""

import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="ekip-test-")
os.environ.update(
    {
        "EKIP_DATA_DIR": _tmp,
        "EKIP_EMBEDDING_MODEL": "hash",
        "EKIP_LLM_BACKEND": "extractive",
        "EKIP_WARMUP_MODELS": "false",
    }
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.core.config import RetrievalConfig, get_retrieval_store  # noqa: E402

# reranker would download a model; disable for tests
get_retrieval_store().set(RetrievalConfig(use_reranker=False, chunk_size=400, chunk_overlap=60))

POLICY_MD = """# Information Security Policy

Top management shall establish an information security policy that is appropriate to the purpose of the organization.
The policy shall be communicated within the organization and be available to interested parties.

## Access Control

Access to systems is granted on a least-privilege basis. Multi-factor authentication is required for all remote access.
Access rights are reviewed every quarter by the system owner.

## Incident Management

Security incidents must be reported to the security operations centre within 24 hours.
The incident response team classifies incidents by severity and coordinates containment, eradication and recovery.

## Backup Policy

Backups of production databases are taken nightly and retained for 35 days.
Backup restoration is tested every six months to verify recoverability.

## Travel Expenses

Employees may claim travel expenses within 30 days of travel. Economy class is mandatory for flights under six hours.
Hotel expenses are reimbursed up to the regional limit published by finance.
"""


@pytest.fixture(scope="session")
def policy_file(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("docs") / "policy.md"
    p.write_text(POLICY_MD)
    return p


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
