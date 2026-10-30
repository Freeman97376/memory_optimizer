"""Replacement strategies and adapters must preserve the competition boundary."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from functools import partial
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from memory_optimizer.api.app import create_app
from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.application import AddService, SearchService, Services
from memory_optimizer.chunking import chunks
from memory_optimizer.config import Settings
from memory_optimizer.domain import (
    IngestCommand,
    IngestReceipt,
    RequestConflict,
    SearchCandidate,
    SourceMessage,
    SourceSlice,
    StorageUnavailable,
    payload_digest,
)
from memory_optimizer.models import AddRequest


class RecordingAdapter:
    def __init__(self):
        self.plans = []
        self.queries = []
        self.opened = False
        self.closed = False
        self.error = None

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def check_ready(self):
        if self.error:
            raise self.error

    def commit(self, plan):
        if self.error:
            raise self.error
        self.plans.append(plan)
        return IngestReceipt(
            plan.command.request_id,
            plan.command.user_id,
            plan.command.session_id,
            "test-batch",
            False,
        )

    def search(self, query):
        if self.error:
            raise self.error
        self.queries.append(query)
        return [SearchCandidate("second", "中文\r\n", 0.8), SearchCandidate("first", "other", 0.3)]


def command():
    return IngestCommand(" r1 ", "u", "s", (SourceMessage("user", "ab🌱cd", 0),))


def test_replacing_slicer_does_not_require_storage_or_http_changes():
    adapter = RecordingAdapter()
    service = AddService(adapter, partial(chunks, size=2, overlap=0))
    receipt = service.add(command())
    assert receipt.request_id == " r1 " and receipt.user_id == "u"
    assert len(adapter.plans) == 1
    plan = adapter.plans[0]
    assert plan.command == command()
    assert [part.content for part in plan.message_slices[0]] == ["ab", "🌱c", "d"]
    assert [(part.start, part.end) for part in plan.message_slices[0]] == [(0, 2), (2, 4), (4, 5)]
    with pytest.raises(FrozenInstanceError):
        plan.command.user_id = "different-scope"


@pytest.mark.parametrize(
    "parts",
    [
        [],
        [SourceSlice(0, 1, 5, "b🌱cd")],
        [SourceSlice(0, 0, 5, "other")],
        [SourceSlice(0, 0, 2, "ab")],
        [SourceSlice(1, 0, 5, "ab🌱cd")],
        [SourceSlice(0, 0, 5, "ab🌱cd"), SourceSlice(1, 0, 5, "ab🌱cd")],
    ],
)
def test_invalid_strategy_cannot_commit_partial_or_rewritten_source(parts):
    adapter = RecordingAdapter()
    with pytest.raises(ValueError, match="source slicer"):
        AddService(adapter, lambda text: iter(parts)).add(command())
    assert adapter.plans == []


def test_canonical_payload_hash_matches_pre_refactor_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/p1-add.json").read_text())
    parsed = AddRequest.model_validate(fixture["payload"])
    assert payload_digest(ingest_command(parsed)) == fixture["payload_hash"]


def test_http_uses_injected_services_preserving_fields_results_and_lifecycle(monkeypatch):
    def forbidden_build(settings):
        raise AssertionError("injected services must not construct PostgreSQL")

    monkeypatch.setattr("memory_optimizer.bootstrap.build_services", forbidden_build)
    adapter = RecordingAdapter()
    settings = Settings(database_url=SecretStr("postgresql://unused"), api_key=SecretStr("test"))
    services = Services(
        AddService(adapter, partial(chunks, size=3, overlap=1)),
        SearchService(adapter),
        adapter,
    )
    with TestClient(create_app(settings, services=services), headers={"X-Api-Key": "test"}) as c:
        assert adapter.opened and not adapter.closed
        payload = {
            "request_id": " r1 ",
            "user_id": " u ",
            "session_id": "s",
            "messages": [
                {"role": "assistant", "content": "🌱\r\n", "timestamp": 0},
                {"role": "user", "content": " unknown ", "timestamp": None},
            ],
        }
        response = c.post("/add", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "request_id": " r1 ",
            "user_id": " u ",
            "session_id": "s",
        }
        assert adapter.plans[0].command.messages == (
            SourceMessage("assistant", "🌱\r\n", 0),
            SourceMessage("user", " unknown ", None),
        )
        result = c.post(
            "/search", json={"query": " cache ", "user_id": " u ", "top_k": 2, "options": ["x"]}
        )
        assert result.status_code == 200
        assert result.json() == {
            "data": [
                {"id": "second", "content": "中文\r\n", "score": 0.8},
                {"id": "first", "content": "other", "score": 0.3},
            ]
        }
        query = adapter.queries[0]
        assert (query.user_id, query.query, query.top_k, query.options) == (
            " u ",
            " cache ",
            2,
            ("x",),
        )
        assert c.get("/health").json() == {"status": "ok"}

        adapter.error = RequestConflict("private-sentinel")
        assert c.post("/add", json=payload).json() == {"error": "request_id_conflict"}
        assert c.post("/add", json=payload).status_code == 409
        adapter.error = StorageUnavailable("private-sentinel")
        for path, body in (("/add", payload), ("/search", {"query": "cache", "user_id": "u"})):
            failure = c.post(path, json=body)
            assert failure.status_code == 503
            assert failure.json() == {"error": "storage_unavailable"}
            assert failure.headers["Retry-After"] == "1"
        assert c.get("/health").status_code == 503
        assert c.get("/health").json() == {"status": "unavailable"}
    assert adapter.closed


def test_application_can_import_without_http_driver_or_api_models():
    script = """
import importlib.abc
import sys
class BlockAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + '.') for item in
               ('fastapi', 'psycopg', 'psycopg_pool', 'pydantic',
                'memory_optimizer.api', 'memory_optimizer.models',
                'memory_optimizer.storage', 'memory_optimizer.config')):
            raise ImportError('unexpected adapter dependency: ' + fullname)
sys.meta_path.insert(0, BlockAdapters())
from memory_optimizer.application import AddService, SearchService
from memory_optimizer.domain import IngestCommand, SourceMessage
from memory_optimizer.ports import IngestRepository, SearchIndex
from memory_optimizer.processing.workflow import ProcessingFlow
from memory_optimizer.processing.selection import select_windows
from memory_optimizer.extraction.workflow import OfflineCandidateFlow
from memory_optimizer.extraction.validation import validate_reply
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
