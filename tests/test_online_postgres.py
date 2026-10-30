"""Full HTTP Add->paid adapter->publish->Search using a fake external transport."""

import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from memory_optimizer.api.app import create_app
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.llm.openai import OpenAITransport
from memory_optimizer.llm.types import Limits
from memory_optimizer.online import build_online_services
from memory_optimizer.storage.llm import PostgresCallLedger
from tests.helpers.s3_material import FIRST
from tests.test_llm import response
from tests.test_online import wire_candidate

pytestmark = pytest.mark.integration


def test_online_http_commit_visibility_replay_scope_and_restart(db_settings, db):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        content = (
            wire_candidate()
            if body["response_format"]["json_schema"]["name"] == "memory_extract"
            else dict(assessments={"c0": "independent"}, relations=[])
        )
        return httpx.Response(
            200,
            json=response(
                choices=[dict(finish_reason="stop", message=dict(content=canonical(content)))]
            ),
        )

    def services():
        return build_online_services(
            db_settings,
            api_key=SecretStr("fake"),
            transport=OpenAITransport(SecretStr("fake"), transport=httpx.MockTransport(handler)),
        )

    payload = dict(
        request_id="online-1",
        user_id="online-user",
        session_id="online-session",
        messages=[dict(role="user", content=FIRST)],
    )
    app = services()
    with TestClient(
        create_app(db_settings, services=app), headers={"Authorization": "Bearer local-test-key"}
    ) as client:
        assert client.post("/add", json=payload).status_code == 200
        assert len(requests) == 2
        hits = client.post(
            "/search", json=dict(query="lock", user_id="online-user", top_k=1)
        ).json()["data"]
        assert hits[0]["content"].startswith("经历证据")
        assert "single-thread passed" in hits[0]["content"]
        assert client.post("/add", json=payload).status_code == 200 and len(requests) == 2
        assert (
            client.post("/search", json=dict(query="lock", user_id="other-user")).json()["data"]
            == []
        )
        summary = PostgresCallLedger(app.lifecycle.transaction).summary(Limits().campaign)
        assert summary["calls"] == 2 and summary["successful_calls"] == 2
        from memory_optimizer.api.mapping import ingest_command
        from memory_optimizer.models import AddRequest
        from memory_optimizer.storage.relations_report import PostgresS3ReportRepository

        reader = PostgresS3ReportRepository(app.lifecycle.transaction)
        reader.FLOW_VERSION = "s3-openai-v1"
        report = reader.report_request(ingest_command(AddRequest.model_validate(payload)))
        assert report["run"]["model_calls"] is None
        assert report["publication"]["model_calls"] is None
        assert report["provider_accounting"]["source"] == "llm_calls"
        assert db.execute("SELECT body->'model_calls' FROM s3_publications").fetchone() == (None,)
        assert client.post("/add", json=payload | {"request_id": "online-2"}).status_code == 200
        assert len(requests) == 3  # extraction cached, new frozen relation snapshot is not
    restored = services()
    with TestClient(
        create_app(db_settings, services=restored),
        headers={"Authorization": "Bearer local-test-key"},
    ) as client:
        assert client.post("/add", json=payload).status_code == 200 and len(requests) == 3
        assert client.get("/health").status_code == 200
    assert db.execute("SELECT count(*) FROM llm_calls").fetchone() == (3,)


@pytest.mark.parametrize("bad", ["refused", "malformed", "auth"])
def test_provider_failures_preserve_source_without_fabricated_episode(db_settings, db, bad):
    requests = []

    def handler(request):
        requests.append(request)
        if bad == "auth":
            return httpx.Response(401)
        message = dict(refusal="no") if bad == "refused" else dict(content="not-json")
        return httpx.Response(
            200, json=response(choices=[dict(finish_reason="stop", message=message)])
        )

    app = build_online_services(
        db_settings,
        api_key=SecretStr("fake"),
        transport=OpenAITransport(SecretStr("fake"), transport=httpx.MockTransport(handler)),
    )
    with TestClient(
        create_app(db_settings, services=app), headers={"Authorization": "Bearer local-test-key"}
    ) as client:
        payload = dict(
            request_id="failure",
            user_id="user",
            session_id="s",
            messages=[dict(role="user", content=FIRST)],
        )
        assert client.post("/add", json=payload).status_code == 200
        assert 1 <= len(requests) <= 4
        assert db.execute("SELECT content FROM messages").fetchone() == (FIRST,)
        assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (0,)
        hits = client.post("/search", json=dict(query="lock", user_id="user")).json()["data"]
        assert hits and not hits[0]["content"].startswith("经历证据")
        calls = len(requests)
        assert client.post("/add", json=payload).status_code == 200 and len(requests) == calls


def test_global_call_cap_stops_second_stage_without_losing_raw(db_settings, db):
    class Transport:
        calls = 0

        async def complete(self, body, timeout):
            self.calls += 1
            from memory_optimizer.llm.types import Result

            return Result("complete", canonical(wire_candidate()), 100, 100)

    transport = Transport()
    app = build_online_services(
        db_settings,
        api_key=SecretStr("fake"),
        transport=transport,
        limits=replace(Limits(), max_calls=1),
    )
    from memory_optimizer.domain import IngestCommand, SourceMessage

    app.lifecycle.open()
    try:
        app.add.add(IngestCommand("limit", "scope", "s", (SourceMessage("user", FIRST),)))
        assert transport.calls == 1
        assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM messages").fetchone() == (1,)
    finally:
        app.lifecycle.close()


def test_feature_only_add_reaches_online_extraction_and_publication(db_settings, db):
    source = "Added optional CSV export for batch processing."
    calls = []

    def handler(request):
        body = json.loads(request.content)
        phase = body["response_format"]["json_schema"]["name"]
        calls.append(phase)
        content = dict(assessments={"c0": "independent"}, relations=[])
        if phase == "memory_extract":
            content = dict(
                episodes=[
                    dict(
                        candidate=dict(
                            problem=None,
                            conditions=[],
                            objects=[],
                            attempts=[
                                dict(
                                    action=dict(text=source, evidence=[dict(evidence_id=0)]),
                                    kind="reported_change",
                                    event_time=None,
                                    before=None,
                                    after=None,
                                    object_refs=[],
                                    observations=[],
                                )
                            ],
                        ),
                        time_proposals=[],
                    )
                ]
            )
        return httpx.Response(
            200,
            json=response(
                choices=[dict(finish_reason="stop", message=dict(content=canonical(content)))]
            ),
        )

    app = build_online_services(
        db_settings,
        api_key=SecretStr("fake"),
        transport=OpenAITransport(SecretStr("fake"), transport=httpx.MockTransport(handler)),
    )
    with TestClient(
        create_app(db_settings, services=app), headers={"Authorization": "Bearer local-test-key"}
    ) as client:
        payload = dict(
            request_id="feature-only",
            user_id="feature-user",
            session_id="s",
            messages=[dict(role="user", content=source)],
        )
        assert client.post("/add", json=payload).status_code == 200
        assert calls == ["memory_extract", "memory_relation"]
        hit = client.post(
            "/search", json=dict(query="CSV", user_id="feature-user", top_k=1)
        ).json()["data"][0]
        assert hit["content"].startswith("经历证据") and source in hit["content"]
        assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (1,)
        assert db.execute("SELECT content FROM messages").fetchone() == (source,)
