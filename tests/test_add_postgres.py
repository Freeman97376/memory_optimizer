import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from memory_optimizer.api.app import create_app
from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.bootstrap import build_services
from memory_optimizer.domain import RequestConflict, StorageUnavailable
from memory_optimizer.models import AddRequest
from memory_optimizer.storage.postgres import PostgresStore, migrate

from .conftest import counts, request_data

pytestmark = pytest.mark.integration


def search(client, user="project-a", query="cache", top_k=100):
    result = client.post("/search", json={"user_id": user, "query": query, "top_k": top_k})
    assert result.status_code == 200, result.text
    return result.json()["data"]


def test_add_is_durable_searchable_and_lossless(client, db):
    data = request_data()
    result = client.post("/add", json=data)
    assert result.status_code == 200
    assert result.json() == {
        "success": True,
        "request_id": data["request_id"],
        "user_id": data["user_id"],
        "session_id": data["session_id"],
    }
    raw = db.execute(
        "SELECT ordinal, role, content, source_timestamp FROM messages ORDER BY ordinal"
    ).fetchall()
    assert raw == [
        (i, m["role"], m["content"], m.get("timestamp")) for i, m in enumerate(data["messages"])
    ]
    assert {hit["content"] for hit in search(client)} == {m["content"] for m in data["messages"]}
    assert counts(db) == (1, 2, 2)
    assert client.get("/health", headers={"Authorization": "wrong"}).status_code == 200


def test_retry_is_stable_and_conflicts_do_not_overwrite(client, db):
    data = request_data()
    assert client.post("/add", json=data).status_code == 200
    original = search(client)
    reordered = dict(reversed(list(data.items())))
    assert (
        client.post(
            "/add", content=json.dumps(reordered), headers={"Content-Type": "application/json"}
        ).status_code
        == 200
    )
    assert search(client) == original
    for change in (
        {"session_id": "other"},
        {"messages": list(reversed(data["messages"]))},
        {"messages": [{"role": "user", "content": "new cache"}]},
    ):
        assert client.post("/add", json=data | change).status_code == 409
    assert counts(db) == (1, 2, 2)
    assert search(client) == original


def test_scopes_are_isolated_but_sessions_are_not_search_scopes(client, db):
    assert client.post("/add", json=request_data()).status_code == 200
    assert not search(client, "project-b")
    assert (
        client.post(
            "/add",
            json=request_data(
                user_id="project-b", messages=[{"role": "user", "content": "cache only-b"}]
            ),
        ).status_code
        == 200
    )
    assert {hit["content"] for hit in search(client, "project-b")} == {"cache only-b"}
    assert not search(client, "project-a", '"only-b"')
    assert (
        client.post(
            "/add",
            json=request_data(
                request_id="req-2",
                session_id="session-2",
                messages=[{"role": "user", "content": "cache session two"}],
            ),
        ).status_code
        == 200
    )
    assert len(search(client)) == 3
    assert counts(db) == (3, 4, 4)


def test_new_service_instance_recovers_persistent_ids(db_settings):
    with TestClient(create_app(db_settings), headers={"X-Api-Key": "local-test-key"}) as first:
        assert first.post("/add", json=request_data()).status_code == 200
        before = search(first)
    with TestClient(
        create_app(db_settings), headers={"Authorization": "Token local-test-key"}
    ) as second:
        assert search(second) == before
        assert second.post("/add", json=request_data()).status_code == 200
        assert search(second) == before


def test_database_failure_rolls_back_everything_and_retry_recovers(client, db):
    db.execute("""CREATE FUNCTION fail_second_chunk() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN IF NEW.message_ordinal = 1 THEN
            RAISE EXCEPTION 'synthetic database failure private-sentinel';
        END IF; RETURN NEW; END $$""")
    db.execute(
        "CREATE TRIGGER fail_chunk BEFORE INSERT ON chunks "
        "FOR EACH ROW EXECUTE FUNCTION fail_second_chunk()"
    )
    result = client.post("/add", json=request_data())
    assert result.status_code == 503
    assert "private-sentinel" not in result.text
    assert counts(db) == (0, 0, 0)
    assert search(client) == []
    db.execute("DROP TRIGGER fail_chunk ON chunks")
    assert client.post("/add", json=request_data()).status_code == 200
    assert counts(db) == (1, 2, 2)


@pytest.mark.parametrize("conflicting", [False, True])
def test_concurrent_retries_have_one_committed_batch(db_settings, db, conflicting):
    services = [build_services(db_settings), build_services(db_settings)]
    barrier = Barrier(2)
    for service in services:
        service.lifecycle.open()

    def submit(index):
        data = request_data()
        if conflicting and index == 1:
            data["messages"][0]["content"] += " changed"
        barrier.wait(timeout=10)
        try:
            services[index].add.add(ingest_command(AddRequest.model_validate(data)))
            return "ok"
        except RequestConflict:
            return "conflict"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(submit, [0, 1]))
        assert sorted(results) == (["conflict", "ok"] if conflicting else ["ok", "ok"])
        assert counts(db) == (1, 2, 2)
    finally:
        for service in services:
            service.lifecycle.close()


def test_long_message_has_complete_source_offsets_and_bounded_results(client, db):
    text = "cache 中文🌱\r\n" * 25
    assert (
        client.post(
            "/add", json=request_data(messages=[{"role": "user", "content": text}])
        ).status_code
        == 200
    )
    assert db.execute("SELECT content FROM messages").fetchone()[0] == text
    slices = db.execute("SELECT start_offset, end_offset, content FROM chunks ORDER BY ordinal")
    recovered = ""
    covered = 0
    for start, end, content in slices:
        assert text[start:end] == content and len(content) <= 60
        assert start <= covered
        recovered += content[max(0, covered - start) :]
        covered = end
    assert recovered == text
    assert len(search(client, top_k=2)) == 2
    assert search(client, query="unfindabletoken") == []
    search(client, query="'; DROP TABLE messages; --")
    assert counts(db)[1] == 1


def test_invalid_requests_cannot_write_or_echo_private_input(client, db):
    result = client.post(
        "/add",
        json=request_data(messages=[{"role": "private-sentinel", "content": "private-sentinel"}]),
    )
    assert result.status_code == 422 and "private-sentinel" not in result.text
    assert counts(db) == (0, 0, 0)
    result = client.post("/add", json=request_data(), headers={"Authorization": "Bearer wrong"})
    assert result.status_code == 401 and counts(db) == (0, 0, 0)


def test_closed_pool_is_unavailable_not_success(client):
    client.app.state.services.lifecycle.close()
    assert client.post("/add", json=request_data()).status_code == 503
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("field", ["private-sentinel", "\ud800"])
def test_unknown_field_names_cannot_break_or_leak_through_error_response(client, db, field):
    result = client.post(
        "/add",
        content=json.dumps(request_data() | {field: "private-sentinel"}),
        headers={"Content-Type": "application/json"},
    )
    assert result.status_code == 422
    assert "private-sentinel" not in result.text
    assert counts(db) == (0, 0, 0)


def test_migration_repeat_preserves_rows_and_detects_drift(client, db_settings, db):
    assert client.post("/add", json=request_data()).status_code == 200
    migrate(db_settings.database)
    assert counts(db) == (1, 2, 2)
    db.execute("UPDATE schema_migrations SET checksum = 'changed'")
    with pytest.raises(RuntimeError, match="checksum"):
        migrate(db_settings.database)
    assert client.get("/health").status_code == 503


def test_unmigrated_schema_refuses_startup(db_settings):
    store = PostgresStore(replace(db_settings, schema=db_settings.schema + "_absent").database)
    with pytest.raises(StorageUnavailable):
        store.open()


def test_existing_p1_batch_replays_without_rewriting_ids_or_content(client, db, db_settings):
    # Recreate the pre-002 state only in this fixture's isolated schema. Seed the
    # unchanged P1 artifact, then exercise the actual additive upgrade/backfill.
    from tests.helpers.s3_material import drop_s3_fixture

    drop_s3_fixture(db)
    db.execute("DROP TABLE candidate_temporal")
    db.execute("DROP FUNCTION guard_candidate_temporal()")
    db.execute("ALTER TABLE candidate_windows DROP CONSTRAINT candidate_windows_temporal_link_key")
    db.execute("DROP TABLE coverage_bindings")
    db.execute("DROP TRIGGER protect_coverage_runtime ON processing_runs")
    db.execute("DROP FUNCTION protect_coverage_binding(), protect_coverage_runtime()")
    db.execute("DROP TABLE planning_attempts, planning_runs")
    db.execute("DROP FUNCTION protect_planning_attempt(), protect_planning_run()")
    db.execute("DROP TABLE candidate_windows, episode_candidates, extraction_attempts")
    db.execute("DROP TABLE processing_windows")
    db.execute("DROP TABLE processing_runs")
    db.execute("DELETE FROM schema_migrations WHERE version >= 2")
    fixture = json.loads((Path(__file__).parent / "fixtures/p1-add.json").read_text())
    payload = fixture["payload"]
    db.execute(
        "INSERT INTO batches "
        "(id, scope_hash, request_hash, user_id, request_id, session_id, payload_hash) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            fixture["batch_id"],
            fixture["scope_hash"],
            fixture["request_hash"],
            payload["user_id"],
            payload["request_id"],
            payload["session_id"],
            fixture["payload_hash"],
        ),
    )
    for ordinal, message in enumerate(payload["messages"]):
        db.execute(
            "INSERT INTO messages (batch_id, ordinal, role, content, source_timestamp) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                fixture["batch_id"],
                ordinal,
                message["role"],
                message["content"],
                message.get("timestamp"),
            ),
        )
    for part in fixture["chunks"]:
        db.execute(
            "INSERT INTO chunks "
            "(id, batch_id, message_ordinal, ordinal, start_offset, end_offset, content) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                part["id"],
                fixture["batch_id"],
                part["message_ordinal"],
                part["ordinal"],
                part["start"],
                part["end"],
                part["content"],
            ),
        )
    migrate(db_settings.database)
    assert db.execute("SELECT stage, outcome FROM processing_runs").fetchone() == (
        "complete",
        "raw_legacy",
    )
    before = search(client)
    assert {hit["id"] for hit in before} == {part["id"] for part in fixture["chunks"]}
    assert client.post("/add", json=payload).status_code == 200
    assert search(client) == before
    assert counts(db) == (1, 2, 2)
    assert client.post("/add", json=payload | {"session_id": "changed"}).status_code == 409
    assert search(client) == before


def test_commit_time_failure_does_not_release_success(client, db):
    db.execute("""CREATE FUNCTION reject_commit() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'deferred private-sentinel'; END $$""")
    db.execute(
        "CREATE CONSTRAINT TRIGGER fail_commit AFTER INSERT ON batches "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_commit()"
    )
    failure = client.post("/add", json=request_data())
    assert failure.status_code == 503
    assert failure.json() == {"error": "storage_unavailable"}
    assert counts(db) == (0, 0, 0)
    db.execute("DROP TRIGGER fail_commit ON batches")
    assert client.post("/add", json=request_data()).status_code == 200
    assert counts(db) == (1, 2, 2)
