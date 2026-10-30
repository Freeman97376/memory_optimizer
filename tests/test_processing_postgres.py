"""Real transactions and process exits; evidence inputs remain synthetic."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from memory_optimizer.api.app import create_app
from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.bootstrap import build_services
from memory_optimizer.domain import ProcessingBusy, ProcessingUnavailable, SearchQuery
from memory_optimizer.models import AddRequest
from memory_optimizer.processing.selection import select_windows
from memory_optimizer.processing.types import ProcessingSpec, SelectionConfig
from memory_optimizer.storage.postgres import migrate

from .conftest import counts, request_data

pytestmark = pytest.mark.integration


def command():
    return ingest_command(AddRequest.model_validate(request_data()))


def state(db):
    return db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchone()[0]


def expire(db):
    # Advance only the test lease's expiry, not the host/database clock.
    db.execute(
        "UPDATE processing_runs SET lease_until = clock_timestamp() - interval '1 second' "
        "WHERE owner IS NOT NULL"
    )


@pytest.fixture
def services(db_settings):
    result = build_services(db_settings)
    result.lifecycle.open()
    try:
        yield result
    finally:
        result.lifecycle.close()


class Interrupt(BaseException):
    pass


def save_raw_only(services, monkeypatch):
    flow = services.add.processing
    with monkeypatch.context() as patch:

        def interrupt(*args):
            raise Interrupt()

        patch.setattr(type(flow), "finish", interrupt)
        with pytest.raises(Interrupt):
            services.add.add(command())
    return flow.repository


def test_terminal_outcomes_are_explicit_and_all_raw_material_is_searchable(client, db):
    payload = request_data(
        messages=[
            {"role": "user", "content": "unselectedtoken welcome"},
            {"role": "assistant", "content": "error; fixed but not tested"},
        ]
    )
    assert client.post("/add", json=payload).status_code == 200
    run = state(db)
    assert (run["stage"], run["outcome"], run["reason"], run["model_calls"]) == (
        "complete",
        "raw_degraded",
        "processor_unavailable",
        0,
    )
    assert run["candidate_count"] == run["selected_count"] == 1
    assert run["owner"] is None and run["lease_until"] is None
    assert db.execute("SELECT status FROM processing_windows").fetchall() == [("deferred",)]
    response = client.post("/search", json={"user_id": "project-a", "query": "unselectedtoken"})
    assert response.json()["data"][0]["content"] == "unselectedtoken welcome"
    assert client.post("/add", json=payload).status_code == 200
    assert state(db) == run


def test_zero_windows_is_normal_raw_success(client, db):
    assert (
        client.post(
            "/add", json=request_data(messages=[{"role": "user", "content": "hello world"}])
        ).status_code
        == 200
    )
    run = state(db)
    assert (run["outcome"], run["reason"], run["candidate_count"], run["selected_count"]) == (
        "raw_not_selected",
        None,
        0,
        0,
    )


@pytest.mark.parametrize(
    "failed_stage,remaining_stage,window_status",
    [
        ("raw_ready", None, None),
        ("selected", "raw_ready", None),
        ("complete", "selected", "pending"),
    ],
)
def test_commit_failures_preserve_exact_recovery_boundary(
    client, db, failed_stage, remaining_stage, window_status
):
    # Deferred trigger fails at COMMIT, after the stage's writes have executed.
    db.execute("""CREATE FUNCTION reject_stage() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'private-sentinel'; END $$""")
    db.execute(
        "CREATE CONSTRAINT TRIGGER reject_stage AFTER INSERT OR UPDATE ON processing_runs "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        f"WHEN (NEW.stage = '{failed_stage}') EXECUTE FUNCTION reject_stage()"
    )
    response = client.post("/add", json=request_data())
    assert response.status_code == 503 and response.json() == {"error": "storage_unavailable"}
    assert "private-sentinel" not in response.text
    if remaining_stage is None:
        assert counts(db) == (0, 0, 0)
        assert db.execute("SELECT count(*) FROM processing_runs").fetchone()[0] == 0
    else:
        assert counts(db) == (1, 2, 2)
        assert state(db)["stage"] == remaining_stage
        hits = client.post("/search", json={"query": "cache", "user_id": "project-a"})
        assert len(hits.json()["data"]) == 2
    statuses = db.execute("SELECT status FROM processing_windows").fetchall()
    assert statuses == ([] if window_status is None else [(window_status,)])
    db.execute("DROP TRIGGER reject_stage ON processing_runs")
    expire(db)
    assert client.post("/add", json=request_data()).status_code == 200
    assert state(db)["stage"] == "complete" and counts(db) == (1, 2, 2)


def test_concurrent_claim_has_single_owner_and_old_owner_is_fenced(services, db, monkeypatch):
    repo = save_raw_only(services, monkeypatch)
    batch_id = state(db)["batch_id"]

    def claim():
        try:
            return repo.claim("project-a", batch_id, 30)
        except ProcessingBusy:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: claim(), range(2)))
    assert sum(c is not None for c in claims) == 1
    old = next(c for c in claims if c is not None)
    selected = select_windows(command().messages, old.spec.selection)
    expire(db)
    # Even without a replacement owner, an expired lease cannot write.
    with pytest.raises(ProcessingBusy):
        repo.checkpoint(old, selected)
    new = repo.claim("project-a", batch_id, 30)
    assert new.fence == old.fence + 1 and new.owner != old.owner
    with pytest.raises(ProcessingBusy):
        repo.checkpoint(old, selected)
    repo.checkpoint(new, selected)
    with pytest.raises(ProcessingBusy):
        repo.finish_raw(replace(old, stage="selected"))
    assert state(db)["stage"] == "selected"
    repo.finish_raw(replace(new, stage="selected"))
    assert state(db)["stage"] == "complete"
    assert db.execute("SELECT count(*) FROM processing_windows").fetchone()[0] == 1


def test_processing_scope_is_required_for_claim_checkpoint_and_completion(
    services, db, monkeypatch
):
    repo = save_raw_only(services, monkeypatch)
    batch_id = state(db)["batch_id"]
    with pytest.raises(ProcessingUnavailable):
        repo.claim("project-b", batch_id, 30)
    lease = repo.claim("project-a", batch_id, 30)
    selection = select_windows(command().messages, lease.spec.selection)
    wrong_scope = replace(lease, user_id="project-b")
    with pytest.raises(ProcessingBusy):
        repo.checkpoint(wrong_scope, selection)
    repo.checkpoint(lease, selection)
    with pytest.raises(ProcessingBusy):
        repo.finish_raw(replace(wrong_scope, stage="selected"))
    repo.finish_raw(replace(lease, stage="selected"))
    with pytest.raises(ProcessingUnavailable):
        repo.claim("project-b", batch_id, 30)


def test_resume_uses_persisted_spec_and_terminal_replay_does_not_select(services, db, monkeypatch):
    save_raw_only(services, monkeypatch)
    original = state(db)["spec"]
    changed = replace(
        services.add.processing,
        spec=ProcessingSpec(
            selection=SelectionConfig(focus_chars=2, context_chars=0, max_windows=1)
        ),
    )
    # The new default would change the focus; the unfinished run must keep its original config.
    add = replace(services.add, processing=changed)
    add.add(command())
    assert state(db)["spec"] == original
    assert db.execute("SELECT start_offset, end_offset FROM processing_windows").fetchone() == (
        0,
        30,
    )
    before = state(db)

    def forbidden(*args):
        raise AssertionError("completed requests cannot select again")

    final = replace(add, processing=replace(changed, selectors={"coding-signals-v1": forbidden}))
    final.add(command())
    assert state(db) == before


def test_selector_limit_is_durable_and_unselected_raw_survives(services, db):
    limited = replace(
        services.add.processing,
        spec=ProcessingSpec(selection=SelectionConfig(max_windows=1, context_chars=0)),
    )
    data = ingest_command(
        AddRequest.model_validate(
            request_data(
                messages=[
                    {"role": "user", "content": "error"},
                    {"role": "assistant", "content": "error omittedtoken"},
                ]
            )
        )
    )
    replace(services.add, processing=limited).add(data)
    run = state(db)
    assert (run["candidate_count"], run["selected_count"], run["outcome"]) == (2, 1, "raw_degraded")
    assert (
        services.search.search(SearchQuery("omittedtoken", "project-a"))[0].content
        == "error omittedtoken"
    )


@pytest.mark.parametrize("invalid", ["flow", "selector", "limits", "missing_run"])
def test_incompatible_unfinished_state_is_not_silently_reinterpreted(
    services, db, monkeypatch, db_settings, invalid
):
    save_raw_only(services, monkeypatch)
    run = state(db)
    spec = run["spec"]
    if invalid == "missing_run":
        db.execute("DELETE FROM processing_runs")
    else:
        if invalid == "limits":
            spec["selection"]["max_windows"] = False
        else:
            spec[f"{invalid}_version"] = "unknown-v99"
        db.execute("UPDATE processing_runs SET spec = %s", (Jsonb(spec),))
    with TestClient(create_app(db_settings), headers={"X-Api-Key": "local-test-key"}) as c:
        response = c.post("/add", json=request_data())
        assert response.status_code == 503
        assert response.json() == {"error": "processing_unavailable"}
        assert response.headers["Retry-After"] == "1"
    assert counts(db) == (1, 2, 2)
    if invalid != "missing_run":
        assert state(db)["stage"] == "raw_ready" and state(db)["spec"] == spec


def test_live_lease_returns_retryable_failure_without_false_success(
    services, db, monkeypatch, db_settings
):
    repo = save_raw_only(services, monkeypatch)
    repo.claim("project-a", state(db)["batch_id"], 30)
    no_wait = replace(
        services,
        add=replace(services.add, processing=replace(services.add.processing, wait_seconds=0)),
    )
    with TestClient(
        create_app(db_settings, services=no_wait), headers={"X-Api-Key": "local-test-key"}
    ) as c:
        response = c.post("/add", json=request_data())
        assert response.status_code == 503 and response.headers["Retry-After"] == "1"
        assert state(db)["stage"] == "raw_ready"
        assert c.post("/add", json=request_data(session_id="changed")).status_code == 409


@pytest.mark.parametrize("mutation", ["missing", "gap", "future", "old_checksum"])
def test_readiness_checks_entire_migration_history(client, db, db_settings, mutation):
    if mutation == "missing":
        db.execute("DELETE FROM schema_migrations WHERE version = 2")
    elif mutation == "gap":
        db.execute("DELETE FROM schema_migrations WHERE version = 1")
    elif mutation == "future":
        db.execute("INSERT INTO schema_migrations(version, checksum) VALUES(99, 'future')")
    else:
        db.execute("UPDATE schema_migrations SET checksum = 'changed' WHERE version = 1")
    assert client.get("/health").status_code == 503
    # Missing history cannot be silently inferred from existing tables either.
    with pytest.raises((RuntimeError, psycopg.Error)):
        migrate(db_settings.database)
    assert db.execute("SELECT count(*) FROM processing_runs").fetchone()[0] == 0


@pytest.mark.parametrize(
    "point,expected", [("raw", "raw_ready"), ("selected", "selected"), ("complete", "complete")]
)
def test_real_process_exit_and_fresh_process_resume(db_settings, db, point, expected):
    env = os.environ.copy()
    env.update(
        {
            "MEMORY_DATABASE_URL": db_settings.database_url.get_secret_value(),
            "MEMORY_DATABASE_SCHEMA": db_settings.schema,
            "MEMORY_API_KEY": "local-test-key",
        }
    )
    helper = Path(__file__).parent / "helpers/processing_crash.py"
    first = subprocess.run(
        [sys.executable, str(helper), point], env=env, capture_output=True, timeout=15
    )
    assert first.returncode == 17, first.stderr.decode()
    run = state(db)
    assert run["stage"] == expected
    old_ids = db.execute("SELECT id FROM processing_windows ORDER BY ordinal").fetchall()
    expire(db)
    resume = "resume" if point == "raw" else "resume-no-selection"
    second = subprocess.run(
        [sys.executable, str(helper), resume], env=env, capture_output=True, timeout=15
    )
    assert second.returncode == 0, second.stderr.decode()
    assert json.loads(second.stdout)["reused"] is True
    assert state(db)["stage"] == "complete" and state(db)["spec"] == run["spec"]
    assert state(db)["id"] == run["id"]
    if old_ids:
        assert (
            db.execute("SELECT id FROM processing_windows ORDER BY ordinal").fetchall() == old_ids
        )
    if point == "complete":
        assert state(db) == run
    assert counts(db) == (1, 2, 2)


def test_sql_cannot_accept_an_incomplete_terminal_state(services, db, monkeypatch):
    save_raw_only(services, monkeypatch)
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("UPDATE processing_runs SET stage = 'complete', finished_at = now()")
    assert state(db)["stage"] == "raw_ready"


@pytest.mark.parametrize("stage", ["checkpoint", "finish"])
def test_lease_expiring_inside_transaction_rolls_back_all_stage_writes(
    services, db, monkeypatch, stage
):
    repo = save_raw_only(services, monkeypatch)
    lease = repo.claim("project-a", state(db)["batch_id"], 30)
    selection = select_windows(command().messages, lease.spec.selection)
    if stage == "finish":
        repo.checkpoint(lease, selection)
        lease = replace(lease, stage="selected")
    before = state(db)
    db.execute("""CREATE FUNCTION expire_owner() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            UPDATE processing_runs SET lease_until = clock_timestamp() - interval '1 second'
            WHERE id = NEW.run_id;
            RETURN NEW;
        END $$""")
    event = "INSERT" if stage == "checkpoint" else "UPDATE"
    db.execute(
        f"CREATE TRIGGER expire_owner AFTER {event} ON processing_windows "
        "FOR EACH ROW EXECUTE FUNCTION expire_owner()"
    )
    with pytest.raises(ProcessingBusy):
        if stage == "checkpoint":
            repo.checkpoint(lease, selection)
        else:
            repo.finish_raw(lease)
    assert state(db) == before
    expected = [] if stage == "checkpoint" else [("pending",)]
    assert db.execute("SELECT status FROM processing_windows").fetchall() == expected
