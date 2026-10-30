"""Real isolated PostgreSQL checkpoints and commit/lease failures, no model calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.bootstrap import build_services
from memory_optimizer.domain import (
    ProcessingBusy,
    ProcessingUnavailable,
    SearchQuery,
    StorageUnavailable,
)
from memory_optimizer.models import AddRequest
from memory_optimizer.processing.checkpoints import RunBudget, diagnostic_ranges
from memory_optimizer.processing.plan_types import ExecutionDiagnostic, PlanLimits, PlanningSpec
from memory_optimizer.processing.planning import build_plan, select_baseline
from memory_optimizer.processing.types import SourceSpan
from memory_optimizer.storage import postgres
from memory_optimizer.storage.planning import PostgresPlanningRepository
from memory_optimizer.storage.postgres import migrate

from .conftest import request_data

pytestmark = pytest.mark.integration
SPEC = PlanningSpec(PlanLimits(10, 2, 2, 1000, 100, 3))


@pytest.fixture
def setup(db_settings):
    services = build_services(db_settings)
    services.lifecycle.open()
    command = ingest_command(
        AddRequest.model_validate(
            request_data(
                messages=[
                    {"role": "user", "content": "error cache🌱\r\n", "timestamp": 7},
                    {"role": "assistant", "content": "fixed and tested"},
                ]
            )
        )
    )
    receipt = services.add.add(command)
    repo = PostgresPlanningRepository(services.lifecycle.transaction)
    try:
        yield repo, services, command, receipt.batch_id
    finally:
        services.lifecycle.close()


def initialize(setup, budget=None):
    repo, _, command, batch = setup
    return repo.initialize(command.user_id, batch, SPEC, budget or RunBudget(120, 4))


def prepare(setup, budget=None):
    saved = initialize(setup, budget)
    repo = setup[0]
    lease = repo.claim(saved.user_id, saved.batch_id, 30)
    ticket = repo.reserve(lease, "planning", str(uuid4())).ticket
    plan = build_plan(saved.messages, saved.spec)
    diagnostic = ExecutionDiagnostic(
        "planning", "completed", diagnostic_ranges(saved, "planning"), plan.fingerprint
    )
    return saved, lease, ticket, plan, diagnostic


def planned(setup):
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    repo = setup[0]
    repo.save_plan(lease, ticket, plan, diagnostic)
    saved = repo.load(saved.user_id, saved.batch_id)
    ticket = repo.reserve(lease, "selection", str(uuid4())).ticket
    selection = select_baseline(plan, saved.messages)
    diagnostic = ExecutionDiagnostic(
        "selection", "completed", diagnostic_ranges(saved, "selection", selection), plan.fingerprint
    )
    return saved, lease, ticket, selection, diagnostic


def expire(db):
    db.execute(
        "UPDATE planning_runs SET lease_until = clock_timestamp() - interval "
        "'1 second' WHERE owner IS NOT NULL"
    )


def rows(db):
    return (
        db.execute("SELECT row_to_json(r) FROM planning_runs r").fetchall(),
        db.execute("SELECT row_to_json(a) FROM planning_attempts a ORDER BY number").fetchall(),
    )


def test_initialize_is_idempotent_concurrent_and_never_changes_legacy_receipt(setup, db):
    repo, services, command, batch = setup
    legacy = db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchone()[0]
    assert db.execute("SELECT count(*) FROM planning_runs").fetchone()[0] == 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        saved = list(pool.map(lambda _: initialize(setup), range(2)))
    assert saved[0] == saved[1]
    assert saved[0].deadline_at - saved[0].created_at == timedelta(seconds=120)
    assert services.add.add(command).reused is True
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchone()[0] == legacy
    assert services.search.search(SearchQuery("cache", command.user_id))
    for spec, budget in [
        (replace(SPEC, limits=replace(SPEC.limits, max_windows=1)), RunBudget(120, 4)),
        (SPEC, RunBudget(121, 4)),
    ]:
        with pytest.raises(ProcessingUnavailable):
            repo.initialize(command.user_id, batch, spec, budget)
    assert initialize(setup) == saved[0]


def test_plan_final_and_diagnostics_roundtrip_and_exact_retries_do_not_rewrite(
    setup, db, db_settings
):
    repo = setup[0]
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    repo.save_plan(lease, ticket, plan, diagnostic)
    before = rows(db)
    repo.save_plan(lease, ticket, plan, diagnostic)
    assert rows(db) == before
    saved = repo.load(saved.user_id, saved.batch_id)
    final = select_baseline(saved.plan, saved.messages)
    ticket = repo.reserve(lease, "selection", str(uuid4())).ticket
    d = ExecutionDiagnostic(
        "selection", "completed", diagnostic_ranges(saved, "selection", final), plan.fingerprint
    )
    repo.save_selection(lease, ticket, final, d)
    before = rows(db)
    repo.save_selection(lease, ticket, final, d)
    assert rows(db) == before
    fresh = build_services(db_settings)
    fresh.lifecycle.open()
    try:
        loaded = PostgresPlanningRepository(fresh.lifecycle.transaction).load(
            saved.user_id, saved.batch_id
        )
    finally:
        fresh.lifecycle.close()
    assert loaded.stage == "selection_ready" and loaded.plan == plan and loaded.selection == final
    assert [a.diagnostic.outcome for a in loaded.attempts] == ["completed", "completed"]
    assert loaded.deadline_at == saved.deadline_at
    with pytest.raises(ProcessingUnavailable, match="already ready"):
        repo.claim(saved.user_id, saved.batch_id, 30)


def test_reservation_retry_uses_one_slot_and_different_key_cannot_duplicate_inflight_work(
    setup, db
):
    saved = initialize(setup)
    repo = setup[0]
    lease = repo.claim(saved.user_id, saved.batch_id, 30)
    key = str(uuid4())
    first = repo.reserve(lease, "planning", key)
    before = rows(db)
    assert repo.reserve(lease, "planning", key) == first
    assert rows(db) == before
    with pytest.raises(ProcessingBusy):
        repo.reserve(lease, "planning", str(uuid4()))
    assert len(repo.load(saved.user_id, saved.batch_id).attempts) == 1


def test_abandoned_attempt_consumes_budget_fences_old_owner_and_preserves_deadline(setup, db):
    saved, old, ticket, plan, diagnostic = prepare(setup)
    repo = setup[0]
    expire(db)
    with pytest.raises(ProcessingBusy):
        repo.save_plan(old, ticket, plan, diagnostic)
    new = repo.claim(saved.user_id, saved.batch_id, 30)
    assert new.fence == old.fence + 1 and new.owner != old.owner
    after = repo.load(saved.user_id, saved.batch_id)
    assert after.attempts[0].diagnostic.outcome == "interrupted"
    assert after.attempts[0].diagnostic.inspected == ()  # No durable progress is known.
    assert after.deadline_at == saved.deadline_at
    with pytest.raises(ProcessingBusy):
        repo.save_plan(old, ticket, plan, diagnostic)
    with pytest.raises(ProcessingUnavailable):
        repo.reserve(new, "planning", ticket.reservation_id)
    next_ticket = repo.reserve(new, "planning", str(uuid4())).ticket
    assert next_ticket.number == 2 and next_ticket.input_hash == ticket.input_hash
    repo.renew(new, 60)
    assert repo.load(saved.user_id, saved.batch_id).deadline_at == saved.deadline_at
    repo.save_plan(new, next_ticket, plan, diagnostic)


def test_concurrent_claim_has_single_owner(setup):
    saved = initialize(setup)
    repo = setup[0]

    def claim(_):
        try:
            return repo.claim(saved.user_id, saved.batch_id, 30)
        except ProcessingBusy:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, range(2)))
    assert sum(value is not None for value in claims) == 1


def test_concurrent_reservations_have_one_durable_slot(setup):
    saved = initialize(setup)
    repo = setup[0]
    lease = repo.claim(saved.user_id, saved.batch_id, 30)

    def reserve(_):
        try:
            return repo.reserve(lease, "planning", str(uuid4()))
        except ProcessingBusy:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = list(pool.map(reserve, range(2)))
    assert sum(value is not None for value in attempts) == 1
    assert len(repo.load(saved.user_id, saved.batch_id).attempts) == 1


def test_selection_recovery_reuses_saved_plan_and_consumes_abandoned_attempt(setup, db):
    saved, old, abandoned, final, diagnostic = planned(setup)
    repo = setup[0]
    plan_row = db.execute("SELECT plan_body,plan_hash,planned_at,deadline_at FROM planning_runs")
    before = plan_row.fetchone()
    expire(db)
    new = repo.claim(saved.user_id, saved.batch_id, 30)
    loaded = repo.load(saved.user_id, saved.batch_id)
    assert loaded.plan == saved.plan and loaded.selection is None
    assert loaded.attempts[-1].diagnostic == ExecutionDiagnostic(
        "selection", "interrupted", (), saved.plan.fingerprint
    )
    with pytest.raises(ProcessingBusy):
        repo.save_selection(old, abandoned, final, diagnostic)
    ticket = repo.reserve(new, "selection", str(uuid4())).ticket
    assert ticket.number == 3 and ticket.input_hash == abandoned.input_hash
    repo.save_selection(new, ticket, final, diagnostic)
    assert (
        db.execute(
            "SELECT plan_body,plan_hash,planned_at,deadline_at FROM planning_runs"
        ).fetchone()
        == before
    )
    assert repo.load(saved.user_id, saved.batch_id).selection == final


def test_attempt_budget_is_shared_by_both_phases_and_diagnostics_are_durable(setup, db):
    saved, lease, first, plan, completed = prepare(setup, RunBudget(120, 2))
    repo = setup[0]
    failed = ExecutionDiagnostic("planning", "failed", (SourceSpan(0, 0, 2),))
    repo.save_diagnostic(lease, first, failed)
    before = rows(db)
    repo.save_diagnostic(lease, first, failed)
    assert rows(db) == before
    second = repo.reserve(lease, "planning", str(uuid4())).ticket
    repo.save_plan(lease, second, plan, completed)
    assert repo.reserve(lease, "selection", str(uuid4())) is None
    after = repo.load(saved.user_id, saved.batch_id)
    assert (
        after.plan == plan
        and after.selection is None
        and after.closed_reason == "attempts_exhausted"
    )
    assert len(after.attempts) == 2
    expire(db)
    new = repo.claim(saved.user_id, saved.batch_id, 30)
    assert repo.reserve(new, "selection", str(uuid4())) is None
    assert repo.load(saved.user_id, saved.batch_id).deadline_at == saved.deadline_at


@pytest.mark.parametrize("phase", ["planning", "selection"])
def test_commit_failure_rolls_back_result_diagnostic_and_ready_marker(setup, db, phase):
    prepared = prepare(setup) if phase == "planning" else planned(setup)
    saved, lease, ticket, value, diagnostic = prepared
    repo = setup[0]
    before = rows(db)
    db.execute(
        "CREATE FUNCTION fail_plan_commit() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'private-sentinel'; END $$"
    )
    stage = "plan_ready" if phase == "planning" else "selection_ready"
    db.execute(
        "CREATE CONSTRAINT TRIGGER fail_plan_commit AFTER UPDATE ON "
        "planning_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        + f"WHEN (NEW.stage = '{stage}') EXECUTE FUNCTION fail_plan_commit()"
    )
    save = repo.save_plan if phase == "planning" else repo.save_selection
    with pytest.raises(StorageUnavailable, match="storage operation failed"):
        save(lease, ticket, value, diagnostic)
    assert rows(db) == before
    db.execute("DROP TRIGGER fail_plan_commit ON planning_runs")
    save(lease, ticket, value, diagnostic)
    assert repo.load(saved.user_id, saved.batch_id).stage == stage


@pytest.mark.parametrize("operation", ["initialize", "reserve", "diagnostic"])
def test_other_commit_failures_never_leave_partial_records(setup, db, operation):
    repo = setup[0]
    if operation == "diagnostic":
        saved, lease, ticket, _, _ = prepare(setup)
    elif operation == "reserve":
        saved = initialize(setup)
        lease = repo.claim(saved.user_id, saved.batch_id, 30)
    before = rows(db)
    table = "planning_runs" if operation == "initialize" else "planning_attempts"
    event = "UPDATE" if operation == "diagnostic" else "INSERT"
    db.execute(
        "CREATE FUNCTION reject_planning_commit() RETURNS trigger LANGUAGE "
        "plpgsql AS $$ BEGIN RAISE EXCEPTION 'test rollback'; END $$"
    )
    db.execute(
        f"CREATE CONSTRAINT TRIGGER reject_planning_commit AFTER {event} ON {table} "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        "EXECUTE FUNCTION reject_planning_commit()"
    )
    with pytest.raises(StorageUnavailable):
        if operation == "initialize":
            initialize(setup)
        elif operation == "reserve":
            repo.reserve(lease, "planning", str(uuid4()))
        else:
            repo.save_diagnostic(lease, ticket, ExecutionDiagnostic("planning", "failed", ()))
    assert rows(db) == before


def test_fixed_deadline_blocks_new_work_but_allows_terminal_failure_record(setup, db):
    saved, lease, ticket, plan, diagnostic = prepare(setup, RunBudget(1, 4))
    repo = setup[0]
    db.execute("SELECT pg_sleep(1.05)")
    repo.renew(lease, 30)
    with pytest.raises(ProcessingBusy):
        repo.save_plan(lease, ticket, plan, diagnostic)
    repo.save_diagnostic(lease, ticket, ExecutionDiagnostic("planning", "deadline_exhausted", ()))
    assert repo.reserve(lease, "planning", str(uuid4())) is None
    after = repo.load(saved.user_id, saved.batch_id)
    assert after.stage == "raw_ready" and after.plan is None
    assert after.closed_reason == "deadline_exhausted" and after.deadline_at == saved.deadline_at
    expire(db)
    new = repo.claim(saved.user_id, saved.batch_id, 30)
    assert repo.reserve(new, "planning", str(uuid4())) is None
    assert initialize(setup, RunBudget(1, 4)).deadline_at == saved.deadline_at


def test_scope_lease_ticket_and_phase_mismatch_cannot_write(setup):
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    repo = setup[0]
    for action in [
        lambda: repo.load("other", saved.batch_id),
        lambda: repo.claim("other", saved.batch_id, 30),
        lambda: repo.initialize("other", saved.batch_id, SPEC, RunBudget(120, 4)),
    ]:
        with pytest.raises(ProcessingUnavailable):
            action()
    for wrong in [
        replace(lease, user_id="other"),
        replace(lease, run_id=str(uuid4())),
        replace(lease, batch_id=str(uuid4())),
        replace(lease, owner=str(uuid4())),
        replace(lease, fence=lease.fence + 1),
    ]:
        with pytest.raises(ProcessingUnavailable):
            repo.save_plan(wrong, ticket, plan, diagnostic)
    with pytest.raises(ProcessingBusy):
        repo.save_plan(lease, replace(ticket, input_hash="0" * 64), plan, diagnostic)
    with pytest.raises(ValueError):
        repo.reserve(lease, "selection", str(uuid4()))
    with pytest.raises(ValueError):
        repo.save_diagnostic(lease, ticket, diagnostic)
    assert repo.load(saved.user_id, saved.batch_id).stage == "raw_ready"


def test_invalid_plan_or_diagnostic_never_sets_ready_marker(setup):
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    repo = setup[0]
    for bad in [replace(plan, source_digest="0" * 64), replace(plan, coverage=())]:
        with pytest.raises(ValueError):
            repo.save_plan(lease, ticket, bad, diagnostic)
    with pytest.raises(ValueError):
        repo.save_plan(lease, ticket, plan, replace(diagnostic, inspected=()))
    with pytest.raises(ValueError):
        repo.save_diagnostic(
            lease, ticket, ExecutionDiagnostic("planning", "deadline_exhausted", ())
        )
    after = repo.load(saved.user_id, saved.batch_id)
    assert after.stage == "raw_ready" and after.attempts[0].diagnostic is None


def test_write_rechecks_actual_sources_after_validation(setup, db, monkeypatch):
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    repo = setup[0]
    original = repo._lock

    def change_source(conn, active):
        db.execute("UPDATE messages SET content = content || 'changed' WHERE ordinal = 0")
        return original(conn, active)

    monkeypatch.setattr(repo, "_lock", change_source)
    with pytest.raises(ProcessingUnavailable, match="input changed"):
        repo.save_plan(lease, ticket, plan, diagnostic)
    assert db.execute("SELECT stage,plan_body FROM planning_runs").fetchone() == ("raw_ready", None)


def test_expiry_after_partial_writes_rolls_back_entire_checkpoint(setup, db, monkeypatch):
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    repo = setup[0]
    original = repo._save_attempt
    before = rows(db)

    def expire_after_write(conn, *args):
        original(conn, *args)
        conn.execute(
            "UPDATE planning_runs SET lease_until = clock_timestamp() - interval '1 second'"
        )

    with monkeypatch.context() as patch:
        patch.setattr(repo, "_save_attempt", expire_after_write)
        with pytest.raises(ProcessingBusy):
            repo.save_plan(lease, ticket, plan, diagnostic)
    assert rows(db) == before
    repo.save_plan(lease, ticket, plan, diagnostic)


@pytest.mark.parametrize(
    "column,expression",
    [
        ("deadline_at", "deadline_at + interval '1 second'"),
        ("max_attempts", "max_attempts + 1"),
        ("source_hash", "repeat('0',64)"),
        ("plan_body", "'{}'::jsonb"),
    ],
)
def test_database_rejects_mutation_of_fixed_identity_and_saved_plan(setup, db, column, expression):
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    setup[0].save_plan(lease, ticket, plan, diagnostic)
    before = rows(db)
    with pytest.raises(psycopg.Error):
        db.execute(f"UPDATE planning_runs SET {column} = {expression}")
    assert rows(db) == before


@pytest.mark.parametrize("corruption", ["version", "plan_hash", "diagnostic", "attempt_fence"])
def test_corrupt_persisted_data_is_rejected_without_repair_or_v1_fallback(setup, db, corruption):
    saved, lease, ticket, plan, diagnostic = prepare(setup)
    repo, services, command, _ = setup
    repo.save_plan(lease, ticket, plan, diagnostic)
    db.execute("ALTER TABLE planning_runs DISABLE TRIGGER protect_planning_run")
    db.execute("ALTER TABLE planning_attempts DISABLE TRIGGER protect_planning_attempt")
    if corruption == "version":
        db.execute(
            "UPDATE planning_runs SET spec = jsonb_set(spec,'{flow_version}','\"future\"'::jsonb)"
        )
    elif corruption == "plan_hash":
        db.execute("UPDATE planning_runs SET plan_hash = repeat('0',64)")
    elif corruption == "attempt_fence":
        db.execute("UPDATE planning_attempts SET fence = fence + 1")
    else:
        db.execute("UPDATE planning_attempts SET diagnostic = %s", (Jsonb({"phase": "future"}),))
    before = rows(db)
    with pytest.raises(ProcessingUnavailable):
        repo.load(saved.user_id, saved.batch_id)
    assert rows(db) == before
    assert services.add.add(command).reused is True


def drop_planning_fixture(db):
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
    db.execute("DELETE FROM schema_migrations WHERE version >= 4")


def test_additive_upgrade_preserves_v1_and_failed_migration_is_atomic(
    setup, db, db_settings, monkeypatch
):
    repo, services, command, batch = setup
    before = db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall()
    sources = db.execute("SELECT * FROM messages ORDER BY ordinal").fetchall()
    drop_planning_fixture(db)
    with pytest.raises(StorageUnavailable):
        services.lifecycle.check_ready()
    original = postgres.migration_source
    with monkeypatch.context() as patch:
        patch.setattr(
            postgres,
            "migration_source",
            lambda name="001_initial.sql": (
                original(name)
                + ("\nSELECT no_such_function();" if name == "004_planning.sql" else "")
            ),
        )
        with pytest.raises(psycopg.Error):
            migrate(db_settings.database)
    assert db.execute("SELECT to_regclass('planning_runs')").fetchone()[0] is None
    assert db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall() == [
        (1,),
        (2,),
        (3,),
    ]
    migrate(db_settings.database)
    migrate(db_settings.database)
    services.lifecycle.check_ready()
    assert db.execute("SELECT * FROM messages ORDER BY ordinal").fetchall() == sources
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall() == before
    assert services.add.add(command).reused is True
    assert db.execute("SELECT count(*) FROM planning_runs").fetchone()[0] == 0
