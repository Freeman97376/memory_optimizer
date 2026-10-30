"""Real PostgreSQL atomic ingest, bounded offline execution and process recovery."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest

from memory_optimizer.bootstrap import build_coverage_services, build_services
from memory_optimizer.domain import (
    ProcessingBusy,
    ProcessingUnavailable,
    SearchQuery,
    StorageUnavailable,
)
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.extraction.types import ExtractionReply
from memory_optimizer.processing import coverage_workflow
from memory_optimizer.processing.checkpoints import RunBudget
from memory_optimizer.processing.coverage_types import candidate_input
from memory_optimizer.storage import postgres
from memory_optimizer.storage.candidates import PostgresCandidateRepository
from memory_optimizer.storage.coverage import CoverageIngestRepository
from memory_optimizer.storage.coverage_candidates import BoundCandidateRepository
from memory_optimizer.storage.planning import PostgresPlanningRepository
from memory_optimizer.storage.postgres import migrate

from .helpers.coverage_material import material

pytestmark = pytest.mark.integration


class Interrupt(Exception):
    pass


@pytest.fixture
def coverage(db_settings):
    command, replay, spec = material()
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        yield command, replay, spec, services
    finally:
        services.lifecycle.close()


def expire(db):
    for table in ("processing_runs", "planning_runs"):
        db.execute(
            f"UPDATE {table} SET lease_until=clock_timestamp()-interval '1 second' "
            "WHERE owner IS NOT NULL"
        )


def state(db):
    return db.execute("SELECT stage,outcome,reason FROM processing_runs").fetchone()


def counts(db):
    return tuple(
        db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in (
            "batches",
            "messages",
            "chunks",
            "planning_runs",
            "processing_runs",
            "coverage_bindings",
            "episode_candidates",
        )
    )


def pause(coverage, monkeypatch, point="ensure_work"):
    command, _, _, services = coverage
    repo = services.add.processing.repository
    with monkeypatch.context() as patch:

        def stop(*args):
            raise Interrupt()

        patch.setattr(repo, point, stop)
        with pytest.raises(Interrupt):
            services.add.add(command)
    return repo


def test_atomic_ingest_full_candidate_path_and_terminal_replay(coverage, db):
    command, _, spec, services = coverage
    receipt = services.add.add(command)
    assert state(db) == ("complete", "raw_degraded", "offline_only")
    assert counts(db) == (1, 3, 3, 1, 1, 1, 2)
    times = db.execute(
        "SELECT b.created_at,p.created_at,p.deadline_at,r.created_at FROM batches b "
        "JOIN planning_runs p ON p.batch_id=b.id JOIN processing_runs r ON "
        "r.batch_id=b.id"
    ).fetchone()
    assert times[0] == times[1] == times[3]
    assert (times[2] - times[1]).total_seconds() == spec.budget.total_seconds
    report = PostgresCandidateRepository(services.lifecycle.transaction).report(
        command.user_id, receipt.batch_id
    )
    assert [a["status"] for a in report["attempts"]] == ["accepted"] * 3
    assert report["run"]["model_calls"] == 0
    before = db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchone()
    assert services.add.add(command).reused
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchone() == before
    assert services.search.search(SearchQuery("rollback", command.user_id))
    assert not services.search.search(SearchQuery("rollback", "other"))


@pytest.mark.parametrize("target", ["planning_runs", "processing_runs"])
def test_deferred_initialization_failure_rolls_back_raw_and_both_runs(coverage, db, target):
    db.execute(
        "CREATE FUNCTION reject_init() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        "RAISE EXCEPTION 'test init'; END $$"
    )
    db.execute(
        f"CREATE CONSTRAINT TRIGGER reject_init AFTER INSERT ON {target} "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_init()"
    )
    with pytest.raises(StorageUnavailable):
        coverage[3].add.add(coverage[0])
    assert counts(db) == (0,) * 7


def test_old_completed_receipts_are_not_reprocessed_and_pending_flows_cannot_mix(
    coverage, db, db_settings, monkeypatch
):
    command, _, _, services = coverage
    old = build_services(db_settings)
    old.lifecycle.open()
    try:
        first = old.add.add(command)
        assert services.add.add(command).batch_id == first.batch_id
        assert db.execute("SELECT count(*) FROM planning_runs").fetchone()[0] == 0
        command = replace(command, request_id="new-coverage")
        current = (command, *coverage[1:])
        pause(current, monkeypatch)
        expire(db)
        with pytest.raises(ProcessingUnavailable):
            old.add.add(command)
        assert db.execute(
            "SELECT stage FROM processing_runs WHERE batch_id<>%s", (first.batch_id,)
        ).fetchone() == ("raw_ready",)
    finally:
        old.lifecycle.close()


def test_changed_pending_config_and_other_scope_rejected(coverage, db, db_settings, monkeypatch):
    repo = pause(coverage, monkeypatch)
    expire(db)
    batch = db.execute("SELECT id::text FROM batches").fetchone()[0]
    with pytest.raises(ProcessingUnavailable):
        repo.claim("other", batch, coverage[2], 30)
    changed = replace(coverage[2], budget=RunBudget(121, 6))
    other = build_coverage_services(db_settings, spec=changed, offline_extractor=coverage[1])
    other.lifecycle.open()
    try:
        with pytest.raises(ProcessingUnavailable):
            other.add.add(coverage[0])
    finally:
        other.lifecycle.close()
    assert state(db)[0] == "raw_ready"


@pytest.mark.parametrize("mode", ["no_signal", "scan_zero", "final_zero", "attempts_one"])
def test_empty_selection_keeps_budget_gaps_distinct(db_settings, db, mode):
    command, replay, spec = material()
    limits = spec.planning.limits
    if mode == "no_signal":
        command = replace(command, messages=(replace(command.messages[0], content="hello"),))
    elif mode == "scan_zero":
        limits = replace(limits, max_scan_chars=0)
    elif mode == "final_zero":
        limits = replace(limits, max_windows=0)
    else:
        spec = replace(spec, budget=RunBudget(120, 1))
    spec = replace(spec, planning=replace(spec.planning, limits=limits))
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        services.add.add(command)
    finally:
        services.lifecycle.close()
    expected = {
        "no_signal": None,
        "scan_zero": "selection_budget",
        "final_zero": "selection_budget",
        "attempts_one": "planning_attempts_exhausted",
    }[mode]
    assert state(db)[2] == expected
    assert db.execute("SELECT count(*) FROM extraction_attempts").fetchone()[0] == 0


@pytest.mark.parametrize(
    "point",
    [
        "initialize",
        "plan",
        "selection",
        "extract",
        "final_commit",
        "candidate_reserve",
        "candidate_save",
    ],
)
def test_one_deadline_rejects_late_results_and_records_terminal_gap(
    db_settings, db, monkeypatch, point
):
    command, replay, spec = material(2)
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()

    def delayed(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            db.execute("SELECT pg_sleep(1.05)")
            return result

        return call

    if point == "initialize":
        monkeypatch.setattr(
            CoverageIngestRepository,
            "initialize",
            staticmethod(delayed(CoverageIngestRepository.initialize)),
        )
    elif point == "plan":
        monkeypatch.setattr(coverage_workflow, "build_plan", delayed(coverage_workflow.build_plan))
    elif point == "selection":
        monkeypatch.setattr(
            coverage_workflow, "select_baseline", delayed(coverage_workflow.select_baseline)
        )
    elif point == "extract":
        monkeypatch.setattr(ReplayExtractor, "extract", delayed(ReplayExtractor.extract))
    elif point == "final_commit":
        monkeypatch.setattr(
            PostgresPlanningRepository,
            "save_selection",
            delayed(PostgresPlanningRepository.save_selection),
        )
    elif point == "candidate_reserve":
        monkeypatch.setattr(
            BoundCandidateRepository, "reserve", delayed(BoundCandidateRepository.reserve)
        )
    else:
        monkeypatch.setattr(
            BoundCandidateRepository, "save", delayed(BoundCandidateRepository.save)
        )
    try:
        receipt = services.add.add(command)
        assert services.search.search(SearchQuery("rollback", command.user_id))
        assert services.add.add(command).batch_id == receipt.batch_id
    finally:
        services.lifecycle.close()
    assert state(db) == ("complete", "raw_degraded", "time_budget_exhausted")
    assert db.execute("SELECT count(*) FROM episode_candidates").fetchone()[0] == (
        1 if point == "candidate_save" else 0
    )
    assert (
        db.execute("SELECT count(*) FROM extraction_attempts WHERE status='reserved'").fetchone()[0]
        == 0
    )
    assert (
        db.execute("SELECT count(*) FROM planning_attempts WHERE status='reserved'").fetchone()[0]
        == 0
    )
    persisted = db.execute(
        "SELECT plan_body IS NOT NULL,selection_body IS NOT "
        "NULL,deadline_at-created_at FROM planning_runs"
    ).fetchone()
    assert (
        persisted[:2]
        == (
            {
                "initialize": (False, False),
                "plan": (False, False),
                "selection": (True, False),
                "extract": (True, True),
                "final_commit": (True, True),
                "candidate_reserve": (True, True),
                "candidate_save": (True, True),
            }[point]
        )
    )
    assert persisted[2].total_seconds() == 2


def test_concurrent_claim_fences_both_owners(coverage, db, monkeypatch):
    repo = pause(coverage, monkeypatch)
    expire(db)
    batch = db.execute("SELECT id::text FROM batches").fetchone()[0]

    def claim(_):
        try:
            return repo.claim(coverage[0].user_id, batch, coverage[2], 30)
        except ProcessingBusy:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        owners = list(pool.map(claim, range(2)))
    lease = next(o for o in owners if o)
    assert sum(o is not None for o in owners) == 1
    for changes in (
        {"user_id": "other"},
        {"batch_id": "other"},
        {"owner": "other"},
        {"fence": lease.fence + 1},
    ):
        with pytest.raises(ProcessingBusy):
            repo.ensure_work(replace(lease, planning=replace(lease.planning, **changes)))
    saved = repo.load_plan(lease)
    with pytest.raises(ProcessingUnavailable):
        with repo.bounded_transaction(lease, snapshot=replace(saved, user_id="other")):
            raise AssertionError("mismatched snapshot reached transaction body")
    expire(db)
    new = claim(0)
    assert new.fence > lease.fence
    with pytest.raises(ProcessingBusy):
        repo.ensure_work(lease)
    assert db.execute(
        "SELECT p.owner=r.owner AND p.fence=r.fence AND p.lease_until=r.lease_until "
        "FROM planning_runs p JOIN processing_runs r ON r.batch_id=p.batch_id"
    ).fetchone()[0]


@pytest.mark.parametrize(
    "table,when",
    [
        ("coverage_bindings", "INSERT"),
        ("episode_candidates", "INSERT"),
        ("processing_runs", "UPDATE"),
    ],
)
def test_new_boundary_commit_failure_never_becomes_success(coverage, db, monkeypatch, table, when):
    condition = " WHEN (NEW.stage='complete')" if table == "processing_runs" else ""
    db.execute(
        "CREATE FUNCTION reject_boundary() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'test boundary'; END $$"
    )
    db.execute(
        f"CREATE CONSTRAINT TRIGGER reject_boundary AFTER {when} ON {table} "
        f"DEFERRABLE INITIALLY DEFERRED FOR EACH ROW{condition} "
        "EXECUTE FUNCTION reject_boundary()"
    )
    with pytest.raises(StorageUnavailable):
        coverage[3].add.add(coverage[0])
    assert state(db)[0] != "complete"
    if table == "coverage_bindings":
        assert db.execute("SELECT count(*) FROM processing_windows").fetchone()[0] == 0
    elif table == "episode_candidates":
        assert db.execute("SELECT count(*) FROM episode_candidates").fetchone()[0] == 0
        assert db.execute("SELECT status FROM extraction_attempts").fetchone() == ("reserved",)
    db.execute(f"DROP TRIGGER reject_boundary ON {table}")
    expire(db)
    if table == "processing_runs":

        def forbidden(*args):
            raise AssertionError("completed candidates were re-extracted")

        monkeypatch.setattr(ReplayExtractor, "extract", forbidden)
    coverage[3].add.add(coverage[0])
    assert state(db)[0] == "complete"


def test_candidate_request_tampering_and_final_window_corruption_fail_closed(
    coverage, db, monkeypatch
):
    repo = pause(coverage, monkeypatch, "candidates")
    expire(db)
    batch = db.execute("SELECT id::text FROM batches").fetchone()[0]
    lease = repo.claim(coverage[0].user_id, batch, coverage[2], 30)
    saved = repo.load_plan(lease)
    bound = repo.candidates(lease)
    window_id, window = bound.windows()[0]
    request = candidate_input(lease, window_id, window, saved, bound.payload_hash)
    with pytest.raises(ProcessingUnavailable):
        bound.reserve(replace(request, payload_hash="wrong"))
    db.execute("UPDATE processing_windows SET end_offset=end_offset-1 WHERE ordinal=0")
    with pytest.raises(ProcessingUnavailable):
        bound.reserve(request)


@pytest.mark.parametrize(
    "point",
    [
        "initialized",
        "plan_reserved",
        "plan_saved",
        "selection_saved",
        "bound",
        "candidate_reserved",
        "candidate_saved",
        "terminal",
    ],
)
def test_real_process_recovery_preserves_deadline_and_committed_work(db_settings, db, point):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
        MEMORY_API_KEY="local-test-key",
    )
    helper = Path(__file__).parent / "helpers/coverage_crash.py"
    first = subprocess.run(
        [sys.executable, str(helper), point], env=env, capture_output=True, timeout=20
    )
    assert first.returncode == 17, first.stderr.decode()
    saved = db.execute(
        "SELECT created_at,deadline_at,plan_body,selection_body FROM planning_runs"
    ).fetchone()
    ids = db.execute("SELECT id FROM episode_candidates").fetchall()
    expire(db)
    second = subprocess.run(
        [sys.executable, str(helper), "resume-" + point], env=env, capture_output=True, timeout=20
    )
    assert second.returncode == 0, second.stderr.decode()
    after = db.execute(
        "SELECT created_at,deadline_at,plan_body,selection_body FROM planning_runs"
    ).fetchone()
    assert saved[:2] == after[:2]
    for i in (2, 3):
        if saved[i] is not None:
            assert after[i] == saved[i]
    assert set(ids) <= set(db.execute("SELECT id FROM episode_candidates").fetchall())
    assert state(db) == ("complete", "raw_degraded", "offline_only")
    if point == "plan_reserved":
        assert db.execute("SELECT count(*) FROM planning_attempts").fetchone()[0] == 3
    if point == "candidate_reserved":
        assert (
            db.execute(
                "SELECT count(*) FROM extraction_attempts WHERE error_code='interrupted'"
            ).fetchone()[0]
            == 1
        )


def test_legacy_reason_constraints_are_not_weakened_by_new_terminal_shapes(
    coverage, db, db_settings
):
    old = build_services(db_settings)
    old.lifecycle.open()
    try:
        old.add.add(coverage[0])
    finally:
        old.lifecycle.close()
    with pytest.raises(psycopg.Error):
        db.execute("UPDATE processing_runs SET reason='time_budget_exhausted'")


def test_failed_planner_uses_only_persisted_shared_attempt_budget(coverage, db, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(1)
        raise ValueError("synthetic planner failure")

    monkeypatch.setattr(coverage_workflow, "build_plan", fail)
    coverage[3].add.add(coverage[0])
    assert len(calls) == coverage[2].budget.max_attempts
    assert state(db) == ("complete", "raw_degraded", "planning_attempts_exhausted")
    assert db.execute("SELECT count(*) FROM planning_attempts WHERE status='failed'").fetchone()[
        0
    ] == len(calls)
    assert db.execute("SELECT plan_body FROM planning_runs").fetchone() == (None,)


@pytest.mark.parametrize("target", ["planning_runs", "processing_runs"])
def test_candidate_transaction_checks_both_leases_after_partial_writes(coverage, db, target):
    db.execute(f"""CREATE FUNCTION expire_coverage() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN UPDATE {target} SET lease_until=clock_timestamp()-interval '1 second';
    RETURN NEW; END $$""")
    db.execute(
        "CREATE TRIGGER expire_coverage AFTER INSERT ON episode_candidates "
        "FOR EACH ROW EXECUTE FUNCTION expire_coverage()"
    )
    with pytest.raises(ProcessingBusy):
        coverage[3].add.add(coverage[0])
    assert state(db)[0] == "selected"
    assert db.execute("SELECT count(*) FROM episode_candidates").fetchone()[0] == 0
    assert db.execute("SELECT status FROM extraction_attempts").fetchall() == [("reserved",)]
    db.execute("DROP TRIGGER expire_coverage ON episode_candidates")
    expire(db)
    coverage[3].add.add(coverage[0])
    assert state(db)[2] == "offline_only"


def test_real_process_resume_after_work_deadline_does_not_start_selection(db_settings, db):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
        MEMORY_API_KEY="local-test-key",
    )
    helper = Path(__file__).parent / "helpers/coverage_crash.py"
    first = subprocess.run(
        [sys.executable, str(helper), "plan_saved", "3"], env=env, capture_output=True, timeout=20
    )
    assert first.returncode == 17, first.stderr.decode()
    before = db.execute("SELECT created_at,deadline_at,plan_body FROM planning_runs").fetchone()
    db.execute("SELECT pg_sleep(2.05)")
    expire(db)
    resumed = subprocess.run(
        [sys.executable, str(helper), "resume-expired", "3"],
        env=env,
        capture_output=True,
        timeout=20,
    )
    assert resumed.returncode == 0, resumed.stderr.decode()
    assert (
        db.execute("SELECT created_at,deadline_at,plan_body FROM planning_runs").fetchone()
        == before
    )
    assert db.execute("SELECT selection_body FROM planning_runs").fetchone() == (None,)
    assert db.execute("SELECT count(*) FROM planning_attempts").fetchone()[0] == 1
    assert state(db)[2] == "time_budget_exhausted"


def test_005_failure_rolls_back_and_upgrade_preserves_legacy_rows(db_settings, db, monkeypatch):
    command, _, _ = material()
    old = build_services(db_settings)
    old.lifecycle.open()
    try:
        receipt = old.add.add(command)
    finally:
        old.lifecycle.close()
    from tests.helpers.s3_material import drop_s3_fixture

    drop_s3_fixture(db)
    db.execute("DROP TABLE candidate_temporal")
    db.execute("DROP FUNCTION guard_candidate_temporal()")
    db.execute("ALTER TABLE candidate_windows DROP CONSTRAINT candidate_windows_temporal_link_key")
    db.execute("DROP TABLE coverage_bindings")
    db.execute("DROP TRIGGER protect_coverage_runtime ON processing_runs")
    db.execute("DROP FUNCTION protect_coverage_runtime(), protect_coverage_binding()")
    prefix = postgres.migration_source("003_candidates.sql").split(
        "ALTER TABLE processing_windows"
    )[0]
    db.execute(prefix.replace("processing_runs_check2", "processing_runs_terminal_check"))
    db.execute("DELETE FROM schema_migrations WHERE version>=5")
    before = db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall()
    original = postgres.migration_source
    with monkeypatch.context() as patch:
        patch.setattr(
            postgres,
            "migration_source",
            lambda name="001_initial.sql": (
                original(name)
                + ("\nSELECT no_such_function();" if name == "005_coverage_runtime.sql" else "")
            ),
        )
        with pytest.raises(psycopg.Error):
            migrate(db_settings.database)
    assert db.execute("SELECT to_regclass('coverage_bindings')").fetchone() == (None,)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone() == (4,)
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall() == before
    migrate(db_settings.database)
    migrate(db_settings.database)
    fresh = build_services(db_settings)
    fresh.lifecycle.open()
    try:
        assert fresh.add.add(command).batch_id == receipt.batch_id
    finally:
        fresh.lifecycle.close()
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall() == before


@pytest.mark.parametrize("empty", [False, True])
def test_offline_reply_failure_and_valid_empty_are_distinct(db_settings, db, empty):
    command, original, spec = material()
    reply = (
        ExtractionReply("complete", '{"schema_version":"episode-candidates-v1","episodes":[]}')
        if empty
        else ExtractionReply("error")
    )
    replay = ReplayExtractor(
        original.source_payload_hash, tuple((i, (reply, reply)) for i, _ in original.responses)
    )
    spec = replace(spec, processor_version=replay.version)
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        services.add.add(command)
    finally:
        services.lifecycle.close()
    assert state(db)[2] == ("offline_only" if empty else "extraction_failed")
    assert db.execute("SELECT status FROM extraction_attempts").fetchall() == (
        [("empty",)] * 3 if empty else [("failed",)] * 6
    )
    assert db.execute("SELECT count(*) FROM episode_candidates").fetchone()[0] == 0


def test_concurrent_source_retries_create_one_atomic_run_pair(coverage, db, monkeypatch):
    command, _, _, services = coverage
    with monkeypatch.context() as patch:
        patch.setattr(coverage_workflow.CoverageFlow, "finish", lambda *args: None)
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda _: services.add.add(command), range(2)))
    assert receipts[0].batch_id == receipts[1].batch_id
    assert sorted(r.reused for r in receipts) == [False, True]
    assert counts(db) == (1, 3, 3, 1, 1, 0, 0)
    before = db.execute("SELECT created_at,deadline_at FROM planning_runs").fetchone()
    services.add.add(command)
    assert db.execute("SELECT created_at,deadline_at FROM planning_runs").fetchone() == before


def test_new_flow_does_not_adopt_pending_v1_run(coverage, db, db_settings, monkeypatch):
    old = build_services(db_settings)
    old.lifecycle.open()

    def stop(*args):
        raise Interrupt()

    try:
        monkeypatch.setattr(old.add.processing.repository, "checkpoint", stop)
        with pytest.raises(Interrupt):
            old.add.add(coverage[0])
    finally:
        old.lifecycle.close()
    expire(db)
    with pytest.raises(ProcessingUnavailable):
        coverage[3].add.add(coverage[0])
    assert db.execute("SELECT count(*) FROM planning_runs").fetchone()[0] == 0
