import json
import os
import subprocess
import sys
from dataclasses import replace

import psycopg
import pytest

from memory_optimizer.bootstrap import build_coverage_services
from memory_optimizer.domain import (
    ProcessingBusy,
    ProcessingUnavailable,
    SearchQuery,
    StorageUnavailable,
)
from memory_optimizer.storage import postgres
from memory_optimizer.storage.coverage_report import PostgresCoverageReportRepository
from memory_optimizer.storage.temporal_report import PostgresTemporalReportRepository

from .helpers.coverage_material import material as old_material
from .helpers.temporal_material import material
from .test_coverage_postgres import expire
from .test_coverage_report_postgres import dump

pytestmark = pytest.mark.integration


@pytest.fixture
def temporal(db_settings):
    command, replay, spec = material()
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        yield command, replay, spec, services
    finally:
        services.lifecycle.close()


def report(services, command):
    return PostgresTemporalReportRepository(services.lifecycle.transaction).report_request(command)


def test_temporal_flow_atomic_sidecars_raw_search_and_read_only_report(temporal, db):
    command, _, _, services = temporal
    receipt = services.add.add(command)
    result = report(services, command)
    assert result["schema_version"] == "temporal-coverage-report-v1"
    entries = result["temporal"]["entries"]
    assert len(entries) == 3
    assert all(e["body"]["events"][0]["binding_status"] == "proposed" for e in entries)
    assert all(ev["sort_key"] is None for e in entries for ev in e["body"]["events"])
    assert any(ev["binding_status"] == "unknown" for e in entries for ev in e["body"]["events"])
    assert services.search.search(SearchQuery("Yesterday", command.user_id))
    assert not services.search.search(SearchQuery("Yesterday", "another-scope"))
    before = dump(db)
    sidecars = db.execute(
        "SELECT row_to_json(t) FROM candidate_temporal t ORDER BY candidate_id,window_id"
    ).fetchall()
    assert services.add.add(command).reused
    assert report(services, command) == result and dump(db) == before
    assert (
        db.execute(
            "SELECT row_to_json(t) FROM candidate_temporal t ORDER BY candidate_id,window_id"
        ).fetchall()
        == sidecars
    )
    with pytest.raises(ProcessingUnavailable):
        PostgresTemporalReportRepository(services.lifecycle.transaction).report(
            "other", receipt.batch_id
        )
    with pytest.raises(ProcessingUnavailable):
        PostgresCoverageReportRepository(services.lifecycle.transaction).report_request(command)


@pytest.mark.parametrize("failure", ["insert", "deferred_commit", "lease_expiry"])
def test_time_candidate_attempt_and_link_are_one_transaction(temporal, db, failure):
    command, _, _, services = temporal
    action = (
        "UPDATE planning_runs SET lease_until=clock_timestamp()-interval '1 second'; RETURN NEW;"
        if failure == "lease_expiry"
        else "RAISE EXCEPTION 'synthetic temporal failure';"
    )
    db.execute(
        f"CREATE FUNCTION fail_time() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN {action} END $$"
    )
    if failure == "deferred_commit":
        db.execute(
            "CREATE CONSTRAINT TRIGGER fail_time AFTER INSERT ON candidate_temporal "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION fail_time()"
        )
    else:
        db.execute(
            "CREATE TRIGGER fail_time AFTER INSERT ON candidate_temporal "
            "FOR EACH ROW EXECUTE FUNCTION fail_time()"
        )
    with pytest.raises(ProcessingBusy if failure == "lease_expiry" else StorageUnavailable):
        services.add.add(command)
    assert db.execute("SELECT count(*) FROM episode_candidates").fetchone() == (0,)
    assert db.execute("SELECT count(*) FROM candidate_windows").fetchone() == (0,)
    assert db.execute("SELECT count(*) FROM candidate_temporal").fetchone() == (0,)
    assert db.execute("SELECT status FROM extraction_attempts").fetchall() == [("reserved",)]
    deadline = db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0]
    db.execute("DROP TRIGGER fail_time ON candidate_temporal")
    expire(db)
    services.add.add(command)
    assert db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0] == deadline
    assert len(report(services, command)["temporal"]["entries"]) == 3


def test_time_compute_and_sidecar_commit_share_original_work_cutoff(db_settings, db):
    command, replay, spec = material(2)
    db.execute(
        "CREATE FUNCTION slow_time() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN PERFORM pg_sleep(1.05); RETURN NEW; END $$"
    )
    db.execute(
        "CREATE TRIGGER slow_time AFTER INSERT ON candidate_temporal "
        "FOR EACH ROW EXECUTE FUNCTION slow_time()"
    )
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        services.add.add(command)
        result = report(services, command)
        assert result["run"]["reason"] == "time_budget_exhausted"
        assert result["temporal"]["entries"] == []
        assert db.execute("SELECT count(*) FROM episode_candidates").fetchone() == (0,)
        assert db.execute("SELECT error_code FROM extraction_attempts").fetchall() == [
            ("time_budget_exhausted",)
        ]
    finally:
        services.lifecycle.close()


@pytest.mark.parametrize("mode", ["complete", "pending"])
def test_old_coverage_is_not_upgraded_and_pending_flow_cannot_mix(
    db_settings, db, monkeypatch, mode
):
    command, replay, spec = old_material()
    old = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    old.lifecycle.open()
    try:
        if mode == "pending":
            with monkeypatch.context() as patch:

                def stop(*args):
                    raise RuntimeError("synthetic stop")

                patch.setattr(old.add.processing.repository, "ensure_work", stop)
                with pytest.raises(RuntimeError):
                    old.add.add(command)
            expire(db)
        else:
            old.add.add(command)
        before = dump(db)
        # Same original input with explicit new runtime; terminal history only replays.
        new = build_coverage_services(
            db_settings,
            spec=replace(spec, flow_version="coverage-temporal-v1"),
            offline_extractor=replay,
        )
        new.lifecycle.open()
        try:
            if mode == "pending":
                with pytest.raises(ProcessingUnavailable):
                    new.add.add(command)
            else:
                assert new.add.add(command).reused
            with pytest.raises(ProcessingUnavailable):
                report(new, command)
        finally:
            new.lifecycle.close()
        assert dump(db) == before
        assert db.execute("SELECT count(*) FROM candidate_temporal").fetchone() == (0,)
    finally:
        old.lifecycle.close()


@pytest.mark.parametrize("damage", ["missing", "normalization", "provenance", "input_hash"])
def test_time_read_validation_rejects_damage_without_repair(temporal, db, damage):
    command, _, _, services = temporal
    services.add.add(command)
    if damage == "missing":
        db.execute("DELETE FROM candidate_temporal")
    else:
        db.execute("ALTER TABLE candidate_temporal DISABLE TRIGGER guard_candidate_temporal")
        if damage == "normalization":
            db.execute(
                "UPDATE candidate_temporal SET body=jsonb_set(body,'{events,0,sort_key}','1')"
            )
        if damage == "provenance":
            db.execute(
                "UPDATE candidate_temporal SET body=jsonb_set(body,"
                "'{proposals,0,time,evidence,0,role}','\"user\"')"
            )
        if damage == "input_hash":
            db.execute("UPDATE candidate_temporal SET input_hash=repeat('0',64)")
    before = dump(db)
    with pytest.raises(ProcessingUnavailable):
        report(services, command)
    assert dump(db) == before


def test_time_rows_cannot_be_rewritten(temporal, db):
    command, _, _, services = temporal
    services.add.add(command)
    with pytest.raises(psycopg.Error):
        db.execute("UPDATE candidate_temporal SET input_hash=repeat('0',64)")
    assert report(services, command)["temporal"]["normalization"] == "disabled"


def cli(settings, tmp_path, read=False, config="examples/temporal-config.json"):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=settings.schema,
    )
    args = [
        sys.executable,
        "-m",
        "memory_optimizer.offline",
        "--flow",
        "coverage-temporal-v1",
        "--request",
        "examples/temporal-add.json",
        "--output",
        str(tmp_path / ("read.json" if read else "report.json")),
    ]
    args += (
        ["--report-only"]
        if read
        else ["--responses", "examples/temporal-responses.json", "--coverage-config", config]
    )
    return subprocess.run(args, env=env, capture_output=True, timeout=20)


def test_explicit_temporal_cli_and_report_only(db_settings, db, tmp_path):
    result = cli(db_settings, tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert data["spec"]["flow_version"] == "coverage-temporal-v1"
    before = dump(db)
    assert cli(db_settings, tmp_path, True).returncode == 0
    data.pop("reused")
    assert json.loads((tmp_path / "read.json").read_text(encoding="utf-8")) == data
    assert dump(db) == before


def test_cli_wrong_flow_config_rejected_before_source_write(db_settings, db, tmp_path):
    assert cli(db_settings, tmp_path, config="examples/coverage-config.json").returncode == 1
    assert db.execute("SELECT count(*) FROM batches").fetchone() == (0,)


@pytest.mark.parametrize(
    "point", ["initialized", "reserved", "inside_transaction", "saved", "terminal"]
)
def test_real_process_resume_retains_temporal_checkpoints_and_deadline(
    db_settings, db, tmp_path, point
):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
    )
    child = subprocess.run(
        [sys.executable, "tests/helpers/temporal_crash.py", point],
        env=env,
        capture_output=True,
        timeout=20,
    )
    assert child.returncode == 17, child.stderr.decode()
    deadline = db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0]
    prior = db.execute(
        "SELECT candidate_id,window_id,body_hash,recorded_at FROM candidate_temporal "
        "ORDER BY candidate_id,window_id"
    ).fetchall()
    if point == "inside_transaction":
        assert db.execute("SELECT count(*) FROM episode_candidates").fetchone() == (0,)
        assert prior == []
    expire(db)
    result = cli(db_settings, tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert data["timing"]["deadline_at"] == deadline.isoformat()
    assert len(data["temporal"]["entries"]) == 3
    after = db.execute(
        "SELECT candidate_id,window_id,body_hash,recorded_at FROM candidate_temporal "
        "ORDER BY candidate_id,window_id"
    ).fetchall()
    assert all(row in after for row in prior)
    if point in {"reserved", "inside_transaction"}:
        assert data["windows"][0]["attempts"][0]["error_code"] == "interrupted"
        assert len(data["windows"][0]["attempts"]) == 2
    elif point in {"saved", "terminal"}:
        assert len(data["windows"][0]["attempts"]) == 1


def test_migration_006_failure_and_retry_preserve_old_rows(db_settings, db, monkeypatch):
    command, replay, spec = old_material()
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        services.add.add(command)
    finally:
        services.lifecycle.close()
    before = dump(db)
    from tests.helpers.s3_material import drop_s3_fixture

    drop_s3_fixture(db)
    db.execute("DROP TABLE candidate_temporal")
    db.execute("DROP FUNCTION guard_candidate_temporal()")
    db.execute("ALTER TABLE candidate_windows DROP CONSTRAINT candidate_windows_temporal_link_key")
    old_sql = postgres.migration_source("005_coverage_runtime.sql")
    db.execute(old_sql.split("CREATE TABLE coverage_bindings")[0])
    function = old_sql.split("CREATE FUNCTION protect_coverage_runtime")[1].split(
        "CREATE TRIGGER protect_coverage_runtime"
    )[0]
    db.execute("CREATE OR REPLACE FUNCTION protect_coverage_runtime" + function)
    db.execute("DELETE FROM schema_migrations WHERE version=6")
    original = postgres.migration_source
    with monkeypatch.context() as patch:
        patch.setattr(
            postgres,
            "migration_source",
            lambda name="001_initial.sql": (
                original(name)
                + (
                    "\nSELECT missing_temporal_function();"
                    if name == "006_temporal_proposals.sql"
                    else ""
                )
            ),
        )
        with pytest.raises(psycopg.Error):
            postgres.migrate(db_settings.database)
    assert db.execute("SELECT to_regclass('candidate_temporal')").fetchone() == (None,)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone() == (5,)
    assert dump(db) == before
    postgres.migrate(db_settings.database)
    postgres.migrate(db_settings.database)
    assert db.execute("SELECT count(*) FROM candidate_temporal").fetchone() == (0,)
    assert dump(db) == before


def test_new_pending_cannot_be_finished_by_old_coverage(db_settings, db, temporal, monkeypatch):
    command, replay, spec, services = temporal
    with monkeypatch.context() as patch:

        def stop(*args):
            raise RuntimeError("synthetic stop")

        patch.setattr(services.add.processing.repository, "ensure_work", stop)
        with pytest.raises(RuntimeError):
            services.add.add(command)
    expire(db)
    before = dump(db)
    old = build_coverage_services(
        db_settings,
        spec=replace(spec, flow_version="coverage-offline-v1"),
        offline_extractor=replay,
    )
    old.lifecycle.open()
    try:
        with pytest.raises(ProcessingUnavailable):
            old.add.add(command)
    finally:
        old.lifecycle.close()
    assert dump(db) == before
