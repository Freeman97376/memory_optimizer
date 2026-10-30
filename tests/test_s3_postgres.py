import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from memory_optimizer.bootstrap import build_s3_services, build_services
from memory_optimizer.domain import (
    ProcessingUnavailable,
    SearchQuery,
    StorageUnavailable,
    payload_digest,
)
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.contracts import StaleTargets
from memory_optimizer.storage import publication
from memory_optimizer.storage.relations_report import PostgresS3ReportRepository
from tests.helpers.s3_material import material, with_response

pytestmark = pytest.mark.integration


@contextmanager
def services(settings, value):
    command, extraction, proposer, spec = value
    app = build_s3_services(
        settings, spec=spec, offline_extractor=extraction, relation_proposer=proposer
    )
    app.lifecycle.open()
    try:
        yield app, command
    finally:
        app.lifecycle.close()


@pytest.mark.parametrize("mode", ["normal", "unknown", "two_windows"])
def test_add_to_published_search(db_settings, db, mode):
    with services(db_settings, material(mode)) as (app, command):
        receipt = app.add.add(command)
        report = PostgresS3ReportRepository(app.lifecycle.transaction).report(
            command.user_id, receipt.batch_id
        )
        assert report["run"]["outcome"] == "s3_published"
        assert report["publication"]["published"]
        result = app.search.search(SearchQuery("lock", command.user_id, 1))
        assert len(result) == 1 and result[0].content.startswith("经历证据")
        if mode == "two_windows":
            assert (
                "single-thread passed" in result[0].content
                and "duplicates under concurrency" in result[0].content
            )
        if mode == "unknown":
            assert "结果/验证: unknown" in result[0].content
        before = db.execute("SELECT version_id FROM episode_heads").fetchall()
        assert app.add.add(command).reused
        assert db.execute("SELECT version_id FROM episode_heads").fetchall() == before
        assert app.search.search(SearchQuery("lock", "other-user", 5)) == []


@pytest.mark.parametrize("mode", ["correction", "continuation"])
def test_cross_add_appends_history_without_losing_old_evidence(db_settings, db, mode):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    old = db.execute("SELECT id,body FROM episode_versions").fetchall()
    with services(db_settings, material(mode)) as (app, command):
        app.add.add(command)
        hits = app.search.search(SearchQuery("single-thread", command.user_id, 1))
        assert hits[0].content.startswith("经历证据")
        assert (
            "duplicates under concurrency" if mode == "correction" else "stress test passed"
        ) in hits[0].content
    assert db.execute("SELECT count(*) FROM episode_heads").fetchone() == (1,)
    assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (2,)
    assert (
        db.execute("SELECT id,body FROM episode_versions WHERE id=%s", (old[0][0],)).fetchall()
        == old
    )
    legacy = build_services(db_settings)
    legacy.lifecycle.open()
    try:
        assert all(
            not h.content.startswith("经历证据")
            for h in legacy.search.search(SearchQuery("lock", command.user_id, 5))
        )
    finally:
        legacy.lifecycle.close()


def expire(db):
    for table in ("planning_runs", "processing_runs"):
        db.execute(
            f"UPDATE {table} SET lease_until=clock_timestamp()-interval '1 second' "
            "WHERE owner IS NOT NULL"
        )


def cli_files(folder, value):
    command, extraction, proposer, spec = value
    folder.mkdir(exist_ok=True)
    values = {
        "request": asdict(command),
        "responses": {
            "source_payload_hash": payload_digest(command),
            "windows": [
                {"ordinal": i, "replies": [asdict(r) for r in replies]}
                for i, replies in extraction.responses
            ],
        },
        "relations": {
            "source_payload_hash": payload_digest(command),
            "replies": [json.loads(r) for r in proposer.replies],
        },
        "config": {k: v for k, v in asdict(spec).items() if k != "processor_version"},
    }
    for name, value in values.items():
        (folder / f"{name}.json").write_text(canonical(value), encoding="utf-8")


def cli(settings, folder, *, report_only=False, output="report.json"):
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
        "s3-offline-v1",
        "--request",
        str(folder / "request.json"),
        "--output",
        str(folder / output),
        "--query",
        "lock",
    ]
    if report_only:
        args += ["--report-only"]
    else:
        args += [
            "--responses",
            str(folder / "responses.json"),
            "--relations",
            str(folder / "relations.json"),
            "--coverage-config",
            str(folder / "config.json"),
        ]
    return subprocess.run(args, env=env, capture_output=True, timeout=25)


@pytest.mark.parametrize(
    "point", ["initialized", "snapshot", "reserved", "decision", "inside_publication", "published"]
)
def test_real_process_exit_then_cli_resume(db_settings, db, tmp_path, point):
    value = material("two_windows")
    cli_files(tmp_path, value)
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
    )
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [sys.executable, "tests/helpers/s3_crash.py", point],
        env=env,
        capture_output=True,
        timeout=25,
    )
    assert child.returncode == 23, child.stderr.decode()
    deadline = db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0]
    candidates = db.execute("SELECT id,fingerprint FROM episode_candidates ORDER BY id").fetchall()
    prior_snapshots = db.execute("SELECT revision,input_hash FROM relation_snapshots").fetchall()
    if point == "inside_publication":
        assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (0,)
    expire(db)
    result = cli(db_settings, tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["timing"]["deadline_at"] == deadline.isoformat()
    assert report["run"]["outcome"] == "s3_published"
    assert report["search"][0]["content"].startswith("经历证据")
    assert "duplicates under concurrency" in report["search"][0]["content"]
    assert set(candidates) <= set(
        db.execute("SELECT id,fingerprint FROM episode_candidates").fetchall()
    )
    assert set(prior_snapshots) <= set(
        db.execute("SELECT revision,input_hash FROM relation_snapshots").fetchall()
    )
    if point == "reserved":
        assert db.execute(
            "SELECT status,error_code FROM relation_attempts ORDER BY number"
        ).fetchall() == [("failed", "interrupted"), ("accepted", None)]


@pytest.mark.parametrize("mode", ["malformed", "omitted", "unresolved"])
def test_failed_or_missing_correction_qualifies_specific_old_episode(db_settings, db, mode):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    response = {"independent": ["c0"], "unresolved": [], "relations": []}
    if mode == "malformed":
        response["extra"] = "invalid"
    if mode == "unresolved":
        response.update(independent=[], unresolved=["c0"])
    value = with_response(material("correction"), response)
    with services(db_settings, value) as (app, command):
        receipt = app.add.add(command)
        report = PostgresS3ReportRepository(app.lifecycle.transaction).report(
            command.user_id, receipt.batch_id
        )
        assert report["run"]["outcome"] == "s3_withheld"
        assert report["publication"]["qualified"]
        hit = app.search.search(SearchQuery("single-thread", command.user_id, 1))[0]
        assert "待核验" in hit.content and "不能据此认定问题已解决" in hit.content
    assert db.execute("SELECT count(*) FROM episode_heads").fetchone() == (1,)


@pytest.mark.parametrize("fault", ["version", "commit"])
def test_publication_failure_rolls_back_visibility_and_terminal(
    db_settings, db, monkeypatch, fault
):
    with services(db_settings, material("two_windows")) as (app, command):
        original = publication.append_version

        def fail(conn, *args):
            result = original(conn, *args)
            if fault == "version":
                raise RuntimeError("synthetic publication failure")
            conn.execute(
                "CREATE FUNCTION fail_publication_commit() RETURNS trigger LANGUAGE plpgs"
                "ql AS $$ BEGIN RAISE EXCEPTION 'synthetic deferred failure'; END $$"
            )
            conn.execute(
                "CREATE CONSTRAINT TRIGGER fail_publication_commit AFTER INSERT ON s3_pub"
                "lications DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION fa"
                "il_publication_commit()"
            )
            return result

        with monkeypatch.context() as patch:
            patch.setattr(publication, "append_version", fail)
            with pytest.raises((RuntimeError, StorageUnavailable)):
                app.add.add(command)
        assert db.execute("SELECT count(*) FROM episode_heads").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM s3_publications").fetchone() == (0,)
        assert db.execute("SELECT stage FROM processing_runs").fetchone() == ("selected",)
        deadline = db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0]
        expire(db)
        app.add.add(command)
        assert db.execute("SELECT deadline_at FROM planning_runs").fetchone() == (deadline,)
        assert db.execute("SELECT count(*) FROM episode_heads").fetchone() == (1,)


def test_shared_attempt_budget_closes_without_extra_relation_calls(db_settings, db):
    with services(db_settings, material("two_windows", budget=4)) as (app, command):
        receipt = app.add.add(command)
        report = PostgresS3ReportRepository(app.lifecycle.transaction).report(
            command.user_id, receipt.batch_id
        )
        assert report["run"]["outcome"] == "s3_withheld"
        assert report["publication"]["attempts_used"] == 4
        assert report["relations"]["attempts"] == []
        assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (0,)


def test_late_publication_uses_original_deadline_and_qualifies_history(
    db_settings, db, monkeypatch
):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    value = material("correction", seconds=3)
    with services(db_settings, value) as (app, command):
        original = app.add.processing.repository.publish

        def late(lease, request, decision):
            db.execute("SELECT pg_sleep(2.05)")
            return original(lease, request, decision)

        monkeypatch.setattr(app.add.processing.repository, "publish", late)
        receipt = app.add.add(command)
        report = PostgresS3ReportRepository(app.lifecycle.transaction).report(
            command.user_id, receipt.batch_id
        )
        assert report["publication"]["reason"] == "time_budget_exhausted"
        assert report["publication"]["qualified"] and not report["publication"]["published"]
        assert (
            "待核验"
            in app.search.search(SearchQuery("single-thread", command.user_id, 1))[0].content
        )
        assert (
            db.execute(
                "SELECT extract(epoch FROM deadline_at-created_at) FROM planning_runs WHE"
                "RE batch_id=%s",
                (receipt.batch_id,),
            ).fetchone()[0]
            == 3
        )


def test_target_version_changes_cannot_publish_stale_response(db_settings, db, monkeypatch):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    with services(db_settings, material("correction")) as (app, command):
        repo = app.add.processing.repository
        original = repo.save_relation
        advanced = False

        def advance(lease, *args):
            nonlocal advanced
            if not advanced:
                advanced = True
                with services(db_settings, material("continuation", request_id="concurrent")) as (
                    other,
                    other_command,
                ):
                    other.add.add(other_command)
                with pytest.raises(StaleTargets):
                    original(lease, *args)
                raise StaleTargets("synthetic coordination race")
            return original(lease, *args)

        monkeypatch.setattr(repo, "save_relation", advance)
        receipt = app.add.add(command)
        report = PostgresS3ReportRepository(app.lifecycle.transaction).report(
            command.user_id, receipt.batch_id
        )
        assert len(report["relations"]["snapshots"]) == 2
        assert report["publication"]["qualified"] or report["publication"]["published"]
        assert all(
            a["status"] != "accepted" for a in report["relations"]["attempts"] if a["revision"] == 1
        )
        assert report["publication"]["attempts_used"] <= 12


def test_read_only_report_and_search_do_not_change_any_rows(db_settings, db, tmp_path):
    from shutil import copyfile

    for source, target in (
        ("s3-add", "request"),
        ("s3-responses", "responses"),
        ("s3-relations", "relations"),
        ("s3-config", "config"),
    ):
        copyfile(Path("examples") / f"{source}.json", tmp_path / f"{target}.json")
    assert cli(db_settings, tmp_path).returncode == 0
    tables = (
        "planning_runs",
        "processing_runs",
        "relation_snapshots",
        "relation_attempts",
        "episode_versions",
        "episode_heads",
        "s3_publications",
    )
    before = {t: db.execute(f"SELECT row_to_json(t) FROM {t} t").fetchall() for t in tables}
    result = cli(db_settings, tmp_path, report_only=True, output="inspect.json")
    assert result.returncode == 0, result.stderr.decode()
    after = {t: db.execute(f"SELECT row_to_json(t) FROM {t} t").fetchall() for t in tables}
    assert before == after


@pytest.mark.parametrize("table", ["relation_snapshots", "episode_versions", "s3_publications"])
def test_immutable_checkpoints_and_publication(db_settings, db, table):
    import psycopg

    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    with pytest.raises(psycopg.Error):
        db.execute(f"UPDATE {table} SET body='{{}}'::jsonb")


def test_explicit_http_service_uses_published_search_without_default_switch(db_settings):
    from fastapi.testclient import TestClient

    from memory_optimizer.api.app import create_app

    value = material("two_windows")
    command, extraction, proposer, spec = value
    app = build_s3_services(
        db_settings, spec=spec, offline_extractor=extraction, relation_proposer=proposer
    )
    with TestClient(
        create_app(db_settings, services=app), headers={"Authorization": "Bearer local-test-key"}
    ) as client:
        assert client.post("/add", json=asdict(command)).status_code == 200
        response = client.post(
            "/search", json={"user_id": command.user_id, "query": "single-thread", "top_k": 1}
        )
        assert response.status_code == 200
        assert "duplicates under concurrency" in response.text


def test_failed_required_qualification_cannot_return_add_success(db_settings, db, monkeypatch):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    value = with_response(
        material("correction"), {"independent": [], "unresolved": ["c0"], "relations": []}
    )
    with services(db_settings, value) as (app, command):
        before = db.execute("SELECT version_id FROM episode_heads").fetchall()
        original = publication.append_version

        def fail(conn, lease, episode_id, body):
            if body["pending"]:
                raise RuntimeError("required qualification failed")
            return original(conn, lease, episode_id, body)

        with monkeypatch.context() as patch:
            patch.setattr(publication, "append_version", fail)
            with pytest.raises(RuntimeError):
                app.add.add(command)
        assert db.execute("SELECT version_id FROM episode_heads").fetchall() == before
        assert db.execute(
            "SELECT stage FROM processing_runs r JOIN batches b ON b.id=r.batch_id WH"
            "ERE b.request_id='correction'"
        ).fetchone() == ("selected",)
        expire(db)
        app.add.add(command)
        assert "待核验" in app.search.search(SearchQuery("lock", command.user_id, 1))[0].content


@pytest.mark.parametrize("target", ["snapshot", "decision", "publication", "episode"])
def test_corrupted_saved_evidence_rejected_without_read_repair(db_settings, db, target):
    with services(db_settings, material()) as (app, command):
        receipt = app.add.add(command)
        table = {
            "snapshot": "relation_snapshots",
            "decision": "relation_attempts",
            "publication": "s3_publications",
            "episode": "episode_versions",
        }[target]
        db.execute(f"ALTER TABLE {table} DISABLE TRIGGER USER")
        field = "decision" if target == "decision" else "body"
        db.execute(f"UPDATE {table} SET {field}={field} || '{{\"tampered\":true}}'::jsonb")
        before = db.execute(f"SELECT row_to_json(t) FROM {table} t").fetchall()
        with pytest.raises(ProcessingUnavailable):
            if target == "episode":
                app.search.search(SearchQuery("lock", command.user_id, 1))
            else:
                PostgresS3ReportRepository(app.lifecycle.transaction).report(
                    command.user_id, receipt.batch_id
                )
        assert before == db.execute(f"SELECT row_to_json(t) FROM {table} t").fetchall()


def test_migration_007_failure_and_retry_preserve_legacy(db_settings, db, monkeypatch):
    import psycopg

    from memory_optimizer.bootstrap import build_coverage_services
    from memory_optimizer.storage import postgres
    from tests.helpers.s3_material import drop_s3_fixture
    from tests.helpers.temporal_material import material as old_material

    command, extractor, spec = old_material()
    app = build_coverage_services(db_settings, spec=spec, offline_extractor=extractor)
    app.lifecycle.open()
    try:
        app.add.add(command)
    finally:
        app.lifecycle.close()
    before = db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall()
    drop_s3_fixture(db)
    sql = postgres.migration_source("006_temporal_proposals.sql")
    db.execute(sql.split("ALTER TABLE candidate_windows")[0])
    fn = sql.split("CREATE FUNCTION guard_candidate_temporal()")[1].split(
        "CREATE TRIGGER guard_candidate_temporal"
    )[0]
    db.execute("CREATE OR REPLACE FUNCTION guard_candidate_temporal()" + fn)
    original = postgres.migration_source
    with monkeypatch.context() as patch:
        patch.setattr(
            postgres,
            "migration_source",
            lambda name="001_initial.sql": (
                original(name)
                + (
                    "\nSELECT nonexistent_s3_function();"
                    if name == "007_s3_publication.sql"
                    else ""
                )
            ),
        )
        with pytest.raises(psycopg.Error):
            postgres.migrate(db_settings.database)
    assert db.execute("SELECT to_regclass('episode_versions')").fetchone() == (None,)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone() == (6,)
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall() == before
    postgres.migrate(db_settings.database)
    postgres.migrate(db_settings.database)
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall() == before
    assert db.execute("SELECT count(*) FROM episode_versions").fetchone() == (0,)


@pytest.mark.parametrize("point", ["before_plan", "before_snapshot"])
def test_expired_before_relation_snapshot_still_qualifies_explicit_history(
    db_settings, db, monkeypatch, point
):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    with services(db_settings, material("correction", seconds=3)) as (app, command):
        if point == "before_plan":
            original = app.add.repository.commit

            def delayed(*args):
                result = original(*args)
                db.execute("SELECT pg_sleep(2.05)")
                return result

            monkeypatch.setattr(app.add.repository, "commit", delayed)
        else:
            original = app.add.processing.repository.prepare

            def delayed(*args, **kwargs):
                if kwargs.get("work", True):
                    db.execute("SELECT pg_sleep(2.05)")
                return original(*args, **kwargs)

            monkeypatch.setattr(app.add.processing.repository, "prepare", delayed)
        receipt = app.add.add(command)
        report = PostgresS3ReportRepository(app.lifecycle.transaction).report(
            command.user_id, receipt.batch_id
        )
        assert report["publication"]["reason"] == "time_budget_exhausted"
        assert report["publication"]["qualified"]
        assert report["relations"]["attempts"] == []
        assert report["relations"]["snapshots"][0]["input"]["meta"]["cleanup_only"]
        assert "待核验" in app.search.search(SearchQuery("lock", command.user_id, 1))[0].content


def test_ambiguous_same_problem_does_not_qualify_all_history(db_settings, db):
    for identity in ("first", "second"):
        with services(db_settings, material(request_id=identity)) as (app, command):
            app.add.add(command)
    before = db.execute("SELECT version_id FROM episode_heads ORDER BY episode_id").fetchall()
    with services(db_settings, material("correction")) as (app, command):
        receipt = app.add.add(command)
        report = PostgresS3ReportRepository(app.lifecycle.transaction).report(
            command.user_id, receipt.batch_id
        )
        assert report["run"]["outcome"] == "s3_withheld"
        assert report["publication"]["qualified"] == []
    assert (
        db.execute("SELECT version_id FROM episode_heads ORDER BY episode_id").fetchall() == before
    )


def test_new_and_old_pending_flows_cannot_adopt_each_other(db_settings, db, monkeypatch):
    from dataclasses import replace

    from memory_optimizer.bootstrap import build_coverage_services

    with services(db_settings, material()) as (app, command):

        def stop(*args, **kwargs):
            raise RuntimeError("stop before planning")

        with monkeypatch.context() as patch:
            patch.setattr(app.add.processing.repository, "ensure_work", stop)
            with pytest.raises(RuntimeError):
                app.add.add(command)
        expire(db)
        _, extractor, _, spec = material()
        old = build_coverage_services(
            db_settings,
            spec=replace(
                spec, flow_version="coverage-temporal-v1", processor_version=extractor.version
            ),
            offline_extractor=extractor,
        )
        old.lifecycle.open()
        try:
            with pytest.raises(ProcessingUnavailable):
                old.add.add(command)
        finally:
            old.lifecycle.close()
        app.add.add(command)


def test_expired_original_deadline_after_actual_process_exit(db_settings, db, tmp_path):
    value = material("two_windows", seconds=3)
    cli_files(tmp_path, value)
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
    )
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [sys.executable, "tests/helpers/s3_crash.py", "reserved", "3"],
        env=env,
        capture_output=True,
        timeout=25,
    )
    assert child.returncode == 23, child.stderr.decode()
    deadline = db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0]
    db.execute("SELECT pg_sleep(2.05)")
    expire(db)
    result = cli(db_settings, tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["timing"]["deadline_at"] == deadline.isoformat()
    assert report["publication"]["reason"] == "time_budget_exhausted"
    assert len(report["relations"]["attempts"]) == 1
    assert report["run"]["outcome"] == "s3_withheld"


def test_same_file_different_problem_remains_separate(db_settings, db):
    from dataclasses import replace

    from memory_optimizer.extraction.replay import ReplayExtractor
    from memory_optimizer.relations.replay import ReplayRelations, processor_version

    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    command, extractor, proposer, spec = material(request_id="parser-problem")
    command = replace(
        command,
        messages=tuple(
            replace(m, content=m.content.replace("duplicate writes", "parser exception"))
            for m in command.messages
        ),
    )
    value = {
        "source_payload_hash": payload_digest(command),
        "windows": [
            {
                "ordinal": i,
                "replies": [
                    dict(
                        status=r.status,
                        content=r.content.replace("duplicate writes", "parser exception"),
                    )
                    for r in replies
                ],
            }
            for i, replies in extractor.responses
        ],
    }
    extractor = ReplayExtractor.from_json(canonical(value))
    proposer = ReplayRelations.from_json(
        canonical(
            {
                "source_payload_hash": payload_digest(command),
                "replies": [json.loads(r) for r in proposer.replies],
            }
        )
    )
    spec = replace(spec, processor_version=processor_version(extractor, proposer))
    with services(db_settings, (command, extractor, proposer, spec)) as (app, command):
        app.add.add(command)
        hit = app.search.search(SearchQuery("parser", command.user_id, 1))[0]
        assert hit.content.startswith("经历证据") and "duplicate writes" not in hit.content
    assert db.execute("SELECT count(*) FROM episode_heads").fetchone() == (2,)
