import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from memory_optimizer.bootstrap import build_coverage_services, build_services
from memory_optimizer.domain import ProcessingUnavailable, SourceMessage, payload_digest
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.extraction.types import ExtractionReply
from memory_optimizer.processing.coverage_types import candidate_input
from memory_optimizer.storage.coverage_report import PostgresCoverageReportRepository

from .helpers.coverage_material import material
from .test_coverage_postgres import coverage as coverage_fixture
from .test_coverage_postgres import expire, pause

coverage = coverage_fixture

pytestmark = pytest.mark.integration


def dump(db):
    return [
        db.execute(f"SELECT row_to_json(t) FROM {table} t ORDER BY 1::text").fetchall()
        for table in (
            "batches",
            "planning_runs",
            "planning_attempts",
            "processing_runs",
            "processing_windows",
            "extraction_attempts",
            "episode_candidates",
        )
    ]


def repo(services):
    return PostgresCoverageReportRepository(services.lifecycle.transaction)


def test_report_is_read_only_deterministic_scoped_and_retains_deadline(coverage, db):
    command, _, _, services = coverage
    receipt = services.add.add(command)
    before = dump(db)
    report = repo(services).report_request(command)
    assert repo(services).report(command.user_id, receipt.batch_id) == report
    assert dump(db) == before
    assert len(report["candidates"]) == 2
    assert [w["state"] for w in report["windows"]] == ["accepted"] * 3
    assert report["coverage"]["selected_context"]["ratio"] == 1
    assert report["coverage"]["candidate_evidence"]["codepoints"] > 0
    assert report["coverage"]["published_evidence"]["ratio"] == 0
    assert (
        report["timing"]["deadline_at"]
        == db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0].isoformat()
    )
    with pytest.raises(ProcessingUnavailable):
        repo(services).report("other", receipt.batch_id)
    with pytest.raises(ProcessingUnavailable):
        repo(services).report_request(replace(command, session_id="changed"))
    assert dump(db) == before


def test_pending_reserved_attempt_remains_unknown_and_report_does_not_claim(
    coverage, db, monkeypatch
):
    command, _, _, services = coverage
    workflow_repo = pause(coverage, monkeypatch, "candidates")
    expire(db)
    batch = str(db.execute("SELECT id FROM batches").fetchone()[0])
    lease = workflow_repo.claim(command.user_id, batch, coverage[2], 30)
    saved = workflow_repo.load_plan(lease)
    candidates = workflow_repo.candidates(lease)
    wid, window = candidates.windows()[0]
    candidates.reserve(candidate_input(lease, wid, window, saved, payload_digest(command)))
    before = dump(db)
    report = repo(services).report_request(command)
    assert report["windows"][0]["state"] == "pending_unknown"
    assert report["coverage"]["recorded_reply_context"]["codepoints"] == 0
    assert report["coverage"]["uncertain_attempt_context"]["codepoints"] > 0
    assert dump(db) == before


@pytest.mark.parametrize(
    "kind",
    [
        "no_signal",
        "scan_budget",
        "final_budget",
        "pool_budget",
        "empty_reply",
        "failed_reply",
    ],
)
def test_reports_distinguish_exclusions_empty_and_failure(db_settings, kind):
    command, replay, spec = material()
    if kind in {"no_signal", "empty_source"}:
        command = replace(
            command, messages=(SourceMessage("user", "hello🌱" if kind == "no_signal" else ""),)
        )
        replay = ReplayExtractor(payload_digest(command), ())
    if kind in {"scan_budget", "final_budget", "pool_budget"}:
        changes = {
            "scan_budget": {"max_scan_chars": 0},
            "final_budget": {"max_windows": 0},
            "pool_budget": {"max_windows": 0, "max_pool_windows": 0},
        }[kind]
        spec = replace(
            spec, planning=replace(spec.planning, limits=replace(spec.planning.limits, **changes))
        )
    if kind == "empty_reply":
        reply = ExtractionReply(
            "complete", '{"schema_version":"episode-candidates-v1","episodes":[]}'
        )
        replay = ReplayExtractor(
            replay.source_payload_hash, tuple((i, (reply,)) for i, _ in replay.responses)
        )
    if kind == "failed_reply":
        replay = ReplayExtractor(replay.source_payload_hash, ())
    spec = replace(spec, processor_version=replay.version)
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        services.add.add(command)
        report = repo(services).report_request(command)
        if kind == "empty_source":
            assert all(v["ratio"] is None for v in report["coverage"]["dispositions"].values())
        elif kind == "no_signal":
            assert report["coverage"]["dispositions"]["no_signal"]["ratio"] == 1
        elif kind == "scan_budget":
            assert report["coverage"]["dispositions"]["scan_budget"]["ratio"] == 1
        elif kind == "final_budget":
            assert report["coverage"]["final_budget_excluded_focus"]["ratio"] == 1
        elif kind == "pool_budget":
            assert report["coverage"]["dispositions"]["pool_budget"]["ratio"] == 1
        else:
            assert {w["state"] for w in report["windows"]} == (
                {"empty"} if kind == "empty_reply" else {"attempts_exhausted"}
            )
            assert report["coverage"]["candidate_evidence"]["codepoints"] == 0
            assert report["coverage"]["recorded_reply_context"]["ratio"] == 1
    finally:
        services.lifecycle.close()


def test_v1_report_is_not_upgraded_or_modified(db_settings):
    command, replay, _ = material()
    services = build_services(db_settings, offline_extractor=replay)
    services.lifecycle.open()
    try:
        receipt = services.add.add(command)
        with pytest.raises(ProcessingUnavailable):
            repo(services).report(command.user_id, receipt.batch_id)
    finally:
        services.lifecycle.close()


@pytest.mark.parametrize("kind", ["quote", "attribution", "attempt_hash", "link"])
def test_corrupt_candidate_state_is_rejected_without_repair(coverage, db, kind):
    command, _, _, services = coverage
    services.add.add(command)
    if kind in {"quote", "attribution"}:
        cid, body = db.execute(
            "SELECT id,body FROM episode_candidates ORDER BY id LIMIT 1"
        ).fetchone()
        claim = body["problem"] or body["attempts"][0]["action"]
        if kind == "quote":
            claim["evidence"][0]["quote"] = "synthetic wrong quote"
        else:
            claim["attribution"] = "invented"
        db.execute("UPDATE episode_candidates SET body=%s WHERE id=%s", (Jsonb(body), cid))
    elif kind == "attempt_hash":
        db.execute("UPDATE extraction_attempts SET input_hash='changed'")
    else:
        db.execute("DELETE FROM candidate_windows")
    before = dump(db)
    with pytest.raises(ProcessingUnavailable):
        repo(services).report_request(command)
    assert dump(db) == before


def cli(db_settings, tmp_path, *, report_only=False, extra=(), output="report.json"):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
    )
    args = [
        sys.executable,
        "-m",
        "memory_optimizer.offline",
        "--flow",
        "coverage-offline-v1",
        "--request",
        "examples/offline-add.json",
        "--output",
        str(tmp_path / output),
    ]
    args += (
        ["--report-only"]
        if report_only
        else [
            "--responses",
            "examples/offline-responses.json",
            "--coverage-config",
            "examples/coverage-config.json",
        ]
    )
    return subprocess.run(args + list(extra), env=env, capture_output=True, timeout=20)


def test_real_cli_explicit_flow_export_replay_and_read_only(db_settings, tmp_path, db):
    result = cli(db_settings, tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "coverage-report-v1" and len(report["candidates"]) == 2
    before = dump(db)
    assert cli(db_settings, tmp_path).returncode == 1
    assert cli(db_settings, tmp_path, output="retry.json").returncode == 0
    retry = json.loads((tmp_path / "retry.json").read_text(encoding="utf-8"))
    assert retry.pop("reused") is True
    report.pop("reused")
    assert retry == report
    assert cli(db_settings, tmp_path, report_only=True, output="read.json").returncode == 0
    assert json.loads((tmp_path / "read.json").read_text(encoding="utf-8")) == report
    assert dump(db) == before


def test_export_failure_keeps_committed_work_and_can_export_without_reexecution(
    db_settings, tmp_path, db
):
    result = cli(db_settings, tmp_path, output="absent/report.json")
    assert result.returncode == 1 and b"Traceback" not in result.stderr
    assert db.execute("SELECT stage FROM processing_runs").fetchone() == ("complete",)
    before = dump(db)
    assert cli(db_settings, tmp_path, report_only=True).returncode == 0
    assert dump(db) == before


def test_read_only_cli_missing_request_creates_no_source(db_settings, tmp_path, db):
    assert cli(db_settings, tmp_path, report_only=True).returncode == 1
    assert db.execute("SELECT count(*) FROM batches").fetchone() == (0,)


def test_report_snapshot_does_not_mix_concurrent_candidate_changes(coverage, db):
    command, _, _, services = coverage
    services.add.add(command)
    original = repo(services).report_request(command)

    @contextmanager
    def transaction():
        with services.lifecycle.transaction() as conn:

            class Connection:
                def execute(self, query, params=None):
                    result = conn.execute(query, params)
                    if "SELECT * FROM processing_runs" in query:
                        db.execute("DELETE FROM candidate_windows")
                    return result

            yield Connection()

    assert (
        PostgresCoverageReportRepository(transaction).report(command.user_id, original["batch_id"])
        == original
    )
    with pytest.raises(ProcessingUnavailable):
        repo(services).report_request(command)


@pytest.mark.parametrize(
    "point", ["initialized", "selection_saved", "candidate_reserved", "candidate_saved"]
)
def test_cli_resumes_real_process_checkpoint_and_preserves_report_identity(
    db_settings, tmp_path, db, point
):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
    )
    first = subprocess.run(
        [sys.executable, "tests/helpers/coverage_crash.py", point],
        env=env,
        capture_output=True,
        timeout=20,
    )
    assert first.returncode == 17, first.stderr.decode()
    deadline = db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0]
    old_candidates = db.execute("SELECT id FROM episode_candidates ORDER BY id").fetchall()
    before = dump(db)
    result = cli(db_settings, tmp_path, report_only=True, output="pending.json")
    assert result.returncode == 0, result.stderr.decode()
    assert dump(db) == before
    pending = json.loads((tmp_path / "pending.json").read_text(encoding="utf-8"))
    assert pending["timing"]["deadline_at"] == deadline.isoformat()
    if point == "initialized":
        assert pending["coverage"]["accounted_focus"]["ratio"] == 0
    if point == "selection_saved":
        assert pending["planning"]["selected_windows"] == 3
        assert {w["state"] for w in pending["windows"]} == {"not_started"}
    if point == "candidate_reserved":
        assert pending["windows"][0]["state"] == "pending_unknown"
    expire(db)
    resumed = cli(db_settings, tmp_path)
    assert resumed.returncode == 0, resumed.stderr.decode()
    final = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert final["timing"]["deadline_at"] == deadline.isoformat()
    assert final["run"]["stage"] == "complete"
    assert all(str(cid[0]) in {c["id"] for c in final["candidates"]} for cid in old_candidates)
    if point == "candidate_saved":
        assert len(final["windows"][0]["attempts"]) == 1
    if point == "candidate_reserved":
        assert final["windows"][0]["attempts"][0]["error_code"] == "interrupted"


def test_cli_expired_original_deadline_exports_without_starting_new_work(
    db_settings, tmp_path, db, monkeypatch
):
    command, replay, spec = material(2)
    services = build_coverage_services(db_settings, spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        pause((command, replay, spec, services), monkeypatch, "bind")
        deadline = db.execute("SELECT deadline_at FROM planning_runs").fetchone()[0]
        db.execute("SELECT pg_sleep(1.05)")
        expire(db)
        config = json.loads(Path("examples/coverage-config.json").read_text())
        config["budget"]["total_seconds"] = 2
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        result = cli(db_settings, tmp_path, extra=("--coverage-config", str(path)))
        assert result.returncode == 0, result.stderr.decode()
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["timing"]["deadline_at"] == deadline.isoformat()
        assert report["run"]["reason"] == "time_budget_exhausted"
        assert report["planning"]["selected_windows"] == 3
        assert report["coverage"]["not_started_context"]["ratio"] == 1
        assert db.execute("SELECT count(*) FROM extraction_attempts").fetchone() == (0,)
        before = dump(db)
        assert cli(db_settings, tmp_path, report_only=True, output="expired.json").returncode == 0
        assert dump(db) == before
    finally:
        services.lifecycle.close()


@pytest.mark.parametrize("stage", ["raw", "selected"])
def test_unfinished_report_has_uncommitted_or_unexecuted_ranges(coverage, db, monkeypatch, stage):
    command, _, _, services = coverage
    pause(coverage, monkeypatch, "ensure_work" if stage == "raw" else "bind")
    before = dump(db)
    report = repo(services).report_request(command)
    if stage == "raw":
        assert report["coverage"]["dispositions"]["no_signal"]["status"] == "unknown"
        assert report["coverage"]["selected_focus"]["codepoints"] is None
    else:
        assert report["planning"]["binding_saved"] is False
        assert report["coverage"]["not_started_context"]["ratio"] == 1
    assert dump(db) == before
