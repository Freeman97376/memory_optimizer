import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from memory_optimizer.bootstrap import build_services
from memory_optimizer.domain import (
    ProcessingBusy,
    ProcessingUnavailable,
    SearchQuery,
    StorageUnavailable,
    payload_digest,
)
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.extraction.types import prepare_input
from memory_optimizer.extraction.validation import validate_reply
from memory_optimizer.extraction.workflow import OfflineCandidateFlow

from .test_extraction import EXAMPLES, offline_material
from .test_processing_postgres import Interrupt, expire, state

pytestmark = pytest.mark.integration


@pytest.fixture
def offline(db_settings):
    command, replay = offline_material()
    services = build_services(db_settings, offline_extractor=replay)
    services.lifecycle.open()
    try:
        yield command, replay, services
    finally:
        services.lifecycle.close()


def stage_repo(services):
    return services.add.processing.processors["offline-candidates-v1"].repository


def responses_with(first_replies):
    data = json.loads((EXAMPLES / "offline-responses.json").read_text())
    data["windows"][0]["replies"] = first_replies
    return ReplayExtractor.from_json(json.dumps(data))


def test_offline_end_to_end_deduplicates_only_exact_candidates_and_keeps_raw(offline, db):
    command, _, services = offline
    receipt = services.add.add(command)
    repo = stage_repo(services)
    report = repo.report(command.user_id, receipt.batch_id)
    assert report["run"]["reason"] == "offline_only" and report["run"]["model_calls"] == 0
    assert [a["status"] for a in report["attempts"]] == ["accepted"] * 3
    assert len(report["candidates"]) == 2
    assert all(c["semantic_status"] == "unreviewed" for c in report["candidates"])
    assert db.execute("SELECT count(*) FROM candidate_windows").fetchone()[0] == 3
    assert services.search.search(SearchQuery("rollback", command.user_id))
    with pytest.raises(ProcessingUnavailable):
        repo.report("another-user", receipt.batch_id)
    assert services.add.add(command).reused
    assert repo.report(command.user_id, receipt.batch_id) == report


@pytest.mark.parametrize("mode", ["retry", "exhausted", "empty"])
def test_retry_empty_and_failure_are_durable_distinct_outcomes(db_settings, db, mode):
    command, original = offline_material()
    good = dict(status="complete", content=original.responses[0][1][0].content)
    bad = dict(status="truncated", content=good["content"])
    replies = (
        [bad, good]
        if mode == "retry"
        else (
            [bad, bad]
            if mode == "exhausted"
            else [
                dict(
                    status="complete",
                    content='{"schema_version":"episode-candidates-v1","episodes":[]}',
                )
            ]
        )
    )
    services = build_services(db_settings, offline_extractor=responses_with(replies))
    services.lifecycle.open()
    try:
        receipt = services.add.add(command)
        report = stage_repo(services).report(command.user_id, receipt.batch_id)
        first = [a for a in report["attempts"] if a["ordinal"] == 0]
        expected = {
            "retry": ["failed", "accepted"],
            "exhausted": ["failed", "failed"],
            "empty": ["empty"],
        }
        assert [a["status"] for a in first] == expected[mode]
        if mode != "empty":
            assert first[0]["error_code"] == "truncated"
        assert report["run"]["reason"] == (
            "extraction_failed" if mode == "exhausted" else "offline_only"
        )
        if mode != "retry":
            assert (
                db.execute(
                    "SELECT count(*) FROM candidate_windows cw JOIN processing_windows w "
                    "ON w.id=cw.window_id WHERE w.ordinal=0"
                ).fetchone()[0]
                == 0
            )
    finally:
        services.lifecycle.close()


def test_invalid_reference_rejects_whole_window_and_never_publishes_partial_candidate(
    db_settings, db
):
    command, original = offline_material()
    value = json.loads(original.responses[0][1][0].content)
    value["episodes"] *= 2
    value["episodes"][1] = dict(
        value["episodes"][1],
        problem={
            "text": "private-sentinel",
            "evidence": [{"message_ordinal": 99, "start": 0, "end": 1, "quote": "x"}],
        },
    )
    bad = dict(status="complete", content=json.dumps(value))
    services = build_services(db_settings, offline_extractor=responses_with([bad, bad]))
    services.lifecycle.open()
    try:
        receipt = services.add.add(command)
        report = stage_repo(services).report(command.user_id, receipt.batch_id)
        assert report["run"]["reason"] == "extraction_failed"
        assert "private-sentinel" not in json.dumps(report)
        assert report["attempts"][0]["error_code"] == "reference_not_provided"
        assert (
            db.execute(
                "SELECT count(*) FROM candidate_windows cw JOIN processing_windows w "
                "ON w.id=cw.window_id WHERE w.ordinal=0"
            ).fetchone()[0]
            == 0
        )
    finally:
        services.lifecycle.close()


def test_same_evidence_different_paraphrases_remain_unreviewed_and_out_of_search(db_settings):
    command, original = offline_material()
    value = json.loads(original.responses[0][1][0].content)
    value["episodes"][0]["problem"]["text"] = "paraphraseonlytoken"
    services = build_services(
        db_settings,
        offline_extractor=responses_with([dict(status="complete", content=json.dumps(value))]),
    )
    services.lifecycle.open()
    try:
        receipt = services.add.add(command)
        candidates = stage_repo(services).report(command.user_id, receipt.batch_id)["candidates"]
        assert len(candidates) == 3
        assert len({c["evidence_key"] for c in candidates}) == 2
        assert services.search.search(SearchQuery("paraphraseonlytoken", command.user_id)) == []
    finally:
        services.lifecycle.close()


def test_candidate_commit_failure_rolls_back_candidates_links_and_attempt_result(offline, db):
    command, _, services = offline
    db.execute("""CREATE FUNCTION fail_candidate() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'private-sentinel'; END $$""")
    db.execute(
        "CREATE CONSTRAINT TRIGGER fail_candidate AFTER UPDATE ON extraction_attempts "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN (NEW.status = 'accepted') "
        "EXECUTE FUNCTION fail_candidate()"
    )
    with pytest.raises(StorageUnavailable):
        services.add.add(command)
    assert state(db)["stage"] == "selected"
    assert db.execute("SELECT status FROM extraction_attempts").fetchall() == [("reserved",)]
    assert db.execute("SELECT count(*) FROM episode_candidates").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM candidate_windows").fetchone()[0] == 0
    db.execute("DROP TRIGGER fail_candidate ON extraction_attempts")
    expire(db)
    receipt = services.add.add(command)
    report = stage_repo(services).report(command.user_id, receipt.batch_id)
    assert report["attempts"][0]["error_code"] == "interrupted"
    assert report["attempts"][1]["status"] == "accepted" and len(report["candidates"]) == 2


def test_reservation_failure_cannot_call_extractor(offline, db, monkeypatch):
    command, _, services = offline
    db.execute("""CREATE FUNCTION fail_reserve() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'private-sentinel'; END $$""")
    db.execute(
        "CREATE TRIGGER fail_reserve BEFORE INSERT ON extraction_attempts "
        "FOR EACH ROW EXECUTE FUNCTION fail_reserve()"
    )

    def forbidden(*args):
        raise AssertionError("no call before committed reservation")

    monkeypatch.setattr(ReplayExtractor, "extract", forbidden)
    with pytest.raises(StorageUnavailable):
        services.add.add(command)
    assert db.execute("SELECT count(*) FROM extraction_attempts").fetchone()[0] == 0


def test_terminal_commit_failure_reuses_candidates_without_extraction(offline, db, monkeypatch):
    command, _, services = offline
    db.execute("""CREATE FUNCTION fail_terminal() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'private-sentinel'; END $$""")
    db.execute(
        "CREATE CONSTRAINT TRIGGER fail_terminal AFTER UPDATE ON processing_runs "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN (NEW.stage = 'complete') "
        "EXECUTE FUNCTION fail_terminal()"
    )
    with pytest.raises(StorageUnavailable):
        services.add.add(command)
    assert state(db)["stage"] == "selected"
    before = db.execute("SELECT id, body FROM episode_candidates ORDER BY id").fetchall()
    assert len(before) == 2
    db.execute("DROP TRIGGER fail_terminal ON processing_runs")
    expire(db)

    def forbidden(*args):
        raise AssertionError("completed windows must not call extractor")

    monkeypatch.setattr(ReplayExtractor, "extract", forbidden)
    services.add.add(command)
    assert state(db)["stage"] == "complete"
    assert db.execute("SELECT id, body FROM episode_candidates ORDER BY id").fetchall() == before


def selected(offline, db, monkeypatch):
    command, replay, services = offline
    with monkeypatch.context() as patch:

        def interrupt(*args):
            raise Interrupt()

        patch.setattr(OfflineCandidateFlow, "process", interrupt)
        with pytest.raises(Interrupt):
            services.add.add(command)
    expire(db)
    lease = services.add.processing.repository.claim(command.user_id, state(db)["batch_id"], 30)
    repo = stage_repo(services)
    window_id, window = repo.windows(lease)[0]
    request = prepare_input(lease, window_id, window, command.messages, payload_digest(command))
    return lease, request, repo


def test_candidate_storage_fences_stale_wrong_scope_and_changed_inputs(offline, db, monkeypatch):
    command, replay, services = offline
    lease, request, repo = selected(offline, db, monkeypatch)
    ticket = repo.reserve(lease, request)
    candidates = validate_reply(replay.extract(request, ticket.number), request)
    with pytest.raises(ProcessingBusy):
        repo.save(replace(lease, user_id="other"), ticket, candidates, None)
    with pytest.raises(ProcessingUnavailable):
        repo.reserve(lease, replace(request, user_id="other"))
    with pytest.raises(ProcessingUnavailable):
        repo.reserve(lease, replace(request, payload_hash="changed"))
    with pytest.raises(ProcessingUnavailable):
        repo.finish(lease)
    expire(db)
    new = services.add.processing.repository.claim(command.user_id, lease.batch_id, 30)
    with pytest.raises(ProcessingBusy):
        repo.save(lease, ticket, candidates, None)
    replacement = repo.reserve(new, request)
    assert replacement.number == 2
    repo.save(new, replacement, candidates, None)
    assert db.execute(
        "SELECT status,error_code FROM extraction_attempts ORDER BY number"
    ).fetchall() == [("failed", "interrupted"), ("accepted", None)]


def test_missing_or_changed_processor_cannot_finish_old_pending_version(
    offline, db, monkeypatch, db_settings
):
    command, replay, _ = offline
    selected(offline, db, monkeypatch)
    expire(db)
    changed = ReplayExtractor(replay.source_payload_hash, ())
    for extractor in (None, changed):
        other = build_services(db_settings, offline_extractor=extractor)
        other.lifecycle.open()
        try:
            with pytest.raises(ProcessingUnavailable):
                other.add.add(command)
            assert state(db)["stage"] == "selected"
        finally:
            other.lifecycle.close()
            expire(db)


def test_repeated_interruptions_exhaust_persisted_limit_without_third_call(
    offline, db, monkeypatch
):
    command, _, services = offline
    lease, request, repo = selected(offline, db, monkeypatch)
    assert repo.reserve(lease, request).number == 1
    expire(db)
    second = services.add.processing.repository.claim(command.user_id, lease.batch_id, 30)
    assert repo.reserve(second, request).number == 2
    expire(db)
    third = services.add.processing.repository.claim(command.user_id, lease.batch_id, 30)
    assert repo.reserve(third, request) is None
    assert db.execute(
        "SELECT status,error_code FROM extraction_attempts ORDER BY number"
    ).fetchall() == [("failed", "interrupted"), ("failed", "interrupted")]
    original = ReplayExtractor.extract

    def no_third_call(self, value, attempt):
        if value.window.ordinal == 0:
            raise AssertionError("exhausted window cannot call again")
        return original(self, value, attempt)

    monkeypatch.setattr(ReplayExtractor, "extract", no_third_call)
    expire(db)
    services.add.add(command)
    assert state(db)["reason"] == "extraction_failed"


@pytest.mark.parametrize("phase", ["reserve", "save"])
def test_candidate_transaction_expiry_cannot_commit_partial_state(offline, db, monkeypatch, phase):
    _, replay, _ = offline
    lease, request, repo = selected(offline, db, monkeypatch)
    ticket = repo.reserve(lease, request) if phase == "save" else None
    before = state(db)
    db.execute("""CREATE FUNCTION expire_candidate_owner() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN UPDATE processing_runs SET lease_until = clock_timestamp() - interval '1 second'
          WHERE id = NEW.run_id; RETURN NEW; END $$""")
    table = "extraction_attempts" if phase == "reserve" else "episode_candidates"
    db.execute(
        f"CREATE TRIGGER expire_candidate_owner AFTER INSERT ON {table} "
        "FOR EACH ROW EXECUTE FUNCTION expire_candidate_owner()"
    )
    with pytest.raises(ProcessingBusy):
        if phase == "reserve":
            repo.reserve(lease, request)
        else:
            candidates = validate_reply(replay.extract(request, ticket.number), request)
            repo.save(lease, ticket, candidates, None)
    assert state(db) == before
    assert db.execute("SELECT count(*) FROM episode_candidates").fetchone()[0] == 0
    assert db.execute("SELECT status FROM extraction_attempts").fetchall() == (
        [] if phase == "reserve" else [("reserved",)]
    )


@pytest.mark.parametrize("point", ["reserve", "save"])
def test_fresh_process_resumes_reserved_or_completed_window(db_settings, db, point):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
        MEMORY_API_KEY="local-test-key",
    )
    helper = Path(__file__).parent / "helpers/extraction_crash.py"
    first = subprocess.run(
        [sys.executable, str(helper), point], env=env, capture_output=True, timeout=15
    )
    assert first.returncode == 17, first.stderr.decode()
    before = db.execute("SELECT id FROM episode_candidates ORDER BY id").fetchall()
    expire(db)
    second = subprocess.run(
        [sys.executable, str(helper), "resume-" + point], env=env, capture_output=True, timeout=15
    )
    assert second.returncode == 0, second.stderr.decode()
    assert state(db)["stage"] == "complete" and state(db)["reason"] == "offline_only"
    if point == "save":
        assert len(before) == 1
        assert set(before) <= set(db.execute("SELECT id FROM episode_candidates").fetchall())
        assert db.execute("SELECT count(*) FROM extraction_attempts").fetchone()[0] == 3
    else:
        assert (
            db.execute(
                "SELECT count(*) FROM extraction_attempts WHERE error_code='interrupted'"
            ).fetchone()[0]
            == 1
        )


def test_real_cli_writes_reviewable_report_and_refuses_overwrite(db_settings, tmp_path):
    env = os.environ.copy()
    env.update(
        MEMORY_DATABASE_URL=db_settings.database_url.get_secret_value(),
        MEMORY_DATABASE_SCHEMA=db_settings.schema,
        MEMORY_API_KEY="local-test-key",
    )
    output = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m",
        "memory_optimizer.offline",
        "--request",
        str(EXAMPLES / "offline-add.json"),
        "--responses",
        str(EXAMPLES / "offline-responses.json"),
        "--output",
        str(output),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    before = output.read_bytes()
    report = json.loads(before)
    assert len(report["candidates"]) == 2 and report["run"]["model_calls"] == 0
    again = subprocess.run(cmd, env=env, capture_output=True, timeout=15)
    assert again.returncode == 1 and output.read_bytes() == before
