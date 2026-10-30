"""Contract cases from hand-authored synthetic sources, not model-quality evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.domain import SourceMessage, payload_digest
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.extraction.types import (
    CandidateValidationError,
    ExtractionReply,
    prepare_input,
)
from memory_optimizer.extraction.validation import validate_reply
from memory_optimizer.models import AddRequest
from memory_optimizer.processing.selection import select_windows
from memory_optimizer.processing.types import ProcessingSpec, RunLease, SelectionConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def offline_material():
    command = ingest_command(
        AddRequest.model_validate_json((EXAMPLES / "offline-add.json").read_text())
    )
    replay = ReplayExtractor.from_json((EXAMPLES / "offline-responses.json").read_text())
    return command, replay


def request_input(ordinal=0):
    command, replay = offline_material()
    spec = ProcessingSpec(flow_version="offline-candidates-v1", processor_version=replay.version)
    lease = RunLease("run", "batch", command.user_id, "owner", 1, "selected", spec)
    window = select_windows(command.messages, spec.selection).windows[ordinal]
    return prepare_input(lease, str(ordinal), window, command.messages, payload_digest(command))


def response_body():
    _, replay = offline_material()
    return json.loads(replay.extract(request_input(), 1).content)


def test_multi_attempts_unknown_time_and_reported_verification_are_preserved():
    _, replay = offline_material()
    first = validate_reply(replay.extract(request_input(), 1), request_input())[0].candidate
    assert first.problem.text == "duplicate writes"
    assert first.problem.evidence[0].source_timestamp == 1792627200000
    action = first.attempts[0]
    assert action.event_time.text == "Yesterday"
    assert action.event_time.evidence[0].source_timestamp is None
    assert action.before is None and action.after is None
    assert action.observations[0].content.attribution == "assistant_report"
    assert not hasattr(action.observations[0], "verified")
    last_request = request_input(2)
    last = validate_reply(replay.extract(last_request, 1), last_request)[0].candidate
    assert last.problem is None and len(last.attempts) == 2
    assert last.attempts[0].observations[1].corrects_attempt == 0
    assert last.attempts[1].kind == "rollback"
    assert last.attempts[1].event_time is None and last.attempts[1].observations == ()


def test_overlapping_windows_share_candidate_identity_but_not_input_identity():
    _, replay = offline_material()
    a, b = request_input(0), request_input(1)
    assert a.fingerprint != b.fingerprint
    assert validate_reply(replay.extract(a, 1), a) == validate_reply(replay.extract(b, 1), b)
    body = response_body()
    body["episodes"] *= 2
    assert len(validate_reply(ExtractionReply("complete", json.dumps(body)), a)) == 1


def test_complete_empty_result_is_valid_and_missing_replay_is_not_empty():
    request = request_input()
    reply = ExtractionReply("complete", '{"schema_version":"episode-candidates-v1","episodes":[]}')
    assert validate_reply(reply, request) == ()
    _, replay = offline_material()
    for broken in (
        replace(request, payload_hash="other"),
        replace(request, window=replace(request.window, ordinal=99)),
    ):
        with pytest.raises(CandidateValidationError, match="error"):
            validate_reply(replay.extract(broken, 1), broken)


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("scope", "invalid_fields"),
        ("missing", "invalid_fields"),
        ("bool_offset", "invalid_reference"),
        ("negative", "reference_not_provided"),
        ("quote", "quote_mismatch"),
        ("outside_context", "reference_not_provided"),
        ("schema", "unsupported_schema"),
        ("verified", "invalid_fields"),
        ("no_evidence", "invalid_count"),
        ("kind", "invalid_enum"),
        ("object", "invalid_reference"),
        ("correction", "invalid_reference"),
        ("time", "nonverbatim_value"),
        ("too_many", "invalid_count"),
        ("nul", "invalid_text"),
        ("surrogate", "invalid_text"),
    ],
)
def test_invalid_candidate_response_is_rejected_as_a_whole(mutation, code):
    body = response_body()
    episode = body["episodes"][0]
    attempt = episode["attempts"][0]
    reference = episode["problem"]["evidence"][0]
    if mutation == "scope":
        body["user_id"] = "other-user"
    elif mutation == "missing":
        del attempt["before"]
    elif mutation == "bool_offset":
        reference["start"] = True
    elif mutation == "negative":
        reference["start"] = -1
    elif mutation == "quote":
        reference["quote"] = "private-sentinel"
    elif mutation == "outside_context":
        reference.update(message_ordinal=2, start=0, end=8, quote="The lock")
    elif mutation == "schema":
        body["schema_version"] = "future"
    elif mutation == "verified":
        attempt["observations"][0]["verified"] = True
    elif mutation == "no_evidence":
        episode["problem"]["evidence"] = []
    elif mutation == "kind":
        attempt["kind"] = "verified_done"
    elif mutation == "object":
        attempt["object_refs"] = [9]
    elif mutation == "correction":
        attempt["observations"][0].update(kind="correction", corrects_attempt=9)
    elif mutation == "time":
        attempt["event_time"]["text"] = "2026-10-22T10:00:00Z"
    elif mutation == "too_many":
        body["episodes"] *= 9
    elif mutation == "nul":
        episode["problem"]["text"] = "\x00"
    elif mutation == "surrogate":
        episode["problem"]["text"] = "\ud800"
    with pytest.raises(CandidateValidationError) as error:
        validate_reply(ExtractionReply("complete", json.dumps(body)), request_input())
    assert error.value.code == code and "private-sentinel" not in str(error.value)


@pytest.mark.parametrize(
    "reply,code",
    [
        (
            ExtractionReply(
                "truncated", '{"schema_version":"episode-candidates-v1","episodes":[]}'
            ),
            "truncated",
        ),
        (ExtractionReply("refused"), "refused"),
        (ExtractionReply("error"), "error"),
        (ExtractionReply([], None), "invalid_status"),
        (ExtractionReply("complete", '{"episodes":[],"episodes":[]}'), "duplicate_json_key"),
        (ExtractionReply("complete", '{"episodes":['), "invalid_json"),
        (ExtractionReply("complete", "NaN"), "invalid_json"),
        (ExtractionReply("complete", "[" * 2000 + "]" * 2000), "invalid_json"),
        (ExtractionReply("complete", "x" * (64 * 1024 + 1)), "response_size"),
    ],
)
def test_incomplete_or_malformed_output_never_becomes_valid_empty(reply, code):
    with pytest.raises(CandidateValidationError) as error:
        validate_reply(reply, request_input())
    assert error.value.code == code


def test_validation_checks_provenance_without_claiming_paraphrase_truth():
    body = response_body()
    body["episodes"][0]["problem"]["text"] = "This paraphrase still requires semantic review."
    result = validate_reply(ExtractionReply("complete", json.dumps(body)), request_input())
    assert result[0].candidate.problem.attribution == "user_report"


def test_identical_source_text_at_different_positions_remains_distinct():
    command, replay = offline_material()
    messages = (SourceMessage("user", "error 🌱\r\n"), SourceMessage("user", "error 🌱\r\n"))
    spec = ProcessingSpec(
        flow_version="offline-candidates-v1",
        processor_version=replay.version,
        selection=SelectionConfig(context_chars=0),
    )
    lease = RunLease("run", "batch", command.user_id, "owner", 1, "selected", spec)
    candidates = []
    for ordinal, window in enumerate(select_windows(messages, spec.selection).windows):
        request = prepare_input(lease, str(ordinal), window, messages, "payload")
        claim = {
            "text": "error",
            "evidence": [
                {"message_ordinal": ordinal, "start": 0, "end": 9, "quote": "error 🌱\r\n"}
            ],
        }
        body = {
            "schema_version": "episode-candidates-v1",
            "episodes": [{"problem": claim, "conditions": [], "objects": [], "attempts": []}],
        }
        candidates.append(validate_reply(ExtractionReply("complete", json.dumps(body)), request)[0])
    assert candidates[0].fingerprint != candidates[1].fingerprint
    assert candidates[0].evidence_key != candidates[1].evidence_key


def test_neighbor_context_alone_cannot_create_an_unrelated_focus_candidate():
    body = response_body()
    body["episodes"][0].update(problem=None, conditions=[], objects=[])
    body["episodes"][0]["attempts"][0]["object_refs"] = []
    # All remaining evidence is in m1, while this window focuses m0.
    with pytest.raises(CandidateValidationError, match="outside_focus"):
        validate_reply(ExtractionReply("complete", json.dumps(body)), request_input())
