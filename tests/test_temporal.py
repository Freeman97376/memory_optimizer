import copy
import json
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta

import pytest

from memory_optimizer.domain import ProcessingUnavailable, payload_digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.extraction.types import (
    SCHEMA_VERSION,
    CandidateValidationError,
    EvidenceText,
    ExtractionReply,
)
from memory_optimizer.extraction.validation import validate_reply
from memory_optimizer.processing.checkpoints import PlanningLease
from memory_optimizer.processing.coverage_types import CoverageLease, candidate_input, final_windows
from memory_optimizer.processing.coverage_workflow import CoverageFlow
from memory_optimizer.processing.planning import build_plan, select_baseline
from memory_optimizer.processing.types import SourceSpan, Window
from memory_optimizer.temporal.proposals import raw_proposals, validate_temporal_reply

from .helpers.temporal_material import material
from .test_plan_checkpoints import snapshot


def inputs():
    command, replay, spec = material()
    plan = build_plan(command.messages, spec.planning)
    saved = replace(
        snapshot(),
        messages=command.messages,
        spec=spec.planning,
        budget=spec.budget,
        plan=plan,
        source_hash=plan.source_digest,
        selection=select_baseline(plan, command.messages),
    )
    saved = replace(
        saved, deadline_at=saved.created_at + timedelta(seconds=spec.budget.total_seconds)
    )
    lease = CoverageLease(
        "r", "batch", "u", "owner", 1, spec, PlanningLease("run", "batch", "u", "owner", 1), 30
    )
    request = candidate_input(lease, "w", final_windows(saved)[0], saved, payload_digest(command))
    envelope = json.loads(replay.extract(request, 1).content)
    return request, envelope


def validate(request, envelope):
    return validate_temporal_reply(ExtractionReply("complete", canonical(envelope)), request)


def test_provenance_binding_conversion_and_each_event_are_separate():
    request, body = inputs()
    (result,) = validate(request, body)
    assert [
        (t.provenance_status, t.binding_status, t.conversion_status) for t in result.events
    ] == [("source_checked", "proposed", "disabled"), ("absent", "unknown", "disabled")]
    assert result.events[0].raw_time.text == "Yesterday"
    assert all(e.normalized_value is None and e.sort_key is None for e in result.events)
    assert result.proposals[0].time.evidence[0].role == "assistant"
    assert result.proposals[0].time.evidence[0].source_timestamp is None
    assert result.body()["binding_rules"] == "disabled"
    restored = copy.deepcopy(body)
    restored["episodes"][0]["time_proposals"] = raw_proposals(
        json.loads(canonical(result.body()))["proposals"]
    )
    assert validate(request, restored) == (result,)


def test_raw_time_with_no_proposal_stays_unknown_and_v1_body_is_unchanged():
    request, body = inputs()
    body["episodes"][0]["time_proposals"] = []
    (result,) = validate(request, body)
    (old,) = validate_reply(
        ExtractionReply(
            "complete",
            canonical(
                {"schema_version": SCHEMA_VERSION, "episodes": [body["episodes"][0]["candidate"]]}
            ),
        ),
        replace(request, schema_version=SCHEMA_VERSION),
    )
    assert result.candidate == old
    assert result.events[0].binding_status == "unknown"
    assert result.events[0].binding_reason == "raw_mention_only"


@pytest.mark.parametrize(
    "case",
    ["unknown_context", "missing_relation", "different_time", "different_context", "planned_event"],
)
def test_ambiguity_and_conflict_never_create_sort_keys(case):
    request, body = inputs()
    item = body["episodes"][0]
    proposal = item["time_proposals"][0]
    if case == "unknown_context":
        proposal["context"] = "unknown"
    if case == "missing_relation":
        proposal["relation_evidence"] = []
    if case == "different_time":
        proposal["time"] = {
            "text": "Tests",
            "evidence": [dict(message_ordinal=1, start=44, end=49, quote="Tests")],
        }
    if case == "different_context":
        other = copy.deepcopy(proposal)
        other["context"] = "quoted"
        item["time_proposals"].append(other)
    if case == "planned_event":
        item["candidate"]["attempts"][0]["kind"] = "proposed"
    (result,) = validate(request, body)
    assert result.events[0].binding_status == (
        "ambiguous" if case in {"unknown_context", "missing_relation"} else "conflict"
    )
    assert result.events[0].sort_key is None


@pytest.mark.parametrize(
    "case",
    [
        "bad_quote",
        "wrong_target",
        "boolean_target",
        "observation_target",
        "outside_event",
        "missing_time",
        "extra_field",
        "empty_relation_type",
        "invented_normalization",
        "duplicate",
        "too_many",
        "foreign_id",
        "bad_context",
        "bad_anchor",
    ],
)
def test_malformed_or_unsupported_proposals_rejected(case):
    request, body = inputs()
    item = body["episodes"][0]
    p = item["time_proposals"][0]
    if case == "bad_quote":
        p["time"]["evidence"][0]["quote"] = "invented"
    if case == "wrong_target":
        p["target"]["attempt"] = 99
    if case == "boolean_target":
        p["target"]["attempt"] = True
    if case == "observation_target":
        p["target"]["observation"] = 99
    if case == "outside_event":
        p["event_evidence"] = p["time"]["evidence"]
    if case == "missing_time":
        del p["time"]
    if case == "extra_field":
        p["confidence"] = 1.0
    if case == "empty_relation_type":
        p["relation_evidence"] = ""
    if case == "invented_normalization":
        p["normalized_value"] = "2026-10-21"
    if case == "duplicate":
        item["time_proposals"].append(copy.deepcopy(p))
    if case == "too_many":
        item["time_proposals"] = [copy.deepcopy(p)] * 33
    if case == "foreign_id":
        p["target"] = {"candidate_id": "external", "attempt": 0, "observation": None}
    if case == "bad_context":
        p["context"] = "verified"
    if case == "bad_anchor":
        p["anchor"] = {"text": "2020", "evidence": p["time"]["evidence"]}
    with pytest.raises(CandidateValidationError):
        validate(request, body)


def test_relation_quote_must_include_both_event_and_time():
    request, body = inputs()
    p = body["episodes"][0]["time_proposals"][0]
    p["relation_evidence"] = p["time"]["evidence"]
    with pytest.raises(CandidateValidationError, match="incomplete_relation"):
        validate(request, body)


def test_outer_source_timestamp_remains_raw_even_outside_datetime_range():
    request, body = inputs()
    request = replace(
        request,
        sources=tuple(replace(s, source_timestamp=9223372036854775807) for s in request.sources),
    )
    (result,) = validate(request, body)
    assert result.proposals[0].time.evidence[0].source_timestamp == 9223372036854775807
    assert result.events[0].sort_key is None


@pytest.mark.parametrize("direction", ["old_reply", "old_input", "old_validator", "old_flow"])
def test_versions_do_not_implicitly_upgrade(direction):
    request, body = inputs()
    with pytest.raises((CandidateValidationError, ProcessingUnavailable)):
        if direction == "old_reply":
            body["schema_version"] = SCHEMA_VERSION
            validate(request, body)
        elif direction == "old_input":
            validate(replace(request, schema_version=SCHEMA_VERSION), body)
        elif direction == "old_validator":
            validate_reply(ExtractionReply("complete", canonical(body)), request)
        else:
            _, replay, spec = material()
            CoverageFlow(None, spec, replay)


@pytest.mark.parametrize("status", ["error", "refused", "truncated"])
def test_adapter_failure_does_not_become_valid_unknown(status):
    request, _ = inputs()
    with pytest.raises(CandidateValidationError, match=status):
        validate_temporal_reply(ExtractionReply(status), request)


def test_pure_temporal_import_has_no_database_http_or_model_dependency():
    script = (
        "import sys; "
        "sys.modules.update({k:None for k in ('psycopg','fastapi','uvicorn','openai')}); "
        "import memory_optimizer.temporal.proposals; import memory_optimizer.temporal.report"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10)
    assert done.returncode == 0, done.stderr.decode()


@pytest.mark.parametrize(
    "phrase",
    [
        "10月21日",
        "昨天",
        "2026-02-30",
        "2099-10-21",
        "2026-11-01T01:30",
        "2026-10-21T09:00:00Z",
        "2030-01-01",
        "明年🌱",
    ],
)
def test_partial_relative_invalid_future_quoted_dates_never_normalize(phrase):
    request, _ = inputs()
    source = f"Quoted log: modified cache.py on {phrase}. Competition deadline 2030-01-01."
    span = SourceSpan(0, 0, len(source))
    request = replace(
        request,
        window=Window(0, span, (span,), ("change",)),
        sources=(EvidenceText(span, source, "user", 9223372036854775807),),
    )

    def ref(text):
        start = source.index(text)
        return dict(message_ordinal=0, start=start, end=start + len(text), quote=text)

    action = dict(text="modified cache.py", evidence=[ref("modified cache.py")])
    time = dict(text=phrase, evidence=[ref(phrase)])
    candidate = dict(
        problem=None,
        conditions=[],
        objects=[],
        attempts=[
            dict(
                action=action,
                kind="reported_change",
                event_time=None,
                before=None,
                after=None,
                object_refs=[],
                observations=[],
            )
        ],
    )
    proposal = dict(
        target=dict(attempt=0, observation=None),
        time=time,
        event_evidence=action["evidence"],
        relation_evidence=[ref(source)],
        context="quoted",
        subject=None,
        branch=None,
        version=None,
        anchor=None,
        timezone=None,
    )
    (result,) = validate(
        request,
        dict(
            schema_version="episode-temporal-v1",
            episodes=[dict(candidate=candidate, time_proposals=[proposal])],
        ),
    )
    assert result.events[0].binding_status == "proposed"
    assert result.events[0].conversion_status == "disabled" and result.events[0].sort_key is None
    assert result.proposals[0].time.text == phrase
    assert result.proposals[0].context == "quoted"


def test_valid_observation_target_does_not_change_action_time():
    request, body = inputs()
    item = body["episodes"][0]
    proposal = item["time_proposals"][0]
    proposal["target"]["observation"] = 0
    proposal["event_evidence"] = item["candidate"]["attempts"][0]["observations"][0]["content"][
        "evidence"
    ]
    (result,) = validate(request, body)
    assert result.events[0].binding_status == "unknown"
    assert result.events[1].binding_status == "proposed"


@pytest.mark.parametrize("field", ["subject", "branch", "version", "anchor", "timezone"])
def test_disagreeing_qualifiers_or_anchors_are_retained_as_conflict(field):
    request, body = inputs()
    item = body["episodes"][0]
    proposal = item["time_proposals"][0]
    evidence = proposal["relation_evidence"]
    proposal[field] = {"text": "cache.py", "evidence": evidence}
    other = copy.deepcopy(proposal)
    other[field]["text"] = "lock"
    item["time_proposals"].append(other)
    (result,) = validate(request, body)
    assert result.events[0].binding_status == "conflict"
    assert len(result.proposals) == 2 and result.events[0].sort_key is None


def test_full_32_proposal_bound_preserves_each_without_picking_a_winner():
    request, body = inputs()
    item = body["episodes"][0]
    base = item["time_proposals"][0]
    relation = base["relation_evidence"]
    text = relation[0]["quote"]
    item["time_proposals"] = []
    for size in range(1, 33):
        proposal = copy.deepcopy(base)
        proposal["subject"] = {"text": text[:size], "evidence": relation}
        item["time_proposals"].append(proposal)
    (result,) = validate(request, body)
    assert len(result.proposals) == 32
    assert result.events[0].binding_status == "conflict"
    assert result.events[0].sort_key is None
