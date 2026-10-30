"""Synthetic value/codec/diagnostic contracts for persistence, without a database."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from memory_optimizer.domain import ProcessingUnavailable, SourceMessage
from memory_optimizer.processing import plan_codec
from memory_optimizer.processing.checkpoints import (
    PlanningSnapshot,
    RunBudget,
    attempt_input,
    diagnostic_ranges,
    validate_diagnostic,
)
from memory_optimizer.processing.plan_types import ExecutionDiagnostic, PlanLimits, PlanningSpec
from memory_optimizer.processing.planning import build_plan, select_baseline
from memory_optimizer.processing.types import SourceSpan


def snapshot():
    messages = (SourceMessage("user", "error🌱\r\n", 7), SourceMessage("assistant", "test"))
    spec = PlanningSpec(PlanLimits(10, 1, 2, 100, 50, 3))
    plan = build_plan(messages, spec)
    now = datetime(2026, 10, 26, tzinfo=UTC)
    return PlanningSnapshot(
        "run",
        "batch",
        "u",
        messages,
        plan.source_digest,
        spec,
        RunBudget(60, 4),
        now,
        now + timedelta(seconds=60),
        "plan_ready",
        plan,
        None,
        (),
        None,
    )


@pytest.mark.parametrize(
    "decoder,field",
    [
        (plan_codec.spec, "spec"),
        (plan_codec.plan, "plan"),
        (plan_codec.selection, "selection"),
        (plan_codec.diagnostic, "diagnostic"),
    ],
)
def test_strict_codec_round_trip_preserves_values_fingerprints_and_unknowns(decoder, field):
    saved = snapshot()
    selection = select_baseline(saved.plan, saved.messages)
    values = dict(
        spec=saved.spec,
        plan=saved.plan,
        selection=selection,
        diagnostic=ExecutionDiagnostic("planning", "interrupted", ()),
    )
    original = values[field]
    encoded = json.loads(json.dumps(asdict(original)))
    assert plan_codec.decode(encoded, decoder) == original
    for bad in (
        {**encoded, "future_field": 1},
        {k: v for k, v in encoded.items() if k != next(iter(encoded))},
    ):
        with pytest.raises(ProcessingUnavailable, match="persisted"):
            plan_codec.decode(bad, decoder)


@pytest.mark.parametrize(
    "seconds,attempts", [(True, 1), (0, 1), (86401, 1), (10, True), (10, 0), (10, 33)]
)
def test_run_budget_rejects_implicit_unlimited_or_bool_values(seconds, attempts):
    with pytest.raises(ValueError):
        RunBudget(seconds, attempts)


def test_attempt_identity_binds_run_scope_configuration_deadline_and_phase():
    saved = snapshot()
    first = attempt_input(saved, "planning")
    for changed in [
        replace(saved, run_id="other"),
        replace(saved, batch_id="other"),
        replace(saved, user_id="other"),
        replace(saved, deadline_at=saved.deadline_at + timedelta(seconds=1)),
        replace(saved, budget=RunBudget(60, 5)),
        replace(saved, source_hash="0" * 64),
    ]:
        assert attempt_input(changed, "planning") != first
    assert first != attempt_input(saved, "selection")
    assert first == attempt_input(replace(saved, plan=None, stage="raw_ready"), "planning")


def test_diagnostics_are_bounded_and_completion_requires_exact_checkpoint_ranges():
    saved = snapshot()
    ranges = (SourceSpan(0, 0, 8), SourceSpan(1, 0, 4))
    assert diagnostic_ranges(saved, "planning") == ranges
    valid = ExecutionDiagnostic("planning", "completed", ranges, saved.plan.fingerprint)
    validate_diagnostic(valid, saved, plan=saved.plan)
    for bad in [
        replace(valid, inspected=()),
        replace(valid, plan_fingerprint="0" * 64),
        replace(valid, inspected=(SourceSpan(0, 0, 9),)),
        replace(valid, inspected=(ranges[1], ranges[0])),
    ]:
        with pytest.raises(ValueError):
            validate_diagnostic(bad, saved, plan=saved.plan)
    validate_diagnostic(ExecutionDiagnostic("planning", "failed", (SourceSpan(0, 0, 2),)), saved)
    with pytest.raises(ValueError):
        validate_diagnostic(
            ExecutionDiagnostic("planning", "failed", (SourceSpan(9, 0, 2),)), saved
        )
    final = select_baseline(saved.plan, saved.messages)
    d = ExecutionDiagnostic(
        "selection",
        "completed",
        diagnostic_ranges(saved, "selection", final),
        saved.plan.fingerprint,
    )
    validate_diagnostic(d, saved, selection=final)
    with pytest.raises(ValueError):
        validate_diagnostic(d, saved)


def test_persisted_unknown_version_and_invalid_spans_never_gain_defaults():
    value = json.loads(json.dumps(asdict(snapshot().plan)))
    value["spec"]["flow_version"] = "future"
    with pytest.raises(ProcessingUnavailable):
        plan_codec.decode(value, plan_codec.plan)
    value = json.loads(json.dumps(asdict(snapshot().plan)))
    value["coverage"][0]["span"]["start"] = True
    with pytest.raises(ProcessingUnavailable):
        plan_codec.decode(value, plan_codec.plan)
