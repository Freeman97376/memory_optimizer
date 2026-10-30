"""Synthetic pure configuration and candidate adapter contracts."""

import json
from dataclasses import asdict, replace
from datetime import timedelta, timezone

import pytest

from memory_optimizer.domain import ProcessingUnavailable
from memory_optimizer.processing.checkpoints import PlanningLease, RunBudget
from memory_optimizer.processing.coverage_types import (
    CoverageLease,
    CoverageSpec,
    candidate_input,
    final_windows,
    read_coverage_spec,
)
from memory_optimizer.processing.planning import select_baseline

from .test_plan_checkpoints import snapshot


@pytest.mark.parametrize(
    "updates",
    [
        {"cleanup_seconds": 0},
        {"cleanup_seconds": True},
        {"cleanup_seconds": 60},
        {"flow_version": "future"},
        {"adapter_version": "future"},
        {"processor_version": ""},
        {"budget": None},
    ],
)
def test_explicit_runtime_rejects_invalid_versions_and_cleanup_budgets(updates):
    saved = snapshot()
    value = CoverageSpec(saved.spec, RunBudget(60, 4), "replay-test", 1)
    with pytest.raises((ValueError, ProcessingUnavailable)):
        replace(value, **updates)


def test_runtime_spec_roundtrip_and_no_implicit_upgrade_defaults():
    value = CoverageSpec(snapshot().spec, RunBudget(60, 4), "replay-test", 1)
    raw = json.loads(json.dumps(asdict(value)))
    assert read_coverage_spec(raw) == value
    for changed in ({**raw, "extra": 1}, {k: v for k, v in raw.items() if k != "cleanup_seconds"}):
        with pytest.raises(ProcessingUnavailable):
            read_coverage_spec(changed)


def test_adapter_preserves_exact_focus_context_and_binds_scope_deadline_and_final_list():
    saved = snapshot()
    saved = replace(saved, selection=select_baseline(saved.plan, saved.messages))
    spec = CoverageSpec(saved.spec, RunBudget(60, 4), "replay-test", 1)
    lease = CoverageLease(
        "r", "b", "u", "owner", 1, spec, PlanningLease("p", "b", "u", "owner", 1), 30
    )
    windows = final_windows(saved)
    for window, index in zip(windows, saved.selection.pool_ordinals, strict=True):
        original = saved.plan.pool[index]
        assert (window.focus, window.context, window.reasons) == (
            original.focus,
            original.context,
            original.signals,
        )
    value = candidate_input(lease, "w", windows[0], saved, "payload")
    assert value.sources[0].source_timestamp == 7
    assert (
        value.sources[0].content
        == saved.messages[0].content[value.sources[0].span.start : value.sources[0].span.end]
    )
    changed = replace(saved, deadline_at=saved.deadline_at + timedelta(seconds=1))
    same_instant = replace(
        saved, deadline_at=saved.deadline_at.astimezone(timezone(timedelta(hours=8)))
    )
    assert (
        value.fingerprint
        == candidate_input(lease, "w", windows[0], same_instant, "payload").fingerprint
    )
    assert (
        value.fingerprint != candidate_input(lease, "w", windows[0], changed, "payload").fingerprint
    )
    assert (
        value.fingerprint
        != candidate_input(
            replace(lease, user_id="other"), "w", windows[0], saved, "payload"
        ).fingerprint
    )
    with pytest.raises(ValueError):
        final_windows(replace(saved, selection=replace(saved.selection, pool_ordinals=())))
