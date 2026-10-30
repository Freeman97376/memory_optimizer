"""Versioned capability selection; original rules and source budgets remain intact."""

import json
from dataclasses import asdict, replace

import pytest

from memory_optimizer.domain import SourceMessage
from memory_optimizer.processing.plan_codec import decode
from memory_optimizer.processing.plan_codec import plan as decode_plan
from memory_optimizer.processing.plan_types import PlanLimits, PlanningSpec
from memory_optimizer.processing.plan_validation import validate_plan
from memory_optimizer.processing.planning import build_plan, select_baseline


def profile(**limits):
    values = dict(
        focus_chars=1000,
        context_chars=0,
        max_windows=3,
        max_scan_chars=10000,
        max_context_chars=10000,
        max_pool_windows=4,
    )
    values.update(limits)
    return PlanningSpec(PlanLimits(**values), selector_version="coding-coverage-v3")


@pytest.mark.parametrize(
    "content",
    [
        "Added an optional CSV export endpoint.",
        "The parser now supports quoted identifiers.",
        "Introduced a configurable retry interval.",
        "Removed the deprecated legacy endpoint.",
        "Implement streaming for large exports.",
        "新增批量导出功能。",
        "支持可选字符检测。",
        "移除旧版接口。",
    ],
)
def test_capability_changes_are_selected_only_by_new_version(content):
    sources = (SourceMessage("user", content),)
    spec = profile()
    old_spec = replace(spec, selector_version="coding-coverage-v2")
    old = build_plan(sources, old_spec)
    new = build_plan(sources, spec)
    assert old.pool == ()
    assert len(new.pool) == 1 and new.pool[0].signals == ("change",)
    validate_plan(new, sources, spec)
    assert select_baseline(new, sources).pool_ordinals == (0,)
    assert old.fingerprint != new.fingerprint


@pytest.mark.parametrize(
    "content", ["Address book listing", "Unsupported-format metadata", "A quiet day."]
)
def test_feature_matching_does_not_use_partial_english_words(content):
    assert build_plan((SourceMessage("user", content),), profile()).pool == ()


def test_new_rule_keeps_old_problem_change_and_verification_signals():
    sources = (SourceMessage("assistant", "bug fixed; tests passed"),)
    new = build_plan(sources, profile())
    old = build_plan(sources, replace(profile(), selector_version="coding-coverage-v2"))
    assert new.coverage == old.coverage and new.pool == old.pool


def test_partial_word_at_scan_boundary_does_not_create_a_feature():
    source = (SourceMessage("user", "supports"),)
    result = build_plan(source, profile(focus_chars=7, max_scan_chars=7))
    assert not result.pool
    assert [(r.span.start, r.span.end, r.disposition) for r in result.coverage] == [
        (0, 7, "no_signal"),
        (7, 8, "scan_budget"),
    ]


def test_capability_pool_and_final_selection_keep_explicit_budget_omissions():
    sources = tuple(
        SourceMessage("user", text)
        for text in ("added CSV output", "introduced XML output", "supports JSON output")
    )
    spec = profile(max_pool_windows=2, max_windows=1)
    result = build_plan(sources, spec)
    assert [r.disposition for r in result.coverage] == ["candidate", "candidate", "pool_budget"]
    assert select_baseline(result, sources).pool_ordinals == (0,)
    restored = decode(json.loads(json.dumps(asdict(result))), decode_plan)
    assert restored == result
    validate_plan(restored, sources, spec)
    with pytest.raises(ValueError):
        validate_plan(restored, sources, replace(spec, selector_version="coding-coverage-v2"))


def test_cross_block_feature_is_counted_once_at_its_original_start():
    sources = (SourceMessage("user", "....supports JSON"),)
    spec = profile(focus_chars=5)
    result = build_plan(sources, spec)
    validate_plan(result, sources, spec)
    assert [(w.focus.start, w.focus.end) for w in result.pool] == [(0, 5)]
