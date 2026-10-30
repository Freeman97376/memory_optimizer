"""Finite synthetic coverage/budget oracles; no semantic recall or runtime claims."""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from itertools import product

import pytest

from memory_optimizer.domain import ProcessingUnavailable, SourceMessage
from memory_optimizer.processing import planning_rules
from memory_optimizer.processing.plan_types import (
    CoverageRange,
    FinalSelection,
    PlanLimits,
    PlanningSpec,
    PlanWindow,
)
from memory_optimizer.processing.plan_validation import validate_final_selection, validate_plan
from memory_optimizer.processing.planning import build_plan, select_baseline
from memory_optimizer.processing.planning_rules import source_digest
from memory_optimizer.processing.selection_contracts import SelectionBinding
from memory_optimizer.processing.types import SourceSpan


def spec(**changes):
    values = dict(
        focus_chars=10,
        context_chars=0,
        max_windows=3,
        max_scan_chars=1000,
        max_context_chars=1000,
        max_pool_windows=3,
    )
    values.update(changes)
    return PlanningSpec(PlanLimits(**values))


def messages(*texts):
    return tuple(SourceMessage("user", text) for text in texts)


def assert_partition(plan, sources):
    # Independent small-fixture oracle: every original code point appears once.
    observed = Counter(
        (item.span.message_ordinal, index)
        for item in plan.coverage
        for index in range(item.span.start, item.span.end)
    )
    expected = Counter(
        (ordinal, index)
        for ordinal, message in enumerate(sources)
        for index in range(len(message.content))
    )
    assert observed == expected
    assert sum(item.span.end - item.span.start for item in plan.coverage) == sum(
        len(message.content) for message in sources
    )


def test_unicode_crlf_empty_message_and_exact_source_order_accounting():
    sources = messages("error", "", "已修复🌱\r\n", "说明", "测试失败")
    config = spec(focus_chars=5, context_chars=2, max_pool_windows=2, max_windows=1)
    result = build_plan(sources, config)
    assert result.coverage == (
        CoverageRange(SourceSpan(0, 0, 5), "candidate", ("problem",)),
        CoverageRange(SourceSpan(2, 0, 5), "candidate", ("change",)),
        CoverageRange(SourceSpan(2, 5, 6), "no_signal"),
        CoverageRange(SourceSpan(3, 0, 2), "no_signal"),
        CoverageRange(SourceSpan(4, 0, 4), "pool_budget", ("problem", "verification")),
    )
    assert result.pool[1].context == (SourceSpan(2, 0, 6),)
    assert select_baseline(result, sources).pool_ordinals == (0,)
    assert_partition(result, sources)
    validate_plan(result, sources, config)


def test_scan_is_whole_unit_prefix_never_reallocated_to_later_small_message():
    sources = messages("error! long text", "错误")
    config = spec(focus_chars=4, max_scan_chars=7)
    result = build_plan(sources, config)
    assert result.coverage == (
        CoverageRange(SourceSpan(0, 0, 4), "no_signal"),
        CoverageRange(SourceSpan(0, 4, 16), "scan_budget"),
        CoverageRange(SourceSpan(1, 0, 2), "scan_budget"),
    )
    assert result.pool == ()
    assert_partition(result, sources)
    validate_plan(result, sources, config)


@pytest.mark.parametrize("text", ["errorless", "error!more", "x修改更多"])
def test_truncated_edge_does_not_manufacture_a_complete_rule_match(text):
    focus = 5 if text.startswith("error") else 3
    sources = messages(text)
    config = spec(focus_chars=focus, max_scan_chars=focus)
    result = build_plan(sources, config)
    assert result.coverage[0].disposition == "no_signal"
    assert result.coverage[1].disposition == "scan_budget"
    assert not result.pool
    validate_plan(result, sources, config)


def test_signal_across_inspected_blocks_belongs_only_to_start_block():
    sources = messages("xx error!x")
    config = spec(focus_chars=4, context_chars=4)
    result = build_plan(sources, config)
    assert result.pool == (PlanWindow(SourceSpan(0, 0, 4), (SourceSpan(0, 0, 8),), ("problem",)),)
    assert result.coverage[1] == CoverageRange(SourceSpan(0, 4, 10), "no_signal")
    validate_plan(result, sources, config)


def test_late_counterevidence_is_explicit_pool_omission_not_semantic_empty():
    sources = messages("test passed", "test passed", "test passed", "but test failed")
    config = spec(focus_chars=100, max_pool_windows=1, max_windows=1)
    result = build_plan(sources, config)
    assert result.pool[0].focus == SourceSpan(0, 0, 11)
    assert [item.disposition for item in result.coverage] == ["candidate"] + ["pool_budget"] * 3
    assert result.coverage[-1].signals == ("problem", "verification")
    assert_partition(result, sources)
    validate_plan(result, sources, config)


def test_repeated_occurrences_are_distinct_and_equal_non_candidate_ranges_compress():
    sources = messages("错误错误错误", "错误", "xyz xyz xyz")
    config = spec(focus_chars=2, max_pool_windows=1, max_windows=1)
    result = build_plan(sources, config)
    assert result.coverage == (
        CoverageRange(SourceSpan(0, 0, 2), "candidate", ("problem",)),
        CoverageRange(SourceSpan(0, 2, 6), "pool_budget", ("problem",)),
        CoverageRange(SourceSpan(1, 0, 2), "pool_budget", ("problem",)),
        CoverageRange(SourceSpan(2, 0, 11), "no_signal"),
    )
    validate_plan(result, sources, config)


@pytest.mark.parametrize("sources", [(), messages(""), messages("", "")])
def test_empty_sources_have_empty_coverage_and_selection_without_fake_percentage(sources):
    config = spec(max_windows=0, max_pool_windows=0, max_scan_chars=0, max_context_chars=0)
    result = build_plan(sources, config)
    assert result.coverage == result.pool == ()
    final = select_baseline(result, sources)
    assert final.pool_ordinals == ()
    assert final.screening == "disabled"
    validate_final_selection(final, result, sources)


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
def test_exact_scan_cutoffs_and_final_short_unit(budget):
    sources = messages("abcdefg", "xy")
    config = spec(focus_chars=4, max_scan_chars=budget)
    result = build_plan(sources, config)
    # Independently grant whole units in order; once one fails, all later units stop.
    granted = []
    used = 0
    for ordinal, start, end in [(0, 0, 4), (0, 4, 7), (1, 0, 2)]:
        if used + end - start > budget:
            break
        granted.extend((ordinal, index) for index in range(start, end))
        used += end - start
    inspected = [
        (item.span.message_ordinal, index)
        for item in result.coverage
        if item.disposition != "scan_budget"
        for index in range(item.span.start, item.span.end)
    ]
    assert inspected == granted
    assert_partition(result, sources)
    validate_plan(result, sources, config)


def test_zero_scan_large_source_stays_compact_and_does_not_scan_uninspected_text(monkeypatch):
    sources = messages("error " * 100_000, "测试" * 100_000)
    calls = []
    original = planning_rules._SIGNALS

    class ScanSpy:
        def finditer(self, text):
            calls.append(len(text))
            return original.finditer(text)

    monkeypatch.setattr(planning_rules, "_SIGNALS", ScanSpy())
    result = build_plan(sources, spec(focus_chars=1, max_scan_chars=0))
    validate_plan(result, sources, result.spec)
    assert len(result.coverage) == 2
    assert {item.disposition for item in result.coverage} == {"scan_budget"}
    assert calls == [0, 0, 0, 0]
    assert result.pool == ()


def test_context_budget_charges_overlap_per_window_and_never_skips_a_blocker():
    sources = messages("error", "fix", "test")
    config = spec(context_chars=2, max_context_chars=13)
    result = build_plan(sources, config)
    assert [sum(s.end - s.start for s in w.context) for w in result.pool] == [7, 7, 6]
    final = select_baseline(result, sources)
    assert final.pool_ordinals == (0,)  # Second does not fit; cheaper third cannot jump it.
    validate_final_selection(final, result, sources)
    with pytest.raises(ValueError, match="prefix"):
        validate_final_selection(replace(final, pool_ordinals=(0, 2)), result, sources)


@pytest.mark.parametrize(
    "budget,expected", [(0, ()), (4, ()), (5, (0,)), (7, (0,)), (8, (0, 1)), (12, (0, 1, 2))]
)
def test_final_context_budget_exact_boundary(budget, expected):
    sources = messages("error", "fix", "test")
    result = build_plan(sources, spec(max_context_chars=budget))
    final = select_baseline(result, sources)
    assert final.pool_ordinals == expected
    validate_final_selection(final, result, sources)


@pytest.mark.parametrize("windows", [0, 1, 2, 3])
def test_final_window_cap_does_not_delete_planned_candidates(windows):
    sources = messages("error", "fix", "test")
    result = build_plan(sources, spec(max_windows=windows))
    assert len(result.pool) == 3
    final = select_baseline(result, sources)
    assert final.pool_ordinals == tuple(range(windows))
    validate_final_selection(final, result, sources)


@pytest.mark.parametrize("field", ["role", "content", "timestamp", "order", "empty_message"])
def test_digest_binds_all_source_fields_including_empty_positions(field):
    sources = (SourceMessage("user", "error", 1), SourceMessage("assistant", "fix"))
    changed = {
        "role": (replace(sources[0], role="assistant"), sources[1]),
        "content": (replace(sources[0], content="ERROR"), sources[1]),
        "timestamp": (replace(sources[0], timestamp=2), sources[1]),
        "order": tuple(reversed(sources)),
        "empty_message": (*sources, SourceMessage("user", "")),
    }[field]
    result = build_plan(sources, spec())
    assert source_digest(sources) != source_digest(changed)
    with pytest.raises(ValueError, match="digest"):
        validate_plan(result, changed, result.spec)


def test_source_digest_has_a_fixed_encoding_and_never_normalizes_unicode():
    sources = (SourceMessage("user", "🌱\r\n", None),)
    expected = 'planning-source-v1\n[{"content":"🌱\\r\\n","role":"user","timestamp":null}]'
    assert source_digest(sources) == hashlib.sha256(expected.encode()).hexdigest()
    assert source_digest(messages("é")) != source_digest(messages("e\u0301"))


@pytest.mark.parametrize(
    "sources",
    [
        [],
        ("error",),
        (SourceMessage("tool", "error"),),
        (SourceMessage("user", 1),),
        (SourceMessage("user", "error", True),),
    ],
)
def test_invalid_source_values_fail_at_pure_boundary(sources):
    with pytest.raises(ValueError):
        build_plan(sources, spec())


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "missing",
        "duplicate",
        "reordered",
        "gap",
        "out_of_bounds",
        "wrong_ordinal",
        "uninspected_lie",
        "no_signal_lie",
        "pool_budget_lie",
        "signal_lie",
        "missing_pool",
        "pool_order",
        "pool_focus",
        "context_bounds",
        "context_omitted",
        "context_reordered",
        "extra_pool",
        "split_no_signal",
        "wrong_grid",
        "premature_scan_stop",
    ],
)
def test_full_validator_rejects_forged_accounting_pool_context_and_budgets(mutation):
    sources = messages("error!", "fix", "test", "abcdef")
    config = spec(context_chars=1, focus_chars=4, max_pool_windows=5)
    result = build_plan(sources, config)
    coverage = list(result.coverage)
    pool = list(result.pool)
    if mutation == "digest":
        result = replace(result, source_digest="0" * 64)
    elif mutation == "missing":
        coverage.pop()
    elif mutation == "duplicate":
        coverage.append(coverage[-1])
    elif mutation == "reordered":
        coverage.reverse()
    elif mutation == "gap":
        coverage[0] = replace(coverage[0], span=SourceSpan(0, 1, 4))
    elif mutation == "out_of_bounds":
        coverage[-1] = replace(coverage[-1], span=SourceSpan(3, 0, 7))
    elif mutation == "wrong_ordinal":
        coverage[-1] = replace(coverage[-1], span=SourceSpan(8, 0, 6))
    elif mutation in {"uninspected_lie", "no_signal_lie", "pool_budget_lie"}:
        disposition = {
            "uninspected_lie": "scan_budget",
            "no_signal_lie": "no_signal",
            "pool_budget_lie": "pool_budget",
        }[mutation]
        coverage[0] = CoverageRange(coverage[0].span, disposition)
    elif mutation == "signal_lie":
        coverage[0] = replace(coverage[0], signals=("change",))
    elif mutation == "missing_pool":
        pool.pop()
    elif mutation == "pool_order":
        pool.reverse()
    elif mutation == "pool_focus":
        pool[0] = replace(pool[0], focus=SourceSpan(0, 0, 3))
    elif mutation == "context_bounds":
        pool[0] = replace(pool[0], context=(SourceSpan(0, 0, 7),))
    elif mutation == "context_omitted":
        pool[1] = replace(pool[1], context=(pool[1].focus,))
    elif mutation == "context_reordered":
        pool[1] = replace(pool[1], context=tuple(reversed(pool[1].context)))
    elif mutation == "extra_pool":
        pool.append(pool[-1])
    elif mutation == "split_no_signal":
        coverage[-1:] = [
            CoverageRange(SourceSpan(3, 0, 4), "no_signal"),
            CoverageRange(SourceSpan(3, 4, 6), "no_signal"),
        ]
    elif mutation == "wrong_grid":
        coverage[0:2] = [
            replace(coverage[0], span=SourceSpan(0, 0, 3)),
            replace(coverage[1], span=SourceSpan(0, 3, 6)),
        ]
    else:
        coverage[-1] = CoverageRange(coverage[-1].span, "scan_budget")
    forged = replace(result, coverage=tuple(coverage), pool=tuple(pool))
    with pytest.raises(ValueError):
        validate_plan(forged, sources, config)
    with pytest.raises(ValueError):
        SelectionBinding(config, lambda *_: forged).select(sources)


@pytest.mark.parametrize("chosen,budget", [((0, 1), 5), ((), 100), ((1, 0), 100)])
def test_final_validator_rejects_overbudget_nonmaximal_or_reordered_selection(chosen, budget):
    sources = messages("error", "fix")
    result = build_plan(sources, spec(max_context_chars=budget))
    final = FinalSelection(result.fingerprint, chosen)
    with pytest.raises(ValueError):
        validate_final_selection(final, result, sources)


def test_new_binding_uses_full_validation_and_version_failure_precedes_source_work():
    sources = messages("test")
    config = spec()
    assert SelectionBinding(config, build_plan).select(sources) == build_plan(sources, config)
    with pytest.raises(ProcessingUnavailable):
        build_plan([], replace(config, selector_version="unknown"))


# Bounded cartesian fixtures, not random fuzzing: 36 configurations, <= 25 source
# code points each. Oracles are partition/budget/order/immutability invariants.
@pytest.mark.parametrize(
    "focus,scan,pool,radius", list(product([1, 4, 20], [0, 7, 100], [0, 3], [0, 2]))
)
def test_finite_budget_grid_preserves_partition_limits_and_determinism(focus, scan, pool, radius):
    sources = messages("错误🌱\r\n", "plain", "", "fix; test", "错误")
    config = spec(
        focus_chars=focus,
        max_scan_chars=scan,
        max_pool_windows=pool,
        max_windows=min(2, pool),
        context_chars=radius,
        max_context_chars=12,
    )
    result = build_plan(sources, config)
    frozen = asdict(result)
    assert_partition(result, sources)
    assert len(result.pool) <= pool
    assert (
        sum(
            item.span.end - item.span.start
            for item in result.coverage
            if item.disposition != "scan_budget"
        )
        <= scan
    )
    validate_plan(result, sources, config)
    final = select_baseline(result, sources)
    assert len(final.pool_ordinals) <= config.limits.max_windows
    assert (
        sum(s.end - s.start for index in final.pool_ordinals for s in result.pool[index].context)
        <= 12
    )
    validate_final_selection(final, result, sources)
    assert asdict(result) == frozen
    assert build_plan(sources, config) == result
    assert select_baseline(result, sources) == final


def test_process_hash_seed_and_clock_do_not_change_pure_plan_or_final_selection():
    code = """
import builtins, json, sys, time
from dataclasses import asdict
from memory_optimizer.domain import SourceMessage
from memory_optimizer.processing.plan_types import PlanLimits, PlanningSpec
from memory_optimizer.processing.planning import build_plan, select_baseline
from memory_optimizer.processing.plan_validation import validate_final_selection
def forbidden(*args, **kwargs):
    raise AssertionError("forbidden external dependency or clock")
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'fastapi','psycopg','httpx','openai','anthropic','git'}:
        return forbidden()
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
time.time = time.monotonic = time.perf_counter = forbidden
def audit(event, args):
    if event in {'open','socket.connect','subprocess.Popen','os.system'}:
        forbidden()
sys.addaudithook(audit)
sources = (SourceMessage('user','error; fix; test',3),)
plan = build_plan(sources, PlanningSpec(PlanLimits(6,2,2,100,30,4)))
final = select_baseline(plan, sources)
validate_final_selection(final, plan, sources)
print(json.dumps([asdict(plan),asdict(final)],sort_keys=True))
"""
    outputs = []
    for seed in ("0", "17", "999"):
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        outputs.append(json.loads(completed.stdout))
    assert outputs[0] == outputs[1] == outputs[2]
