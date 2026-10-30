"""Deterministic coverage plan and disabled-screening baseline; no durable runtime."""

from dataclasses import replace

from memory_optimizer.domain import SourceMessage
from memory_optimizer.processing.feature_rules import signal_blocks
from memory_optimizer.processing.plan_types import (
    CoverageRange,
    FinalSelection,
    PlanningSpec,
    PlanWindow,
    SelectionPlan,
    require_planning_spec,
)
from memory_optimizer.processing.plan_validation import validate_plan
from memory_optimizer.processing.planning_rules import (
    context_cost,
    inspection_ends,
    source_digest,
    window_context,
)
from memory_optimizer.processing.types import SourceSpan


def _append_range(coverage: list[CoverageRange], item: CoverageRange) -> None:
    if coverage and item.disposition != "candidate":
        previous = coverage[-1]
        if (
            previous.span.message_ordinal == item.span.message_ordinal
            and previous.span.end == item.span.start
            and (previous.disposition, previous.signals) == (item.disposition, item.signals)
        ):
            coverage[-1] = replace(previous, span=replace(previous.span, end=item.span.end))
            return
    coverage.append(item)


def build_plan(messages: tuple[SourceMessage, ...], spec: PlanningSpec) -> SelectionPlan:
    """Account for all source, classify a whole-unit prefix and retain a bounded pool."""
    require_planning_spec(spec)
    digest = source_digest(messages)
    limits = spec.limits
    coverage = []
    pool = []
    for ordinal, (message, inspected_end) in enumerate(
        zip(messages, inspection_ends(messages, limits), strict=True)
    ):
        size = len(message.content)
        cursor = 0
        for block, signals in signal_blocks(
            message.content, inspected_end, limits.focus_chars, spec.selector_version
        ):
            start = block * limits.focus_chars
            if cursor < start:
                _append_range(
                    coverage, CoverageRange(SourceSpan(ordinal, cursor, start), "no_signal")
                )
            focus = SourceSpan(ordinal, start, min(start + limits.focus_chars, size))
            disposition = "pool_budget"
            if len(pool) < limits.max_pool_windows:
                disposition = "candidate"
                pool.append(
                    PlanWindow(
                        focus, window_context(messages, focus, limits.context_chars), signals
                    )
                )
            _append_range(coverage, CoverageRange(focus, disposition, signals))
            cursor = focus.end
        if cursor < inspected_end:
            _append_range(
                coverage, CoverageRange(SourceSpan(ordinal, cursor, inspected_end), "no_signal")
            )
        if inspected_end < size:
            _append_range(
                coverage, CoverageRange(SourceSpan(ordinal, inspected_end, size), "scan_budget")
            )
    return SelectionPlan(spec, digest, tuple(coverage), tuple(pool))


def select_baseline(plan: SelectionPlan, messages: tuple[SourceMessage, ...]) -> FinalSelection:
    """Take the affordable source-order prefix; never trim context or skip a blocker."""
    if type(plan) is not SelectionPlan:
        raise ValueError("baseline requires a planning result")
    validate_plan(plan, messages, plan.spec)
    limits = plan.spec.limits
    used = 0
    selected = []
    for ordinal, window in enumerate(plan.pool):
        cost = context_cost(window.context)
        if len(selected) == limits.max_windows or used + cost > limits.max_context_chars:
            break
        selected.append(ordinal)
        used += cost
    return FinalSelection(plan.fingerprint, tuple(selected))
