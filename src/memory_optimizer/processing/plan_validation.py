"""Pure full-source validation, independent of plan generation and legacy selection."""

from memory_optimizer.domain import SourceMessage
from memory_optimizer.processing.feature_rules import signal_blocks
from memory_optimizer.processing.plan_types import (
    FinalSelection,
    PlanningSpec,
    SelectionPlan,
    require_planning_spec,
)
from memory_optimizer.processing.planning_rules import (
    context_cost,
    inspection_ends,
    source_digest,
    window_context,
)
from memory_optimizer.processing.types import SourceSpan


def validate_plan_contract(plan: SelectionPlan, spec: PlanningSpec) -> None:
    """Type/version check only; consumers of actual source must use validate_plan."""
    require_planning_spec(spec)
    if type(plan) is not SelectionPlan or plan.spec != spec:
        raise ValueError("selector result does not match planning contract")


def validate_final_contract(selection: FinalSelection, plan: SelectionPlan) -> None:
    """Identity/reference check only, not full source or baseline validation."""
    if type(plan) is not SelectionPlan:
        raise ValueError("final selection requires a planning result")
    validate_plan_contract(plan, plan.spec)
    if type(selection) is not FinalSelection or selection.plan_fingerprint != plan.fingerprint:
        raise ValueError("final selection refers to a different plan")
    if len(selection.pool_ordinals) > plan.spec.limits.max_windows or any(
        index >= len(plan.pool) for index in selection.pool_ordinals
    ):
        raise ValueError("invalid final selection references or budget")


def validate_plan(
    plan: SelectionPlan, messages: tuple[SourceMessage, ...], spec: PlanningSpec
) -> None:
    """Validate full partition, rule accounting, pool order/context and input identity.

    The validator checks intervals and candidates directly; it never builds a
    second plan or accepts a self-reported digest/coverage total as proof.
    """
    validate_plan_contract(plan, spec)
    if plan.source_digest != source_digest(messages):
        raise ValueError("planning source digest mismatch")
    limits = spec.limits
    ranges = iter(plan.coverage)
    current = next(ranges, None)
    pool_index = 0
    scanned = 0
    for ordinal, (message, inspected_end) in enumerate(
        zip(messages, inspection_ends(messages, limits), strict=True)
    ):
        size = len(message.content)
        cursor = 0
        previous = None
        blocks = iter(
            signal_blocks(message.content, inspected_end, limits.focus_chars, spec.selector_version)
        )
        matched = next(blocks, None)
        while cursor < size:
            if (
                current is None
                or current.span.message_ordinal != ordinal
                or current.span.start != cursor
                or current.span.end > size
            ):
                raise ValueError("coverage must partition every source exactly once in order")
            end = current.span.end
            if (
                previous is not None
                and current.disposition != "candidate"
                and (current.disposition, current.signals)
                == (previous.disposition, previous.signals)
            ):
                raise ValueError("coverage must compress adjacent equal dispositions")
            if cursor == inspected_end:
                if end != size or current.disposition != "scan_budget" or current.signals:
                    raise ValueError("uninspected source must retain its exact budget range")
            else:
                if (
                    end > inspected_end
                    or cursor % limits.focus_chars
                    or (end != size and end % limits.focus_chars)
                ):
                    raise ValueError("inspected coverage must align with whole focus units")
                for start in range(cursor, end, limits.focus_chars):
                    focus = SourceSpan(ordinal, start, min(start + limits.focus_chars, size))
                    signals = ()
                    if matched is not None and matched[0] == start // limits.focus_chars:
                        signals = matched[1]
                        matched = next(blocks, None)
                    disposition = "no_signal"
                    if signals:
                        disposition = (
                            "candidate" if pool_index < limits.max_pool_windows else "pool_budget"
                        )
                    if (current.disposition, current.signals) != (disposition, signals):
                        raise ValueError("coverage disposition or rule signals mismatch")
                    if disposition == "candidate":
                        if current.span != focus or pool_index >= len(plan.pool):
                            raise ValueError(
                                "candidate coverage and pool must correspond one to one"
                            )
                        window = plan.pool[pool_index]
                        if (
                            window.focus != focus
                            or window.signals != signals
                            or window.context
                            != window_context(messages, focus, limits.context_chars)
                        ):
                            raise ValueError("invalid pool order, signal or bounded source context")
                        pool_index += 1
                scanned += end - cursor
            cursor = end
            previous = current
            current = next(ranges, None)
    if current is not None or pool_index != len(plan.pool) or scanned > limits.max_scan_chars:
        raise ValueError("extra coverage/pool entries or scan budget exceeded")


def validate_final_selection(
    selection: FinalSelection, plan: SelectionPlan, messages: tuple[SourceMessage, ...]
) -> None:
    """Require the maximal affordable ordered prefix, with screening disabled."""
    validate_final_contract(selection, plan)
    validate_plan(plan, messages, plan.spec)
    count = len(selection.pool_ordinals)
    if selection.pool_ordinals != tuple(range(count)):
        raise ValueError("baseline selection must keep pool prefix order")
    used = sum(context_cost(window.context) for window in plan.pool[:count])
    limits = plan.spec.limits
    if used > limits.max_context_chars:
        raise ValueError("final context budget exceeded")
    if count < min(len(plan.pool), limits.max_windows):
        if used + context_cost(plan.pool[count].context) <= limits.max_context_chars:
            raise ValueError("baseline omitted an affordable next window")
