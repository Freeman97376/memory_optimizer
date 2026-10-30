"""Strict JSON-value decoding for saved plans; missing fields never gain defaults."""

from dataclasses import fields

from memory_optimizer.domain import ProcessingUnavailable
from memory_optimizer.processing.plan_types import (
    CoverageRange,
    ExecutionDiagnostic,
    FinalSelection,
    PlanLimits,
    PlanningSpec,
    PlanWindow,
    SelectionPlan,
    require_planning_spec,
)
from memory_optimizer.processing.types import SourceSpan


def object_fields(value, cls):
    if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
        raise ValueError("invalid persisted planning fields")
    return dict(value)


def sequence(value, decoder):
    if type(value) is not list:
        raise ValueError("invalid persisted planning array")
    return tuple(decoder(item) for item in value)


def span(value):
    return SourceSpan(**object_fields(value, SourceSpan))


def spec(value):
    data = object_fields(value, PlanningSpec)
    data["limits"] = PlanLimits(**object_fields(data["limits"], PlanLimits))
    result = PlanningSpec(**data)
    require_planning_spec(result)
    return result


def coverage(value):
    data = object_fields(value, CoverageRange)
    data["span"] = span(data["span"])
    data["signals"] = sequence(data["signals"], lambda x: x)
    return CoverageRange(**data)


def window(value):
    data = object_fields(value, PlanWindow)
    data["focus"] = span(data["focus"])
    data["context"] = sequence(data["context"], span)
    data["signals"] = sequence(data["signals"], lambda x: x)
    return PlanWindow(**data)


def plan(value):
    data = object_fields(value, SelectionPlan)
    data["spec"] = spec(data["spec"])
    data["coverage"] = sequence(data["coverage"], coverage)
    data["pool"] = sequence(data["pool"], window)
    return SelectionPlan(**data)


def selection(value):
    data = object_fields(value, FinalSelection)
    data["pool_ordinals"] = sequence(data["pool_ordinals"], lambda x: x)
    return FinalSelection(**data)


def diagnostic(value):
    data = object_fields(value, ExecutionDiagnostic)
    data["inspected"] = sequence(data["inspected"], span)
    return ExecutionDiagnostic(**data)


def decode(value, decoder):
    try:
        return decoder(value)
    except (KeyError, TypeError, ValueError, RecursionError):
        raise ProcessingUnavailable("incompatible persisted planning data") from None
