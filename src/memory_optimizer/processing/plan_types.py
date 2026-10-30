"""Value contracts for coverage planning; no planner, persistence or readiness claim."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Literal

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.processing.types import SourceSpan

PLANNING_FLOW = "coverage-selection-v2"
PLANNING_SELECTOR = "coding-coverage-v2"
CAPABILITY_SELECTOR = "coding-coverage-v3"
PLAN_SCHEMA = "coverage-plan-v1"
SEGMENTATION_VERSION = "message-focus-v1"
MEASUREMENT_VERSION = "unicode-codepoints-v1"


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid planning integer")


def _fingerprint(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid planning fingerprint")


def _tuple_of(values, item_type):
    if type(values) is not tuple or any(type(value) is not item_type for value in values):
        raise ValueError("invalid immutable planning sequence")


def _span(value):
    if type(value) is not SourceSpan:
        raise ValueError("invalid planning span")
    _integer(value.message_ordinal)
    _integer(value.start)
    _integer(value.end, value.start + 1)


def _signals(values):
    _tuple_of(values, str)
    if len(set(values)) != len(values) or not set(values) <= {"problem", "change", "verification"}:
        raise ValueError("invalid planning signals")


@dataclass(frozen=True)
class PlanLimits:
    """Explicit code-point budgets; defaults await the baseline planner experiment."""

    focus_chars: int
    context_chars: int
    max_windows: int
    max_scan_chars: int
    max_context_chars: int
    max_pool_windows: int

    def __post_init__(self):
        _integer(self.focus_chars, 1)
        for value in (
            self.context_chars,
            self.max_windows,
            self.max_scan_chars,
            self.max_context_chars,
            self.max_pool_windows,
        ):
            _integer(value)
        if self.max_windows > self.max_pool_windows:
            raise ValueError("selection budget exceeds candidate pool budget")


@dataclass(frozen=True)
class PlanningSpec:
    limits: PlanLimits
    flow_version: str = PLANNING_FLOW
    selector_version: str = PLANNING_SELECTOR
    schema_version: str = PLAN_SCHEMA
    segmentation_version: str = SEGMENTATION_VERSION
    measurement_version: str = MEASUREMENT_VERSION

    def __post_init__(self):
        if type(self.limits) is not PlanLimits:
            raise ValueError("invalid planning limits")
        for value in (
            self.flow_version,
            self.selector_version,
            self.schema_version,
            self.segmentation_version,
            self.measurement_version,
        ):
            if type(value) is not str or not 1 <= len(value) <= 100:
                raise ValueError("invalid planning version")


def require_planning_spec(spec: PlanningSpec) -> None:
    """Closed version family shared by planning and its independent validator."""
    if (
        type(spec) is not PlanningSpec
        or (
            spec.flow_version,
            spec.schema_version,
            spec.segmentation_version,
            spec.measurement_version,
        )
        != (
            PLANNING_FLOW,
            PLAN_SCHEMA,
            SEGMENTATION_VERSION,
            MEASUREMENT_VERSION,
        )
        or spec.selector_version not in (PLANNING_SELECTOR, CAPABILITY_SELECTOR)
    ):
        raise ProcessingUnavailable("unsupported selection contract combination")


@dataclass(frozen=True)
class CoverageRange:
    """A source interval, possibly compressing adjacent equally disposed units."""

    span: SourceSpan
    disposition: Literal["candidate", "no_signal", "scan_budget", "pool_budget"]
    signals: tuple[str, ...] = ()

    def __post_init__(self):
        _span(self.span)
        _signals(self.signals)
        if self.disposition not in ("candidate", "no_signal", "scan_budget", "pool_budget"):
            raise ValueError("invalid coverage disposition")
        if self.disposition in ("no_signal", "scan_budget") and self.signals:
            raise ValueError("uninspected/no-signal range cannot claim signals")


@dataclass(frozen=True)
class PlanWindow:
    """Pool order is tuple order; no legacy Window or fabricated signal is needed."""

    focus: SourceSpan
    context: tuple[SourceSpan, ...]
    signals: tuple[str, ...] = ()

    def __post_init__(self):
        _span(self.focus)
        _tuple_of(self.context, SourceSpan)
        if not self.context:
            raise ValueError("planning window requires source context")
        for span in self.context:
            _span(span)
        _signals(self.signals)


@dataclass(frozen=True)
class SelectionPlan:
    spec: PlanningSpec
    source_digest: str
    coverage: tuple[CoverageRange, ...]
    pool: tuple[PlanWindow, ...]

    def __post_init__(self):
        if type(self.spec) is not PlanningSpec:
            raise ValueError("invalid planning specification")
        _fingerprint(self.source_digest)
        _tuple_of(self.coverage, CoverageRange)
        _tuple_of(self.pool, PlanWindow)
        if len(self.pool) > self.spec.limits.max_pool_windows:
            raise ValueError("planning pool budget exceeded")

    @property
    def fingerprint(self) -> str:
        # Content identity is not user identity or permission to reuse across scopes.
        return digest(
            json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )


@dataclass(frozen=True)
class FinalSelection:
    plan_fingerprint: str
    pool_ordinals: tuple[int, ...]
    screening: Literal["disabled"] = "disabled"
    schema_version: str = "final-selection-v1"

    def __post_init__(self):
        _fingerprint(self.plan_fingerprint)
        _tuple_of(self.pool_ordinals, int)
        for ordinal in self.pool_ordinals:
            _integer(ordinal)
        if len(set(self.pool_ordinals)) != len(self.pool_ordinals):
            raise ValueError("duplicate final selection reference")
        if self.screening != "disabled" or self.schema_version != "final-selection-v1":
            raise ValueError("unsupported final selection contract")


@dataclass(frozen=True)
class ExecutionDiagnostic:
    """Execution progress is neither a replacement plan nor a persisted stage marker."""

    phase: Literal["planning", "selection"]
    outcome: Literal["interrupted", "deadline_exhausted", "failed", "completed"]
    inspected: tuple[SourceSpan, ...]
    plan_fingerprint: str | None = None

    def __post_init__(self):
        if self.phase not in ("planning", "selection") or self.outcome not in (
            "interrupted",
            "deadline_exhausted",
            "failed",
            "completed",
        ):
            raise ValueError("invalid planning execution diagnostic")
        _tuple_of(self.inspected, SourceSpan)
        for span in self.inspected:
            _span(span)
        if self.plan_fingerprint is not None:
            _fingerprint(self.plan_fingerprint)
        elif self.phase == "selection" or self.outcome == "completed":
            raise ValueError("completed planning/selection requires a plan reference")
