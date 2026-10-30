"""Small persisted planning contracts; no storage driver or execution clock."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from memory_optimizer.domain import SourceMessage, digest
from memory_optimizer.processing.plan_types import (
    ExecutionDiagnostic,
    FinalSelection,
    PlanningSpec,
    SelectionPlan,
)
from memory_optimizer.processing.planning_rules import inspection_ends
from memory_optimizer.processing.types import SourceSpan

Phase = Literal["planning", "selection"]


@dataclass(frozen=True)
class RunBudget:
    total_seconds: int
    max_attempts: int

    def __post_init__(self):
        if type(self.total_seconds) is not int or not 1 <= self.total_seconds <= 86400:
            raise ValueError("invalid planning total duration")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 32:
            raise ValueError("invalid planning attempt budget")


@dataclass(frozen=True)
class PlanningLease:
    run_id: str
    batch_id: str
    user_id: str
    owner: str
    fence: int


@dataclass(frozen=True)
class PlanningTicket:
    number: int
    reservation_id: str
    phase: Phase
    input_hash: str
    fence: int


@dataclass(frozen=True)
class PlanningAttempt:
    ticket: PlanningTicket
    diagnostic: ExecutionDiagnostic | None
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class PlanningSnapshot:
    run_id: str
    batch_id: str
    user_id: str
    messages: tuple[SourceMessage, ...]
    source_hash: str
    spec: PlanningSpec
    budget: RunBudget
    created_at: datetime
    deadline_at: datetime
    stage: Literal["raw_ready", "plan_ready", "selection_ready"]
    plan: SelectionPlan | None
    selection: FinalSelection | None
    attempts: tuple[PlanningAttempt, ...]
    closed_reason: Literal["deadline_exhausted", "attempts_exhausted"] | None


def attempt_input(snapshot: PlanningSnapshot, phase: Phase) -> str:
    if phase not in ("planning", "selection") or (phase == "selection" and snapshot.plan is None):
        raise ValueError("invalid planning attempt phase")
    return digest(
        json.dumps(
            {
                "version": "planning-attempt-v1",
                "run_id": snapshot.run_id,
                "batch_id": snapshot.batch_id,
                "user_id": snapshot.user_id,
                "source_hash": snapshot.source_hash,
                "spec": asdict(snapshot.spec),
                "budget": asdict(snapshot.budget),
                "deadline_at": snapshot.deadline_at.astimezone(UTC).isoformat(),
                "phase": phase,
                "plan_fingerprint": snapshot.plan.fingerprint if phase == "selection" else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def merge_spans(spans: tuple[SourceSpan, ...]) -> tuple[SourceSpan, ...]:
    result = []
    for span in sorted(spans, key=lambda s: (s.message_ordinal, s.start, s.end)):
        if (
            result
            and result[-1].message_ordinal == span.message_ordinal
            and result[-1].end >= span.start
        ):
            previous = result.pop()
            result.append(
                SourceSpan(span.message_ordinal, previous.start, max(previous.end, span.end))
            )
        else:
            result.append(span)
    return tuple(result)


def diagnostic_ranges(snapshot: PlanningSnapshot, phase: Phase, selection=None):
    """Accounted algorithm ranges, never a claim that a model read the text."""
    if phase == "planning":
        return tuple(
            SourceSpan(i, 0, end)
            for i, end in enumerate(inspection_ends(snapshot.messages, snapshot.spec.limits))
            if end
        )
    if phase != "selection" or snapshot.plan is None:
        raise ValueError("selection diagnostic requires a plan")
    indices = selection.pool_ordinals if selection is not None else range(len(snapshot.plan.pool))
    return merge_spans(tuple(s for i in indices for s in snapshot.plan.pool[i].context))


def validate_diagnostic(diagnostic, snapshot, *, plan=None, selection=None):
    if type(diagnostic) is not ExecutionDiagnostic:
        raise ValueError("invalid execution diagnostic type")
    completed = diagnostic.outcome == "completed"
    target = plan if diagnostic.phase == "planning" else snapshot.plan
    expected_hash = target.fingerprint if target is not None else None
    if diagnostic.plan_fingerprint != expected_hash:
        raise ValueError("diagnostic plan reference mismatch")
    if completed and (target is None or (diagnostic.phase == "selection" and selection is None)):
        raise ValueError("completed diagnostic requires its checkpoint")
    allowed = diagnostic_ranges(snapshot, diagnostic.phase, selection if completed else None)
    actual = diagnostic.inspected
    if actual != merge_spans(actual):
        raise ValueError("diagnostic ranges must be ordered, disjoint and compressed")
    if completed and actual != allowed:
        raise ValueError("completed diagnostic must account for the checkpoint ranges")
    if any(
        not any(
            a.message_ordinal == s.message_ordinal and a.start <= s.start < s.end <= a.end
            for a in allowed
        )
        for s in actual
    ):
        raise ValueError("diagnostic claims unaccounted source")


class PlanningRepository(Protocol):
    def initialize(
        self, user_id: str, batch_id: str, spec: PlanningSpec, budget: RunBudget
    ) -> PlanningSnapshot: ...
    def load(self, user_id: str, batch_id: str) -> PlanningSnapshot: ...
    def claim(self, user_id: str, batch_id: str, lease_seconds: int) -> PlanningLease: ...
    def renew(self, lease: PlanningLease, lease_seconds: int) -> None: ...
    def reserve(
        self, lease: PlanningLease, phase: Phase, reservation_id: str
    ) -> PlanningAttempt | None: ...
    def save_plan(
        self,
        lease: PlanningLease,
        ticket: PlanningTicket,
        plan: SelectionPlan,
        diagnostic: ExecutionDiagnostic,
    ) -> None: ...
    def save_selection(
        self,
        lease: PlanningLease,
        ticket: PlanningTicket,
        selection: FinalSelection,
        diagnostic: ExecutionDiagnostic,
    ) -> None: ...
    def save_diagnostic(
        self, lease: PlanningLease, ticket: PlanningTicket, diagnostic: ExecutionDiagnostic
    ) -> None: ...
