"""Explicit offline runtime and the narrow final-list-to-candidate adapter."""

from dataclasses import asdict, dataclass
from datetime import UTC
from typing import Protocol

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.extraction.types import EvidenceText, ExtractionInput
from memory_optimizer.processing import plan_codec
from memory_optimizer.processing.checkpoints import PlanningLease, RunBudget
from memory_optimizer.processing.plan_types import PlanningSpec, require_planning_spec
from memory_optimizer.processing.plan_validation import validate_final_selection
from memory_optimizer.processing.types import Window


class WorkTimeExhausted(ProcessingUnavailable):
    """No work time remains; coordination may still record a terminal result."""


@dataclass(frozen=True)
class CoverageSpec:
    planning: PlanningSpec
    budget: RunBudget
    processor_version: str
    cleanup_seconds: int
    flow_version: str = "coverage-offline-v1"
    adapter_version: str = "final-windows-v1"

    def __post_init__(self):
        require_planning_spec(self.planning)
        if type(self.budget) is not RunBudget:
            raise ValueError("invalid coverage budget")
        if type(self.cleanup_seconds) is not int or not (
            1 <= self.cleanup_seconds < self.budget.total_seconds
        ):
            raise ValueError("invalid cleanup reservation")
        if type(self.processor_version) is not str or not 1 <= len(self.processor_version) <= 100:
            raise ValueError("invalid offline processor version")
        if (self.flow_version, self.adapter_version) not in {
            ("coverage-offline-v1", "final-windows-v1"),
            ("coverage-temporal-v1", "final-windows-v1"),
            ("s3-offline-v1", "final-windows-v1"),
            ("s3-openai-v1", "final-windows-v1"),
        }:
            raise ValueError("unsupported coverage runtime")


def read_coverage_spec(value):
    def decode(raw):
        fields = plan_codec.object_fields(raw, CoverageSpec)
        fields["planning"] = plan_codec.spec(fields["planning"])
        fields["budget"] = RunBudget(**plan_codec.object_fields(fields["budget"], RunBudget))
        return CoverageSpec(**fields)

    return plan_codec.decode(value, decode)


@dataclass(frozen=True)
class CoverageLease:
    run_id: str
    batch_id: str
    user_id: str
    owner: str
    fence: int
    spec: CoverageSpec
    planning: PlanningLease
    lease_seconds: int


def final_windows(snapshot):
    validate_final_selection(snapshot.selection, snapshot.plan, snapshot.messages)
    return tuple(
        Window(i, item.focus, item.context, item.signals)
        for i, index in enumerate(snapshot.selection.pool_ordinals)
        for item in (snapshot.plan.pool[index],)
    )


def final_fingerprint(snapshot):
    return digest(canonical(asdict(snapshot.selection)))


def has_selection_gap(snapshot):
    return any(c.disposition in {"scan_budget", "pool_budget"} for c in snapshot.plan.coverage) or (
        len(snapshot.selection.pool_ordinals) < len(snapshot.plan.pool)
    )


def candidate_input(lease, window_id, window, snapshot, payload_hash):
    # Bind the same persisted deadline as well as the final list; no new clock/budget.
    configuration = {
        "spec": asdict(lease.spec),
        "deadline_at": snapshot.deadline_at.astimezone(UTC).isoformat(),
        "selection": final_fingerprint(snapshot),
    }
    return ExtractionInput(
        lease.user_id,
        lease.batch_id,
        lease.run_id,
        window_id,
        window,
        payload_hash,
        lease.spec.processor_version,
        digest(canonical(configuration)),
        tuple(
            EvidenceText(
                span,
                snapshot.messages[span.message_ordinal].content[span.start : span.end],
                snapshot.messages[span.message_ordinal].role,
                snapshot.messages[span.message_ordinal].timestamp,
            )
            for span in window.context
        ),
        schema_version="episode-temporal-v1"
        if lease.spec.flow_version in {"coverage-temporal-v1", "s3-offline-v1", "s3-openai-v1"}
        else "episode-candidates-v1",
    )


class CoverageRepository(Protocol):
    def claim(self, user_id, batch_id, spec, lease_seconds) -> CoverageLease | None: ...
    def load_plan(self, lease): ...
    def plans(self, lease, *, work=True): ...
    def ensure_work(self, lease): ...
    def bind(self, lease): ...
    def candidates(self, lease): ...
    def complete(self, lease, reason=None): ...


class BoundCandidates(Protocol):
    def windows(self): ...
    def reserve(self, request): ...
    def save(self, ticket, candidates, error): ...
