"""Synchronous, explicit offline orchestration; no SQL or provider SDK."""

from dataclasses import dataclass
from typing import ClassVar
from uuid import uuid4

from memory_optimizer.domain import ProcessingUnavailable, payload_digest
from memory_optimizer.extraction.types import (
    MAX_ATTEMPTS,
    CandidateValidationError,
    OfflineExtractor,
)
from memory_optimizer.extraction.validation import validate_reply
from memory_optimizer.processing.checkpoints import diagnostic_ranges
from memory_optimizer.processing.coverage_types import (
    CoverageRepository,
    CoverageSpec,
    WorkTimeExhausted,
    candidate_input,
)
from memory_optimizer.processing.plan_types import ExecutionDiagnostic
from memory_optimizer.processing.planning import build_plan, select_baseline


@dataclass(frozen=True)
class CoverageFlow:
    FLOW_VERSION: ClassVar[str] = "coverage-offline-v1"
    repository: CoverageRepository
    spec: CoverageSpec
    extractor: OfflineExtractor
    lease_seconds: int = 30

    def __post_init__(self):
        if (
            type(self.spec) is not CoverageSpec
            or self.spec.flow_version != self.FLOW_VERSION
            or self.extractor.version != self.spec.processor_version
        ):
            raise ProcessingUnavailable("coverage offline processor mismatch")
        if type(self.lease_seconds) is not int or not 1 <= self.lease_seconds <= 300:
            raise ValueError("invalid coverage lease")

    def finish(self, command, receipt):
        lease = self.repository.claim(
            command.user_id, receipt.batch_id, self.spec, self.lease_seconds
        )
        if lease is None:  # All completed versions replay; never retrofit a legacy receipt.
            return
        try:
            if not self._plan(lease):
                self.repository.complete(lease, "planning_attempts_exhausted")
                return
            self.repository.bind(lease)
            self._extract(lease, command)
            self.repository.complete(lease)
        except WorkTimeExhausted:
            self.repository.complete(lease, "time_budget_exhausted")

    def _plan(self, lease):
        plans = self.repository.plans(lease)
        while True:
            saved = self.repository.load_plan(lease)
            if saved.selection is not None:
                return True
            self.repository.ensure_work(lease)
            phase = "planning" if saved.plan is None else "selection"
            attempt = plans.reserve(lease.planning, phase, str(uuid4()))
            if attempt is None:
                return False
            # A failed deterministic calculation is persisted, but storage/lease errors
            # propagate. Never convert an ambiguous commit failure into success.
            try:
                value = (
                    build_plan(saved.messages, saved.spec)
                    if phase == "planning"
                    else select_baseline(saved.plan, saved.messages)
                )
            except (ValueError, ProcessingUnavailable):
                plans.save_diagnostic(
                    lease.planning,
                    attempt.ticket,
                    ExecutionDiagnostic(
                        phase,
                        "failed",
                        (),
                        saved.plan.fingerprint if phase == "selection" else None,
                    ),
                )
                continue
            if phase == "planning":
                d = ExecutionDiagnostic(
                    phase, "completed", diagnostic_ranges(saved, phase), value.fingerprint
                )
                plans.save_plan(lease.planning, attempt.ticket, value, d)
            else:
                d = ExecutionDiagnostic(
                    phase,
                    "completed",
                    diagnostic_ranges(saved, phase, value),
                    saved.plan.fingerprint,
                )
                plans.save_selection(lease.planning, attempt.ticket, value, d)

    def _extract(self, lease, command):
        saved = self.repository.load_plan(lease)
        candidates = self.repository.candidates(lease)
        for window_id, window in candidates.windows():
            request = candidate_input(lease, window_id, window, saved, payload_digest(command))
            for _ in range(MAX_ATTEMPTS):
                ticket = candidates.reserve(request)
                if ticket is None:
                    break
                self.repository.ensure_work(lease)
                try:
                    accepted = self.validate_reply(
                        self.extract_reply(lease, request, ticket.number), request
                    )
                    error = None
                except CandidateValidationError as exc:
                    accepted, error = (), exc.code
                candidates.save(ticket, accepted, error)
                if error is None:
                    break

    def extract_reply(self, lease, request, attempt):
        return self.extractor.extract(request, attempt)

    validate_reply = staticmethod(validate_reply)
