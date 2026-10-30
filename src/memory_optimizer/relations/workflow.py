"""Explicit S3 orchestration; old flows retain their original finish behavior."""

from dataclasses import dataclass

from memory_optimizer.extraction.types import CandidateValidationError
from memory_optimizer.processing.coverage_types import WorkTimeExhausted
from memory_optimizer.processing.coverage_workflow import CoverageFlow
from memory_optimizer.relations.contracts import (
    FLOW_VERSION,
    AttemptBudgetExhausted,
    StaleTargets,
    validate_reply,
)
from memory_optimizer.temporal.proposals import validate_temporal_reply


@dataclass(frozen=True)
class S3Flow(CoverageFlow):
    FLOW_VERSION = FLOW_VERSION
    proposer: object = None
    validate_reply = staticmethod(validate_temporal_reply)

    def finish(self, command, receipt):
        lease = self.repository.claim(
            command.user_id, receipt.batch_id, self.spec, self.lease_seconds
        )
        if lease is None:
            return
        try:
            if self.repository.latest(lease) is None:
                if not self._plan(lease):
                    self.repository.complete(lease, "planning_attempts_exhausted")
                    return
                self.repository.bind(lease)
                reason = None
                try:
                    self._extract(lease, command)
                except AttemptBudgetExhausted:
                    reason = "attempt_budget_exhausted"
                self.repository.prepare(lease, reason=reason)
            for rebuild in range(2):
                revision, request = self.repository.prepare(lease, rebuild=bool(rebuild))
                try:
                    decision = self.repository.result(lease, revision, request)
                    for _ in range(2):
                        if decision is not None:
                            break
                        number = self.repository.reserve_relation(lease, revision, request)
                        if number is None:
                            break
                        self.repository.ensure_work(lease)
                        reply = self.relation_reply(lease, request, number)
                        try:
                            decision, error = validate_reply(reply, request), None
                        except CandidateValidationError as exc:
                            decision, error = None, exc.code
                        self.repository.save_relation(
                            lease, revision, number, request, reply, decision, error
                        )
                    if decision is None:
                        self.repository.complete(lease)
                    else:
                        self.repository.publish(lease, request, decision)
                    return
                except StaleTargets:
                    continue
            self.repository.complete(lease, "stale_targets")
        except (WorkTimeExhausted, AttemptBudgetExhausted, StaleTargets) as exc:
            self.repository.complete(
                lease,
                "time_budget_exhausted"
                if isinstance(exc, WorkTimeExhausted)
                else "attempt_or_target_exhausted",
            )

    def relation_reply(self, lease, request, attempt):
        return self.proposer.propose(request, attempt)
