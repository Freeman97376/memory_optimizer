from dataclasses import dataclass
from typing import Protocol

from memory_optimizer.extraction.types import ExtractionInput, ValidatedCandidate
from memory_optimizer.processing.types import RunLease, Window


@dataclass(frozen=True)
class AttemptTicket:
    window_id: str
    number: int
    input_hash: str


class CandidateRepository(Protocol):
    def windows(self, lease: RunLease) -> tuple[tuple[str, Window], ...]: ...

    def reserve(self, lease: RunLease, request: ExtractionInput) -> AttemptTicket | None:
        """Persist an attempt, or skip an already terminal/exhausted window."""
        ...

    def save(
        self,
        lease: RunLease,
        ticket: AttemptTicket,
        candidates: tuple[ValidatedCandidate, ...],
        error: str | None,
    ) -> None: ...

    def finish(self, lease: RunLease) -> None: ...
