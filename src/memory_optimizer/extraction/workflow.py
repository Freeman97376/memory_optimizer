"""Offline candidate stage. Provider work and validation occur outside transactions."""

from dataclasses import dataclass

from memory_optimizer.domain import IngestCommand, ProcessingUnavailable, payload_digest
from memory_optimizer.extraction.ports import CandidateRepository
from memory_optimizer.extraction.types import (
    MAX_ATTEMPTS,
    CandidateValidationError,
    OfflineExtractor,
    prepare_input,
)
from memory_optimizer.extraction.validation import validate_reply
from memory_optimizer.processing.types import ProcessingSpec, RunLease


@dataclass(frozen=True)
class OfflineCandidateFlow:
    repository: CandidateRepository
    extractor: OfflineExtractor

    def check_version(self, spec: ProcessingSpec) -> None:
        if (
            spec.flow_version != "offline-candidates-v1"
            or spec.processor_version != self.extractor.version
        ):
            raise ProcessingUnavailable("required offline processor version unavailable")

    def process(self, lease: RunLease, command: IngestCommand) -> None:
        self.check_version(lease.spec)
        source_hash = payload_digest(command)
        for window_id, window in self.repository.windows(lease):
            request = prepare_input(lease, window_id, window, command.messages, source_hash)
            # Storage counts reservations (including interrupted ones), not this loop.
            for _ in range(MAX_ATTEMPTS):
                ticket = self.repository.reserve(lease, request)
                if ticket is None:
                    break
                try:
                    candidates = validate_reply(
                        self.extractor.extract(request, ticket.number), request
                    )
                    error = None
                except CandidateValidationError as exc:
                    candidates, error = (), exc.code
                self.repository.save(lease, ticket, candidates, error)
                if error is None:
                    break
        self.repository.finish(lease)
