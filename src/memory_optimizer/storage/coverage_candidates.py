"""Bounded adapter reusing the existing unpublished candidate persistence."""

from uuid import UUID, uuid5

from memory_optimizer.domain import ProcessingUnavailable
from memory_optimizer.processing.coverage_types import candidate_input, final_windows
from memory_optimizer.storage.candidates import PostgresCandidateRepository


class BoundCandidateRepository:
    def __init__(self, owner, lease, snapshot, payload_hash):
        self.lease, self.snapshot, self.payload_hash = lease, snapshot, payload_hash
        self.repository = PostgresCandidateRepository(
            lambda: owner.bounded_transaction(lease, snapshot=snapshot)
        )
        self.expected = {
            str(uuid5(UUID(lease.run_id), f"window:{w.ordinal}")): w
            for w in final_windows(snapshot)
        }

    def windows(self):
        return self.repository.windows(self.lease)

    def reserve(self, request):
        window = self.expected.get(request.window_id)
        if window is None or request != candidate_input(
            self.lease, request.window_id, window, self.snapshot, self.payload_hash
        ):
            raise ProcessingUnavailable("coverage candidate input mismatch")
        return self.repository.reserve(self.lease, request)

    def save(self, ticket, candidates, error):
        self.repository.save(self.lease, ticket, candidates, error)
