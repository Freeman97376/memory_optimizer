"""Atomically persist temporal sidecars with existing candidate attempts and links."""

from contextlib import contextmanager
from uuid import UUID, uuid5

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.storage.candidates import PostgresCandidateRepository
from memory_optimizer.storage.coverage import PostgresCoverageRepository
from memory_optimizer.storage.coverage_candidates import BoundCandidateRepository
from memory_optimizer.temporal.proposals import FLOW_VERSION, TIME_SCHEMA


class TemporalCandidateRepository(BoundCandidateRepository):
    FLOW_VERSION = FLOW_VERSION

    def __init__(self, owner, lease, snapshot, payload_hash):
        super().__init__(owner, lease, snapshot, payload_hash)
        self.owner = owner

    def save(self, ticket, candidates, error):
        if self.lease.spec.flow_version != self.FLOW_VERSION or (candidates and error is not None):
            raise ProcessingUnavailable("temporal persistence version mismatch")
        with self.owner.bounded_transaction(self.lease, snapshot=self.snapshot) as conn:

            @contextmanager
            def same_transaction():
                yield conn

            PostgresCandidateRepository(same_transaction).save(
                self.lease,
                ticket,
                tuple(c.candidate for c in candidates),
                error,
            )
            for candidate in candidates:
                cid = uuid5(UUID(self.lease.run_id), "candidate:" + candidate.candidate.fingerprint)
                body = candidate.body()
                encoded = canonical(body)
                conn.execute(
                    """INSERT INTO candidate_temporal(candidate_id,window_id,run_id,batch_id,
                       schema_version,input_hash,body,body_hash)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        cid,
                        ticket.window_id,
                        self.lease.run_id,
                        self.lease.batch_id,
                        TIME_SCHEMA,
                        ticket.input_hash,
                        Jsonb(body),
                        digest(encoded),
                    ),
                )


class PostgresTemporalRepository(PostgresCoverageRepository):
    def candidate_adapter(self, lease, snapshot, payload):
        return TemporalCandidateRepository(self, lease, snapshot, payload)
