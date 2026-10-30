"""Finite relation checkpoints over the original coverage lease and deadline."""

from contextlib import contextmanager

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingBusy, ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.contracts import (
    FLOW_VERSION,
    AttemptBudgetExhausted,
    RelationInput,
    StaleTargets,
    validate_reply,
)
from memory_optimizer.storage.relations_snapshot import build_input
from memory_optimizer.storage.temporal import (
    PostgresTemporalRepository,
    TemporalCandidateRepository,
)


class S3Candidates(TemporalCandidateRepository):
    FLOW_VERSION = FLOW_VERSION

    def reserve(self, request):
        with self.owner.bounded_transaction(self.lease) as conn:
            done = conn.execute(
                (
                    "SELECT 1 FROM extraction_attempts WHERE window_id=%s AND status IN "
                    "('accepted','empty')"
                ),
                (request.window_id,),
            ).fetchone()
            if (
                not done
                and self.owner.attempts_used(conn, self.lease)
                >= self.lease.spec.budget.max_attempts
            ):
                raise AttemptBudgetExhausted("S3 attempt budget exhausted")
        return super().reserve(request)


class PostgresRelationsRepository(PostgresTemporalRepository):
    candidate_type = S3Candidates

    def candidate_adapter(self, lease, snapshot, payload):
        return self.candidate_type(self, lease, snapshot, payload)

    @staticmethod
    def attempts_used(conn, lease):
        return conn.execute(
            """SELECT (SELECT count(*) FROM planning_attempts WHERE run_id=%s)
            +(SELECT count(*) FROM extraction_attempts WHERE run_id=%s)
            +(SELECT count(*) FROM relation_attempts WHERE run_id=%s) AS n""",
            (lease.planning.run_id, lease.run_id, lease.run_id),
        ).fetchone()["n"]

    @contextmanager
    def scoped_transaction(self, lease, *, work=True):
        # All publication writers lock scope before their own batch; no cross-batch deadlock.
        with self.transaction() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", ("s3:" + lease.user_id,)
            )
            self._guard(conn, lease, work=work)
            yield conn
            self._guard(conn, lease, work=work)
            self._renew(conn, lease)

    @staticmethod
    def check_targets(conn, request):
        for node in request.value["nodes"]:
            if node["ref"].startswith("h"):
                row = conn.execute(
                    "SELECT version_id FROM episode_heads WHERE episode_id=%s AND user_id=%s",
                    (node["id"], request.value["meta"]["user_id"]),
                ).fetchone()
                if row is None or str(row["version_id"]) != node["version"]:
                    raise StaleTargets("relation target changed")

    def latest(self, lease):
        with self.transaction() as conn:
            row = conn.execute(
                (
                    "SELECT s.* FROM relation_snapshots s JOIN processing_runs r ON r.id=s.run_id "
                    "JOIN batches b ON b.id=r.batch_id WHERE s.run_id=%s AND b.user_id=%s ORDER BY "
                    "revision DESC LIMIT 1"
                ),
                (lease.run_id, lease.user_id),
            ).fetchone()
        if row is None:
            return None
        request = RelationInput(canonical(row["body"]))
        if (
            request.fingerprint != row["input_hash"]
            or request.value["meta"]["run_id"] != lease.run_id
            or request.value["meta"]["user_id"] != lease.user_id
        ):
            raise ProcessingUnavailable("changed frozen input")
        return row["revision"], request

    def prepare(self, lease, *, rebuild=False, reason=None, work=True):
        latest = self.latest(lease)
        if latest is not None and not rebuild:
            return latest
        revision = latest[0] + 1 if latest else 1
        if revision > 2:
            raise AttemptBudgetExhausted("snapshot rebuild limit")
        request = build_input(self, lease, revision, reason)
        with self.scoped_transaction(lease, work=work) as conn:
            self.check_targets(conn, request)
            if self.attempts_used(conn, lease) >= lease.spec.budget.max_attempts and rebuild:
                raise AttemptBudgetExhausted("snapshot attempt budget exhausted")
            conn.execute(
                "INSERT INTO relation_snapshots(run_id,revision,body,input_hash) VALUES ("
                "%s,%s,%s,%s)",
                (lease.run_id, revision, Jsonb(request.value), request.fingerprint),
            )
        return revision, request

    def result(self, lease, revision, request):
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM relation_attempts WHERE run_id=%s AND revision=%s AND stat"
                "us='accepted'",
                (lease.run_id, revision),
            ).fetchone()
        if row is None:
            return None
        decision = validate_reply(row["reply"], request)
        if (
            row["input_hash"] != request.fingerprint
            or canonical(decision) != canonical(row["decision"])
            or digest(canonical(decision)) != row["decision_hash"]
        ):
            raise ProcessingUnavailable("changed saved decision")
        return decision

    def reserve_relation(self, lease, revision, request):
        with self.scoped_transaction(lease) as conn:
            self.check_targets(conn, request)
            rows = conn.execute(
                "SELECT * FROM relation_attempts WHERE run_id=%s AND revision=%s ORDER BY number",
                (lease.run_id, revision),
            ).fetchall()
            if any(r["status"] == "accepted" for r in rows):
                return None
            if any(r["status"] == "reserved" and r["fence"] == lease.fence for r in rows):
                raise ProcessingBusy("relation attempt already reserved")
            conn.execute(
                (
                    "UPDATE relation_attempts SET "
                    "status='failed',error_code='interrupted',finished_at=clock_timestamp() WHERE "
                    "run_id=%s AND status='reserved'"
                ),
                (lease.run_id,),
            )
            if len(rows) >= 2 or self.attempts_used(conn, lease) >= lease.spec.budget.max_attempts:
                return None
            number = len(rows) + 1
            conn.execute(
                (
                    "INSERT INTO relation_attempts(run_id,revision,number,fence,status,input_hash) "
                    "VALUES (%s,%s,%s,%s,'reserved',%s)"
                ),
                (lease.run_id, revision, number, lease.fence, request.fingerprint),
            )
        return number

    def save_relation(self, lease, revision, number, request, reply, decision, error):
        with self.scoped_transaction(lease) as conn:
            self.check_targets(conn, request)
            row = conn.execute(
                (
                    "UPDATE relation_attempts SET "
                    "status=%s,reply=%s,decision=%s,decision_hash=%s,error_code=%s,finished_a"
                    "t=clock_timestamp() WHERE run_id=%s AND revision=%s AND number=%s AND fe"
                    "nce=%s AND status='reserved' RETURNING number"
                ),
                (
                    "accepted" if decision is not None else "failed",
                    reply,
                    Jsonb(decision) if decision is not None else None,
                    digest(canonical(decision)) if decision is not None else None,
                    error,
                    lease.run_id,
                    revision,
                    number,
                    lease.fence,
                ),
            ).fetchone()
            if row is None:
                raise ProcessingBusy("relation reservation no longer owned")

    def publish(self, lease, request, decision):
        from memory_optimizer.storage.publication import finalize

        return finalize(self, lease, request, decision)

    def complete(self, lease, reason=None):
        from memory_optimizer.storage.publication import finalize

        latest = self.latest(lease)
        if latest is None:
            latest = self.prepare(lease, reason=reason or "relation_unresolved", work=False)
        return finalize(
            self, lease, latest[1] if latest else None, None, reason=reason or "relation_unresolved"
        )
