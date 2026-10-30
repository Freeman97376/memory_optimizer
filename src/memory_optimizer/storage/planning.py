"""Atomic planning checkpoints. Pure validation runs outside write transactions."""

import json
from dataclasses import asdict
from uuid import UUID, uuid4, uuid5

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingBusy, ProcessingUnavailable, SourceMessage
from memory_optimizer.processing import plan_codec
from memory_optimizer.processing.checkpoints import (
    PlanningAttempt,
    PlanningLease,
    PlanningSnapshot,
    PlanningTicket,
    RunBudget,
    attempt_input,
    validate_diagnostic,
)
from memory_optimizer.processing.plan_types import ExecutionDiagnostic, require_planning_spec
from memory_optimizer.processing.plan_validation import validate_final_selection, validate_plan
from memory_optimizer.processing.planning_rules import source_digest


def body(value):
    return json.loads(json.dumps(asdict(value)))


def duration(seconds):
    if type(seconds) is not int or not 1 <= seconds <= 300:
        raise ValueError("invalid planning lease duration")


def insert_plan_run(conn, batch_id, source_hash, spec, budget, created_at=None):
    """Storage initializer shared with atomic source ingest; caller owns transaction."""
    conn.execute(
        """WITH stamp AS (SELECT COALESCE(%s::timestamptz, clock_timestamp()) AS now)
        INSERT INTO planning_runs(id,batch_id,source_hash,spec,
            total_seconds,max_attempts,created_at,deadline_at)
        SELECT %s,%s,%s,%s,%s,%s,now,now + make_interval(secs => %s) FROM stamp""",
        (
            created_at,
            uuid5(UUID(str(batch_id)), "coverage-planning:1"),
            batch_id,
            source_hash,
            Jsonb(body(spec)),
            budget.total_seconds,
            budget.max_attempts,
            budget.total_seconds,
        ),
    )


def read_attempt(row):
    return PlanningAttempt(
        PlanningTicket(
            row["number"], str(row["reservation_id"]), row["phase"], row["input_hash"], row["fence"]
        ),
        plan_codec.decode(row["diagnostic"], plan_codec.diagnostic)
        if row["diagnostic"] is not None
        else None,
        row["started_at"],
        row["finished_at"],
    )


class PostgresPlanningRepository:
    def __init__(self, transaction):
        self.transaction = transaction

    @staticmethod
    def _batch(conn, user_id, batch_id, write=False):
        # All methods lock batch -> run -> source/attempt rows in the same order.
        # UPDATE also blocks source inserts through their FK during checkpoint writes.
        suffix = " FOR UPDATE" if write else " FOR SHARE"
        row = conn.execute(
            "SELECT id FROM batches WHERE id = %s AND user_id = %s" + suffix, (batch_id, user_id)
        ).fetchone()
        if row is None:
            raise ProcessingUnavailable("planning source unavailable")

    @staticmethod
    def _sources(conn, batch_id):
        rows = conn.execute(
            "SELECT ordinal,role,content,source_timestamp FROM messages "
            "WHERE batch_id = %s ORDER BY ordinal FOR SHARE",
            (batch_id,),
        ).fetchall()
        if [r["ordinal"] for r in rows] != list(range(len(rows))) or not rows:
            raise ProcessingUnavailable("incomplete planning source")
        return tuple(SourceMessage(r["role"], r["content"], r["source_timestamp"]) for r in rows)

    def _row(self, conn, user_id, batch_id, write=False):
        self._batch(conn, user_id, batch_id, write)
        row = conn.execute(
            "SELECT * FROM planning_runs WHERE batch_id = %s AND revision = 1"
            + (" FOR UPDATE" if write else " FOR SHARE"),
            (batch_id,),
        ).fetchone()
        if row is None:
            raise ProcessingUnavailable("planning run unavailable")
        return row

    def _lock(self, conn, lease):
        if type(lease) is not PlanningLease:
            raise ProcessingBusy("invalid planning lease")
        row = self._row(conn, lease.user_id, lease.batch_id, True)
        now = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
        if (
            str(row["id"]) != lease.run_id
            or str(row["owner"]) != lease.owner
            or row["fence"] != lease.fence
            or row["lease_until"] is None
            or row["lease_until"] <= now
        ):
            raise ProcessingBusy("planning lease lost")
        return row

    @staticmethod
    def _finish_guard(conn, lease, work=False):
        row = conn.execute(
            "SELECT id FROM planning_runs WHERE id = %s AND owner = %s AND fence = %s "
            "AND lease_until > clock_timestamp()"
            + (" AND deadline_at > clock_timestamp() AND closed_reason IS NULL" if work else ""),
            (lease.run_id, lease.owner, lease.fence),
        ).fetchone()
        if row is None:
            raise ProcessingBusy("planning lease or deadline expired during transaction")

    def _unchanged(self, conn, row, snapshot):
        if (
            row["source_hash"] != snapshot.source_hash
            or row["spec"] != body(snapshot.spec)
            or row["deadline_at"] != snapshot.deadline_at
            or (row["total_seconds"], row["max_attempts"])
            != (snapshot.budget.total_seconds, snapshot.budget.max_attempts)
            or self._sources(conn, snapshot.batch_id) != snapshot.messages
        ):
            raise ProcessingUnavailable("planning input changed before checkpoint")

    def initialize(self, user_id, batch_id, spec, budget):
        require_planning_spec(spec)
        if type(budget) is not RunBudget:
            raise ValueError("invalid planning run budget")
        with self.transaction() as conn:
            self._batch(conn, user_id, batch_id, True)
            source_hash = source_digest(self._sources(conn, batch_id))
            row = conn.execute(
                "SELECT * FROM planning_runs WHERE batch_id = %s AND revision = 1 FOR UPDATE",
                (batch_id,),
            ).fetchone()
            if row is not None:
                if (
                    row["spec"] != body(spec)
                    or row["source_hash"] != source_hash
                    or row["total_seconds"] != budget.total_seconds
                    or row["max_attempts"] != budget.max_attempts
                ):
                    raise ProcessingUnavailable(
                        "planning initialization conflicts with saved input"
                    )
            else:
                insert_plan_run(conn, batch_id, source_hash, spec, budget)
        return self.load(user_id, batch_id)

    def load(self, user_id, batch_id):
        with self.transaction() as conn:
            row = self._row(conn, user_id, batch_id)
            messages = self._sources(conn, batch_id)
            attempts = conn.execute(
                "SELECT * FROM planning_attempts WHERE run_id = %s ORDER BY number", (row["id"],)
            ).fetchall()
        return decode_snapshot(row, messages, attempts, user_id, batch_id)

    @staticmethod
    def _end_reserved(conn, row, outcome):
        pending = conn.execute(
            "SELECT number,phase FROM planning_attempts WHERE run_id = %s AND status = 'reserved'",
            (row["id"],),
        ).fetchall()
        for item in pending:
            diagnostic = ExecutionDiagnostic(
                item["phase"],
                outcome,
                (),
                row["plan_hash"] if item["phase"] == "selection" else None,
            )
            conn.execute(
                "UPDATE planning_attempts SET status = %s,diagnostic = %s,finished_at"
                " = clock_timestamp() "
                "WHERE run_id = %s AND number = %s",
                (outcome, Jsonb(body(diagnostic)), row["id"], item["number"]),
            )

    def claim(self, user_id, batch_id, lease_seconds):
        duration(lease_seconds)
        snapshot = self.load(user_id, batch_id)
        with self.transaction() as conn:
            row = self._row(conn, user_id, batch_id, True)
            self._unchanged(conn, row, snapshot)
            now = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
            if row["stage"] == "selection_ready":
                raise ProcessingUnavailable("planning selection already ready; load checkpoint")
            if row["lease_until"] is not None and row["lease_until"] > now:
                raise ProcessingBusy("planning run already leased")
            self._end_reserved(
                conn, row, "deadline_exhausted" if now >= row["deadline_at"] else "interrupted"
            )
            owner = uuid4()
            updated = conn.execute(
                """UPDATE planning_runs SET owner = %s,fence = fence + 1,
                lease_until = clock_timestamp() + make_interval(secs => %s)
                WHERE id = %s RETURNING fence""",
                (owner, lease_seconds, row["id"]),
            ).fetchone()
            lease = PlanningLease(str(row["id"]), batch_id, user_id, str(owner), updated["fence"])
            self._finish_guard(conn, lease)
        return lease

    def renew(self, lease, lease_seconds):
        duration(lease_seconds)
        with self.transaction() as conn:
            self._lock(conn, lease)
            conn.execute(
                "UPDATE planning_runs SET lease_until = clock_timestamp() + "
                "make_interval(secs => %s) WHERE id = %s",
                (lease_seconds, lease.run_id),
            )
            self._finish_guard(conn, lease)

    def reserve(self, lease, phase, reservation_id):
        reservation_id = str(UUID(reservation_id))
        snapshot = self.load(lease.user_id, lease.batch_id)
        input_hash = attempt_input(snapshot, phase)
        result = None
        with self.transaction() as conn:
            row = self._lock(conn, lease)
            self._unchanged(conn, row, snapshot)
            existing = conn.execute(
                "SELECT * FROM planning_attempts WHERE run_id = %s AND reservation_id = %s",
                (lease.run_id, reservation_id),
            ).fetchone()
            if existing is not None:
                if (existing["phase"], existing["input_hash"], existing["fence"]) != (
                    phase,
                    input_hash,
                    lease.fence,
                ):
                    raise ProcessingUnavailable("reservation identity conflict")
                result = read_attempt(existing)
            else:
                expired = conn.execute(
                    "SELECT deadline_at <= clock_timestamp() AS expired FROM "
                    "planning_runs WHERE id = %s",
                    (lease.run_id,),
                ).fetchone()["expired"]
                count = conn.execute(
                    "SELECT count(*) AS n,count(*) FILTER (WHERE status = 'reserved') AS pending "
                    "FROM planning_attempts WHERE run_id = %s",
                    (lease.run_id,),
                ).fetchone()
                reason = row["closed_reason"]
                if expired:
                    self._end_reserved(conn, row, "deadline_exhausted")
                    reason = reason or "deadline_exhausted"
                elif count["pending"]:
                    raise ProcessingBusy("planning attempt already reserved")
                elif count["n"] >= row["max_attempts"]:
                    reason = reason or "attempts_exhausted"
                if reason:
                    conn.execute(
                        "UPDATE planning_runs SET closed_reason = %s WHERE id = %s",
                        (reason, lease.run_id),
                    )
                else:
                    expected = "raw_ready" if phase == "planning" else "plan_ready"
                    if row["stage"] != expected:
                        raise ProcessingUnavailable("planning phase does not match checkpoint")
                    stored = conn.execute(
                        """INSERT INTO planning_attempts(
                            run_id,number,reservation_id,phase,input_hash,fence)
                        VALUES (%s,%s,%s,%s,%s,%s) RETURNING *""",
                        (
                            lease.run_id,
                            count["n"] + 1,
                            reservation_id,
                            phase,
                            input_hash,
                            lease.fence,
                        ),
                    ).fetchone()
                    result = read_attempt(stored)
                    self._finish_guard(conn, lease, work=True)
            self._finish_guard(conn, lease)
        return result

    @staticmethod
    def _attempt(conn, lease, ticket):
        row = conn.execute(
            "SELECT * FROM planning_attempts WHERE run_id = %s AND number = %s",
            (lease.run_id, ticket.number),
        ).fetchone()
        if row is None or read_attempt(row).ticket != ticket or ticket.fence != lease.fence:
            raise ProcessingBusy("planning attempt is no longer owned")
        return row

    @staticmethod
    def _save_attempt(conn, lease, ticket, diagnostic):
        conn.execute(
            "UPDATE planning_attempts SET status = %s,diagnostic = %s,finished_at"
            " = clock_timestamp() "
            "WHERE run_id = %s AND number = %s",
            (diagnostic.outcome, Jsonb(body(diagnostic)), lease.run_id, ticket.number),
        )

    def save_plan(self, lease, ticket, plan, diagnostic):
        snapshot = self.load(lease.user_id, lease.batch_id)
        validate_plan(plan, snapshot.messages, snapshot.spec)
        validate_diagnostic(diagnostic, snapshot, plan=plan)
        self._checkpoint(lease, ticket, snapshot, diagnostic, plan, "planning")

    def save_selection(self, lease, ticket, selection, diagnostic):
        snapshot = self.load(lease.user_id, lease.batch_id)
        validate_final_selection(selection, snapshot.plan, snapshot.messages)
        validate_diagnostic(diagnostic, snapshot, selection=selection)
        self._checkpoint(lease, ticket, snapshot, diagnostic, selection, "selection")

    def _checkpoint(self, lease, ticket, snapshot, diagnostic, value, phase):
        if ticket.phase != phase or diagnostic.phase != phase or diagnostic.outcome != "completed":
            raise ValueError("checkpoint requires a matching completed diagnostic")
        with self.transaction() as conn:
            row = self._lock(conn, lease)
            self._unchanged(conn, row, snapshot)
            attempt = self._attempt(conn, lease, ticket)
            saved = row["plan_body"] if phase == "planning" else row["selection_body"]
            if (
                attempt["status"] == "completed"
                and saved == body(value)
                and attempt["diagnostic"] == body(diagnostic)
            ):
                self._finish_guard(conn, lease)
                return  # Exact retry is a read-only acknowledgement; no timestamps change.
            if attempt["status"] != "reserved" or row["stage"] != (
                "raw_ready" if phase == "planning" else "plan_ready"
            ):
                raise ProcessingBusy("planning checkpoint no longer writable")
            self._finish_guard(conn, lease, work=True)
            self._save_attempt(conn, lease, ticket, diagnostic)
            if phase == "planning":
                conn.execute(
                    "UPDATE planning_runs SET plan_body = %s,plan_hash = %s,planned_at = "
                    "clock_timestamp(),stage = 'plan_ready' WHERE id = %s",
                    (Jsonb(body(value)), value.fingerprint, lease.run_id),
                )
            else:
                conn.execute(
                    "UPDATE planning_runs SET selection_body = %s,selected_at = "
                    "clock_timestamp(),stage = 'selection_ready' WHERE id = %s",
                    (Jsonb(body(value)), lease.run_id),
                )
            self._finish_guard(conn, lease, work=True)

    def save_diagnostic(self, lease, ticket, diagnostic):
        snapshot = self.load(lease.user_id, lease.batch_id)
        if diagnostic.outcome == "completed" or diagnostic.phase != ticket.phase:
            raise ValueError("completed diagnostics must commit with their checkpoint")
        validate_diagnostic(diagnostic, snapshot)
        with self.transaction() as conn:
            row = self._lock(conn, lease)
            self._unchanged(conn, row, snapshot)
            attempt = self._attempt(conn, lease, ticket)
            if attempt["status"] == diagnostic.outcome and attempt["diagnostic"] == body(
                diagnostic
            ):
                self._finish_guard(conn, lease)
                return
            if attempt["status"] != "reserved":
                raise ProcessingBusy("planning attempt already finished")
            expired = conn.execute(
                "SELECT deadline_at <= clock_timestamp() AS expired FROM "
                "planning_runs WHERE id = %s",
                (lease.run_id,),
            ).fetchone()["expired"]
            if diagnostic.outcome == "deadline_exhausted" and not expired:
                raise ValueError("planning total deadline has not expired")
            self._save_attempt(conn, lease, ticket, diagnostic)
            if expired:
                conn.execute(
                    "UPDATE planning_runs SET closed_reason = "
                    "COALESCE(closed_reason,'deadline_exhausted') WHERE id = %s",
                    (lease.run_id,),
                )
            self._finish_guard(conn, lease)


def decode_snapshot(row, messages, attempts, user_id, batch_id):
    # Decode/validate after releasing the read transaction, never repair corrupt data.
    spec = plan_codec.decode(row["spec"], plan_codec.spec)
    plan = (
        plan_codec.decode(row["plan_body"], plan_codec.plan)
        if row["plan_body"] is not None
        else None
    )
    selection = (
        plan_codec.decode(row["selection_body"], plan_codec.selection)
        if row["selection_body"] is not None
        else None
    )
    snapshot = PlanningSnapshot(
        str(row["id"]),
        batch_id,
        user_id,
        messages,
        row["source_hash"],
        spec,
        RunBudget(row["total_seconds"], row["max_attempts"]),
        row["created_at"],
        row["deadline_at"],
        row["stage"],
        plan,
        selection,
        tuple(read_attempt(a) for a in attempts),
        row["closed_reason"],
    )
    try:
        if source_digest(messages) != snapshot.source_hash:
            raise ValueError("source changed")
        if plan is not None:
            validate_plan(plan, messages, spec)
            if plan.fingerprint != row["plan_hash"]:
                raise ValueError("plan hash mismatch")
        if selection is not None:
            validate_final_selection(selection, plan, messages)
        if [a.ticket.number for a in snapshot.attempts] != list(range(1, len(attempts) + 1)) or len(
            attempts
        ) > snapshot.budget.max_attempts:
            raise ValueError("invalid attempt ledger")
        completed = {"planning": 0, "selection": 0}
        stage = "raw_ready"
        for attempt, stored in zip(snapshot.attempts, attempts, strict=True):
            expected = "planning" if stage == "raw_ready" else "selection"
            if (
                stage == "selection_ready"
                or attempt.ticket.phase != expected
                or not 0 < attempt.ticket.fence <= row["fence"]
            ):
                raise ValueError("invalid attempt order or owner generation")
            if attempt.ticket.input_hash != attempt_input(snapshot, attempt.ticket.phase):
                raise ValueError("attempt input mismatch")
            d = attempt.diagnostic
            if stored["status"] != (d.outcome if d is not None else "reserved"):
                raise ValueError("attempt diagnostic mismatch")
            if d is None and (
                attempt.ticket.number != len(attempts)
                or attempt.ticket.fence != row["fence"]
                or snapshot.closed_reason is not None
            ):
                raise ValueError("invalid pending attempt")
            if d is not None:
                if d.phase != attempt.ticket.phase:
                    raise ValueError("attempt phase mismatch")
                validate_diagnostic(
                    d,
                    snapshot,
                    plan=plan if d.outcome == "completed" else None,
                    selection=selection if d.outcome == "completed" else None,
                )
                if d.outcome == "completed":
                    completed[d.phase] += 1
                    stage = "plan_ready" if d.phase == "planning" else "selection_ready"
        if stage != snapshot.stage:
            raise ValueError("stage does not match attempt history")
        if completed != {
            "planning": int(plan is not None),
            "selection": int(selection is not None),
        }:
            raise ValueError("checkpoint has no matching completed attempt")
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ProcessingUnavailable("incompatible persisted planning checkpoint") from None
    return snapshot
