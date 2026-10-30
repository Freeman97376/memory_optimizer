"""Atomic initialization and ownership for the explicit offline coverage runtime."""

from contextlib import contextmanager
from dataclasses import asdict
from uuid import UUID, uuid4, uuid5

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingBusy, ProcessingUnavailable
from memory_optimizer.extraction.types import MAX_ATTEMPTS
from memory_optimizer.processing.checkpoints import PlanningLease
from memory_optimizer.processing.coverage_types import (
    CoverageLease,
    CoverageSpec,
    WorkTimeExhausted,
    final_fingerprint,
    final_windows,
    has_selection_gap,
    read_coverage_spec,
)
from memory_optimizer.processing.planning_rules import source_digest
from memory_optimizer.storage.planning import (
    PostgresPlanningRepository,
    body,
    duration,
    insert_plan_run,
)


class CoverageIngestRepository:
    def __init__(self, store):
        self.store = store

    def commit(self, plan):
        if type(plan.processing) is not CoverageSpec:
            raise ProcessingUnavailable("coverage initialization requires its specification")
        source_hash = source_digest(plan.command.messages)
        return self.store.commit_initialized(
            plan, lambda conn, batch, value: self.initialize(conn, batch, value, source_hash)
        )

    @staticmethod
    def initialize(conn, batch_id, plan, source_hash):
        spec = plan.processing
        created = conn.execute(
            "SELECT created_at FROM batches WHERE id = %s", (batch_id,)
        ).fetchone()["created_at"]
        insert_plan_run(conn, batch_id, source_hash, spec.planning, spec.budget, created)
        conn.execute(
            """INSERT INTO processing_runs(id,batch_id,revision,spec,stage,created_at)
            VALUES (%s,%s,1,%s,'raw_ready',%s)""",
            (uuid5(batch_id, "processing:1"), batch_id, Jsonb(body(spec)), created),
        )


class PostgresCoverageRepository:
    def __init__(self, transaction):
        self.transaction = transaction
        self.planning = PostgresPlanningRepository(transaction)

    def _rows(self, conn, user_id, batch_id):
        # Same order as independent planning: batch -> planning -> runtime.
        self.planning._batch(conn, user_id, batch_id, True)
        plan = conn.execute(
            "SELECT * FROM planning_runs WHERE batch_id=%s AND revision=1 FOR UPDATE", (batch_id,)
        ).fetchone()
        run = conn.execute(
            "SELECT * FROM processing_runs WHERE batch_id=%s AND revision=1 FOR UPDATE", (batch_id,)
        ).fetchone()
        if run is None:
            raise ProcessingUnavailable("coverage runtime unavailable")
        return plan, run

    @staticmethod
    def _configuration(plan, run, spec):
        if (
            plan is None
            or run["spec"] != body(spec)
            or plan["spec"] != body(spec.planning)
            or (plan["total_seconds"], plan["max_attempts"])
            != (spec.budget.total_seconds, spec.budget.max_attempts)
        ):
            raise ProcessingUnavailable("coverage configuration mismatch")

    def claim(self, user_id, batch_id, spec, lease_seconds):
        duration(lease_seconds)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT r.stage,r.spec FROM processing_runs r JOIN batches b ON b.id=r.batch_id "
                "WHERE r.batch_id=%s AND b.user_id=%s AND r.revision=1",
                (batch_id, user_id),
            ).fetchone()
        if row is None:
            raise ProcessingUnavailable("coverage runtime unavailable")
        if row["stage"] == "complete":
            return None
        if read_coverage_spec(row["spec"]) != spec:
            raise ProcessingUnavailable("required coverage configuration unavailable")
        snapshot = self.planning.load(user_id, batch_id)
        with self.transaction() as conn:
            plan, run = self._rows(conn, user_id, batch_id)
            if run["stage"] == "complete":
                return None
            self._configuration(plan, run, spec)
            self.planning._unchanged(conn, plan, snapshot)
            now = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
            if any(r["lease_until"] is not None and r["lease_until"] > now for r in (plan, run)):
                raise ProcessingBusy("coverage runtime already leased")
            self.planning._end_reserved(
                conn, plan, "deadline_exhausted" if now >= plan["deadline_at"] else "interrupted"
            )
            owner, fence = str(uuid4()), max(plan["fence"], run["fence"]) + 1
            for table, identity in (("planning_runs", plan["id"]), ("processing_runs", run["id"])):
                conn.execute(
                    f"UPDATE {table} SET owner=%s,fence=%s,"
                    "lease_until=%s + make_interval(secs => %s) WHERE id=%s",
                    (owner, fence, now, lease_seconds, identity),
                )
            lease = CoverageLease(
                str(run["id"]),
                batch_id,
                user_id,
                owner,
                fence,
                spec,
                PlanningLease(str(plan["id"]), batch_id, user_id, owner, fence),
                lease_seconds,
            )
            self._guard(conn, lease)
        return lease

    def _guard(self, conn, lease, *, work=False):
        if (
            lease.planning.batch_id,
            lease.planning.user_id,
            lease.planning.owner,
            lease.planning.fence,
        ) != (lease.batch_id, lease.user_id, lease.owner, lease.fence):
            raise ProcessingBusy("coverage lease pair mismatch")
        plan, run = self._rows(conn, lease.user_id, lease.batch_id)
        self._configuration(plan, run, lease.spec)
        now = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
        if (
            str(plan["id"]) != lease.planning.run_id
            or str(run["id"]) != lease.run_id
            or run["stage"] == "complete"
            or any(
                str(r["owner"]) != lease.owner
                or r["fence"] != lease.fence
                or r["lease_until"] is None
                or r["lease_until"] <= now
                for r in (plan, run)
            )
        ):
            raise ProcessingBusy("coverage ownership lost")
        remaining = (plan["deadline_at"] - now).total_seconds()
        if work and remaining <= lease.spec.cleanup_seconds:
            raise WorkTimeExhausted("coverage work time exhausted")
        return plan, run, remaining

    def _renew(self, conn, lease):
        self._guard(conn, lease)
        stamp = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
        for table, identity in (
            ("planning_runs", lease.planning.run_id),
            ("processing_runs", lease.run_id),
        ):
            conn.execute(
                f"UPDATE {table} SET lease_until=%s + make_interval(secs => %s) WHERE id=%s",
                (stamp, lease.lease_seconds, identity),
            )

    @contextmanager
    def bounded_transaction(self, lease, *, work=True, snapshot=None):
        if snapshot is not None and (snapshot.run_id, snapshot.batch_id, snapshot.user_id) != (
            lease.planning.run_id,
            lease.batch_id,
            lease.user_id,
        ):
            raise ProcessingUnavailable("coverage snapshot scope mismatch")
        windows = final_windows(snapshot) if snapshot is not None else None
        with self.transaction() as conn:
            plan, _, _ = self._guard(conn, lease, work=work)
            if snapshot is not None:
                self.planning._unchanged(conn, plan, snapshot)
                self._check_binding(conn, lease, snapshot, windows)
            yield conn
            self._guard(conn, lease, work=work)
            self._renew(conn, lease)

    def load_plan(self, lease):
        return self.planning.load(lease.user_id, lease.batch_id)

    def plans(self, lease, *, work=True):
        return PostgresPlanningRepository(lambda: self.bounded_transaction(lease, work=work))

    def ensure_work(self, lease):
        with self.bounded_transaction(lease):
            pass

    def bind(self, lease):
        snapshot = self.load_plan(lease)
        windows = final_windows(snapshot)
        with self.bounded_transaction(lease) as conn:
            plan, run, _ = self._guard(conn, lease, work=True)
            self.planning._unchanged(conn, plan, snapshot)
            if run["stage"] == "selected":
                self._check_binding(conn, lease, snapshot, windows)
                return
            if plan["stage"] != "selection_ready" or run["stage"] != "raw_ready":
                raise ProcessingUnavailable("coverage final selection unavailable")
            for window in windows:
                conn.execute(
                    """INSERT INTO processing_windows(id,run_id,batch_id,ordinal,message_ordinal,
                    start_offset,end_offset,context_spans,reasons)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        uuid5(UUID(lease.run_id), f"window:{window.ordinal}"),
                        lease.run_id,
                        lease.batch_id,
                        window.ordinal,
                        window.focus.message_ordinal,
                        window.focus.start,
                        window.focus.end,
                        Jsonb([asdict(s) for s in window.context]),
                        Jsonb(list(window.reasons)),
                    ),
                )
            conn.execute(
                "INSERT INTO "
                "coverage_bindings(run_id,batch_id,planning_id,selection_hash,adapter_version) "
                "VALUES (%s,%s,%s,%s,%s)",
                (
                    lease.run_id,
                    lease.batch_id,
                    lease.planning.run_id,
                    final_fingerprint(snapshot),
                    lease.spec.adapter_version,
                ),
            )
            conn.execute(
                "UPDATE processing_runs SET "
                "stage='selected',candidate_count=%s,selected_count=%s,"
                "selected_at=clock_timestamp() WHERE id=%s",
                (len(snapshot.plan.pool), len(windows), lease.run_id),
            )

    @staticmethod
    def _check_binding(conn, lease, snapshot, windows):
        binding = conn.execute(
            "SELECT * FROM coverage_bindings WHERE run_id=%s", (lease.run_id,)
        ).fetchone()
        rows = conn.execute(
            "SELECT * FROM processing_windows WHERE run_id=%s ORDER BY ordinal", (lease.run_id,)
        ).fetchall()
        run = conn.execute(
            "SELECT stage,candidate_count,selected_count FROM processing_runs WHERE id=%s",
            (lease.run_id,),
        ).fetchone()
        if (
            binding is None
            or str(binding["planning_id"]) != lease.planning.run_id
            or str(binding["batch_id"]) != lease.batch_id
            or binding["selection_hash"] != final_fingerprint(snapshot)
            or binding["adapter_version"] != lease.spec.adapter_version
            or len(rows) != len(windows)
            or run["stage"] != "selected"
            or run["candidate_count"] != len(snapshot.plan.pool)
            or run["selected_count"] != len(windows)
        ):
            raise ProcessingUnavailable("incompatible coverage candidate binding")
        for row, window in zip(rows, windows, strict=True):
            if (
                str(row["id"]) != str(uuid5(UUID(lease.run_id), f"window:{window.ordinal}"))
                or (row["ordinal"], row["message_ordinal"], row["start_offset"], row["end_offset"])
                != (
                    window.ordinal,
                    window.focus.message_ordinal,
                    window.focus.start,
                    window.focus.end,
                )
                or row["context_spans"] != [body(s) for s in window.context]
                or row["reasons"] != list(window.reasons)
                or row["status"] != "pending"
            ):
                raise ProcessingUnavailable("coverage candidate window changed")

    def candidates(self, lease):
        snapshot = self.load_plan(lease)
        with self.bounded_transaction(lease, snapshot=snapshot) as conn:
            payload = conn.execute(
                "SELECT payload_hash FROM batches WHERE id=%s", (lease.batch_id,)
            ).fetchone()["payload_hash"]
        return self.candidate_adapter(lease, snapshot, payload)

    def candidate_adapter(self, lease, snapshot, payload):
        from memory_optimizer.storage.coverage_candidates import BoundCandidateRepository

        return BoundCandidateRepository(self, lease, snapshot, payload)

    def complete(self, lease, reason=None):
        snapshot = self.load_plan(lease)
        windows = final_windows(snapshot) if snapshot.selection is not None else None
        with self.transaction() as conn:
            plan, run, remaining = self._guard(conn, lease)
            self.planning._unchanged(conn, plan, snapshot)
            if remaining <= lease.spec.cleanup_seconds:
                reason = "time_budget_exhausted"
            elif reason == "time_budget_exhausted":
                raise ProcessingUnavailable("coverage work time remains")
            elif reason == "planning_attempts_exhausted":
                if plan["closed_reason"] != "attempts_exhausted" or snapshot.selection is not None:
                    raise ProcessingUnavailable("planning attempts remain")
            elif reason is not None:
                raise ValueError("unsupported coverage terminal reason")
            if run["stage"] == "selected":
                self._check_binding(conn, lease, snapshot, windows)
            rows = conn.execute(
                """SELECT w.id,count(a.number) AS attempts,
                count(*) FILTER (WHERE a.status IN ('accepted','empty')) AS completed,
                count(*) FILTER (WHERE a.status='reserved') AS pending
                FROM processing_windows w LEFT JOIN extraction_attempts a ON a.window_id=w.id
                WHERE w.run_id=%s GROUP BY w.id""",
                (lease.run_id,),
            ).fetchall()
            if reason is None:
                if run["stage"] != "selected" or any(
                    r["pending"] or (not r["completed"] and r["attempts"] < MAX_ATTEMPTS)
                    for r in rows
                ):
                    raise ProcessingUnavailable("coverage candidate processing incomplete")
                reason = (
                    "extraction_failed"
                    if any(not r["completed"] for r in rows)
                    else "selection_budget"
                    if has_selection_gap(snapshot)
                    else "offline_only"
                    if rows
                    else None
                )
            self.planning._end_reserved(
                conn, plan, "deadline_exhausted" if remaining <= 0 else "interrupted"
            )
            conn.execute(
                "UPDATE extraction_attempts SET "
                "status='failed',error_code='time_budget_exhausted',finished_at=clock_timestamp() "
                "WHERE run_id=%s AND status='reserved'",
                (lease.run_id,),
            )
            conn.execute(
                "UPDATE processing_windows SET status='deferred' WHERE run_id=%s", (lease.run_id,)
            )
            self._guard(conn, lease)
            updated = conn.execute(
                """UPDATE processing_runs SET stage='complete',outcome=%s,reason=%s,owner=NULL,
                lease_until=NULL,finished_at=clock_timestamp()
                WHERE id=%s AND lease_until>clock_timestamp() RETURNING id""",
                ("raw_degraded" if reason else "raw_not_selected", reason, lease.run_id),
            ).fetchone()
            cleared = conn.execute(
                "UPDATE planning_runs SET owner=NULL,lease_until=NULL WHERE id=%s AND "
                "lease_until>clock_timestamp() RETURNING id",
                (lease.planning.run_id,),
            ).fetchone()
            if updated is None or cleared is None:
                raise ProcessingBusy("coverage terminal ownership expired")
