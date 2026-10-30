"""PostgreSQL processing adapter; it stores decisions, never chooses source windows."""

from dataclasses import asdict
from uuid import UUID, uuid4, uuid5

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingBusy, ProcessingUnavailable
from memory_optimizer.processing.types import ProcessingSpec, RunLease, Selection, SelectionConfig
from memory_optimizer.storage.lease import lock_run


def create_run(conn, batch_id: UUID, spec: ProcessingSpec | None) -> None:
    """Called inside the SAME transaction as the initial raw/index commit."""
    conn.execute(
        """INSERT INTO processing_runs
           (id, batch_id, revision, spec, stage, outcome, finished_at)
           VALUES (%s, %s, 1, %s, %s, %s,
                   CASE WHEN %s THEN clock_timestamp() ELSE NULL END)""",
        (
            uuid5(batch_id, "processing:1"),
            batch_id,
            Jsonb(asdict(spec) if spec else {"flow_version": "legacy-raw-v1"}),
            "raw_ready" if spec else "complete",
            None if spec else "raw_legacy",
            spec is None,
        ),
    )


def read_spec(value: dict) -> ProcessingSpec:
    try:
        fields = dict(value)
        fields["selection"] = SelectionConfig(**fields["selection"])
        return ProcessingSpec(**fields)
    except (KeyError, TypeError, ValueError):
        raise ProcessingUnavailable("incompatible persisted processing specification") from None


class PostgresProcessingRepository:
    def __init__(self, transaction):
        self.transaction = transaction

    def claim(self, user_id: str, batch_id: str, lease_seconds: int) -> RunLease:
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 300:
            raise ValueError("invalid lease duration")
        with self.transaction() as conn:
            row = conn.execute(
                """SELECT r.*, r.lease_until > clock_timestamp() AS live
                   FROM processing_runs r JOIN batches b ON b.id = r.batch_id
                   WHERE r.batch_id = %s AND b.user_id = %s AND r.revision = 1
                   FOR UPDATE OF r""",
                (batch_id, user_id),
            ).fetchone()
            if row is None:
                raise ProcessingUnavailable("processing run unavailable")
            if row["stage"] == "complete":
                return RunLease(
                    str(row["id"]), batch_id, user_id, None, row["fence"], "complete", None
                )
            if row["live"]:
                raise ProcessingBusy("processing run already leased")
            spec = read_spec(row["spec"])
            owner = uuid4()
            updated = conn.execute(
                """UPDATE processing_runs SET owner = %s, fence = fence + 1,
                   lease_until = clock_timestamp() + make_interval(secs => %s)
                   WHERE id = %s RETURNING fence""",
                (owner, lease_seconds, row["id"]),
            ).fetchone()
            lease = RunLease(
                str(row["id"]), batch_id, user_id, str(owner), updated["fence"], row["stage"], spec
            )
        return lease

    _lock = staticmethod(lock_run)

    def checkpoint(self, lease: RunLease, selection: Selection) -> None:
        with self.transaction() as conn:
            self._lock(conn, lease, "raw_ready")
            for window in selection.windows:
                conn.execute(
                    """INSERT INTO processing_windows
                       (id, run_id, batch_id, ordinal, message_ordinal, start_offset,
                        end_offset, context_spans, reasons)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        uuid5(UUID(lease.run_id), f"window:{window.ordinal}"),
                        lease.run_id,
                        lease.batch_id,
                        window.ordinal,
                        window.focus.message_ordinal,
                        window.focus.start,
                        window.focus.end,
                        Jsonb([asdict(span) for span in window.context]),
                        Jsonb(list(window.reasons)),
                    ),
                )
            updated = conn.execute(
                """UPDATE processing_runs SET stage = 'selected', candidate_count = %s,
                   selected_count = %s, selected_at = clock_timestamp()
                   WHERE id = %s AND lease_until > clock_timestamp() RETURNING id""",
                (selection.candidate_count, len(selection.windows), lease.run_id),
            ).fetchone()
            if updated is None:
                raise ProcessingBusy("processing lease expired during checkpoint")

    def finish_raw(self, lease: RunLease) -> None:
        with self.transaction() as conn:
            row = self._lock(conn, lease, "selected")
            count = conn.execute(
                "SELECT count(*) AS count FROM processing_windows "
                "WHERE run_id = %s AND status = 'pending'",
                (lease.run_id,),
            ).fetchone()["count"]
            if count != row["selected_count"]:
                raise ProcessingUnavailable("incomplete window checkpoint")
            conn.execute(
                "UPDATE processing_windows SET status = 'deferred' WHERE run_id = %s",
                (lease.run_id,),
            )
            updated = conn.execute(
                """UPDATE processing_runs SET stage = 'complete', outcome = %s, reason = %s,
                   owner = NULL, lease_until = NULL, finished_at = clock_timestamp()
                   WHERE id = %s AND lease_until > clock_timestamp() RETURNING id""",
                (
                    "raw_not_selected" if row["candidate_count"] == 0 else "raw_degraded",
                    None if row["candidate_count"] == 0 else "processor_unavailable",
                    lease.run_id,
                ),
            ).fetchone()
            if updated is None:
                raise ProcessingBusy("processing lease expired during finalization")
