from dataclasses import asdict
from uuid import UUID, uuid5

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingBusy, ProcessingUnavailable
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.extraction.ports import AttemptTicket
from memory_optimizer.extraction.types import MAX_ATTEMPTS, SCHEMA_VERSION
from memory_optimizer.processing.types import SourceSpan, Window
from memory_optimizer.storage.lease import lock_run, renew


class PostgresCandidateRepository:
    def __init__(self, transaction):
        self.transaction = transaction

    def report(self, user_id, batch_id):
        """Explicit local inspection only; candidates never enter Search here."""
        with self.transaction() as conn:
            run = conn.execute(
                """SELECT r.id, r.stage, r.outcome, r.reason, r.spec, r.model_calls,
                   r.candidate_count, r.selected_count FROM processing_runs r
                   JOIN batches b ON b.id = r.batch_id
                   WHERE r.batch_id = %s AND b.user_id = %s AND r.revision = 1""",
                (batch_id, user_id),
            ).fetchone()
            if run is None:
                raise ProcessingUnavailable("candidate report unavailable")
            run_id = run.pop("id")
            attempts = conn.execute(
                """SELECT w.ordinal, a.number, a.status, a.error_code, a.input_hash
                   FROM extraction_attempts a JOIN processing_windows w ON w.id = a.window_id
                   WHERE a.run_id = %s ORDER BY w.ordinal, a.number""",
                (run_id,),
            ).fetchall()
            candidates = conn.execute(
                """SELECT id::text, evidence_key, semantic_status, body FROM episode_candidates
                   WHERE run_id = %s ORDER BY id""",
                (run_id,),
            ).fetchall()
        return {"run_id": str(run_id), "run": run, "attempts": attempts, "candidates": candidates}

    def windows(self, lease):
        with self.transaction() as conn:
            run = lock_run(conn, lease, "selected")
            rows = conn.execute(
                "SELECT * FROM processing_windows WHERE run_id = %s ORDER BY ordinal",
                (lease.run_id,),
            ).fetchall()
            if len(rows) != run["selected_count"] or any(r["status"] != "pending" for r in rows):
                raise ProcessingUnavailable("incomplete selection checkpoint")
            result = tuple(
                (
                    str(r["id"]),
                    Window(
                        r["ordinal"],
                        SourceSpan(r["message_ordinal"], r["start_offset"], r["end_offset"]),
                        tuple(SourceSpan(**s) for s in r["context_spans"]),
                        tuple(r["reasons"]),
                    ),
                )
                for r in rows
            )
            renew(conn, lease)
        return result

    def reserve(self, lease, request):
        if (request.user_id, request.batch_id, request.run_id) != (
            lease.user_id,
            lease.batch_id,
            lease.run_id,
        ):
            raise ProcessingUnavailable("candidate request scope mismatch")
        with self.transaction() as conn:
            run = lock_run(conn, lease, "selected")
            if run["spec"].get("processor_version") != request.processor_version:
                raise ProcessingUnavailable("candidate processor mismatch")
            window = conn.execute(
                "SELECT id FROM processing_windows "
                "WHERE id = %s AND run_id = %s AND status = 'pending'",
                (request.window_id, lease.run_id),
            ).fetchone()
            if window is None:
                raise ProcessingUnavailable("candidate window unavailable")
            rows = conn.execute(
                "SELECT * FROM extraction_attempts WHERE window_id = %s ORDER BY number",
                (request.window_id,),
            ).fetchall()
            if any(r["input_hash"] != request.fingerprint for r in rows):
                raise ProcessingUnavailable("candidate input changed")
            if any(r["status"] in {"accepted", "empty"} for r in rows):
                renew(conn, lease)
                return None
            if any(r["status"] == "reserved" and r["fence"] == lease.fence for r in rows):
                raise ProcessingBusy("candidate attempt already reserved by this owner")
            conn.execute(
                """UPDATE extraction_attempts SET status = 'failed', error_code = 'interrupted',
                   finished_at = clock_timestamp() WHERE window_id = %s AND status = 'reserved'""",
                (request.window_id,),
            )
            ticket = None
            if len(rows) < MAX_ATTEMPTS:
                ticket = AttemptTicket(request.window_id, len(rows) + 1, request.fingerprint)
                conn.execute(
                    """INSERT INTO extraction_attempts
                       (window_id, run_id, batch_id, number, input_hash, fence, status)
                       VALUES (%s, %s, %s, %s, %s, %s, 'reserved')""",
                    (
                        request.window_id,
                        lease.run_id,
                        lease.batch_id,
                        ticket.number,
                        ticket.input_hash,
                        lease.fence,
                    ),
                )
            renew(conn, lease)
        return ticket

    def save(self, lease, ticket, candidates, error):
        if candidates and error is not None:
            raise ValueError("failed extraction cannot save candidates")
        with self.transaction() as conn:
            lock_run(conn, lease, "selected")
            row = conn.execute(
                """SELECT window_id FROM extraction_attempts WHERE window_id = %s AND number = %s
                   AND run_id = %s AND input_hash = %s AND fence = %s AND status = 'reserved'""",
                (ticket.window_id, ticket.number, lease.run_id, ticket.input_hash, lease.fence),
            ).fetchone()
            if row is None:
                raise ProcessingBusy("candidate attempt is no longer owned")
            for candidate in candidates:
                candidate_id = uuid5(UUID(lease.run_id), "candidate:" + candidate.fingerprint)
                body = asdict(candidate.candidate)
                conn.execute(
                    """INSERT INTO episode_candidates
                       (id, run_id, batch_id, schema_version, fingerprint, evidence_key, body)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (run_id, fingerprint) DO NOTHING""",
                    (
                        candidate_id,
                        lease.run_id,
                        lease.batch_id,
                        SCHEMA_VERSION,
                        candidate.fingerprint,
                        candidate.evidence_key,
                        Jsonb(body),
                    ),
                )
                stored = conn.execute(
                    "SELECT body FROM episode_candidates WHERE id = %s",
                    (candidate_id,),
                ).fetchone()
                # Jsonb converts tuples to arrays; compare canonical JSON representations.
                if stored is None or canonical(stored["body"]) != canonical(body):
                    raise ProcessingUnavailable("candidate identity collision")
                conn.execute(
                    """INSERT INTO candidate_windows(candidate_id, window_id, run_id, batch_id)
                       VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                    (candidate_id, ticket.window_id, lease.run_id, lease.batch_id),
                )
            status = "failed" if error is not None else ("accepted" if candidates else "empty")
            conn.execute(
                """UPDATE extraction_attempts SET status = %s, error_code = %s,
                   finished_at = clock_timestamp() WHERE window_id = %s AND number = %s""",
                (status, error, ticket.window_id, ticket.number),
            )
            renew(conn, lease)

    def finish(self, lease):
        with self.transaction() as conn:
            run = lock_run(conn, lease, "selected")
            rows = conn.execute(
                """SELECT w.id, count(a.number) AS attempts,
                   count(*) FILTER (WHERE a.status IN ('accepted','empty')) AS completed,
                   count(*) FILTER (WHERE a.status = 'reserved') AS reserved
                   FROM processing_windows w LEFT JOIN extraction_attempts a ON a.window_id = w.id
                   WHERE w.run_id = %s GROUP BY w.id""",
                (lease.run_id,),
            ).fetchall()
            if len(rows) != run["selected_count"] or any(
                r["reserved"] or (not r["completed"] and r["attempts"] < MAX_ATTEMPTS) for r in rows
            ):
                raise ProcessingUnavailable("candidate processing incomplete")
            failed = any(not r["completed"] for r in rows)
            conn.execute(
                "UPDATE processing_windows SET status = 'deferred' WHERE run_id = %s",
                (lease.run_id,),
            )
            updated = conn.execute(
                """UPDATE processing_runs SET stage = 'complete', outcome = %s, reason = %s,
                   owner = NULL, lease_until = NULL, finished_at = clock_timestamp()
                   WHERE id = %s AND lease_until > clock_timestamp() RETURNING id""",
                (
                    "raw_degraded" if rows else "raw_not_selected",
                    ("extraction_failed" if failed else "offline_only") if rows else None,
                    lease.run_id,
                ),
            ).fetchone()
            if updated is None:
                raise ProcessingBusy("candidate finalization lease expired")
