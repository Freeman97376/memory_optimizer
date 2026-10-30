"""Read one MVCC snapshot; reporting never claims, renews, repairs or executes work."""

from memory_optimizer.domain import ProcessingUnavailable, SourceMessage, payload_digest
from memory_optimizer.processing.coverage_report import CoverageReportInput, coverage_report
from memory_optimizer.storage.planning import decode_snapshot


class PostgresCoverageReportRepository:
    FLOW_VERSION = "coverage-offline-v1"

    def __init__(self, transaction):
        self.transaction = transaction

    def report_request(self, command):
        with self.transaction() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            row = conn.execute(
                "SELECT id FROM batches WHERE user_id=%s AND request_id=%s AND payload_hash=%s",
                (command.user_id, command.request_id, payload_digest(command)),
            ).fetchone()
        if row is None:
            raise ProcessingUnavailable("coverage request unavailable")
        report = self.report(command.user_id, str(row["id"]))
        if report["source"]["payload_hash"] != payload_digest(command):
            raise ProcessingUnavailable("coverage request changed")
        return report

    def report(self, user_id, batch_id):
        with self.transaction() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            batch = conn.execute(
                "SELECT payload_hash FROM batches WHERE id=%s AND user_id=%s",
                (batch_id, user_id),
            ).fetchone()
            if batch is None:
                raise ProcessingUnavailable("coverage report unavailable")
            run = conn.execute(
                "SELECT * FROM processing_runs WHERE batch_id=%s AND revision=1", (batch_id,)
            ).fetchone()
            plan = conn.execute(
                "SELECT * FROM planning_runs WHERE batch_id=%s AND revision=1", (batch_id,)
            ).fetchone()
            if run is None or plan is None or run["spec"].get("flow_version") != self.FLOW_VERSION:
                raise ProcessingUnavailable("coverage report requires an existing coverage run")
            messages = conn.execute(
                "SELECT ordinal,role,content,source_timestamp FROM messages "
                "WHERE batch_id=%s ORDER BY ordinal",
                (batch_id,),
            ).fetchall()
            if not messages or [m["ordinal"] for m in messages] != list(range(len(messages))):
                raise ProcessingUnavailable("coverage sources incomplete")
            planning_attempts = conn.execute(
                "SELECT * FROM planning_attempts WHERE run_id=%s ORDER BY number", (plan["id"],)
            ).fetchall()
            binding = conn.execute(
                "SELECT * FROM coverage_bindings WHERE run_id=%s", (run["id"],)
            ).fetchone()
            rows = {}
            for name, order in (
                ("processing_windows", "ordinal"),
                ("extraction_attempts", "window_id,number"),
                ("episode_candidates", "id"),
                ("candidate_windows", "candidate_id,window_id"),
            ):
                rows[name] = tuple(
                    conn.execute(
                        f"SELECT * FROM {name} WHERE run_id=%s ORDER BY {order}", (run["id"],)
                    ).fetchall()
                )
            extra = self.read_extra(conn, run)
        # Pure validation/union arithmetic occurs after closing the database snapshot.
        snapshot = decode_snapshot(
            plan,
            tuple(SourceMessage(m["role"], m["content"], m["source_timestamp"]) for m in messages),
            planning_attempts,
            user_id,
            batch_id,
        )
        return self.project(
            CoverageReportInput(
                snapshot,
                run,
                binding,
                batch["payload_hash"],
                rows["processing_windows"],
                rows["extraction_attempts"],
                rows["episode_candidates"],
                rows["candidate_windows"],
            ),
            extra,
        )

    def read_extra(self, conn, run):
        return None

    def project(self, data, extra):
        return coverage_report(data)
