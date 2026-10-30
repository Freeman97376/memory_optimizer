"""Read-only S3 inspection using the existing consistent snapshot reader."""

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.contracts import FLOW_VERSION, RelationInput, validate_reply
from memory_optimizer.storage.temporal_report import PostgresTemporalReportRepository
from memory_optimizer.temporal.report import temporal_report


class PostgresS3ReportRepository(PostgresTemporalReportRepository):
    FLOW_VERSION = FLOW_VERSION

    def read_extra(self, conn, run):
        return {
            "temporal": super().read_extra(conn, run),
            "snapshots": conn.execute(
                "SELECT * FROM relation_snapshots WHERE run_id=%s ORDER BY revision", (run["id"],)
            ).fetchall(),
            "attempts": conn.execute(
                "SELECT * FROM relation_attempts WHERE run_id=%s ORDER BY revision,number",
                (run["id"],),
            ).fetchall(),
            "publication": conn.execute(
                "SELECT * FROM s3_publications WHERE run_id=%s", (run["id"],)
            ).fetchone(),
        }

    def project(self, data, extra):
        report = temporal_report(data, extra["temporal"], flow_version=self.FLOW_VERSION)
        snapshots = {}
        for row in extra["snapshots"]:
            request = RelationInput(canonical(row["body"]))
            meta = request.value["meta"]
            if (
                request.fingerprint != row["input_hash"]
                or meta["run_id"] != report["run_id"]
                or meta["user_id"] != report["user_id"]
                or meta["deadline_at"] != report["timing"]["deadline_at"]
            ):
                raise ProcessingUnavailable("changed relation snapshot")
            snapshots[row["revision"]] = request
        attempts = []
        for row in extra["attempts"]:
            request = snapshots.get(row["revision"])
            if request is None or request.fingerprint != row["input_hash"]:
                raise ProcessingUnavailable("changed relation attempt input")
            if row["status"] == "accepted":
                try:
                    decision = validate_reply(row["reply"], request)
                except ValueError:
                    raise ProcessingUnavailable("invalid saved relation reply") from None
                if (
                    canonical(decision) != canonical(row["decision"])
                    or digest(canonical(decision)) != row["decision_hash"]
                ):
                    raise ProcessingUnavailable("changed relation decision")
            attempts.append(
                {
                    k: row[k]
                    for k in (
                        "revision",
                        "number",
                        "status",
                        "error_code",
                        "input_hash",
                        "decision",
                    )
                }
            )
        publication = extra["publication"]
        if publication and digest(canonical(publication["body"])) != publication["body_hash"]:
            raise ProcessingUnavailable("changed publication record")
        report["schema_version"] = "s3-report-v1"
        report["relations"] = {
            "snapshots": [
                {"revision": r, "input_hash": s.fingerprint, "input": s.value}
                for r, s in snapshots.items()
            ],
            "attempts": attempts,
        }
        report["publication"] = publication["body"] if publication else None
        if self.FLOW_VERSION == "s3-openai-v1":
            # The old runtime column is an offline-only counter. Paid usage belongs
            # to the persistent campaign ledger; never present that legacy zero.
            report["run"]["model_calls"] = None
            if report["publication"] is not None:
                report["publication"] = dict(report["publication"], model_calls=None)
            report["provider_accounting"] = {
                "source": "llm_calls",
                "scope": "campaign",
                "per_run_calls": "not_attributed",
            }
        # Candidate coverage is not a measure of publication or semantic correctness.
        report["coverage"].pop("published_evidence", None)
        return report
