"""Read validation and explicit temporal handoff; never assign a normalized time."""

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.extraction.types import ExtractionReply
from memory_optimizer.processing.checkpoints import PlanningLease
from memory_optimizer.processing.coverage_report import coverage_report, raw_candidate
from memory_optimizer.processing.coverage_types import (
    CoverageLease,
    candidate_input,
    final_windows,
    read_coverage_spec,
)
from memory_optimizer.temporal.proposals import (
    FLOW_VERSION,
    REPLY_SCHEMA,
    TIME_SCHEMA,
    raw_proposals,
    validate_temporal_reply,
)


def temporal_report(data, rows, *, flow_version=FLOW_VERSION):
    report = coverage_report(data, flow_version=flow_version)
    try:
        saved = data.planning
        spec = read_coverage_spec(data.run["spec"])
        lease = CoverageLease(
            report["run_id"],
            saved.batch_id,
            saved.user_id,
            "report",
            0,
            spec,
            PlanningLease(saved.run_id, saved.batch_id, saved.user_id, "report", 0),
            1,
        )
        windows = final_windows(saved) if saved.selection is not None else ()
        windows = {w["id"]: windows[w["ordinal"]] for w in report["windows"] if w["id"] is not None}
        candidates = {c["id"]: c for c in report["candidates"]}
        expected = {(c["id"], wid) for c in report["candidates"] for wid in c["window_ids"]}
        if {(str(r["candidate_id"]), str(r["window_id"])) for r in rows} != expected or len(
            rows
        ) != len(expected):
            raise ValueError("missing temporal association")
        entries = []
        for row in rows:
            cid, wid = str(row["candidate_id"]), str(row["window_id"])
            request = candidate_input(lease, wid, windows[wid], saved, data.payload_hash)
            (rebuilt,) = validate_temporal_reply(
                ExtractionReply(
                    "complete",
                    canonical(
                        {
                            "schema_version": REPLY_SCHEMA,
                            "episodes": [
                                {
                                    "candidate": raw_candidate(candidates[cid]["body"]),
                                    "time_proposals": raw_proposals(row["body"]["proposals"]),
                                }
                            ],
                        }
                    ),
                ),
                request,
            )
            if (
                row["schema_version"] != TIME_SCHEMA
                or row["input_hash"] != request.fingerprint
                or canonical(rebuilt.body()) != canonical(row["body"])
                or digest(canonical(row["body"])) != row["body_hash"]
                or str(row["run_id"]) != lease.run_id
                or str(row["batch_id"]) != lease.batch_id
            ):
                raise ValueError("changed temporal state")
            entries.append(
                {
                    "candidate_id": cid,
                    "window_id": wid,
                    "input_hash": row["input_hash"],
                    "body_hash": row["body_hash"],
                    "body": row["body"],
                }
            )
        report["schema_version"] = "temporal-coverage-report-v1"
        report["temporal"] = {
            "schema_version": TIME_SCHEMA,
            "binding_rules": "disabled",
            "normalization": "disabled",
            "definite_time_filtering": False,
            "entries": entries,
        }
        return report
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        raise ProcessingUnavailable("incompatible temporal report snapshot") from None
