"""Versioned processing coverage, not semantic recall. No I/O or execution clock."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, timedelta
from uuid import UUID, uuid5

from memory_optimizer.domain import ProcessingUnavailable
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.extraction.types import MAX_ATTEMPTS, SCHEMA_VERSION, ExtractionReply
from memory_optimizer.extraction.validation import validate_reply
from memory_optimizer.processing.checkpoints import PlanningLease, PlanningSnapshot, merge_spans
from memory_optimizer.processing.coverage_types import (
    CoverageLease,
    candidate_input,
    final_fingerprint,
    final_windows,
    read_coverage_spec,
)
from memory_optimizer.processing.types import SourceSpan


@dataclass(frozen=True)
class CoverageReportInput:
    """Detached storage snapshot. Row mappings are read-only inputs to this pure projection."""

    planning: PlanningSnapshot
    run: dict
    binding: dict | None
    payload_hash: str
    windows: tuple[dict, ...]
    attempts: tuple[dict, ...]
    candidates: tuple[dict, ...]
    links: tuple[dict, ...]


def measure(spans, total, *, available=True):
    merged = merge_spans(tuple(spans)) if available else ()
    count = sum(s.end - s.start for s in merged) if available else None
    if count is not None and not 0 <= count <= total:
        raise ValueError("invalid coverage numerator")
    return {
        "codepoints": count,
        "total_codepoints": total,
        "ratio": count / total if available and total else None,
        "status": "known"
        if available and total
        else ("not_applicable" if available else "unknown"),
        "ranges": [asdict(s) for s in merged],
    }


def raw_candidate(value):
    # Reconstitute the original candidate schema, then re-use its evidence validator.
    if isinstance(value, list):
        return [raw_candidate(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"text", "evidence", "attribution"}:
            return {
                "text": value["text"],
                "evidence": [dict(c["span"], quote=c["quote"]) for c in value["evidence"]],
            }
        return {k: raw_candidate(v) for k, v in value.items()}
    return value


def _citations(value):
    if isinstance(value, dict):
        if set(value) == {"text", "evidence", "attribution"}:
            yield from (SourceSpan(**c["span"]) for c in value["evidence"])
        else:
            for item in value.values():
                yield from _citations(item)
    elif isinstance(value, list):
        for item in value:
            yield from _citations(item)


def _stamp(value):
    return value.astimezone(UTC).isoformat() if value is not None else None


def coverage_report(data: CoverageReportInput, *, flow_version="coverage-offline-v1"):
    try:
        if data.run["spec"]["flow_version"] != flow_version:
            raise ValueError("report flow mismatch")
        return _report(data)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        raise ProcessingUnavailable("incompatible coverage report snapshot") from None


def _report(data):
    saved, run = data.planning, data.run
    spec = read_coverage_spec(run["spec"])
    run_id = str(run["id"])
    if (
        str(run["batch_id"]) != saved.batch_id
        or spec.planning != saved.spec
        or spec.budget != saved.budget
        or run["created_at"] != saved.created_at
        or saved.deadline_at != saved.created_at + timedelta(seconds=spec.budget.total_seconds)
        or run["model_calls"] != 0
    ):
        raise ValueError("report runtime mismatch")
    lease = CoverageLease(
        run_id,
        saved.batch_id,
        saved.user_id,
        "report",
        0,
        spec,
        PlanningLease(saved.run_id, saved.batch_id, saved.user_id, "report", 0),
        1,
    )
    windows = final_windows(saved) if saved.selection is not None else ()
    bound = data.binding is not None
    if bound:
        b = data.binding
        if (
            saved.selection is None
            or str(b["run_id"]) != run_id
            or str(b["batch_id"]) != saved.batch_id
            or str(b["planning_id"]) != saved.run_id
            or b["adapter_version"] != spec.adapter_version
            or b["selection_hash"] != final_fingerprint(saved)
            or run["candidate_count"] != len(saved.plan.pool)
            or run["selected_count"] != len(windows)
            or len(data.windows) != len(windows)
        ):
            raise ValueError("report binding mismatch")
    elif (
        data.windows
        or data.attempts
        or data.candidates
        or data.links
        or run["selected_at"] is not None
    ):
        raise ValueError("unbound extraction data")

    requests, accepted_ids, output_windows, ranges = _extraction(data, lease, windows, bound)
    returned, uncertain, accepted, empty = ranges
    evidence, candidates = _candidates(data, requests, accepted_ids, run_id)

    total = sum(len(m.content) for m in saved.messages)
    plan = saved.plan
    dispositions = {
        name: measure([c.span for c in plan.coverage if c.disposition == name], total)
        if plan is not None
        else measure((), total, available=False)
        for name in ("candidate", "no_signal", "pool_budget", "scan_budget")
    }
    selected = {i for i in saved.selection.pool_ordinals} if saved.selection else set()
    excluded = [w.focus for i, w in enumerate(plan.pool) if i not in selected] if plan else []
    return {
        "schema_version": "coverage-report-v1",
        "user_id": saved.user_id,
        "batch_id": saved.batch_id,
        "run_id": run_id,
        "planning_id": saved.run_id,
        "source": {
            "digest": saved.source_hash,
            "payload_hash": data.payload_hash,
            "message_codepoints": [len(m.content) for m in saved.messages],
            "measurement_version": saved.spec.measurement_version,
        },
        "run": {k: run[k] for k in ("stage", "outcome", "reason", "model_calls")},
        "spec": asdict(spec),
        "timing": {
            "created_at": _stamp(saved.created_at),
            "deadline_at": _stamp(saved.deadline_at),
            "work_deadline_at": _stamp(saved.deadline_at - timedelta(seconds=spec.cleanup_seconds)),
            "finished_at": _stamp(run["finished_at"]),
            "elapsed_seconds": (run["finished_at"] - saved.created_at).total_seconds()
            if run["finished_at"]
            else None,
            "hard_latency_guarantee": False,
        },
        "planning": {
            "stage": saved.stage,
            "closed_reason": saved.closed_reason,
            "plan_fingerprint": plan.fingerprint if plan else None,
            "selection_fingerprint": final_fingerprint(saved) if saved.selection else None,
            "pool_windows": len(plan.pool) if plan else None,
            "selected_windows": len(windows) if saved.selection else None,
            "binding_saved": bound,
            "screening": "disabled",
            "attempts_used": len(saved.attempts),
            "attempts_limit": saved.budget.max_attempts,
            "attempts": [
                {
                    "number": a.ticket.number,
                    "phase": a.ticket.phase,
                    "diagnostic": asdict(a.diagnostic) if a.diagnostic else None,
                }
                for a in saved.attempts
            ],
        },
        "coverage": {
            "accounted_focus": measure([c.span for c in plan.coverage] if plan else (), total),
            "dispositions": dispositions,
            "selected_focus": measure(
                [w.focus for w in windows], total, available=saved.selection is not None
            ),
            "selected_context": measure(
                [s for w in windows for s in w.context],
                total,
                available=saved.selection is not None,
            ),
            "final_budget_excluded_focus": measure(
                excluded, total, available=saved.selection is not None
            ),
            "recorded_reply_context": measure(returned, total),
            "uncertain_attempt_context": measure(uncertain, total),
            "accepted_context": measure(accepted, total),
            "valid_empty_context": measure(empty, total),
            "not_started_context": measure(
                [
                    SourceSpan(**s)
                    for w in output_windows
                    if w["state"] == "not_started"
                    for s in w["context"]
                ],
                total,
                available=saved.selection is not None,
            ),
            "candidate_evidence": measure(evidence, total),
            "published_evidence": measure((), total),
        },
        "windows": output_windows,
        "candidates": candidates,
        "limitations": {
            "model_seen_context": "not_applicable_offline_replay",
            "semantic_recall": "unavailable_without_independent_labels",
            "publication": "not_implemented",
            "episode_search": "not_implemented",
            "interrupted_attempt_read": "unknown",
            "coverage_is_semantic_completeness": False,
        },
    }


def _extraction(data, lease, windows, bound):
    saved, run = data.planning, data.run
    run_id = str(run["id"])
    by_window = {}
    for row, window in zip(data.windows, windows if bound else (), strict=True):
        wid = str(row["id"])
        if (
            wid != str(uuid5(UUID(run_id), f"window:{window.ordinal}"))
            or (row["ordinal"], row["message_ordinal"], row["start_offset"], row["end_offset"])
            != (window.ordinal, window.focus.message_ordinal, window.focus.start, window.focus.end)
            or row["context_spans"] != [asdict(s) for s in window.context]
            or row["reasons"] != list(window.reasons)
            or row["status"] != ("deferred" if run["stage"] == "complete" else "pending")
        ):
            raise ValueError("report window mismatch")
        by_window[wid] = window

    returned, uncertain, accepted, empty, output_windows = [], [], [], [], []
    requests, accepted_ids = {}, set()
    for wid, window in by_window.items():
        request = candidate_input(lease, wid, window, saved, data.payload_hash)
        requests[wid] = request
        attempts = [a for a in data.attempts if str(a["window_id"]) == wid]
        if [a["number"] for a in attempts] != list(range(1, len(attempts) + 1)) or len(
            attempts
        ) > MAX_ATTEMPTS:
            raise ValueError("invalid extraction attempt sequence")
        successes = []
        for index, attempt in enumerate(attempts):
            status, error = attempt["status"], attempt["error_code"]
            if (
                attempt["input_hash"] != request.fingerprint
                or status not in {"reserved", "accepted", "empty", "failed"}
                or (status == "failed") != (error is not None)
                or (status == "reserved") != (attempt["finished_at"] is None)
                or ((status in {"reserved", "accepted", "empty"}) and index != len(attempts) - 1)
                or (run["stage"] == "complete" and status == "reserved")
            ):
                raise ValueError("invalid extraction attempt")
            if status == "reserved" or error in {"interrupted", "time_budget_exhausted"}:
                uncertain.extend(window.context)
            else:
                returned.extend(window.context)
            if status in {"accepted", "empty"}:
                successes.append(status)
        state = (
            successes[0]
            if successes
            else (
                "not_started"
                if not attempts
                else (
                    "pending_unknown"
                    if attempts[-1]["status"] == "reserved"
                    else ("attempts_exhausted" if len(attempts) == MAX_ATTEMPTS else "incomplete")
                )
            )
        )
        if state == "accepted":
            accepted.extend(window.context)
            accepted_ids.add(wid)
        elif state == "empty":
            empty.extend(window.context)
        output_windows.append(
            {
                "id": wid,
                "ordinal": window.ordinal,
                "focus": asdict(window.focus),
                "context": [asdict(s) for s in window.context],
                "state": state,
                "attempts": [
                    {k: a[k] for k in ("number", "status", "error_code", "input_hash")}
                    for a in attempts
                ],
            }
        )
    if any(str(a["window_id"]) not in by_window for a in data.attempts):
        raise ValueError("unknown extraction window")
    if not bound:
        output_windows = [
            {
                "id": None,
                "ordinal": w.ordinal,
                "focus": asdict(w.focus),
                "context": [asdict(s) for s in w.context],
                "state": "not_started",
                "attempts": [],
            }
            for w in windows
        ]

    return requests, accepted_ids, output_windows, (returned, uncertain, accepted, empty)


def _candidates(data, requests, accepted_ids, run_id):
    evidence, candidates, linked = [], [], set()
    candidate_ids = {str(c["id"]) for c in data.candidates}
    if len(candidate_ids) != len(data.candidates):
        raise ValueError("duplicate candidate")
    if any(
        str(link["candidate_id"]) not in candidate_ids or str(link["window_id"]) not in accepted_ids
        for link in data.links
    ):
        raise ValueError("invalid candidate association")
    for candidate in data.candidates:
        cid = str(candidate["id"])
        ids = sorted(
            str(link["window_id"]) for link in data.links if str(link["candidate_id"]) == cid
        )
        if not ids or len(set(ids)) != len(ids) or candidate["semantic_status"] != "unreviewed":
            raise ValueError("unassociated or published candidate")
        reply = ExtractionReply(
            "complete",
            canonical(
                {"schema_version": SCHEMA_VERSION, "episodes": [raw_candidate(candidate["body"])]}
            ),
        )
        for wid in ids:
            (validated,) = validate_reply(
                reply, replace(requests[wid], schema_version=SCHEMA_VERSION)
            )
            if (
                canonical(asdict(validated.candidate)) != canonical(candidate["body"])
                or validated.fingerprint != candidate["fingerprint"]
                or validated.evidence_key != candidate["evidence_key"]
                or candidate["schema_version"] != SCHEMA_VERSION
                or cid != str(uuid5(UUID(run_id), "candidate:" + validated.fingerprint))
            ):
                raise ValueError("changed candidate evidence")
        linked.update(ids)
        evidence.extend(_citations(candidate["body"]))
        candidates.append(
            {
                "id": cid,
                "schema_version": SCHEMA_VERSION,
                "semantic_status": "unreviewed",
                "window_ids": ids,
                "evidence_key": candidate["evidence_key"],
                "body": candidate["body"],
            }
        )
    if linked != accepted_ids:
        raise ValueError("accepted window missing candidates")

    return evidence, candidates
