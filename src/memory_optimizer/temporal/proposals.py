"""Pure temporal validation. Literal provenance never establishes event attribution."""

from dataclasses import asdict, dataclass, replace

from memory_optimizer.extraction.json_codec import canonical, strict_json
from memory_optimizer.extraction.types import (
    MAX_RESPONSE_BYTES,
    SCHEMA_VERSION,
    CandidateValidationError,
    Claim,
    ExtractionReply,
    ValidatedCandidate,
)
from memory_optimizer.extraction.validation import (
    EvidenceValidator,
    array,
    choice,
    fields,
    index,
    validate_reply,
)

FLOW_VERSION = "coverage-temporal-v1"
REPLY_SCHEMA = "episode-temporal-v1"
TIME_SCHEMA = "time-proposals-v1"


@dataclass(frozen=True)
class EventTarget:
    attempt: int
    observation: int | None


@dataclass(frozen=True)
class TimeProposal:
    target: EventTarget
    time: Claim
    event_evidence: tuple
    relation_evidence: tuple
    context: str
    subject: Claim | None
    branch: Claim | None
    version: Claim | None
    anchor: Claim | None
    timezone: Claim | None


@dataclass(frozen=True)
class EventTime:
    target: EventTarget
    event_kind: str
    raw_time: Claim | None
    proposal_ordinals: tuple[int, ...]
    provenance_status: str
    binding_status: str
    binding_reason: str
    conversion_status: str = "disabled"
    normalized_value: None = None
    sort_key: None = None


@dataclass(frozen=True)
class TemporalCandidate:
    candidate: ValidatedCandidate
    proposals: tuple[TimeProposal, ...]
    events: tuple[EventTime, ...]

    def body(self):
        return {
            "schema_version": TIME_SCHEMA,
            "proposals": [asdict(p) for p in self.proposals],
            "events": [asdict(e) for e in self.events],
            "binding_rules": "disabled",
            "normalization": "disabled",
        }


def _references(raw, validator):
    # Existing evidence validation supplies the outer role and exact source timestamp.
    return validator.claim({"text": "evidence reference", "evidence": raw}).evidence


def _contains(container, citation):
    a, b = container.span, citation.span
    return a.message_ordinal == b.message_ordinal and a.start <= b.start < b.end <= a.end


def _event(candidate, target):
    attempt = candidate.attempts[index(target.attempt, len(candidate.attempts))]
    if target.observation is None:
        return attempt.action, attempt.event_time, attempt.kind
    observation = attempt.observations[index(target.observation, len(attempt.observations))]
    return observation.content, observation.event_time, observation.kind


def assess(candidate, raw_proposals, request):
    """Return all event slots, including those without a time or without proposals."""
    proposals = []
    for raw in array(raw_proposals, 32):
        item = fields(
            raw,
            "target time event_evidence relation_evidence context "
            "subject branch version anchor timezone",
        )
        target = EventTarget(**fields(item["target"], "attempt observation"))
        event, _, _ = _event(candidate.candidate, target)
        validator = EvidenceValidator(request)
        event_refs = _references(item["event_evidence"], validator)
        if any(
            not any(_contains(ref, citation) for ref in event.evidence) for citation in event_refs
        ):
            raise CandidateValidationError("time_wrong_event_evidence")
        time = validator.claim(item["time"], verbatim=True)
        relation_raw = array(item["relation_evidence"], 8)
        relation = _references(relation_raw, validator) if relation_raw else ()
        if relation and any(
            not any(_contains(r, c) for r in relation) for c in (*time.evidence, *event_refs)
        ):
            raise CandidateValidationError("time_incomplete_relation_evidence")
        proposals.append(
            TimeProposal(
                target,
                time,
                event_refs,
                relation,
                choice(item["context"], "occurred planned hypothetical quoted unknown"),
                *(
                    validator.claim(item[key], optional=True, verbatim=True)
                    for key in ("subject", "branch", "version", "anchor", "timezone")
                ),
            )
        )
    if len({canonical(asdict(p)) for p in proposals}) != len(proposals):
        raise CandidateValidationError("duplicate_time_proposal")
    targets = [
        EventTarget(i, j)
        for i, a in enumerate(candidate.candidate.attempts)
        for j in (None, *range(len(a.observations)))
    ]
    events = []
    for target in targets:
        _, raw_time, kind = _event(candidate.candidate, target)
        ordinals = tuple(i for i, p in enumerate(proposals) if p.target == target)
        selected = [proposals[i] for i in ordinals]
        # Conservative disagreement in raw mentions/context is not resolved by parsing dates.
        phrases = {p.time.text for p in selected}
        if raw_time is not None:
            phrases.add(raw_time.text)
        contexts = {p.context for p in selected if p.context != "unknown"}
        qualifier_conflict = any(
            len({getattr(p, key).text for p in selected if getattr(p, key) is not None}) > 1
            for key in ("subject", "branch", "version", "anchor", "timezone")
        )
        if (
            len(phrases) > 1
            or len(contexts) > 1
            or qualifier_conflict
            or (kind == "proposed" and "occurred" in contexts)
        ):
            status, reason = "conflict", "inconsistent_raw_proposals"
        elif not selected:
            status, reason = "unknown", "raw_mention_only" if raw_time else "not_provided"
        elif any(not p.relation_evidence or p.context == "unknown" for p in selected):
            status, reason = "ambiguous", "insufficient_relation_or_context"
        else:
            status, reason = "proposed", "unverified_relation"
        events.append(
            EventTime(
                target,
                kind,
                raw_time,
                ordinals,
                "source_checked" if raw_time or selected else "absent",
                status,
                reason,
            )
        )
    return TemporalCandidate(candidate, tuple(proposals), tuple(events))


def validate_temporal_reply(reply, request):
    if request.schema_version != REPLY_SCHEMA:
        raise CandidateValidationError("unsupported_temporal_input")
    if reply.status != "complete":
        # Reuse v1 refusal/error/truncation taxonomy, without accepting its envelope.
        return validate_reply(reply, request)
    envelope = fields(strict_json(reply.content, MAX_RESPONSE_BYTES), "schema_version episodes")
    if envelope["schema_version"] != REPLY_SCHEMA:
        raise CandidateValidationError("unsupported_temporal_schema")
    result = []
    for value in array(envelope["episodes"], 8):
        item = fields(value, "candidate time_proposals")
        (base,) = validate_reply(
            ExtractionReply(
                "complete",
                canonical(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "episodes": [item["candidate"]],
                    }
                ),
            ),
            replace(request, schema_version=SCHEMA_VERSION),
        )
        result.append(assess(base, item["time_proposals"], request))
    if len({v.candidate.fingerprint for v in result}) != len(result):
        raise CandidateValidationError("duplicate_temporal_candidate")
    return tuple(result)


def raw_proposals(stored):
    """Stored normalized claims back to the closed response shape for read validation."""

    def claim(value):
        if value is None:
            return None
        return {"text": value["text"], "evidence": refs(value["evidence"])}

    def refs(values):
        return [dict(c["span"], quote=c["quote"]) for c in values]

    return [
        dict(
            target=p["target"],
            time=claim(p["time"]),
            event_evidence=refs(p["event_evidence"]),
            relation_evidence=refs(p["relation_evidence"]),
            context=p["context"],
            **{k: claim(p[k]) for k in ("subject", "branch", "version", "anchor", "timezone")},
        )
        for p in stored
    ]
