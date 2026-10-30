"""Schema and exact-source checks only: paraphrase correctness remains unreviewed."""

from dataclasses import asdict

from memory_optimizer.domain import digest
from memory_optimizer.extraction.json_codec import canonical, strict_json
from memory_optimizer.extraction.types import (
    MAX_RESPONSE_BYTES,
    SCHEMA_VERSION,
    Attempt,
    CandidateValidationError,
    Citation,
    Claim,
    EpisodeCandidate,
    ExtractionInput,
    ExtractionReply,
    ObjectMention,
    Observation,
    ValidatedCandidate,
)
from memory_optimizer.processing.types import SourceSpan


def fields(value, expected):
    if type(value) is not dict or set(value) != set(expected.split()):
        raise CandidateValidationError("invalid_fields")
    return value


def array(value, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise CandidateValidationError("invalid_count")
    return value


def text(value, maximum=2048, *, allow_whitespace=False):
    if (
        type(value) is not str
        or not value
        or (not allow_whitespace and not value.strip())
        or len(value) > maximum
        or "\x00" in value
    ):
        raise CandidateValidationError("invalid_text")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise CandidateValidationError("invalid_text") from None
    return value


def choice(value, values):
    if type(value) is not str or value not in values.split():
        raise CandidateValidationError("invalid_enum")
    return value


def index(value, size):
    if type(value) is not int or not 0 <= value < size:
        raise CandidateValidationError("invalid_reference")
    return value


class EvidenceValidator:
    def __init__(self, request: ExtractionInput):
        self.request = request
        self.references = []

    def claim(self, value, *, optional=False, verbatim=False):
        if value is None and optional:
            return None
        item = fields(value, "text evidence")
        statement = text(item["text"])
        citations = []
        for raw in array(item["evidence"], 8, 1):
            ref = fields(raw, "message_ordinal start end quote")
            if any(type(ref[key]) is not int for key in ("message_ordinal", "start", "end")):
                raise CandidateValidationError("invalid_reference")
            span = SourceSpan(ref["message_ordinal"], ref["start"], ref["end"])
            quote = text(ref["quote"], 16_000, allow_whitespace=True)
            source = next(
                (
                    s
                    for s in self.request.sources
                    if (
                        s.span.message_ordinal == span.message_ordinal
                        and s.span.start <= span.start < span.end <= s.span.end
                    )
                ),
                None,
            )
            if source is None:
                raise CandidateValidationError("reference_not_provided")
            if (
                quote
                != source.content[span.start - source.span.start : span.end - source.span.start]
            ):
                raise CandidateValidationError("quote_mismatch")
            citations.append(Citation(span, quote, source.role, source.source_timestamp))
            self.references.append(span)
        citations = tuple(
            sorted(set(citations), key=lambda c: (c.span.message_ordinal, c.span.start, c.span.end))
        )
        if verbatim and not any(statement in c.quote for c in citations):
            raise CandidateValidationError("nonverbatim_value")
        roles = {c.role for c in citations}
        attribution = next(iter(roles)) + "_report" if len(roles) == 1 else "mixed_report"
        return Claim(statement, citations, attribution)

    def episode(self, value):
        item = fields(value, "problem conditions objects attempts")
        problem = self.claim(item["problem"], optional=True)
        conditions = tuple(self.claim(c) for c in array(item["conditions"], 8))
        objects = []
        for raw in array(item["objects"], 16):
            obj = fields(raw, "name kind qualifiers")
            objects.append(
                ObjectMention(
                    self.claim(obj["name"], verbatim=True),
                    choice(obj["kind"], "repository directory file symbol configuration unknown"),
                    tuple(self.claim(c) for c in array(obj["qualifiers"], 8)),
                )
            )
        attempts = []
        raw_attempts = array(item["attempts"], 8)
        if problem is None and not raw_attempts:
            raise CandidateValidationError("empty_candidate")
        for raw in raw_attempts:
            attempt = fields(raw, "action kind event_time before after object_refs observations")
            observations = []
            for raw_observation in array(attempt["observations"], 8):
                observation = fields(raw_observation, "content kind event_time corrects_attempt")
                kind = choice(observation["kind"], "result verification correction")
                target = observation["corrects_attempt"]
                if target is not None:
                    index(target, len(raw_attempts))
                    if kind != "correction":
                        raise CandidateValidationError("invalid_correction")
                observations.append(
                    Observation(
                        self.claim(observation["content"]),
                        kind,
                        self.claim(observation["event_time"], optional=True, verbatim=True),
                        target,
                    )
                )
            object_refs = tuple(index(i, len(objects)) for i in array(attempt["object_refs"], 16))
            if len(set(object_refs)) != len(object_refs):
                raise CandidateValidationError("duplicate_object_reference")
            attempts.append(
                Attempt(
                    self.claim(attempt["action"]),
                    choice(attempt["kind"], "proposed reported_change rollback unknown"),
                    self.claim(attempt["event_time"], optional=True, verbatim=True),
                    self.claim(attempt["before"], optional=True),
                    self.claim(attempt["after"], optional=True),
                    object_refs,
                    tuple(observations),
                )
            )
        return EpisodeCandidate(problem, conditions, tuple(objects), tuple(attempts))


def validate_reply(
    reply: ExtractionReply, request: ExtractionInput
) -> tuple[ValidatedCandidate, ...]:
    if type(reply.status) is not str:
        raise CandidateValidationError("invalid_status")
    if reply.status != "complete":
        code = (
            reply.status if reply.status in {"truncated", "refused", "error"} else "invalid_status"
        )
        raise CandidateValidationError(code)
    envelope = fields(strict_json(reply.content, MAX_RESPONSE_BYTES), "schema_version episodes")
    if envelope["schema_version"] != SCHEMA_VERSION or request.schema_version != SCHEMA_VERSION:
        raise CandidateValidationError("unsupported_schema")
    accepted = {}
    for value in array(envelope["episodes"], 8):
        validator = EvidenceValidator(request)
        candidate = validator.episode(value)
        focus = request.window.focus
        if not any(
            r.message_ordinal == focus.message_ordinal
            and r.start < focus.end
            and r.end > focus.start
            for r in validator.references
        ):
            raise CandidateValidationError("outside_focus")
        fingerprint = digest(
            canonical({"schema_version": SCHEMA_VERSION, "candidate": asdict(candidate)})
        )
        refs = sorted({(r.message_ordinal, r.start, r.end) for r in validator.references})
        accepted[fingerprint] = ValidatedCandidate(candidate, fingerprint, digest(canonical(refs)))
    return tuple(accepted.values())
