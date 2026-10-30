"""Wire adapters preserve the existing candidate and conservative relation validators."""

from dataclasses import asdict

from memory_optimizer.extraction.json_codec import canonical, strict_json
from memory_optimizer.extraction.types import CandidateValidationError, ExtractionReply
from memory_optimizer.extraction.validation import array, choice, fields, index, text
from memory_optimizer.llm.contracts import (
    EXTRACTION_PROMPT,
    EXTRACTION_SCHEMA,
    RELATION_PROMPT,
    RELATION_SCHEMA,
    VERSION,
    enum,
    obj,
)
from memory_optimizer.llm.types import CallBlocked
from memory_optimizer.relations.contracts import REPLY_VERSION, validate_reply
from memory_optimizer.temporal.proposals import REPLY_SCHEMA, validate_temporal_reply


def quote_span(quote, sources):
    quote = text(quote, 16000, allow_whitespace=True)
    matches = set()
    for ordinal, start, content in sources:
        position = content.find(quote)
        while position >= 0:
            matches.add((ordinal, start + position, start + position + len(quote)))
            if len(matches) > 1:
                raise CandidateValidationError("ambiguous_provider_quote")
            position = content.find(quote, position + 1)
    if len(matches) != 1:
        raise CandidateValidationError("missing_provider_quote")
    ordinal, start, end = next(iter(matches))
    return dict(message_ordinal=ordinal, start=start, end=end, quote=quote)


def evidence_units(request):
    """Deterministic exact-source pieces; no whitespace normalization or synthetic text."""
    units = []
    for source in request.sources:
        # Small provenance units bound rendering while keeping line endings byte-for-byte.
        # A long physical line may span units; a claim can cite multiple consecutive IDs.
        start = 0
        while start < len(source.content):
            end = min(start + 800, len(source.content))
            if end < len(source.content):
                boundary = source.content.rfind("\n", start + 200, end)
                if boundary >= 0:
                    end = boundary + 1
            units.append(
                dict(
                    evidence_id=len(units),
                    message_ordinal=source.span.message_ordinal,
                    start=source.span.start + start,
                    end=source.span.start + end,
                    quote=source.content[start:end],
                    role=source.role,
                    source_timestamp=source.source_timestamp,
                )
            )
            start = end
    if not units or len(units) > 128:
        raise CandidateValidationError("provider_evidence_unit_limit")
    return units


def extraction_schema(request):
    from copy import deepcopy

    schema = deepcopy(EXTRACTION_SCHEMA)
    ids = [u["evidence_id"] for u in evidence_units(request)]

    def bind(value):
        if isinstance(value, dict):
            if value.get("type") == "object" and set(value.get("properties", {})) == {
                "evidence_id"
            }:
                value["properties"]["evidence_id"] = {"type": "integer", "enum": ids}
            for child in value.values():
                bind(child)
        elif isinstance(value, list):
            for child in value:
                bind(child)

    bind(schema)
    return schema


def extraction_reply(content, request):
    envelope = fields(strict_json(content, 65536), "episodes")
    array(envelope["episodes"], 8)
    units = evidence_units(request)

    def walk(value, depth=0):
        if depth > 24:
            raise CandidateValidationError("provider_nesting")
        if type(value) is dict:
            if set(value) == {"evidence_id"}:
                unit = units[index(value["evidence_id"], len(units))]
                return {key: unit[key] for key in ("message_ordinal", "start", "end", "quote")}
            return {k: walk(v, depth + 1) for k, v in value.items()}
        if type(value) is list:
            return [walk(v, depth + 1) for v in value]
        return value

    return canonical(dict(schema_version=REPLY_SCHEMA, episodes=walk(envelope["episodes"])))


def relation_reply(content, request):
    raw = fields(strict_json(content, 65536), "assessments relations")
    nodes = {n["ref"]: n for n in request.value["nodes"]}
    current = [r for r in nodes if r.startswith("c")]
    assessments = fields(raw["assessments"], " ".join(current))
    for state in assessments.values():
        choice(state, "independent unresolved related")
    relations = []
    for item in array(raw["relations"], 32):
        item = dict(
            fields(item, "kind source target source_evidence target_evidence link_evidence")
        )
        for key, ref in [
            ("source_evidence", item["source"]),
            ("target_evidence", item["target"]),
            ("link_evidence", item["source"]),
        ]:
            if type(ref) is not str or ref not in nodes:
                raise CandidateValidationError("relation_target")
            cite = fields(item[key], "member message_ordinal quote")
            members = nodes[ref]["members"]
            member = members[index(cite["member"], len(members))]
            if type(cite["message_ordinal"]) is not int:
                raise CandidateValidationError("relation_reference")
            span = quote_span(
                cite["quote"],
                [
                    (s["message_ordinal"], s["start"], s["content"])
                    for s in member["sources"]
                    if s["message_ordinal"] == cite["message_ordinal"]
                ],
            )
            item[key] = dict(member=cite["member"], **span)
        relations.append(item)
    return canonical(
        dict(
            schema_version=REPLY_VERSION,
            input_hash=request.fingerprint,
            independent=[r for r in current if assessments[r] == "independent"],
            unresolved=[r for r in current if assessments[r] == "unresolved"],
            relations=relations,
        )
    )


class OpenAIExtractor:
    version = VERSION

    def __init__(self, model):
        self.model = model

    def extract(self, request, attempt, *, deadline):
        # Exclude run UUIDs/deadline from the prompt. Cache only the same scoped content,
        # focus, roles, source timestamps and frozen processor version.
        data = canonical(
            dict(
                version=self.version,
                focus=asdict(request.window.focus),
                evidence_units=evidence_units(request),
            )
        )

        def validate(content):
            validate_temporal_reply(
                ExtractionReply("complete", extraction_reply(content, request)), request
            )

        try:
            result = self.model.call(
                scope=request.user_id,
                phase="extract",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": data},
                ],
                schema=extraction_schema(request),
                deadline=deadline,
                validate=validate,
            )
        except CallBlocked:
            return ExtractionReply("error")
        if result.status != "complete":
            return ExtractionReply(result.status)
        return ExtractionReply("complete", extraction_reply(result.content, request))


def relation_schema(request):
    from copy import deepcopy

    refs = [n["ref"] for n in request.value["nodes"]]
    current = [r for r in refs if r.startswith("c")]
    if not current:
        return None
    schema = deepcopy(RELATION_SCHEMA)
    schema["properties"]["assessments"] = obj(
        **{ref: enum("independent", "unresolved", "related") for ref in current}
    )
    relation = schema["properties"]["relations"]["items"]["properties"]
    relation["source"] = {"type": "string", "enum": current}
    relation["target"] = {"type": "string", "enum": refs}
    return schema


class OpenAIRelations:
    version = VERSION

    def __init__(self, model):
        self.model = model

    def propose(self, request, attempt, *, deadline):
        schema = relation_schema(request)
        if schema is None:
            content = relation_reply(canonical(dict(assessments={}, relations=[])), request)
            validate_reply(content, request)
            return content

        def validate(content):
            validate_reply(relation_reply(content, request), request)

        try:
            result = self.model.call(
                scope=request.value["meta"]["user_id"],
                phase="relation",
                messages=[
                    {"role": "system", "content": RELATION_PROMPT},
                    {"role": "user", "content": request.encoded},
                ],
                schema=schema,
                deadline=deadline,
                validate=validate,
                max_output=4096,
            )
        except CallBlocked:
            return None
        return relation_reply(result.content, request) if result.status == "complete" else None
