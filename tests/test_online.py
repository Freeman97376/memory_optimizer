"""Provider wire conversion against independent source and existing evidence contracts."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.extraction.types import (
    CandidateValidationError,
    EvidenceText,
    ExtractionInput,
    ExtractionReply,
)
from memory_optimizer.llm.adapters import OpenAIExtractor, extraction_reply, quote_span
from memory_optimizer.llm.contracts import EXTRACTION_SCHEMA, RELATION_SCHEMA, VERSION
from memory_optimizer.llm.types import CallBlocked, Result
from memory_optimizer.processing.types import SourceSpan, Window
from memory_optimizer.temporal.proposals import validate_temporal_reply
from tests.helpers.s3_material import FIRST, candidate


def wire_candidate():
    def wire(value):
        if isinstance(value, dict):
            if set(value) == {"message_ordinal", "start", "end", "quote"}:
                return dict(evidence_id=0)
            return {k: wire(v) for k, v in value.items()}
        if isinstance(value, list):
            return [wire(v) for v in value]
        return value

    return dict(
        episodes=[
            dict(
                candidate=wire(
                    candidate(FIRST, "duplicate writes", "added lock", "single-thread passed")
                ),
                time_proposals=[],
            )
        ]
    )


def request():
    span = SourceSpan(4, 17, 17 + len(FIRST))
    return ExtractionInput(
        "user",
        "batch",
        "run",
        "window",
        Window(0, span, (span,), ("problem",)),
        "payload",
        VERSION,
        "configuration",
        (EvidenceText(span, FIRST, "assistant", 123),),
        schema_version="episode-temporal-v1",
    )


def test_wire_quotes_resolve_to_exact_original_unicode_offsets():
    assert quote_span("修复🌱", [(7, 100, "日志:修复🌱")]) == dict(
        message_ordinal=7, start=103, end=106, quote="修复🌱"
    )
    reply = extraction_reply(canonical(wire_candidate()), request())
    accepted = validate_temporal_reply(ExtractionReply("complete", reply), request())
    claim = accepted[0].candidate.candidate.problem
    assert claim.evidence[0].span.message_ordinal == 4
    assert claim.evidence[0].span.start == 17
    assert claim.evidence[0].quote == FIRST
    assert claim.attribution == "assistant_report"
    assert accepted[0].events[0].binding_status == "unknown"


@pytest.mark.parametrize(
    "quote,sources",
    [
        ("missing", [(0, 0, "source")]),
        ("repeat", [(0, 0, "repeat repeat")]),
        ("same", [(0, 0, "same"), (1, 0, "same")]),
    ],
)
def test_ambiguous_or_absent_quotes_rejected(quote, sources):
    with pytest.raises(CandidateValidationError):
        quote_span(quote, sources)


def test_overlapping_context_duplicates_do_not_create_false_ambiguity():
    assert quote_span("unique", [(0, 0, "a unique"), (0, 2, "unique")])["start"] == 2


def test_provider_cannot_invent_offsets_or_reference_outside_sources():
    for value in [-1, 1, True]:
        raw = wire_candidate()
        raw["episodes"][0]["candidate"]["problem"]["evidence"][0]["evidence_id"] = value
        with pytest.raises(CandidateValidationError):
            extraction_reply(canonical(raw), request())
    raw = wire_candidate()
    raw["episodes"][0]["candidate"]["problem"]["evidence"][0]["evidence_id"] = 999
    with pytest.raises(CandidateValidationError):
        extraction_reply(canonical(raw), request())


def test_schema_uses_strict_required_objects_and_all_refs_resolve():
    def visit(value, root):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            if "$ref" in value:
                assert value["$ref"].removeprefix("#/$defs/") in root["$defs"]
            for v in value.values():
                visit(v, root)
        elif isinstance(value, list):
            for v in value:
                visit(v, root)

    for schema in [EXTRACTION_SCHEMA, RELATION_SCHEMA]:
        visit(schema, schema)


def test_adapter_prompt_excludes_run_ids_but_binds_source_role_and_timestamp():
    class Model:
        calls = []

        def call(self, **kwargs):
            self.calls.append(kwargs)
            content = canonical(wire_candidate())
            kwargs["validate"](content)
            return Result("complete", content)

    model = Model()
    extractor = OpenAIExtractor(model)
    deadline = datetime.now(UTC) + timedelta(seconds=30)
    extractor.extract(request(), 1, deadline=deadline)
    extractor.extract(replace(request(), run_id="new", batch_id="new"), 2, deadline=deadline)
    assert model.calls[0]["messages"] == model.calls[1]["messages"]
    payload = json.loads(model.calls[0]["messages"][1]["content"])
    assert payload["evidence_units"][0]["role"] == "assistant"
    assert payload["evidence_units"][0]["source_timestamp"] == 123
    assert model.calls[0]["deadline"] == deadline


def test_closed_gate_yields_failed_extraction_not_empty_success():
    class Blocked:
        def call(self, **kwargs):
            raise CallBlocked("exhausted")

    reply = OpenAIExtractor(Blocked()).extract(request(), 1, deadline=datetime.now(UTC))
    assert reply.status == "error"


def test_wrapped_source_quotes_preserve_newlines_without_normalization():
    sources = [(0, 0, "Fixed a bug in\n  HTTPAdapter.")]
    assert quote_span("HTTPAdapter", sources)["start"] == 17
    assert quote_span("bug in\n  HTTPAdapter", sources)["quote"] == "bug in\n  HTTPAdapter"
    with pytest.raises(CandidateValidationError, match="missing_provider_quote"):
        quote_span("bug in HTTPAdapter", sources)


def test_evidence_ids_copy_exact_wrapped_text_and_reject_fabricated_ids():
    from memory_optimizer.llm.adapters import evidence_units, extraction_schema

    source = "Problem: custom SSLContexts in sub-classes of\n  HTTPAdapter. Changed configuration."
    span = SourceSpan(3, 10, 10 + len(source))
    req = replace(
        request(),
        sources=(EvidenceText(span, source, "user", None),),
        window=Window(0, span, (span,), ("problem",)),
    )
    units = evidence_units(req)
    assert units[0]["quote"] == source and units[0]["start"] == 10
    raw = dict(
        episodes=[
            dict(
                candidate=dict(
                    problem=dict(text="custom SSLContexts", evidence=[dict(evidence_id=0)]),
                    conditions=[],
                    objects=[],
                    attempts=[],
                ),
                time_proposals=[],
            )
        ]
    )
    accepted = validate_temporal_reply(
        ExtractionReply("complete", extraction_reply(canonical(raw), req)), req
    )
    assert accepted[0].candidate.candidate.problem.evidence[0].quote == source
    assert extraction_schema(req)["$defs"]["claim"]["properties"]["evidence"]["items"][
        "properties"
    ]["evidence_id"]["enum"] == [0]


def test_evidence_units_cover_source_without_rewriting_or_cross_message_joining():
    from memory_optimizer.llm.adapters import evidence_units

    content = ("long physical line 🌱 " * 100) + "\n  indented\n"
    span = SourceSpan(4, 17, 17 + len(content))
    req = replace(request(), sources=(EvidenceText(span, content, "assistant", 123),))
    units = evidence_units(req)
    assert "".join(u["quote"] for u in units) == content
    assert all(u["end"] - u["start"] == len(u["quote"]) <= 800 for u in units)
    assert all(a["end"] == b["start"] for a, b in zip(units, units[1:], strict=False))


def test_relation_schema_cannot_classify_history_as_current():
    from types import SimpleNamespace

    from memory_optimizer.llm.adapters import relation_schema

    request = SimpleNamespace(value=dict(nodes=[dict(ref="c0"), dict(ref="h0")]))
    schema = relation_schema(request)
    states = schema["properties"]["assessments"]
    assert states["required"] == ["c0"] and set(states["properties"]) == {"c0"}
    assert states["additionalProperties"] is False
    assert states["properties"]["c0"]["enum"] == ["independent", "unresolved", "related"]
    relation = schema["properties"]["relations"]["items"]["properties"]
    assert relation["source"]["enum"] == ["c0"] and relation["target"]["enum"] == ["c0", "h0"]


def test_empty_current_needs_no_paid_relation_call():
    from memory_optimizer.llm.adapters import OpenAIRelations
    from memory_optimizer.relations.contracts import freeze, validate_reply

    class NoNetwork:
        def call(self, **kwargs):
            raise AssertionError("No current nodes require no model call")

    request = freeze(dict(user_id="user"), [], [], {})
    reply = OpenAIRelations(NoNetwork()).propose(request, 1, deadline=datetime.now(UTC))
    assert validate_reply(reply, request)["relations"] == []


@pytest.mark.parametrize(
    "assessments",
    [
        {},
        {"c0": "independent", "h0": "unresolved"},
        {"c0": ["independent", "unresolved"]},
        {"c0": "invented"},
    ],
)
def test_relation_wire_requires_one_valid_state_for_every_current_node(assessments):
    from types import SimpleNamespace

    from memory_optimizer.llm.adapters import relation_reply

    req = SimpleNamespace(value=dict(nodes=[dict(ref="c0"), dict(ref="h0")]), fingerprint="frozen")
    with pytest.raises(CandidateValidationError):
        relation_reply(canonical(dict(assessments=assessments, relations=[])), req)


def test_relation_states_convert_to_disjoint_legacy_lists_without_touching_history():
    from types import SimpleNamespace

    from memory_optimizer.llm.adapters import relation_reply

    req = SimpleNamespace(
        value=dict(nodes=[dict(ref="c0"), dict(ref="c1"), dict(ref="h0")]), fingerprint="frozen"
    )
    raw = dict(assessments={"c0": "independent", "c1": "unresolved"}, relations=[])
    result = json.loads(relation_reply(canonical(raw), req))
    assert result["independent"] == ["c0"] and result["unresolved"] == ["c1"]
    assert result["input_hash"] == "frozen"


def test_duplicate_assessment_keys_are_rejected_before_conversion():
    from types import SimpleNamespace

    from memory_optimizer.llm.adapters import relation_reply

    req = SimpleNamespace(value=dict(nodes=[dict(ref="c0")]), fingerprint="frozen")
    with pytest.raises(CandidateValidationError):
        relation_reply('{"assessments":{"c0":"independent","c0":"unresolved"},"relations":[]}', req)
