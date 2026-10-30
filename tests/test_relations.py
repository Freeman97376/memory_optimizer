import json
from dataclasses import asdict

import pytest

from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.processing.coverage_types import candidate_input, final_windows
from memory_optimizer.processing.planning import build_plan, select_baseline
from memory_optimizer.relations.contracts import RelationInput, freeze, validate_reply
from memory_optimizer.relations.publication import publication_plan
from memory_optimizer.relations.render import render
from memory_optimizer.temporal.proposals import validate_temporal_reply
from tests.helpers.s3_material import FIRST, material


def pure_input():
    # Reuse the actual candidate validator with source-owned windows, not a fake episode row.
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from memory_optimizer.domain import payload_digest
    from memory_optimizer.processing.checkpoints import PlanningLease
    from memory_optimizer.processing.coverage_types import CoverageLease

    command, extraction, proposer, spec = material("two_windows")
    plan = build_plan(command.messages, spec.planning)
    saved = SimpleNamespace(
        messages=command.messages,
        plan=plan,
        selection=select_baseline(plan, command.messages),
        deadline_at=datetime(2026, 10, 27, tzinfo=UTC),
    )
    lease = CoverageLease(
        "run",
        "batch",
        command.user_id,
        "owner",
        1,
        spec,
        PlanningLease("plan", "batch", command.user_id, "owner", 1),
        30,
    )
    nodes = []
    for i, window in enumerate(final_windows(saved)):
        request = candidate_input(lease, str(i), window, saved, payload_digest(command))
        (accepted,) = validate_temporal_reply(extraction.extract(request, 1), request)
        body = asdict(accepted.candidate.candidate)
        # Canonical JSON gives the same persisted representation as JSONB.
        body = json.loads(canonical(body))
        sources = [
            dict(
                message_ordinal=s.span.message_ordinal,
                start=s.span.start,
                end=s.span.end,
                content=s.content,
                role=s.role,
                timestamp=s.source_timestamp,
                message_length=len(command.messages[s.span.message_ordinal].content),
            )
            for s in request.sources
        ]
        nodes.append(
            dict(
                id=f"candidate-{i}",
                version=f"version-{i}",
                members=[
                    dict(
                        candidate_id=f"candidate-{i}",
                        batch_id="batch",
                        body=body,
                        temporal=[json.loads(canonical(accepted.body()))],
                        sources=sources,
                    )
                ],
                relations=[],
                pending=False,
            )
        )
    request = freeze({"run_id": "run", "user_id": command.user_id}, nodes, [], {})
    return request, json.loads(proposer.propose(request, 1))


def test_two_window_correction_publishes_one_complete_group():
    request, reply = pure_input()
    decision = validate_reply(canonical(reply), request)
    assert decision["relations"][0]["status"] == "source_supported"
    plan = publication_plan(request, decision)
    assert len(plan["groups"]) == 1 and plan["withheld"] == []
    content = render(plan["groups"][0]["body"], query="single-thread")
    assert "single-thread passed" in content and "duplicates under concurrency" in content
    assert "转换: disabled" in content


@pytest.mark.parametrize(
    "change",
    [
        "unknown_target",
        "wrong_hash",
        "wrong_quote",
        "outside_span",
        "wrong_endpoint",
        "duplicate",
        "missing_assessment",
        "mixed_assessment",
        "bool_member",
        "extra_field",
    ],
)
def test_malformed_relations_rejected(change):
    request, reply = pure_input()
    r = reply["relations"][0]
    if change == "unknown_target":
        r["target"] = "h999"
    if change == "wrong_hash":
        reply["input_hash"] = "0" * 64
    if change == "wrong_quote":
        r["source_evidence"]["quote"] = "invented"
    if change == "outside_span":
        r["link_evidence"]["end"] += 100
    if change == "wrong_endpoint":
        r["target_evidence"].update(start=0, end=7, quote=FIRST[:7])
    if change == "duplicate":
        reply["relations"].append(dict(r))
    if change == "missing_assessment":
        reply["independent"] = []
    if change == "mixed_assessment":
        reply["independent"].append("c1")
    if change == "bool_member":
        r["source_evidence"]["member"] = True
    if change == "extra_field":
        r["confidence"] = 1
    with pytest.raises(ValueError):
        validate_reply(canonical(reply), request)


@pytest.mark.parametrize(
    "change",
    [
        "object_continuity",
        "multiple_matching_problems",
        "condition_conflict",
        "omission",
        "omitted_correction",
    ],
)
def test_uncertainty_never_becomes_unqualified_publication(change):
    request, reply = pure_input()
    value = request.value
    if change == "object_continuity":
        reply["relations"][0]["kind"] = "object_continuity"
    if change == "multiple_matching_problems":
        node = json.loads(canonical(value["nodes"][0]))
        node["ref"] = "h0"
        node["id"] = "historical"
        value["nodes"].append(node)
    if change == "condition_conflict":
        value["nodes"][0]["members"][0]["body"]["conditions"] = [
            dict(value["nodes"][0]["members"][0]["body"]["problem"], text="branch A")
        ]
        value["nodes"][1]["members"][0]["body"]["conditions"] = [
            dict(value["nodes"][1]["members"][0]["body"]["problem"], text="branch B")
        ]
    if change == "omission":
        value["omissions"]["history"] = 1
    if change == "omitted_correction":
        reply.update(independent=["c0", "c1"], relations=[])
    request = RelationInput(canonical(value))
    reply["input_hash"] = request.fingerprint
    decision = validate_reply(canonical(reply), request)
    assert "c1" in decision["blocked"]
    plan = publication_plan(request, decision)
    assert "c1" in plan["withheld"]


def test_safe_length_limit_drops_whole_unit():
    request, reply = pure_input()
    body = publication_plan(request, validate_reply(canonical(reply), request))["groups"][0]["body"]
    assert render(body, limit=40) is None


def test_context_budget_does_not_cut_a_source_in_half():
    request, _ = pure_input()
    node = request.value["nodes"][0]
    node["members"][0]["sources"][0]["content"] = "x" * 24001
    frozen = freeze({}, [node], [], {}).value
    assert frozen["nodes"] == [] and frozen["omissions"]["c"] == 1


def test_full_input_byte_budget_includes_candidate_and_temporal_content():
    request, _ = pure_input()
    node = request.value["nodes"][0]
    node["members"][0]["body"]["problem"]["text"] = "界" * 90000
    frozen = freeze({}, [node], [], {}).value
    assert frozen["nodes"] == [] and frozen["omissions"]["c"] == 1


@pytest.mark.parametrize("prefix", ["Quoted log: ", "If this happens: ", "Do not apply: ", "```\n"])
def test_quote_hypothesis_negation_or_fence_prefix_cannot_support_relation(prefix):
    request, reply = pure_input()
    data = request.value
    member = data["nodes"][1]["members"][0]
    source = member["sources"][0]
    source["content"] = prefix + source["content"]
    source["end"] += len(prefix)
    source["message_length"] += len(prefix)

    def shift(value):
        if isinstance(value, dict):
            if set(value) == {"message_ordinal", "start", "end"}:
                value["start"] += len(prefix)
                value["end"] += len(prefix)
            else:
                for v in value.values():
                    shift(v)
        elif isinstance(value, list):
            for v in value:
                shift(v)

    shift(member["body"])
    relation = reply["relations"][0]
    relation["source_evidence"]["start"] += len(prefix)
    relation["source_evidence"]["end"] += len(prefix)
    relation["link_evidence"]["quote"] = source["content"]
    relation["link_evidence"]["end"] = source["end"]
    request = RelationInput(canonical(data))
    reply["input_hash"] = request.fingerprint
    decision = validate_reply(canonical(reply), request)
    assert decision["relations"][0]["status"] == "unresolved"
    assert publication_plan(request, decision)["groups"] == []


def test_two_opposite_relations_do_not_create_a_cycle():
    request, reply = pure_input()
    nodes = request.value["nodes"]
    a = nodes[0]["members"][0]
    b = nodes[1]["members"][0]

    def raw(claim):
        c = claim["evidence"][0]
        return dict(c["span"], quote=c["quote"], member=0)

    reply["independent"] = []
    reply["relations"].append(
        {
            "source": "c0",
            "target": "c1",
            "kind": "continuation",
            "source_evidence": raw(a["body"]["attempts"][0]["action"]),
            "target_evidence": raw(b["body"]["problem"]),
            "link_evidence": dict(
                message_ordinal=0, start=0, end=len(FIRST), quote=FIRST, member=0
            ),
        }
    )
    decision = validate_reply(canonical(reply), request)
    assert all(r["status"] == "unresolved" for r in decision["relations"])
    assert publication_plan(request, decision)["groups"] == []


def test_pure_relationship_and_workflow_imports_do_not_load_storage_or_http():
    import subprocess
    import sys

    code = """
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith(('psycopg','fastapi','httpx','memory_optimizer.storage')):
        raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import memory_optimizer.relations.contracts
import memory_optimizer.relations.publication
import memory_optimizer.relations.workflow
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("bad", [{"schema_version": "invented"}, {"input_hash": "invented"}])
def test_replay_cannot_override_program_owned_transport_fields(bad):
    from memory_optimizer.relations.replay import ReplayRelations

    with pytest.raises(ValueError):
        ReplayRelations.from_json(canonical({"source_payload_hash": "a" * 64, "replies": [bad]}))
