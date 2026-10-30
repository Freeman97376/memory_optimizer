"""Pure frozen input and conservative source-supported relationship decisions."""

import json
from dataclasses import dataclass

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical, strict_json
from memory_optimizer.extraction.types import CandidateValidationError
from memory_optimizer.extraction.validation import array, choice, fields, index, text

FLOW_VERSION = "s3-offline-v1"
INPUT_VERSION = "relation-input-v1"
REPLY_VERSION = "relation-proposals-v1"
LIMITS = {
    "current": 16,
    "history": 16,
    "members": 32,
    "context_chars": 24000,
    "input_bytes": 262144,
}


class AttemptBudgetExhausted(ProcessingUnavailable):
    pass


class StaleTargets(ProcessingUnavailable):
    pass


@dataclass(frozen=True)
class RelationInput:
    encoded: str

    @property
    def value(self):
        return json.loads(self.encoded)

    @property
    def fingerprint(self):
        return digest(self.encoded)


def freeze(meta, current, history, omissions):
    """Keep complete nodes; never trim evidence into apparently complete context."""
    nodes, chars, omitted = [], 0, dict(omissions)
    for prefix, source, maximum in (("c", current, 16), ("h", history, 16)):
        for ordinal, node in enumerate(source):
            size = sum(len(s["content"]) for m in node["members"] for s in m["sources"])
            if (
                ordinal >= maximum
                or len(node["members"]) > 32
                or chars + size > 24000
                or len(canonical([*nodes, node]).encode("utf-8")) > LIMITS["input_bytes"] - 4096
            ):
                omitted[prefix] = omitted.get(prefix, 0) + 1
                continue
            nodes.append(
                dict(node, ref=f"{prefix}{len([n for n in nodes if n['ref'][0] == prefix])}")
            )
            chars += size
    return RelationInput(
        canonical(
            {
                "schema_version": INPUT_VERSION,
                "meta": meta,
                "limits": LIMITS,
                "nodes": nodes,
                "omissions": omitted,
            }
        )
    )


def problems(node):
    return [m["body"]["problem"]["text"] for m in node["members"] if m["body"]["problem"]]


def marker(kind, problem):
    if kind == "correction":
        return (f'Correction to "{problem}": ', f"更正“{problem}”：")
    if kind == "continuation":
        return (f'Continue "{problem}": ', f"继续处理“{problem}”：")
    return ()


def explicit_targets(source, nodes):
    """Safety signal even when an offline response omits a clear correction."""
    result = set()
    for member in source["members"]:
        for s in member["sources"]:
            if s["start"] != 0:
                continue
            for target in nodes:
                if target["ref"] == source["ref"]:
                    continue
                if any(
                    s["content"].startswith(marker("correction", p))
                    and sum(p in problems(n) for n in nodes) == 1
                    for p in problems(target)
                ):
                    result.add(target["ref"])
    return result


def citation(raw, node):
    raw = fields(raw, "member message_ordinal start end quote")
    member = node["members"][index(raw["member"], len(node["members"]))]
    if any(type(raw[k]) is not int for k in ("message_ordinal", "start", "end")):
        raise CandidateValidationError("relation_reference")
    quote = text(raw["quote"], 16000, allow_whitespace=True)
    for source in member["sources"]:
        if (
            source["message_ordinal"] == raw["message_ordinal"]
            and source["start"] <= raw["start"] < raw["end"] <= source["end"]
        ):
            if (
                quote
                != source["content"][raw["start"] - source["start"] : raw["end"] - source["start"]]
            ):
                raise CandidateValidationError("relation_quote_mismatch")
            return (
                dict(raw, role=source["role"], source_timestamp=source["timestamp"]),
                member,
                source,
            )
    raise CandidateValidationError("relation_reference_not_provided")


def _overlaps_claim(ref, claim):
    return claim is not None and any(
        c["span"]["message_ordinal"] == ref["message_ordinal"]
        and c["span"]["start"] <= ref["start"] < ref["end"] <= c["span"]["end"]
        for c in claim["evidence"]
    )


def _claims(body):
    yield body["problem"]
    for a in body["attempts"]:
        yield a["action"]
        yield from (o["content"] for o in a["observations"])


def _qualifiers(member):
    body = member["body"]
    return {c["text"] for c in body["conditions"]} | {
        q["text"] for o in body["objects"] for q in o["qualifiers"]
    }


def validate_reply(content, request):
    data = request.value
    if data["schema_version"] != INPUT_VERSION or data["limits"] != LIMITS:
        raise CandidateValidationError("relation_input_version")
    value = fields(
        strict_json(content, 65536), "schema_version input_hash independent unresolved relations"
    )
    if value["schema_version"] != REPLY_VERSION or value["input_hash"] != request.fingerprint:
        raise CandidateValidationError("relation_reply_version")
    nodes = {n["ref"]: n for n in data["nodes"]}
    current = {r for r in nodes if r.startswith("c")}
    independent = array(value["independent"], 16)
    unresolved = array(value["unresolved"], 16)
    for refs in (independent, unresolved):
        if any(type(r) is not str or r not in current for r in refs) or len(set(refs)) != len(refs):
            raise CandidateValidationError("relation_assessment_reference")
    relations = []
    for raw in array(value["relations"], 32):
        r = fields(raw, "kind source target source_evidence target_evidence link_evidence")
        kind = choice(r["kind"], "correction continuation object_continuity")
        if (
            type(r["source"]) is not str
            or type(r["target"]) is not str
            or r["source"] not in current
            or r["target"] not in nodes
            or r["source"] == r["target"]
        ):
            raise CandidateValidationError("relation_target")
        source, target = nodes[r["source"]], nodes[r["target"]]
        sr, sm, _ = citation(r["source_evidence"], source)
        tr, tm, _ = citation(r["target_evidence"], target)
        lr, lm, ls = citation(r["link_evidence"], source)
        if not any(_overlaps_claim(sr, c) for c in _claims(sm["body"])) or not _overlaps_claim(
            tr, tm["body"]["problem"]
        ):
            raise CandidateValidationError("relation_wrong_endpoint_evidence")
        problem = tm["body"]["problem"]["text"]
        supported = (
            kind != "object_continuity"
            and tr["quote"] == problem
            and sum(problem in problems(n) for n in nodes.values()) == 1
            and lr["member"] == sr["member"]
            and lr["message_ordinal"] == sr["message_ordinal"]
            and lr["start"] == ls["start"] == 0
            and lr["end"] == ls["end"] == ls["message_length"]
            and lr["quote"].startswith(marker(kind, problem))
            and lr["start"] <= sr["start"] < sr["end"] <= lr["end"]
            and not (_qualifiers(sm) and _qualifiers(tm) and _qualifiers(sm) != _qualifiers(tm))
        )
        relations.append(
            dict(
                r,
                source_evidence=sr,
                target_evidence=tr,
                link_evidence=lr,
                source_id=source["id"],
                target_id=target["id"],
                source_version=source["version"],
                target_version=target["version"],
                status="source_supported" if supported else "unresolved",
            )
        )
    if len({canonical(r) for r in relations}) != len(relations):
        raise CandidateValidationError("duplicate_relation")
    related = {r["source"] for r in relations}
    if (
        set(independent) & (set(unresolved) | related)
        or set(unresolved) & related
        or set(independent) | set(unresolved) | related != current
    ):
        raise CandidateValidationError("incomplete_relation_assessment")
    # Multiple targets and cycles are retained as unresolved, never silently merged.
    edges = {}
    for r in relations:
        edges.setdefault(r["source"], set()).add(r["target"])

    def reaches(start, goal, visited):
        return start == goal or (
            start not in visited
            and any(reaches(t, goal, visited | {start}) for t in edges.get(start, ()))
        )

    for r in relations:
        if len(edges[r["source"]]) > 1 or reaches(r["target"], r["source"], set()):
            r["status"] = "unresolved"
    blocked = set(unresolved) | {
        r["source"] for r in relations if r["status"] != "source_supported"
    }
    blocked.update(
        r["target"]
        for r in relations
        if r["kind"] == "correction"
        and r["status"] != "source_supported"
        and r["target"].startswith("c")
    )
    risks = {
        r["target"]
        for r in relations
        if r["status"] != "source_supported"
        and r["target"].startswith("h")
        and any(
            sum(p in problems(n) for n in nodes.values()) == 1 for p in problems(nodes[r["target"]])
        )
    }
    for ref in current:
        hints = explicit_targets(nodes[ref], list(nodes.values()))
        accepted = {
            r["target"]
            for r in relations
            if r["source"] == ref
            and r["kind"] == "correction"
            and r["status"] == "source_supported"
        }
        if hints - accepted:
            blocked.add(ref)
            blocked.update(t for t in hints - accepted if t.startswith("c"))
            risks.update(h for h in hints - accepted if h.startswith("h"))
        if (
            any(
                o["kind"] == "correction" and o["corrects_attempt"] is None
                for m in nodes[ref]["members"]
                for a in m["body"]["attempts"]
                for o in a["observations"]
            )
            and not accepted
        ):
            blocked.add(ref)
    if any(data["omissions"].values()):
        blocked.update(current)
        risks.update(r["target"] for r in relations if r["target"].startswith("h"))
    return {
        "schema_version": "relation-decisions-v1",
        "input_hash": request.fingerprint,
        "relations": relations,
        "independent": independent,
        "unresolved": unresolved,
        "blocked": sorted(blocked),
        "risk_history": sorted(risks),
    }
