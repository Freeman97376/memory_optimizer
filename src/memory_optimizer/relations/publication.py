"""Pure grouping: preserve every accepted dependency or withhold the whole group."""

from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.contracts import explicit_targets
from memory_optimizer.relations.render import render


def publication_plan(request, decision=None):
    data = request.value
    nodes = {n["ref"]: n for n in data["nodes"]}
    current = {ref for ref in nodes if ref.startswith("c")}
    blocked = set(decision["blocked"]) if decision else set(current)
    risks = set(decision["risk_history"]) if decision else set()
    relations = decision["relations"] if decision else []
    supported_ids = {
        r["target_id"]
        for r in relations
        if r["status"] == "source_supported"
        and r["kind"] == "correction"
        and r["target"].startswith("h")
    }
    alert_ids = set(data["meta"].get("alert_history_ids", ())) - supported_ids
    if decision is None:
        risks.update(
            t
            for c in current
            for t in explicit_targets(nodes[c], list(nodes.values()))
            if t.startswith("h")
        )
    parents = {r: r for r in nodes}

    def root(ref):
        while parents[ref] != ref:
            ref = parents[ref]
        return ref

    for r in relations:
        if r["status"] == "source_supported" and r["kind"] != "object_continuity":
            parents[root(r["source"])] = root(r["target"])
    groups = []
    withheld = []
    for key in sorted({root(r) for r in current}):
        refs = sorted(r for r in nodes if root(r) == key)
        history = [r for r in refs if r.startswith("h")]
        members = {
            m["candidate_id"]: m
            for r in sorted(refs, key=lambda r: not r.startswith("h"))
            for m in nodes[r]["members"]
        }
        group_relations = {canonical(r): r for ref in refs for r in nodes[ref]["relations"]}
        group_relations.update(
            {
                canonical(r): r
                for r in relations
                if r["status"] == "source_supported" and r["source"] in refs and r["target"] in refs
            }
        )
        body = {
            "schema_version": "episode-publication-v1",
            "members": list(members.values()),
            "relations": list(group_relations.values()),
            "pending": any(nodes[r]["pending"] for r in refs),
        }
        content = render(body)
        if set(refs) & blocked or len(history) > 1 or len(members) > 32 or content is None:
            withheld.extend(r for r in refs if r.startswith("c"))
            risks.update(history)
        else:
            groups.append(
                {
                    "refs": refs,
                    "history": history[0] if history else None,
                    "body": body,
                    "content": content,
                }
            )
    return {
        "groups": groups,
        "withheld": sorted(withheld),
        "risk_history": sorted(risks),
        "risk_episode_ids": sorted(alert_ids),
        "omissions": data["omissions"],
    }
