"""Pure evidence-preserving retrieval assembly; no answer generation."""

import re


def _claim(value):
    if value is None:
        return "unknown / 未提供"
    refs = ", ".join(
        f"m{e['span']['message_ordinal']}:{e['span']['start']}-{e['span']['end']}"
        for e in value["evidence"]
    )
    return f"{value['text']} [{value['attribution']}; {refs}]"


def render(body, query=None, limit=12000):
    tokens = re.findall(r"\w+", query.casefold()) if query else []
    linked = bool(body["relations"]) or len(body["members"]) > 1
    lines = ["经历证据 / episode evidence: 来源陈述，未经执行核验；时间未知或仅为提议。"]
    lines.append("按已选材料整理；未覆盖的原文不表示没有失败或反证。")
    if body.get("pending"):
        lines.append("待核验：存在尚未协调的新证据，不能据此认定问题已解决。")
    for member in body["members"]:
        item = member["body"]
        lines.append(f"来源 batch={member['batch_id']} candidate={member['candidate_id']}")
        lines.append("问题: " + _claim(item["problem"]))
        lines.extend("条件: " + _claim(c) for c in item["conditions"])
        for obj in item["objects"]:
            lines.append("对象提及: " + _claim(obj["name"]))
            lines.extend("限定: " + _claim(c) for c in obj["qualifiers"])
        selected = item["attempts"]
        # Local corrections refer to attempt ordinals; preserve all attempts in that case.
        has_correction = any(o["kind"] == "correction" for a in selected for o in a["observations"])
        if tokens and not linked and not has_correction:
            matches = [a for a in selected if any(t in str(a).casefold() for t in tokens)]
            selected = matches or selected
        for attempt in selected:
            lines.append(f"行动 ({attempt['kind']}): " + _claim(attempt["action"]))
            lines.append("原始事件时间: " + _claim(attempt["event_time"]))
            for key in ("before", "after"):
                if attempt[key] is not None:
                    lines.append(key + ": " + _claim(attempt[key]))
            if not attempt["observations"]:
                lines.append("结果/验证: unknown / 未提供")
            for obs in attempt["observations"]:
                lines.append(f"观察 ({obs['kind']}): " + _claim(obs["content"]))
                lines.append("观察时间: " + _claim(obs["event_time"]))
        for temporal in member["temporal"]:
            for event in temporal["events"]:
                lines.append(
                    f"时间归属: {event['binding_status']}; 转换: {event['conversion_status']}"
                )
            for proposal in temporal["proposals"]:
                lines.append("时间提议: " + _claim(proposal["time"]))
    for r in body["relations"]:
        lines.append(f"来源明确关联 ({r['kind']}): {r['link_evidence']['quote']}")
    content = "\n".join(lines)
    return content if len(content) <= limit else None
