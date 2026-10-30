"""Prepare bounded joint evidence from validated current and published history."""

import re

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.contracts import freeze, problems
from memory_optimizer.storage.relations_report import PostgresS3ReportRepository


def build_input(owner, lease, revision, reason=None):
    reader = PostgresS3ReportRepository(owner.transaction)
    reader.FLOW_VERSION = lease.spec.flow_version
    report = reader.report(lease.user_id, lease.batch_id)
    saved = owner.load_plan(lease)
    # Safety-only pointers survive failed extraction and exhausted work time.
    # They permit a pending qualification, never a merge or a factual correction.
    alerts = []
    remaining = saved.spec.limits.max_scan_chars
    for ordinal, message in enumerate(saved.messages):
        if remaining <= 0:
            break
        prefix = message.content[: min(remaining, 2200)]
        remaining -= min(remaining, len(message.content))
        match = re.match(
            r'^(?:Correction to "([^"\n]{1,2048})": |更正“([^”\n]{1,2048})”：)', prefix
        )
        if match and len(alerts) < 16:
            alerts.append(
                {
                    "problem": match.group(1) or match.group(2),
                    "message_ordinal": ordinal,
                    "quote": match.group(0),
                }
            )
    windows = {w["id"]: w for w in report["windows"]}
    temporal = report["temporal"]["entries"]
    current = []
    ordered = sorted(
        report["candidates"],
        key=lambda c: (
            min(windows[w]["ordinal"] for w in c["window_ids"]),
            digest(canonical(c["body"])),
        ),
    )
    for candidate in ordered[:17]:
        spans = {
            tuple(sorted(s.items()))
            for wid in candidate["window_ids"]
            for s in windows[wid]["context"]
        }
        sources = []
        for encoded in sorted(spans):
            s = dict(encoded)
            message = saved.messages[s["message_ordinal"]]
            sources.append(
                dict(
                    s,
                    content=message.content[s["start"] : s["end"]],
                    role=message.role,
                    timestamp=message.timestamp,
                    message_length=len(message.content),
                )
            )
        member = {
            "candidate_id": candidate["id"],
            "batch_id": lease.batch_id,
            "body": candidate["body"],
            "sources": sources,
            "temporal": [t["body"] for t in temporal if t["candidate_id"] == candidate["id"]],
        }
        current.append(
            {
                "id": candidate["id"],
                "version": digest(canonical(candidate["body"])),
                "members": [member],
                "relations": [],
                "pending": False,
            }
        )
    # Read a bounded history page. Explicit target phrases get priority over recency.
    terms = sorted(
        {p for n in current for p in problems(n)}
        | {a["problem"] for a in alerts}
        | {
            s["content"].split('"')[1]
            for n in current
            for m in n["members"]
            for s in m["sources"]
            if s["content"].startswith(('Correction to "', 'Continue "'))
            and len(s["content"].split('"')) >= 3
        }
        | {
            s["content"].split("“")[1].split("”")[0]
            for n in current
            for m in n["members"]
            for s in m["sources"]
            if s["content"].startswith(("更正“", "继续处理“")) and "”" in s["content"]
        }
    )
    with owner.transaction() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        rows = conn.execute(
            """SELECT v.* FROM episode_heads h JOIN episode_versions v ON v.id=h.version_id
            WHERE h.user_id=%s ORDER BY EXISTS(SELECT 1 FROM unnest(%s::text[]) t
            WHERE position(t in v.content)>0) DESC,
            v.created_at DESC,v.id LIMIT 17""",
            (lease.user_id, terms),
        ).fetchall()
        count = conn.execute(
            "SELECT count(*) AS n FROM episode_heads WHERE user_id=%s", (lease.user_id,)
        ).fetchone()["n"]
        unique_alerts = conn.execute(
            """SELECT problem,min(episode_id::text) AS episode_id
            FROM (SELECT DISTINCT h.episode_id,m->'body'->'problem'->>'text' AS problem
              FROM episode_heads h JOIN episode_versions v ON v.id=h.version_id
              CROSS JOIN LATERAL jsonb_array_elements(v.body->'members') m
              WHERE h.user_id=%s AND m->'body'->'problem'->>'text'=ANY(%s::text[])) matches
            GROUP BY problem HAVING count(*)=1""",
            (lease.user_id, [a["problem"] for a in alerts]),
        ).fetchall()
    history = []
    for row in rows:
        if row["user_id"] != lease.user_id or digest(canonical(row["body"])) != row["body_hash"]:
            raise ProcessingUnavailable("invalid published history")
        history.append(dict(row["body"], id=str(row["episode_id"]), version=str(row["id"])))
    gaps = {
        "current_not_loaded": max(0, len(ordered) - 17),
        "history_not_loaded": max(0, count - 17),
        "extraction": sum(w["state"] not in {"accepted", "empty"} for w in report["windows"]),
        "selection": int(
            any(
                report["coverage"]["dispositions"][k]["codepoints"]
                for k in ("scan_budget", "pool_budget")
            )
            or bool(report["coverage"]["final_budget_excluded_focus"]["codepoints"])
        ),
        "closed_early": int(reason is not None),
    }
    meta = {
        "user_id": lease.user_id,
        "batch_id": lease.batch_id,
        "run_id": lease.run_id,
        "deadline_at": report["timing"]["deadline_at"],
        "processor_version": lease.spec.processor_version,
        "revision": revision,
        "source_hash": saved.source_hash,
        "raw_correction_alerts": alerts,
        "alert_history_ids": sorted({r["episode_id"] for r in unique_alerts}),
        "cleanup_only": reason is not None,
    }
    return freeze(meta, current, history, gaps)
