"""Explicit online retrieval: source-ranked anchors with complete episode context.

The original offline index and stored episode rendering remain unchanged. Reads
use one snapshot, never call a provider, and never rewrite stored evidence.
"""

from memory_optimizer.domain import ProcessingUnavailable, SearchCandidate, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.render import render

SEARCH_VERSION = "source-anchored-s3-v1"
RAW_LABEL = "原文证据片段，未形成已确认结论；可能缺少后续更正："


def assemble(body, sources, query, limit=12000):
    evidence = []
    seen = set()
    for source in sources:
        identity = (
            str(source["batch_id"]),
            source["message_ordinal"],
            source["start_offset"],
            source["end_offset"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        evidence.append(
            f"对应原文 batch={identity[0]} m{identity[1]}:{identity[2]}-{identity[3]}：\n"
            + source["content"]
        )
    tail = "\n".join(evidence)
    header = RAW_LABEL if body is None else render(body, query, limit=limit - len(tail) - 1)
    if header is None:
        return None
    content = header + "\n" + tail
    return content if len(content) <= limit else None


def _associated(episode, source):
    return any(
        str(member["batch_id"]) == str(source["batch_id"])
        and any(
            s["message_ordinal"] == source["message_ordinal"]
            and s["start"] < source.get("match_end", source["end_offset"])
            and source.get("match_start", source["start_offset"]) < s["end"]
            for s in member["sources"]
        )
        for member in episode["body"]["members"]
    )


class EvidenceSearchIndex:
    def __init__(self, store):
        self.store = store

    def search(self, query):
        with self.store.transaction() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            raw = conn.execute(
                """SELECT c.id,c.batch_id,c.message_ordinal,
                    c.start_offset AS match_start,c.end_offset AS match_end,
                    CASE WHEN length(m.content)<=6000 THEN 0
                         ELSE c.start_offset END AS start_offset,
                    CASE WHEN length(m.content)<=6000 THEN length(m.content)
                         ELSE c.end_offset END AS end_offset,
                    CASE WHEN length(m.content)<=6000 THEN m.content ELSE c.content END AS content,
                    ts_rank_cd(c.search_vector,q.query) AS score
                FROM chunks c JOIN batches b ON b.id=c.batch_id
                JOIN messages m ON m.batch_id=c.batch_id AND m.ordinal=c.message_ordinal
                CROSS JOIN websearch_to_tsquery('simple',%s) AS q(query)
                WHERE b.scope_hash=%s AND b.user_id=%s AND
                    (c.search_vector @@ q.query OR strpos(lower(c.content),lower(%s))>0)
                ORDER BY score DESC,b.created_at DESC,b.id,c.message_ordinal,c.ordinal LIMIT %s""",
                (query.query, digest(query.user_id), query.user_id, query.query, query.top_k),
            ).fetchall()
            batches = list({str(r["batch_id"]) for r in raw})
            associated = (
                conn.execute(
                    """SELECT v.*,ts_rank_cd(v.search_vector,q.query) AS score
                FROM episode_heads h JOIN episode_versions v ON v.id=h.version_id
                CROSS JOIN websearch_to_tsquery('simple',%s) AS q(query)
                WHERE h.user_id=%s AND v.user_id=%s AND EXISTS
                    (SELECT 1 FROM jsonb_array_elements(v.body->'members') m
                     WHERE m->>'batch_id'=ANY(%s))""",
                    (query.query, query.user_id, query.user_id, batches),
                ).fetchall()
                if batches
                else []
            )
            matches = conn.execute(
                """SELECT v.*,ts_rank_cd(v.search_vector,q.query) AS score
                FROM episode_heads h JOIN episode_versions v ON v.id=h.version_id
                CROSS JOIN websearch_to_tsquery('simple',%s) AS q(query)
                WHERE h.user_id=%s AND v.user_id=%s AND v.search_vector @@ q.query
                ORDER BY score DESC,v.episode_id LIMIT %s""",
                (query.query, query.user_id, query.user_id, query.top_k),
            ).fetchall()
            episodes = {str(row["episode_id"]): row for row in [*associated, *matches]}
            ordered = sorted(episodes.values(), key=lambda r: (-r["score"], str(r["episode_id"])))
            for row in ordered:
                if (
                    digest(canonical(row["body"])) != row["body_hash"]
                    or render(row["body"]) != row["content"]
                ):
                    raise ProcessingUnavailable("changed published episode")
            groups = {}
            for source in raw:
                episode = next((r for r in ordered if _associated(r, source)), None)
                identity = str(episode["episode_id"] if episode else source["id"])
                if identity not in groups:
                    groups[identity] = dict(
                        body=episode["body"] if episode else None, sources=[], score=source["score"]
                    )
                groups[identity]["sources"].append(source)
            result = []
            for identity, group in groups.items():
                content = assemble(group["body"], group["sources"], query.query)
                if content is not None:
                    result.append(SearchCandidate(identity, content, group["score"]))
            # Derived-only matches may fill remaining slots; they cannot crowd out
            # source anchors. Include their frozen source context, never a bare summary.
            for row in matches:
                identity = str(row["episode_id"])
                if identity in groups or len(result) >= query.top_k:
                    continue
                sources = [
                    dict(
                        batch_id=m["batch_id"],
                        message_ordinal=s["message_ordinal"],
                        start_offset=s["start"],
                        end_offset=s["end"],
                        content=s["content"],
                    )
                    for m in row["body"]["members"]
                    for s in m["sources"]
                ]
                content = assemble(row["body"], sources, query.query)
                if content is not None:
                    result.append(SearchCandidate(identity, content, row["score"]))
            return result
