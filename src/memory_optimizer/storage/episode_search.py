"""Explicit S3 index. Query-time reads never repair, infer, or fetch external evidence."""

from memory_optimizer.domain import ProcessingUnavailable, SearchCandidate, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.render import render


class EpisodeSearchIndex:
    def __init__(self, store):
        self.store = store

    def search(self, query):
        with self.store.transaction() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            rows = conn.execute(
                """SELECT v.*,ts_rank_cd(v.search_vector,q.query) AS score
                FROM episode_heads h JOIN episode_versions v ON v.id=h.version_id
                CROSS JOIN websearch_to_tsquery('simple',%s) AS q(query)
                WHERE h.user_id=%s AND v.user_id=%s AND v.search_vector @@ q.query
                ORDER BY score DESC,v.episode_id LIMIT %s""",
                (query.query, query.user_id, query.user_id, query.top_k),
            ).fetchall()
            result = []
            for row in rows:
                if (
                    digest(canonical(row["body"])) != row["body_hash"]
                    or render(row["body"]) != row["content"]
                ):
                    raise ProcessingUnavailable("changed published episode")
                content = render(row["body"], query.query)
                if content is not None:
                    result.append(SearchCandidate(str(row["episode_id"]), content, row["score"]))
            # Same MVCC snapshot for raw fallback; do not mix pre/post-correction reads.
            if len(result) < query.top_k:
                raw = conn.execute(
                    """SELECT c.id,c.content,ts_rank_cd(c.search_vector,q.query) AS score
                    FROM chunks c JOIN batches b ON b.id=c.batch_id
                    CROSS JOIN websearch_to_tsquery('simple',%s) AS q(query)
                    WHERE b.user_id=%s AND b.scope_hash=%s AND c.search_vector @@ q.query
                    ORDER BY score DESC,b.id,c.message_ordinal,c.ordinal LIMIT %s""",
                    (query.query, query.user_id, digest(query.user_id), query.top_k - len(result)),
                ).fetchall()
                result.extend(
                    SearchCandidate(
                        str(r["id"]),
                        "原文证据片段，未形成已确认结论；可能缺少后续更正：\n" + r["content"],
                        r["score"],
                    )
                    for r in raw
                )
        return result
