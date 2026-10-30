"""One atomic visibility boundary for episodes, corrections, indexes and terminal state."""

from uuid import UUID, uuid5

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingBusy, ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.relations.publication import publication_plan
from memory_optimizer.relations.render import render


def append_version(conn, lease, episode_id, body):
    previous = conn.execute(
        (
            "SELECT v.revision FROM episode_heads h JOIN episode_versions v ON "
            "v.id=h.version_id WHERE h.episode_id=%s AND h.user_id=%s"
        ),
        (episode_id, lease.user_id),
    ).fetchone()
    revision = previous["revision"] + 1 if previous else 1
    version_id = str(uuid5(UUID(episode_id), f"revision:{revision}"))
    content = render(body)
    if content is None:
        raise ProcessingUnavailable("publication unit exceeds safe output limit")
    conn.execute(
        (
            "INSERT INTO "
            "episode_versions(id,episode_id,user_id,revision,created_by_run,body,body"
            "_hash,content) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
        ),
        (
            version_id,
            episode_id,
            lease.user_id,
            revision,
            lease.run_id,
            Jsonb(body),
            digest(canonical(body)),
            content,
        ),
    )
    conn.execute(
        (
            "INSERT INTO episode_heads(episode_id,user_id,version_id) VALUES (%s,%s,%s) ON "
            "CONFLICT(episode_id) DO UPDATE SET version_id=EXCLUDED.version_id WHERE "
            "episode_heads.user_id=EXCLUDED.user_id"
        ),
        (episode_id, lease.user_id, version_id),
    )
    return {"episode_id": episode_id, "version_id": version_id}


def finalize(owner, lease, request, decision, *, reason=None):
    if decision is not None:
        latest = owner.latest(lease)
        if (
            latest is None
            or request.fingerprint != latest[1].fingerprint
            or canonical(owner.result(lease, latest[0], request)) != canonical(decision)
        ):
            raise ProcessingUnavailable("publication requires saved matching decision")
    plan = (
        publication_plan(request, decision)
        if request
        else {"groups": [], "risk_history": [], "withheld": [], "omissions": {"not_prepared": 1}}
    )
    nodes = {n["ref"]: n for n in request.value["nodes"]} if request else {}
    # Failed publication must still qualify known historical targets, even if
    # the final deadline expired or a target advanced since the frozen snapshot.
    if reason and request:
        saved = owner.result(lease, request.value["meta"]["revision"], request)
        if saved:
            plan["risk_history"] = sorted(
                set(plan["risk_history"])
                | {r["target"] for r in saved["relations"] if r["target"].startswith("h")}
            )
    with owner.transaction() as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", ("s3:" + lease.user_id,)
        )
        planning, run, remaining = owner._guard(conn, lease, work=reason is None)
        if request and decision is not None:
            owner.check_targets(conn, request)
        published, qualified, changed = [], [], set()
        for group in plan["groups"]:
            history = group["history"]
            episode_id = (
                nodes[history]["id"]
                if history
                else str(uuid5(UUID(lease.run_id), "episode:" + nodes[group["refs"][0]]["id"]))
            )
            body = dict(
                group["body"], pending=group["body"]["pending"] or history in plan["risk_history"]
            )
            published.append(append_version(conn, lease, episode_id, body))
            changed.add(episode_id)
        risk_ids = {nodes[ref]["id"] for ref in plan["risk_history"]} | set(
            plan.get("risk_episode_ids", ())
        )
        for episode_id in sorted(risk_ids):
            if episode_id in changed:
                continue
            current = conn.execute(
                (
                    "SELECT v.* FROM episode_heads h JOIN episode_versions v ON v.id=h.version_id "
                    "WHERE h.episode_id=%s AND h.user_id=%s"
                ),
                (episode_id, lease.user_id),
            ).fetchone()
            if current is None or digest(canonical(current["body"])) != current["body_hash"]:
                raise ProcessingUnavailable("required historical qualification unavailable")
            qualified.append(
                append_version(conn, lease, episode_id, dict(current["body"], pending=True))
            )
        publication = {
            "schema_version": "s3-publication-v1",
            "input_hash": request.fingerprint if request else None,
            "published": published,
            "qualified": qualified,
            "withheld": plan["withheld"],
            "omissions": plan["omissions"],
            "reason": reason,
            "attempts_used": owner.attempts_used(conn, lease),
            "model_calls": None if lease.spec.flow_version == "s3-openai-v1" else 0,
        }
        conn.execute(
            "INSERT INTO s3_publications(run_id,body,body_hash) VALUES (%s,%s,%s)",
            (lease.run_id, Jsonb(publication), digest(canonical(publication))),
        )
        owner.planning._end_reserved(
            conn, planning, "deadline_exhausted" if remaining <= 0 else "interrupted"
        )
        conn.execute(
            (
                "UPDATE extraction_attempts SET "
                "status='failed',error_code='s3_stage_closed',finished_at=clock_timestamp() "
                "WHERE run_id=%s AND status='reserved'"
            ),
            (lease.run_id,),
        )
        conn.execute(
            (
                "UPDATE relation_attempts SET "
                "status='failed',error_code='s3_stage_closed',finished_at=clock_timestamp() "
                "WHERE run_id=%s AND status='reserved'"
            ),
            (lease.run_id,),
        )
        conn.execute(
            "UPDATE processing_windows SET status='deferred' WHERE run_id=%s", (lease.run_id,)
        )
        owner._guard(conn, lease, work=reason is None)
        row = conn.execute(
            (
                "UPDATE processing_runs SET "
                "stage='complete',outcome=%s,reason=NULL,owner=NULL,lease_until=NULL,fini"
                "shed_at=clock_timestamp() WHERE id=%s AND lease_until>clock_timestamp() "
                "RETURNING id"
            ),
            ("s3_published" if published else "s3_withheld", lease.run_id),
        ).fetchone()
        cleared = conn.execute(
            (
                "UPDATE planning_runs SET owner=NULL,lease_until=NULL WHERE id=%s AND "
                "lease_until>clock_timestamp() RETURNING id"
            ),
            (lease.planning.run_id,),
        ).fetchone()
        if row is None or cleared is None:
            raise ProcessingBusy("S3 terminal ownership expired")
    return publication
