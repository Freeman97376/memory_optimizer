"""Durable global campaign cap and per-scope single-flight result cache."""

from dataclasses import asdict
from datetime import timedelta
from uuid import uuid4

from psycopg.types.json import Jsonb

from memory_optimizer.domain import ProcessingUnavailable, digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.llm.types import (
    CONTEXT_TOKENS,
    INPUT_NUSD,
    MODEL,
    OUTPUT_NUSD,
    CallBlocked,
    Reservation,
    Result,
)


class PostgresCallLedger:
    def __init__(self, transaction):
        self.transaction = transaction

    def reserve(self, limits, scope, key, deadline, max_output):
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO llm_campaigns
                (id,max_calls,max_nusd,model,input_nusd,output_nusd)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (
                    limits.campaign,
                    limits.max_calls,
                    limits.max_nusd,
                    MODEL,
                    INPUT_NUSD,
                    OUTPUT_NUSD,
                ),
            )
            campaign = conn.execute(
                "SELECT * FROM llm_campaigns WHERE id=%s FOR UPDATE", (limits.campaign,)
            ).fetchone()
            if (
                campaign["max_calls"],
                campaign["max_nusd"],
                campaign["model"],
                campaign["input_nusd"],
                campaign["output_nusd"],
            ) != (limits.max_calls, limits.max_nusd, MODEL, INPUT_NUSD, OUTPUT_NUSD):
                raise CallBlocked("campaign_configuration_changed")
            now = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
            if deadline <= now:
                raise CallBlocked("deadline_exhausted")
            identity = (limits.campaign, digest(scope), key)
            cached = conn.execute(
                """SELECT result,result_hash FROM llm_calls WHERE
                campaign_id=%s AND scope_hash=%s AND cache_key=%s AND state='complete'
                ORDER BY created_at DESC LIMIT 1""",
                identity,
            ).fetchone()
            if cached:
                if digest(canonical(cached["result"])) != cached["result_hash"]:
                    raise ProcessingUnavailable("provider cache integrity failure")
                return Result(**cached["result"])
            # A crashed/in-flight call stays fully charged. Expiry allows another attempt,
            # never a refund or reuse of its unknown result.
            conn.execute(
                """UPDATE llm_calls SET state='unknown' WHERE campaign_id=%s
                AND scope_hash=%s AND cache_key=%s AND state='reserved' AND expires_at<=%s""",
                (*identity, now),
            )
            pending = conn.execute(
                """SELECT id FROM llm_calls WHERE campaign_id=%s
                AND scope_hash=%s AND cache_key=%s AND state='reserved'""",
                identity,
            ).fetchone()
            if pending:
                raise CallBlocked("call_in_progress")
            total = conn.execute(
                """SELECT count(*) AS calls,COALESCE(sum(charged_nusd),0) AS cost
                FROM llm_calls WHERE campaign_id=%s""",
                (limits.campaign,),
            ).fetchone()
            reserve = CONTEXT_TOKENS * INPUT_NUSD + max_output * OUTPUT_NUSD
            if total["calls"] >= limits.max_calls or total["cost"] + reserve > limits.max_nusd:
                raise CallBlocked("campaign_budget_exhausted")
            ticket = Reservation(str(uuid4()), reserve, min(deadline, now + timedelta(seconds=90)))
            conn.execute(
                """INSERT INTO llm_calls
                (id,campaign_id,scope_hash,cache_key,state,reserved_nusd,charged_nusd,expires_at)
                VALUES (%s,%s,%s,%s,'reserved',%s,%s,%s)""",
                (ticket.id, *identity, reserve, reserve, ticket.expires_at),
            )
            return ticket

    def finish(self, reservation, result):
        data = asdict(result)
        status = "complete" if result.status == "complete" else "error"
        cost = result.known_cost
        if cost is None:
            cost = reservation.reserved_nusd
        if cost > reservation.reserved_nusd:
            raise ProcessingUnavailable("provider usage exceeded reserved cost")
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM llm_calls WHERE id=%s FOR UPDATE", (reservation.id,)
            ).fetchone()
            if row is None or row["reserved_nusd"] != reservation.reserved_nusd:
                raise ProcessingUnavailable("unknown provider reservation")
            if row["state"] != "reserved":
                raise ProcessingUnavailable("provider result already settled or expired")
            now = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
            if now >= row["expires_at"]:
                status = "error"
                data["status"], data["content"], data["error"] = "error", None, "late_reply"
            conn.execute(
                """UPDATE llm_calls SET state=%s,charged_nusd=%s,result=%s,
                result_hash=%s WHERE id=%s""",
                (status, cost, Jsonb(data), digest(canonical(data)), reservation.id),
            )

        return Result(**data)

    def summary(self, campaign):
        with self.transaction() as conn:
            row = conn.execute(
                """SELECT count(*) AS calls,
                COALESCE(sum(charged_nusd),0) AS charged_nusd,
                count(*) FILTER (WHERE state IN ('reserved','unknown')
                    OR result->>'input_tokens' IS NULL
                    OR result->>'output_tokens' IS NULL) AS uncertain_calls,
                count(*) FILTER (WHERE state='complete') AS successful_calls
                FROM llm_calls WHERE campaign_id=%s""",
                (campaign,),
            ).fetchone()

        return {key: int(value) for key, value in row.items()}
