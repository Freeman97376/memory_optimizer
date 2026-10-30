"""Synthetic raw messages and manually specified offline claims, not published fixtures."""

from dataclasses import replace

from memory_optimizer.domain import IngestCommand, SourceMessage, payload_digest
from memory_optimizer.extraction.json_codec import canonical
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.processing.checkpoints import RunBudget
from memory_optimizer.processing.coverage_types import CoverageSpec
from memory_optimizer.processing.plan_types import PlanLimits, PlanningSpec
from memory_optimizer.relations.contracts import FLOW_VERSION
from memory_optimizer.relations.replay import ReplayRelations, processor_version

FIRST = (
    "Problem: duplicate writes. Changed cache.py: added lock. Verification: single-thread passed."
)
CORRECTION = (
    'Correction to "duplicate writes": Problem: concurrency failure. Changed '
    "cache.py: lock still produces duplicates under concurrency."
)
CONTINUATION = (
    'Continue "duplicate writes": Problem: retry writes. Changed cache.py: retry '
    "with jitter. Verification: stress test passed."
)


def ref(message, phrase, ordinal=0):
    start = message.index(phrase)
    return {"message_ordinal": ordinal, "start": start, "end": start + len(phrase), "quote": phrase}


def claim(message, phrase, ordinal=0):
    return {"text": phrase, "evidence": [ref(message, phrase, ordinal)]}


def candidate(message, problem, action, observation=None, ordinal=0, correction=False):
    return {
        "problem": claim(message, problem, ordinal),
        "conditions": [],
        "objects": [
            {"name": claim(message, "cache.py", ordinal), "kind": "file", "qualifiers": []}
        ],
        "attempts": [
            {
                "action": claim(message, action, ordinal),
                "kind": "reported_change",
                "event_time": None,
                "before": None,
                "after": None,
                "object_refs": [0],
                "observations": []
                if observation is None
                else [
                    {
                        "content": claim(message, observation, ordinal),
                        "kind": "correction" if correction else "verification",
                        "event_time": None,
                        "corrects_attempt": None,
                    }
                ],
            }
        ],
    }


def relation(message, ordinal, source, target, kind="correction"):
    phrase = "lock still produces duplicates" if kind == "correction" else "retry with jitter"
    return {
        "kind": kind,
        "source": source,
        "target": target,
        "source_evidence": dict(ref(message, phrase, ordinal), member=0),
        "target_evidence": dict(ref(FIRST, "duplicate writes"), member=0),
        "link_evidence": dict(ref(message, message, ordinal), member=0),
    }


def material(mode="normal", *, user="s3-project", request_id=None, seconds=120, budget=12):
    if mode in {"normal", "unknown"}:
        messages = [FIRST]
        candidates = [
            candidate(
                FIRST,
                "duplicate writes",
                "added lock",
                None if mode == "unknown" else "single-thread passed",
            )
        ]
        response = {"independent": ["c0"], "unresolved": [], "relations": []}
    elif mode == "two_windows":
        messages = [FIRST, CORRECTION]
        candidates = [
            candidate(FIRST, "duplicate writes", "added lock", "single-thread passed"),
            candidate(
                CORRECTION,
                "concurrency failure",
                "lock still produces duplicates",
                "duplicates under concurrency",
                1,
                True,
            ),
        ]
        response = {
            "independent": ["c0"],
            "unresolved": [],
            "relations": [relation(CORRECTION, 1, "c1", "c0")],
        }
    else:
        continuation = mode == "continuation"
        message = CONTINUATION if continuation else CORRECTION
        messages = [message]
        candidates = [
            candidate(
                message,
                "retry writes" if continuation else "concurrency failure",
                "retry with jitter" if continuation else "lock still produces duplicates",
                "stress test passed" if continuation else "duplicates under concurrency",
                correction=not continuation,
            )
        ]
        response = {
            "independent": [],
            "unresolved": [],
            "relations": [
                relation(message, 0, "c0", "h0", "continuation" if continuation else "correction")
            ],
        }
    command = IngestCommand(
        request_id or mode, user, "offline-s3", tuple(SourceMessage("user", m) for m in messages)
    )
    extraction_value = {
        "source_payload_hash": payload_digest(command),
        "windows": [
            {
                "ordinal": i,
                "replies": [
                    {
                        "status": "complete",
                        "content": canonical(
                            {
                                "schema_version": "episode-temporal-v1",
                                "episodes": [{"candidate": c, "time_proposals": []}],
                            }
                        ),
                    }
                ]
                * 2,
            }
            for i, c in enumerate(candidates)
        ],
    }
    extraction = ReplayExtractor.from_json(canonical(extraction_value))
    relations_value = {"source_payload_hash": payload_digest(command), "replies": [response] * 2}
    proposer = ReplayRelations.from_json(canonical(relations_value))
    spec = CoverageSpec(
        PlanningSpec(PlanLimits(2000, 0, 32, 100000, 100000, 64)),
        RunBudget(seconds, budget),
        processor_version(extraction, proposer),
        1,
        flow_version=FLOW_VERSION,
    )
    return command, extraction, proposer, spec


def with_response(material_value, response):
    command, extraction, _, spec = material_value
    proposer = ReplayRelations.from_json(
        canonical({"source_payload_hash": payload_digest(command), "replies": [response] * 2})
    )
    return (
        command,
        extraction,
        proposer,
        replace(spec, processor_version=processor_version(extraction, proposer)),
    )


def drop_s3_fixture(db):
    """Only used by isolated pre-migration fixtures; never normal application cleanup."""
    db.execute(
        "DROP TABLE "
        "s3_publications,episode_heads,episode_versions,relation_attempts,relation_snapshots"
    )
    db.execute("DROP TRIGGER s3_planning_budget ON planning_attempts")
    db.execute("DROP TRIGGER s3_extraction_budget ON extraction_attempts")
    db.execute("DROP FUNCTION s3_attempt_budget(),guard_relation_attempt()")
    db.execute("ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_outcome_check")
    db.execute(
        "ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_outcome_check "
        "CHECK(outcome IN ('raw_legacy','raw_not_selected','raw_degraded'))"
    )
    db.execute("DROP TABLE llm_calls,llm_campaigns")
    db.execute("DROP FUNCTION protect_llm_call(),protect_llm_campaign()")
    db.execute("DELETE FROM schema_migrations WHERE version>=7")
