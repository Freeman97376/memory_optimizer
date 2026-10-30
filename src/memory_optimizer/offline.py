"""Explicit local replay runner: python -m memory_optimizer.offline --help."""

import argparse
import json
from pathlib import Path

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.bootstrap import build_services
from memory_optimizer.config import Settings
from memory_optimizer.domain import (
    ProcessingUnavailable,
    RequestConflict,
    StorageUnavailable,
    payload_digest,
)
from memory_optimizer.extraction.json_codec import strict_json
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.models import AddRequest
from memory_optimizer.storage.candidates import PostgresCandidateRepository


def read_bounded(path: str, limit: int) -> str:
    with Path(path).open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("offline input too large")
    return data.decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Replay local extraction responses without models")
    parser.add_argument("--request", required=True)
    parser.add_argument("--responses")
    parser.add_argument("--output", required=True, help="New local report file; never overwrite")
    parser.add_argument(
        "--flow",
        choices=["v1", "coverage-offline-v1", "coverage-temporal-v1", "s3-offline-v1"],
        default="v1",
    )
    parser.add_argument("--coverage-config", help="Explicit coverage budgets and versions JSON")
    parser.add_argument(
        "--report-only", action="store_true", help="Read existing coverage state only"
    )
    parser.add_argument("--relations", help="S3 offline relation responses")
    parser.add_argument("--query", help="S3 Search query after execution or read-only inspection")
    args = parser.parse_args()
    if args.flow == "s3-offline-v1":
        from memory_optimizer.s3_cli import run_s3

        run_s3(args, parser)
        return
    if args.relations or args.query:
        parser.error("relations/query require explicit S3 flow")
    if args.flow in {"coverage-offline-v1", "coverage-temporal-v1"}:
        from memory_optimizer.coverage_cli import run_coverage

        run_coverage(args, parser)
        return
    if not args.responses or args.coverage_config or args.report_only:
        parser.error("v1 requires --responses and does not accept coverage options")
    try:
        if Path(args.output).exists():
            raise ValueError("output already exists")
        settings = Settings.from_env()
        command = ingest_command(
            AddRequest.model_validate(
                strict_json(
                    read_bounded(args.request, settings.max_body_bytes), settings.max_body_bytes
                )
            )
        )
        replay = ReplayExtractor.from_json(read_bounded(args.responses, 1024 * 1024))
        if replay.source_payload_hash != payload_digest(command):
            raise ValueError("response fixture belongs to another request")
        services = build_services(settings, offline_extractor=replay)
        services.lifecycle.open()
        try:
            receipt = services.add.add(command)
            report = PostgresCandidateRepository(services.lifecycle.transaction).report(
                command.user_id, receipt.batch_id
            )
            report["reused"] = receipt.reused
            with Path(args.output).open("x", encoding="utf-8") as output:
                json.dump(report, output, ensure_ascii=False, indent=2)
                output.write("\n")
            print(
                json.dumps(
                    {
                        "stage": report["run"]["stage"],
                        "reason": report["run"]["reason"],
                        "candidates": len(report["candidates"]),
                        "model_calls": 0,
                    }
                )
            )
        finally:
            services.lifecycle.close()
    except (ValueError, OSError, ProcessingUnavailable, StorageUnavailable, RequestConflict):
        # Do not expose source text, invalid fields, database details or credentials.
        parser.exit(1, "Offline extraction did not complete; check input, version and storage.\n")


if __name__ == "__main__":
    main()
