"""Explicit local S3 execution, read-only inspection and published Search."""

import json
from dataclasses import asdict
from pathlib import Path

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.bootstrap import build_s3_services
from memory_optimizer.config import Settings
from memory_optimizer.coverage_cli import load_spec, write_report
from memory_optimizer.domain import (
    ProcessingUnavailable,
    RequestConflict,
    SearchQuery,
    StorageUnavailable,
    payload_digest,
)
from memory_optimizer.extraction.json_codec import strict_json
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.models import AddRequest
from memory_optimizer.relations.contracts import FLOW_VERSION
from memory_optimizer.relations.replay import ReplayRelations, processor_version
from memory_optimizer.storage.episode_search import EpisodeSearchIndex
from memory_optimizer.storage.postgres import PostgresStore
from memory_optimizer.storage.relations_report import PostgresS3ReportRepository


def run_s3(args, parser):
    from memory_optimizer.offline import read_bounded

    if args.report_only:
        if args.responses or args.relations or args.coverage_config:
            parser.error("S3 report-only accepts no execution responses/configuration")
    elif not all((args.responses, args.relations, args.coverage_config)):
        parser.error("S3 execution requires responses, relations and coverage-config")
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
        if args.query is not None and not 1 <= len(args.query.strip()) <= 2000:
            raise ValueError("invalid query")
        if args.report_only:
            store = PostgresStore(settings.database)
            store.open()
            try:
                report = PostgresS3ReportRepository(store.transaction).report_request(command)
                if args.query:
                    report["search"] = [
                        asdict(r)
                        for r in EpisodeSearchIndex(store).search(
                            SearchQuery(args.query, command.user_id, 10)
                        )
                    ]
            finally:
                store.close()
        else:
            extraction = ReplayExtractor.from_json(read_bounded(args.responses, 1024 * 1024))
            relations = ReplayRelations.from_json(read_bounded(args.relations, 1024 * 1024))
            if extraction.source_payload_hash != payload_digest(
                command
            ) or relations.source_payload_hash != payload_digest(command):
                raise ValueError("response fixture belongs to another request")
            spec = load_spec(
                read_bounded(args.coverage_config, 65536), processor_version(extraction, relations)
            )
            if spec.flow_version != FLOW_VERSION:
                raise ValueError("S3 flow configuration mismatch")
            services = build_s3_services(
                settings, spec=spec, offline_extractor=extraction, relation_proposer=relations
            )
            services.lifecycle.open()
            try:
                receipt = services.add.add(command)
                report = PostgresS3ReportRepository(services.lifecycle.transaction).report(
                    command.user_id, receipt.batch_id
                )
                report["reused"] = receipt.reused
                if args.query:
                    report["search"] = [
                        asdict(r)
                        for r in services.search.search(
                            SearchQuery(args.query, command.user_id, 10)
                        )
                    ]
            finally:
                services.lifecycle.close()
        write_report(args.output, report)
        print(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "stage": report["run"]["stage"],
                    "publication": report["publication"],
                    "model_calls": 0,
                }
            )
        )
    except (ValueError, OSError, ProcessingUnavailable, StorageUnavailable, RequestConflict):
        parser.exit(
            1,
            (
                "S3 execution/export did not complete; saved work may exist. Inspect "
                "input/version/storage and use --report-only.\n"
            ),
        )
