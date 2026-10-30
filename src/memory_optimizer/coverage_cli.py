"""Explicit coverage CLI adapter; v1 assembly and output remain separate."""

import json
import os
import tempfile
from pathlib import Path

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.bootstrap import build_coverage_services
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
from memory_optimizer.processing.coverage_types import read_coverage_spec
from memory_optimizer.storage.coverage_report import PostgresCoverageReportRepository
from memory_optimizer.storage.postgres import PostgresStore


def load_spec(content, processor_version):
    value = strict_json(content, 64 * 1024)
    if type(value) is not dict or "processor_version" in value:
        raise ValueError("processor identity is derived from the supplied responses")
    return read_coverage_spec(dict(value, processor_version=processor_version))


def write_report(path, report):
    """Publish only a complete file, exclusively; an output failure never reruns Add."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".coverage-",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if os.name == "nt":
            os.rename(temporary, destination)  # Windows rename refuses an existing target.
        else:
            os.link(temporary, destination)  # POSIX rename would overwrite the target.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_coverage(args, parser):
    from memory_optimizer.offline import read_bounded

    report_repository = PostgresCoverageReportRepository
    if args.flow == "coverage-temporal-v1":
        from memory_optimizer.storage.temporal_report import PostgresTemporalReportRepository

        report_repository = PostgresTemporalReportRepository

    if args.report_only:
        if args.responses or args.coverage_config:
            parser.error("--report-only reads saved state; omit responses and coverage config")
    elif not args.responses or not args.coverage_config:
        parser.error("coverage execution requires --responses and --coverage-config")
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
        if args.report_only:
            store = PostgresStore(settings.database)
            store.open()
            try:
                report = report_repository(store.transaction).report_request(command)
            finally:
                store.close()
        else:
            replay = ReplayExtractor.from_json(read_bounded(args.responses, 1024 * 1024))
            if replay.source_payload_hash != payload_digest(command):
                raise ValueError("response fixture belongs to another request")
            spec = load_spec(read_bounded(args.coverage_config, 64 * 1024), replay.version)
            if spec.flow_version != args.flow:
                raise ValueError("CLI and configuration flow mismatch")
            services = build_coverage_services(settings, spec=spec, offline_extractor=replay)
            services.lifecycle.open()
            try:
                receipt = services.add.add(command)
                report = report_repository(services.lifecycle.transaction).report(
                    command.user_id, receipt.batch_id
                )
                report["reused"] = receipt.reused
            finally:
                services.lifecycle.close()
        write_report(args.output, report)
        print(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "stage": report["run"]["stage"],
                    "reason": report["run"]["reason"],
                    "candidates": len(report["candidates"]),
                    "model_calls": 0,
                }
            )
        )
    except (ValueError, OSError, ProcessingUnavailable, StorageUnavailable, RequestConflict):
        parser.exit(
            1,
            "Coverage execution or report export did not complete; "
            "saved work may exist. Check input/version/storage; use --report-only to inspect.\n",
        )
