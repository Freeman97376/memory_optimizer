"""Sequential, restartable, explicitly requested real-provider validation.

One persistent schema/campaign; never drop/reset accounting or retry report failures.
Reports contain excerpts: use only the public, licensed manifest supplied for this task.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from psycopg.conninfo import conninfo_to_dict
from pydantic import SecretStr

from memory_optimizer.config import Settings
from memory_optimizer.coverage_cli import write_report
from memory_optimizer.domain import IngestCommand, SearchQuery, SourceMessage, digest
from memory_optimizer.env_file import load_env_file
from memory_optimizer.llm.contracts import VERSION
from memory_optimizer.llm.types import CAMPAIGN, Limits
from memory_optimizer.online import FLOW, build_online_services
from memory_optimizer.storage.evidence_search import SEARCH_VERSION
from memory_optimizer.storage.llm import PostgresCallLedger
from memory_optimizer.storage.postgres import migrate
from memory_optimizer.storage.relations_report import PostgresS3ReportRepository

PREFLIGHT_SCHEMA = "amc_preflight_20261029"


def load_manifest(path, split):
    root = path.resolve().parent
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "public-coding-diagnostic-v1":
        raise ValueError("unsupported material manifest")
    cases = []
    for case in data["cases"]:
        if case["split"] != split:
            continue
        file = (root / case["source_file"]).resolve()
        if not file.is_relative_to(root):
            raise ValueError("source file outside material directory")
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != case["sha256"]:
            raise ValueError("material hash changed")
        source = raw.decode("utf-8")
        if not source or len(source) > 120000 or not case["required_quotes"]:
            raise ValueError("unbounded material or missing evidence oracle")
        if any(not quote or quote not in source for quote in case["required_quotes"]):
            raise ValueError("expected evidence is not in frozen source")
        if not 1 <= case["top_k"] <= 100 or not 1 <= len(case["query"]) <= 2000:
            raise ValueError("invalid diagnostic query")
        cases.append((case, source))
    if not cases or len(cases) > 30 or len({c["id"] for c, _ in cases}) != len(cases):
        raise ValueError("invalid case set")
    return data, cases


def evaluate(case, results, char_budget=12000):
    # Identical top_k and codepoint presentation budgets for both retrieval methods.
    content = "\n".join(r.content for r in results)[:char_budget]
    found = [q in content for q in case["required_quotes"]]
    return dict(
        required=len(found),
        found=sum(found),
        all_found=all(found),
        presented_chars=len(content),
        results=[asdict(r) for r in results],
        presented_content=content,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["diagnostic", "heldout"], default="diagnostic")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 30:
        parser.error("limit must be between 1 and 30")
    data, cases = load_manifest(args.manifest, args.split)
    cases = cases[: args.limit]
    load_env_file(args.env_file)
    settings = replace(Settings.from_env(), schema=PREFLIGHT_SCHEMA)
    info = conninfo_to_dict(settings.database_url.get_secret_value())
    if info.get("host") != "127.0.0.1" or info.get("dbname") not in {
        "memory_optimizer",
        "memory_optimizer_test",
    }:
        parser.error("real preflight requires the named local development database")
    key = os.environ.get("OPENAI_API_KEY")
    if not args.report_only and not key:
        parser.error("OPENAI_API_KEY is required")
    # No reset option: changing output locations cannot reset the database campaign.
    if not args.report_only:
        migrate(settings.database)
    services = build_online_services(settings, api_key=SecretStr(key or "report-only-no-network"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    services.lifecycle.open()
    try:
        ledger = PostgresCallLedger(services.lifecycle.transaction)
        reader = PostgresS3ReportRepository(services.lifecycle.transaction)
        reader.FLOW_VERSION = FLOW
        for case, source in cases:
            # Versioned, deterministic request ID means a process restart never creates
            # a second Add merely because report export previously failed.
            request_id = "preflight:" + digest(canonical_case(case) + VERSION)
            command = IngestCommand(
                request_id,
                "public-" + case["split"] + ":" + case["project"] + ":" + digest(VERSION)[:12],
                case["id"],
                (SourceMessage("user", source),),
            )
            output = args.output_dir / (case["id"] + ".json")
            if output.exists():
                old = json.loads(output.read_text(encoding="utf-8"))
                if old.get("request_id") != request_id:
                    raise ValueError("existing report belongs to another version")
                print(json.dumps(dict(case=case["id"], status="saved_report_exists")), flush=True)
                continue
            before = ledger.summary(CAMPAIGN)
            started = time.monotonic()
            if not args.report_only:
                if before["calls"] >= Limits().max_calls:
                    print("Approved call limit reached; stopped.", flush=True)
                    break
                services.add.add(command)
            report = reader.report_request(command)
            query = SearchQuery(case["query"], command.user_id, case["top_k"])
            result = dict(
                schema_version="real-preflight-v1",
                search_version=SEARCH_VERSION,
                observation_mode="report-only-snapshot" if args.report_only else "after-each-add",
                case_id=case["id"],
                material_role=data["material_role"],
                split=case["split"],
                source_sha256=case["sha256"],
                source_url=case["source_url"],
                processor_version=VERSION,
                request_id=request_id,
                timing_seconds=time.monotonic() - started,
                ledger_before=before,
                ledger_after=ledger.summary(CAMPAIGN),
                processing=report,
                raw=evaluate(case, services.lifecycle.search(query)),
                s3=evaluate(case, services.search.search(query)),
                semantic_review="pending; literal evidence recall is not semantic correctness",
            )
            write_report(output, result)
            print(
                json.dumps(
                    dict(
                        case=case["id"],
                        outcome=report["run"]["outcome"],
                        raw_all=result["raw"]["all_found"],
                        s3_all=result["s3"]["all_found"],
                        ledger=result["ledger_after"],
                    )
                ),
                flush=True,
            )
        print(json.dumps(dict(campaign=CAMPAIGN, ledger=ledger.summary(CAMPAIGN))), flush=True)
    finally:
        services.lifecycle.close()


def canonical_case(case):
    return json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
