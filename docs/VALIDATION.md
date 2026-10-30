# Validation snapshot: 2026-10-30

## Automated and container checks

657 tests passed: 433 without PostgreSQL and 224 PostgreSQL integration tests; zero failures, errors or skips. Tests cover scope isolation, idempotency, bounded attempts, crash recovery, migration rollback, persistent budget accounting, source-owned evidence, relation wire constraints, current corrections and source-anchored retrieval. Ruff lint and formatting passed for 115 Python files.

A dedicated disposable PostgreSQL/container fixture passed explicit migrations, public health, authenticated Add/Search, unauthorized rejection, user isolation, and application restart with source/idempotency retention. It ran without network or real credentials, made zero provider calls, and used a synthetic source without extraction signals. This verifies container wiring, not provider quality or public HTTPS routing.

## Small real-provider comparison

One public Requests HISTORY.md revision supplied 30 fixed excerpts: 20 diagnostic and 10 held out. Necessary literal source strings were derived checks, not human-confirmed semantic gold. Both methods used identical queries, top-k and a 12,000-character presentation budget. S3 results include exact original evidence and raw fallback; this comparison does not isolate the value of generated summaries.

| Split, final corpus snapshot | Raw: all required strings | S3 aggregate: all required strings | Published episodes |
| --- | --- | --- | --- |
| Diagnostic, 20 excerpts | 17/20 | 18/20 | 15/20 |
| Held out, 10 excerpts | 9/10 | 10/10 | 10/10 |

No diagnostic case that passed raw retrieval failed S3 aggregate retrieval. Two broad diagnostic top-1 queries still selected another release; neither check nor query was changed to conceal the misses. Four diagnostic inputs had no selected signals and one retained strict object-quote failure fallback.

Code and prompts were frozen before the held-out run at 2026-10-30T19:19:10Z. Runtime fingerprint: `sha256:03b5affc9aecc9344716506482cf0d6a7d8be2e7173a09b170a83b105b4f0885` (sorted source paths and file hashes plus pyproject.toml and uv.lock). Processor: `s3-openai:aa4ecffbe68a393a23c14968727548bf4a17ea50fc575c4000442b176005abc9`; Search: `source-anchored-s3-v1`. Held-out after-each-Add and final read-only snapshots had the same aggregate scores. No held-out tuning occurred.

Cumulative development and validation: 106 actual provider calls including failures/retries, 81 successful, zero uncertain; estimated charged usage USD 0.057546. This is the application's usage ledger, not an account invoice, and includes earlier unsuccessful candidates. Costs and calls were not reset between versions.

## What this does not establish

Thirty excerpts from one changelog are not 30 independent coding conversations. Literal evidence retention does not establish correct summaries, natural cross-Add corrections, long-dialogue coverage, large-corpus performance, official compatibility or benchmark success. Formal semantic acceptance remains open. No official Smoke/Full evaluation or production load test has been run. Full requires separately measured capacity and budget.
