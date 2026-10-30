# Runbook

Use the root README for the default local setup. Keep `.env`, real inputs, database backups, generated reports and populated deployment configuration private.

## Local tests

```powershell
uv run --locked --env-file .env python -m pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

Without `MEMORY_TEST_DATABASE_URL`, PostgreSQL tests skip. A passing run with skips is not a complete integration result. The fixture accepts a named loopback development database and creates/deletes only its own random `mo_test_...` schema.

## Explicit real preflight

Configure `OPENAI_API_KEY` locally, keep the same persistent PostgreSQL database, and inspect the cost/call limits in METHOD.md before running. These commands incur real provider calls:

```powershell
uv run --locked python -m memory_optimizer.preflight --env-file .env --manifest evals/materials/requests-2.32.3/manifest.json --limit 5 --output-dir evals/results/first-five
```

Inspect published episodes, unknowns, coverage and raw comparison before expanding to `--limit 20`. Freeze code/prompts before `--split heldout --limit 10`; do not tune on those held-out results. New output directories do not reset spending or erase previous attempts. Re-running a completed version reuses its idempotent state. `--report-only` reads existing state without new model calls; choose a fresh output directory. Reports can contain source data and should not be committed.

The preflight command is restricted to the named loopback development database and schema `amc_preflight_20261029`. The shared campaign is `amc-20261029-preflight`. This finite preflight configuration is not a validated Full evaluation budget.

## Dedicated container deployment

`deploy/compose.yml` uses one non-root app worker (512 MiB cap), a separate PostgreSQL volume (768 MiB cap), a read-only app filesystem, dropped capabilities and no-new-privileges. App HTTP binds to host loopback port 58001; PostgreSQL has no published port. The app container needs outbound access to the official OpenAI endpoint in online mode.

Make a populated copy of `deploy/production.env.example` outside the source directory with restrictive file permissions, independent database and inbound API credentials, and the authorized provider key. Do not print `docker compose config` with real credentials. Always name the env file explicitly:

```sh
docker compose --env-file /secure/path/amc.env -f deploy/compose.yml build api
docker compose --env-file /secure/path/amc.env -f deploy/compose.yml up -d postgres
# Restore the existing entire campaign schema here when moving an existing preflight.
# Keep the old paid writer stopped; verify accounting and migrations before startup.
docker compose --env-file /secure/path/amc.env -f deploy/compose.yml run --rm migrate
docker compose --env-file /secure/path/amc.env -f deploy/compose.yml up -d api
```

Only run fresh migrations for a genuinely new independent installation; moving an existing approved campaign must never receive a fresh budget. `deploy/nginx-location.conf.example` is an `/amc/` location fragment for an existing HTTPS server block, not a complete TLS configuration. Test direct app, local proxy and public HTTPS separately, including unauthorized rejection, authenticated Add/Search, scope isolation and replay.

Stop the app without deleting its database volume. Keep source, image, migration checksums and configuration identity together. Rollback must retain all current ledger rows; restoring an older backup cannot refund spending. A previous binary is not automatically compatible with newer schema checksums. Large-input quality, resource contention and official evaluation capacity remain unverified.
