# Memory Optimizer

Evidence-preserving Add/Search service for coding-agent memory. The service stores all source messages, selectively extracts episodes, and retrieves current episodes together with exact source evidence. Answer generation and evaluation are outside the service.

The default `v1` mode does not call a model. Explicit `s3-openai-v1` uses the official OpenAI API with `gpt-4o-mini-2024-07-18`, bounded attempts, a persistent call ledger, and conservative fallback to source retrieval.

- [Method and boundaries](docs/METHOD.md)
- [Run and deploy](docs/RUNBOOK.md)
- [Validation and limitations](docs/VALIDATION.md)

## Quick start: Windows / PowerShell

Requires Python 3.12, uv, and Docker with Linux containers.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Initialize-Local.ps1
uv sync --locked
docker compose up -d --wait
uv run --locked --env-file .env python -m memory_optimizer migrate
uv run --locked --env-file .env python -m memory_optimizer serve
```

The initializer generates independent local secrets without replacing an existing `.env`. PostgreSQL listens on `127.0.0.1:55433`; the default API listens on `127.0.0.1:8000`. Database migration is explicit, never a side effect of starting Search.

`GET /health` is public. `POST /add` and `POST /search` require `Authorization: Bearer <MEMORY_API_KEY>` (Token and X-API-Key are also accepted). [Add example](examples/add.json):

```json
{"query":"retry_budget","user_id":"demo-project","top_k":5}
```

This is a Search request; results are returned under `data` with `id`, `content`, and `score`. Requests are strictly validated. Data is isolated by user scope; repeat Add with the same request and identical payload is idempotent, while a conflicting payload is rejected. Online mode admits one Add at a time and returns 503 while busy.

## Validation status

On 2026-10-30, 657 local tests passed, including 224 PostgreSQL integration tests, with no skips. A small real-provider check used 20 diagnostic and 10 held-out excerpts from one public Requests changelog. These are literal evidence checks, not an official benchmark or a semantic-quality score. See the validation document for comparison results, costs, and missing coverage.

No official Smoke or Full result is claimed. Object identity, full hierarchy, normalized event time, vector retrieval, and general historical reprocessing remain incomplete or out of scope.

## License

Project code: [MIT](LICENSE), copyright 2026 Freeman97376. The Requests test materials retain their [Apache-2.0 license](evals/materials/requests-2.32.3/LICENSE) and [NOTICE](evals/materials/requests-2.32.3/NOTICE); the [manifest](evals/materials/requests-2.32.3/manifest.json) pins their source revision and hashes. Dependency licenses remain their respective owners'.
