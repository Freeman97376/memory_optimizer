# Method and implementation boundaries

Memory Optimizer implements synchronous Add and read-only Search. It preserves source messages independently of derived memory, so extraction failure does not erase evidence.

## Add

1. Validate the request, scope and idempotency key, then atomically persist source messages, searchable chunks and processing state.
2. In explicit online mode, use versioned `coding-coverage-v3` heuristics to select bounded source windows. Capability verbs complement the older error/change signals. These are selection heuristics, not claims that a change succeeded. Legacy selector versions retain their behavior.
3. Ask the fixed OpenAI snapshot for structured episode candidates. Evidence is selected through program-owned source-unit IDs; the program recovers exact Unicode spans, roles and source timestamps. Candidate and temporal validators reject unsupported references and preserve unknowns. A release-note claim is not independent test verification.
4. Ask for one relation assessment per current candidate: independent, unresolved, or related. A dynamic JSON schema separates current references from historical targets. Existing conservative evidence checks decide whether a relation can publish. Same filename or topic alone does not establish identity or continuity.
5. Atomically publish validated episodes, necessary qualifications and current-version pointers. Every Add keeps its original deadline; planning, extraction and relations share a bounded attempt allowance. Interrupted or invalid work is recorded and can degrade to retained raw evidence.

## Search

Online `source-anchored-s3-v1` reads current episodes and source in one consistent PostgreSQL snapshot. Raw source matches anchor ranking so unrelated summaries cannot crowd out a better original match. A literal substring fallback supports qualified code symbols. The query is parameterized and user scoped.

Matching source is associated with current published episodes through source spans. Exact whole messages up to 6,000 characters can accompany the complete episode; longer messages contribute the matched chunk. Necessary correction context is preserved. A complete retrieval unit exceeding 12,000 characters is omitted instead of being clipped. Additional relevant episodes may fill remaining result slots. Search performs no model calls, writes or migrations, and never generates a final answer.

This is lexical retrieval with source preservation, not semantic or vector ranking. Broad top-1 queries can still select another relevant release. Substring matching and JSONB association have not been load-tested at benchmark scale.

## Limits and version identity

Online defaults: 600-second total Add deadline, 15-second cleanup reserve, 16 shared stage attempts, 90-second per-request transport timeout, four initial windows, 120,000 scanned characters, 40 pooled candidates and 24,000 total context characters. These are configured bounds, not throughput guarantees.

Model, prompts, schemas, wire codec and selector identity bind the processor version. Search has a separate version and must also be bound to the source commit. Source hashes, user scope and input metadata bind cached responses; reuse still validates the response. No automatic conversion of unknown source times into a deterministic event order occurs.

The explicit preflight campaign has immutable cumulative caps of USD 10 and 200 provider calls, including failed and interrupted calls. Reserve worst-case usage before external calls; reduce it only when trustworthy usage is available. There are no implicit provider retries or redirects. Migrating a campaign requires carrying its entire database state, never resetting or cloning a live budget for simultaneous use.

## Code map

- `src/memory_optimizer/online.py`: explicit online assembly.
- `processing/`: versioned planning, deadlines and recovery values.
- `llm/`: pure wire contracts, official transport and budgeted calls.
- `relations/`: conservative coordination and complete rendering.
- `storage/`: source, checkpoints, ledger, atomic publication and search adapters.
- `tests/`: synthetic contract and real-PostgreSQL regressions with fake provider transport unless explicitly running the separate real preflight command.
