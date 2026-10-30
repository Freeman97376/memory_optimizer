from contextlib import contextmanager
from importlib.resources import files
from uuid import uuid4, uuid5

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

from memory_optimizer.config import DatabaseSettings
from memory_optimizer.domain import (
    IngestPlan,
    IngestReceipt,
    RequestConflict,
    SearchCandidate,
    SearchQuery,
    StorageUnavailable,
    digest,
)
from memory_optimizer.storage.processing import create_run

DRIVER_ERRORS = (psycopg.Error, PoolTimeout, PoolClosed)


MIGRATIONS = (
    (1, "001_initial.sql"),
    (2, "002_processing.sql"),
    (3, "003_candidates.sql"),
    (4, "004_planning.sql"),
    (5, "005_coverage_runtime.sql"),
    (6, "006_temporal_proposals.sql"),
    (7, "007_s3_publication.sql"),
    (8, "008_provider_budget.sql"),
    (9, "009_openai_s3.sql"),
)


def migration_source(name: str = "001_initial.sql") -> str:
    return (
        files("memory_optimizer.storage").joinpath(f"migrations/{name}").read_text(encoding="utf-8")
    )


def set_schema(connection, schema: str):
    connection.execute(
        sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(sql.Identifier(schema))
    )


def migrate(settings: DatabaseSettings):
    sources = {version: migration_source(name) for version, name in MIGRATIONS}
    with psycopg.connect(settings.database_url.get_secret_value(), connect_timeout=5) as conn:
        conn.execute("SET LOCAL statement_timeout = '60s'")
        conn.execute("SET LOCAL lock_timeout = '5s'")
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (settings.schema,))
        conn.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(settings.schema))
        )
        set_schema(conn, settings.schema)
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version integer PRIMARY KEY, checksum text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now())""")
        previous = dict(conn.execute("SELECT version, checksum FROM schema_migrations").fetchall())
        if sorted(previous) != list(sources)[: len(previous)]:
            raise RuntimeError("unknown or non-contiguous migration history")
        for version, checksum in previous.items():
            if checksum != digest(sources[version]):
                raise RuntimeError("migration checksum mismatch; refusing to change schema")
        for version, source in sources.items():
            if version not in previous:
                conn.execute(source)
                conn.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                    (version, digest(source)),
                )


class PostgresStore:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self.pool = ConnectionPool(
            conninfo=settings.database_url.get_secret_value(),
            min_size=1,
            max_size=settings.pool_size,
            open=False,
            timeout=5,
            reconnect_timeout=10,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "connect_timeout": 5,
                "options": "-c statement_timeout=60000 -c lock_timeout=5000",
            },
        )

    def open(self):
        try:
            self.pool.open(wait=True, timeout=10)
            self.check_ready()
        except DRIVER_ERRORS:
            self.close()
            raise StorageUnavailable("storage startup failed") from None
        except Exception:
            self.close()
            raise

    def close(self):
        self.pool.close()

    @contextmanager
    def transaction(self):
        try:
            with self.pool.connection() as conn, conn.transaction():
                set_schema(conn, self.settings.schema)
                yield conn
        except DRIVER_ERRORS:
            raise StorageUnavailable("storage operation failed") from None

    def check_ready(self):
        with self.transaction() as conn:
            rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
            actual = {row["version"]: row["checksum"] for row in rows}
            expected = {version: digest(migration_source(name)) for version, name in MIGRATIONS}
            if actual != expected:
                raise StorageUnavailable("database migration is missing or incompatible")

    def commit(self, plan: IngestPlan) -> IngestReceipt:
        return self.commit_initialized(
            plan, lambda conn, batch, value: create_run(conn, batch, value.processing)
        )

    def commit_initialized(self, plan: IngestPlan, initialize) -> IngestReceipt:
        """Storage-only seam: initialize the chosen runtime in the source transaction."""
        request = plan.command
        batch_id = uuid4()
        scope_hash, request_hash = digest(request.user_id), digest(request.request_id)
        payload_hash = plan.payload_hash
        with self.transaction() as conn:
            inserted = conn.execute(
                """
                INSERT INTO batches
                    (id, scope_hash, request_hash, user_id, request_id, session_id, payload_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scope_hash, request_hash) DO NOTHING RETURNING id
                """,
                (
                    batch_id,
                    scope_hash,
                    request_hash,
                    request.user_id,
                    request.request_id,
                    request.session_id,
                    payload_hash,
                ),
            ).fetchone()
            if inserted is None:
                existing = conn.execute(
                    """
                    SELECT id, user_id, request_id, session_id, payload_hash FROM batches
                    WHERE scope_hash = %s AND request_hash = %s
                    """,
                    (scope_hash, request_hash),
                ).fetchone()
                expected = {
                    "user_id": request.user_id,
                    "request_id": request.request_id,
                    "payload_hash": payload_hash,
                }
                if existing is None or any(
                    existing[key] != value for key, value in expected.items()
                ):
                    raise RequestConflict()
                receipt = IngestReceipt(
                    request_id=existing["request_id"],
                    user_id=existing["user_id"],
                    session_id=existing["session_id"],
                    batch_id=str(existing["id"]),
                    reused=True,
                )
            else:
                self._write_sources(conn, plan, batch_id)
                initialize(conn, batch_id, plan)
                receipt = IngestReceipt(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    batch_id=str(batch_id),
                    reused=False,
                )
        # The receipt is released only after the transaction context has committed.
        return receipt

    def _write_sources(self, conn, plan: IngestPlan, batch_id):
        for ordinal, (message, parts) in enumerate(
            zip(plan.command.messages, plan.message_slices, strict=True)
        ):
            conn.execute(
                """
                    INSERT INTO messages (batch_id, ordinal, role, content, source_timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                (batch_id, ordinal, message.role, message.content, message.timestamp),
            )
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                        INSERT INTO chunks
                            (id, batch_id, message_ordinal, ordinal, start_offset, end_offset,
                             content)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                    (
                        (
                            uuid5(batch_id, f"{ordinal}:{part.ordinal}"),
                            batch_id,
                            ordinal,
                            part.ordinal,
                            part.start,
                            part.end,
                            part.content,
                        )
                        for part in parts
                    ),
                )

    def search(self, request: SearchQuery) -> list[SearchCandidate]:
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.content, ts_rank_cd(c.search_vector, q.query) AS score
                FROM chunks c JOIN batches b ON b.id = c.batch_id
                CROSS JOIN websearch_to_tsquery('simple', %s) AS q(query)
                WHERE b.scope_hash = %s AND b.user_id = %s AND c.search_vector @@ q.query
                ORDER BY score DESC, b.created_at DESC, b.id, c.message_ordinal, c.ordinal
                LIMIT %s
                """,
                (request.query, digest(request.user_id), request.user_id, request.top_k),
            ).fetchall()
            return [
                SearchCandidate(id=str(row["id"]), content=row["content"], score=row["score"])
                for row in rows
            ]
