import os
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from pydantic import SecretStr

from memory_optimizer.api.app import create_app
from memory_optimizer.config import Settings
from memory_optimizer.storage.postgres import migrate


@pytest.fixture
def db_settings():
    url = os.environ.get("MEMORY_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MEMORY_TEST_DATABASE_URL is required for real PostgreSQL tests")
    info = conninfo_to_dict(url)
    if info.get("host") != "127.0.0.1" or info.get("dbname") not in {
        "memory_optimizer",
        "memory_optimizer_test",
    }:
        pytest.fail("integration tests require the explicitly named loopback development database")
    schema = f"mo_test_{uuid4().hex}"
    settings = Settings(
        database_url=SecretStr(url),
        schema=schema,
        api_key=SecretStr("local-test-key"),
        chunk_chars=60,
        chunk_overlap=10,
    )
    migrate(settings.database)
    try:
        yield settings
    finally:
        # Only this fixture's newly created random schema can be removed.
        assert schema.startswith("mo_test_") and len(schema) == 40
        with psycopg.connect(url) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def client(db_settings):
    with TestClient(
        create_app(db_settings), headers={"Authorization": "Bearer local-test-key"}
    ) as c:
        yield c


@pytest.fixture
def db(db_settings):
    with psycopg.connect(db_settings.database_url.get_secret_value(), autocommit=True) as conn:
        # Each query uses a fresh transaction; a persistent path is appropriate for test inspection.
        conn.execute(
            sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(db_settings.schema))
        )
        yield conn


def counts(db):
    return tuple(
        db.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
        for table in ("batches", "messages", "chunks")
    )


def request_data(**updates):
    data = {
        "request_id": "req-1",
        "user_id": "project-a",
        "session_id": "session-1",
        "messages": [
            {
                "role": "user",
                "content": "  def cache():\r\n    return '中文🌱'\n",
                "timestamp": 1792627200000,
            },
            {"role": "assistant", "content": "cache was proposed; not tested"},
        ],
    }
    return data | updates
