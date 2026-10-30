"""Real PostgreSQL ledger: caps survive process objects, concurrency and crashes."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from time import sleep

import psycopg
import pytest

from memory_optimizer.domain import digest
from memory_optimizer.llm.types import CallBlocked, Limits, Result
from memory_optimizer.storage.llm import PostgresCallLedger
from memory_optimizer.storage.postgres import PostgresStore

pytestmark = pytest.mark.integration


@pytest.fixture
def ledger(db_settings):
    store = PostgresStore(db_settings.database)
    store.open()
    try:
        yield PostgresCallLedger(store.transaction)
    finally:
        store.close()


def reserve(ledger, *, limits=None, scope="user-a", key="same", seconds=30):
    return ledger.reserve(
        limits or Limits(), scope, digest(key), datetime.now(UTC) + timedelta(seconds=seconds), 100
    )


def test_shared_atomic_call_cap_and_persistent_restart(ledger):
    limits = Limits(max_calls=2)

    def attempt(i):
        try:
            return reserve(ledger, limits=limits, scope=f"user-{i}")
        except CallBlocked:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(attempt, range(8)))
    assert sum(x is not None for x in accepted) == 2
    restarted = PostgresCallLedger(ledger.transaction)
    assert restarted.summary(limits.campaign)["calls"] == 2
    with pytest.raises(CallBlocked, match="campaign_budget_exhausted"):
        reserve(restarted, limits=limits, scope="another")


def test_dollar_cap_reserves_worst_case_before_network(ledger):
    limits = Limits(max_nusd=20_000_000)
    ticket = reserve(ledger, limits=limits)
    assert ticket.reserved_nusd == 19_260_000
    with pytest.raises(CallBlocked, match="campaign_budget_exhausted"):
        reserve(ledger, limits=limits, key="second")
    ledger.finish(ticket, Result("complete", "{}", 10, 5))
    reserve(ledger, limits=limits, key="second")
    assert ledger.summary(limits.campaign)["calls"] == 2


def test_same_scope_single_flight_cache_and_cross_scope_isolation(ledger):
    ticket = reserve(ledger)
    with pytest.raises(CallBlocked, match="call_in_progress"):
        reserve(ledger)
    settled = ledger.finish(ticket, Result("complete", "{}", 10, 5))
    assert reserve(ledger) == settled
    other = reserve(ledger, scope="user-b")
    assert other.id != ticket.id
    assert ledger.summary(Limits().campaign)["calls"] == 2


def test_failed_and_unknown_calls_charge_attempts_but_are_not_cache_hits(ledger):
    ticket = reserve(ledger)
    ledger.finish(ticket, Result("error", input_tokens=10, output_tokens=5))
    second = reserve(ledger, seconds=0.05)
    sleep(0.08)
    third = reserve(PostgresCallLedger(ledger.transaction))
    summary = ledger.summary(Limits().campaign)
    assert summary["calls"] == 3 and summary["uncertain_calls"] == 2
    assert summary["charged_nusd"] == 4500 + second.reserved_nusd + third.reserved_nusd


def test_ledger_cannot_reset_limits_refund_unknown_or_delete_records(ledger, db):
    ticket = reserve(ledger)
    with pytest.raises(CallBlocked, match="campaign_configuration_changed"):
        reserve(ledger, limits=replace(Limits(), max_calls=1))
    for query in [
        "UPDATE llm_campaigns SET max_calls=1",
        "DELETE FROM llm_campaigns",
        "UPDATE llm_calls SET charged_nusd=0,state='unknown'",
        "DELETE FROM llm_calls",
    ]:
        with pytest.raises(psycopg.Error):
            db.execute(query)
    assert ledger.summary(Limits().campaign)["charged_nusd"] == ticket.reserved_nusd


def test_late_db_settlement_never_releases_usable_response(ledger):
    ticket = reserve(ledger, seconds=0.05)
    sleep(0.08)
    result = ledger.finish(ticket, Result("complete", "{}", 10, 5))
    assert result.status == "error" and result.content is None
    assert ledger.summary(Limits().campaign)["successful_calls"] == 0


def test_repeated_settlement_cannot_refund_original_cost(ledger):
    ticket = reserve(ledger)
    ledger.finish(ticket, Result("error"))
    from memory_optimizer.domain import ProcessingUnavailable

    with pytest.raises(ProcessingUnavailable):
        ledger.finish(ticket, Result("complete", "{}", 0, 0))
    assert ledger.summary(Limits().campaign)["charged_nusd"] == ticket.reserved_nusd


@pytest.mark.parametrize("failure_file", ["008_provider_budget.sql", "009_openai_s3.sql"])
def test_provider_migration_is_atomic_and_preserves_legacy(
    db_settings, db, monkeypatch, failure_file
):
    from memory_optimizer.bootstrap import build_services
    from memory_optimizer.domain import IngestCommand, SourceMessage
    from memory_optimizer.storage import postgres

    old = build_services(db_settings)
    command = IngestCommand(
        "legacy", "user", "session", (SourceMessage("user", "Changed old cache."),)
    )
    old.lifecycle.open()
    try:
        receipt = old.add.add(command)
    finally:
        old.lifecycle.close()
    rows = db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall()
    source = db.execute("SELECT * FROM messages").fetchall()
    # Recreate pre-008 contract only inside this fixture's random mo_test_ schema.
    db.execute(
        postgres.migration_source("009_openai_s3.sql").replace(
            "'s3-offline-v1','s3-openai-v1'", "'s3-offline-v1'"
        )
    )
    db.execute("DROP TABLE llm_calls,llm_campaigns")
    db.execute("DROP FUNCTION protect_llm_call(),protect_llm_campaign()")
    db.execute("DELETE FROM schema_migrations WHERE version>=8")
    original = postgres.migration_source
    with monkeypatch.context() as patch:
        patch.setattr(
            postgres,
            "migration_source",
            lambda name: (
                original(name)
                + ("\nSELECT missing_preflight_function();" if name == failure_file else "")
            ),
        )
        with pytest.raises(psycopg.Error):
            postgres.migrate(db_settings.database)
    assert db.execute("SELECT to_regclass('llm_calls')").fetchone() == (None,)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone() == (7,)
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall() == rows
    postgres.migrate(db_settings.database)
    postgres.migrate(db_settings.database)
    assert db.execute("SELECT * FROM messages").fetchall() == source
    assert db.execute("SELECT row_to_json(r) FROM processing_runs r").fetchall() == rows
    restored = build_services(db_settings)
    restored.lifecycle.open()
    try:
        assert restored.add.add(command).batch_id == receipt.batch_id
    finally:
        restored.lifecycle.close()
