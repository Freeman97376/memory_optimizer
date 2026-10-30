"""Real database source-priority and identifier regression checks, no paid calls."""

import pytest

from memory_optimizer.bootstrap import build_services
from memory_optimizer.domain import IngestCommand, SearchQuery, SourceMessage
from memory_optimizer.storage.evidence_search import EvidenceSearchIndex
from tests.helpers.s3_material import FIRST, material
from tests.test_s3_postgres import services

pytestmark = pytest.mark.integration


def test_published_episode_keeps_exact_source_in_same_result(db_settings):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
        hits = EvidenceSearchIndex(app.lifecycle).search(SearchQuery("lock", command.user_id, 1))
        assert len(hits) == 1 and hits[0].content.startswith("经历证据")
        assert FIRST in hits[0].content
        assert EvidenceSearchIndex(app.lifecycle).search(SearchQuery("lock", "other", 5)) == []


def test_episode_does_not_displace_more_relevant_unprocessed_source(db_settings):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    raw = build_services(db_settings)
    raw.lifecycle.open()
    try:
        content = "lock " * 12 + "Original applicability: only version 9 and no other version."
        raw.add.add(
            IngestCommand("raw-high-score", command.user_id, "s", (SourceMessage("user", content),))
        )
        query = SearchQuery("lock", command.user_id, 1)
        assert content.startswith(raw.lifecycle.search(query)[0].content)
        hits = EvidenceSearchIndex(raw.lifecycle).search(query)
        assert len(hits) == 1 and content in hits[0].content
        assert not hits[0].content.startswith("经历证据")
    finally:
        raw.lifecycle.close()


def test_qualified_code_identifier_and_literal_percent_match_without_wildcards(db_settings):
    raw = build_services(db_settings)
    raw.lifecycle.open()
    try:
        raw.add.add(
            IngestCommand(
                "symbol",
                "symbols",
                "s",
                (SourceMessage("user", "library.transport.TLSProblemHandler wraps a failure."),),
            )
        )
        assert raw.lifecycle.search(SearchQuery("TLSProblemHandler", "symbols", 1)) == []
        index = EvidenceSearchIndex(raw.lifecycle)
        assert (
            "library.transport.TLSProblemHandler"
            in index.search(SearchQuery("TLSProblemHandler", "symbols", 1))[0].content
        )
        assert index.search(SearchQuery("%", "symbols", 1)) == []
        assert index.search(SearchQuery("TLSProblemHandler", "different-scope", 1)) == []
    finally:
        raw.lifecycle.close()


def test_source_anchor_retains_current_cross_add_correction(db_settings):
    with services(db_settings, material()) as (app, command):
        app.add.add(command)
    with services(db_settings, material("correction")) as (app, command):
        app.add.add(command)
        result = EvidenceSearchIndex(app.lifecycle).search(
            SearchQuery("single-thread", command.user_id, 1)
        )
        assert len(result) == 1
        assert "single-thread passed" in result[0].content
        assert "duplicates under concurrency" in result[0].content
        assert "来源明确关联 (correction)" in result[0].content
