import json
from pathlib import Path

import pytest

from memory_optimizer.env_file import load_env_file
from memory_optimizer.preflight import evaluate, load_manifest


def test_env_file_is_explicit_literal_atomic_and_preserves_inherited_values(tmp_path, monkeypatch):
    monkeypatch.setenv("EXISTING", "inherited")
    monkeypatch.delenv("MO_TEST_LITERAL", raising=False)
    file = tmp_path / "config.env"
    file.write_text('EXISTING=file\nMO_TEST_LITERAL="$(not-a-command)"\n', encoding="utf-8")
    load_env_file(file)
    import os

    assert os.environ["EXISTING"] == "inherited"
    assert os.environ["MO_TEST_LITERAL"] == "$(not-a-command)"
    monkeypatch.delenv("MO_TEST_LITERAL")
    file.write_text("MO_TEST_LITERAL=private-value\nnot an assignment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or duplicate") as error:
        load_env_file(file)
    assert "private-value" not in str(error.value) and "MO_TEST_LITERAL" not in os.environ


def test_frozen_material_splits_hashes_and_evidence():
    path = Path(__file__).parents[1] / "evals/materials/requests-2.32.3/manifest.json"
    _, development = load_manifest(path, "diagnostic")
    _, heldout = load_manifest(path, "heldout")
    assert len(development) == 20 and len(heldout) == 10
    assert {c["sha256"] for c, _ in development}.isdisjoint(c["sha256"] for c, _ in heldout)
    assert all(c["license"] == "Apache-2.0" for c, _ in development + heldout)


@pytest.mark.parametrize("mutation", ["hash", "outside", "missing_quote"])
def test_material_changes_fail_before_network(tmp_path, mutation):
    import hashlib

    source = tmp_path / "source.txt"
    source.write_text("frozen evidence", encoding="utf-8")
    case = dict(
        id="case",
        split="diagnostic",
        source_file="source.txt",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        required_quotes=["evidence"],
        query="evidence",
        top_k=1,
    )
    if mutation == "hash":
        case["sha256"] = "0" * 64
    if mutation == "outside":
        case["source_file"] = "../elsewhere.txt"
    if mutation == "missing_quote":
        case["required_quotes"] = ["absent"]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(dict(schema_version="public-coding-diagnostic-v1", cases=[case])),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_manifest(path, "diagnostic")


def test_comparison_uses_identical_presentation_budget():
    from memory_optimizer.domain import SearchCandidate

    case = dict(required_quotes=["evidence"])
    results = [SearchCandidate("id", "x" * 20 + "evidence", 1.0)]
    assert evaluate(case, results, char_budget=20)["all_found"] is False
    assert evaluate(case, results, char_budget=28)["all_found"] is True


def test_default_http_entry_never_selects_paid_flow(monkeypatch):
    import sys

    from memory_optimizer import __main__ as entry
    from memory_optimizer.api import app

    monkeypatch.setattr(sys, "argv", ["memory_optimizer", "serve"])
    monkeypatch.setenv("MEMORY_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("MEMORY_API_KEY", "local")
    observed = []
    monkeypatch.setattr(app, "create_app", lambda settings, services: observed.append(services))
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **kw: None)
    entry.main()
    assert observed == [None]


def test_online_http_rejects_budget_schema_reset_before_assembly(monkeypatch):
    import sys

    from memory_optimizer import __main__ as entry

    monkeypatch.setattr(sys, "argv", ["memory_optimizer", "serve", "--flow", "s3-openai-v1"])
    monkeypatch.setenv("MEMORY_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("MEMORY_API_KEY", "local")
    monkeypatch.setenv("MEMORY_DATABASE_SCHEMA", "another_budget")
    with pytest.raises(SystemExit):
        entry.main()


def test_single_add_rejects_concurrency_and_releases_after_failure():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from memory_optimizer.admission import SingleAdd
    from memory_optimizer.domain import ProcessingBusy

    entered, release = Event(), Event()

    class Delegate:
        def add(self, command):
            entered.set()
            assert release.wait(2)
            if command == "fail":
                raise ValueError("failure")
            return command

    gate = SingleAdd(Delegate())
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gate.add, "fail")
        assert entered.wait(1)
        with pytest.raises(ProcessingBusy):
            gate.add("second")
        release.set()
        with pytest.raises(ValueError):
            future.result(timeout=1)
    assert gate.add("next") == "next"
