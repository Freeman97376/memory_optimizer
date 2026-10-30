import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from memory_optimizer.coverage_cli import load_spec, write_report
from memory_optimizer.domain import ProcessingUnavailable
from memory_optimizer.processing.coverage_report import (
    CoverageReportInput,
    coverage_report,
    measure,
)
from memory_optimizer.processing.coverage_types import CoverageSpec
from memory_optimizer.processing.planning import select_baseline
from memory_optimizer.processing.types import SourceSpan

from .test_plan_checkpoints import snapshot


def unbound(saved):
    spec = CoverageSpec(saved.spec, saved.budget, "replay-test", 1)
    run = dict(
        id="00000000-0000-0000-0000-000000000001",
        batch_id=saved.batch_id,
        spec=json.loads(json.dumps(asdict(spec))),
        created_at=saved.created_at,
        finished_at=None,
        stage="raw_ready",
        outcome=None,
        reason=None,
        selected_at=None,
        model_calls=0,
    )
    return CoverageReportInput(saved, run, None, "payload", (), (), (), ())


def test_union_uses_unicode_coordinates_per_message_not_overlapping_sum():
    metric = measure((SourceSpan(0, 0, 6), SourceSpan(0, 2, 8), SourceSpan(1, 0, 4)), 12)
    assert metric["codepoints"] == 12 and metric["ratio"] == 1
    assert len(metric["ranges"]) == 2
    assert measure((), 0)["ratio"] is None
    assert measure((), 0)["status"] == "not_applicable"
    assert measure((), 12, available=False)["codepoints"] is None


@pytest.mark.parametrize("phase", ["raw", "planned", "selected"])
def test_pure_partial_report_does_not_invent_execution_or_semantic_metrics(phase):
    saved = snapshot()
    if phase == "raw":
        saved = replace(saved, plan=None, stage="raw_ready")
    if phase == "selected":
        saved = replace(
            saved, selection=select_baseline(saved.plan, saved.messages), stage="selection_ready"
        )
    data = unbound(saved)
    report = coverage_report(data)
    assert coverage_report(data) == report
    assert report["coverage"]["accounted_focus"]["ratio"] == (0 if phase == "raw" else 1)
    assert report["coverage"]["selected_focus"]["ratio"] == (1 if phase == "selected" else None)
    assert report["coverage"]["recorded_reply_context"]["ratio"] == 0
    assert report["coverage"]["published_evidence"]["ratio"] == 0
    assert report["limitations"]["semantic_recall"] == "unavailable_without_independent_labels"
    assert report["timing"]["deadline_at"] == saved.deadline_at.isoformat()
    if phase == "selected":
        assert report["coverage"]["not_started_context"]["ratio"] == 1
        assert all(w["id"] is None and w["state"] == "not_started" for w in report["windows"])


@pytest.mark.parametrize("change", ["deadline", "version", "model", "scope"])
def test_report_rejects_mixed_runtime_identity(change):
    data = unbound(snapshot())
    if change == "deadline":
        data = replace(data, planning=replace(data.planning, deadline_at=data.planning.created_at))
    elif change == "version":
        data.run["spec"]["flow_version"] = "v1"
    elif change == "model":
        data.run["model_calls"] = 1
    else:
        data.run["batch_id"] = "other"
    with pytest.raises(ProcessingUnavailable):
        coverage_report(data)


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "extra", "version", "bool", "processor", "oversize"]
)
def test_cli_config_is_explicit_strict_and_bounded(change):
    value = json.loads(Path("examples/coverage-config.json").read_text())
    if change == "missing":
        del value["budget"]
    if change == "extra":
        value["api_key"] = "synthetic"
    if change == "version":
        value["planning"]["selector_version"] = "future"
    if change == "bool":
        value["budget"]["total_seconds"] = True
    if change == "processor":
        value["processor_version"] = "invented"
    content = json.dumps(value)
    if change == "duplicate":
        content = content[:-1] + ',"cleanup_seconds":2}'
    if change == "oversize":
        content = " " * 65537
    with pytest.raises((ValueError, ProcessingUnavailable)):
        load_spec(content, "replay-test")


def test_config_processor_identity_is_derived_from_responses():
    content = Path("examples/coverage-config.json").read_text()
    assert load_spec(content, "replay-a").processor_version == "replay-a"
    assert load_spec(content, "replay-b").processor_version == "replay-b"


@pytest.mark.parametrize("failure", ["serialization", "publish", "exists"])
def test_output_failure_never_publishes_partial_or_overwrites(tmp_path, monkeypatch, failure):
    output = tmp_path / "report.json"
    report = {"safe": "中文🌱"}
    if failure == "serialization":
        report["invalid"] = object()
    if failure == "publish":

        def fail(*args):
            raise OSError("synthetic publish failure")

        monkeypatch.setattr(os, "rename" if os.name == "nt" else "link", fail)
    if failure == "exists":
        output.write_text("owned")
    with pytest.raises((TypeError, OSError)):
        write_report(output, report)
    assert output.read_text() == "owned" if output.exists() else failure != "exists"
    assert not list(tmp_path.glob(".coverage-*.tmp"))


def test_atomic_report_roundtrip(tmp_path):
    output = tmp_path / "report.json"
    write_report(output, {"text": "中文🌱", "ratio": None})
    assert json.loads(output.read_text(encoding="utf-8")) == {"text": "中文🌱", "ratio": None}


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--coverage-config", "x"],
        ["--report-only"],
        ["--flow", "coverage-offline-v1"],
        ["--flow", "coverage-offline-v1", "--report-only", "--responses", "x"],
    ],
)
def test_invalid_cli_combinations_fail_before_database(args, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "memory_optimizer.offline",
            "--request",
            "absent",
            "--output",
            str(tmp_path / "out"),
            *args,
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2 and b"Traceback" not in result.stderr
