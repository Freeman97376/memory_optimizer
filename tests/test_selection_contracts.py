"""Synthetic type/version boundaries, not planner or semantic-quality validation."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from memory_optimizer.domain import (
    IngestCommand,
    IngestReceipt,
    ProcessingUnavailable,
    SourceMessage,
)
from memory_optimizer.processing import selection_contracts
from memory_optimizer.processing.plan_types import (
    CoverageRange,
    ExecutionDiagnostic,
    FinalSelection,
    PlanLimits,
    PlanningSpec,
    PlanWindow,
    SelectionPlan,
)
from memory_optimizer.processing.planning import build_plan
from memory_optimizer.processing.selection import select_windows
from memory_optimizer.processing.selection_contracts import (
    SelectionBinding,
    selection_contract,
    validate_final_contract,
)
from memory_optimizer.processing.types import (
    ProcessingSpec,
    RunLease,
    Selection,
    SelectionConfig,
    SourceSpan,
)
from memory_optimizer.processing.workflow import ProcessingFlow

MESSAGES = (SourceMessage("user", "error"), SourceMessage("assistant", "未知🌱"))
LEGACY_SPEC = ProcessingSpec()


def planning_spec():
    return PlanningSpec(PlanLimits(10, 0, 1, 100, 100, 2))


def plan(spec=None):
    first, second = SourceSpan(0, 0, 5), SourceSpan(1, 0, 3)
    return SelectionPlan(
        spec or planning_spec(),
        "a" * 64,
        (CoverageRange(first, "candidate", ("problem",)), CoverageRange(second, "candidate")),
        (PlanWindow(first, (first,), ("problem",)), PlanWindow(second, (second,))),
    )


class RecordingRepository:
    def __init__(self, spec=LEGACY_SPEC, stage="raw_ready"):
        self.lease = RunLease("run", "batch", "u", "owner", 1, stage, spec)
        self.checkpoints = []
        self.finished = []

    def claim(self, user_id, batch_id, lease_seconds):
        assert (user_id, batch_id) == ("u", "batch")
        return self.lease

    def checkpoint(self, lease, selection):
        self.checkpoints.append((lease, selection))

    def finish_raw(self, lease):
        self.finished.append(lease)


def finish(flow):
    flow.finish(
        IngestCommand("r", "u", "s", MESSAGES),
        IngestReceipt("r", "u", "s", "batch", False),
    )


def forbidden(*args, **kwargs):
    raise AssertionError("wrong contract invoked")


def test_legacy_persisted_spec_shape_and_defaults_are_unchanged():
    assert asdict(ProcessingSpec()) == {
        "flow_version": "selection-v1",
        "selector_version": "coding-signals-v1",
        "selection": {"focus_chars": 2000, "context_chars": 300, "max_windows": 32},
        "processor_version": None,
    }


@pytest.mark.parametrize("flow_version", ["selection-v1", "offline-candidates-v1"])
def test_both_legacy_flows_use_the_v1_selector_and_validator(flow_version, monkeypatch):
    spec = ProcessingSpec(flow_version=flow_version, selection=SelectionConfig(context_chars=0))
    calls = []
    original = selection_contracts.validate_selection

    def checked(result, messages, config):
        calls.append((result, messages, config))
        original(result, messages, config)

    monkeypatch.setattr(selection_contracts, "validate_selection", checked)
    result = SelectionBinding(spec, select_windows).select(MESSAGES)
    assert result.candidate_count == 1
    assert result.windows[0].focus == SourceSpan(0, 0, 5)
    assert result.windows[0].reasons == ("problem",)
    assert calls == [(result, MESSAGES, spec.selection)]


def test_new_plan_routes_to_its_contract_without_legacy_validation(monkeypatch):
    result = build_plan(MESSAGES, planning_spec())
    calls = []

    def fixture_selector(messages, spec):
        calls.append((messages, spec))
        return result

    monkeypatch.setattr(selection_contracts, "validate_selection", forbidden)
    assert SelectionBinding(result.spec, fixture_selector).select(MESSAGES) is result
    assert calls == [(MESSAGES, result.spec)]
    assert len(result.pool) == 1
    assert result.coverage[1].disposition == "no_signal"
    assert not isinstance(result, Selection)


@pytest.mark.parametrize("result", [plan(), FinalSelection("a" * 64, ()), None])
def test_nonlegacy_results_never_reach_v1_validator_or_checkpoint(result, monkeypatch):
    repository = RecordingRepository()
    monkeypatch.setattr(selection_contracts, "validate_selection", forbidden)
    flow = ProcessingFlow(repository, selectors={"coding-signals-v1": lambda *_: result})
    with pytest.raises(ValueError, match="legacy selection contract"):
        finish(flow)
    assert repository.checkpoints == repository.finished == []


def test_invalid_legacy_result_still_hits_original_validation_before_checkpoint():
    repository = RecordingRepository()
    invalid = Selection((), 1)
    with pytest.raises(ValueError, match="invalid selection coverage"):
        finish(ProcessingFlow(repository, selectors={"coding-signals-v1": lambda *_: invalid}))
    assert repository.checkpoints == repository.finished == []


@pytest.mark.parametrize(
    "result",
    [
        Selection((), 0),
        FinalSelection("a" * 64, ()),
        replace(
            plan(),
            spec=replace(
                planning_spec(), limits=replace(planning_spec().limits, max_scan_chars=50)
            ),
        ),
    ],
)
def test_planning_rejects_legacy_final_or_different_config_results(result):
    with pytest.raises(ValueError, match="planning contract"):
        SelectionBinding(planning_spec(), lambda *_: result).select(MESSAGES)


@pytest.mark.parametrize(
    "spec",
    [
        ProcessingSpec(flow_version="unknown"),
        ProcessingSpec(selector_version="coding-coverage-v2"),
        ProcessingSpec(flow_version="coverage-selection-v2"),
        replace(planning_spec(), flow_version="selection-v1"),
        replace(planning_spec(), selector_version="coding-signals-v1"),
        replace(planning_spec(), schema_version="coverage-plan-v99"),
        replace(planning_spec(), segmentation_version="unknown"),
        replace(planning_spec(), measurement_version="token-guess"),
        None,
    ],
)
def test_unknown_or_mixed_combinations_rejected_before_selector_call(spec):
    with pytest.raises(ProcessingUnavailable, match="contract combination"):
        SelectionBinding(spec, forbidden)


def test_new_flow_is_not_accidentally_installed_in_legacy_orchestration():
    repository = RecordingRepository(planning_spec())
    with pytest.raises(ProcessingUnavailable, match="separate processing flow"):
        ProcessingFlow(repository, spec=planning_spec())
    with pytest.raises(ProcessingUnavailable, match="processing version"):
        finish(ProcessingFlow(repository, selectors={"coding-coverage-v2": forbidden}))
    assert repository.checkpoints == repository.finished == []


def test_declared_but_unknown_selector_has_no_default_fallback():
    repository = RecordingRepository(ProcessingSpec(selector_version="unregistered"))
    with pytest.raises(ProcessingUnavailable, match="contract combination"):
        finish(ProcessingFlow(repository, selectors={"unregistered": forbidden}))
    assert repository.checkpoints == repository.finished == []


def test_v1_checkpoints_and_resumes_without_reselection():
    repository = RecordingRepository()
    finish(ProcessingFlow(repository))
    assert len(repository.checkpoints) == len(repository.finished) == 1
    assert repository.checkpoints[0][1].candidate_count == 1
    assert repository.finished[0].stage == "selected"
    resumed = RecordingRepository(stage="selected")
    finish(ProcessingFlow(resumed, selectors={"coding-signals-v1": forbidden}))
    assert resumed.checkpoints == [] and len(resumed.finished) == 1
    completed = RecordingRepository(spec=None, stage="complete")
    finish(ProcessingFlow(completed, selectors={"coding-signals-v1": forbidden}))
    assert completed.checkpoints == completed.finished == []


def test_offline_processor_still_owns_its_version_and_post_selection_work():
    spec = ProcessingSpec(flow_version="offline-candidates-v1", processor_version="fixture")
    repository = RecordingRepository(spec)
    calls = []

    class Processor:
        def check_version(self, value):
            assert value == spec
            calls.append("version")

        def process(self, lease, command):
            assert lease.stage == "selected" and command.messages == MESSAGES
            calls.append("process")

    finish(ProcessingFlow(repository, processors={"offline-candidates-v1": Processor()}))
    assert calls == ["version", "process"]
    assert len(repository.checkpoints) == 1 and repository.finished == []


@pytest.mark.parametrize(
    "changes",
    [
        {"focus_chars": 0},
        {"max_windows": True},
        {"max_scan_chars": -1},
        {"max_pool_windows": 0},
        {"context_chars": 0.5},
    ],
)
def test_invalid_budget_shapes_cannot_be_saved_as_planning_specs(changes):
    with pytest.raises(ValueError):
        replace(planning_spec().limits, **changes)


def test_zero_budgets_are_explicit_and_do_not_inherit_v1_nonzero_minimum():
    spec = PlanningSpec(PlanLimits(10, 0, 0, 0, 0, 0))
    assert selection_contract(spec) == "coverage-plan-v1"
    empty = SelectionPlan(spec, "a" * 64, (), ())
    validate_final_contract(FinalSelection(empty.fingerprint, ()), empty)


def test_plan_values_are_deeply_immutable_under_normal_construction():
    value = plan()
    with pytest.raises(FrozenInstanceError):
        value.spec.limits.max_windows = 20
    with pytest.raises(FrozenInstanceError):
        value.pool[0].focus.start = 2
    with pytest.raises(ValueError, match="immutable"):
        replace(value, coverage=list(value.coverage))
    with pytest.raises(ValueError, match="signals"):
        CoverageRange(SourceSpan(0, 0, 5), "scan_budget", ("problem",))
    with pytest.raises(ValueError, match="integer"):
        PlanWindow(SourceSpan(True, 0, 5), (SourceSpan(0, 0, 5),))


def test_final_references_bind_to_exact_plan_and_never_rewrite_it():
    value = plan()
    original = value.fingerprint
    chosen = FinalSelection(original, (1,))
    validate_final_contract(chosen, value)
    ExecutionDiagnostic("selection", "deadline_exhausted", (), original)
    assert value.fingerprint == original
    for changed in (
        replace(value, source_digest="b" * 64),
        replace(value, pool=tuple(reversed(value.pool))),
        replace(
            value, spec=replace(value.spec, limits=replace(value.spec.limits, max_scan_chars=99))
        ),
    ):
        with pytest.raises(ValueError, match="different plan"):
            validate_final_contract(chosen, changed)
    for ordinals in ((2,), (0, 1)):
        with pytest.raises(ValueError, match="references or budget"):
            validate_final_contract(FinalSelection(original, ordinals), value)
    with pytest.raises(ValueError, match="duplicate"):
        FinalSelection(original, (0, 0))
    with pytest.raises(ValueError, match="immutable"):
        FinalSelection(original, (True,))


def test_incomplete_planning_diagnostic_cannot_masquerade_as_completed_selection():
    ExecutionDiagnostic("planning", "interrupted", (SourceSpan(0, 0, 5),))
    for phase, outcome in (("planning", "completed"), ("selection", "deadline_exhausted")):
        with pytest.raises(ValueError, match="plan reference"):
            ExecutionDiagnostic(phase, outcome, ())


def test_core_contracts_run_without_http_database_provider_or_process_dependencies():
    script = """
import importlib.abc
import sys
class BlockAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ('fastapi', 'pydantic', 'psycopg', 'psycopg_pool', 'openai', 'anthropic',
                 'git', 'requests', 'httpx', 'memory_optimizer.storage',
                 'memory_optimizer.api', 'memory_optimizer.bootstrap')
        if any(fullname == name or fullname.startswith(name + '.') for name in roots):
            raise ImportError('unexpected adapter: ' + fullname)
sys.meta_path.insert(0, BlockAdapters())
from memory_optimizer.processing.plan_types import PlanLimits, PlanningSpec
from memory_optimizer.processing.selection_contracts import SelectionBinding, selection_contract
from memory_optimizer.processing.planning import build_plan
from memory_optimizer.processing.selection import select_windows
from memory_optimizer.processing.types import ProcessingSpec
from memory_optimizer.domain import SourceMessage
binding = SelectionBinding(ProcessingSpec(), select_windows)
assert binding.select((SourceMessage('user', 'error'),)).candidate_count == 1
assert selection_contract(PlanningSpec(PlanLimits(10, 0, 0, 0, 0, 0))) == 'coverage-plan-v1'
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
