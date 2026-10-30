"""Shared synthetic material for in-process and abrupt-process coverage tests."""

from pathlib import Path

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.models import AddRequest
from memory_optimizer.processing.checkpoints import RunBudget
from memory_optimizer.processing.coverage_types import CoverageSpec
from memory_optimizer.processing.plan_types import PlanLimits, PlanningSpec


def material(seconds=120):
    examples = Path(__file__).resolve().parents[2] / "examples"
    command = ingest_command(
        AddRequest.model_validate_json((examples / "offline-add.json").read_text())
    )
    replay = ReplayExtractor.from_json((examples / "offline-responses.json").read_text())
    spec = CoverageSpec(
        PlanningSpec(PlanLimits(2000, 300, 32, 100000, 100000, 64)),
        RunBudget(seconds, 6),
        replay.version,
        1,
    )
    return command, replay, spec
