"""Synthetic explicit temporal input; no provider or clock during extraction."""

import json
from dataclasses import replace
from pathlib import Path

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.models import AddRequest
from memory_optimizer.processing.checkpoints import RunBudget
from memory_optimizer.processing.coverage_types import read_coverage_spec


def material(seconds=120):
    examples = Path(__file__).resolve().parents[2] / "examples"
    command = ingest_command(
        AddRequest.model_validate_json((examples / "temporal-add.json").read_text(encoding="utf-8"))
    )
    replay = ReplayExtractor.from_json(
        (examples / "temporal-responses.json").read_text(encoding="utf-8")
    )
    value = json.loads((examples / "temporal-config.json").read_text())
    spec = read_coverage_spec(dict(value, processor_version=replay.version))
    return command, replay, replace(spec, budget=RunBudget(seconds, spec.budget.max_attempts))
