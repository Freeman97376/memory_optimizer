"""Abrupt exits around real offline extraction checkpoints; test-only hooks."""

import os
import sys
from pathlib import Path

from memory_optimizer.api.mapping import ingest_command
from memory_optimizer.bootstrap import build_services
from memory_optimizer.config import Settings
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.models import AddRequest


def main():
    examples = Path(__file__).resolve().parents[2] / "examples"
    command = ingest_command(
        AddRequest.model_validate_json((examples / "offline-add.json").read_text())
    )
    replay = ReplayExtractor.from_json((examples / "offline-responses.json").read_text())
    point = sys.argv[1]
    if point == "resume-save":
        original_extract = ReplayExtractor.extract

        def extract(self, request, attempt):
            if request.window.ordinal == 0:
                raise AssertionError("completed first window was re-extracted")
            return original_extract(self, request, attempt)

        ReplayExtractor.extract = extract
    services = build_services(Settings.from_env(), offline_extractor=replay)
    if point in {"reserve", "save"}:
        repo = services.add.processing.processors["offline-candidates-v1"].repository
        original = getattr(repo, point)

        def crash(*args):
            original(*args)
            os._exit(17)

        setattr(repo, point, crash)
    services.lifecycle.open()
    try:
        services.add.add(command)
    finally:
        services.lifecycle.close()


if __name__ == "__main__":
    main()
