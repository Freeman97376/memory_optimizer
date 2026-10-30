"""Test-only abrupt exits; no hooks in product code."""

import os
import sys

from temporal_material import material

from memory_optimizer.bootstrap import build_coverage_services
from memory_optimizer.config import Settings
from memory_optimizer.storage.coverage import CoverageIngestRepository, PostgresCoverageRepository
from memory_optimizer.storage.temporal import TemporalCandidateRepository
from memory_optimizer.temporal.proposals import TemporalCandidate


def main():
    point = sys.argv[1]
    cls, name = {
        "initialized": (CoverageIngestRepository, "commit"),
        "reserved": (TemporalCandidateRepository, "reserve"),
        "inside_transaction": (TemporalCandidate, "body"),
        "saved": (TemporalCandidateRepository, "save"),
        "terminal": (PostgresCoverageRepository, "complete"),
    }[point]
    original = getattr(cls, name)

    def crash(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(17)

    setattr(cls, name, crash)
    command, replay, spec = material()
    services = build_coverage_services(Settings.from_env(), spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        services.add.add(command)
    finally:
        services.lifecycle.close()


if __name__ == "__main__":
    main()
