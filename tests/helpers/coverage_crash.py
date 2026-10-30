"""Test-only abrupt exits; no product hooks or provider calls."""

import os
import sys

from coverage_material import material

from memory_optimizer.bootstrap import build_coverage_services
from memory_optimizer.config import Settings
from memory_optimizer.extraction.replay import ReplayExtractor
from memory_optimizer.processing import coverage_workflow
from memory_optimizer.storage.coverage import CoverageIngestRepository, PostgresCoverageRepository
from memory_optimizer.storage.coverage_candidates import BoundCandidateRepository
from memory_optimizer.storage.planning import PostgresPlanningRepository


def main():
    point = sys.argv[1]
    command, replay, spec = material(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    hooks = {
        "initialized": (CoverageIngestRepository, "commit"),
        "plan_reserved": (PostgresPlanningRepository, "reserve"),
        "plan_saved": (PostgresPlanningRepository, "save_plan"),
        "selection_saved": (PostgresPlanningRepository, "save_selection"),
        "bound": (PostgresCoverageRepository, "bind"),
        "candidate_reserved": (BoundCandidateRepository, "reserve"),
        "candidate_saved": (BoundCandidateRepository, "save"),
        "terminal": (PostgresCoverageRepository, "complete"),
    }
    if point in hooks:
        cls, name = hooks[point]
        original = getattr(cls, name)

        def crash(*args, **kwargs):
            original(*args, **kwargs)
            os._exit(17)

        setattr(cls, name, crash)
    else:
        resumed = point.removeprefix("resume-")

        def forbidden(*args):
            raise AssertionError("persisted planning work was repeated")

        if resumed not in {"initialized", "plan_reserved"}:
            coverage_workflow.build_plan = forbidden
        if resumed in {
            "selection_saved",
            "bound",
            "candidate_reserved",
            "candidate_saved",
            "terminal",
            "expired",
        }:
            coverage_workflow.select_baseline = forbidden
        if resumed in {"candidate_saved", "terminal"}:
            original = ReplayExtractor.extract

            def extract(self, request, attempt):
                if request.window.ordinal == 0:
                    raise AssertionError("accepted candidate window was repeated")
                return original(self, request, attempt)

            ReplayExtractor.extract = extract
    services = build_coverage_services(Settings.from_env(), spec=spec, offline_extractor=replay)
    services.lifecycle.open()
    try:
        services.add.add(command)
    finally:
        services.lifecycle.close()


if __name__ == "__main__":
    main()
