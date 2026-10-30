"""Test-only abrupt process termination after a real committed checkpoint."""

import json
import os
import sys
from dataclasses import asdict, replace

from memory_optimizer.bootstrap import build_services
from memory_optimizer.config import Settings
from memory_optimizer.domain import IngestCommand, SourceMessage


def main():
    services = build_services(Settings.from_env())
    point = sys.argv[1]
    if point in {"raw", "selected", "complete"}:
        target, method = (
            (services.lifecycle, "commit")
            if point == "raw"
            else (
                services.add.processing.repository,
                "checkpoint" if point == "selected" else "finish_raw",
            )
        )
        original = getattr(target, method)

        def crash(*args):
            original(*args)
            os._exit(17)

        setattr(target, method, crash)
    elif point == "resume-no-selection":

        def forbidden(*args):
            raise AssertionError("committed windows must not be reselected")

        services = replace(
            services,
            add=replace(
                services.add,
                processing=replace(
                    services.add.processing, selectors={"coding-signals-v1": forbidden}
                ),
            ),
        )
    services.lifecycle.open()
    try:
        result = services.add.add(
            IngestCommand(
                "restart-1",
                "project-a",
                "s",
                (
                    SourceMessage("user", "error 🌱\r\n cache"),
                    SourceMessage("assistant", "fixed but not tested"),
                ),
            )
        )
        print(json.dumps(asdict(result)))
    finally:
        services.lifecycle.close()


if __name__ == "__main__":
    main()
