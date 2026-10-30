"""Actual abrupt exit hooks at production S3 boundaries; tests only."""

import os
import sys

from memory_optimizer.bootstrap import build_s3_services
from memory_optimizer.config import Settings
from memory_optimizer.storage import publication
from tests.helpers.s3_material import material

point = sys.argv[1]
command, extraction, proposer, spec = material(
    "two_windows", seconds=int(sys.argv[2]) if len(sys.argv) > 2 else 120
)
app = build_s3_services(
    Settings.from_env(), spec=spec, offline_extractor=extraction, relation_proposer=proposer
)
app.lifecycle.open()


def after(owner, method):
    original = getattr(owner, method)

    def stopped(*args, **kwargs):
        result = original(*args, **kwargs)
        os._exit(23)
        return result

    setattr(owner, method, stopped)


if point == "initialized":
    after(app.add.repository, "commit")
elif point == "snapshot":
    after(app.add.processing.repository, "prepare")
elif point == "reserved":
    after(app.add.processing.repository, "reserve_relation")
elif point == "decision":
    after(app.add.processing.repository, "save_relation")
elif point == "inside_publication":
    after(publication, "append_version")
elif point == "published":
    after(app.add.processing.repository, "publish")
else:
    raise ValueError("unknown test hook")
app.add.add(command)
