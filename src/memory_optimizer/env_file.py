"""Explicit local env-file loading, without shell expansion or secret-bearing errors."""

import os
import re
from pathlib import Path


def load_env_file(path):
    # Read and validate all entries before changing this process. Never overwrite
    # inherited environment and never read a second, implicitly discovered file.
    values = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match or match[1] in values:
            raise ValueError("invalid or duplicate environment assignment")
        value = match[2]
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("invalid environment quoting")
            value = value[1:-1]
        values[match[1]] = value
    for name, value in values.items():
        os.environ.setdefault(name, value)
