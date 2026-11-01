from __future__ import annotations

import re
from pathlib import Path


_DUPLICATE_SUFFIX_PATTERN = re.compile(r"\(\d+\)(?=\.[^.]+$)")


def normalize_filename(name: str) -> str:
    filename = Path(name).name
    return _DUPLICATE_SUFFIX_PATTERN.sub("", filename)
