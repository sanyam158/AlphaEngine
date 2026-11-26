"""Immutable store for raw vendor payloads (``raw/<source>/<yyyy>/<name>``).

Raw files are written once and never modified, so curated data can always be rebuilt and
re-downloads are avoided.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, source: str, day: date, name: str) -> Path:
        return self.root / source / f"{day:%Y}" / name

    def get(self, source: str, day: date, name: str) -> bytes | None:
        p = self.path_for(source, day, name)
        return p.read_bytes() if p.exists() else None

    def put(self, source: str, day: date, name: str, payload: bytes) -> Path:
        """Write ``payload`` unless the file already exists (first write wins)."""
        p = self.path_for(source, day, name)
        if p.exists():
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, p)
        return p
