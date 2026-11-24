"""Prefixed, sortable identifiers (e.g. ``run_01J...``) for runs, orders and jobs."""

from __future__ import annotations

import os
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        chars.append(_ALPHABET[rem])
    return "".join(reversed(chars))


def new_id(prefix: str) -> str:
    """Return a ULID-style id: 48-bit ms timestamp + 80 random bits, lexicographically sortable."""
    if not prefix.isidentifier() or not prefix.islower():
        raise ValueError(f"prefix must be a lowercase identifier, got {prefix!r}")
    ts_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    return f"{prefix}_{_encode(ts_ms, 10)}{_encode(rand, 16)}"
