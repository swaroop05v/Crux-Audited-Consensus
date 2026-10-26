"""Deterministic per-call seeds, so any single call can be re-run without replaying the whole stage."""

from __future__ import annotations

import hashlib


def stable_seed(base_seed: int, *parts: object) -> int:
    """A 31-bit seed derived from ``base_seed`` and identifying parts (case id, consultant, sample index, ...)."""
    key = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF
