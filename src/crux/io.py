"""JSONL reading and writing for stage files, with the label firewall.

``read_jsonl`` validates every record and, by default, drops the ``eval`` block so method code cannot see labels.
Only evaluation code (Track 4) and the disagreement filter (Track 3) should pass ``with_labels=True``.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from crux.schema import LABEL_BLOCK, validate_record

REPO_ROOT = Path(__file__).resolve().parents[2]


class SchemaError(ValueError):
    """Raised when a record in a stage file fails validation."""


def git_sha(repo: Path = REPO_ROOT) -> Optional[str]:
    """Current commit of ``repo``, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def stage_meta(stage: str, **params: Any) -> Dict[str, Any]:
    """The ``meta`` block every stage writes: stage name, code version, timestamp, plus stage params."""
    return {
        "stage": stage,
        "code_git_sha": git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **params,
    }


def strip_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key != LABEL_BLOCK}


def _raise_if_invalid(record: Any, stage: str, where: str) -> None:
    errors = validate_record(record, stage)
    if errors:
        raise SchemaError(f"{where}: " + "; ".join(errors))


def read_jsonl(path: Path, stage: str, *, with_labels: bool = False, validate: bool = True) -> List[Dict[str, Any]]:
    """Load a stage file. Labels (the ``eval`` block) are removed unless ``with_labels`` is True."""
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if validate:
                _raise_if_invalid(record, stage, f"{path}:{line_no}")
            records.append(record if with_labels else strip_labels(record))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], stage: str) -> int:
    """Validate and write records atomically (temp file, then replace). Returns the number written."""
    path = Path(path)
    records = list(records)
    seen_ids = set()
    for position, record in enumerate(records):
        _raise_if_invalid(record, stage, f"record {position}")
        if record["id"] in seen_ids:
            raise SchemaError(f"record {position}: duplicate id {record['id']!r}")
        seen_ids.add(record["id"])

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    return len(records)


def index_by_id(records: Iterable[Dict[str, Any]], key: str = "id") -> Dict[str, Dict[str, Any]]:
    """Map records by ``key`` (``id`` or ``crux_id``) for joining stage files; duplicate keys are an error."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record[key] in indexed:
            raise SchemaError(f"duplicate {key} {record[key]!r}")
        indexed[record[key]] = record
    return indexed
