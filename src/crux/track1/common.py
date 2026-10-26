"""Helpers shared by the Track 1 stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crux.backends import Backend, BackendError, Registry, get_backend
from crux.backends.fake import FakeBackend
from crux.io import REPO_ROOT, read_jsonl

CONSULTANT_ROLES = ("consultant_a", "consultant_b")
DEFAULT_CASES = REPO_ROOT / "data" / "seed" / "seed_30.jsonl"
RUNS_DIR = REPO_ROOT / "data" / "runs"


def run_path(run: str, stage: str) -> Path:
    return RUNS_DIR / run / f"{stage}.jsonl"


def other_role(role: str) -> str:
    return "consultant_b" if role == "consultant_a" else "consultant_a"


def side(role: str) -> str:
    return role[-1]


def load_cases(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Cases with labels stripped: Track 1 never sees ground truth."""
    cases = read_jsonl(path, "cases")
    return cases[:limit] if limit is not None else cases


def format_reasoning(steps: List[str]) -> str:
    """Numbered steps, as run.py's render_reasoning shows them to consultants."""
    if not steps:
        return "No reasoning provided."
    return "\n".join(f"{position + 1}. {step}" for position, step in enumerate(steps))


def parse_json_object(text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse a JSON object, falling back to the outermost {...} (models wrap JSON in prose or code fences)."""
    if not text or not text.strip():
        return None, "empty output"
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, None
    return None, "no JSON object found"


def make_backend(name: str, registry: Registry, fake_responder: Any) -> Backend:
    """The consultants' backend; ``fake`` uses the stage's canned responder so CLI plumbing runs offline."""
    if name == "fake":
        return FakeBackend(registry, responder=fake_responder)
    return get_backend(registry, name)


def safe_chat(backend: Backend, model_key: str, messages: List[Dict[str, str]],
              **kwargs: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """One completion as (text, params, call_error). A failed call is recorded, never treated as model output."""
    try:
        result = backend.chat(model_key, messages, **kwargs)
    except BackendError as exc:
        return None, None, str(exc)
    return result.text, result.params, None


def params_without_seed(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Call params shared by all calls of a stage; per-call seeds are stored with each output."""
    if params is None:
        return None
    return {key: value for key, value in params.items() if key not in ("seed", "seeds")}
