"""Helpers shared by the Track 1 stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


@dataclass
class CallOutcome:
    """One completion. A failed call is recorded, never treated as model output."""

    text: Optional[str]
    params: Optional[Dict[str, Any]]
    call_error: Optional[str]
    finish_reason: Optional[str]

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def fields(self) -> Dict[str, Any]:
        """Per-output fields every Track 1 record stores."""
        return {"raw_output": self.text, "call_error": self.call_error, "finish_reason": self.finish_reason}


def safe_chat(backend: Backend, model_key: str, messages: List[Dict[str, str]], **kwargs: Any) -> CallOutcome:
    try:
        result = backend.chat(model_key, messages, **kwargs)
    except BackendError as exc:
        return CallOutcome(None, None, str(exc), None)
    return CallOutcome(result.text, result.params, None, result.finish_reason)


# Exactly one per failed output, checked in this order: the call failed; the reply was cut off at max_tokens
# (never parsed, even if a prefix happens to be valid JSON); the reply was complete but did not parse.
FAILURE_TYPES = ("call_error", "truncated", "parse_error")


def parse_outcome(outcome: CallOutcome, parse: Callable[[Optional[str]], Dict[str, Any]],
                  empty: Dict[str, Any]) -> Dict[str, Any]:
    """``parse(text)`` returns a dict with ``parse_ok``/``parse_error``; ``empty`` holds the fields to null out
    when there is nothing to parse. Adds ``failure_type``."""
    if outcome.call_error is not None:
        return {**empty, "parse_ok": False, "parse_error": "call failed", "failure_type": "call_error"}
    if outcome.truncated:
        return {**empty, "parse_ok": False, "parse_error": "truncated at max_tokens", "failure_type": "truncated"}
    parsed = parse(outcome.text)
    return {**parsed, "failure_type": None if parsed["parse_ok"] else "parse_error"}


def failure_summary(outputs: List[Dict[str, Any]]) -> str:
    """Counts such as "38 ok, 0 call_error, 1 truncated, 1 parse_error", for stage logs."""
    counts = {kind: sum(o.get("failure_type") == kind for o in outputs) for kind in FAILURE_TYPES}
    ok = sum(o.get("failure_type") is None for o in outputs)
    return ", ".join([f"{ok} ok"] + [f"{n} {kind}" for kind, n in counts.items()])


_PER_CALL_KEYS = ("seed", "seeds", "finish_reasons", "truncated", "usage")


def shared_call_params(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Call params common to all calls of a stage. Per-call values (seed, finish reason, token usage) are
    stored with each output instead."""
    if params is None:
        return None
    return {key: value for key, value in params.items() if key not in _PER_CALL_KEYS}
