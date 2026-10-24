"""Shared JSONL data contract: per-stage record validation and the label firewall.

Every pipeline stage writes its own JSONL file, one record per case (or per crux), joined on ``id`` / ``crux_id``
(docs/decisions.md D7). Each record carries:

- ``schema_version``, ``stage``, ``id``: the envelope shared by all stages;
- ``meta``: who wrote the record and how (``stage``, ``code_git_sha``, ``created_at``, plus stage-specific params);
- ``eval`` (optional): every label-derived field. Method code must never read it; ``crux.io.read_jsonl`` strips
  it unless called with ``with_labels=True``.

Only the ``cases`` stage is defined so far. Later stages add a ``validate_<stage>`` function and an entry in
``STAGE_VALIDATORS`` when they are built.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List

SCHEMA_VERSION = "0.1"

LABEL_BLOCK = "eval"

# Keys that reveal or derive from the ground truth. Allowed only inside the top-level ``eval`` block.
FORBIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "ground_truth_text",
        "correct_choice",
        "correct_answer",
        "Correct Answer",
        "answer_letter",
        "rationale",
        "debug_meta",
        "raw_parse_source",
        "is_agreement_trap",
        "derived_crux_label",
        "initially_right",
        "exactly_one_right",
    }
)
FORBIDDEN_KEY_PATTERNS = (
    re.compile(r".*_correct$"),  # model_correct, a_correct, final_correct, ...
    re.compile(r"^Incorrect Answer"),
    re.compile(r"^raw_model_"),
)
# Keys that are legitimate when nested (a consultant's "answer") but leak at the top level
# (SuperGPQA's "answer" holds the correct option text).
FORBIDDEN_TOP_LEVEL_KEYS = frozenset({"answer"})

DATASETS = frozenset({"gpqa", "gpqa_diamond", "supergpqa"})
SOURCES = frozenset({"frontier_precomputed", "8b_run"})
SPLITS = frozenset({"seed", "unassigned", "dev", "test"})
CONSULTANTS = ("consultant_a", "consultant_b")

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def question_sha1(question: str) -> str:
    """Stable hash of a question for dedupe and joins: lowercased, whitespace collapsed."""
    normalized = " ".join(question.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def is_forbidden_key(key: str, *, top_level: bool = False) -> bool:
    if key in FORBIDDEN_KEYS or (top_level and key in FORBIDDEN_TOP_LEVEL_KEYS):
        return True
    return any(pattern.match(key) for pattern in FORBIDDEN_KEY_PATTERNS)


def find_label_leaks(record: Dict[str, Any]) -> List[str]:
    """Return the paths of label-revealing keys found anywhere outside the top-level ``eval`` block."""
    leaks: List[str] = []

    def walk(node: Any, path: str, top_level: bool) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}" if path else key
                if top_level and key == LABEL_BLOCK:
                    continue
                if is_forbidden_key(key, top_level=top_level):
                    leaks.append(child)
                walk(value, child, False)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                walk(item, f"{path}[{idx}]", False)

    walk(record, "", True)
    return leaks


def _check_choices(choices: Any, errors: List[str]) -> None:
    if not isinstance(choices, dict) or len(choices) < 2:
        errors.append("choices: must be a dict with at least 2 options")
        return
    expected = [chr(ord("A") + i) for i in range(len(choices))]
    if list(choices) != expected:
        errors.append(f"choices: labels must be {expected[0]}..{expected[-1]} in order, got {list(choices)}")
    for label, text in choices.items():
        if not isinstance(text, str) or not text.strip():
            errors.append(f"choices.{label}: must be a non-empty string")


def _check_consultant_init(name: str, block: Any, choices: Dict[str, Any], errors: List[str]) -> None:
    if not isinstance(block, dict):
        errors.append(f"{name}: must be an object")
        return
    for key in ("model", "answer"):
        if not isinstance(block.get(key), str) or not block.get(key):
            errors.append(f"{name}.{key}: must be a non-empty string")
    if not isinstance(block.get("answer_valid"), bool):
        errors.append(f"{name}.answer_valid: must be a bool")
    elif block["answer_valid"] and block.get("answer") not in choices:
        errors.append(f"{name}.answer: {block.get('answer')!r} is marked valid but is not a choice label")
    reasoning = block.get("reasoning")
    if not isinstance(reasoning, list) or not all(isinstance(step, str) for step in reasoning):
        errors.append(f"{name}.reasoning: must be a list of strings")
    if not isinstance(block.get("meta"), dict):
        errors.append(f"{name}.meta: must be an object")


def validate_cases(record: Dict[str, Any]) -> List[str]:
    """Stage ``cases``: one disagreement case with both consultants' initial answers."""
    errors: List[str] = []
    for key in ("dataset", "source", "split", "question", "question_sha1"):
        if not isinstance(record.get(key), str) or not record.get(key):
            errors.append(f"{key}: must be a non-empty string")
    if record.get("dataset") not in DATASETS:
        errors.append(f"dataset: {record.get('dataset')!r} not in {sorted(DATASETS)}")
    if record.get("source") not in SOURCES:
        errors.append(f"source: {record.get('source')!r} not in {sorted(SOURCES)}")
    if record.get("split") not in SPLITS:
        errors.append(f"split: {record.get('split')!r} not in {sorted(SPLITS)}")
    # Seed records are plumbing only and must never land in a kill-test split (CLAUDE.md).
    if (record.get("source") == "frontier_precomputed") != (record.get("split") == "seed"):
        errors.append("split: 'seed' is required for, and reserved to, source='frontier_precomputed'")
    if isinstance(record.get("question"), str) and record.get("question_sha1") != question_sha1(record["question"]):
        errors.append("question_sha1: does not match question")
    if not isinstance(record.get("provenance"), dict):
        errors.append("provenance: must be an object")

    choices = record.get("choices")
    _check_choices(choices, errors)
    choices = choices if isinstance(choices, dict) else {}
    for name in CONSULTANTS:
        _check_consultant_init(f"{name}_init", record.get(f"{name}_init"), choices, errors)

    labels = record.get(LABEL_BLOCK)
    if labels is not None:
        if not isinstance(labels, dict):
            errors.append(f"{LABEL_BLOCK}: must be an object")
        elif labels.get("ground_truth") not in choices:
            errors.append(f"{LABEL_BLOCK}.ground_truth: {labels.get('ground_truth')!r} is not a choice label")
    return errors


STAGE_VALIDATORS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "cases": validate_cases,
}


def validate_record(record: Any, stage: str) -> List[str]:
    """Validate one record for ``stage``. Returns a list of error strings; empty means valid."""
    if not isinstance(record, dict):
        return ["record: must be a JSON object"]
    if stage not in STAGE_VALIDATORS:
        return [f"stage: unknown stage {stage!r}; known: {sorted(STAGE_VALIDATORS)}"]

    errors: List[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION!r}, got {record.get('schema_version')!r}")
    if record.get("stage") != stage:
        errors.append(f"stage: expected {stage!r}, got {record.get('stage')!r}")
    if not isinstance(record.get("id"), str) or not record.get("id"):
        errors.append("id: must be a non-empty string")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta: must be an object")
    else:
        if meta.get("stage") != stage:
            errors.append(f"meta.stage: expected {stage!r}, got {meta.get('stage')!r}")
        for key in ("code_git_sha", "created_at"):
            if key not in meta:
                errors.append(f"meta.{key}: missing")

    errors.extend(f"label leak outside {LABEL_BLOCK}: {path}" for path in find_label_leaks(record))
    errors.extend(STAGE_VALIDATORS[stage](record))
    return errors
