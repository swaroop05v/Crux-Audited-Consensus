"""Shared JSONL data contract: per-stage record validation and the label firewall.

Every pipeline stage writes its own JSONL file, one record per case (or per crux), joined on ``id`` / ``crux_id``
(docs/decisions.md D7). Each record carries:

- ``schema_version``, ``stage``, ``id``: the envelope shared by all stages;
- ``meta``: who wrote the record and how (``stage``, ``code_git_sha``, ``created_at``, plus stage-specific params);
- ``eval`` (optional): every label-derived field. Method code must never read it; ``crux.io.read_jsonl`` strips
  it unless called with ``with_labels=True``.

Defined stages: ``cases`` (keyed by ``id``) and the Track 1 stages ``crux_proposals``, ``leakage`` and
``decisiveness`` (keyed by ``crux_id`` = ``"{id}#a"`` / ``"{id}#b"``, the consultant who proposed it). Later stages
add a ``validate_<stage>`` function plus entries in ``STAGE_VALIDATORS`` and ``STAGE_KEYS`` when they are built.
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


UNDETERMINED = "UNDETERMINED"
_CRUX_ID_RE = re.compile(r"^(?P<id>.+)#(?P<side>[ab])$")


def _check_crux_ref(record: Dict[str, Any], errors: List[str]) -> None:
    """``crux_id`` is ``"{id}#a"`` or ``"{id}#b"`` and agrees with ``id`` and ``proposed_by``."""
    match = _CRUX_ID_RE.match(str(record.get("crux_id") or ""))
    if not match:
        errors.append(f"crux_id: {record.get('crux_id')!r} must look like '<id>#a' or '<id>#b'")
        return
    if match.group("id") != record.get("id"):
        errors.append("crux_id: prefix does not match id")
    if record.get("proposed_by") != f"consultant_{match.group('side')}":
        errors.append(f"proposed_by: must be consultant_{match.group('side')} for {record['crux_id']}")


def _check_optional(record: Dict[str, Any], key: str, types: tuple, errors: List[str]) -> None:
    if record.get(key) is not None and not isinstance(record.get(key), types):
        errors.append(f"{key}: must be {' or '.join(t.__name__ for t in types)} or null")


FAILURE_TYPES = (None, "call_error", "truncated", "parse_error")


def _check_failure(output: Dict[str, Any], path: str, errors: List[str]) -> None:
    """``failure_type`` is set exactly when ``parse_ok`` is false; ``finish_reason`` is a string or null."""
    failure = output.get("failure_type")
    if failure not in FAILURE_TYPES:
        errors.append(f"{path}failure_type: {failure!r} not in {list(FAILURE_TYPES)}")
    elif isinstance(output.get("parse_ok"), bool) and output["parse_ok"] != (failure is None):
        errors.append(f"{path}failure_type: must be null exactly when parse_ok is true")
    if output.get("finish_reason") is not None and not isinstance(output["finish_reason"], str):
        errors.append(f"{path}finish_reason: must be a string or null")


def _check_mapping(value: Any, path: str, errors: List[str]) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"if_true", "if_false"}:
        errors.append(f"{path}: must be null or {{if_true, if_false}}")


def validate_crux_proposals(record: Dict[str, Any]) -> List[str]:
    """Stage ``crux_proposals``: one Prompt 1A output per consultant per case."""
    errors: List[str] = []
    _check_crux_ref(record, errors)
    if not isinstance(record.get("parse_ok"), bool):
        errors.append("parse_ok: must be a bool")
    elif record["parse_ok"] and not (isinstance(record.get("crux_claim"), str) and record["crux_claim"].strip()):
        errors.append("crux_claim: must be a non-empty string when parse_ok")
    elif not record["parse_ok"] and record.get("crux_claim") is not None:
        errors.append("crux_claim: must be null when not parse_ok")
    for key in ("raw_output", "call_error", "parse_error", "disagreement_analysis"):
        _check_optional(record, key, (str,), errors)
    _check_optional(record, "proposer_own_verdict", (bool,), errors)
    _check_failure(record, "", errors)
    return errors


def validate_leakage(record: Dict[str, Any]) -> List[str]:
    """Stage ``leakage``: L1 wording/number check, P5 flags (log only), optional L2 restatement probe."""
    errors: List[str] = []
    _check_crux_ref(record, errors)
    status = record.get("status")
    if status == "checked":
        for key in ("l1", "p5_flags"):
            if not isinstance(record.get(key), dict):
                errors.append(f"{key}: must be an object")
        if isinstance(record.get("l1"), dict) and not isinstance(record["l1"].get("flagged"), bool):
            errors.append("l1.flagged: must be a bool")
        _check_optional(record, "l2", (dict,), errors)
        if not isinstance(record.get("is_leaky"), bool):
            errors.append("is_leaky: must be a bool")
        probe_failed = record.get("leak_probe_failed")
        if record.get("l2") is None and probe_failed is not None:
            errors.append("leak_probe_failed: must be null when L2 did not run")
        if record.get("l2") is not None and not isinstance(probe_failed, bool):
            errors.append("leak_probe_failed: must be a bool when L2 ran")
    elif status == "no_claim":
        if record.get("is_leaky") is not None:
            errors.append("is_leaky: must be null when status is no_claim")
    else:
        errors.append(f"status: {status!r} not in ['checked', 'no_claim']")
    return errors


def validate_decisiveness(record: Dict[str, Any]) -> List[str]:
    """Stage ``decisiveness``: per-consultant Prompt 1B samples plus the d / veto summary (P1, P6)."""
    errors: List[str] = []
    _check_crux_ref(record, errors)
    status = record.get("status")
    if status == "no_claim":
        return errors
    if status != "evaluated":
        return errors + [f"status: {status!r} not in ['evaluated', 'no_claim']"]
    k = record.get("k")
    if not isinstance(k, int) or k < 1:
        return errors + ["k: must be an int >= 1"]
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) != 2 * k:
        errors.append(f"samples: must be a list of 2*k = {2 * k} per-consultant samples")
        samples = []
    for position, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("consultant") not in CONSULTANTS:
            errors.append(f"samples[{position}]: consultant must be one of {CONSULTANTS}")
            continue
        if not isinstance(sample.get("parse_ok"), bool):
            errors.append(f"samples[{position}].parse_ok: must be a bool")
        _check_failure(sample, f"samples[{position}].", errors)
        order = sample.get("answer_order")
        if not (isinstance(order, list) and len(order) == 2):
            errors.append(f"samples[{position}].answer_order: must list the two answers in prompt order")
        for branch in ("if_true", "if_false"):
            value = sample.get(branch)
            if value is not None and value != UNDETERMINED and (not isinstance(order, list) or value not in order):
                errors.append(f"samples[{position}].{branch}: {value!r} is not an answer in play or UNDETERMINED")
    for key in ("d", "d_a", "d_b", "flip_rate_a", "flip_rate_b"):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            errors.append(f"{key}: must be a number in [0, 1]")
    _check_mapping(record.get("agreed_mapping"), "agreed_mapping", errors)
    if not isinstance(record.get("veto_flag"), bool):
        errors.append("veto_flag: must be a bool")
    return errors


STAGE_VALIDATORS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "cases": validate_cases,
    "crux_proposals": validate_crux_proposals,
    "leakage": validate_leakage,
    "decisiveness": validate_decisiveness,
}

# The field each stage file is keyed by (unique per file); ``id`` always joins back to ``cases``.
STAGE_KEYS: Dict[str, str] = {
    "cases": "id",
    "crux_proposals": "crux_id",
    "leakage": "crux_id",
    "decisiveness": "crux_id",
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
