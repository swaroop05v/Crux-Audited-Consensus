"""Stage ``crux_proposals``: each consultant proposes one crux per case with Prompt 1A (crux_proposal).

Consultant A argues from consultant_a_init's side, B from consultant_b_init's. One record per (case, consultant),
keyed by ``crux_id = "{id}#a" | "{id}#b"``. Prompt v2 adds ``own_verdict`` (audit 3.2 item 10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from crux.backends import Backend, Registry
from crux.data.choices import format_choices
from crux.io import stage_meta, write_jsonl
from crux.prompts import Prompt, load_prompt
from crux.schema import SCHEMA_VERSION
from crux.seeding import stable_seed
from crux.track1.common import (CONSULTANT_ROLES, format_reasoning, load_cases, make_backend, other_role,
                                parse_json_object, run_path, safe_chat, side)

STAGE = "crux_proposals"
PROMPT_NAME = "crux_proposal"
DEFAULT_PROMPT_VERSION = "v2"
DEFAULT_MAX_TOKENS = 1024


def proposal_messages(case: Dict[str, Any], role: str, prompt: Prompt) -> List[Dict[str, str]]:
    mine, theirs = case[f"{role}_init"], case[f"{other_role(role)}_init"]
    fields = {
        "my_answer": mine["answer"],
        "other_answer": theirs["answer"],
        "question": case["question"],
        "choices_formatted": format_choices(case["choices"]),
        "my_reasoning": format_reasoning(mine["reasoning"]),
        "other_reasoning": format_reasoning(theirs["reasoning"]),
    }
    return prompt.messages(**fields)


def _verdict(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().upper()
    if text.startswith("TRUE"):
        return True
    if text.startswith("FALSE"):
        return False
    return None


def parse_proposal(text: Optional[str]) -> Dict[str, Any]:
    """Parsed fields. ``parse_ok`` needs a non-empty ``crux_claim``; a missing own_verdict is not a failure."""
    parsed, error = parse_json_object(text)
    claim = parsed.get("crux_claim") if parsed else None
    if parsed is not None and not (isinstance(claim, str) and claim.strip()):
        error = "crux_claim missing or empty"
    ok = error is None
    analysis = parsed.get("disagreement_analysis") if parsed else None
    return {
        "parse_ok": ok,
        "parse_error": error,
        "crux_claim": claim.strip() if ok else None,
        "disagreement_analysis": analysis.strip() if isinstance(analysis, str) else None,
        "proposer_own_verdict": _verdict(parsed.get("own_verdict")) if parsed else None,
    }


def propose(cases: List[Dict[str, Any]], backend: Backend, registry: Registry, *,
            prompt_version: str = DEFAULT_PROMPT_VERSION, max_tokens: int = DEFAULT_MAX_TOKENS,
            seed: int = 42) -> List[Dict[str, Any]]:
    prompt = load_prompt(PROMPT_NAME, prompt_version)
    sampling = registry.sampling("consultant")
    records = []
    for case in cases:
        for role in CONSULTANT_ROLES:
            call_seed = stable_seed(seed, STAGE, case["id"], role)
            text, params, call_error = safe_chat(backend, role, proposal_messages(case, role, prompt),
                                                 **sampling, max_tokens=max_tokens, seed=call_seed)
            parsed = parse_proposal(text) if call_error is None else {
                "parse_ok": False, "parse_error": "call failed", "crux_claim": None,
                "disagreement_analysis": None, "proposer_own_verdict": None}
            records.append({
                "schema_version": SCHEMA_VERSION,
                "stage": STAGE,
                "id": case["id"],
                "crux_id": f"{case['id']}#{side(role)}",
                "proposed_by": role,
                **parsed,
                "raw_output": text,
                "call_error": call_error,
                "seed": call_seed,
                "meta": stage_meta(STAGE, **prompt.meta(), call=params,
                                   params={"max_tokens": max_tokens, "base_seed": seed}),
            })
    return records


def fake_responder(model_key: str, messages: List[Dict[str, str]], call_index: int) -> str:
    """Canned 1A reply for offline plumbing runs (``--backend fake``)."""
    return json.dumps({
        "disagreement_analysis": f"Fake analysis by {model_key}.",
        "crux_claim": f"Fake self-contained claim number {call_index} proposed by {model_key}.",
        "own_verdict": "TRUE",
    })


def run_stage(*, cases_path: Path, run: str, backend_name: str, prompt_version: str = DEFAULT_PROMPT_VERSION,
              max_tokens: int = DEFAULT_MAX_TOKENS, seed: int = 42, limit: Optional[int] = None,
              registry: Optional[Registry] = None) -> Path:
    registry = registry or Registry.load()
    backend = make_backend(backend_name, registry, fake_responder)
    records = propose(load_cases(cases_path, limit), backend, registry, prompt_version=prompt_version,
                      max_tokens=max_tokens, seed=seed)
    for record in records:
        record["meta"]["inputs"] = {"cases": str(cases_path)}
    out = run_path(run, STAGE)
    write_jsonl(out, records, STAGE)
    ok = sum(record["parse_ok"] for record in records)
    print(f"{STAGE}: wrote {len(records)} records to {out} ({ok} parsed, {len(records) - ok} failed)")
    return out
