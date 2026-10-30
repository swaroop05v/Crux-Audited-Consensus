"""Stage ``leakage``: flag cruxes that restate an answer option (Doc 2 Track 1, decisions D1).

- L1 wording (Doc 2): lowercase, strip punctuation and stopwords; flag if any option's text appears inside the crux,
  or if the crux covers more than ``overlap_threshold`` of an option's content words.
- L1 numbers (P4): flag if a quantity in the crux is within ``rel_tol`` (1%) of a quantity in a quantitative option,
  after unit normalization.
- P5 flags (multi-clause, deictic/meta-reference): logged only, never used to filter (decisions D1).
- L2 restatement probe (Prompt 1C, P4): run by both consultants at T=0; leaky if either says YES. Optional.

``is_leaky`` = L1 flagged, or L2 ran and flagged. If both probe replies are unusable (call error, truncated, or
no RESTATES line), ``is_leaky`` falls back to L1 and ``leak_probe_failed`` is true. Leaky cruxes are kept in the
file (Track 4 reports their d).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from crux.backends import Backend, Registry
from crux.io import index_by_id, read_jsonl, stage_meta, write_jsonl
from crux.prompts import Prompt, load_prompt
from crux.schema import SCHEMA_VERSION
from crux.seeding import stable_seed
from crux.track1.common import (CONSULTANT_ROLES, load_cases, make_backend, parallel_map, parse_outcome, run_path,
                                safe_chat, shared_call_params)
from crux.track1.quantities import Quantity, comparable, extract_quantities, relative_difference

STAGE = "leakage"
PROBE_PROMPT = "restatement_probe"
OVERLAP_THRESHOLD = 0.5
NUMERIC_REL_TOL = 0.01
MAX_NON_NUMERIC_TOKENS = 2   # options with more remaining content words are not treated as quantitative
DEFAULT_PROBE_MAX_TOKENS = 512

STOPWORDS = frozenset(
    "a an the of in on at to for from by with without and or but if then than as is are was were be been being "
    "it its this that these those which who whom whose what when where why how not no nor so such into onto over "
    "under between among about above below up down out off again further once all any both each few more most "
    "other some same only own very can will just should would could may might must do does did has have had "
    "there their they them he she his her we our you your i me my".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
_NUMBER_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?$")

MULTI_CLAUSE_RE = re.compile(r";|\b(and|or|but|because|since|whereas|while|although|therefore|thus|hence|"
                             r"which|whereby)\b", re.IGNORECASE)
DEICTIC_RE = re.compile(
    r"\b(the (reaction|compound|product|molecule|starting material|question|problem|options?|answers?|above|"
    r"given|scenario|system|mixture|experiment|figure|table|passage|following|other expert)|"
    r"this (reaction|compound|product|molecule|question|problem|scenario|system|step)|"
    r"(step|product|compound|reaction) \d+|(option|choice|answer) \(?[A-J]\)?|"
    r"above|consultant)\b",
    re.IGNORECASE,
)
FIRST_PERSON_RE = re.compile(r"\b(I|[Mm]y|[Ww]e|[Oo]ur)\b")  # case-sensitive "I"; rule 4 of Prompt 1A


def content_tokens(text: str, *, keep_numbers: bool) -> List[str]:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return tokens if keep_numbers else [t for t in tokens if not _NUMBER_TOKEN_RE.match(t)]


def _contains_sequence(haystack: List[str], needle: List[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _option_quantities(option: str) -> List[Quantity]:
    """Quantities of a quantitative option: one that is mostly numbers (e.g. "50,300 g/mole", "Both 50,300 g/mole
    and 50.3 kg/mole are correct"), not a name with locants or a list like "Patient 2, Patient 1, Patient 3"."""
    quantities = extract_quantities(option)
    if not quantities:
        return []
    remainder = option
    for quantity in quantities:
        remainder = remainder.replace(quantity.raw, " ")
    words = content_tokens(remainder, keep_numbers=False)
    return quantities if len(words) <= MAX_NON_NUMERIC_TOKENS else []


def l1_check(claim: str, choices: Dict[str, str], *, overlap_threshold: float = OVERLAP_THRESHOLD,
             rel_tol: float = NUMERIC_REL_TOL) -> Dict[str, Any]:
    claim_seq = content_tokens(claim, keep_numbers=True)
    claim_words = set(content_tokens(claim, keep_numbers=False))
    claim_quantities = extract_quantities(claim)

    contained, overlaps, numeric_matches = [], {}, []
    for label, option in choices.items():
        option_seq = content_tokens(option, keep_numbers=True)
        if _contains_sequence(claim_seq, option_seq):
            contained.append(label)
        option_words = set(content_tokens(option, keep_numbers=False))
        if option_words:
            overlaps[label] = len(option_words & claim_words) / len(option_words)
        for oq in _option_quantities(option):
            for cq in claim_quantities:
                if comparable(cq, oq) and relative_difference(cq.value, oq.value) <= rel_tol:
                    numeric_matches.append({"option": label, "crux_quantity": cq.raw, "option_quantity": oq.raw,
                                            "rel_diff": round(relative_difference(cq.value, oq.value), 6)})

    max_label = max(overlaps, key=overlaps.get) if overlaps else None
    max_overlap = overlaps[max_label] if max_label else 0.0
    reasons = []
    if contained:
        reasons.append("option_text_contained")
    if max_overlap > overlap_threshold:
        reasons.append("lexical_overlap")
    if numeric_matches:
        reasons.append("numeric_match")
    return {
        "flagged": bool(reasons),
        "reasons": reasons,
        "contained_options": contained,
        "lexical_overlap_max": round(max_overlap, 4),
        "lexical_overlap_option": max_label,
        "numeric_matches": numeric_matches,
    }


def p5_flags(claim: str) -> Dict[str, Any]:
    """Atomicity / self-containment flags (P5). Logged only."""
    multi = sorted({m.group(0).lower() for m in MULTI_CLAUSE_RE.finditer(claim)})
    deictic = sorted({m.group(0) for regex in (DEICTIC_RE, FIRST_PERSON_RE) for m in regex.finditer(claim)})
    return {"multi_clause": bool(multi), "multi_clause_matches": multi,
            "deictic": bool(deictic), "deictic_matches": deictic}


def parse_restates(text: Optional[str]) -> Optional[bool]:
    """The last ``RESTATES: YES|NO`` line, or None if absent."""
    matches = re.findall(r"RESTATES:\s*(YES|NO)\b", text or "", re.IGNORECASE)
    return matches[-1].upper() == "YES" if matches else None


def l2_probe(claim: str, case: Dict[str, Any], backend: Backend, registry: Registry, prompt: Prompt, *,
             crux_id: str, max_tokens: int, seed: int) -> Dict[str, Any]:
    answer_a, answer_b = case["consultant_a_init"]["answer"], case["consultant_b_init"]["answer"]
    messages = prompt.messages(answer_1=answer_a, choice_text_1=case["choices"][answer_a],
                               answer_2=answer_b, choice_text_2=case["choices"][answer_b], crux_claim=claim)
    by_consultant, calls = {}, {}
    for role in CONSULTANT_ROLES:
        call_seed = stable_seed(seed, STAGE, crux_id, role)
        outcome = safe_chat(backend, role, messages, **registry.sampling("leak_probe"), max_tokens=max_tokens,
                            seed=call_seed)
        parsed = parse_outcome(outcome, _parse_probe, {"restates": None})
        by_consultant[role] = {**parsed, **outcome.fields(), "seed": call_seed}
        calls[role] = shared_call_params(outcome.params)
    verdicts = [entry["restates"] for entry in by_consultant.values()]
    # Leaky if either consultant says YES (P4); clean only if both answered NO; otherwise undetermined.
    flagged = True if True in verdicts else (False if None not in verdicts else None)
    return {"flagged": flagged, "all_failed": all(v is None for v in verdicts), "by_consultant": by_consultant,
            "calls": calls, **prompt.meta()}


def _parse_probe(text: Optional[str]) -> Dict[str, Any]:
    restates = parse_restates(text)
    return {"restates": restates, "parse_ok": restates is not None,
            "parse_error": None if restates is not None else "no 'RESTATES: YES|NO' line"}


def check(proposals: List[Dict[str, Any]], cases_by_id: Dict[str, Dict[str, Any]], *,
          backend: Optional[Backend] = None, registry: Optional[Registry] = None, run_l2: bool = False,
          probe_max_tokens: int = DEFAULT_PROBE_MAX_TOKENS, seed: int = 42,
          overlap_threshold: float = OVERLAP_THRESHOLD, rel_tol: float = NUMERIC_REL_TOL,
          workers: int = 1) -> List[Dict[str, Any]]:
    if run_l2 and (backend is None or registry is None):
        raise ValueError("L2 needs a backend and a registry")
    prompt = load_prompt(PROBE_PROMPT, "v1") if run_l2 else None
    meta = stage_meta(STAGE, params={"overlap_threshold": overlap_threshold, "numeric_rel_tol": rel_tol,
                                     "l2": run_l2, "probe_max_tokens": probe_max_tokens if run_l2 else None,
                                     "base_seed": seed, "workers": workers})

    def one(proposal: Dict[str, Any]) -> Dict[str, Any]:
        record = {"schema_version": SCHEMA_VERSION, "stage": STAGE, "id": proposal["id"],
                  "crux_id": proposal["crux_id"], "proposed_by": proposal["proposed_by"]}
        claim = proposal.get("crux_claim")
        if not proposal.get("parse_ok") or not claim:
            return {**record, "status": "no_claim", "leak_probe_failed": None, "is_leaky": None, "meta": meta}
        case = cases_by_id[proposal["id"]]
        l1 = l1_check(claim, case["choices"], overlap_threshold=overlap_threshold, rel_tol=rel_tol)
        l2 = (l2_probe(claim, case, backend, registry, prompt, crux_id=proposal["crux_id"],
                       max_tokens=probe_max_tokens, seed=seed) if run_l2 else None)
        return {**record, "status": "checked", "crux_claim": claim, "l1": l1, "l2": l2,
                "p5_flags": p5_flags(claim),
                # If both probe replies are unusable, fall back to L1 and mark it so it can be counted.
                "leak_probe_failed": l2["all_failed"] if l2 else None,
                "is_leaky": l1["flagged"] or bool(l2 and l2["flagged"]), "meta": meta}

    return parallel_map(one, list(proposals), workers if run_l2 else 1)


def fake_responder(model_key: str, messages: List[Dict[str, str]], call_index: int) -> str:
    """Canned 1C reply for offline plumbing runs."""
    return "The statement is an intermediate step.\nRESTATES: NO"


def run_stage(*, cases_path: Path, run: str, backend_name: str = "fake", run_l2: bool = False,
              probe_max_tokens: int = DEFAULT_PROBE_MAX_TOKENS, seed: int = 42, workers: int = 1,
              registry: Optional[Registry] = None) -> Path:
    registry = registry or Registry.load()
    backend = make_backend(backend_name, registry, fake_responder) if run_l2 else None
    proposals = read_jsonl(run_path(run, "crux_proposals"), "crux_proposals")
    cases_by_id = index_by_id(load_cases(cases_path))
    records = check(proposals, cases_by_id, backend=backend, registry=registry, run_l2=run_l2,
                    probe_max_tokens=probe_max_tokens, seed=seed, workers=workers)
    for record in records:
        record["meta"] = {**record["meta"], "inputs": {"cases": str(cases_path),
                                                       "crux_proposals": str(run_path(run, "crux_proposals"))}}
    out = run_path(run, STAGE)
    write_jsonl(out, records, STAGE)
    leaky = sum(bool(record["is_leaky"]) for record in records)
    print(f"{STAGE}: wrote {len(records)} records to {out} ({leaky} leaky)")
    return out
