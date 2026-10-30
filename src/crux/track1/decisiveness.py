"""Stage ``decisiveness``: does settling the crux settle the answer, for both consultants? (Doc 2 Prompt 1B)

Each consultant answers Prompt 1B (v2) K times without being told who proposed the crux. P6: the two answers in
play are listed in a random order per consultant per sample, and each branch may be UNDETERMINED.

Per-consultant samples are stored as they came back (not pre-paired). From them:

- a sample is *decisive* if both branches name an answer in play and the two differ; its mapping is
  (if_true, if_false);
- ``flip_rate_x``: share of consultant x's K samples that are decisive;
- ``mapping_x`` / ``d_x``: x's majority mapping and the share of x's samples that give it;
- all K x K cross-pairs (a_i, b_j) pass when both are decisive with the same mapping; ``agreed_mapping`` is the
  most common passing mapping (null on a tie), and ``d`` = pairs passing with it / K^2;
- ``mapping_agree_rate``: among pairs where both are decisive, the share with equal mappings;
- P1 ``veto_flag``: exactly one consultant is decisive on a majority of samples (flip rate >= ``veto_threshold``),
  so that consultant alone sinks d. ``vetoed_by`` names the other one. Provisional: flag only (decisions D1).
- ``mapping_conflict``: both are decisive but their majority mappings disagree.
- per-branch evidence (``d_true``, ``d_false``, ``flip_rate_{a,b}_{true,false}``): see ``branch_summary``. Not used
  by the gate until the team decides on the decisiveness definition.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crux.backends import Backend, Registry
from crux.data.choices import format_choices
from crux.io import index_by_id, read_jsonl, stage_meta, write_jsonl
from crux.prompts import Prompt, load_prompt
from crux.schema import SCHEMA_VERSION, UNDETERMINED
from crux.seeding import stable_seed
from crux.track1.common import (CONSULTANT_ROLES, failure_summary, load_cases, make_backend, parallel_map,
                                parse_json_object, parse_outcome, run_path, safe_chat, shared_call_params, side)

STAGE = "decisiveness"
PROMPT_NAME = "decisiveness"
DEFAULT_PROMPT_VERSION = "v2"
DEFAULT_K = 3
DEFAULT_MAX_TOKENS = 1024  # was 768: seed30_v1 truncated 10/480 samples (2.1%), all Llama
VETO_THRESHOLD = 0.5

Mapping = Tuple[str, str]


def answer_order(base_seed: int, crux_id: str, role: str, k: int, pair: Tuple[str, str]) -> List[str]:
    """P6: which answer is listed first, drawn independently per consultant per sample."""
    rng = random.Random(stable_seed(base_seed, STAGE, "order", crux_id, role, k))
    return list(pair) if rng.random() < 0.5 else [pair[1], pair[0]]


def sample_messages(case: Dict[str, Any], claim: str, order: List[str], prompt: Prompt) -> List[Dict[str, str]]:
    return prompt.messages(question=case["question"], choices_formatted=format_choices(case["choices"]),
                           answer_1=order[0], answer_2=order[1], crux_claim=claim)


_LETTER_RE = re.compile(r"^\(?(?:OPTION|ANSWER|CHOICE)?\s*\(?([A-Z])\)?[.:]?$")


def normalize_branch(value: Any, pair: Tuple[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """(answer, error): an answer in ``pair``, UNDETERMINED, or an error for anything else."""
    text = str(value or "").strip().upper()
    if not text:
        return None, "missing"
    if re.fullmatch(r"UNDETERMINED[.!]?", text):  # not a substring test: an echoed template must not count
        return UNDETERMINED, None
    match = _LETTER_RE.match(text)
    letter = match.group(1) if match else text
    if letter in pair:
        return letter, None
    return None, f"{value!r} is not one of {list(pair)} or UNDETERMINED"


def parse_sample(text: Optional[str], pair: Tuple[str, str]) -> Dict[str, Any]:
    parsed, error = parse_json_object(text)
    branches: Dict[str, Optional[str]] = {"if_true": None, "if_false": None}
    if parsed is not None:
        errors = []
        for branch, key in (("if_true", "answer_if_true"), ("if_false", "answer_if_false")):
            branches[branch], branch_error = normalize_branch(parsed.get(key), pair)
            if branch_error:
                errors.append(f"{key}: {branch_error}")
        error = "; ".join(errors) or None
    return {"parse_ok": error is None, "parse_error": error, **branches}


def sample_mapping(sample: Dict[str, Any], pair: Tuple[str, str]) -> Optional[Mapping]:
    """The (if_true, if_false) mapping of a decisive sample, else None."""
    if not sample.get("parse_ok"):
        return None
    if_true, if_false = sample.get("if_true"), sample.get("if_false")
    if if_true in pair and if_false in pair and if_true != if_false:
        return if_true, if_false
    return None


def _unique_top(counter: Counter) -> Tuple[Optional[Mapping], bool]:
    """Most common key, or (None, True) on a tie for first place."""
    ranked = counter.most_common()
    if not ranked:
        return None, False
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, True
    return ranked[0][0], False


def _as_dict(mapping: Optional[Mapping]) -> Optional[Dict[str, str]]:
    return {"if_true": mapping[0], "if_false": mapping[1]} if mapping else None


def summarize(samples: List[Dict[str, Any]], pair: Tuple[str, str], proposed_by: str, k: int,
              veto_threshold: float = VETO_THRESHOLD) -> Dict[str, Any]:
    mappings = {role: [sample_mapping(s, pair) for s in sorted((s for s in samples if s["consultant"] == role),
                                                                  key=lambda s: s["k"])]
                for role in CONSULTANT_ROLES}
    summary: Dict[str, Any] = {}
    majority: Dict[str, Optional[Mapping]] = {}
    for role in CONSULTANT_ROLES:
        decisive = [m for m in mappings[role] if m is not None]
        majority[role], _ = _unique_top(Counter(decisive))
        summary[f"flip_rate_{side(role)}"] = len(decisive) / k
        summary[f"d_{side(role)}"] = (sum(m == majority[role] for m in decisive) / k) if majority[role] else 0.0
        summary[f"mapping_{side(role)}"] = _as_dict(majority[role])

    both_decisive = [(ma, mb) for ma in mappings["consultant_a"] for mb in mappings["consultant_b"]
                     if ma is not None and mb is not None]
    passing = Counter(ma for ma, mb in both_decisive if ma == mb)
    agreed, tie = _unique_top(passing)
    summary.update({
        "n_pairs": k * k,
        "passing_pairs": sum(passing.values()),
        "agreed_mapping": _as_dict(agreed),
        "mapping_tie": tie,
        "d": passing[agreed] / (k * k) if agreed else 0.0,
        "mapping_agree_rate": (sum(ma == mb for ma, mb in both_decisive) / len(both_decisive)
                               if both_decisive else None),
    })

    decisive_a = summary["flip_rate_a"] >= veto_threshold
    decisive_b = summary["flip_rate_b"] >= veto_threshold
    veto = decisive_a != decisive_b
    vetoed_by = ("consultant_b" if decisive_a else "consultant_a") if veto else None
    summary.update({
        "veto_flag": veto,
        "vetoed_by": vetoed_by,
        "veto_by_proposer": (vetoed_by == proposed_by) if veto else None,
        "mapping_conflict": bool(decisive_a and decisive_b and majority["consultant_a"]
                                 and majority["consultant_b"]
                                 and majority["consultant_a"] != majority["consultant_b"]),
    })
    summary.update(branch_summary(samples, pair, k))
    return summary


BRANCHES = ("true", "false")


def branch_answer(sample: Dict[str, Any], pair: Tuple[str, str], branch: str) -> Optional[str]:
    """The answer a sample gives on one branch, counted only if the crux matters there: a definite answer in play
    that the other branch does not repeat (the other branch is the other answer or UNDETERMINED)."""
    if not sample.get("parse_ok"):
        return None
    this = sample.get(f"if_{branch}")
    other = sample.get("if_false" if branch == "true" else "if_true")
    return this if this in pair and other != this else None


def branch_summary(samples: List[Dict[str, Any]], pair: Tuple[str, str], k: int) -> Dict[str, Any]:
    """Per-branch decisiveness, as evidence for the team's decision (the gate still uses the two-way ``d``).

    seed30_v1 showed most cruxes are one-way ("if TRUE then X; if FALSE, undetermined"). For each branch:
    ``flip_rate_{a,b}_{branch}`` = share of that consultant's K samples where the branch counts (``branch_answer``);
    ``d_{branch}`` = share of the K x K cross-pairs where both consultants name the same answer on that branch, for
    the most common such answer (``agreed_if_{branch}``; 0 and null on a tie). ``branch_conflict``: both branches
    agree on the same answer, which contradicts itself.
    """
    out: Dict[str, Any] = {}
    for branch in BRANCHES:
        answers = {role: [branch_answer(s, pair, branch)
                          for s in sorted((s for s in samples if s["consultant"] == role), key=lambda s: s["k"])]
                   for role in CONSULTANT_ROLES}
        for role in CONSULTANT_ROLES:
            out[f"flip_rate_{side(role)}_{branch}"] = sum(a is not None for a in answers[role]) / k
        passing = Counter(x for x in answers["consultant_a"] for y in answers["consultant_b"]
                          if x is not None and x == y)
        agreed, _ = _unique_top(passing)
        out[f"d_{branch}"] = passing[agreed] / (k * k) if agreed else 0.0
        out[f"agreed_if_{branch}"] = agreed
    out["branch_conflict"] = bool(out["agreed_if_true"] and out["agreed_if_true"] == out["agreed_if_false"])
    return out


def rescore(records: List[Dict[str, Any]], *, reparse: bool = False) -> Tuple[List[Dict[str, Any]], int]:
    """Recompute every summary field from the stored samples, with no model calls. With ``reparse``, complete
    replies (not truncated, not failed calls) are parsed again with the current parser first. Returns the new
    records and how many samples changed parse result."""
    changed = 0
    out = []
    for record in records:
        record = json.loads(json.dumps(record))  # deep copy
        if record.get("status") == "evaluated":
            pair = tuple(record["answers_in_play"])
            if reparse:
                for sample in record["samples"]:
                    if sample.get("failure_type") in (None, "parse_error") and sample.get("raw_output"):
                        parsed = parse_sample(sample["raw_output"], pair)
                        parsed["failure_type"] = None if parsed["parse_ok"] else "parse_error"
                        changed += parsed["parse_ok"] != sample["parse_ok"]
                        sample.update(parsed)
            threshold = (record["meta"].get("params") or {}).get("veto_threshold", VETO_THRESHOLD)
            record.update(summarize(record["samples"], pair, record["proposed_by"], record["k"], threshold))
        record["meta"] = {**record["meta"], "rescored": stage_meta(STAGE, reparse=reparse)}
        out.append(record)
    return out, changed


def run_rescore(*, run: str, reparse: bool = False) -> Path:
    """``crux track1 rescore``: rewrite decisiveness.jsonl with recomputed summaries; the original is kept once
    as decisiveness.pre_rescore.jsonl."""
    path = run_path(run, STAGE)
    backup = path.with_name("decisiveness.pre_rescore.jsonl")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    records, changed = rescore(read_jsonl(path, STAGE), reparse=reparse)
    write_jsonl(path, records, STAGE)
    print(f"{STAGE}: rescored {len(records)} records in {path} (reparse={reparse}, {changed} samples changed "
          f"parse result; original kept at {backup.name})")
    return path


def evaluate_crux(proposal: Dict[str, Any], case: Dict[str, Any], backend: Backend, registry: Registry,
                  prompt: Prompt, *, k: int, max_tokens: int, seed: int,
                  veto_threshold: float = VETO_THRESHOLD) -> Dict[str, Any]:
    pair = (case["consultant_a_init"]["answer"], case["consultant_b_init"]["answer"])
    samples, calls = [], {}
    for role in CONSULTANT_ROLES:
        for index in range(k):
            order = answer_order(seed, proposal["crux_id"], role, index, pair)
            call_seed = stable_seed(seed, STAGE, proposal["crux_id"], role, index)
            outcome = safe_chat(backend, role, sample_messages(case, proposal["crux_claim"], order, prompt),
                                **registry.sampling("consultant"), max_tokens=max_tokens, seed=call_seed)
            parsed = parse_outcome(outcome, lambda text: parse_sample(text, pair),
                                   {"if_true": None, "if_false": None})
            samples.append({"consultant": role, "k": index, "answer_order": order, "seed": call_seed,
                            **parsed, **outcome.fields()})
            if outcome.params is not None:
                calls.setdefault(role, shared_call_params(outcome.params))
    return {"status": "evaluated", "k": k, "answers_in_play": list(pair), "samples": samples,
            **summarize(samples, pair, proposal["proposed_by"], k, veto_threshold),
            "calls": calls}


def evaluate(proposals: List[Dict[str, Any]], cases_by_id: Dict[str, Dict[str, Any]], backend: Backend,
             registry: Registry, *, k: int = DEFAULT_K, prompt_version: str = DEFAULT_PROMPT_VERSION,
             max_tokens: int = DEFAULT_MAX_TOKENS, seed: int = 42,
             veto_threshold: float = VETO_THRESHOLD, workers: int = 1) -> List[Dict[str, Any]]:
    """Every parsed proposal is evaluated, leaky or not (Track 4 reports d for leaky cruxes too).
    With ``workers`` > 1, cruxes are evaluated concurrently; each crux's 2K samples stay sequential."""
    prompt = load_prompt(PROMPT_NAME, prompt_version)

    def one(proposal: Dict[str, Any]) -> Dict[str, Any]:
        record = {"schema_version": SCHEMA_VERSION, "stage": STAGE, "id": proposal["id"],
                  "crux_id": proposal["crux_id"], "proposed_by": proposal["proposed_by"]}
        if not proposal.get("parse_ok") or not proposal.get("crux_claim"):
            result: Dict[str, Any] = {"status": "no_claim"}
            calls = None
        else:
            result = evaluate_crux(proposal, cases_by_id[proposal["id"]], backend, registry, prompt, k=k,
                                   max_tokens=max_tokens, seed=seed, veto_threshold=veto_threshold)
            calls = result.pop("calls")
        return {**record, **result,
                "meta": stage_meta(STAGE, **prompt.meta(), calls=calls,
                                   params={"k": k, "max_tokens": max_tokens, "base_seed": seed,
                                           "veto_threshold": veto_threshold, "workers": workers})}

    return parallel_map(one, list(proposals), workers)


_CANDIDATES_RE = re.compile(r"\[CANDIDATE ANSWERS\]\s*\n- (\S+)\s*\n- (\S+)")


def fake_responder(model_key: str, messages: List[Dict[str, str]], call_index: int) -> str:
    """Canned 1B reply for offline plumbing: TRUE maps to the alphabetically first answer, whatever the order."""
    match = _CANDIDATES_RE.search(messages[-1]["content"])
    first, second = sorted(match.groups()) if match else ("A", "B")
    return json.dumps({"reasoning_if_true": "fake", "answer_if_true": first,
                       "reasoning_if_false": "fake", "answer_if_false": second})


def run_stage(*, cases_path: Path, run: str, backend_name: str, k: int = DEFAULT_K,
              prompt_version: str = DEFAULT_PROMPT_VERSION, max_tokens: int = DEFAULT_MAX_TOKENS, seed: int = 42,
              workers: int = 1, registry: Optional[Registry] = None) -> Path:
    registry = registry or Registry.load()
    backend = make_backend(backend_name, registry, fake_responder)
    proposals = read_jsonl(run_path(run, "crux_proposals"), "crux_proposals")
    records = evaluate(proposals, index_by_id(load_cases(cases_path)), backend, registry, k=k,
                       prompt_version=prompt_version, max_tokens=max_tokens, seed=seed, workers=workers)
    for record in records:
        record["meta"]["inputs"] = {"cases": str(cases_path),
                                    "crux_proposals": str(run_path(run, "crux_proposals"))}
    out = run_path(run, STAGE)
    write_jsonl(out, records, STAGE)
    evaluated = [r for r in records if r["status"] == "evaluated"]
    vetoes = sum(r["veto_flag"] for r in evaluated)
    mean_d = sum(r["d"] for r in evaluated) / len(evaluated) if evaluated else 0.0
    samples = [s for r in evaluated for s in r["samples"]]
    print(f"{STAGE}: wrote {len(records)} records to {out} ({len(evaluated)} evaluated, "
          f"mean d {mean_d:.2f}, {vetoes} veto flags; samples: {failure_summary(samples)})")
    return out
