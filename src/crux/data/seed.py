"""Build seed_30.jsonl (stage ``cases``) from the paper's precomputed frontier answers in collaborative-dr/data.

Plumbing data only: every record has ``source = "frontier_precomputed"`` and ``split = "seed"`` and must never
enter kill-test numbers. Only whitelisted fields are copied; labels go only into ``eval``.

    python -m crux.data.seed --out data/seed/seed_30.jsonl --per-dataset 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crux.data.choices import ChoiceParseError, parse_choices
from crux.io import REPO_ROOT, git_sha, stage_meta, write_jsonl
from crux.schema import SCHEMA_VERSION, question_sha1

BASE_REPO = REPO_ROOT / "collaborative-dr"
STAGE = "cases"

# Pair A only (docs/decisions.md D8). File order fixes the roles: consultant_a = GPT-4o, consultant_b = Claude.
PAIR_A = (
    ("consultant_a", "gpt_4o.json", "openai/gpt-4o"),
    ("consultant_b", "claude.json", "anthropic/claude-sonnet-4"),
)

# Per dataset: source folder, question key, key holding the correct option's text (used only as a sanity check),
# and whitelisted metadata copied into ``domain``.
DATASETS = {
    "gpqa": {"dir": "GPQA", "question_key": "Question", "answer_text_key": "Correct Answer",
             "domain_keys": {"domain": "High-level domain"}},
    "supergpqa": {"dir": "SuperGPQA", "question_key": "Question", "answer_text_key": "answer",
                  "domain_keys": {"discipline": "discipline", "field": "field", "subfield": "subfield"}},
}

# Matches Track 3's kill-test filter (Doc 2 Track 3 step 1).
EXCLUDED_DISCIPLINES = {"Medicine"}


def _load(dataset_dir: str, filename: str) -> Dict[int, Dict[str, Any]]:
    path = BASE_REPO / "data" / dataset_dir / filename
    with path.open("r", encoding="utf-8") as handle:
        return {entry["index"]: entry for entry in json.load(handle)}


def _domain(entry: Dict[str, Any], domain_keys: Dict[str, str]) -> Dict[str, Optional[str]]:
    domain = {}
    for out_key, src_key in domain_keys.items():
        value = entry.get(src_key)
        domain[out_key] = None if value in (None, "", "Unknown") else str(value)
    return domain


def build_case(dataset: str, index: int, entries: Dict[str, Dict[str, Any]]) -> Tuple[Optional[Dict], str]:
    """Return (record, "") for a usable case, or (None, reason) when it is skipped."""
    spec = DATASETS[dataset]
    a, b = entries["consultant_a"], entries["consultant_b"]

    question = str(a.get(spec["question_key"]) or "").strip()
    if not question or question != str(b.get(spec["question_key"]) or "").strip():
        return None, "question missing or differs between files"
    if a.get("choices") != b.get("choices"):
        return None, "choices differ between files"
    try:
        choices = parse_choices(a["choices"])
    except ChoiceParseError:
        return None, "choices unparseable"

    ground_truth = str(a.get("correct_choice") or "").strip().upper()
    if ground_truth not in choices or ground_truth != str(b.get("correct_choice") or "").strip().upper():
        return None, "ground truth missing or inconsistent"
    answer_text = " ".join(str(a.get(spec["answer_text_key"]) or "").split())
    if answer_text != " ".join(choices[ground_truth].split()):
        return None, "ground-truth text does not match parsed option"
    if dataset == "supergpqa" and a.get("discipline") in EXCLUDED_DISCIPLINES:
        return None, "excluded discipline"

    inits = {}
    correct = {}
    for role, filename, model in PAIR_A:
        entry = entries[role]
        answer = str(entry.get("model_answer") or "").strip().upper()
        if answer not in choices:
            return None, "invalid model answer"
        correct[role] = answer == ground_truth
        inits[f"{role}_init"] = {
            "model": model,
            "answer": answer,
            "answer_valid": True,
            "reasoning": [str(step).strip() for step in entry.get("model_reasoning") or [] if str(step).strip()],
            "meta": {"backend": "frontier_precomputed", "source_file": f"collaborative-dr/data/{spec['dir']}/{filename}"},
        }
    if inits["consultant_a_init"]["answer"] == inits["consultant_b_init"]["answer"] or sum(correct.values()) != 1:
        return None, "not a disagreement with exactly one right"

    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "id": f"{dataset}:pairA:{index}",
        "dataset": dataset,
        "source": "frontier_precomputed",
        "split": "seed",
        "question": question,
        "question_sha1": question_sha1(question),
        "choices": choices,
        "domain": _domain(a, spec["domain_keys"]),
        "provenance": {"base_repo": "collaborative-dr", "dataset_dir": spec["dir"], "pair": "A", "index": index},
        **inits,
        "eval": {
            "ground_truth": ground_truth,
            "ground_truth_text": choices[ground_truth],
            "a_correct": correct["consultant_a"],
            "b_correct": correct["consultant_b"],
            "initially_right": "a" if correct["consultant_a"] else "b",
        },
    }
    return record, ""


def build_seed(per_dataset: int, seed: int) -> Tuple[List[Dict[str, Any]], Counter]:
    rng = random.Random(seed)
    skipped: Counter = Counter()
    selected: List[Dict[str, Any]] = []
    seen_questions = set()
    for dataset, spec in DATASETS.items():
        files = {role: _load(spec["dir"], filename) for role, filename, _ in PAIR_A}
        by_side: Dict[str, List[Dict[str, Any]]] = {"a": [], "b": []}
        for index in sorted(set(files["consultant_a"]) & set(files["consultant_b"])):
            record, reason = build_case(dataset, index, {role: files[role][index] for role in files})
            if record is None:
                skipped[f"{dataset}: {reason}"] += 1
            elif record["question_sha1"] in seen_questions:
                skipped[f"{dataset}: duplicate question"] += 1
            else:
                seen_questions.add(record["question_sha1"])
                by_side[record["eval"]["initially_right"]].append(record)
        # Balanced: half the cases where consultant A is initially right, half where B is.
        for side in ("a", "b"):
            want = per_dataset // 2 + (per_dataset % 2 if side == "a" else 0)
            if len(by_side[side]) < want:
                raise SystemExit(f"{dataset}: only {len(by_side[side])} usable cases with {side} right, need {want}")
            selected.extend(rng.sample(by_side[side], want))
    selected.sort(key=lambda record: (record["dataset"], record["provenance"]["index"]))
    return selected, skipped


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "seed" / "seed_30.jsonl")
    parser.add_argument("--per-dataset", type=int, default=20, help="Cases per dataset (GPQA, SuperGPQA).")
    parser.add_argument("--seed", type=int, default=42)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    run(parser.parse_args(argv))


def run(args: argparse.Namespace) -> None:

    records, skipped = build_seed(args.per_dataset, args.seed)
    meta = stage_meta(
        STAGE,
        builder="crux.data.seed",
        params={"per_dataset": args.per_dataset, "seed": args.seed, "pair": "A",
                "excluded_disciplines": sorted(EXCLUDED_DISCIPLINES)},
        base_repo_git_sha=git_sha(BASE_REPO),
    )
    for record in records:
        record["meta"] = meta
    count = write_jsonl(args.out, records, STAGE)

    print(f"Wrote {count} records to {args.out}")
    print("By dataset / initially right:", dict(Counter((r["dataset"], r["eval"]["initially_right"]) for r in records)))
    for reason, n in sorted(skipped.items()):
        print(f"  skipped {n:3d}  {reason}")


if __name__ == "__main__":
    main()
