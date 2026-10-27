
"""Weak-judge utilities for Track 2."""

from __future__ import annotations

import re
from typing import Any, Dict, Literal

from crux.backends.base import Backend, Registry
from crux.io import stage_meta
from crux.prompts import load_prompt

Verdict = Literal["TRUE", "FALSE"]

_VERDICT_RE = re.compile(r"(?m)^VERDICT:\s*(TRUE|FALSE)\s*$")

_FINAL_ANSWER_RE = re.compile(
    r"(?m)^FINAL ANSWER:\s*([A-Z])\s*$"
)


def parse_final_answer(completion: str, answer_1: str, answer_2: str) -> str:
    """Extract one final answer and ensure it matches one of the two candidates."""
    matches = _FINAL_ANSWER_RE.findall(completion)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one 'FINAL ANSWER: X' line; found {len(matches)}."
        )

    answer = matches[0]
    allowed = {answer_1, answer_2}

    if answer not in allowed:
        raise ValueError(
            f"Model selected {answer!r}, which is not one of the candidates "
            f"{sorted(allowed)}."
        )

    return answer

def judge_full_question(
    question: str,
    answer_1: str,
    answer_2: str,
    choice_text_1: str,
    choice_text_2: str,
    *,
    model_key: str,
    backend: Backend,
    registry: Registry,
    seed: int,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Judge the full question and choose between two candidate answers."""
    fields = {
        "question": question,
        "answer_1": answer_1,
        "answer_2": answer_2,
        "choice_text_1": choice_text_1,
        "choice_text_2": choice_text_2,
    }
    if any(not isinstance(value, str) or not value.strip()
           for value in fields.values()):
        raise ValueError("Question, answer labels, and choice texts must be non-empty strings")
    if answer_1 == answer_2:
        raise ValueError("answer_1 and answer_2 must be different")

    prompt = load_prompt("judge_full_q_2way")
    messages = prompt.messages(**fields)

    result = backend.chat(
        model_key,
        messages,
        **registry.sampling("judge_verdict"),
        max_tokens=max_tokens,
        seed=seed,
    )
    answer = parse_final_answer(result.text, answer_1, answer_2)

    return {
        "answer": answer,
        "raw_completion": result.text,
        "meta": stage_meta(
            "judge_full_question",
            model_call=result.params,
            **prompt.meta(),
        ),
    }


def judge_question_plus_crux(
    question: str,
    claim: str,
    *,
    model_key: str,
    backend: Backend,
    registry: Registry,
    seed: int,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Verify a crux claim while providing the original question as context."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    prompt = load_prompt("judge_q_plus_crux")
    messages = prompt.messages(
        question=question.strip(),
        crux_claim=claim.strip(),
    )

    result = backend.chat(
        model_key,
        messages,
        **registry.sampling("judge_verdict"),
        max_tokens=max_tokens,
        seed=seed,
    )
    verdict = parse_verdict(result.text)

    return {
        "verdict": verdict,
        "raw_completion": result.text,
        "meta": stage_meta(
            "judge_question_plus_crux",
            model_call=result.params,
            **prompt.meta(),
        ),
    }


def parse_verdict(completion: str) -> Verdict:
    """Extract an explicit TRUE/FALSE verdict; reject missing or ambiguous output."""
    matches = _VERDICT_RE.findall(completion)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one 'VERDICT: TRUE/FALSE' line; found {len(matches)}."
        )
    return matches[0]  # type: ignore[return-value]


def judge_crux(
    claim: str,
    *,
    model_key: str,
    backend: Backend,
    registry: Registry,
    seed: int,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Ask a configured judge to verify a crux claim and return its auditable result."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    prompt = load_prompt("judge_crux")
    messages = prompt.messages(crux_claim=claim.strip())

    result = backend.chat(
        model_key,
        messages,
        **registry.sampling("judge_verdict"),
        max_tokens=max_tokens,
        seed=seed,
    )
    verdict = parse_verdict(result.text)

    return {
        "verdict": verdict,
        "raw_completion": result.text,
        "meta": stage_meta(
            "judge_crux",
            model_call=result.params,
            **prompt.meta(),
        ),
    }


def measure_c_prob(
    claim: str,
    completion: str,
    *,
    model_key: str,
    backend: Backend,
) -> Dict[str, Any]:
    """Measure TRUE/FALSE next-token probability mass after the verdict marker."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    # The backend must support full next-token distributions (currently hf_local).
    prompt = load_prompt("judge_crux")
    messages = prompt.messages(crux_claim=claim.strip())

    probabilities = backend.verdict_probs(
        model_key,
        messages,
        completion,
        marker="VERDICT:",
    )

    return {
        "p_true_raw": probabilities["p_true_raw"],
        "p_false_raw": probabilities["p_false_raw"],
        "p_mass_tf": probabilities["p_mass_tf"],
        "p_true": probabilities["p_true"],
        "meta": {
            "model_key": model_key,
            "method": "next_token_probability_mass",
            "marker": "VERDICT:",
            "true_token_ids": probabilities["true_token_ids"],
            "false_token_ids": probabilities["false_token_ids"],
        },
    }


def measure_c_cons(
    claim: str,
    *,
    model_key: str,
    backend: Backend,
    registry: Registry,
    seeds: list[int],
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Measure five-sample TRUE/FALSE self-consistency for a crux claim."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    if len(seeds) != 5 or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must contain exactly five integers")
    if len(set(seeds)) != 5:
        raise ValueError("each self-consistency sample must use a distinct seed")

    prompt = load_prompt("judge_crux")
    messages = prompt.messages(crux_claim=claim.strip())

    samples = []
    for seed in seeds:
        result = backend.chat(
            model_key,
            messages,
            **registry.sampling("judge_cons"),
            max_tokens=max_tokens,
            seed=seed,
        )
        verdict = parse_verdict(result.text)
        samples.append(
            {
                "seed": seed,
                "verdict": verdict,
                "raw_completion": result.text,
                "model_call": result.params,
            }
        )

    true_count = sum(sample["verdict"] == "TRUE" for sample in samples)
    false_count = len(samples) - true_count
    majority_verdict = (
        "TRUE" if true_count > false_count
        else "FALSE" if false_count > true_count
        else "TIE"
    )

    return {
        "samples": samples,
        "true_count": true_count,
        "false_count": false_count,
        "agreement_rate": max(true_count, false_count) / len(samples),
        "majority_verdict": majority_verdict,
        "meta": stage_meta(
            "c_cons",
            model_key=model_key,
            sample_count=len(samples),
            temperature=0.7,
            seeds=list(seeds),
            **prompt.meta(),
        ),
    }
