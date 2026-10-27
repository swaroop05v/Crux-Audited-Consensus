
"""Checkability measurements for Track 2 weak-judge experiments."""

from __future__ import annotations

from typing import Any, Dict, Literal

from crux.backends.base import Backend, Registry
from crux.track2.weak_judge import judge_crux

Verdict = Literal["TRUE", "FALSE"]

MODEL_FAMILIES = {
    "judge_l1_qwen": "qwen",
    "judge_l2_llama": "llama",
    "judge_l2_gemma": "gemma",
}


def compare_verdicts(
    verdict_a: str,
    verdict_b: str,
    *,
    model_a: str,
    model_b: str,
    family_a: str,
    family_b: str,
) -> Dict[str, Any]:
    """Compare two verdicts using explicit model-family identities."""
    if verdict_a not in ("TRUE", "FALSE"):
        raise ValueError("verdict_a must be TRUE or FALSE")
    if verdict_b not in ("TRUE", "FALSE"):
        raise ValueError("verdict_b must be TRUE or FALSE")

    for name, value in (
        ("model_a", model_a),
        ("model_b", model_b),
        ("family_a", family_a),
        ("family_b", family_b),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    return {
        "verdict_a": verdict_a,
        "verdict_b": verdict_b,
        "model_a": model_a,
        "model_b": model_b,
        "family_a": family_a,
        "family_b": family_b,
        "agreement": verdict_a == verdict_b,
        "comparison": (
            "same_family" if family_a == family_b else "cross_family"
        ),
    }


def compare_judges_on_claim(
    claim: str,
    *,
    model_a: str,
    model_b: str,
    backend: Backend,
    registry: Registry,
    seed: int,
) -> Dict[str, Any]:
    """Run two configured judges on the same claim and compare their verdicts."""
    if model_a not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_a {model_a!r}")
    if model_b not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_b {model_b!r}")

    result_a = judge_crux(
        claim,
        model_key=model_a,
        backend=backend,
        registry=registry,
        seed=seed,
    )
    result_b = judge_crux(
        claim,
        model_key=model_b,
        backend=backend,
        registry=registry,
        seed=seed,
    )

    comparison = compare_verdicts(
        result_a["verdict"],
        result_b["verdict"],
        model_a=model_a,
        model_b=model_b,
        family_a=MODEL_FAMILIES[model_a],
        family_b=MODEL_FAMILIES[model_b],
    )

    return {
        **comparison,
        "claim": claim,
        "seed": seed,
        "judge_a": result_a,
        "judge_b": result_b,
    }
