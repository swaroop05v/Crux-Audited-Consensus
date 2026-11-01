
"""Selection gate for Crux-Audited Consensus (D10)."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional


SELECTION_REASONS = {
    "selected",
    "no_candidates",
    "all_leaky",
    "d_zero",
    "veto",
    "candidates_disagree",
    "layers_disagree",
    "below_tau",
}

CHECKABILITY_SIGNALS = {"c_cons", "c_prob", "c_2judge"}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_answer(value: Any, field: str) -> None:
    if value is not None and (
        not isinstance(value, str) or not value.strip()
    ):
        raise ValueError(f"{field} must be a non-empty string or None")


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    if not isinstance(candidate, Mapping):
        raise ValueError("Each candidate must be a mapping")

    crux_id = candidate.get("crux_id")
    if not isinstance(crux_id, str) or not crux_id.strip():
        raise ValueError("Each candidate needs a non-empty crux_id")

    for field in ("parsed", "leaky", "veto_flag"):
        if field not in candidate or not isinstance(candidate[field], bool):
            raise ValueError(f"{crux_id}: {field} must be a boolean")

    if "d" not in candidate or not _is_finite_number(candidate["d"]):
        raise ValueError(f"{crux_id}: d must be a finite number")
    if not 0 <= candidate["d"] <= 1:
        raise ValueError(f"{crux_id}: d must be between 0 and 1")

    if "mapped_answer" not in candidate:
        raise ValueError(f"{crux_id}: mapped_answer is required")
    _validate_answer(candidate["mapped_answer"], f"{crux_id}: mapped_answer")

    rate = candidate.get("mapping_agreement_rate")
    if not _is_finite_number(rate) or not 0 <= rate <= 1:
        raise ValueError(
            f"{crux_id}: mapping_agreement_rate must be between 0 and 1"
        )

    verdicts = candidate.get("layer_verdicts")
    if not isinstance(verdicts, Mapping):
        raise ValueError(f"{crux_id}: layer_verdicts must be a mapping")

    for layer, verdict in verdicts.items():
        if not isinstance(layer, str) or not layer.strip():
            raise ValueError(f"{crux_id}: layer names must be non-empty strings")
        if verdict not in ("TRUE", "FALSE"):
            raise ValueError(
                f"{crux_id}: layer {layer!r} must have a TRUE or FALSE verdict"
            )

    for signal in CHECKABILITY_SIGNALS:
        if signal in candidate and (
            not _is_finite_number(candidate[signal])
            or not 0 <= candidate[signal] <= 1
        ):
            raise ValueError(f"{crux_id}: {signal} must be between 0 and 1")


def _layers_agree(candidate: Mapping[str, Any]) -> bool:
    verdicts = list(candidate["layer_verdicts"].values())
    return len(verdicts) >= 2 and len(set(verdicts)) == 1


def _fallback(
    dr_consensus_answer: Optional[str],
    dr_judge_answer: Optional[str],
) -> Dict[str, Any]:
    if dr_consensus_answer is not None:
        return {
            "final_answer": dr_consensus_answer,
            "final_source": "dr_consensus",
        }
    if dr_judge_answer is not None:
        return {
            "final_answer": dr_judge_answer,
            "final_source": "dr_judge",
        }
    return {"final_answer": None, "final_source": None}


def _reason_without_eligible(
    candidates: list[Mapping[str, Any]],
) -> str:
    if not candidates:
        return "no_candidates"
    if all(c["leaky"] for c in candidates):
        return "all_leaky"
    if not any(c["parsed"] and not c["leaky"] for c in candidates):
        return "no_candidates"
    # Without its veto_flag, such a candidate would have been eligible.
    if any(
        c["parsed"] and not c["leaky"] and c["d"] > 0
        and _layers_agree(c) and c["veto_flag"]
        for c in candidates
    ):
        return "veto"
    if not any(
        c["parsed"] and not c["leaky"] and c["d"] > 0
        for c in candidates
    ):
        return "d_zero"
    return "layers_disagree"


def _rank_key(candidate: Mapping[str, Any]) -> tuple:
    return (
        -candidate["d"],
        -candidate["mapping_agreement_rate"],
        candidate["crux_id"],
    )


def apply_gate(
    candidates: Iterable[Mapping[str, Any]],
    *,
    tau: float,
    c_signal: str,
    dr_consensus_answer: Optional[str] = None,
    dr_judge_answer: Optional[str] = None,
    p1_mode: str = "flag",
) -> Dict[str, Any]:
    """Apply D10 selection, scoring, and abstention rules.

    In P1 flag mode, vetoed candidates are logged and excluded, while the
    remaining candidates continue through P2. In abstain mode, any veto
    causes fallback. The counterfactual P1-abstain result is always logged.
    """
    if not _is_finite_number(tau):
        raise ValueError("tau must be a finite number")
    if c_signal not in CHECKABILITY_SIGNALS:
        raise ValueError(f"c_signal must be one of {sorted(CHECKABILITY_SIGNALS)}")
    if p1_mode not in ("flag", "abstain"):
        raise ValueError("p1_mode must be 'flag' or 'abstain'")

    _validate_answer(dr_consensus_answer, "dr_consensus_answer")
    _validate_answer(dr_judge_answer, "dr_judge_answer")

    candidate_list = list(candidates)
    for candidate in candidate_list:
        _validate_candidate(candidate)
        if candidate["parsed"] and not candidate["leaky"]:
            if c_signal not in candidate:
                raise ValueError(
                    f"{candidate['crux_id']}: missing configured signal {c_signal}"
                )

    ids = [c["crux_id"] for c in candidate_list]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate crux_id values must be unique")

    parsed_nonleaky = [
        c for c in candidate_list if c["parsed"] and not c["leaky"]
    ]
    vetoed = [c for c in parsed_nonleaky if c["veto_flag"]]

    # The counterfactual P1 abstention would stop the crux route on any veto.
    p1_abstain_would_abstain = bool(vetoed)

    # In flag mode, vetoed candidates cannot themselves be selected.
    p1_remaining = [
        c for c in parsed_nonleaky if not c["veto_flag"]
    ]

    # A veto in abstain mode falls back immediately.
    if p1_mode == "abstain" and p1_abstain_would_abstain:
        # The actual abstain policy stops, but audit the counterfactual
        # flag policy, which excludes vetoed candidates and continues.
        flag_eligible = [
            c for c in p1_remaining
            if c["d"] > 0 and _layers_agree(c)
        ]

        flag_ranked = sorted(flag_eligible, key=_rank_key)
        p2_alt_answer = (
            flag_ranked[0]["mapped_answer"] if flag_ranked else None
        )

        dr_consensus_relation = "unavailable"
        if (
            p2_alt_answer is not None
            and dr_consensus_answer is not None
        ):
            dr_consensus_relation = (
                "confirms"
                if p2_alt_answer == dr_consensus_answer
                else "overturns"
            )

        fallback = _fallback(dr_consensus_answer, dr_judge_answer)

        return {
            "selected_crux_id": None,
            "selection_reason": "veto",
            "crux_score": None,
            "tau": tau,
            "c_signal": c_signal,
            "p1_mode": p1_mode,
            "p1_veto_crux_ids": [c["crux_id"] for c in vetoed],
            "p1_abstain_would_abstain": True,
            "eligible_crux_ids": [
                c["crux_id"] for c in flag_eligible
            ],
            "final_answer": fallback["final_answer"],
            "final_source": fallback["final_source"],
            "dr_consensus_relation": dr_consensus_relation,
            "p2_alt_answer": p2_alt_answer,
        }

    eligible = [
        c for c in p1_remaining
        if c["d"] > 0 and _layers_agree(c)
    ]

    base = {
        "selected_crux_id": None,
        "selection_reason": None,
        "crux_score": None,
        "tau": tau,
        "c_signal": c_signal,
        "p1_mode": p1_mode,
        "p1_veto_crux_ids": [c["crux_id"] for c in vetoed],
        "p1_abstain_would_abstain": p1_abstain_would_abstain,
        "eligible_crux_ids": [c["crux_id"] for c in eligible],
        "final_answer": None,
        "final_source": None,
        "dr_consensus_relation": "unavailable",
        "p2_alt_answer": None,
    }

    def finish(
        reason: str,
        *,
        selected: Optional[Mapping[str, Any]] = None,
        score: Optional[float] = None,
        p2_alt_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = dict(base)
        result["selection_reason"] = reason
        result["crux_score"] = score
        result["p2_alt_answer"] = p2_alt_answer

        fallback = _fallback(dr_consensus_answer, dr_judge_answer)
        result.update(fallback)

        answer_for_relation = p2_alt_answer
        if selected is not None:
            result["selected_crux_id"] = selected["crux_id"]
            result["selected_mapping"] = selected["mapped_answer"]
            if reason == "selected":
                result["final_answer"] = selected["mapped_answer"]
                result["final_source"] = "crux"
                answer_for_relation = selected["mapped_answer"]

        if (
            answer_for_relation is not None
            and dr_consensus_answer is not None
        ):
            result["dr_consensus_relation"] = (
                "confirms"
                if answer_for_relation == dr_consensus_answer
                else "overturns"
            )

        return result

    if not eligible:
        return finish(_reason_without_eligible(candidate_list))

    ranked = sorted(eligible, key=_rank_key)
    highest_d_candidate = ranked[0]
    p2_alt_answer = highest_d_candidate["mapped_answer"]

    mappings = {c["mapped_answer"] for c in eligible}
    if None in mappings or len(mappings) != 1:
        return finish(
            "candidates_disagree",
            p2_alt_answer=p2_alt_answer,
        )

    selected = highest_d_candidate
    score = selected["d"] * selected[c_signal]

    if score < tau:
        return finish(
            "below_tau",
            selected=selected,
            score=score,
            p2_alt_answer=p2_alt_answer,
        )

    return finish(
        "selected",
        selected=selected,
        score=score,
        p2_alt_answer=p2_alt_answer,
    )
