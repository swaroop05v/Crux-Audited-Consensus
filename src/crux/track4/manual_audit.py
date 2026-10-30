"""50-sample Manual Crux Audit for Track 4.

Implements:
- Stratified sampling of 50 cases for human annotation.
- Rubric column definitions for inter-rater agreement.
- Cohen's kappa and percent agreement calculations.
- Go/no-go threshold agreement skeleton.

The 50-sample manual audit validates the automated pipeline by checking
whether human annotators agree with the crux extraction, leakage flags,
and decisiveness labels produced by the system.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Rubric columns for the manual audit spreadsheet
# ---------------------------------------------------------------------------

RUBRIC_COLUMNS = [
    # Case identification
    "case_id",
    "crux_id",
    "dataset",
    "split",
    # The crux itself
    "crux_claim",
    "proposed_by",
    # Automated pipeline outputs
    "auto_leakage_flag",         # bool: system's leakage call
    "auto_d",                    # float: system's decisiveness score
    "auto_layer1_verdict",       # TRUE/FALSE
    "auto_layer2_verdict",       # TRUE/FALSE
    "auto_selected",             # bool: was this crux selected by the gate?
    "auto_selection_reason",     # from SELECTION_REASONS
    "auto_crux_route_answer",    # the gated final answer
    # Human annotations (to be filled by annotators)
    "human_crux_valid",          # 1 = valid crux, 0 = not a real crux
    "human_crux_clear",          # 1 = clearly stated, 0 = ambiguous
    "human_leakage_flag",        # 1 = leaks answer info, 0 = clean
    "human_decisive",            # 1 = would flip answer, 0 = wouldn't
    "human_verdict_correct",     # 1 = weak judge got it right, 0 = wrong
    "human_notes",               # free-text notes
    # Annotator metadata
    "annotator_id",
    "annotation_timestamp",
]

# Minimal rubric description for each human-annotation column
RUBRIC_DESCRIPTIONS = {
    "human_crux_valid": (
        "Is this a genuine crux (a factual claim whose truth value would "
        "decisively change the answer)?  1 = yes, 0 = no."
    ),
    "human_crux_clear": (
        "Is the crux stated clearly enough to be independently verifiable? "
        "1 = yes, 0 = ambiguous or unclear."
    ),
    "human_leakage_flag": (
        "Does the crux wording reveal or strongly hint at the correct answer "
        "without requiring reasoning?  1 = leaks, 0 = clean."
    ),
    "human_decisive": (
        "If this crux's truth value were known, would it flip the consultant's "
        "answer?  1 = yes (decisive), 0 = no."
    ),
    "human_verdict_correct": (
        "Did the weak judge (Layer 1 + Layer 2) arrive at the correct verdict "
        "for this crux?  1 = correct, 0 = incorrect.  Compare against your "
        "own expert judgment."
    ),
}


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    cases: Sequence[Dict[str, Any]],
    n: int = 50,
    *,
    strata_key: str = "dataset",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Draw a stratified random sample of *n* cases.

    Strata are defined by ``strata_key`` (default: dataset).  Within each
    stratum the sample is proportional to stratum size.  If a stratum has
    fewer cases than its proportional share, all its cases are included and
    the remainder is redistributed.

    Only cases with ``source != "frontier_precomputed"`` are eligible.
    """
    rng = random.Random(seed)

    eligible = [c for c in cases if c.get("source") != "frontier_precomputed"]
    if len(eligible) <= n:
        return list(eligible)

    # Group by stratum
    strata: Dict[str, List[Dict[str, Any]]] = {}
    for c in eligible:
        key = c.get(strata_key, "unknown")
        strata.setdefault(key, []).append(c)

    # Proportional allocation
    total = len(eligible)
    allocation: Dict[str, int] = {}
    remainder = n
    for key, members in sorted(strata.items()):
        share = max(1, round(n * len(members) / total))
        share = min(share, len(members), remainder)
        allocation[key] = share
        remainder -= share

    # Redistribute any leftover
    while remainder > 0:
        for key in sorted(strata.keys()):
            if allocation[key] < len(strata[key]) and remainder > 0:
                allocation[key] += 1
                remainder -= 1

    # Sample within each stratum
    sampled: List[Dict[str, Any]] = []
    for key, count in sorted(allocation.items()):
        sampled.extend(rng.sample(strata[key], count))

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Audit sheet generation
# ---------------------------------------------------------------------------


def build_audit_rows(
    sampled_cases: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Convert sampled case records into audit spreadsheet rows.

    Each row has all ``RUBRIC_COLUMNS``, with automated fields pre-filled
    and human-annotation fields left blank.
    """
    rows: List[Dict[str, str]] = []
    for case in sampled_cases:
        row: Dict[str, str] = {}
        row["case_id"] = str(case.get("id", ""))
        row["crux_id"] = str(case.get("selected_crux_id") or case.get("crux_id", ""))
        row["dataset"] = str(case.get("dataset", ""))
        row["split"] = str(case.get("split", ""))
        row["crux_claim"] = str(case.get("crux_claim", ""))
        row["proposed_by"] = str(case.get("proposed_by", ""))

        # Automated outputs
        row["auto_leakage_flag"] = str(case.get("is_leaky", ""))
        row["auto_d"] = str(case.get("crux_d") if case.get("crux_d") is not None else case.get("d", ""))
        row["auto_layer1_verdict"] = str(case.get("layer1_verdict", ""))
        row["auto_layer2_verdict"] = str(case.get("layer2_verdict", ""))
        row["auto_selected"] = str(bool(case.get("selected_crux_id") or case.get("selection_reason") == "selected"))
        row["auto_selection_reason"] = str(case.get("selection_reason", ""))
        row["auto_crux_route_answer"] = str(case.get("final_answer") or case.get("crux_audited_answer", ""))

        # Human annotation fields (blank)
        for col in RUBRIC_COLUMNS:
            if col.startswith("human_") or col in ("annotator_id", "annotation_timestamp"):
                row[col] = ""

        rows.append(row)
    return rows


def write_audit_csv(rows: Sequence[Dict[str, str]], path: str) -> None:
    """Write audit rows to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUBRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def audit_rows_to_csv_string(rows: Sequence[Dict[str, str]]) -> str:
    """Render audit rows as a CSV string (for notebook display)."""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=RUBRIC_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Inter-rater agreement metrics
# ---------------------------------------------------------------------------


def percent_agreement(
    ratings_a: Sequence[int], ratings_b: Sequence[int]
) -> float:
    """Simple percent agreement between two raters on binary labels."""
    if len(ratings_a) != len(ratings_b):
        raise ValueError("Rating sequences must have the same length")
    if not ratings_a:
        return 0.0
    agree = sum(a == b for a, b in zip(ratings_a, ratings_b))
    return agree / len(ratings_a)


def cohens_kappa(
    ratings_a: Sequence[int], ratings_b: Sequence[int]
) -> float:
    """Cohen's kappa for two raters on binary (0/1) labels.

    Returns kappa in [-1, 1].  1 = perfect agreement, 0 = chance agreement.
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError("Rating sequences must have the same length")
    n = len(ratings_a)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = sum(a == b for a, b in zip(ratings_a, ratings_b)) / n

    # Expected agreement by chance
    a1 = sum(ratings_a) / n  # P(rater A = 1)
    b1 = sum(ratings_b) / n  # P(rater B = 1)
    p_e = a1 * b1 + (1 - a1) * (1 - b1)

    if p_e == 1.0:
        return 1.0  # Both raters gave the same label to everything
    return (p_o - p_e) / (1 - p_e)


def per_column_agreement(
    annotations_a: Sequence[Dict[str, int]],
    annotations_b: Sequence[Dict[str, int]],
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute percent agreement and Cohen's kappa per annotation column.

    Each entry in ``annotations_a`` / ``annotations_b`` is a dict mapping
    column names to binary (0/1) ratings, one per audited case, paired by
    position.

    Returns ``{column: {"percent_agreement": ..., "cohens_kappa": ...}}``.
    """
    if columns is None:
        columns = [c for c in RUBRIC_COLUMNS if c.startswith("human_") and c != "human_notes"]

    result: Dict[str, Dict[str, float]] = {}
    for col in columns:
        ra = [a.get(col, 0) for a in annotations_a]
        rb = [b.get(col, 0) for b in annotations_b]
        # Filter to cases where both raters provided a value
        pairs = [(a, b) for a, b in zip(ra, rb) if a is not None and b is not None]
        if not pairs:
            result[col] = {"percent_agreement": 0.0, "cohens_kappa": 0.0, "n": 0}
            continue
        ra_filtered, rb_filtered = zip(*pairs)
        result[col] = {
            "percent_agreement": percent_agreement(list(ra_filtered), list(rb_filtered)),
            "cohens_kappa": cohens_kappa(list(ra_filtered), list(rb_filtered)),
            "n": len(pairs),
        }
    return result


# ---------------------------------------------------------------------------
# Audit agreement thresholds (inter-rater reliability checks)
# ---------------------------------------------------------------------------

# These thresholds define the minimum inter-rater agreement levels required
# to trust the 50-sample manual audit annotations.  They are NOT the kill-
# criteria go/no-go thresholds (those live in README.md per CLAUDE.md rules).

AUDIT_AGREEMENT_THRESHOLDS = {
    "human_crux_valid": {
        "min_kappa": 0.6,
        "min_percent_agreement": 0.80,
        "description": "Annotators must agree on crux validity at ≥0.6 kappa.",
    },
    "human_leakage_flag": {
        "min_kappa": 0.7,
        "min_percent_agreement": 0.85,
        "description": "Leakage flags need high agreement — false negatives are costly.",
    },
    "human_decisive": {
        "min_kappa": 0.5,
        "min_percent_agreement": 0.75,
        "description": "Decisiveness is inherently subjective; moderate agreement suffices.",
    },
    "human_verdict_correct": {
        "min_kappa": 0.6,
        "min_percent_agreement": 0.80,
        "description": "Weak-judge accuracy agreement must be solid to trust calibration.",
    },
}


def check_audit_agreement(
    agreement_results: Dict[str, Dict[str, float]],
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Check whether inter-rater agreement meets audit agreement thresholds.

    Returns a dict with per-column pass/fail and an overall verdict.
    """
    if thresholds is None:
        thresholds = AUDIT_AGREEMENT_THRESHOLDS

    column_results: Dict[str, Dict[str, Any]] = {}
    all_pass = True

    for col, thresh in thresholds.items():
        actual = agreement_results.get(col, {})
        kappa = actual.get("cohens_kappa", 0.0)
        pct = actual.get("percent_agreement", 0.0)

        kappa_pass = kappa >= thresh["min_kappa"]
        pct_pass = pct >= thresh["min_percent_agreement"]
        col_pass = kappa_pass and pct_pass

        column_results[col] = {
            "kappa": kappa,
            "kappa_threshold": thresh["min_kappa"],
            "kappa_pass": kappa_pass,
            "percent_agreement": pct,
            "pct_threshold": thresh["min_percent_agreement"],
            "pct_pass": pct_pass,
            "pass": col_pass,
            "description": thresh["description"],
        }

        if not col_pass:
            all_pass = False

    return {
        "columns": column_results,
        "go": all_pass,
        "verdict": "GO" if all_pass else "NO-GO",
    }


# Backward-compatible aliases (deprecated — use the new names)
GO_NO_GO_THRESHOLDS = AUDIT_AGREEMENT_THRESHOLDS
check_go_no_go = check_audit_agreement
