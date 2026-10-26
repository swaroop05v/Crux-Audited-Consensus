"""``crux track1 report --run <run>``: the Track 1 numbers decisions D1 asks for before P1 becomes a hard rule.

Reads whichever Track 1 stage files exist in data/runs/<run>/ and reports, per stage:
- failure rates by type (call_error, truncated, parse_error), so truncation is visible on its own;
- leakage: how many cruxes L1 (by reason), L2 and the P5 flags each flagged, and leak_probe_failed;
- decisiveness: veto rate (P1), UNDETERMINED rate, d distribution, mapping conflicts and ties.

Prints a text summary and writes data/runs/<run>/track1_report.json (not a stage file).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from crux.io import read_jsonl
from crux.schema import UNDETERMINED
from crux.track1.common import FAILURE_TYPES, run_path

REPORT_NAME = "track1_report.json"


def _rate(count: int, total: int) -> Dict[str, Any]:
    return {"count": count, "of": total, "rate": round(count / total, 4) if total else None}


def failure_breakdown(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {kind: _rate(sum(o.get("failure_type") == kind for o in outputs), len(outputs)) for kind in FAILURE_TYPES}


def _load(run: str, stage: str) -> Optional[List[Dict[str, Any]]]:
    path = run_path(run, stage)
    return read_jsonl(path, stage) if path.exists() else None


def proposals_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    verdicts = Counter({True: "true", False: "false", None: "missing"}[r.get("proposer_own_verdict")]
                       for r in records if r["parse_ok"])
    return {"n": len(records), "parsed": _rate(sum(r["parse_ok"] for r in records), len(records)),
            "failures": failure_breakdown(records), "own_verdict": dict(verdicts)}


def leakage_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    checked = [r for r in records if r["status"] == "checked"]
    n = len(checked)
    with_l2 = [r for r in checked if r.get("l2") is not None]
    probes = [entry for r in with_l2 for entry in r["l2"]["by_consultant"].values()]
    reasons = Counter(reason for r in checked for reason in r["l1"]["reasons"])
    return {
        "n_checked": n,
        "no_claim": len(records) - n,
        "is_leaky": _rate(sum(r["is_leaky"] for r in checked), n),
        "l1_flagged": _rate(sum(r["l1"]["flagged"] for r in checked), n),
        "l1_by_reason": {reason: _rate(count, n) for reason, count in sorted(reasons.items())},
        "l2_ran_on": len(with_l2),
        "l2_flagged": _rate(sum(r["l2"]["flagged"] is True for r in with_l2), len(with_l2)),
        "l2_undetermined": _rate(sum(r["l2"]["flagged"] is None for r in with_l2), len(with_l2)),
        "leak_probe_failed": _rate(sum(bool(r.get("leak_probe_failed")) for r in with_l2), len(with_l2)),
        "probe_call_failures": failure_breakdown(probes),
        "p5_multi_clause": _rate(sum(r["p5_flags"]["multi_clause"] for r in checked), n),
        "p5_deictic": _rate(sum(r["p5_flags"]["deictic"] for r in checked), n),
    }


def _d_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    histogram = Counter(f"{v:.2f}" for v in values)
    return {"n": len(values), "mean": round(statistics.mean(values), 4), "median": statistics.median(values),
            "histogram": dict(sorted(histogram.items()))}


def decisiveness_report(records: List[Dict[str, Any]],
                        leakage: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    evaluated = [r for r in records if r["status"] == "evaluated"]
    n = len(evaluated)
    samples = [s for r in evaluated for s in r["samples"]]
    parsed = [s for s in samples if s["parse_ok"]]
    branches = [s[b] for s in parsed for b in ("if_true", "if_false")]
    vetoed = [r for r in evaluated if r["veto_flag"]]
    report: Dict[str, Any] = {
        "n_evaluated": n,
        "no_claim": len(records) - n,
        "sample_failures": failure_breakdown(samples),
        "undetermined": {
            "samples_with_any": _rate(sum(UNDETERMINED in (s["if_true"], s["if_false"]) for s in parsed),
                                      len(parsed)),
            "branches": _rate(branches.count(UNDETERMINED), len(branches)),
        },
        "d": _d_stats([r["d"] for r in evaluated]),
        "veto": {
            **_rate(len(vetoed), n),
            "vetoed_by": dict(Counter(r["vetoed_by"] for r in vetoed)),
            "by_proposer": sum(r["veto_by_proposer"] is True for r in vetoed),
            "by_opponent": sum(r["veto_by_proposer"] is False for r in vetoed),
        },
        "mapping_conflict": _rate(sum(r["mapping_conflict"] for r in evaluated), n),
        "mapping_tie": _rate(sum(r["mapping_tie"] for r in evaluated), n),
    }
    if leakage is not None:
        leaky = {r["crux_id"]: r["is_leaky"] for r in leakage if r["status"] == "checked"}
        for label, wanted in (("non_leaky", False), ("leaky", True)):
            group = [r for r in evaluated if leaky.get(r["crux_id"]) is wanted]
            report[f"d_{label}"] = _d_stats([r["d"] for r in group])
            report[f"veto_{label}"] = _rate(sum(r["veto_flag"] for r in group), len(group))
    return report


def build_report(run: str) -> Dict[str, Any]:
    proposals, leakage, decisive = (_load(run, s) for s in ("crux_proposals", "leakage", "decisiveness"))
    if proposals is None and leakage is None and decisive is None:
        raise FileNotFoundError(f"no Track 1 stage files in {run_path(run, 'x').parent}")
    report: Dict[str, Any] = {"run": run}
    if proposals is not None:
        report["crux_proposals"] = proposals_report(proposals)
    if leakage is not None:
        report["leakage"] = leakage_report(leakage)
    if decisive is not None:
        report["decisiveness"] = decisiveness_report(decisive, leakage)
    return report


def _fmt(entry: Dict[str, Any]) -> str:
    if entry.get("rate") is None:
        return f"{entry['count']}/{entry['of']}"
    return f"{entry['count']}/{entry['of']} ({100 * entry['rate']:.1f}%)"


def _fmt_failures(failures: Dict[str, Any]) -> str:
    return ", ".join(f"{kind} {_fmt(entry)}" for kind, entry in failures.items())


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"Track 1 report: run {report['run']}"]
    if "crux_proposals" in report:
        p = report["crux_proposals"]
        lines += ["", "crux_proposals", f"  parsed            {_fmt(p['parsed'])}",
                  f"  failures          {_fmt_failures(p['failures'])}",
                  f"  own_verdict       {p['own_verdict']}"]
    if "leakage" in report:
        lk = report["leakage"]
        lines += ["", "leakage", f"  checked           {lk['n_checked']} (no claim: {lk['no_claim']})",
                  f"  is_leaky          {_fmt(lk['is_leaky'])}",
                  f"  L1 flagged        {_fmt(lk['l1_flagged'])}"]
        lines += [f"    {reason:<22} {_fmt(entry)}" for reason, entry in lk["l1_by_reason"].items()]
        if lk["l2_ran_on"]:
            lines += [f"  L2 flagged        {_fmt(lk['l2_flagged'])}",
                      f"  L2 undetermined   {_fmt(lk['l2_undetermined'])}",
                      f"  leak_probe_failed {_fmt(lk['leak_probe_failed'])}",
                      f"  probe failures    {_fmt_failures(lk['probe_call_failures'])}"]
        else:
            lines.append("  L2                not run")
        lines += [f"  P5 multi-clause   {_fmt(lk['p5_multi_clause'])}  (log only)",
                  f"  P5 deictic        {_fmt(lk['p5_deictic'])}  (log only)"]
    if "decisiveness" in report:
        dc = report["decisiveness"]
        lines += ["", "decisiveness", f"  evaluated         {dc['n_evaluated']} (no claim: {dc['no_claim']})",
                  f"  sample failures   {_fmt_failures(dc['sample_failures'])}",
                  f"  UNDETERMINED      samples {_fmt(dc['undetermined']['samples_with_any'])}, "
                  f"branches {_fmt(dc['undetermined']['branches'])}",
                  f"  d                 {dc['d']}",
                  f"  veto (P1)         {_fmt(dc['veto'])}  by opponent {dc['veto']['by_opponent']}, "
                  f"by proposer {dc['veto']['by_proposer']}, vetoed_by {dc['veto']['vetoed_by']}",
                  f"  mapping conflict  {_fmt(dc['mapping_conflict'])}",
                  f"  mapping tie       {_fmt(dc['mapping_tie'])}"]
        for label in ("non_leaky", "leaky"):
            if f"d_{label}" in dc:
                lines.append(f"  {label:<17} d {dc[f'd_{label}']}, veto {_fmt(dc[f'veto_{label}'])}")
    return "\n".join(lines)


def run_report(*, run: str) -> Path:
    report = build_report(run)
    out = run_path(run, "x").parent / REPORT_NAME
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_report(report))
    print(f"\nWrote {out}")
    return out
