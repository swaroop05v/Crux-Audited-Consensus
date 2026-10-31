"""``crux track2 report --run <run>``: what each judge layer did on this run's cruxes.

Per layer: TRUE rate over valid verdicts (split leaky / non-leaky when leakage.jsonl exists), failures by type, the raw
TRUE+FALSE mass behind c_prob, and the c_prob and c_cons distributions. Across layers: agreement on cruxes where both
have a valid verdict.

Prints a text summary and writes data/runs/<run>/track2_report.json (not a stage file).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from crux.io import read_jsonl
from crux.track1.common import run_path
from crux.track1.report import _d_stats, _fmt, _fmt_failures, _load, _rate, failure_breakdown

REPORT_NAME = "track2_report.json"
# Below this, most of the next-token probability is not on a verdict token: a format failure, not a confident verdict.
LOW_MASS_THRESHOLD = 0.5


def _binned_stats(values: List[float]) -> Dict[str, Any]:
    """Like _d_stats, with bins of width 0.1, because c_prob is continuous."""
    if not values:
        return {"n": 0}
    bins = Counter(min(int(value * 10), 9) for value in values)
    return {**_d_stats(values),
            "histogram": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": count for b, count in sorted(bins.items())}}


def _true_rate(verdicts: List[Optional[str]]) -> Dict[str, Any]:
    valid = [verdict for verdict in verdicts if verdict is not None]
    return _rate(valid.count("TRUE"), len(valid))


def _mass_report(masses: List[float]) -> Dict[str, Any]:
    low = _rate(sum(mass < LOW_MASS_THRESHOLD for mass in masses), len(masses))
    if not masses:
        return {"n": 0, "low_mass": low}
    return {"n": len(masses), "mean": round(statistics.mean(masses), 4), "median": statistics.median(masses),
            "min": min(masses), "low_mass": low}


def _c_cons_report(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [block["c_cons"] for block in blocks if block["c_cons"] is not None]
    return {"values": _d_stats(values), "no_valid_samples": len(blocks) - len(values),
            "sample_failures": failure_breakdown([sample for block in blocks for sample in block["samples"]])}


def layer_report(layer: str, judged: List[Dict[str, Any]], leaky: Optional[Dict[str, bool]]) -> Dict[str, Any]:
    entries = {r["crux_id"]: r["layers"][layer] for r in judged if layer in r["layers"]}
    verdicts = [entry["verdict"] for entry in entries.values()]
    report: Dict[str, Any] = {
        "n": len(entries),
        "true_rate": _true_rate(verdicts),
        "failures": failure_breakdown(list(entries.values())),
        "p_mass_tf": _mass_report([e["p_mass_tf"] for e in entries.values() if e["p_mass_tf"] is not None]),
        "c_prob": _binned_stats([e["c_prob"] for e in entries.values() if e["c_prob"] is not None]),
        "c_prob_missing": sum(e["c_prob"] is None for e in entries.values()),
    }
    if leaky is not None:
        for label, wanted in (("non_leaky", False), ("leaky", True)):
            report[f"true_rate_{label}"] = _true_rate([entry["verdict"] for crux_id, entry in entries.items()
                                                       if leaky.get(crux_id) is wanted])
        report["leakage_unknown"] = sum(crux_id not in leaky for crux_id in entries)
    c_cons = [entry["c_cons"] for entry in entries.values() if entry["c_cons"] is not None]
    if c_cons:
        report["c_cons"] = _c_cons_report(c_cons)
    return report


def pairwise_agreement(layers: List[str], judged: List[Dict[str, Any]]) -> Dict[str, Any]:
    report = {}
    for first, second in combinations(layers, 2):
        pairs = [(r["layers"][first]["verdict"], r["layers"][second]["verdict"]) for r in judged
                 if first in r["layers"] and second in r["layers"]]
        both_valid = [(a, b) for a, b in pairs if a is not None and b is not None]
        report[f"{first} vs {second}"] = _rate(sum(a == b for a, b in both_valid), len(both_valid))
    return report


def build_report(run: str) -> Dict[str, Any]:
    records = read_jsonl(run_path(run, "judges"), "judges")
    leakage = _load(run, "leakage")
    leaky = {r["crux_id"]: r["is_leaky"] for r in leakage if r["status"] == "checked"} if leakage is not None else None
    judged = [r for r in records if r["status"] == "judged"]
    layers = list(dict.fromkeys(layer for r in judged for layer in r["layers"]))
    return {
        "run": run,
        "n_records": len(records),
        "n_judged": len(judged),
        "no_claim": len(records) - len(judged),
        "leakage_joined": leaky is not None,
        "low_mass_threshold": LOW_MASS_THRESHOLD,
        "layers": {layer: layer_report(layer, judged, leaky) for layer in layers},
        "pairwise_agreement": pairwise_agreement(layers, judged),
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"Track 2 report: run {report['run']}",
             f"  records           {report['n_records']} ({report['n_judged']} judged, {report['no_claim']} no claim)"]
    for layer, r in report["layers"].items():
        mass = r["p_mass_tf"]
        lines += ["", layer, f"  TRUE rate         {_fmt(r['true_rate'])} of valid verdicts"]
        for label in ("non_leaky", "leaky"):
            if f"true_rate_{label}" in r:
                lines.append(f"    {label:<15} {_fmt(r[f'true_rate_{label}'])}")
        if "leakage_unknown" in r:
            lines.append(f"    {'unknown':<15} {r['leakage_unknown']} cruxes not in leakage.jsonl")
        lines += [f"  failures          {_fmt_failures(r['failures'])}",
                  f"  p_mass_tf         {mass}",
                  f"  low mass (< {report['low_mass_threshold']})   {_fmt(mass['low_mass'])}",
                  f"  c_prob            {r['c_prob']} (missing: {r['c_prob_missing']})"]
        if "c_cons" in r:
            lines += [f"  c_cons            {r['c_cons']['values']} (no valid samples: "
                      f"{r['c_cons']['no_valid_samples']})",
                      f"  c_cons samples    {_fmt_failures(r['c_cons']['sample_failures'])}"]
    if report["pairwise_agreement"]:
        lines += ["", "Pairwise agreement (both verdicts valid)"]
        lines += [f"  {pair:<34} {_fmt(entry)}" for pair, entry in report["pairwise_agreement"].items()]
    return "\n".join(lines)


def run_report(*, run: str) -> Path:
    report = build_report(run)
    out = run_path(run, "x").parent / REPORT_NAME
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_report(report))
    print(f"\nWrote {out}")
    return out
