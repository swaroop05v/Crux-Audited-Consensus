"""Command line: one subcommand per stage, each writing exactly one stage file.

    crux seed build
    crux track1 propose  --run seed_fake --backend fake
    crux track1 leakage  --run seed_fake [--l2 --backend fake]
    crux track1 decisive --run seed_fake --backend fake --k 3
    crux track1 report   --run seed_fake
    crux smoke consultants --backend vllm --n 2   (needs live vLLM servers)

Stage files go to data/runs/<run>/<stage>.jsonl (gitignored). Without installing: PYTHONPATH=src python -m crux ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from crux.track1.common import DEFAULT_CASES


def _common(parser: argparse.ArgumentParser, *, backend_default: Optional[str] = "fake") -> None:
    parser.add_argument("--run", required=True, help="Run name; files go to data/runs/<run>/.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Stage file of cases (default: seed_30).")
    parser.add_argument("--backend", default=backend_default,
                        help="fake (canned replies, offline) or a backend from configs/models.yaml, e.g. vllm.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; per-call seeds derive from it.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent requests (match the servers' --max-num-seqs). Output order and per-call "
                             "seeds do not depend on it.")


def build_parser() -> argparse.ArgumentParser:
    from crux import smoke
    from crux.data import seed
    from crux.track1 import decisiveness, extractor, leakage, report

    parser = argparse.ArgumentParser(prog="crux", description="Crux-Audited Consensus pipeline stages.")
    groups = parser.add_subparsers(dest="group", required=True)

    seed_group = groups.add_parser("seed", help="Seed data.").add_subparsers(dest="command", required=True)
    build = seed_group.add_parser("build", help="Build seed_30.jsonl from collaborative-dr/data (pair A).")
    seed.add_arguments(build)
    build.set_defaults(func=seed.run)

    track1 = groups.add_parser("track1", help="Crux proposal, leakage, decisiveness.").add_subparsers(
        dest="command", required=True)

    propose = track1.add_parser("propose", help="Prompt 1A per consultant -> crux_proposals.jsonl")
    _common(propose)
    propose.add_argument("--prompt-version", default=extractor.DEFAULT_PROMPT_VERSION)
    propose.add_argument("--max-tokens", type=int, default=extractor.DEFAULT_MAX_TOKENS)
    propose.add_argument("--limit", type=int, help="Only the first N cases.")
    propose.set_defaults(func=lambda a: extractor.run_stage(
        cases_path=a.cases, run=a.run, backend_name=a.backend, prompt_version=a.prompt_version,
        max_tokens=a.max_tokens, seed=a.seed, limit=a.limit, workers=a.workers))

    leak = track1.add_parser("leakage", help="L1 (+ optional L2 probe) and P5 flags -> leakage.jsonl")
    _common(leak)
    leak.add_argument("--l2", action="store_true", help="Also run the restatement probe (Prompt 1C) on both "
                                                         "consultants.")
    leak.add_argument("--probe-max-tokens", type=int, default=leakage.DEFAULT_PROBE_MAX_TOKENS)
    leak.set_defaults(func=lambda a: leakage.run_stage(
        cases_path=a.cases, run=a.run, backend_name=a.backend, run_l2=a.l2,
        probe_max_tokens=a.probe_max_tokens, seed=a.seed, workers=a.workers))

    decisive = track1.add_parser("decisive", help="Prompt 1B (v2) K samples per consultant -> decisiveness.jsonl")
    _common(decisive)
    decisive.add_argument("--k", type=int, default=decisiveness.DEFAULT_K)
    decisive.add_argument("--prompt-version", default=decisiveness.DEFAULT_PROMPT_VERSION)
    decisive.add_argument("--max-tokens", type=int, default=decisiveness.DEFAULT_MAX_TOKENS)
    decisive.set_defaults(func=lambda a: decisiveness.run_stage(
        cases_path=a.cases, run=a.run, backend_name=a.backend, k=a.k, prompt_version=a.prompt_version,
        max_tokens=a.max_tokens, seed=a.seed, workers=a.workers))

    smoke_group = groups.add_parser("smoke", help="Live-server checks.").add_subparsers(dest="command",
                                                                                        required=True)
    consultants = smoke_group.add_parser(
        "consultants", help="Prompt 1A per consultant on a few cases; checks the RAW reply (no <think>, JSON "
                            "parses, finished, right model). Exit code 1 on any failure.")
    consultants.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    consultants.add_argument("--n", type=int, default=2, help="Number of cases.")
    consultants.add_argument("--backend", default="vllm")
    consultants.add_argument("--max-tokens", type=int, default=extractor.DEFAULT_MAX_TOKENS)
    consultants.add_argument("--seed", type=int, default=42)
    consultants.set_defaults(func=lambda a: smoke.run_smoke(
        cases_path=a.cases, n=a.n, backend_name=a.backend, max_tokens=a.max_tokens, seed=a.seed))

    summary = track1.add_parser("report", help="Failure/truncation, leakage, veto, UNDETERMINED and d summary "
                                                "-> track1_report.json")
    summary.add_argument("--run", required=True)
    summary.set_defaults(func=lambda a: report.run_report(run=a.run))
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
