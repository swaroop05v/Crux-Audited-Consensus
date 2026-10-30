"""``crux smoke consultants``: live check of the consultant servers through our own backend.

Sends Prompt 1A for each consultant on a few seed cases exactly as the Track 1 stages do (same backend, so the same
per-request ``chat_template_kwargs`` such as Qwen's ``enable_thinking: false``), but with ``<think>`` stripping
turned OFF, so a thinking block cannot be hidden by our own clean-up. Checks per reply:

- no ``<think>`` / ``</think>`` in the raw text (thinking is really off);
- ``finish_reason == "stop"`` (not cut off at max_tokens);
- the JSON parses into a crux claim;
- the server reports the served model name from models.yaml.

Prints finish reasons, token counts and latency. Exits with code 1 if any check fails.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from crux.backends import Backend, BackendError, Registry, get_backend
from crux.prompts import load_prompt
from crux.seeding import stable_seed
from crux.track1 import extractor
from crux.track1.common import CONSULTANT_ROLES, load_cases

THINK_RE = re.compile(r"</?think>", re.IGNORECASE)


def smoke_consultants(cases: List[Dict[str, Any]], backend: Backend, registry: Registry, *,
                      max_tokens: int = extractor.DEFAULT_MAX_TOKENS, seed: int = 42) -> List[Dict[str, Any]]:
    prompt = load_prompt(extractor.PROMPT_NAME, extractor.DEFAULT_PROMPT_VERSION)
    results = []
    for case in cases:
        for role in CONSULTANT_ROLES:
            spec = registry.spec(role)
            entry: Dict[str, Any] = {"id": case["id"], "model_key": role, "served_name": spec.served_name}
            start = time.monotonic()
            try:
                result = backend.chat(role, extractor.proposal_messages(case, role, prompt),
                                      **registry.sampling("consultant"), max_tokens=max_tokens,
                                      seed=stable_seed(seed, "smoke", case["id"], role))
            except BackendError as exc:
                results.append({**entry, "ok": False, "problems": [f"call failed: {exc}"]})
                continue
            problems = []
            if THINK_RE.search(result.text):
                problems.append("raw reply contains <think>: thinking is not off")
            if result.finish_reason != "stop":
                problems.append(f"finish_reason {result.finish_reason!r}, expected 'stop'")
            parsed = extractor.parse_proposal(result.text)
            if not parsed["parse_ok"]:
                problems.append(f"JSON did not parse: {parsed['parse_error']}")
            served = result.params.get("served_model") or []
            if spec.served_name and served and spec.served_name not in served:
                problems.append(f"server ran {served}, expected {spec.served_name!r}")
            results.append({**entry, "ok": not problems, "problems": problems,
                            "finish_reason": result.finish_reason, "usage": result.params.get("usage"),
                            "seconds": round(time.monotonic() - start, 1), "crux_claim": parsed["crux_claim"],
                            "raw_head": result.text[:200]})
    return results


def run_smoke(*, cases_path: Path, n: int, backend_name: str, max_tokens: int = extractor.DEFAULT_MAX_TOKENS,
              seed: int = 42, registry: Optional[Registry] = None) -> List[Dict[str, Any]]:
    registry = (registry or Registry.load()).without_think_stripping()
    backend = get_backend(registry, backend_name)
    results = smoke_consultants(load_cases(cases_path, n), backend, registry, max_tokens=max_tokens, seed=seed)
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{status}] {r['model_key']:<13} {r['id']:<22} finish={r.get('finish_reason')} "
              f"usage={r.get('usage')} {r.get('seconds')}s")
        for problem in r["problems"]:
            print(f"        - {problem}")
        if r.get("raw_head"):
            print(f"        raw: {r['raw_head']!r}")
    failed = sum(not r["ok"] for r in results)
    print(f"smoke consultants: {len(results) - failed}/{len(results)} passed")
    if failed:
        raise SystemExit(1)
    return results
