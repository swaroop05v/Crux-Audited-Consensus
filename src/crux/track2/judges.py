"""Stage ``judges``: the weak judges' verdict and checkability on each crux (Doc 2 Prompt 2A, audit 4.2).

One record per crux with one entry per layer, so the gate can pick which layers count. Every crux with a parsed claim
is judged, leaky or not: Kill Criterion 3 and Track 4 Table 3 compare the two, and the gate drops leaky cruxes later.
The stage reads only crux_proposals.jsonl, so a judge sees the claim and nothing else.

Per crux and layer: the verdict at that layer's sampling profile (decisions D3), and ``c_prob`` from the same
completion. On ``--c-cons-layers`` (Layer 1 by default), ``c_cons`` from five seeded samples. A failed call, truncation
or unparseable reply is recorded with verdict None and Track 1's failure fields; configuration errors raise before any
call. If every verdict call to a layer fails with call_error, the run stops instead of writing a file full of None.

Layers run one at a time and each model is released before the next loads, for a small GPU. A finished layer is
checkpointed in data/runs/<run>/judges_checkpoints/, so rerunning the same command skips it; different params refuse
to resume. The checkpoints are removed once judges.jsonl is written. There is no --workers: ``hf_local`` is in-process
and ``set_seed`` is process-global, so concurrent calls would break the per-call seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from crux.backends import Backend, BackendError, Registry
from crux.io import read_jsonl, stage_meta, write_jsonl
from crux.prompts import load_prompt
from crux.schema import SCHEMA_VERSION, check_judge_layer
from crux.seeding import stable_seed
from crux.track1.common import failure_summary, make_backend, run_path
from crux.track2.weak_judge import judge_crux_recorded, measure_c_cons, measure_c_prob

STAGE = "judges"
PROMPT_NAME = "judge_crux"
DEFAULT_LAYERS = ("judge_l1_qwen", "judge_l2_llama", "judge_l2_gemma")
DEFAULT_C_CONS_LAYERS = ("judge_l1_qwen",)
DEFAULT_MAX_TOKENS = 512
C_CONS_SAMPLES = 5
C_CONS_PROFILE = "judge_cons"
VERDICT_PROFILES = {
    "judge_l1_qwen": "judge_verdict",
    "judge_l2_llama": "judge_layer2",
    "judge_l2_gemma": "judge_layer2",
}
CHECKPOINT_FOLDER = "judges_checkpoints"
PROGRESS_EVERY = 10

EMPTY_C_PROB = {"c_prob": None, "p_true": None, "p_true_raw": None, "p_false_raw": None, "p_mass_tf": None,
                "c_prob_meta": None}


def check_config(layers: Sequence[str], c_cons_layers: Sequence[str], backend: Backend, registry: Registry) -> None:
    """Raise on a bad layer list, backend or sampling profile; ``safe_chat`` would hide these as failed calls."""
    if not layers:
        raise ValueError("layers: name at least one judge layer")
    for name, values in (("layers", layers), ("c_cons_layers", c_cons_layers)):
        if len(set(values)) != len(values):
            raise ValueError(f"{name}: duplicates in {list(values)}")
    unknown = [layer for layer in layers if layer not in VERDICT_PROFILES]
    if unknown:
        raise ValueError(f"layers: {unknown} are not judge layers; known: {sorted(VERDICT_PROFILES)}")
    extra = [layer for layer in c_cons_layers if layer not in layers]
    if extra:
        raise ValueError(f"c_cons_layers: {extra} are not in layers {list(layers)}")
    for layer in layers:
        backend.spec(layer)
        registry.sampling(VERDICT_PROFILES[layer])
    if c_cons_layers:
        registry.sampling(C_CONS_PROFILE)


def layer_calls(layer: str, backend: Backend, registry: Registry, *, with_c_cons: bool,
                max_tokens: int) -> Dict[str, Any]:
    """What this layer is set up to run. From the registry, so a record whose calls all failed still has it."""
    profile = VERDICT_PROFILES[layer]
    c_cons = {"profile": C_CONS_PROFILE, **registry.sampling(C_CONS_PROFILE)} if with_c_cons else None
    return {**backend.spec(layer).describe(), "backend": backend.name,
            "sampling": {"verdict": {"profile": profile, **registry.sampling(profile)}, "c_cons": c_cons},
            "max_tokens": max_tokens}


def c_prob_fields(claim: str, sample: Dict[str, Any], *, model_key: str, backend: Backend) -> Dict[str, Any]:
    """c_prob from the verdict's own completion. Anything that stops it is recorded in ``c_prob_error``."""
    if sample["verdict"] is None:
        return {**EMPTY_C_PROB, "c_prob_error": f"no verdict ({sample['failure_type']})"}
    try:
        measured = measure_c_prob(claim, sample["raw_output"], model_key=model_key, backend=backend)
    except NotImplementedError as exc:
        return {**EMPTY_C_PROB, "c_prob_error": f"unsupported: {exc}"}
    except BackendError as exc:
        return {**EMPTY_C_PROB, "c_prob_error": str(exc)}
    return {"c_prob": measured["c_prob"], "p_true": measured["p_true"], "p_true_raw": measured["p_true_raw"],
            "p_false_raw": measured["p_false_raw"], "p_mass_tf": measured["p_mass_tf"],
            "c_prob_error": None if measured["c_prob"] is not None else "no TRUE/FALSE mass",
            "c_prob_meta": measured["meta"]}


def _progress(layer: str, step: str, position: int, total: int) -> None:
    if position % PROGRESS_EVERY == 0 or position == total:
        print(f"{STAGE}: {layer}: {step} {position}/{total}", flush=True)


def _stop_if_all_calls_failed(layer: str, entries: Dict[str, Dict[str, Any]]) -> None:
    if entries and all(entry["failure_type"] == "call_error" for entry in entries.values()):
        first = next(iter(entries.values()))["call_error"]
        raise BackendError(f"{layer}: all {len(entries)} verdict calls failed with call_error, so judges.jsonl was "
                           f"not written (first error: {first})")


def run_layer(cruxes: List[Dict[str, Any]], layer: str, backend: Backend, registry: Registry, *, with_c_cons: bool,
              max_tokens: int, seed: int) -> Dict[str, Dict[str, Any]]:
    """One layer across all cruxes, keyed by crux_id. The model is released even if the layer stops early."""
    entries: Dict[str, Dict[str, Any]] = {}
    try:
        for position, crux in enumerate(cruxes, start=1):
            crux_id, claim = crux["crux_id"], crux["crux_claim"]
            sample = judge_crux_recorded(claim, model_key=layer, backend=backend, registry=registry,
                                         seed=stable_seed(seed, STAGE, "verdict", crux_id, layer),
                                         max_tokens=max_tokens, sampling_profile=VERDICT_PROFILES[layer])
            entries[crux_id] = {**sample, **c_prob_fields(claim, sample, model_key=layer, backend=backend),
                                "c_cons": None}
            _progress(layer, "verdicts", position, len(cruxes))
        _stop_if_all_calls_failed(layer, entries)  # before c_cons, which would add 5 failing calls per crux
        if with_c_cons:
            for position, crux in enumerate(cruxes, start=1):
                crux_id = crux["crux_id"]
                seeds = [stable_seed(seed, STAGE, "c_cons", crux_id, layer, index) for index in range(C_CONS_SAMPLES)]
                entries[crux_id]["c_cons"] = measure_c_cons(crux["crux_claim"], model_key=layer, backend=backend,
                                                            registry=registry, seeds=seeds, max_tokens=max_tokens)
                _progress(layer, "c_cons", position, len(cruxes))
    finally:
        backend.release(layer)
    return entries


def _differences(old: Any, new: Any, path: str = "") -> List[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        return [line for key in sorted(set(old) | set(new))
                for line in _differences(old.get(key), new.get(key), f"{path}.{key}" if path else key)]
    return [] if old == new else [f"{path}: checkpoint has {old!r}, this run has {new!r}"]


def _read_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("fingerprint"), dict):
        return None
    return checkpoint


def _checkpoint_problem(checkpoint: Dict[str, Any], layer: str, crux_ids: List[str]) -> Optional[str]:
    """Why a checkpoint with the right params still cannot be used, or None if it can."""
    entries = checkpoint.get("entries")
    written = checkpoint.get("written")
    if checkpoint.get("layer") != layer:
        return f"it holds layer {checkpoint.get('layer')!r}"
    if not isinstance(written, dict) or not {"code_git_sha", "created_at"} <= set(written):
        return "it does not say when it was written"
    if not isinstance(entries, dict) or set(entries) != set(crux_ids):
        return "its crux ids do not match this run"
    errors: List[str] = []
    for crux_id, entry in entries.items():
        check_judge_layer(entry, crux_id, errors)
    return "; ".join(errors[:3]) or None


class LayerCheckpoints:
    """Finished layers on disk, one JSON file per layer. Each is written atomically, so a file holds a whole layer."""

    def __init__(self, folder: Path, fingerprint: Dict[str, Any]):
        self.folder = Path(folder)
        self.fingerprint = json.loads(json.dumps(fingerprint))  # same shape as one read back from disk

    def load(self, crux_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Usable checkpoints by layer. Refuses if any file came from different params; a damaged file is ignored."""
        found: Dict[str, Dict[str, Any]] = {}
        for path in sorted(self.folder.glob("*.json")):
            checkpoint = _read_checkpoint(path)
            if checkpoint is None:
                print(f"{STAGE}: ignoring unreadable checkpoint {path.name}", flush=True)
                continue
            differences = _differences(checkpoint["fingerprint"], self.fingerprint)
            if differences:
                raise BackendError(f"cannot resume from {path}, its params differ: " + "; ".join(differences)
                                   + f". Rerun with the original params, or delete {self.folder} to start over.")
            problem = _checkpoint_problem(checkpoint, path.stem, crux_ids)
            if problem:
                print(f"{STAGE}: ignoring checkpoint {path.name}: {problem}", flush=True)
                continue
            found[path.stem] = checkpoint
        return found

    def save(self, layer: str, entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{layer}.json"
        checkpoint = {"layer": layer, "fingerprint": self.fingerprint, "written": stage_meta(STAGE),
                      "entries": entries}
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(checkpoint, handle, ensure_ascii=False)
        os.replace(tmp_path, path)
        return checkpoint


def clear_checkpoints(folder: Path) -> None:
    """The records hold everything the checkpoints do, and leftovers would refuse the next run with other params."""
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.name.endswith((".json", ".json.tmp")):
            path.unlink()
    if not any(folder.iterdir()):
        folder.rmdir()


def _layer_run(written: Dict[str, Any], *, resumed: bool) -> Dict[str, Any]:
    return {"code_git_sha": written["code_git_sha"], "created_at": written["created_at"], "resumed": resumed}


def evaluate(proposals: List[Dict[str, Any]], backend: Backend, registry: Registry, *,
             layers: Sequence[str] = DEFAULT_LAYERS, c_cons_layers: Sequence[str] = DEFAULT_C_CONS_LAYERS,
             max_tokens: int = DEFAULT_MAX_TOKENS, seed: int = 42, limit: Optional[int] = None,
             input_sha256: Optional[str] = None,
             checkpoint_folder: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Records in proposal order. ``limit`` and ``input_sha256`` only identify the run for its checkpoints."""
    layers, c_cons_layers = list(layers), list(c_cons_layers)
    check_config(layers, c_cons_layers, backend, registry)
    prompt = load_prompt(PROMPT_NAME)
    calls = {layer: layer_calls(layer, backend, registry, with_c_cons=layer in c_cons_layers, max_tokens=max_tokens)
             for layer in layers}
    cruxes = [p for p in proposals if p.get("parse_ok") and p.get("crux_claim")]

    checkpoints, finished = None, {}
    if checkpoint_folder is not None:
        checkpoints = LayerCheckpoints(checkpoint_folder, {
            "layers": layers, "c_cons_layers": c_cons_layers, "max_tokens": max_tokens, "seed": seed,
            "backend": backend.name, "limit": limit, "input_sha256": input_sha256,
            "prompt_sha256": prompt.sha256, "calls": calls})
        finished = checkpoints.load([crux["crux_id"] for crux in cruxes])

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    layer_runs: Dict[str, Dict[str, Any]] = {}
    for layer in layers:
        if layer in finished:
            print(f"{STAGE}: {layer}: resumed from checkpoint", flush=True)
            results[layer] = finished[layer]["entries"]
            layer_runs[layer] = _layer_run(finished[layer]["written"], resumed=True)
            continue
        print(f"{STAGE}: {layer}: starting on {len(cruxes)} cruxes", flush=True)
        results[layer] = run_layer(cruxes, layer, backend, registry, with_c_cons=layer in c_cons_layers,
                                   max_tokens=max_tokens, seed=seed)
        written = checkpoints.save(layer, results[layer])["written"] if checkpoints else stage_meta(STAGE)
        layer_runs[layer] = _layer_run(written, resumed=False)

    def record(proposal: Dict[str, Any]) -> Dict[str, Any]:
        crux_id = proposal["crux_id"]
        if proposal.get("parse_ok") and proposal.get("crux_claim"):
            body = {"status": "judged", "crux_claim": proposal["crux_claim"], "no_claim_reason": None,
                    "layers": {layer: results[layer][crux_id] for layer in layers}}
        else:
            body = {"status": "no_claim", "crux_claim": None, "no_claim_reason": proposal.get("failure_type"),
                    "layers": {}}
        return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "id": proposal["id"], "crux_id": crux_id,
                "proposed_by": proposal["proposed_by"], **body,
                "meta": stage_meta(STAGE, **prompt.meta(), calls=calls, layer_runs=layer_runs,
                                   params={"layers": layers, "c_cons_layers": c_cons_layers,
                                           "c_cons_samples": C_CONS_SAMPLES, "max_tokens": max_tokens,
                                           "base_seed": seed})}

    return [record(proposal) for proposal in proposals]


def fake_responder(model_key: str, messages: List[Dict[str, str]], call_index: int) -> str:
    """Canned 2A reply for offline plumbing. Depends only on the model and prompt, so layers can disagree."""
    tag = zlib.crc32(f"{model_key}|{messages[-1]['content']}".encode("utf-8"))
    return f"Fake reasoning.\nVERDICT: {'TRUE' if tag % 2 == 0 else 'FALSE'}"


def run_stage(*, run: str, backend_name: str, layers: Sequence[str] = DEFAULT_LAYERS,
              c_cons_layers: Sequence[str] = DEFAULT_C_CONS_LAYERS, limit: Optional[int] = None,
              max_tokens: int = DEFAULT_MAX_TOKENS, seed: int = 42, registry: Optional[Registry] = None) -> Path:
    registry = registry or Registry.load()
    backend = make_backend(backend_name, registry, fake_responder)
    source = run_path(run, "crux_proposals")
    input_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    proposals = read_jsonl(source, "crux_proposals")
    proposals = proposals[:limit] if limit is not None else proposals
    out = run_path(run, STAGE)
    folder = out.with_name(CHECKPOINT_FOLDER)
    records = evaluate(proposals, backend, registry, layers=layers, c_cons_layers=c_cons_layers,
                       max_tokens=max_tokens, seed=seed, limit=limit, input_sha256=input_sha256,
                       checkpoint_folder=folder)
    for record in records:
        record["meta"]["inputs"] = {"crux_proposals": str(source), "crux_proposals_sha256": input_sha256}
    write_jsonl(out, records, STAGE)
    clear_checkpoints(folder)
    judged = [r for r in records if r["status"] == "judged"]
    for layer in layers:
        print(f"{STAGE}: {layer}: {failure_summary([r['layers'][layer] for r in judged])}")
    print(f"{STAGE}: wrote {len(records)} records to {out} ({len(judged)} judged, "
          f"{len(records) - len(judged)} no claim)")
    return out
