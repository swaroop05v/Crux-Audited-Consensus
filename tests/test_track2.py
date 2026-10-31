"""Track 2 stage tests with the fake backend: no GPU, no network."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crux.backends import BackendError, Registry
from crux.backends.base import Backend
from crux.backends.fake import FakeBackend
from crux.backends.hf_local import HFLocalBackend, VerdictMarkerMissing
from crux.cli import main as cli_main
from crux.io import read_jsonl, stage_meta, write_jsonl
from crux.prompts import PromptError, load_prompt
from crux.schema import SCHEMA_VERSION, find_label_leaks, validate_record
from crux.track2 import judges
from crux.track2.checkability import MODEL_FAMILIES

REGISTRY = Registry.load()
L1, L2, GEMMA = judges.DEFAULT_LAYERS
TRUE, FALSE = "Reasoning.\nVERDICT: TRUE", "Reasoning.\nVERDICT: FALSE"
QUIET = contextlib.redirect_stdout(io.StringIO())


def setUpModule():
    # stage_meta starts a git process per call, and these runs make thousands of calls.
    patcher = mock.patch("crux.io.git_sha", return_value="0" * 40)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def proposal(number, side="a", ok=True, failure="parse_error"):
    return {"schema_version": SCHEMA_VERSION, "stage": "crux_proposals", "id": f"case{number}",
            "crux_id": f"case{number}#{side}", "proposed_by": f"consultant_{side}", "parse_ok": ok,
            "crux_claim": f"Claim number {number}{side} holds." if ok else None,
            "parse_error": None if ok else "bad reply", "failure_type": None if ok else failure,
            "raw_output": "raw", "call_error": None, "finish_reason": "stop",
            "disagreement_analysis": None, "proposer_own_verdict": None, "meta": stage_meta("crux_proposals")}


def claim(number, side="a"):
    return f"Claim number {number}{side} holds."


PROPOSALS = [proposal(0), proposal(0, "b"), proposal(1), proposal(1, "b")]


def by_claim(replies, default=TRUE):
    """A responder scripted on the claim text; a BackendError value is raised instead of returned."""
    def responder(model_key, messages, index):
        for text, reply in replies.items():
            if text in messages[-1]["content"]:
                if isinstance(reply, BackendError):
                    raise reply
                return reply
        return default
    return responder


class SpyBackend(FakeBackend):
    """Logs attempted calls and releases in order; a call that raises still counts as attempted."""

    def __init__(self, responder=TRUE, **options):
        super().__init__(options.pop("registry", REGISTRY), responder=responder, **options)
        self.events = []

    def _generate(self, spec, messages, **options):
        self.events.append(("call", spec.key))
        return super()._generate(spec, messages, **options)

    def release(self, model_key):
        self.events.append(("release", model_key))

    def attempts(self, layer):
        return self.events.count(("call", layer))


def run(proposals=PROPOSALS, backend=None, **options):
    with QUIET:
        return judges.evaluate(proposals, backend or SpyBackend(), options.pop("registry", REGISTRY), **options)


def comparable(record):
    """A record's layers without the wall-clock field, which is the one thing allowed to differ between runs."""
    layers = copy.deepcopy(record["layers"])
    for entry in layers.values():
        if entry["c_cons"] is not None:
            entry["c_cons"]["meta"].pop("created_at")
    return layers


def registry_with(**sampling):
    return Registry(REGISTRY.models, REGISTRY.backends, {**REGISTRY._sampling, **sampling})


def write_proposals(folder, proposals=PROPOSALS):
    write_jsonl(folder / "r" / "crux_proposals.jsonl", proposals, "crux_proposals")


class StageTest(unittest.TestCase):
    def test_records_are_valid_and_cover_every_layer(self):
        records = run()
        self.assertEqual([r["crux_id"] for r in records], [p["crux_id"] for p in PROPOSALS])
        for record in records:
            self.assertEqual(validate_record(record, "judges"), [])
            self.assertEqual(find_label_leaks(record), [])
            self.assertEqual(record["status"], "judged")
            self.assertEqual(list(record["layers"]), [L1, L2, GEMMA])
            self.assertEqual(record["crux_claim"], next(p["crux_claim"] for p in PROPOSALS
                                                        if p["crux_id"] == record["crux_id"]))
        entry = records[0]["layers"][L1]
        self.assertEqual(entry["verdict"], "TRUE")
        self.assertIsNone(entry["failure_type"])
        self.assertEqual(entry["model_call"]["model_key"], L1)
        self.assertAlmostEqual(entry["p_true"], 0.75)  # FakeBackend: p_true_raw 0.6, p_false_raw 0.2
        self.assertAlmostEqual(entry["c_prob"], 0.5)
        self.assertAlmostEqual(entry["p_mass_tf"], 0.8)
        self.assertIsNone(entry["c_prob_error"])
        self.assertEqual(entry["c_prob_meta"]["marker"], "VERDICT:")

    def test_c_cons_only_on_requested_layers(self):
        records = run()
        self.assertEqual(records[0]["layers"][L1]["c_cons"]["c_cons"], 1.0)
        self.assertEqual(len(records[0]["layers"][L1]["c_cons"]["samples"]), 5)
        self.assertIsNone(records[0]["layers"][L2]["c_cons"])

        backend = SpyBackend()
        records = run(backend=backend, layers=[L1, L2], c_cons_layers=[L2])
        self.assertIsNone(records[0]["layers"][L1]["c_cons"])
        self.assertIsNotNone(records[0]["layers"][L2]["c_cons"])
        self.assertEqual((backend.attempts(L1), backend.attempts(L2)), (4, 24))

    def test_empty_c_cons_layers_skips_it(self):
        backend = SpyBackend()
        run(backend=backend, layers=[L1], c_cons_layers=[])
        self.assertEqual(backend.attempts(L1), 4)

    def test_prompts_carry_the_claim_and_nothing_from_the_case(self):
        backend = SpyBackend()
        run(backend=backend, layers=[L1])
        expected = load_prompt("judge_crux").messages(crux_claim=claim(0))
        self.assertEqual(backend.calls[0]["messages"], expected)

    def test_leaky_cruxes_are_judged(self):
        """The stage never reads leakage.jsonl, so a crux flagged leaky is judged like any other."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)), QUIET:
            folder = Path(tmp)
            write_proposals(folder)
            write_jsonl(folder / "r" / "leakage.jsonl", [
                {"schema_version": SCHEMA_VERSION, "stage": "leakage", "id": p["id"], "crux_id": p["crux_id"],
                 "proposed_by": p["proposed_by"], "status": "checked", "l1": {"flagged": True, "reasons": []},
                 "p5_flags": {}, "l2": None, "leak_probe_failed": None, "is_leaky": True,
                 "meta": stage_meta("leakage")} for p in PROPOSALS], "leakage")
            cli_main(["track2", "judge", "--run", "r", "--backend", "fake", "--layers", L1])
            records = read_jsonl(folder / "r" / "judges.jsonl", "judges")
        self.assertEqual([r["status"] for r in records], ["judged"] * 4)
        self.assertTrue(all(r["layers"][L1]["verdict"] for r in records))

    def test_no_claim_crux_is_recorded_and_skipped(self):
        proposals = [proposal(0), proposal(0, "b", ok=False, failure="truncated")]
        backend = SpyBackend()
        records = run(proposals, backend, layers=[L1], c_cons_layers=[])
        self.assertEqual(backend.attempts(L1), 1)
        skipped = records[1]
        self.assertEqual(validate_record(skipped, "judges"), [])
        self.assertEqual((skipped["status"], skipped["no_claim_reason"]), ("no_claim", "truncated"))
        self.assertEqual(skipped["layers"], {})

    def test_all_no_claim_is_not_a_systematic_failure(self):
        records = run([proposal(0, ok=False)], layers=[L1])
        self.assertEqual(records[0]["status"], "no_claim")


class SamplingProfileTest(unittest.TestCase):
    def setUp(self):
        self.registry = registry_with(judge_verdict={"temperature": 0.1, "top_p": None},
                                      judge_layer2={"temperature": 0.2, "top_p": None},
                                      judge_cons={"temperature": 0.3, "top_p": None})

    def test_each_layer_uses_its_profile_and_c_cons_its_own(self):
        backend = SpyBackend(registry=self.registry)
        records = run(PROPOSALS[:1], backend, registry=self.registry)
        used = [(call["model_key"], call["temperature"]) for call in backend.calls]
        self.assertEqual(used, [(L1, 0.1)] + [(L1, 0.3)] * 5 + [(L2, 0.2), (GEMMA, 0.2)])

        record = records[0]
        calls = record["meta"]["calls"]
        self.assertEqual(calls[L1]["sampling"]["verdict"], {"profile": "judge_verdict", "temperature": 0.1,
                                                            "top_p": None})
        self.assertEqual(calls[L2]["sampling"]["verdict"]["profile"], "judge_layer2")
        self.assertEqual(calls[GEMMA]["sampling"]["verdict"]["profile"], "judge_layer2")
        self.assertEqual(calls[L1]["sampling"]["c_cons"], {"profile": "judge_cons", "temperature": 0.3, "top_p": None})
        self.assertIsNone(calls[L2]["sampling"]["c_cons"])
        self.assertEqual(record["layers"][L1]["c_cons"]["meta"]["temperature"], 0.3)

    def test_every_judge_has_a_profile(self):
        self.assertEqual(set(judges.VERDICT_PROFILES), set(MODEL_FAMILIES))
        self.assertEqual(set(judges.DEFAULT_LAYERS), set(judges.VERDICT_PROFILES))


class FailureTest(unittest.TestCase):
    def test_each_verdict_failure_is_recorded_without_crashing(self):
        proposals = [proposal(n) for n in range(4)]
        backend = SpyBackend(
            by_claim({claim(1): "I think it is true.", claim(2): BackendError("HTTP 503"), claim(3): "No marker here"}),
            finish_reason=lambda key, index: "length" if index == 0 else "stop")
        records = run(proposals, backend, layers=[L1], c_cons_layers=[])
        entries = [r["layers"][L1] for r in records]
        for record in records:
            self.assertEqual(validate_record(record, "judges"), [], record["crux_id"])

        truncated, unparsed, failed, unmarked = entries
        self.assertEqual((truncated["verdict"], truncated["failure_type"]), (None, "truncated"))
        self.assertEqual(truncated["finish_reason"], "length")
        self.assertEqual((unparsed["verdict"], unparsed["failure_type"]), (None, "parse_error"))
        self.assertEqual(unparsed["raw_output"], "I think it is true.")
        self.assertEqual((failed["verdict"], failed["failure_type"]), (None, "call_error"))
        self.assertEqual(failed["call_error"], "HTTP 503")
        self.assertIsNone(failed["model_call"])
        self.assertEqual((unmarked["verdict"], unmarked["failure_type"]), (None, "parse_error"))
        for entry in entries:
            self.assertIsNone(entry["c_prob"])
            self.assertIn("no verdict", entry["c_prob_error"])
        self.assertIn("truncated", truncated["c_prob_error"])

    def test_truncated_reply_with_a_verdict_is_not_trusted(self):
        backend = SpyBackend(TRUE, finish_reason="length")
        entry = run(PROPOSALS[:1], backend, layers=[L1], c_cons_layers=[])[0]["layers"][L1]
        self.assertIsNone(entry["verdict"])
        self.assertEqual(entry["failure_type"], "truncated")

    def test_c_cons_sample_failures_are_recorded(self):
        replies = iter([TRUE, TRUE, "no verdict", BackendError("HTTP 503"), FALSE, TRUE])

        def responder(model_key, messages, index):
            reply = next(replies)
            if isinstance(reply, BackendError):
                raise reply
            return reply

        record = run(PROPOSALS[:1], SpyBackend(responder), layers=[L1])[0]
        self.assertEqual(validate_record(record, "judges"), [])
        c_cons = record["layers"][L1]["c_cons"]
        self.assertEqual([s["failure_type"] for s in c_cons["samples"]],
                         [None, "parse_error", "call_error", None, None])
        self.assertEqual((c_cons["n_valid"], c_cons["n_failed"]), (3, 2))
        self.assertAlmostEqual(c_cons["c_cons"], 1 / 3)

    def test_c_cons_still_runs_when_the_verdict_failed(self):
        replies = iter(["no verdict"] + [TRUE] * 5)
        record = run(PROPOSALS[:1], SpyBackend(lambda *_: next(replies)), layers=[L1])[0]
        self.assertIsNone(record["layers"][L1]["verdict"])
        self.assertEqual(record["layers"][L1]["c_cons"]["c_cons"], 1.0)

    def test_c_prob_failure_keeps_the_verdict(self):
        backend = SpyBackend()
        with mock.patch.object(backend, "verdict_probs", side_effect=VerdictMarkerMissing("marker not found")):
            record = run(PROPOSALS[:1], backend, layers=[L1], c_cons_layers=[])[0]
        self.assertEqual(validate_record(record, "judges"), [])
        entry = record["layers"][L1]
        self.assertEqual(entry["verdict"], "TRUE")
        self.assertIsNone(entry["c_prob"])
        self.assertEqual(entry["c_prob_error"], "marker not found")

    def test_backend_without_full_distributions_is_recorded_as_unsupported(self):
        class NoDistribution(SpyBackend):
            verdict_probs = Backend.verdict_probs

        entry = run(PROPOSALS[:1], NoDistribution(), layers=[L1], c_cons_layers=[])[0]["layers"][L1]
        self.assertEqual(entry["verdict"], "TRUE")
        self.assertIsNone(entry["c_prob"])
        self.assertIn("unsupported", entry["c_prob_error"])

    def test_zero_mass_gives_no_c_prob(self):
        backend = SpyBackend(verdict={"p_true_raw": 0.0, "p_false_raw": 0.0})
        entry = run(PROPOSALS[:1], backend, layers=[L1], c_cons_layers=[])[0]["layers"][L1]
        self.assertIsNone(entry["c_prob"])
        self.assertEqual(entry["p_mass_tf"], 0.0)
        self.assertIn("no TRUE/FALSE mass", entry["c_prob_error"])


class SystematicFailureTest(unittest.TestCase):
    def failing(self, broken_layer):
        def responder(model_key, messages, index):
            if model_key == broken_layer:
                raise BackendError("model would not load")
            return TRUE
        return SpyBackend(responder)

    def test_abort_names_the_layer_and_stops(self):
        backend = self.failing(L1)
        with self.assertRaisesRegex(BackendError, f"{L1}: all 4 verdict calls failed.*model would not load"):
            run(backend=backend)
        self.assertEqual(backend.attempts(L1), 4)  # no c_cons calls after the verdict pass failed
        self.assertEqual(backend.attempts(L2), 0)
        self.assertIn(("release", L1), backend.events)

    def test_abort_writes_no_judges_file_and_keeps_earlier_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)), QUIET:
            folder = Path(tmp)
            write_proposals(folder)
            backend = self.failing(L2)
            with mock.patch("crux.track2.judges.make_backend", return_value=backend):
                with self.assertRaisesRegex(BackendError, L2):
                    judges.run_stage(run="r", backend_name="fake")
            self.assertFalse((folder / "r" / "judges.jsonl").exists())
            names = sorted(p.name for p in (folder / "r" / judges.CHECKPOINT_FOLDER).iterdir())
            self.assertEqual(names, [f"{L1}.json"])

    def test_partial_failure_does_not_abort(self):
        backend = SpyBackend(by_claim({claim(n): BackendError("HTTP 503") for n in (0,)}))
        records = run([proposal(0), proposal(1)], backend, layers=[L1], c_cons_layers=[])
        self.assertEqual([r["layers"][L1]["failure_type"] for r in records], ["call_error", None])

    def test_all_parse_errors_do_not_abort(self):
        records = run(backend=SpyBackend("no verdict here"), layers=[L1], c_cons_layers=[])
        self.assertTrue(all(r["layers"][L1]["failure_type"] == "parse_error" for r in records))


class ConfigTest(unittest.TestCase):
    def assert_refused(self, exception, backend=None, **options):
        backend = backend or SpyBackend()
        with self.assertRaises(exception):
            run(backend=backend, **options)
        self.assertEqual(backend.events, [])

    def test_bad_layer_lists(self):
        self.assert_refused(ValueError, layers=[])
        self.assert_refused(ValueError, layers=["consultant_a"], c_cons_layers=[])
        self.assert_refused(ValueError, layers=[L1, L1], c_cons_layers=[])
        self.assert_refused(ValueError, layers=[L1], c_cons_layers=[L2])
        self.assert_refused(ValueError, layers=[L1], c_cons_layers=[L1, L1])

    def test_backend_a_judge_is_not_pinned_to(self):
        class WrongBackend(SpyBackend):
            name = "vllm"
            spec = Backend.spec

        self.assert_refused(BackendError, WrongBackend(), layers=[L1])

    def test_missing_sampling_profiles(self):
        for missing in ("judge_verdict", "judge_cons"):
            with self.subTest(missing=missing):
                registry = Registry(REGISTRY.models, REGISTRY.backends,
                                    {k: v for k, v in REGISTRY._sampling.items() if k != missing})
                self.assert_refused(BackendError, SpyBackend(registry=registry), registry=registry, layers=[L1])
        registry = Registry(REGISTRY.models, REGISTRY.backends,
                            {k: v for k, v in REGISTRY._sampling.items() if k != "judge_layer2"})
        self.assert_refused(BackendError, SpyBackend(registry=registry), registry=registry, layers=[L1, L2])

    def test_missing_prompt_file(self):
        with tempfile.TemporaryDirectory() as empty, mock.patch("crux.prompts.PROMPTS_DIR", Path(empty)):
            self.assert_refused(PromptError)

    def test_max_tokens_below_one_raises(self):
        with self.assertRaises(ValueError):
            run(max_tokens=0, layers=[L1])

    def test_hf_local_model_pinning_is_not_swallowed(self):
        with self.assertRaisesRegex(ValueError, "not judge layers"):
            run(backend=HFLocalBackend(REGISTRY), layers=["consultant_a"], c_cons_layers=[])


class SeedTest(unittest.TestCase):
    def test_seeds_are_distinct_across_cruxes_layers_and_samples(self):
        records = run(c_cons_layers=[L1, L2, GEMMA])
        seeds = [entry["seed"] for r in records for entry in r["layers"].values()]
        seeds += [s["seed"] for r in records for e in r["layers"].values() for s in e["c_cons"]["samples"]]
        self.assertEqual(len(seeds), 4 * 3 * 6)
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_output_does_not_depend_on_crux_or_layer_order(self):
        base = {r["crux_id"]: comparable(r) for r in run()}
        reversed_cruxes = {r["crux_id"]: comparable(r) for r in run(list(reversed(PROPOSALS)))}
        reversed_layers = {r["crux_id"]: comparable(r) for r in run(layers=list(reversed(judges.DEFAULT_LAYERS)))}
        self.assertEqual(reversed_cruxes, base)
        self.assertEqual(reversed_layers, base)

    def test_base_seed_changes_every_call_seed(self):
        first, second = run(seed=1)[0]["layers"][L1], run(seed=2)[0]["layers"][L1]
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertNotEqual(first["c_cons"]["meta"]["seeds"], second["c_cons"]["meta"]["seeds"])

    def test_same_seed_repeats(self):
        self.assertEqual([comparable(r) for r in run()], [comparable(r) for r in run()])


class OneLayerAtATimeTest(unittest.TestCase):
    def test_each_model_is_released_before_the_next_starts(self):
        backend = SpyBackend()
        run(PROPOSALS[:2], backend, c_cons_layers=[L1])
        expected = ([("call", L1)] * 12 + [("release", L1)] + [("call", L2)] * 2 + [("release", L2)]
                    + [("call", GEMMA)] * 2 + [("release", GEMMA)])
        self.assertEqual(backend.events, expected)


class ResumeTest(unittest.TestCase):
    """Resume through run_stage, the way a restarted command reaches it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch("crux.track1.common.RUNS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = Path(self.tmp.name)
        write_proposals(self.folder)
        self.checkpoints = self.folder / "r" / judges.CHECKPOINT_FOLDER
        self.out = self.folder / "r" / "judges.jsonl"

    def stage(self, backend=None, **options):
        backend = backend or SpyBackend()
        with mock.patch("crux.track2.judges.make_backend", return_value=backend), QUIET:
            judges.run_stage(run="r", backend_name="fake", **options)
        return backend

    def stop_at(self, layer, **options):
        """A run that aborts on ``layer``, leaving the layers before it in the checkpoints."""
        def responder(model_key, messages, index):
            if model_key == layer:
                raise BackendError("no such luck")
            return TRUE

        with self.assertRaises(BackendError):
            self.stage(SpyBackend(responder), **options)

    def test_resume_skips_a_finished_layer_and_matches_an_uninterrupted_run(self):
        self.stop_at(L2)
        backend = self.stage()
        self.assertEqual(backend.attempts(L1), 0)
        self.assertNotIn(("release", L1), backend.events)
        self.assertEqual((backend.attempts(L2), backend.attempts(GEMMA)), (4, 4))
        resumed = read_jsonl(self.out, "judges")
        self.assertTrue(resumed[0]["meta"]["layer_runs"][L1]["resumed"])
        self.assertFalse(resumed[0]["meta"]["layer_runs"][L2]["resumed"])
        self.assertTrue(resumed[0]["meta"]["layer_runs"][L1]["created_at"])

        self.out.unlink()
        self.stage()
        straight = read_jsonl(self.out, "judges")
        self.assertEqual([comparable(r) for r in resumed], [comparable(r) for r in straight])
        self.assertFalse(any(run_["resumed"] for run_ in straight[0]["meta"]["layer_runs"].values()))

    def test_crash_in_a_later_layer_keeps_earlier_ones(self):
        class Crash(SpyBackend):
            def _generate(self, spec, messages, **options):
                if spec.key == GEMMA:
                    raise KeyboardInterrupt
                return super()._generate(spec, messages, **options)

        with self.assertRaises(KeyboardInterrupt):
            self.stage(Crash())
        self.assertEqual(sorted(p.stem for p in self.checkpoints.iterdir()), sorted([L1, L2]))
        self.assertEqual(self.stage().attempts(L2), 0)

    def test_checkpoints_are_removed_after_judges_file_is_written(self):
        self.stage()
        self.assertTrue(self.out.exists())
        self.assertFalse(self.checkpoints.exists())

    def test_smoke_run_does_not_block_the_full_run(self):
        self.stage(limit=2)
        self.assertEqual(len(read_jsonl(self.out, "judges")), 2)
        self.stage()
        self.assertEqual(len(read_jsonl(self.out, "judges")), 4)

    def test_different_params_refuse_and_name_the_param(self):
        changes = {
            "layers": dict(layers=[L1, L2]),
            "c_cons_layers": dict(c_cons_layers=[]),
            "max_tokens": dict(max_tokens=99),
            "seed": dict(seed=7),
            "limit": dict(limit=2),
        }
        self.stop_at(L2)
        for param, options in changes.items():
            with self.subTest(param=param):
                backend = SpyBackend()
                with self.assertRaisesRegex(BackendError, param) as caught:
                    self.stage(backend, **options)
                self.assertEqual(backend.events, [])
                self.assertIn(str(self.checkpoints), str(caught.exception))
        self.assertFalse(self.out.exists())

    def test_different_layer_order_refuses(self):
        self.stop_at(L2)
        with self.assertRaisesRegex(BackendError, "layers"):
            self.stage(layers=[L2, L1, GEMMA])

    def test_different_backend_refuses(self):
        self.stop_at(L2)
        other = SpyBackend()
        other.name = "fake2"
        with self.assertRaisesRegex(BackendError, "backend"):
            self.stage(other)
        self.assertEqual(other.events, [])

    def test_changed_input_file_refuses(self):
        self.stop_at(L2)
        write_proposals(self.folder, PROPOSALS[:3])
        with self.assertRaisesRegex(BackendError, "input_sha256"):
            self.stage()

    def test_changed_prompt_refuses(self):
        self.stop_at(L2)
        real = load_prompt("judge_crux")
        edited = type(real)(real.name, real.version, real.text, "0" * 64, real.path)
        with mock.patch("crux.track2.judges.load_prompt", return_value=edited):
            with self.assertRaisesRegex(BackendError, "prompt_sha256"):
                self.stage()

    def test_changed_model_revision_refuses(self):
        self.stop_at(L2)
        models = dict(REGISTRY.models)
        spec = models[L1]
        models[L1] = type(spec)(spec.key, spec.hf_repo, "1" * 40, spec.backend, spec.dtype, spec.quantization)
        registry = Registry(models, REGISTRY.backends, REGISTRY._sampling)
        with self.assertRaisesRegex(BackendError, f"calls.{L1}.revision"):
            self.stage(SpyBackend(registry=registry), registry=registry)

    def test_changed_sampling_value_refuses(self):
        self.stop_at(L2)
        registry = registry_with(judge_verdict={"temperature": 0.4, "top_p": None})
        with self.assertRaisesRegex(BackendError, "temperature"):
            self.stage(SpyBackend(registry=registry), registry=registry)

    def test_unreadable_checkpoint_reruns_that_layer(self):
        self.stop_at(L2)
        (self.checkpoints / f"{L1}.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(self.stage().attempts(L1), 24)

    def test_checkpoint_with_other_cruxes_reruns_that_layer(self):
        self.stop_at(L2)
        path = self.checkpoints / f"{L1}.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        del checkpoint["entries"][PROPOSALS[0]["crux_id"]]
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        self.assertEqual(self.stage().attempts(L1), 24)

    def test_checkpoint_with_a_bad_entry_reruns_that_layer(self):
        self.stop_at(L2)
        path = self.checkpoints / f"{L1}.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        checkpoint["entries"][PROPOSALS[0]["crux_id"]]["verdict"] = "MAYBE"
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        self.assertEqual(self.stage().attempts(L1), 24)

    def test_checkpoint_without_its_provenance_reruns_that_layer(self):
        self.stop_at(L2)
        path = self.checkpoints / f"{L1}.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        del checkpoint["written"]
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        self.assertEqual(self.stage().attempts(L1), 24)

    def test_failed_cruxes_in_a_finished_layer_are_not_retried(self):
        def responder(model_key, messages, index):
            if model_key == L2:
                raise BackendError("no such luck")
            if claim(0) in messages[-1]["content"]:
                raise BackendError("HTTP 503")
            return TRUE

        with self.assertRaises(BackendError):
            self.stage(SpyBackend(responder), layers=[L1, L2], c_cons_layers=[])
        backend = self.stage(layers=[L1, L2], c_cons_layers=[])
        self.assertEqual(backend.attempts(L1), 0)
        record = read_jsonl(self.out, "judges")[0]
        self.assertEqual(record["layers"][L1]["failure_type"], "call_error")


class MetaTest(unittest.TestCase):
    def test_meta_records_how_each_layer_ran(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)), QUIET:
            write_proposals(Path(tmp))
            judges.run_stage(run="r", backend_name="fake", max_tokens=64, seed=9)
            meta = read_jsonl(Path(tmp) / "r" / "judges.jsonl", "judges")[0]["meta"]
            source = str(Path(tmp) / "r" / "crux_proposals.jsonl")

        spec = REGISTRY.spec(L1)
        call = meta["calls"][L1]
        self.assertEqual((call["model_key"], call["revision"], call["quantization"]), (L1, spec.revision, "none"))
        self.assertEqual((call["backend"], call["max_tokens"]), ("fake", 64))
        self.assertEqual(call["sampling"]["verdict"]["temperature"],
                         REGISTRY.sampling("judge_verdict")["temperature"])
        prompt = load_prompt("judge_crux")
        self.assertEqual((meta["prompt_name"], meta["prompt_version"], meta["prompt_sha256"]),
                         ("judge_crux", "v1", prompt.sha256))
        self.assertIn("code_git_sha", meta)
        self.assertEqual(meta["inputs"]["crux_proposals"], source)
        self.assertEqual(len(meta["inputs"]["crux_proposals_sha256"]), 64)
        self.assertEqual(meta["params"], {"layers": list(judges.DEFAULT_LAYERS),
                                          "c_cons_layers": list(judges.DEFAULT_C_CONS_LAYERS),
                                          "c_cons_samples": 5, "max_tokens": 64, "base_seed": 9})
        self.assertEqual(meta["layer_runs"][L1]["resumed"], False)

    def test_meta_is_complete_when_every_call_failed(self):
        backend = SpyBackend(by_claim({claim(0): BackendError("HTTP 503")}))
        record = run([proposal(0), proposal(1)], backend, layers=[L1], c_cons_layers=[])[0]
        self.assertIsNone(record["layers"][L1]["model_call"])
        self.assertEqual(record["meta"]["calls"][L1]["revision"], REGISTRY.spec(L1).revision)


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.good = run(PROPOSALS[:1], layers=[L1])[0]

    def invalid(self, change):
        record = copy.deepcopy(self.good)
        change(record)
        self.assertNotEqual(validate_record(record, "judges"), [])

    def test_good_records_pass(self):
        self.assertEqual(validate_record(self.good, "judges"), [])
        skipped = run([proposal(0, ok=False)], layers=[L1])[0]
        self.assertEqual(validate_record(skipped, "judges"), [])

    def test_bad_records_are_rejected(self):
        def entry(record):
            return record["layers"][L1]

        changes = {
            "status": lambda r: r.update(status="done"),
            "crux_id": lambda r: r.update(crux_id="case0"),
            "proposed_by": lambda r: r.update(proposed_by="consultant_b"),
            "blank claim": lambda r: r.update(crux_claim="  "),
            "no layers": lambda r: r.update(layers={}),
            "layer not an object": lambda r: r["layers"].update({L1: "TRUE"}),
            "verdict value": lambda r: entry(r).update(verdict="MAYBE"),
            "verdict without parse_ok": lambda r: entry(r).update(parse_ok=False, failure_type="parse_error"),
            "no verdict with parse_ok": lambda r: entry(r).update(verdict=None),
            "failure type": lambda r: entry(r).update(failure_type="oops"),
            "parse_ok not bool": lambda r: entry(r).update(parse_ok="yes"),
            "failure with parse_ok": lambda r: entry(r).update(failure_type="parse_error"),
            "seed": lambda r: entry(r).update(seed="1"),
            "raw output": lambda r: entry(r).update(raw_output=5),
            "c_prob range": lambda r: entry(r).update(c_prob=1.5),
            "c_prob type": lambda r: entry(r).update(c_prob=True),
            "p_true range": lambda r: entry(r).update(p_true=-0.1),
            "mass range": lambda r: entry(r).update(p_mass_tf=1.1),
            "c_cons shape": lambda r: entry(r).update(c_cons=0.5),
            "c_cons range": lambda r: entry(r)["c_cons"].update(c_cons=2),
            "c_cons sample": lambda r: entry(r)["c_cons"]["samples"].__setitem__(0, "TRUE"),
            "c_cons sample failure": lambda r: entry(r)["c_cons"]["samples"][0].update(failure_type="oops"),
            "label leak": lambda r: entry(r).update(ground_truth="A"),
            "meta stage": lambda r: r["meta"].update(stage="other"),
        }
        for name, change in changes.items():
            with self.subTest(name):
                self.invalid(change)

    def test_c_prob_without_a_verdict_is_rejected(self):
        def change(record):
            record["layers"][L1].update(verdict=None, parse_ok=False, failure_type="parse_error")

        self.invalid(change)  # c_prob is still set from the good record

    def test_no_claim_with_results_is_rejected(self):
        skipped = run([proposal(0, ok=False)], layers=[L1])[0]
        self.assertNotEqual(validate_record({**skipped, "layers": self.good["layers"]}, "judges"), [])
        self.assertNotEqual(validate_record({**skipped, "crux_claim": "x"}, "judges"), [])

    def test_mass_just_above_one_is_a_float_sum_not_an_error(self):
        record = copy.deepcopy(self.good)
        record["layers"][L1]["p_mass_tf"] = 1.0000001
        self.assertEqual(validate_record(record, "judges"), [])

    def test_stage_key_catches_duplicate_crux_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "duplicate crux_id"):
                write_jsonl(Path(tmp) / "judges.jsonl", [self.good, self.good], "judges")


def entry(verdict, c_prob=None, mass=None, c_cons=None, failure=None):
    return {"seed": 1, "verdict": verdict, "parse_ok": verdict is not None, "parse_error": None,
            "failure_type": failure, "raw_output": "x", "call_error": None, "finish_reason": "stop",
            "model_call": None, "c_prob": c_prob, "p_true": None, "p_true_raw": None, "p_false_raw": None,
            "p_mass_tf": mass, "c_prob_error": None, "c_prob_meta": None, "c_cons": c_cons}


def cons(value, failed=0):
    """A c_cons block with ``failed`` failed samples out of five."""
    samples = [{"verdict": "TRUE", "parse_ok": True, "failure_type": None}] * (5 - failed)
    samples += [{"verdict": None, "parse_ok": False, "failure_type": "parse_error"}] * failed
    return {"samples": samples, "c_cons": value}


def judged(number, side, **layers):
    return {"schema_version": SCHEMA_VERSION, "stage": "judges", "id": f"case{number}",
            "crux_id": f"case{number}#{side}", "proposed_by": f"consultant_{side}", "status": "judged",
            "crux_claim": "A claim.", "no_claim_reason": None, "layers": {key: value for key, value in layers.items()},
            "meta": stage_meta("judges")}


def leak(number, side, is_leaky):
    return {"schema_version": SCHEMA_VERSION, "stage": "leakage", "id": f"case{number}",
            "crux_id": f"case{number}#{side}", "proposed_by": f"consultant_{side}", "status": "checked",
            "l1": {"flagged": is_leaky, "reasons": []}, "p5_flags": {}, "l2": None, "leak_probe_failed": None,
            "is_leaky": is_leaky, "meta": stage_meta("leakage")}


def fixture_records():
    """Four judged cruxes and one with no claim. Every number in ReportTest is worked out from this by hand."""
    return [
        judged(0, "a", judge_l1_qwen=entry("TRUE", 0.9, 0.95, cons(1.0)),
               judge_l2_llama=entry("TRUE", 0.5, 0.4), judge_l2_gemma=entry("FALSE", 0.1, 0.9)),
        judged(0, "b", judge_l1_qwen=entry("TRUE", 0.25, 0.8, cons(0.6)),
               judge_l2_llama=entry("FALSE", 0.7, 0.9), judge_l2_gemma=entry("FALSE", 0.1, 0.9)),
        judged(1, "a", judge_l1_qwen=entry("FALSE", 0.05, 0.6, cons(0.2, failed=1)),
               judge_l2_llama=entry(None, failure="parse_error"), judge_l2_gemma=entry("TRUE", 0.1, 0.9)),
        judged(1, "b", judge_l1_qwen=entry(None, failure="call_error", c_cons=cons(None, failed=5)),
               judge_l2_llama=entry("TRUE", 0.1, 0.3), judge_l2_gemma=entry(None, failure="truncated")),
        {"schema_version": SCHEMA_VERSION, "stage": "judges", "id": "case2", "crux_id": "case2#a",
         "proposed_by": "consultant_a", "status": "no_claim", "crux_claim": None, "no_claim_reason": "parse_error",
         "layers": {}, "meta": stage_meta("judges")},
    ]


class ReportTest(unittest.TestCase):
    def report(self, *, leakage=True):
        from crux.track2 import report

        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)), QUIET:
            write_jsonl(Path(tmp) / "r" / "judges.jsonl", fixture_records(), "judges")
            if leakage:
                write_jsonl(Path(tmp) / "r" / "leakage.jsonl",
                            [leak(0, "a", False), leak(0, "b", True), leak(1, "a", False)], "leakage")
            cli_main(["track2", "report", "--run", "r"])
            result = json.loads((Path(tmp) / "r" / report.REPORT_NAME).read_text(encoding="utf-8"))
        return result

    def test_counts(self):
        result = self.report()
        self.assertEqual((result["n_records"], result["n_judged"], result["no_claim"]), (5, 4, 1))
        self.assertEqual(result["low_mass_threshold"], 0.5)
        self.assertEqual(list(result["layers"]), [L1, L2, GEMMA])

    def test_true_rates_overall_and_by_leakage(self):
        l1 = self.report()["layers"][L1]
        self.assertEqual((l1["true_rate"]["count"], l1["true_rate"]["of"], l1["true_rate"]["rate"]), (2, 3, 0.6667))
        self.assertEqual((l1["true_rate_non_leaky"]["count"], l1["true_rate_non_leaky"]["of"]), (1, 2))
        self.assertEqual((l1["true_rate_leaky"]["count"], l1["true_rate_leaky"]["of"]), (1, 1))
        self.assertEqual(l1["leakage_unknown"], 1)

    def test_failures(self):
        layers = self.report()["layers"]
        counts = {layer: {kind: v["count"] for kind, v in layers[layer]["failures"].items()} for layer in layers}
        self.assertEqual(counts[L1], {"call_error": 1, "truncated": 0, "parse_error": 0})
        self.assertEqual(counts[L2], {"call_error": 0, "truncated": 0, "parse_error": 1})
        self.assertEqual(counts[GEMMA], {"call_error": 0, "truncated": 1, "parse_error": 0})
        self.assertEqual(layers[L1]["failures"]["call_error"]["rate"], 0.25)

    def test_mass(self):
        layers = self.report()["layers"]
        l1, l2 = layers[L1]["p_mass_tf"], layers[L2]["p_mass_tf"]
        self.assertEqual((l1["n"], l1["mean"], l1["median"], l1["min"]), (3, 0.7833, 0.8, 0.6))
        self.assertEqual((l1["low_mass"]["count"], l1["low_mass"]["of"]), (0, 3))
        self.assertEqual((l2["low_mass"]["count"], l2["low_mass"]["of"]), (2, 3))

    def test_c_prob_distribution(self):
        layers = self.report()["layers"]
        c_prob = layers[L1]["c_prob"]
        self.assertEqual((c_prob["n"], c_prob["mean"], c_prob["median"]), (3, 0.4, 0.25))
        self.assertEqual(c_prob["histogram"], {"0.0-0.1": 1, "0.2-0.3": 1, "0.9-1.0": 1})
        self.assertEqual(layers[L1]["c_prob_missing"], 1)

    def test_c_prob_of_exactly_one_lands_in_the_top_bin(self):
        from crux.track2.report import _binned_stats

        self.assertEqual(_binned_stats([1.0])["histogram"], {"0.9-1.0": 1})
        self.assertEqual(_binned_stats([]), {"n": 0})

    def test_c_cons_distribution(self):
        layers = self.report()["layers"]
        c_cons = layers[L1]["c_cons"]
        self.assertEqual((c_cons["values"]["n"], c_cons["values"]["mean"], c_cons["values"]["median"]), (3, 0.6, 0.6))
        self.assertEqual(c_cons["values"]["histogram"], {"0.20": 1, "0.60": 1, "1.00": 1})
        self.assertEqual(c_cons["no_valid_samples"], 1)
        self.assertEqual(c_cons["sample_failures"]["parse_error"]["count"], 6)
        self.assertEqual(c_cons["sample_failures"]["parse_error"]["of"], 20)
        self.assertNotIn("c_cons", layers[L2])

    def test_pairwise_agreement_uses_cruxes_where_both_are_valid(self):
        pairs = self.report()["pairwise_agreement"]
        self.assertEqual((pairs[f"{L1} vs {L2}"]["count"], pairs[f"{L1} vs {L2}"]["of"]), (1, 2))
        self.assertEqual((pairs[f"{L1} vs {GEMMA}"]["count"], pairs[f"{L1} vs {GEMMA}"]["of"]), (0, 3))
        self.assertEqual((pairs[f"{L2} vs {GEMMA}"]["count"], pairs[f"{L2} vs {GEMMA}"]["of"]), (1, 2))

    def test_without_leakage_file_the_split_is_skipped(self):
        result = self.report(leakage=False)
        self.assertFalse(result["leakage_joined"])
        self.assertNotIn("true_rate_leaky", result["layers"][L1])
        self.assertNotIn("leakage_unknown", result["layers"][L1])

    def test_missing_judges_file_raises(self):
        from crux.track2 import report

        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)):
            with self.assertRaises(FileNotFoundError):
                report.build_report("r")


class CliTest(unittest.TestCase):
    def test_tiny_run_through_track1_judge_and_report(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)), QUIET:
            cli_main(["track1", "propose", "--run", "t", "--backend", "fake", "--limit", "2"])
            cli_main(["track1", "leakage", "--run", "t"])
            cli_main(["track2", "judge", "--run", "t", "--backend", "fake"])
            cli_main(["track2", "report", "--run", "t"])
            folder = Path(tmp) / "t"
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ["crux_proposals.jsonl", "judges.jsonl", "leakage.jsonl", "track2_report.json"])
            records = read_jsonl(folder / "judges.jsonl", "judges")
            report = json.loads((folder / "track2_report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(records), 4)
        self.assertTrue(all(r["status"] == "judged" for r in records))
        self.assertEqual(report["n_judged"], 4)
        self.assertEqual(report["layers"][L1]["c_cons"]["values"]["n"], 4)
        self.assertEqual(report["layers"][L1]["p_mass_tf"]["low_mass"]["count"], 0)

    def test_options_reach_the_stage(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)), QUIET:
            write_proposals(Path(tmp))
            cli_main(["track2", "judge", "--run", "r", "--backend", "fake", "--layers", f"{L1}, {L2}",
                      "--c-cons-layers", "", "--limit", "3", "--max-tokens", "64", "--seed", "5"])
            records = read_jsonl(Path(tmp) / "r" / "judges.jsonl", "judges")
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["meta"]["params"]["layers"], [L1, L2])
        self.assertEqual(records[0]["meta"]["params"]["c_cons_layers"], [])
        self.assertEqual(records[0]["meta"]["params"]["max_tokens"], 64)
        self.assertEqual(records[0]["meta"]["params"]["base_seed"], 5)

    def test_backend_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli_main(["track2", "judge", "--run", "r"])

    def test_hf_local_refuses_a_consultant_as_layer_before_loading_anything(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)), QUIET:
            write_proposals(Path(tmp))
            with self.assertRaisesRegex(ValueError, "not judge layers"):
                cli_main(["track2", "judge", "--run", "r", "--backend", "hf_local", "--layers", "consultant_a",
                          "--c-cons-layers", ""])


if __name__ == "__main__":
    unittest.main()
