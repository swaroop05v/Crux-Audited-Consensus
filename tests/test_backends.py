"""Backend tests. Fake backend and fake HTTP session only: no network, no GPU, no real model."""

import json
import unittest

from crux.backends import BackendError, Registry, get_backend, strip_think
from crux.backends.fake import FakeBackend
from crux.backends.hf_local import (HFLocalBackend, VerdictMarkerMissing, cut_after_marker, verdict_readout,
                                    verdict_token_ids)
from crux.backends.openai_compat import OpenAICompatBackend
from crux.io import stage_meta
from crux.prompts import load_prompt

REGISTRY = Registry.load()
MESSAGES = [{"role": "user", "content": "hi"}]
JUDGE = dict(temperature=0.0, top_p=None, max_tokens=16, seed=1)


class RegistryTest(unittest.TestCase):
    def test_models_and_pins(self):
        self.assertEqual(set(REGISTRY.models),
                         {"consultant_a", "consultant_b", "judge_l1_qwen", "judge_l2_llama", "judge_l2_gemma"})
        self.assertEqual(REGISTRY.spec("consultant_a").hf_repo, "Qwen/Qwen3-8B")
        self.assertEqual(REGISTRY.spec("consultant_a").chat_template_kwargs, {"enable_thinking": False})
        self.assertEqual(REGISTRY.spec("judge_l2_gemma").hf_repo, "google/gemma-3-1b-it")
        for key in ("consultant_a", "consultant_b"):
            self.assertEqual(REGISTRY.spec(key).backend, "vllm")
        for key in ("judge_l1_qwen", "judge_l2_llama", "judge_l2_gemma"):
            self.assertEqual(REGISTRY.spec(key).backend, "hf_local")

    def test_sampling_profiles_match_decisions_d3(self):
        self.assertEqual(REGISTRY.sampling("consultant"), {"temperature": 0.7, "top_p": 0.9})
        self.assertEqual(REGISTRY.sampling("leak_probe"), {"temperature": 0.0, "top_p": None})
        self.assertEqual(REGISTRY.sampling("judge_verdict"), {"temperature": 0.0, "top_p": None})
        self.assertEqual(REGISTRY.sampling("judge_cons"), {"temperature": 0.7, "top_p": None})
        self.assertEqual(REGISTRY.sampling("judge_layer2"), {"temperature": 0.0, "top_p": None})
        with self.assertRaises(BackendError):
            REGISTRY.sampling("nope")


class FakeBackendTest(unittest.TestCase):
    def test_canned_response_and_logged_params(self):
        backend = FakeBackend(REGISTRY, responder='{"answer": "B"}')
        result = backend.chat("consultant_a", MESSAGES, **REGISTRY.sampling("consultant"), max_tokens=64, seed=3)
        self.assertEqual(json.loads(result.text), {"answer": "B"})
        params = result.params
        self.assertEqual(params["backend"], "fake")
        self.assertEqual(params["revision"], REGISTRY.spec("consultant_a").revision)
        self.assertEqual((params["temperature"], params["top_p"], params["seed"]), (0.7, 0.9, 3))
        self.assertEqual(backend.calls[0]["messages"], MESSAGES)

    def test_params_go_into_stage_meta(self):
        backend = FakeBackend(REGISTRY)
        prompt = load_prompt("judge_crux")
        result = backend.chat("judge_l1_qwen", prompt.messages(crux_claim="1 + 1 = 2"), **JUDGE)
        meta = stage_meta("judge", calls={"layer1": result.params}, **prompt.meta())
        self.assertEqual(meta["calls"]["layer1"]["hf_repo"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(meta["prompt_sha256"], prompt.sha256)
        json.dumps(meta)  # must be JSON-serializable for stage files

    def test_cycled_responses_and_n(self):
        backend = FakeBackend(REGISTRY, responder=["x", "y"])
        result = backend.chat("judge_l1_qwen", MESSAGES, **JUDGE, n=3)
        self.assertEqual(result.texts, ["x", "y", "x"])
        with self.assertRaises(BackendError):
            result.text

    def test_strip_think_applied_for_qwen3(self):
        backend = FakeBackend(REGISTRY, responder='<think>\n{"no": 1}\n</think>\n{"answer": "C"}')
        result = backend.chat("consultant_a", MESSAGES, **REGISTRY.sampling("consultant"), max_tokens=8, seed=0)
        self.assertEqual(result.text, '{"answer": "C"}')
        self.assertEqual(strip_think("plain"), "plain")

    def test_argument_validation(self):
        backend = FakeBackend(REGISTRY)
        with self.assertRaises(TypeError):
            backend.chat("judge_l1_qwen", MESSAGES, max_tokens=8, seed=0)  # temperature/top_p are required
        with self.assertRaises(ValueError):
            backend.chat("judge_l1_qwen", MESSAGES, temperature=-1, top_p=None, max_tokens=8, seed=0)
        with self.assertRaises(ValueError):
            backend.chat("judge_l1_qwen", [{"role": "tool", "content": "x"}], **JUDGE)
        with self.assertRaises(BackendError):
            backend.chat("unknown_model", MESSAGES, **JUDGE)

    def test_fake_verdict_probs(self):
        readout = FakeBackend(REGISTRY).verdict_probs("judge_l1_qwen", MESSAGES, "work...\nVERDICT: TRUE")
        self.assertAlmostEqual(readout["p_true"], 0.75)
        with self.assertRaises(VerdictMarkerMissing):
            FakeBackend(REGISTRY).verdict_probs("judge_l1_qwen", MESSAGES, "no marker here")


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def ok(*texts, model="qwen3-8b"):
    return FakeResponse(200, {"model": model, "choices": [{"message": {"content": t}} for t in texts],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 2}})


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.payloads.append(json)
        return self.responses.pop(0)


def make_backend(responses, name="vllm", supports_n=True, send_kwargs=True, max_retries=3):
    session, sleeps = FakeSession(responses), []
    backend = OpenAICompatBackend(REGISTRY, name, base_url="http://localhost:8000/v1", supports_n=supports_n,
                                  send_chat_template_kwargs=send_kwargs, max_retries=max_retries,
                                  session=session, sleep=sleeps.append)
    return backend, session, sleeps


CONSULTANT = dict(temperature=0.7, top_p=0.9, max_tokens=32, seed=11)


class OpenAICompatTest(unittest.TestCase):
    def test_payload_and_params(self):
        backend, session, sleeps = make_backend([ok('{"answer": "A"}')])
        result = backend.chat("consultant_a", MESSAGES, **CONSULTANT)
        payload = session.payloads[0]
        self.assertEqual(payload["model"], "qwen3-8b")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((payload["temperature"], payload["top_p"], payload["seed"], payload["n"]),
                         (0.7, 0.9, 11, 1))
        self.assertEqual(result.params["served_model"], ["qwen3-8b"])
        self.assertEqual(result.params["endpoint_host"], "localhost:8000")
        self.assertEqual(sleeps, [])

    def test_retries_429_and_5xx_then_succeeds(self):
        backend, session, sleeps = make_backend(
            [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(503), ok("done")])
        self.assertEqual(backend.chat("consultant_a", MESSAGES, **CONSULTANT).text, "done")
        self.assertEqual(len(session.payloads), 3)
        self.assertEqual(sleeps[0], 7.0)  # Retry-After honoured
        self.assertEqual(len(sleeps), 2)

    def test_provider_error_in_200_body_is_retried(self):
        backend, _, sleeps = make_backend([FakeResponse(200, {"error": {"code": 429, "message": "rate"}}), ok("x")])
        self.assertEqual(backend.chat("consultant_a", MESSAGES, **CONSULTANT).text, "x")
        self.assertEqual(len(sleeps), 1)

    def test_gives_up_after_max_retries(self):
        backend, session, _ = make_backend([FakeResponse(500)] * 3, max_retries=2)
        with self.assertRaises(BackendError):
            backend.chat("consultant_a", MESSAGES, **CONSULTANT)
        self.assertEqual(len(session.payloads), 3)

    def test_client_error_is_not_retried(self):
        backend, session, sleeps = make_backend([FakeResponse(400, {"error": "bad"})])
        with self.assertRaises(BackendError):
            backend.chat("consultant_a", MESSAGES, **CONSULTANT)
        self.assertEqual((len(session.payloads), sleeps), (1, []))

    def test_without_n_support_loops_with_incrementing_seeds(self):
        backend, session, _ = make_backend([ok("a"), ok("b"), ok("c")], supports_n=False)
        result = backend.chat("consultant_a", MESSAGES, **CONSULTANT, n=3)
        self.assertEqual(result.texts, ["a", "b", "c"])
        self.assertEqual([p["seed"] for p in session.payloads], [11, 12, 13])
        self.assertTrue(all("n" not in p for p in session.payloads))
        self.assertEqual(result.params["seeds"], [11, 12, 13])

    def test_refuses_to_drop_chat_template_kwargs(self):
        backend, session, _ = make_backend([ok("x")], send_kwargs=False)
        with self.assertRaises(BackendError):
            backend.chat("consultant_a", MESSAGES, **CONSULTANT)
        self.assertEqual(session.payloads, [])

    def test_one_backend_per_model(self):
        backend, _, _ = make_backend([ok("x")])
        with self.assertRaises(BackendError):
            backend.chat("judge_l1_qwen", MESSAGES, **JUDGE)  # pinned to hf_local
        other, _, _ = make_backend([ok("x")], name="openrouter")
        with self.assertRaises(BackendError):
            other.chat("consultant_a", MESSAGES, **CONSULTANT)  # pinned to vllm

    def test_get_backend_requires_base_url(self):
        import os
        saved = os.environ.pop("VLLM_BASE_URL", None)
        try:
            os.environ["VLLM_BASE_URL"] = ""
            with self.assertRaises(BackendError):
                get_backend(REGISTRY, "vllm")
        finally:
            os.environ.pop("VLLM_BASE_URL", None)
            if saved is not None:
                os.environ["VLLM_BASE_URL"] = saved


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else []


class VerdictReadoutTest(unittest.TestCase):
    def test_cut_after_last_marker(self):
        text = "I will end with VERDICT: soon.\nWork...\nVERDICT: TRUE"
        self.assertEqual(cut_after_marker(text), "I will end with VERDICT: soon.\nWork...\nVERDICT:")

    def test_token_ids_union_of_spellings(self):
        tok = FakeTokenizer({" TRUE": 10, "TRUE": 11, " True": 10, " FALSE": 20, "FALSE": 21, "false": 22})
        self.assertEqual(verdict_token_ids(tok), ([10, 11], [20, 21, 22]))

    def test_overlapping_first_tokens_raise(self):
        tok = FakeTokenizer({" TRUE": 5, " FALSE": 5})
        with self.assertRaises(BackendError):
            verdict_token_ids(tok)

    def test_readout_renormalizes_and_logs_mass(self):
        probs = {10: 0.30, 11: 0.10, 20: 0.05, 99: 0.55}
        readout = verdict_readout(lambda i: probs.get(i, 0.0), [10, 11], [20])
        self.assertAlmostEqual(readout["p_mass_tf"], 0.45)
        self.assertAlmostEqual(readout["p_true"], 0.40 / 0.45)

    def test_zero_mass(self):
        self.assertIsNone(verdict_readout(lambda i: 0.0, [1], [2])["p_true"])

    def test_hf_local_constructs_without_loading_models(self):
        backend = HFLocalBackend(REGISTRY)
        self.assertEqual(backend._loaded, {})
        with self.assertRaises(BackendError):
            backend.spec("consultant_a")  # consultants are pinned to vllm


if __name__ == "__main__":
    unittest.main()
