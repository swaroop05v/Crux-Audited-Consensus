import unittest

from crux.backends import Registry, get_backend
from crux.track2.checkability import (
    MODEL_FAMILIES,
    compare_verdicts,
    compare_judges_on_claim,
)


class CompareVerdictsTest(unittest.TestCase):
    def test_agreeing_cross_family_verdicts(self):
        result = compare_verdicts(
            "TRUE",
            "TRUE",
            model_a="judge_l1_qwen",
            model_b="judge_l2_llama",
            family_a="qwen",
            family_b="llama",
        )

        self.assertTrue(result["agreement"])
        self.assertEqual(result["comparison"], "cross_family")
        self.assertEqual(result["family_a"], "qwen")
        self.assertEqual(result["family_b"], "llama")

    def test_disagreeing_cross_family_verdicts(self):
        result = compare_verdicts(
            "TRUE",
            "FALSE",
            model_a="judge_l1_qwen",
            model_b="judge_l2_llama",
            family_a="qwen",
            family_b="llama",
        )

        self.assertFalse(result["agreement"])
        self.assertEqual(result["comparison"], "cross_family")

    def test_same_family_different_models(self):
        result = compare_verdicts(
            "FALSE",
            "FALSE",
            model_a="qwen_model_a",
            model_b="qwen_model_b",
            family_a="qwen",
            family_b="qwen",
        )

        self.assertTrue(result["agreement"])
        self.assertEqual(result["comparison"], "same_family")

    def test_invalid_verdict_rejected(self):
        with self.assertRaises(ValueError):
            compare_verdicts(
                "MAYBE",
                "TRUE",
                model_a="judge_l1_qwen",
                model_b="judge_l2_llama",
                family_a="qwen",
                family_b="llama",
            )

    def test_empty_model_key_rejected(self):
        with self.assertRaises(ValueError):
            compare_verdicts(
                "TRUE",
                "FALSE",
                model_a=" ",
                model_b="judge_l2_llama",
                family_a="qwen",
                family_b="llama",
            )

    def test_empty_family_rejected(self):
        with self.assertRaises(ValueError):
            compare_verdicts(
                "TRUE",
                "FALSE",
                model_a="judge_l1_qwen",
                model_b="judge_l2_llama",
                family_a="",
                family_b="llama",
            )


class CompareJudgesOnClaimTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.load()

    def test_runs_both_judges_and_returns_comparison(self):
        backend = get_backend(
            self.registry,
            "fake",
            responder=["VERDICT: TRUE", "VERDICT: TRUE"],
        )

        result = compare_judges_on_claim(
            "The Earth orbits the Sun.",
            model_a="judge_l1_qwen",
            model_b="judge_l2_llama",
            backend=backend,
            registry=self.registry,
            seed=42,
        )

        self.assertEqual(result["verdict_a"], "TRUE")
        self.assertEqual(result["verdict_b"], "TRUE")
        self.assertTrue(result["agreement"])
        self.assertEqual(result["comparison"], "cross_family")
        self.assertEqual(result["claim"], "The Earth orbits the Sun.")
        self.assertEqual(result["seed"], 42)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(
            [call["model_key"] for call in backend.calls],
            ["judge_l1_qwen", "judge_l2_llama"],
        )

    def test_disagreeing_judges_are_reported(self):
        backend = get_backend(
            self.registry,
            "fake",
            responder=["VERDICT: TRUE", "VERDICT: FALSE"],
        )

        result = compare_judges_on_claim(
            "Test claim",
            model_a="judge_l1_qwen",
            model_b="judge_l2_gemma",
            backend=backend,
            registry=self.registry,
            seed=7,
        )

        self.assertFalse(result["agreement"])
        self.assertEqual(result["comparison"], "cross_family")
        self.assertEqual(result["verdict_a"], "TRUE")
        self.assertEqual(result["verdict_b"], "FALSE")

    def test_same_family_comparison(self):
        backend = get_backend(
            self.registry,
            "fake",
            responder=["VERDICT: TRUE", "VERDICT: FALSE"],
        )

        result = compare_judges_on_claim(
            "Test claim",
            model_a="judge_l2_llama",
            model_b="judge_l2_llama",
            backend=backend,
            registry=self.registry,
            seed=11,
        )

        self.assertEqual(result["comparison"], "same_family")
        self.assertFalse(result["agreement"])

    def test_unknown_first_model_is_rejected(self):
        backend = get_backend(
            self.registry,
            "fake",
            responder="VERDICT: TRUE",
        )

        with self.assertRaises(ValueError):
            compare_judges_on_claim(
                "Test claim",
                model_a="unknown_model",
                model_b="judge_l2_llama",
                backend=backend,
                registry=self.registry,
                seed=1,
            )

        self.assertEqual(len(backend.calls), 0)

    def test_unknown_second_model_is_rejected(self):
        backend = get_backend(
            self.registry,
            "fake",
            responder="VERDICT: TRUE",
        )

        with self.assertRaises(ValueError):
            compare_judges_on_claim(
                "Test claim",
                model_a="judge_l1_qwen",
                model_b="unknown_model",
                backend=backend,
                registry=self.registry,
                seed=1,
            )

        self.assertEqual(len(backend.calls), 0)

    def test_model_family_mapping_contains_expected_models(self):
        self.assertEqual(MODEL_FAMILIES["judge_l1_qwen"], "qwen")
        self.assertEqual(MODEL_FAMILIES["judge_l2_llama"], "llama")
        self.assertEqual(MODEL_FAMILIES["judge_l2_gemma"], "gemma")


if __name__ == "__main__":
    unittest.main()
