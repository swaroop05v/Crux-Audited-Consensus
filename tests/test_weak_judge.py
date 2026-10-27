import unittest

from crux.backends import Registry, get_backend
from crux.backends.hf_local import VerdictMarkerMissing
from crux.track2.weak_judge import (
    judge_crux,
    judge_full_question,
    judge_question_plus_crux,
    measure_c_cons,
    measure_c_prob,
    parse_final_answer,
    parse_verdict,
)


class ParseVerdictTest(unittest.TestCase):
    def test_true_verdict(self):
        self.assertEqual(parse_verdict("Reasoning here.\nVERDICT: TRUE"), "TRUE")

    def test_false_verdict(self):
        self.assertEqual(parse_verdict("Reasoning here.\nVERDICT: FALSE"), "FALSE")

    def test_missing_verdict_raises(self):
        with self.assertRaises(ValueError):
            parse_verdict("I think the statement is true.")

    def test_lowercase_verdict_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_verdict("VERDICT: true")

    def test_multiple_verdicts_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_verdict("VERDICT: TRUE\nVERDICT: FALSE")

class ParseFinalAnswerTest(unittest.TestCase):
    def test_valid_first_candidate(self):
        self.assertEqual(
            parse_final_answer("Reasoning\nFINAL ANSWER: A", "A", "B"),
            "A",
        )

    def test_valid_second_candidate(self):
        self.assertEqual(
            parse_final_answer("Reasoning\nFINAL ANSWER: B", "A", "B"),
            "B",
        )

    def test_rejects_answer_outside_candidates(self):
        with self.assertRaises(ValueError):
            parse_final_answer("FINAL ANSWER: C", "A", "B")

    def test_rejects_missing_final_answer(self):
        with self.assertRaises(ValueError):
            parse_final_answer("The answer is probably A.", "A", "B")

    def test_rejects_multiple_final_answers(self):
        with self.assertRaises(ValueError):
            parse_final_answer(
                "FINAL ANSWER: A\nFINAL ANSWER: B", "A", "B"
            )


class JudgeCruxTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.load()
        self.backend = get_backend(
            self.registry,
            "fake",
            responder="Reasoning here.\nVERDICT: TRUE",
        )

    def test_calls_backend_and_returns_auditable_result(self):
        result = judge_crux(
            "7 * 13 = 91",
            model_key="judge_l1_qwen",
            backend=self.backend,
            registry=self.registry,
            seed=42,
        )

        self.assertEqual(result["verdict"], "TRUE")
        self.assertIn("VERDICT: TRUE", result["raw_completion"])
        self.assertEqual(result["meta"]["stage"], "judge_crux")
        self.assertEqual(result["meta"]["prompt_name"], "judge_crux")
        self.assertEqual(result["meta"]["model_call"]["model_key"], "judge_l1_qwen")
        self.assertEqual(result["meta"]["model_call"]["seed"], 42)
        self.assertEqual(len(self.backend.calls), 1)

        sent_messages = self.backend.calls[0]["messages"]
        self.assertIn("7 * 13 = 91", sent_messages[1]["content"])

    def test_empty_claim_is_rejected_before_call(self):
        with self.assertRaises(ValueError):
            judge_crux(
                "  ",
                model_key="judge_l1_qwen",
                backend=self.backend,
                registry=self.registry,
                seed=42,
            )
        self.assertEqual(len(self.backend.calls), 0)

    def test_invalid_model_response_raises(self):
        backend = get_backend(self.registry, "fake", responder="I think it is true.")

        with self.assertRaises(ValueError):
            judge_crux(
                "7 * 13 = 91",
                model_key="judge_l1_qwen",
                backend=backend,
                registry=self.registry,
                seed=42,
            )

class JudgeFullQuestionTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.load()
        self.backend = get_backend(
            self.registry,
            "fake",
            responder="Reasoning here.\nFINAL ANSWER: B",
        )

    def test_calls_backend_and_returns_auditable_result(self):
        result = judge_full_question(
            "What is 6 * 7?",
            "A",
            "B",
            "41",
            "42",
            model_key="judge_l1_qwen",
            backend=self.backend,
            registry=self.registry,
            seed=42,
        )

        self.assertEqual(result["answer"], "B")
        self.assertIn("FINAL ANSWER: B", result["raw_completion"])
        self.assertEqual(result["meta"]["stage"], "judge_full_question")
        self.assertEqual(
            result["meta"]["prompt_name"], "judge_full_q_2way"
        )
        self.assertEqual(
            result["meta"]["model_call"]["model_key"], "judge_l1_qwen"
        )
        self.assertEqual(result["meta"]["model_call"]["seed"], 42)
        self.assertEqual(len(self.backend.calls), 1)

        sent_messages = self.backend.calls[0]["messages"]
        self.assertIn("What is 6 * 7?", sent_messages[1]["content"])
        self.assertIn("Option A: 41", sent_messages[1]["content"])
        self.assertIn("Option B: 42", sent_messages[1]["content"])

    def test_rejects_invalid_model_response(self):
        backend = get_backend(
            self.registry, "fake", responder="The answer is B."
        )

        with self.assertRaises(ValueError):
            judge_full_question(
                "What is 6 * 7?",
                "A",
                "B",
                "41",
                "42",
                model_key="judge_l1_qwen",
                backend=backend,
                registry=self.registry,
                seed=42,
            )

    def test_rejects_duplicate_answer_labels(self):
        with self.assertRaises(ValueError):
            judge_full_question(
                "What is 6 * 7?",
                "A",
                "A",
                "41",
                "42",
                model_key="judge_l1_qwen",
                backend=self.backend,
                registry=self.registry,
                seed=42,
            )


class JudgeQuestionPlusCruxTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.load()
        self.backend = get_backend(
            self.registry,
            "fake",
            responder="Reasoning here.\nVERDICT: TRUE",
        )

    def test_calls_backend_and_returns_auditable_result(self):
        result = judge_question_plus_crux(
            "What is 6 * 7?",
            "6 * 7 = 42",
            model_key="judge_l1_qwen",
            backend=self.backend,
            registry=self.registry,
            seed=43,
        )

        self.assertEqual(result["verdict"], "TRUE")
        self.assertEqual(
            result["meta"]["stage"], "judge_question_plus_crux"
        )
        self.assertEqual(
            result["meta"]["prompt_name"], "judge_q_plus_crux"
        )
        self.assertEqual(result["meta"]["model_call"]["seed"], 43)
        self.assertEqual(len(self.backend.calls), 1)

        user_message = self.backend.calls[0]["messages"][1]["content"]
        self.assertIn("What is 6 * 7?", user_message)
        self.assertIn("6 * 7 = 42", user_message)

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            judge_question_plus_crux(
                " ",
                "6 * 7 = 42",
                model_key="judge_l1_qwen",
                backend=self.backend,
                registry=self.registry,
                seed=43,
            )

    def test_rejects_invalid_verdict(self):
        backend = get_backend(
            self.registry, "fake", responder="The claim is correct."
        )

        with self.assertRaises(ValueError):
            judge_question_plus_crux(
                "What is 6 * 7?",
                "6 * 7 = 42",
                model_key="judge_l1_qwen",
                backend=backend,
                registry=self.registry,
                seed=43,
            )

class MeasureCProbTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.load()
        self.backend = get_backend(
            self.registry,
            "fake",
            responder="VERDICT: TRUE",
            verdict={"p_true_raw": 0.6, "p_false_raw": 0.2},
        )

    def test_returns_probability_masses_and_metadata(self):
        result = measure_c_prob(
            "7 * 13 = 91",
            "Reasoning here.\nVERDICT: TRUE",
            model_key="judge_l1_qwen",
            backend=self.backend,
        )

        self.assertAlmostEqual(result["p_true_raw"], 0.6)
        self.assertAlmostEqual(result["p_false_raw"], 0.2)
        self.assertAlmostEqual(result["p_mass_tf"], 0.8)
        self.assertAlmostEqual(result["p_true"], 0.75)
        self.assertEqual(result["meta"]["method"], "next_token_probability_mass")
        self.assertEqual(result["meta"]["marker"], "VERDICT:")

    def test_missing_verdict_marker_is_rejected(self):
        with self.assertRaises(VerdictMarkerMissing):
            measure_c_prob(
                "7 * 13 = 91",
                "No verdict supplied.",
                model_key="judge_l1_qwen",
                backend=self.backend,
            )

    def test_empty_claim_is_rejected(self):
        with self.assertRaises(ValueError):
            measure_c_prob(
                " ",
                "VERDICT: TRUE",
                model_key="judge_l1_qwen",
                backend=self.backend,
            )


class MeasureCConsTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.load()
        self.backend = get_backend(
            self.registry,
            "fake",
            responder=[
                "Reasoning\nVERDICT: TRUE",
                "Reasoning\nVERDICT: TRUE",
                "Reasoning\nVERDICT: FALSE",
                "Reasoning\nVERDICT: TRUE",
                "Reasoning\nVERDICT: TRUE",
            ],
        )
        self.seeds = [101, 102, 103, 104, 105]

    def test_returns_five_samples_and_correct_agreement(self):
        result = measure_c_cons(
            "7 * 13 = 91",
            model_key="judge_l1_qwen",
            backend=self.backend,
            registry=self.registry,
            seeds=self.seeds,
        )

        self.assertEqual(len(result["samples"]), 5)
        self.assertEqual(result["true_count"], 4)
        self.assertEqual(result["false_count"], 1)
        self.assertAlmostEqual(result["agreement_rate"], 0.8)
        self.assertEqual(result["majority_verdict"], "TRUE")
        self.assertEqual(
            [s["seed"] for s in result["samples"]], self.seeds
        )
        self.assertEqual(len(self.backend.calls), 5)

    def test_uses_consistency_sampling_profile(self):
        measure_c_cons(
            "7 * 13 = 91",
            model_key="judge_l1_qwen",
            backend=self.backend,
            registry=self.registry,
            seeds=self.seeds,
        )

        for call in self.backend.calls:
            self.assertEqual(call["temperature"], 0.7)
            self.assertIsNone(call["top_p"])

    def test_rejects_wrong_number_of_seeds(self):
        with self.assertRaises(ValueError):
            measure_c_cons(
                "7 * 13 = 91",
                model_key="judge_l1_qwen",
                backend=self.backend,
                registry=self.registry,
                seeds=[101, 102],
            )

    def test_rejects_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            measure_c_cons(
                "7 * 13 = 91",
                model_key="judge_l1_qwen",
                backend=self.backend,
                registry=self.registry,
                seeds=[101, 102, 103, 104, 104],
            )

    def test_rejects_invalid_sample_verdict(self):
        backend = get_backend(
            self.registry,
            "fake",
            responder="The claim is correct.",
        )

        with self.assertRaises(ValueError):
            measure_c_cons(
                "7 * 13 = 91",
                model_key="judge_l1_qwen",
                backend=backend,
                registry=self.registry,
                seeds=self.seeds,
            )

if __name__ == "__main__":
    unittest.main()
