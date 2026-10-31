import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crux.backends import BackendError, Registry, get_backend
from crux.backends.hf_local import HFLocalBackend, VerdictMarkerMissing
from crux.prompts import PromptError
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

    def test_accepted_formats(self):
        cases = {
            "**VERDICT: TRUE**": "TRUE",
            "**VERDICT: FALSE**": "FALSE",
            "*VERDICT: TRUE*": "TRUE",
            "***VERDICT: FALSE***": "FALSE",
            "VERDICT: true": "TRUE",
            "VERDICT: false": "FALSE",
            "VERDICT: True": "TRUE",
            "VERDICT: fAlSe": "FALSE",
            "VERDICT: TRUE.": "TRUE",
            "VERDICT: false.": "FALSE",
            "**VERDICT: TRUE.**": "TRUE",
            "**VERDICT: False**.": "FALSE",
        }
        for line, expected in cases.items():
            with self.subTest(line=line):
                self.assertEqual(parse_verdict(f"Reasoning.\n{line}"), expected)

    def test_windows_line_ending_is_accepted(self):
        self.assertEqual(parse_verdict("Reasoning.\r\n**VERDICT: true.**\r\n"), "TRUE")

    def test_identical_verdict_lines_are_accepted(self):
        self.assertEqual(parse_verdict("VERDICT: TRUE\nVERDICT: TRUE"), "TRUE")
        self.assertEqual(parse_verdict("VERDICT: FALSE\n**VERDICT: false.**"), "FALSE")

    def test_multiple_verdicts_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_verdict("VERDICT: TRUE\nVERDICT: FALSE")

    def test_conflicting_verdicts_across_formats_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_verdict("**VERDICT: true**\nVERDICT: FALSE.")

    def test_keyword_stays_uppercase_and_alone_on_the_line(self):
        for text in (
            "verdict: TRUE",
            "Verdict: TRUE",
            "**VERDICT:** TRUE",
            "VERDICT: **TRUE**",
            "My VERDICT: TRUE",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_verdict(text)

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

    def measure(self, responder, **backend_options):
        backend = get_backend(
            self.registry, "fake", responder=responder, **backend_options
        )
        return measure_c_cons(
            "7 * 13 = 91",
            model_key="judge_l1_qwen",
            backend=backend,
            registry=self.registry,
            seeds=self.seeds,
        )

    def test_c_cons_values(self):
        for true_count, false_count, expected in (
            (5, 0, 1.0),
            (4, 1, 0.6),
            (3, 2, 0.2),
            (2, 3, 0.2),
            (0, 5, 1.0),
        ):
            with self.subTest(true=true_count, false=false_count):
                result = self.measure(
                    ["VERDICT: TRUE"] * true_count
                    + ["VERDICT: FALSE"] * false_count
                )
                self.assertAlmostEqual(result["c_cons"], expected)
                self.assertEqual(result["true_count"], true_count)
                self.assertEqual(result["false_count"], false_count)
                self.assertEqual(result["n_valid"], 5)
                self.assertEqual(result["n_failed"], 0)
                self.assertIsNone(result["c_cons_error"])

    def test_valid_samples_record_finish_reason_and_no_failure(self):
        result = self.measure("VERDICT: TRUE")

        for sample in result["samples"]:
            self.assertTrue(sample["parse_ok"])
            self.assertIsNone(sample["parse_error"])
            self.assertIsNone(sample["failure_type"])
            self.assertIsNone(sample["call_error"])
            self.assertEqual(sample["finish_reason"], "stop")
            self.assertEqual(sample["raw_output"], "VERDICT: TRUE")
            self.assertEqual(sample["model_call"]["model_key"], "judge_l1_qwen")

    def test_one_failed_sample_is_left_out_of_c_cons(self):
        result = self.measure(
            [
                "VERDICT: TRUE",
                "VERDICT: TRUE",
                "The claim is correct.",
                "VERDICT: TRUE",
                "VERDICT: FALSE",
            ]
        )

        self.assertEqual(len(result["samples"]), 5)
        self.assertEqual(result["true_count"], 3)
        self.assertEqual(result["false_count"], 1)
        self.assertEqual(result["n_valid"], 4)
        self.assertEqual(result["n_failed"], 1)
        self.assertAlmostEqual(result["c_cons"], 0.5)
        self.assertAlmostEqual(result["agreement_rate"], 0.75)
        self.assertEqual(result["majority_verdict"], "TRUE")
        self.assertIsNone(result["c_cons_error"])

        failed = result["samples"][2]
        self.assertEqual(failed["seed"], 103)
        self.assertIsNone(failed["verdict"])
        self.assertFalse(failed["parse_ok"])
        self.assertIn("VERDICT", failed["parse_error"])
        self.assertEqual(failed["failure_type"], "parse_error")
        self.assertEqual(failed["raw_output"], "The claim is correct.")

    def test_all_failed_samples_give_no_measurement(self):
        result = self.measure("The claim is correct.")

        self.assertEqual(len(result["samples"]), 5)
        self.assertEqual(result["true_count"], 0)
        self.assertEqual(result["false_count"], 0)
        self.assertEqual(result["n_valid"], 0)
        self.assertEqual(result["n_failed"], 5)
        self.assertIsNone(result["c_cons"])
        self.assertIsNone(result["agreement_rate"])
        self.assertIsNone(result["majority_verdict"])
        self.assertIn("no valid samples", result["c_cons_error"])
        self.assertIn("5 parse_error", result["c_cons_error"])

    def test_even_split_after_failure_is_a_tie(self):
        result = self.measure(
            [
                "VERDICT: TRUE",
                "VERDICT: FALSE",
                "The claim is correct.",
                "VERDICT: TRUE",
                "VERDICT: FALSE",
            ]
        )

        self.assertEqual(result["n_valid"], 4)
        self.assertAlmostEqual(result["c_cons"], 0.0)
        self.assertAlmostEqual(result["agreement_rate"], 0.5)
        self.assertEqual(result["majority_verdict"], "TIE")

    def test_truncated_sample_is_not_trusted(self):
        result = self.measure(
            "VERDICT: TRUE",
            finish_reason=lambda key, index: "length" if index == 1 else "stop",
        )

        truncated = result["samples"][1]
        self.assertIsNone(truncated["verdict"])
        self.assertFalse(truncated["parse_ok"])
        self.assertEqual(truncated["failure_type"], "truncated")
        self.assertEqual(truncated["parse_error"], "truncated at max_tokens")
        self.assertEqual(truncated["finish_reason"], "length")
        self.assertTrue(truncated["model_call"]["truncated"])
        self.assertEqual(result["n_valid"], 4)
        self.assertEqual(result["n_failed"], 1)
        self.assertEqual(result["true_count"], 4)
        self.assertAlmostEqual(result["c_cons"], 1.0)

    def test_failed_call_is_recorded_and_sampling_continues(self):
        attempts = []

        def flaky(model_key, messages, index):
            # FakeBackend records a call only after the responder returns, so index stays put after a raise.
            attempts.append(index)
            if len(attempts) == 1:
                raise BackendError("HTTP 503 after retries")
            return "VERDICT: TRUE"

        result = self.measure(flaky)

        failed = result["samples"][0]
        self.assertIsNone(failed["verdict"])
        self.assertFalse(failed["parse_ok"])
        self.assertEqual(failed["failure_type"], "call_error")
        self.assertEqual(failed["call_error"], "HTTP 503 after retries")
        self.assertIsNone(failed["finish_reason"])
        self.assertIsNone(failed["raw_output"])
        self.assertIsNone(failed["model_call"])
        self.assertEqual(len(attempts), 5)
        self.assertEqual(result["n_valid"], 4)
        self.assertEqual(result["n_failed"], 1)

    def test_unknown_model_key_raises(self):
        with self.assertRaisesRegex(BackendError, "unknown model key"):
            measure_c_cons(
                "7 * 13 = 91",
                model_key="no_such_model",
                backend=self.backend,
                registry=self.registry,
                seeds=self.seeds,
            )
        self.assertEqual(len(self.backend.calls), 0)

    def test_backend_mismatch_raises(self):
        # FakeBackend skips the pin check by design, so this needs a real backend class.
        with self.assertRaisesRegex(BackendError, "pinned to backend"):
            measure_c_cons(
                "7 * 13 = 91",
                model_key="consultant_a",
                backend=HFLocalBackend(self.registry),
                registry=self.registry,
                seeds=self.seeds,
            )

    def test_missing_prompt_file_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            with mock.patch("crux.prompts.PROMPTS_DIR", Path(empty)):
                with self.assertRaises(PromptError):
                    measure_c_cons(
                        "7 * 13 = 91",
                        model_key="judge_l1_qwen",
                        backend=self.backend,
                        registry=self.registry,
                        seeds=self.seeds,
                    )
        self.assertEqual(len(self.backend.calls), 0)

    def test_unknown_sampling_profile_raises(self):
        registry = Registry(self.registry.models, self.registry.backends, {})

        with self.assertRaisesRegex(BackendError, "unknown sampling profile"):
            measure_c_cons(
                "7 * 13 = 91",
                model_key="judge_l1_qwen",
                backend=self.backend,
                registry=registry,
                seeds=self.seeds,
            )
        self.assertEqual(len(self.backend.calls), 0)

if __name__ == "__main__":
    unittest.main()
