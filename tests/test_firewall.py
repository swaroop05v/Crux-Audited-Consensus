"""The label firewall: no ground-truth-revealing field may appear outside the top-level ``eval`` block."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from crux.data.seed import BASE_REPO, build_seed
from crux.io import SchemaError, read_jsonl, stage_meta, write_jsonl
from crux.schema import find_label_leaks, validate_record

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = REPO_ROOT / "data" / "seed" / "seed_30.jsonl"

# Exactly the top-level keys a ``cases`` record may carry; anything copied from the source files beyond this fails.
CASES_TOP_LEVEL_KEYS = {
    "schema_version", "stage", "id", "dataset", "source", "split", "question", "question_sha1", "choices",
    "domain", "provenance", "consultant_a_init", "consultant_b_init", "meta", "eval",
}


def _seed_records():
    with SEED_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SeedFileTest(unittest.TestCase):
    def test_seed_has_40_balanced_valid_records(self):
        records = _seed_records()
        self.assertEqual(len(records), 40)
        for record in records:
            self.assertEqual(validate_record(record, "cases"), [], record["id"])
        sides = [(r["dataset"], r["eval"]["initially_right"]) for r in records]
        for key in [("gpqa", "a"), ("gpqa", "b"), ("supergpqa", "a"), ("supergpqa", "b")]:
            self.assertEqual(sides.count(key), 10, key)

    def test_no_forbidden_field_outside_eval_in_seed(self):
        for record in _seed_records():
            self.assertEqual(find_label_leaks(record), [], record["id"])
            self.assertEqual(set(record) - CASES_TOP_LEVEL_KEYS, set(), record["id"])

    def test_seed_is_plumbing_only(self):
        for record in _seed_records():
            self.assertEqual(record["source"], "frontier_precomputed")
            self.assertEqual(record["split"], "seed")


@unittest.skipUnless((BASE_REPO / "data").exists(), "collaborative-dr submodule not checked out")
class SeedBuilderTest(unittest.TestCase):
    def test_builder_output_has_no_leaks(self):
        records, _ = build_seed(per_dataset=20, seed=42)
        for record in records:
            record["meta"] = stage_meta("cases")
            self.assertEqual(find_label_leaks(record), [], record["id"])
            self.assertEqual(set(record) - CASES_TOP_LEVEL_KEYS, set(), record["id"])
            self.assertEqual(validate_record(record, "cases"), [], record["id"])

    def test_builder_is_deterministic(self):
        first, _ = build_seed(per_dataset=20, seed=42)
        second, _ = build_seed(per_dataset=20, seed=42)
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])


class LoaderFirewallTest(unittest.TestCase):
    def test_read_strips_eval_by_default(self):
        for record in read_jsonl(SEED_PATH, "cases"):
            self.assertNotIn("eval", record)
            self.assertEqual(find_label_leaks(record), [])

    def test_read_with_labels_keeps_eval(self):
        for record in read_jsonl(SEED_PATH, "cases", with_labels=True):
            self.assertIn("ground_truth", record["eval"])


class ValidatorCatchesLeaksTest(unittest.TestCase):
    def setUp(self):
        self.record = _seed_records()[0]

    def _assert_leak(self, mutate, expected_path):
        record = copy.deepcopy(self.record)
        mutate(record)
        self.assertIn(expected_path, find_label_leaks(record))
        self.assertTrue(any(expected_path in error for error in validate_record(record, "cases")))

    def test_top_level_ground_truth(self):
        self._assert_leak(lambda r: r.update(ground_truth="A"), "ground_truth")

    def test_nested_correct_choice(self):
        self._assert_leak(lambda r: r["consultant_a_init"].update(correct_choice="A"),
                          "consultant_a_init.correct_choice")

    def test_correct_suffix_pattern(self):
        self._assert_leak(lambda r: r["consultant_b_init"].update(model_correct=True),
                          "consultant_b_init.model_correct")

    def test_gpqa_source_fields(self):
        self._assert_leak(lambda r: r.update({"Correct Answer": "x"}), "Correct Answer")
        self._assert_leak(lambda r: r["provenance"].update({"Incorrect Answer 2": "y"}),
                          "provenance.Incorrect Answer 2")

    def test_supergpqa_answer_text_at_top_level(self):
        self._assert_leak(lambda r: r.update(answer="some option text"), "answer")

    def test_leak_inside_list(self):
        self._assert_leak(lambda r: r.update(extra=[{"derived_crux_label": True}]), "extra[0].derived_crux_label")

    def test_consultant_answer_is_not_a_leak(self):
        self.assertEqual(find_label_leaks(self.record), [])
        self.assertIn("answer", self.record["consultant_a_init"])

    def test_write_refuses_leaky_record(self):
        record = copy.deepcopy(self.record)
        record["consultant_a_init"]["correct_choice"] = "A"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cases.jsonl"
            with self.assertRaises(SchemaError):
                write_jsonl(out, [record], "cases")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
