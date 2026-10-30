"""Tests for Track 4: Kill-Test Evaluation & Manual Crux Audit."""

import unittest

from crux.track4.kill_test_eval import (
    CaseRecord,
    auroc,
    balanced_subsample,
    filter_split,
    mcnemar_test,
    plot1_calibration,
    select_tau,
    summary_stats,
    table1_kill_criterion_1,
    table2_oversight,
    table3_leakage_and_layers,
)
from crux.track4.manual_audit import (
    GO_NO_GO_THRESHOLDS,
    RUBRIC_COLUMNS,
    audit_rows_to_csv_string,
    build_audit_rows,
    check_go_no_go,
    cohens_kappa,
    per_column_agreement,
    percent_agreement,
    stratified_sample,
)


# ---------------------------------------------------------------------------
# Helpers to build test fixtures
# ---------------------------------------------------------------------------


def _case(
    cid="case_01",
    split="test",
    source="8b_run",
    ground_truth="A",
    initially_right="consultant_a",
    baseline_answer="A",
    crux_answer="A",
    final_answer="A",
    final_source="crux",
    selected_crux_id="case_01#a",
    crux_d=0.8,
    c_cons=0.9,
    layers_agree=True,
    is_leaky=False,
    is_agreement_trap=False,
    **kwargs,
) -> CaseRecord:
    return CaseRecord(
        id=cid,
        dataset="gpqa",
        source=source,
        split=split,
        ground_truth=ground_truth,
        initially_right=initially_right,
        baseline_full_q_answer=baseline_answer,
        crux_audited_answer=crux_answer,
        final_answer=final_answer,
        final_source=final_source,
        selected_crux_id=selected_crux_id,
        crux_d=crux_d,
        c_cons=c_cons,
        layers_agree=layers_agree,
        is_leaky=is_leaky,
        is_agreement_trap=is_agreement_trap,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------


class TestMcNemar(unittest.TestCase):
    def test_identical(self):
        result = mcnemar_test([True, True, False], [True, True, False])
        self.assertEqual(result["b"], 0)
        self.assertEqual(result["c"], 0)
        self.assertEqual(result["p_value"], 1.0)

    def test_all_discordant(self):
        result = mcnemar_test([True, False, True, False], [False, True, False, True])
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["c"], 2)
        # With b==c, p-value should be 1.0 (symmetric)
        self.assertAlmostEqual(result["p_value"], 1.0, places=3)

    def test_asymmetric(self):
        # A is always right, B is always wrong → b=n, c=0
        a = [True] * 10
        b = [False] * 10
        result = mcnemar_test(a, b)
        self.assertEqual(result["b"], 10)
        self.assertEqual(result["c"], 0)
        self.assertLess(result["p_value"], 0.01)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            mcnemar_test([True], [True, False])


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


class TestAUROC(unittest.TestCase):
    def test_perfect(self):
        # All positives have higher score than negatives
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [True, True, False, False]
        self.assertAlmostEqual(auroc(scores, labels), 1.0)

    def test_inverted(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [True, True, False, False]
        self.assertAlmostEqual(auroc(scores, labels), 0.0)

    def test_random(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [True, True, False, False]
        self.assertAlmostEqual(auroc(scores, labels), 0.5)

    def test_empty(self):
        import math
        self.assertTrue(math.isnan(auroc([], [])))

    def test_no_positives(self):
        import math
        self.assertTrue(math.isnan(auroc([0.5], [False])))


# ---------------------------------------------------------------------------
# filter_split
# ---------------------------------------------------------------------------


class TestFilterSplit(unittest.TestCase):
    def test_excludes_frontier_precomputed(self):
        cases = [
            _case(cid="c1", split="test", source="frontier_precomputed"),
            _case(cid="c2", split="test", source="8b_run"),
        ]
        result = filter_split(cases, "test")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "c2")

    def test_filters_by_split(self):
        cases = [
            _case(cid="c1", split="dev"),
            _case(cid="c2", split="test"),
        ]
        self.assertEqual(len(filter_split(cases, "dev")), 1)
        self.assertEqual(len(filter_split(cases, "test")), 1)


# ---------------------------------------------------------------------------
# balanced_subsample
# ---------------------------------------------------------------------------


class TestBalancedSubsample(unittest.TestCase):
    def test_balances(self):
        cases = (
            [_case(cid=f"a{i}", initially_right="consultant_a") for i in range(10)]
            + [_case(cid=f"b{i}", initially_right="consultant_b") for i in range(3)]
        )
        bal = balanced_subsample(cases)
        a_count = sum(1 for c in bal if c.initially_right == "consultant_a")
        b_count = sum(1 for c in bal if c.initially_right == "consultant_b")
        self.assertEqual(a_count, b_count)
        self.assertEqual(a_count, 3)


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------


class TestTable1(unittest.TestCase):
    def test_basic_structure(self):
        cases = [
            _case(cid="c1", baseline_answer="A", crux_answer="A", ground_truth="A"),
            _case(cid="c2", baseline_answer="B", crux_answer="A", ground_truth="A"),
        ]
        t1 = table1_kill_criterion_1(cases)
        methods = [r["method"] for r in t1]
        self.assertIn("full_question_2way", methods)
        self.assertIn("crux_route_all", methods)
        self.assertIn("crux_route_d1", methods)
        self.assertIn("crux_route_gated", methods)
        self.assertIn("chance", methods)

    def test_accuracy_values(self):
        cases = [
            _case(cid="c1", baseline_answer="A", crux_answer="A", ground_truth="A"),
            _case(cid="c2", baseline_answer="A", crux_answer="B", ground_truth="A"),
        ]
        t1 = table1_kill_criterion_1(cases)
        baseline_row = next(r for r in t1 if r["method"] == "full_question_2way")
        self.assertAlmostEqual(baseline_row["accuracy"], 1.0)  # both baseline answers are "A" = ground_truth

        crux_row = next(r for r in t1 if r["method"] == "crux_route_all")
        self.assertAlmostEqual(crux_row["accuracy"], 0.5)  # one right, one wrong


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------


class TestTable2(unittest.TestCase):
    def test_basic_structure(self):
        cases = [_case(cid="c1")]
        t2 = table2_oversight(cases)
        methods = [r["method"] for r in t2]
        self.assertIn("base_dr", methods)
        self.assertIn("crux_audited_consensus", methods)


# ---------------------------------------------------------------------------
# Plot 1 (calibration)
# ---------------------------------------------------------------------------


class TestPlot1(unittest.TestCase):
    def test_returns_all_signals(self):
        cases = [
            _case(cid="c1", c_cons=0.9, crux_answer="A", ground_truth="A"),
            _case(cid="c2", c_cons=0.1, crux_answer="B", ground_truth="A"),
        ]
        cal = plot1_calibration(cases)
        self.assertIn("c_cons", cal)
        self.assertIn("d_times_c_cons", cal)

    def test_empty_cases(self):
        cal = plot1_calibration([])
        for sig in cal.values():
            self.assertEqual(sig["n_total"], 0)


# ---------------------------------------------------------------------------
# Table 3
# ---------------------------------------------------------------------------


class TestTable3(unittest.TestCase):
    def test_leakage_stats(self):
        cases = [_case(cid="c1")]
        candidates = [
            {"crux_id": "c1#a", "is_leaky": True, "d": 0.5},
            {"crux_id": "c1#b", "is_leaky": False, "d": 0.8},
        ]
        t3 = table3_leakage_and_layers(cases, candidates)
        self.assertEqual(t3["leakage"]["n_candidates"], 2)
        self.assertEqual(t3["leakage"]["n_leaky"], 1)
        self.assertAlmostEqual(t3["leakage"]["leaky_share"], 0.5)


# ---------------------------------------------------------------------------
# Tau selection
# ---------------------------------------------------------------------------


class TestTauSelection(unittest.TestCase):
    def test_returns_sweep(self):
        cases = [
            _case(cid="c1", split="dev", crux_d=1.0, c_cons=0.9, layers_agree=True,
                  crux_answer="A", ground_truth="A"),
        ]
        result = select_tau(cases)
        self.assertIn("best_tau", result)
        self.assertIn("sweep", result)
        self.assertEqual(len(result["sweep"]), 9)  # default candidates


# ---------------------------------------------------------------------------
# Manual audit: sampling
# ---------------------------------------------------------------------------


class TestStratifiedSample(unittest.TestCase):
    def test_returns_n(self):
        cases = [{"id": str(i), "dataset": "gpqa", "source": "8b_run"} for i in range(100)]
        sample = stratified_sample(cases, n=50)
        self.assertEqual(len(sample), 50)

    def test_excludes_frontier(self):
        cases = [{"id": str(i), "source": "frontier_precomputed"} for i in range(10)]
        sample = stratified_sample(cases, n=5)
        self.assertEqual(len(sample), 0)

    def test_proportional(self):
        cases = (
            [{"id": f"g{i}", "dataset": "gpqa", "source": "8b_run"} for i in range(60)]
            + [{"id": f"s{i}", "dataset": "supergpqa", "source": "8b_run"} for i in range(40)]
        )
        sample = stratified_sample(cases, n=10, seed=123)
        gpqa_count = sum(1 for c in sample if c["dataset"] == "gpqa")
        sgpqa_count = sum(1 for c in sample if c["dataset"] == "supergpqa")
        # Should be roughly 6:4 ratio
        self.assertGreater(gpqa_count, 0)
        self.assertGreater(sgpqa_count, 0)
        self.assertEqual(gpqa_count + sgpqa_count, 10)


# ---------------------------------------------------------------------------
# Manual audit: agreement
# ---------------------------------------------------------------------------


class TestAgreement(unittest.TestCase):
    def test_percent_agreement_perfect(self):
        self.assertAlmostEqual(percent_agreement([1, 0, 1], [1, 0, 1]), 1.0)

    def test_percent_agreement_none(self):
        self.assertAlmostEqual(percent_agreement([1, 1, 1], [0, 0, 0]), 0.0)

    def test_cohens_kappa_perfect(self):
        self.assertAlmostEqual(cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_cohens_kappa_chance(self):
        # When agreement equals chance, kappa = 0
        # Rater A: all 1, Rater B: half 1, half 0 → kappa ~ 0
        k = cohens_kappa([1, 1, 1, 1], [1, 1, 0, 0])
        self.assertAlmostEqual(k, 0.0, places=5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            percent_agreement([1], [1, 0])
        with self.assertRaises(ValueError):
            cohens_kappa([1], [1, 0])


class TestPerColumnAgreement(unittest.TestCase):
    def test_basic(self):
        a = [{"human_crux_valid": 1, "human_leakage_flag": 0}]
        b = [{"human_crux_valid": 1, "human_leakage_flag": 1}]
        result = per_column_agreement(a, b, columns=["human_crux_valid", "human_leakage_flag"])
        self.assertAlmostEqual(result["human_crux_valid"]["percent_agreement"], 1.0)
        self.assertAlmostEqual(result["human_leakage_flag"]["percent_agreement"], 0.0)


# ---------------------------------------------------------------------------
# Go/No-Go
# ---------------------------------------------------------------------------


class TestGoNoGo(unittest.TestCase):
    def test_all_pass(self):
        agreement = {
            "human_crux_valid": {"percent_agreement": 0.95, "cohens_kappa": 0.85},
            "human_leakage_flag": {"percent_agreement": 0.95, "cohens_kappa": 0.85},
            "human_decisive": {"percent_agreement": 0.90, "cohens_kappa": 0.70},
            "human_verdict_correct": {"percent_agreement": 0.90, "cohens_kappa": 0.75},
        }
        result = check_go_no_go(agreement)
        self.assertTrue(result["go"])
        self.assertEqual(result["verdict"], "GO")

    def test_one_fails(self):
        agreement = {
            "human_crux_valid": {"percent_agreement": 0.50, "cohens_kappa": 0.10},
            "human_leakage_flag": {"percent_agreement": 0.95, "cohens_kappa": 0.85},
            "human_decisive": {"percent_agreement": 0.90, "cohens_kappa": 0.70},
            "human_verdict_correct": {"percent_agreement": 0.90, "cohens_kappa": 0.75},
        }
        result = check_go_no_go(agreement)
        self.assertFalse(result["go"])
        self.assertEqual(result["verdict"], "NO-GO")


# ---------------------------------------------------------------------------
# Audit row generation
# ---------------------------------------------------------------------------


class TestBuildAuditRows(unittest.TestCase):
    def test_columns_present(self):
        cases = [{"id": "c1", "dataset": "gpqa", "split": "test"}]
        rows = build_audit_rows(cases)
        self.assertEqual(len(rows), 1)
        for col in RUBRIC_COLUMNS:
            self.assertIn(col, rows[0])

    def test_csv_output(self):
        cases = [{"id": "c1", "dataset": "gpqa", "split": "test"}]
        rows = build_audit_rows(cases)
        csv_str = audit_rows_to_csv_string(rows)
        self.assertIn("case_id", csv_str)
        self.assertIn("human_crux_valid", csv_str)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


class TestSummaryStats(unittest.TestCase):
    def test_counts(self):
        cases = [
            _case(cid="c1", source="8b_run"),
            _case(cid="c2", source="frontier_precomputed", split="seed"),
        ]
        stats = summary_stats(cases)
        self.assertEqual(stats["n_total"], 2)
        self.assertEqual(stats["sources"]["8b_run"], 1)


if __name__ == "__main__":
    unittest.main()
