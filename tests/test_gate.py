
"""Tests for the D10 Track 2 selection gate."""

import unittest

from crux.gate import apply_gate


def candidate(
    crux_id="c1",
    *,
    parsed=True,
    leaky=False,
    d=0.8,
    veto_flag=False,
    mapped_answer="A",
    mapping_agreement_rate=0.9,
    layer_verdicts=None,
    c_cons=0.9,
    c_prob=0.8,
    c_2judge=0.85,
):
    if layer_verdicts is None:
        layer_verdicts = {
            "judge_l1_qwen": "TRUE",
            "judge_l2_llama": "TRUE",
        }

    return {
        "crux_id": crux_id,
        "parsed": parsed,
        "leaky": leaky,
        "d": d,
        "veto_flag": veto_flag,
        "mapped_answer": mapped_answer,
        "mapping_agreement_rate": mapping_agreement_rate,
        "layer_verdicts": layer_verdicts,
        "c_cons": c_cons,
        "c_prob": c_prob,
        "c_2judge": c_2judge,
    }


class TestApplyGate(unittest.TestCase):
    def test_selects_highest_d_not_highest_checkability(self):
        result = apply_gate(
            [
                candidate("c1", d=0.7, c_cons=1.0),
                candidate("c2", d=0.9, c_cons=0.8),
            ],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "selected")
        self.assertEqual(result["selected_crux_id"], "c2")
        self.assertAlmostEqual(result["crux_score"], 0.72)
        self.assertEqual(result["final_answer"], "A")
        self.assertEqual(result["final_source"], "crux")

    def test_ties_break_by_mapping_agreement_then_id(self):
        result = apply_gate(
            [
                candidate("c2", d=0.8, mapping_agreement_rate=0.8),
                candidate("c1", d=0.8, mapping_agreement_rate=0.95),
                candidate("c0", d=0.8, mapping_agreement_rate=0.95),
            ],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selected_crux_id"], "c0")

    def test_checkability_does_not_change_candidate_ranking(self):
        result = apply_gate(
            [
                candidate("c1", d=0.9, c_cons=0.1),
                candidate("c2", d=0.8, c_cons=1.0),
            ],
            tau=0.0,
            c_signal="c_cons",
        )
        self.assertEqual(result["selected_crux_id"], "c1")

    def test_p2_disagreement_abstains_and_logs_alternative(self):
        result = apply_gate(
            [
                candidate("c1", d=0.9, mapped_answer="A"),
                candidate("c2", d=0.8, mapped_answer="B"),
            ],
            tau=0.5,
            c_signal="c_cons",
            dr_consensus_answer="B",
        )
        self.assertEqual(result["selection_reason"], "candidates_disagree")
        self.assertEqual(result["final_answer"], "B")
        self.assertEqual(result["final_source"], "dr_consensus")
        self.assertEqual(result["p2_alt_answer"], "A")
        self.assertEqual(result["dr_consensus_relation"], "overturns")

    def test_disagreeing_layer_candidate_drops_out(self):
        result = apply_gate(
            [
                candidate(
                    "c1",
                    d=0.95,
                    layer_verdicts={
                        "judge_l1_qwen": "TRUE",
                        "judge_l2_llama": "FALSE",
                    },
                ),
                candidate("c2", d=0.8, mapped_answer="B"),
            ],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "selected")
        self.assertEqual(result["selected_crux_id"], "c2")
        self.assertEqual(result["final_answer"], "B")

    def test_single_eligible_candidate_is_selected(self):
        result = apply_gate(
            [candidate("c1")],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "selected")

    def test_p1_flag_mode_excludes_vetoed_candidate_and_continues(self):
        result = apply_gate(
            [
                candidate("c1", veto_flag=True, d=0.9),
                candidate("c2", d=0.8),
            ],
            tau=0.5,
            c_signal="c_cons",
            p1_mode="flag",
        )
        self.assertEqual(result["selection_reason"], "selected")
        self.assertEqual(result["selected_crux_id"], "c2")
        self.assertEqual(result["p1_veto_crux_ids"], ["c1"])
        self.assertTrue(result["p1_abstain_would_abstain"])

    def test_p1_abstain_mode_falls_back(self):
        result = apply_gate(
            [
                candidate(
                    "c1",
                    veto_flag=True,
                    d=0.9,
                    mapped_answer="A",
                ),
                candidate(
                    "c2",
                    d=0.8,
                    mapped_answer="B",
                ),
            ],
            tau=0.5,
            c_signal="c_cons",
            dr_consensus_answer="B",
            p1_mode="abstain",
        )
        self.assertEqual(result["selection_reason"], "veto")
        self.assertIsNone(result["selected_crux_id"])
        self.assertEqual(result["final_answer"], "B")
        self.assertEqual(result["final_source"], "dr_consensus")
        self.assertEqual(result["p1_veto_crux_ids"], ["c1"])
        self.assertTrue(result["p1_abstain_would_abstain"])
        self.assertEqual(result["eligible_crux_ids"], ["c2"])
        self.assertEqual(result["p2_alt_answer"], "B")
        self.assertEqual(result["dr_consensus_relation"], "confirms")

    def test_all_leaky_candidates_abstain(self):
        result = apply_gate(
            [candidate("c1", leaky=True)],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "all_leaky")

    def test_no_parsed_candidates_abstain(self):
        result = apply_gate(
            [candidate("c1", parsed=False)],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "no_candidates")

    def test_zero_decisiveness_abstains(self):
        result = apply_gate(
            [candidate("c1", d=0.0)],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "d_zero")

    def test_all_layers_disagree_abstains(self):
        result = apply_gate(
            [
                candidate(
                    "c1",
                    layer_verdicts={
                        "judge_l1_qwen": "TRUE",
                        "judge_l2_llama": "FALSE",
                    },
                )
            ],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "layers_disagree")

    def test_below_threshold_falls_back(self):
        result = apply_gate(
            [candidate("c1", d=0.5, c_cons=0.5)],
            tau=0.3,
            c_signal="c_cons",
            dr_consensus_answer="B",
        )
        self.assertEqual(result["selection_reason"], "below_tau")
        self.assertAlmostEqual(result["crux_score"], 0.25)
        self.assertEqual(result["final_answer"], "B")
        self.assertEqual(result["final_source"], "dr_consensus")

    def test_score_equal_to_threshold_is_selected(self):
        result = apply_gate(
            [candidate("c1", d=0.5, c_cons=0.8)],
            tau=0.4,
            c_signal="c_cons",
        )
        self.assertEqual(result["selection_reason"], "selected")
        self.assertAlmostEqual(result["crux_score"], 0.4)

    def test_leaky_candidate_is_excluded(self):
        result = apply_gate(
            [
                candidate("c1", leaky=True, d=0.99),
                candidate("c2", d=0.8),
            ],
            tau=0.5,
            c_signal="c_cons",
        )
        self.assertEqual(result["selected_crux_id"], "c2")

    def test_confirms_dr_consensus(self):
        result = apply_gate(
            [candidate("c1", mapped_answer="A")],
            tau=0.5,
            c_signal="c_cons",
            dr_consensus_answer="A",
        )
        self.assertEqual(result["dr_consensus_relation"], "confirms")

    def test_overturns_dr_consensus(self):
        result = apply_gate(
            [candidate("c1", mapped_answer="A")],
            tau=0.5,
            c_signal="c_cons",
            dr_consensus_answer="B",
        )
        self.assertEqual(result["dr_consensus_relation"], "overturns")

    def test_invalid_c_signal_rejected(self):
        with self.assertRaises(ValueError):
            apply_gate([candidate()], tau=0.5, c_signal="unknown")

    def test_invalid_p1_mode_rejected(self):
        with self.assertRaises(ValueError):
            apply_gate([candidate()], tau=0.5, c_signal="c_cons", p1_mode="maybe")

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            apply_gate(
                [candidate("c1"), candidate("c1")],
                tau=0.5,
                c_signal="c_cons",
            )


if __name__ == "__main__":
    unittest.main()
