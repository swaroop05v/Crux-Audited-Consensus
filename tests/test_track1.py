"""Track 1 tests: canned fake-backend replies on seed_30 cases. No real model, no network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crux.backends import BackendError, Registry
from crux.backends.fake import FakeBackend
from crux.cli import main as cli_main
from crux.io import index_by_id, read_jsonl, write_jsonl
from crux.schema import UNDETERMINED, find_label_leaks, validate_record
from crux.track1 import decisiveness, extractor, leakage
from crux.track1.common import DEFAULT_CASES, load_cases
from crux.track1.quantities import extract_quantities

REGISTRY = Registry.load()
CASES = load_cases(DEFAULT_CASES)
CASE = CASES[0]
PAIR = (CASE["consultant_a_init"]["answer"], CASE["consultant_b_init"]["answer"])
X, Y = PAIR  # X = consultant A's answer, Y = consultant B's


def assert_stage_valid(test, records, stage):
    for record in records:
        test.assertEqual(validate_record(record, stage), [], record.get("crux_id"))
        test.assertEqual(find_label_leaks(record), [])
    with tempfile.TemporaryDirectory() as tmp:
        write_jsonl(Path(tmp) / f"{stage}.jsonl", records, stage)  # validates again and checks crux_id uniqueness


def proposal(claim, side="a", case=CASE, ok=True):
    return {"id": case["id"], "crux_id": f"{case['id']}#{side}", "proposed_by": f"consultant_{side}",
            "parse_ok": ok, "crux_claim": claim if ok else None}


# --------------------------------------------------------------------------- extractor (Prompt 1A)

class ExtractorTest(unittest.TestCase):
    def test_clean_proposal_records(self):
        reply = json.dumps({"disagreement_analysis": "They differ on step 2.",
                            "crux_claim": "7 * 13 = 91", "own_verdict": "TRUE"})
        backend = FakeBackend(REGISTRY, responder=reply)
        records = extractor.propose([CASE], backend, REGISTRY)
        self.assertEqual([r["crux_id"] for r in records], [f"{CASE['id']}#a", f"{CASE['id']}#b"])
        for record in records:
            self.assertTrue(record["parse_ok"])
            self.assertEqual(record["crux_claim"], "7 * 13 = 91")
            self.assertIs(record["proposer_own_verdict"], True)
            self.assertEqual(record["meta"]["prompt_version"], "v2")
            self.assertEqual(record["meta"]["call"]["temperature"], 0.7)  # decisions D3
            self.assertEqual(record["meta"]["call"]["quantization"], "awq")  # decisions D9
        assert_stage_valid(self, records, "crux_proposals")

    def test_each_consultant_argues_its_own_side(self):
        backend = FakeBackend(REGISTRY, responder='{"crux_claim": "c"}')
        extractor.propose([CASE], backend, REGISTRY)
        user_a, user_b = (call["messages"][-1]["content"] for call in backend.calls)
        self.assertIn(f"You answered {X}. The other expert answered {Y}.", user_a)
        self.assertIn(f"You answered {Y}. The other expert answered {X}.", user_b)
        self.assertEqual([call["model_key"] for call in backend.calls], ["consultant_a", "consultant_b"])

    def test_malformed_json(self):
        backend = FakeBackend(REGISTRY, responder="Sure! The crux is that 7 * 13 = 91.")
        records = extractor.propose([CASE], backend, REGISTRY)
        for record in records:
            self.assertFalse(record["parse_ok"])
            self.assertIsNone(record["crux_claim"])
            self.assertEqual(record["parse_error"], "no JSON object found")
            self.assertEqual(record["raw_output"], "Sure! The crux is that 7 * 13 = 91.")
        assert_stage_valid(self, records, "crux_proposals")

    def test_missing_claim_and_v1_prompt(self):
        backend = FakeBackend(REGISTRY, responder='```json\n{"disagreement_analysis": "x", "crux_claim": " "}\n```')
        record = extractor.propose([CASE], backend, REGISTRY, prompt_version="v1")[0]
        self.assertFalse(record["parse_ok"])
        self.assertEqual(record["parse_error"], "crux_claim missing or empty")
        self.assertEqual(record["meta"]["prompt_version"], "v1")

    def test_failed_call_is_recorded_not_parsed(self):
        def boom(model_key, messages, index):
            raise BackendError("HTTP 503 after retries")
        records = extractor.propose([CASE], FakeBackend(REGISTRY, responder=boom), REGISTRY)
        self.assertEqual(records[0]["call_error"], "HTTP 503 after retries")
        self.assertIsNone(records[0]["raw_output"])
        assert_stage_valid(self, records, "crux_proposals")


# --------------------------------------------------------------------------- leakage (L1, P5, L2)

MOLAR_MASS = {"A": "46,000 g/mole", "B": "5,030 g/mole", "C": "50.3 kg/mole", "D": "150,000 g/mole",
              "E": "50,300 g/mole", "F": "92,000 g/mole"}
SPLICING = {"A": "lariat", "B": "R-loops", "C": "polyA tail", "D": "antisense"}
CHEMISTRY = {"A": "2,3-diphenylbut-3-en-2-ol", "B": "(1S,2S,5R)-5-isopropyl-2-methoxy-2-methylcyclohexyl propionate",
             "C": "3,3-diphenylbutan-2-one", "D": "A = benzoquinone, B = dimethyl fumarate"}


class L1Test(unittest.TestCase):
    def test_leaky_crux_caught_by_wording(self):
        result = leakage.l1_check("R-loops play no role in Morpholino-induced exon skipping.", SPLICING)
        self.assertTrue(result["flagged"])
        self.assertEqual(result["contained_options"], ["B"])

    def test_lexical_overlap_threshold(self):
        options = {"A": "dimethyl fumarate and benzoquinone", "B": "maleic anhydride"}
        self.assertTrue(leakage.l1_check("Benzoquinone reacts with dimethyl fumarate as a dienophile.",
                                         options)["flagged"])
        result = leakage.l1_check("Dimethyl esters undergo hydrolysis under basic conditions.", options)
        self.assertEqual(result["lexical_overlap_option"], "A")
        self.assertFalse(result["flagged"])  # 1 of 3 content words: below 0.5

    def test_numeric_near_match_with_unit_normalization(self):
        result = leakage.l1_check("The polymer's molar mass equals 5.03 × 10^4 g/mol.", MOLAR_MASS)
        self.assertTrue(result["flagged"])
        self.assertEqual(result["reasons"], ["numeric_match"])
        self.assertEqual(sorted(m["option"] for m in result["numeric_matches"]), ["C", "E"])

    def test_numeric_within_one_percent_only(self):
        self.assertTrue(leakage.l1_check("The molar mass is 50,600 g/mol.", MOLAR_MASS)["flagged"])     # 0.6%
        self.assertFalse(leakage.l1_check("The molar mass is 51,100 g/mol.", MOLAR_MASS)["flagged"])    # 1.6%

    def test_different_units_do_not_match(self):
        result = leakage.l1_check("The monomer's molar mass is 50.3 g/mol.", MOLAR_MASS)
        self.assertEqual(result["numeric_matches"], [])  # 50.3 g/mol is not 50.3 kg/mol
        self.assertFalse(result["flagged"])

    def test_chemistry_locants_are_not_quantities(self):
        claim = "Protonation of the diol gives a tertiary carbocation that undergoes a 1,2-phenyl shift."
        self.assertFalse(leakage.l1_check(claim, CHEMISTRY)["flagged"])
        self.assertEqual(extract_quantities(CHEMISTRY["B"]), [])

    def test_clean_intermediate_step(self):
        self.assertFalse(leakage.l1_check("7 * 13 = 91", {"A": "prime", "B": "not prime"})["flagged"])

    def test_p5_flags_are_logged(self):
        flags = leakage.p5_flags("The product of step 2 is a ketone and it is chiral.")
        self.assertTrue(flags["multi_clause"])
        self.assertTrue(flags["deictic"])
        self.assertIn("step 2", flags["deictic_matches"])
        self.assertTrue(leakage.p5_flags("Option B is wrong.")["deictic"])
        self.assertTrue(leakage.p5_flags("I believe water boils at 100 C.")["deictic"])
        self.assertEqual(leakage.p5_flags("Water boils at 100 C at 1 atm."),
                         {"multi_clause": False, "multi_clause_matches": [], "deictic": False,
                          "deictic_matches": []})


class LeakageStageTest(unittest.TestCase):
    def test_records_and_no_claim(self):
        proposals = [proposal("7 * 13 = 91", "a"), proposal(None, "b", ok=False)]
        records = leakage.check(proposals, index_by_id(CASES))
        self.assertEqual([r["status"] for r in records], ["checked", "no_claim"])
        self.assertIsNone(records[0]["l2"])
        self.assertIsNone(records[1]["is_leaky"])
        assert_stage_valid(self, records, "leakage")

    def test_l2_either_consultant_yes_flags(self):
        def probe(model_key, messages, index):
            return "Reasoning...\nRESTATES: " + ("YES" if model_key == "consultant_b" else "NO")
        backend = FakeBackend(REGISTRY, responder=probe)
        record = leakage.check([proposal("7 * 13 = 91")], index_by_id(CASES), backend=backend,
                               registry=REGISTRY, run_l2=True)[0]
        self.assertFalse(record["l1"]["flagged"])
        self.assertTrue(record["l2"]["flagged"])
        self.assertTrue(record["is_leaky"])
        self.assertEqual({c["temperature"] for c in record["l2"]["calls"].values()}, {0.0})  # decisions D3
        user = backend.calls[0]["messages"][-1]["content"]
        self.assertIn(f"- {X}: {CASE['choices'][X]}", user)
        self.assertIn(f"- {Y}: {CASE['choices'][Y]}", user)
        assert_stage_valid(self, [record], "leakage")

    def test_l2_unparseable_probe(self):
        backend = FakeBackend(REGISTRY, responder="I cannot tell.")
        record = leakage.check([proposal("7 * 13 = 91")], index_by_id(CASES), backend=backend,
                               registry=REGISTRY, run_l2=True)[0]
        self.assertIsNone(record["l2"]["flagged"])
        self.assertFalse(record["is_leaky"])  # undetermined L2 never overrides a clean L1 silently...
        self.assertFalse(record["l2"]["by_consultant"]["consultant_a"]["parse_ok"])  # ...but is visible here
        self.assertEqual(leakage.parse_restates("x\nRESTATES: no"), False)


# --------------------------------------------------------------------------- decisiveness (Prompt 1B v2)

def candidates(messages):
    match = decisiveness._CANDIDATES_RE.search(messages[-1]["content"])
    return list(match.groups())


def reply(if_true, if_false):
    return json.dumps({"reasoning_if_true": "r", "answer_if_true": if_true,
                       "reasoning_if_false": "r", "answer_if_false": if_false})


def scripted(behaviour):
    """behaviour(role, sample_index_for_role, listed_order) -> raw reply."""
    counts = {}

    def responder(model_key, messages, index):
        k = counts.get(model_key, 0)
        counts[model_key] = k + 1
        return behaviour(model_key, k, candidates(messages))
    return responder


def run_decisive(behaviour, side="a", k=3):
    backend = FakeBackend(REGISTRY, responder=scripted(behaviour))
    record = decisiveness.evaluate([proposal("7 * 13 = 91", side)], index_by_id(CASES), backend, REGISTRY, k=k)[0]
    return record, backend


class DecisivenessTest(unittest.TestCase):
    def test_clean_decisive_crux(self):
        record, backend = run_decisive(lambda role, k, order: reply(X, Y))
        self.assertEqual((record["d"], record["d_a"], record["d_b"]), (1.0, 1.0, 1.0))
        self.assertEqual(record["agreed_mapping"], {"if_true": X, "if_false": Y})
        self.assertEqual((record["passing_pairs"], record["n_pairs"]), (9, 9))
        self.assertFalse(record["veto_flag"])
        self.assertEqual(record["mapping_agree_rate"], 1.0)
        assert_stage_valid(self, [record], "decisiveness")

    def test_samples_stored_per_consultant_with_shown_order(self):
        record, backend = run_decisive(lambda role, k, order: reply(X, Y))
        self.assertEqual([(s["consultant"], s["k"]) for s in record["samples"]],
                         [("consultant_a", i) for i in range(3)] + [("consultant_b", i) for i in range(3)])
        shown = [candidates(call["messages"]) for call in backend.calls]
        self.assertEqual([s["answer_order"] for s in record["samples"]], shown)  # P6: recorded order = shown order
        self.assertTrue(all(sorted(order) == sorted(PAIR) for order in shown))
        self.assertEqual(len({s["seed"] for s in record["samples"]}), 6)

    def test_order_randomization_is_seeded_and_varies(self):
        orders = [tuple(decisiveness.answer_order(42, f"x#{i}", "consultant_a", 0, ("A", "B"))) for i in range(40)]
        self.assertEqual(set(orders), {("A", "B"), ("B", "A")})
        self.assertEqual(orders, [tuple(decisiveness.answer_order(42, f"x#{i}", "consultant_a", 0, ("A", "B")))
                                  for i in range(40)])

    def test_one_consultant_veto(self):
        # B never lets the crux decide anything: it keeps its own answer under both branches.
        record, _ = run_decisive(lambda role, k, order: reply(X, Y) if role == "consultant_a" else reply(Y, Y))
        self.assertEqual(record["d"], 0.0)
        self.assertEqual((record["flip_rate_a"], record["flip_rate_b"]), (1.0, 0.0))
        self.assertTrue(record["veto_flag"])
        self.assertEqual(record["vetoed_by"], "consultant_b")
        self.assertIs(record["veto_by_proposer"], False)  # A proposed it; the opponent vetoed (attack A1 shape)
        self.assertFalse(record["mapping_conflict"])
        assert_stage_valid(self, [record], "decisiveness")

    def test_undetermined_branch(self):
        record, _ = run_decisive(
            lambda role, k, order: reply(X, Y) if role == "consultant_a" else reply(X, UNDETERMINED))
        b_samples = [s for s in record["samples"] if s["consultant"] == "consultant_b"]
        self.assertTrue(all(s["parse_ok"] and s["if_false"] == UNDETERMINED for s in b_samples))
        self.assertEqual((record["flip_rate_b"], record["d"]), (0.0, 0.0))
        self.assertTrue(record["veto_flag"])

    def test_partial_undetermined_lowers_d_without_veto(self):
        record, _ = run_decisive(lambda role, k, order: reply(UNDETERMINED, Y)
                                 if role == "consultant_b" and k == 0 else reply(X, Y))
        self.assertAlmostEqual(record["flip_rate_b"], 2 / 3)
        self.assertAlmostEqual(record["d"], 6 / 9)
        self.assertFalse(record["veto_flag"])

    def test_malformed_json_sample(self):
        record, _ = run_decisive(lambda role, k, order: "{not json" if role == "consultant_b" and k == 1
                                 else reply(X, Y))
        bad = [s for s in record["samples"] if not s["parse_ok"]]
        self.assertEqual(len(bad), 1)
        self.assertEqual((bad[0]["consultant"], bad[0]["k"], bad[0]["raw_output"]), ("consultant_b", 1, "{not json"))
        self.assertIsNone(bad[0]["if_true"])
        self.assertAlmostEqual(record["d"], 6 / 9)
        assert_stage_valid(self, [record], "decisiveness")

    def test_mapping_conflict(self):
        record, _ = run_decisive(lambda role, k, order: reply(X, Y) if role == "consultant_a" else reply(Y, X))
        self.assertEqual((record["d"], record["mapping_agree_rate"]), (0.0, 0.0))
        self.assertTrue(record["mapping_conflict"])
        self.assertFalse(record["veto_flag"])

    def test_mapping_tie_gives_no_agreed_mapping(self):
        m1, m2 = (X, Y), (Y, X)
        script = {"consultant_a": [m1, m1, m2], "consultant_b": [m1, m2, m2]}
        record, _ = run_decisive(lambda role, k, order: reply(*script[role][k]))
        self.assertTrue(record["mapping_tie"])
        self.assertIsNone(record["agreed_mapping"])
        self.assertEqual(record["d"], 0.0)

    def test_branch_normalization(self):
        pair = ("B", "D")
        self.assertEqual(decisiveness.normalize_branch("Option B", pair), ("B", None))
        self.assertEqual(decisiveness.normalize_branch("(D)", pair), ("D", None))
        self.assertEqual(decisiveness.normalize_branch("undetermined.", pair), (UNDETERMINED, None))
        self.assertIsNone(decisiveness.normalize_branch("B, D, or UNDETERMINED", pair)[0])  # echoed template
        self.assertIsNone(decisiveness.normalize_branch("C", pair)[0])  # an answer not in play

    def test_no_claim_proposal_is_skipped(self):
        record = decisiveness.evaluate([proposal(None, ok=False)], index_by_id(CASES), FakeBackend(REGISTRY),
                                       REGISTRY)[0]
        self.assertEqual(record["status"], "no_claim")
        assert_stage_valid(self, [record], "decisiveness")


# --------------------------------------------------------------------------- CLI

class CliTest(unittest.TestCase):
    def test_three_stages_write_one_file_each(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("crux.track1.common.RUNS_DIR", Path(tmp)):
            cli_main(["track1", "propose", "--run", "t", "--backend", "fake", "--limit", "2"])
            cli_main(["track1", "leakage", "--run", "t", "--l2", "--backend", "fake"])
            cli_main(["track1", "decisive", "--run", "t", "--backend", "fake", "--k", "2"])
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "t").iterdir()),
                             ["crux_proposals.jsonl", "decisiveness.jsonl", "leakage.jsonl"])
            for stage in ("crux_proposals", "leakage", "decisiveness"):
                records = read_jsonl(Path(tmp) / "t" / f"{stage}.jsonl", stage)
                self.assertEqual(len(records), 4)
                self.assertTrue(all(r["meta"]["inputs"]["cases"] for r in records))
            decisive = read_jsonl(Path(tmp) / "t" / "decisiveness.jsonl", "decisiveness")
            self.assertTrue(all(r["d"] == 1.0 and r["k"] == 2 for r in decisive))


if __name__ == "__main__":
    unittest.main()
