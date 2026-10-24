import hashlib
import re
import unittest
from pathlib import Path

from crux.prompts import PromptError, load_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC2 = REPO_ROOT / "docs" / "Doc2_Prompts_and_Member_Playbook_v2.md"

EXPECTED_PLACEHOLDERS = {
    "crux_proposal": ["choices_formatted", "my_answer", "my_reasoning", "other_answer", "other_reasoning", "question"],
    "decisiveness": ["answer_1", "answer_2", "choices_formatted", "crux_claim", "question"],
    "restatement_probe": ["answer_1", "answer_2", "choice_text_1", "choice_text_2", "crux_claim"],
    "judge_crux": ["crux_claim"],
    "judge_full_q_2way": ["answer_1", "answer_2", "choice_text_1", "choice_text_2", "question"],
    "judge_q_plus_crux": ["crux_claim", "question"],
}


def _fill(prompt):
    return {name: f"<{name}>" for name in prompt.placeholders}


class PromptFilesTest(unittest.TestCase):
    def test_v1_prompts_are_verbatim_doc2_blocks(self):
        doc = DOC2.read_text(encoding="utf-8").replace("\r\n", "\n")
        blocks = re.findall(r"```text\n(.*?)\n```", doc, re.DOTALL)
        for name in EXPECTED_PLACEHOLDERS:
            prompt = load_prompt(name)
            self.assertIn(prompt.text.rstrip("\n"), blocks, name)

    def test_placeholders(self):
        for name, expected in EXPECTED_PLACEHOLDERS.items():
            self.assertEqual(load_prompt(name).placeholders, expected, name)

    def test_sha256_is_over_lf_text(self):
        prompt = load_prompt("judge_crux")
        self.assertEqual(prompt.sha256, hashlib.sha256(prompt.text.encode("utf-8")).hexdigest())
        self.assertNotIn("\r", prompt.text)
        self.assertEqual(prompt.meta()["prompt_file"], "prompts/v1/judge_crux.txt")


class PromptRenderTest(unittest.TestCase):
    def test_messages_split_system_and_user(self):
        prompt = load_prompt("judge_crux")
        messages = prompt.messages(crux_claim="7 * 13 = 91")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn('"7 * 13 = 91"', messages[1]["content"])
        self.assertTrue(messages[1]["content"].rstrip().endswith("VERDICT: FALSE"))

    def test_json_braces_survive_and_placeholders_fill(self):
        prompt = load_prompt("decisiveness")
        user = prompt.messages(**_fill(prompt))[1]["content"]
        self.assertIn('"answer_if_true": "<answer_1> or <answer_2>"', user)
        self.assertIn("{\n", user)
        self.assertNotRegex(user, r"\{[a-z_0-9]+\}")

    def test_field_values_are_not_resubstituted(self):
        prompt = load_prompt("judge_q_plus_crux")
        rendered = prompt.render(question="Is {crux_claim} literal?", crux_claim="x")
        self.assertIn("Is {crux_claim} literal?", rendered)

    def test_missing_and_unknown_fields_raise(self):
        prompt = load_prompt("judge_crux")
        with self.assertRaises(PromptError):
            prompt.render()
        with self.assertRaises(PromptError):
            prompt.render(crux_claim="x", question="extra")

    def test_unknown_prompt(self):
        with self.assertRaises(PromptError):
            load_prompt("does_not_exist")


if __name__ == "__main__":
    unittest.main()
