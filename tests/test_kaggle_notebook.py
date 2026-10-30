"""The Kaggle notebook's commands must match our CLI and configs/models.yaml. Nothing here starts a server."""

import json
import unittest
from pathlib import Path

from crux.backends import Registry
from crux.cli import build_parser
from crux.prompts import load_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_consultants.ipynb"


def tagged_source(notebook, tag):
    cells = [c for c in notebook["cells"] if tag in c.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1, f"expected exactly one cell tagged {tag!r}"
    source = cells[0]["source"]
    return "".join(source) if isinstance(source, list) else source


class KaggleNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.env = {}
        exec(tagged_source(cls.notebook, "config"), cls.env)
        cls.env["REPO_DIR"] = str(REPO_ROOT)  # read models.yaml from this checkout instead of /kaggle/working
        exec(tagged_source(cls.notebook, "server_cmds"), cls.env)
        exec(tagged_source(cls.notebook, "steps"), cls.env)

    def test_valid_notebook_with_code_cells_that_parse(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<cell>", "exec")

    def test_settings(self):
        env = self.env
        self.assertEqual(env["VLLM_VERSION"], "0.32.0")
        self.assertEqual(env["MAX_NUM_SEQS"], 8)
        self.assertGreaterEqual(env["WORKERS"], env["MAX_NUM_SEQS"])
        self.assertEqual(env["GPUS"], {"consultant_a": "0", "consultant_b": "1"})

    def test_server_commands_follow_models_yaml(self):
        registry = Registry.load()
        for key, cmd in self.env["SERVER_CMDS"].items():
            spec = registry.spec(key)
            flags = {cmd[i]: cmd[i + 1] for i in range(3, len(cmd) - 1) if cmd[i].startswith("--")}
            self.assertEqual(cmd[1:3], ["serve", spec.hf_repo])
            self.assertEqual(flags["--revision"], spec.revision)
            self.assertEqual(flags["--tokenizer-revision"], spec.revision)
            self.assertEqual(flags["--served-model-name"], spec.served_name)
            self.assertEqual((flags["--quantization"], flags["--dtype"]), ("awq", "float16"))  # decisions D9
            self.assertEqual(flags["--max-model-len"], str(spec.extra["max_model_len"]))
            self.assertEqual(flags["--tensor-parallel-size"], "1")
            self.assertEqual(flags["--max-num-seqs"], "8")
            self.assertEqual(flags["--port"], str(self.env["PORTS"][key]))

    def test_env_file_variables_match_backend(self):
        registry = Registry.load()
        self.assertEqual({registry.spec(k).extra["base_url_env"] for k in self.env["PORTS"]},
                         {"VLLM_BASE_URL_A", "VLLM_BASE_URL_B"})

    def test_crux_commands_parse(self):
        parser = build_parser()
        smoke = parser.parse_args(self.env["SMOKE_CMD"])
        self.assertEqual((smoke.group, smoke.command, smoke.backend, smoke.n), ("smoke", "consultants", "vllm", 2))
        steps = self.env["crux_steps"]()
        self.assertEqual([s[:2] for s in steps], [["track1", "propose"], ["track1", "leakage"],
                                                  ["track1", "decisive"], ["track1", "report"]])
        for argv in steps[:3]:
            args = parser.parse_args(argv)
            self.assertEqual((args.backend, args.workers, args.run), ("vllm", self.env["WORKERS"],
                                                                      self.env["RUN_NAME"]))
            self.assertTrue(callable(args.func))
        self.assertTrue(parser.parse_args(steps[1]).l2)
        propose = parser.parse_args(steps[0])
        self.assertEqual(propose.prompt_version, self.env["PROPOSAL_PROMPT_VERSION"])
        load_prompt("crux_proposal", propose.prompt_version)  # the configured version exists
        self.assertEqual(parser.parse_args(steps[2]).k, 3)
        self.assertEqual(parser.parse_args(steps[3]).run, self.env["RUN_NAME"])


if __name__ == "__main__":
    unittest.main()
