# Capstone: Crux-Audited Consensus for Weak-Judge Oversight

Team PW26_BJD_09 (Swaroop, Shreesh, Sruthi, Sunay), guide Dr. Bhaskarjyoti Das.
Built on `collaborative-dr/`, a clone of https://github.com/ChicagoHAI/collaborative-dr, the code for
Jiang et al., "Collaborative Disagreement Resolution for Scalable Oversight" (ICML 2026, arXiv 2607.01251).

## Source of truth
- `docs/Doc1_Team_Plan_and_Claude_Brief_v2.md`: method, kill criteria, schedule, who owns what
- `docs/Doc2_Prompts_and_Member_Playbook_v2.md`: prompts, shared JSONL schema, per-track playbooks
- `docs/2607.01251v1.pdf`: the base paper

If the code and these docs disagree, flag it and ask before changing either one.

## Base repo facts (checked)
- `collaborative-dr/data/{GPQA,SuperGPQA,HLE}/` holds the paper's precomputed initial answers from frontier models:
  `gpt_4o.json` + `claude.json` (pair A) and `glm.json` + `kimi.json` (pair B). Each file is a list of records with
  `index`, `choices`, `correct_choice`, `model_answer`, `model_reasoning`, `model_correct`. GPQA and SuperGPQA use a
  `Question` field; HLE uses `question`.
- Within each pair the files are already aligned by index and already filtered to the paper's disagreement set: every record
  has the two models disagreeing and exactly one of them right (620 cases in total). These are the source for seed_30.
- `run.py` calls models only through OpenRouter (`call_openrouter`) and reads its config from `collaborative-dr/.env`
  (see `.env.example`). Running our 8B or local models needs a backend adapter on our side.
- `baselines/llm_debate/` is a git submodule (Khan et al., 2024 debate code). It is only needed for the debate baseline.

## Rules
- Do not edit files inside `collaborative-dr/` unless explicitly asked. Our code lives in its own package and hooks into theirs.
- Every pipeline stage reads and writes the shared JSONL schema (Doc 2, Section 1). Any schema change must be called out in the commit message, because 4 people depend on it.
- Records with `source = "frontier_precomputed"` (seed_30) are for plumbing only. Never include them in kill-test metrics.
- One backend per model for the whole kill test. Every record fills `run_meta`.
- Judges run locally (Hugging Face transformers or vLLM) because `c_prob` needs token log-probabilities.
- Prompts live as versioned text files in a prompts folder, never as strings inside code.
- No API keys in code or git. Use a `.env` file that is gitignored.
- Choose tau and tune prompts on the dev split (or seed_30) only, never on the test split.
- The go/no-go thresholds are written in README.md before the 300-case run. Do not change them after results come in.
