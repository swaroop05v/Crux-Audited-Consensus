# Capstone: Crux-Audited Consensus for Weak-Judge Oversight

Team PW26_BJD_09 (Swaroop, Shreesh, Sruthi, Sunay), guide Dr. Bhaskarjyoti Das.
Built on `collaborative-dr-main/`, the code for Jiang et al., "Collaborative Disagreement
Resolution for Scalable Oversight" (ICML 2026, arXiv 2607.01251).

## Source of truth
- `docs/Doc1_Team_Plan_v2.md`: method, kill criteria, schedule, who owns what
- `docs/Doc2_Playbook_v2.md`: prompts, shared JSONL schema, per-track playbooks
- `docs/paper.pdf`: the base paper

If the code and these docs disagree, flag it and ask before changing either one.

## Rules
- Do not edit files inside `collaborative-dr-main/` unless explicitly asked. Our code lives in its own package and hooks into theirs.
- Every pipeline stage reads and writes the shared JSONL schema (Doc 2, Section 1). Any schema change must be called out in the commit message, because 4 people depend on it.
- Records with `source = "frontier_precomputed"` (seed_30) are for plumbing only. Never include them in kill-test metrics.
- One backend per model for the whole kill test. Every record fills `run_meta`.
- Judges run locally (Hugging Face transformers or vLLM) because `c_prob` needs token log-probabilities.
- Prompts live as versioned text files in a prompts folder, never as strings inside code.
- No API keys in code or git. Use a `.env` file that is gitignored.
- Choose tau and tune prompts on the dev split (or seed_30) only, never on the test split.
- The go/no-go thresholds are written in README.md before the 300-case run. Do not change them after results come in.
