# Team Decisions Log

Decisions that refine Doc 1 / Doc 2 without editing them. Newest first. Section references (§, P1–P7) point to
[step0_audit.md](step0_audit.md).

## 2026-10-24: Step-0 open questions

### D1. Protocol fixes for the kill test (audit §3.3)

| Fix | Status | Detail |
|---|---|---|
| P1 (no unilateral veto) | **Adopted, provisional** | Before P1 becomes a hard abstain rule, report the veto rate on seed_30, because an honest but stubborn consultant also vetoes. |
| P2 (audit both candidates, abstain if mapped answers disagree) | **Adopted** | Replaces the tie-break on Layer-1 c. |
| P3 (polarity symmetrization, judge C and ¬C) | Deferred | |
| P4 (leakage probe by both consultants + numeric normalization in L1) | **Adopted** | Flag leaky if either consultant's probe says YES. |
| P5 (atomicity / deixis checks) | **Log flags only** | Not used as filters in the kill test. |
| P6 (randomized answer order in 1B + `UNDETERMINED` branch) | **Adopted** | |
| P7 (per-consultant injection slot in 1A/1B/1C) | Deferred | Needed for the sabotage condition later. |

### D2. Kill Criterion 1 scoring (audit §3.2 item 5)

- **Primary:** paired McNemar on **covered cases only**: the crux route against the full-question two-way judge
  (Prompt 2B) on the same cases. A minimum coverage is pre-registered alongside the go/no-go thresholds in README.md.
- **Secondary:** end-to-end accuracy with DR-consensus fallback when the audit abstains (Doc 2 Table 2).
- Abstentions are **never** scored as right or wrong.

### D3. Sampling parameters (audit §4.1 item 3)

| Call | Temperature | top_p |
|---|---|---|
| Consultants: initial answers, DR turns, Prompt 1A, Prompt 1B | 0.7 | 0.9 |
| Leakage probe (Prompt 1C) | 0 | — |
| Judges: verdict and `c_prob` pass | 0 | — |
| Judges: `c_cons` samples | 0.7 | — |
| Both Layer-2 variants | 0 | — |

Every call records its resolved parameters in its stage `meta`.

### D4. Judge layers (audit §3.2 item 7)

Keep Llama-3.2-1B-Instruct as Layer 2 and log the family confound. Layer 1 is Qwen and shares a family with
consultant A; Layer 2 is Llama and shares a family with consultant B. Add **Gemma-3-1B-it** as a neutral-family
Layer-2 ablation.

### D5. CLAUDE.md

CLAUDE.md's "Base repo facts" was corrected by the team (GPQA `kimi.json` `Question` gap, HLE `question` key, A–D
choice parsing, answer-revealing fields).

### D6. Explainer

"Our Capstone Idea, Explained Simply" stays as historical context. Doc 1 v2 supersedes it where they differ.

### D7. Schema (audit §5)

- **Adopted now (§5.1):**
  - per-stage JSONL files keyed by `id` / `crux_id`;
  - per-stage `meta`;
  - a separate `gate` stage;
  - an `eval` block as the label firewall.
- **§5.2 fields are added as each stage is built**, not up front.

### D8. seed_30

40 cases from GPQA and SuperGPQA pair A (GPT-4o vs Claude Sonnet 4), field whitelist, labels only in `eval`.

## Implementation choices (flag any you disagree with)

- **Consultant roles in seed records.** `consultant_a` is GPT-4o and `consultant_b` is Claude Sonnet 4 (file order
  in pair A). In the kill test, `consultant_a` is Qwen3-8B and `consultant_b` is Llama-3.1-8B-Instruct.
- **Stratification.** Each dataset contributes 20 cases: 10 where consultant A is initially right and 10 where B is.
- **SuperGPQA Medicine excluded from seed_30.** This matches Track 3's kill-test filter.
- **`reasoning` is a list of steps**, as in the base data and `run.py`. Doc 2 shows a single string.
- **The first stage is named `cases`.** It holds question, choices, both consultants' initial answers, and `eval`.
  Track 3's 8B runs write the same stage.
