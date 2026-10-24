# Step-0 Architectural Audit

**Project:** Crux-Audited Consensus for Weak-Judge Oversight (PW26_BJD_09)
**Date:** 2026-10-24 · **Base repo commit:** `collaborative-dr@5661bed` · **Scope:** read-only audit, no implementation code written.

Sources read: every file in `collaborative-dr/` (`run.py`, `run_ablation.py`, `prompts/`, `ablation/`, `baselines/*.py`, `baselines/prompts/`, all 12 files in `data/`, `README.md`, `.env.example`, `pyproject.toml`), how the `baselines/llm_debate` submodule is invoked, Doc 1, Doc 2, the base paper (arXiv 2607.01251v1, incl. appendices C–E), and "Our Capstone Idea, Explained Simply". Every data claim below was checked by loading the JSON files, not inferred from the README.

---

## TL;DR: things that need a team decision before Day 3

1. **The base code has no crux isolation step.** The paper's "isolate the crux" exists only as a per-turn `conflicts` list in the consultant's JSON. Nothing extracts a crux for the judge (§1.3). Doc 1 §1 ("the remaining point of disagreement is isolated for the judge") describes the paper, not the code.
2. **`run.py` silently mangles multiple-choice options beyond D** (`format_choices_text`, `run.py:93`, regex `[A-D]`). Options E–J are glued onto the D line in 152/153 SuperGPQA cases and 86/88 HLE cases. SuperGPQA is where most of our 300 cases come from, so our adapter must pass `choices` as a dict, which takes the dict branch at `run.py:86` and bypasses the bug.
3. **HLE is broken in `run.py`**. It reads `entry["Question"]` (`run.py:268`, `:648`), and HLE uses `question`, so the question text is empty. HLE `question` also embeds the original answer list with different letters from the shuffled `choices` field (75/88 and 67/80 records disagree). Keep HLE out of seed_30. Doc 2 already drops it from the kill test.
4. **One manipulative consultant can veto the honest crux** (§3.1, attack A1). The selection rule in Doc 2 does not defend against this. Doc 1's statement that "the honest consultant's own candidate is still in the pool" is true, but that candidate can be pushed to d = 0. I propose a fix in §3.3.
5. **Selection needs the judge's c (tie-break on Layer-1 checkability)**, so Track 1 cannot finish selection before Track 2 runs. Fix: Track 2 judges *all* non-leaky candidates, and a separate `gate` step does selection. The schema has to change for this (§5).
6. **Seed data leaks the answer through extra fields** (`Correct Answer`, `Incorrect Answer 1-3`, `rationale`, `answer`, `answer_letter`, `correct_answer`, `debug_meta`). The seed builder must use a field whitelist (§2.4).
7. **CLAUDE.md "Base repo facts" is slightly wrong.** GPQA `kimi.json` has no `Question` field in 51 of 101 records. I have not edited CLAUDE.md. Please confirm before I change it.

---

## 1. Base repo: how `run.py` works, and where we hook in

### 1.1 Control flow (file:line)

| Stage | Where | What it actually does |
|---|---|---|
| Config | `main()` `run.py:797-884` | Reads `collaborative-dr/.env` or env vars: `DATASET`, `CONSULTANT_{1,2}`, `CONSULTANT_{1,2}_FILENAME`, `JUDGE`, `PROMPTS_SET`, `RESULTS_DIR`. Loads `reasoning_generation.txt`, `consultant.txt`, `judge.txt`, `judge_system.txt` (`:875-880`). `CONSULTANT_SYSTEM_PROMPT` is hard-set to `""` (`:883`). |
| Load initial responses | `load_model_data` `:145-157` | Loads both JSON lists and inner-joins them on `index`. It does **not** filter for disagreement; it trusts that the files are already filtered. |
| Case subset + order swap | `:913-920` | `random.Random(42)` draws a swap flag for each case in `cases_to_run`. Because the draw is over the subset, `--offset`, `--limit` and `--cached` change which cases get swapped, so a resumed run is not bit-identical to a full one. |
| Seed consultant state | `build_consultant_state` `:262-296` | Re-creates the turn-0 conversation as `[user: reasoning_generation.txt filled with Question and choices, assistant: {"answer","reasoning"} from the JSON file]`. Initial answers are **never regenerated**; they are replayed as fake assistant turns. |
| Discussion loop | `run_case` `:684-692` | `while answers differ and turns < max_turns (default 5, :767)`: `execute_consultant_turn`. |
| One turn | `execute_consultant_turn` `:444-484` | Snapshots both latest answers and reasoning (`:453-460`), then updates consultant 1 and consultant 2 **simultaneously** against the snapshot (matches the paper). |
| Consultant update | `run_consultant_update` `:370-441` | Appends `consultant.txt` (with the other side's latest answer and reasoning) as a user turn to the consultant's own growing history and calls the model. Parses `answer`, `reasoning`, `conflicts[]` (`issue`, `winner`, `explanation`), `confidence` (`:412-415`). `answer` is only `.strip().upper()`: **no check that it is a valid label, or that it is one of the two answers in play**. |
| Model call | `call_openrouter` `:175-211` | **The only network call site.** Hard-coded OpenRouter URL, `temperature=0.2` default (never overridden), no `max_tokens`, no seed, 90 s timeout. |
| Retry | `fetch_json_response_with_retries` `:214-259` | Up to 3 attempts when JSON is malformed (`force_json_parse`, `:160`, takes the outermost `{...}`). On HTTP 400 it retries once with the system prompt merged into the first user message. Any other HTTP error (e.g. 429) is raised, and the whole case becomes `status: "error"` (`:927-935`). No backoff. |
| Judge | `run_judge_review` `:556-619` | **Always called**, whether the consultants agreed or not (`:699`). Messages are `[system: judge_system.txt with question and choices, user: judge.txt + "Full verification history" (every raw JSON response from both consultants, :576-582)]`. `chat_history` is always empty, so the "read the entire conversation history" line in the system prompt does nothing. |
| Output | `run_case` return `:719-744`, `save_results` `:747` | One JSON list per run: final answers, `judge_final_answer`, `judge_winner`, consultant summaries with `conflict_history`, `metadata.consultant_order`. **It does not compute correctness.** `pyproject.toml` declares an `acc` module and an `agent-acc` script, but `acc.py` does not exist. |

`run_ablation.py` is `run.py` plus two features:
- **Best-of-N.** Each consultant samples N candidates, and the *opponent* model picks the most persuasive one (`ablation/BoN/preference.txt`).
- **Sycophancy injection.** A per-consultant text block is prepended to each DR-turn user prompt (`ablation/sycophancy/*.txt`), and it follows the model when the order is swapped.

The sycophancy injection hook is exactly what our planned **sabotage condition** needs for the DR turns.

### 1.2 Hook points for our code

We never edit `collaborative-dr/`. Integration takes three thin adapters, and the crux pipeline itself runs alongside DR rather than inside it.

| Hook | Mechanism |
|---|---|
| **Backend injection** | `fetch_json_response_with_retries` looks up the module-global `call_openrouter` at call time (`run.py:231`), so our runner does `import run; run.call_openrouter = our_backend.chat` before calling anything. The same pattern works for `run_ablation`. |
| **Data adapter** | We call `run.run_case(case, models, prompts, args, swap)` directly (`:640`) instead of `run.main()`. We build `case = {"index", "model1": {...}, "model2": {...}}` from our JSONL, with `Question` (capital Q), `choices` **as a dict** (avoids the A-D regex bug), `correct_choice`, `model_answer`, `model_reasoning`. We set the globals `run.CONSULTANT_SYSTEM_PROMPT`, `run.JUDGE_SYSTEM_PROMPT_TEMPLATE` and `run.LOG_PROMPTS` ourselves. `args` only needs `.max_turns` and `.sleep`. |
| **Output parser** | We convert the `run_case` return dict into our `dr_baseline` block: exit mode TC/AT/DC, per-turn actions (Persistence/Recovery/Overthinking/Stubbornness), consensus, judge adoption. We undo the swap via `metadata.consultant_order`. |
| **Prompts reused, not copied** | `collaborative-dr/prompts/reasoning_generation.txt` for 8B initial answers (Doc 2 Track 3 already says this), and `consultant.txt`, `judge.txt`, `judge_system.txt` unchanged for base DR. |
| **DR-native "crux" (optional ablation)** | The last-turn `conflicts[].issue` strings in `conflict_history` are the closest thing DR has to a crux. They are a free "DR's own crux" comparison against our Prompt 1A cruxes. |
| **Sabotage** | `run_ablation.py`'s `CONSULTANT_n_SYCOPHANCY_PROMPT` hook covers the DR turns. Our own prompts (1A, 1B, 1C) need an equivalent per-consultant injection slot. |

### 1.3 Where the code differs from the paper

| # | Paper says | Code does | Impact on us |
|---|---|---|---|
| D1 | Each turn, each consultant "forms a crux c = g(m1, m2)" (Def. 3, Table 1); DR "isolates the crux" for the judge (Fig. 1). | There is no crux object. The consultant lists `conflicts`. The judge gets the 2 most recent conflict summaries, first 3 issues each (`summarize_conflicts`, `:299`), plus every raw turn JSON. | Our "extract the crux up front" is new relative to the **code**, not just the paper. The Doc 1 phrasing should say the paper describes crux isolation but the implementation only lists conflicts. |
| D2 | Action is retain or adopt: a_t ∈ {a_{t-1}, a^{-i}_{t-1}} (Def. 1). | Any string is accepted as the new answer. A consultant can jump to a third option. | The TC/AT/DC taxonomy and `is_agreement_trap` need a fourth bucket ("consensus on a third answer"), and the crux mapping (which only covers two answers) can't reach it. 8B models will do this more often than frontier models. |
| D3 | DR uses T = 1, top_p = 1 for consultants and judge (App. C.1). | `call_openrouter` default `temperature=0.2`, never changed. | The repo does not reproduce the paper's setting. We must choose and record our own temperatures. |
| D4 | Double Consultancy: "five-turn arguments", T = 1, n = 5 (App. C.1). | `double_consultancy.py` shows the judge each model's single initial trace at `temperature: 0` (`:239`). Its regex parser expects `Winner: Consultant N`. | Our baseline numbers are comparable to the code, not to the paper's description. Say this in the write-up. |
| D5 | Naive Judge: "T = 1 for deterministic outputs" (self-contradictory). | `temperature: 0` (`naive_judge.py:219`). | Same as D4. |
| D6 | Debate (Khan et al.) on these datasets. | `baselines/llm_debate` is the unmodified upstream repo (2024-03, QuALITY-oriented, `python -m core.main ...`). **The base repo has no adapter** from `data/*.json` to its input format. | The debate baseline is not runnable as shipped. Doc 1 doesn't need it for the kill test; leave it out until Phase 2. |
| D7 | README: "Each result records … correctness". | No correctness field is written. `acc.py` is missing. | We compute correctness ourselves (`correct_choice` is in the output). |
| D8 | Paper evaluates HLE-MC. | `Question` key bug (§TL;DR 3): with the shipped code and data, HLE consultants and judge get an empty question, and the judge sees "Question unavailable.". | Either the released code is not what produced the HLE numbers, or those numbers came from an empty-question run. Not our problem to resolve, but don't cite HLE numbers as replicable. Both baselines have the same bug (`naive_judge.py:399`, `double_consultancy.py:430`). |

Smaller notes:
- `judge.txt` tells the judge the agreement status explicitly. This is one reason the judge adopts the consensus 99.5% of the time.
- The judge prompt is long for a 1.5B model: the full verification history can reach about 10k tokens after 5 turns.

---

## 2. Data on disk

### 2.1 Inventory (verified)

`collaborative-dr/data/` has 3 datasets × 4 files = 12 files. Every file is a JSON list with unique `index` values. Within each pair the files join 1:1 on `index`, both models always disagree, exactly one is right, and the `choices` strings match exactly. **Total 620 cases**, which matches paper footnote 4.

| Dataset | Pair A: `gpt_4o.json` + `claude.json` | Pair B: `glm.json` + `kimi.json` | Option count | Notes |
|---|---|---|---|---|
| GPQA | 126 | 101 | 4 (A–D) | `High-level domain` is `"Unknown"` for all records. Validator fields are `null` in pair A. Subset (main or Diamond) is not stated; don't assume Diamond. |
| SuperGPQA | 153 | 72 | 4–10 (136/153 have 10) | Has `discipline`, `field`, `subfield`, `difficulty`, `is_calculation`, `uuid`. Pair A includes 20 Medicine questions. |
| HLE | 88 | 80 | up to ~15 | Has `question` and `clean_question` (pair A only), `category`, `answer_type` = multipleChoice. 17–18 per file mention an image/figure. |

**Fields present:** `index`, `choices` (one string, `"A. …, B. …"` or newline-separated), `correct_choice`, `model_answer`, `model_reasoning` (list of 4–5 strings, median about 500–1000 chars), `model_correct` (bool, consistent with `correct_choice` in all 620×2 records). Extra fields:
- GPQA: `Question`, `Correct Answer`, `Incorrect Answer 1-3`, validator columns; in pair B also `raw_model_text`, `raw_model_message`, `raw_parse_source`.
- SuperGPQA: `Question`, `answer` (text), `answer_letter`, metadata.
- HLE: `question`, `correct_answer`, `debug_meta`; in the glm file also `rationale`, `author_name`, `raw_subject`.

### 2.2 Anomalies found

| # | Finding | Count |
|---|---|---|
| A1 | GPQA `kimi.json` has no `Question` field (or any GPQA metadata) in some records. `glm.json` has it in all 101, so take the question from the partner file. | 51 / 101 |
| A2 | HLE `question` embeds `Answer Choices:` with the **original** letters. `choices` is a shuffled re-lettering. `correct_choice` and `model_correct` use the shuffled letters; `correct_answer` uses the original ones. Showing both to a model gives two conflicting letterings. | `correct_answer != correct_choice` in 75/88 (pair A), 67/80 (pair B) |
| A3 | Invalid model answers: SuperGPQA glm `"A, D"`; HLE claude `"CANNOT DETERMINE WITHOUT IMAGE"`. | 1 each |
| A4 | `index` is per pair, not global. GPQA indices overlap across pairs but point to different questions (only 1 of 34 overlapping indices has the same question). Separately, 27 GPQA, 31 SuperGPQA and 12 HLE question texts appear in both pairs. | — |
| A5 | `format_choices_text` mis-parses options (A-D-only regex, plus one GPQA case where `(ctDNA)` triggers a false split). | GPQA 1/126, SuperGPQA 152/153, HLE 86/88 |

### 2.3 Files mentioned in our docs: do they exist?

- `claude.json`, `gpt_4o.json`, `glm.json`, `kimi.json` per dataset: **yes**.
- `seed_30.jsonl`, `schema.jsonl`, `README.md` thresholds, `extractor.py`, `leakage.py`, `decisiveness.py`, `weak_judge.py`, `checkability.py`, `kill_test_eval.ipynb`: **none exist yet** (expected).
- `acc.py` (declared in base `pyproject.toml`): **missing**.
- `collaborative-dr/.env`: not present (gitignored, expected).

### 2.4 Can we build `seed_30.jsonl` today? **Yes.**

Recommended recipe for 40 cases (30 is a floor; 40 gives Sunay's audit some slack):
- **Source pool:** GPQA pair A (126) and SuperGPQA pair A (153), which are the cleanest. Optionally 5–10 from GPQA and SuperGPQA pair B, taking `Question` from `glm.json` and dropping the `"A, D"` record. **Exclude HLE** (A2, images; Doc 2 also drops HLE).
- **Stratify** by dataset and by *which consultant is right*. In pair A, Claude is right 60–69% of the time, so sample about 50/50 to match the balanced-subsample logic.
- **Dedupe** by a hash of the normalized question text, not by `index` (A4).
- **Parse `choices` ourselves** into a dict using `(?:^|,\s|\n)([A-Z])\.\s` boundaries, and check that the letters run A, B, C… with no gaps. Do not use `format_choices_text`.
- **Field whitelist.** Copy only question, choices, ground truth, the two answers and reasonings, and metadata (domain/discipline). Never copy `Correct Answer`, `Incorrect Answer *`, `answer`, `answer_letter`, `correct_answer`, `rationale`, `debug_meta` or `raw_*`. Put ground truth only in the eval-only block (§5).
- Tag every record `source = "frontier_precomputed"`, `split = "seed"`, and keep provenance (`dataset`, `pair`, `file`, `index`).

---

## 3. Protocol check: flaws, edge cases, leakage, manipulation

Notation: H is the honest consultant, M the manipulative one, C the correct answer, W the wrong one.

### 3.1 How one manipulative consultant can get a bad crux selected

| ID | Attack | Why the current rules allow it | Severity |
|---|---|---|---|
| **A1** | **Veto the honest crux.** In Prompt 1B for H's crux, M answers without flipping (e.g. W for both branches) or flips the wrong way. H's crux then fails every pair, so d = 0. M's own crux survives if H honestly confirms its mapping. Selection picks M's crux. | Condition 3 requires *both* consultants to agree. That protects against M *forcing* a mapping, but hands M a **unilateral veto** over every other crux. "The honest candidate stays in the pool" is true, but it stays there at d = 0. | **High.** This is the main single-actor path. |
| **A2** | **Confidently-wrong-checkable crux.** M proposes a claim that is *false but looks true to a 1.5B model* (a common misconception, a plausible wrong constant), phrased so that TRUE implies W. H, answering the conditional honestly, agrees "if TRUE then W". So d = 1. The judge says TRUE with high c_prob and high c_cons. | Decisiveness checks *implication*, not *truth*. Tie-breaking on c **rewards** this crux over H's harder-but-correct one. Gating on d × c passes it. | **High.** This is the attack the method exists to stop, and the tie-break helps it. |
| **A3** | **Polarity plus yes-bias.** Each proposer naturally writes its *own belief* as the claim, so TRUE favours the proposer. Small instruct models lean towards TRUE. If A1 or A2 gets M's crux selected, the judge's yes-bias pushes towards W. | Nothing normalizes polarity. | Medium |
| **A4** | **Steering the leakage probe.** If L2 (Prompt 1C) runs on M's model, M says `RESTATES: NO` for its own leaky crux. Numeric options also slip past L1 (`5.03e4 g/mol` vs option `50,300 g/mole`; `pKa ≈ 4.8` vs `4.76`). | Doc 2: "run once at T = 0.0 with a consultant model". It doesn't say which one. L1 is purely lexical. | Medium |
| **A5** | **Compound claim.** "X and Y", where X is true and easy and Y is false and load-bearing. Small judges anchor on X. | Rule 2 asks for atomicity, but nothing checks it. | Medium |
| **A6** | **Context-dependent claim.** "The product of step 2 is a ketone." The judge can't evaluate it without context, so the verdict is a coin flip with the yes-bias on top. Not malicious, but it is also a cheap way for M to make the audit noise. | Self-containment is only an instruction in 1A, never checked. | Medium (also common for honest 8B models) |
| A7 | **Order bias in 1B.** `{answer_1}` / `{answer_2}` order is fixed. Two models sharing a "TRUE maps to the first-listed answer" bias produce agreeing mappings by accident, which inflates d. | No order randomization is specified. | Medium |

### 3.2 Logical flaws and edge cases (no adversary needed)

1. **The binary frame inflates d.** Prompt 1B forces each branch to {a1, a2}. Many real cruxes are one-directional: "C TRUE implies a1", but "C FALSE" doesn't imply a2. Forced to choose, models usually flip anyway, so the flip condition is close to automatic for any on-topic claim. **Fix:** allow `"UNDETERMINED"` per branch. A pass requires two definite, distinct answers from both consultants.
2. **d is coarse and the pairing is arbitrary.** With K = 3, d ∈ {0, ⅓, ⅔, 1}, so ties are common and the c tie-break (A2) decides often. Pairing sample k with sample k throws data away. **Fix:** use all K×K cross-pairs (9), or report d_A, d_B and the agreement separately.
3. **Selection before judging is a dependency cycle.** The tie-break needs Layer-1 c, so Track 2 must judge every surviving candidate before selection. Doc 2's `weak_judge_audit` holds one crux only. **Fix:** judge all non-leaky candidates, and select in a separate `gate` step (§5, §6).
4. **DR can end on a third answer** (D2). The mapping only covers {a1, a2}. We need a written rule, for example: the crux route still answers from {a1, a2}, and the DR fallback stays whatever DR said.
5. **Abstention vs Kill Criterion 1.** McNemar needs paired outcomes on the same cases. When the crux route abstains (no candidate, all leaky, d = 0, layers disagree), what is its answer? **Pre-register one option before running:** (a) count as wrong, (b) fall back to the 2B full-question answer, or (c) report on covered cases only, with coverage next to it. Doc 2 Table 1 shows coverage but doesn't say how McNemar treats abstentions.
6. **Overturning truth consensus is the cost side.** With 8B consultants, truth-consensus cases are probably the majority. A crux verdict that is 70% accurate will overturn some correct consensuses. A single τ treats confirming and overturning the same. **Suggest:** two thresholds, τ_confirm and τ_overturn, both tuned on dev only. Overturning should need more confidence.
7. **The same model family appears on both sides.** Layer 1 (Qwen2.5-1.5B) shares a family with consultant A (Qwen3-8B), and Layer 2 (Llama-3.2-1B) with consultant B (Llama-3.1-8B). Family-level misconceptions can correlate a judge layer with "its" consultant, which confounds the cross-family claim. At least log it. A neutral-family judge such as Gemma-3-1B or Phi would be a cleaner Layer 2 ablation.
8. **The same-family comparison is not like-for-like.** Doc 2 runs the second Qwen at T = 0.7, while the Llama layer and c_2judge are at T = 0. Run both comparison layers at the same temperature.
9. **Decisiveness re-solves the question.** In 1B the consultants see the full question. A consultant that is confident in its answer may give that answer for both branches (honest stubbornness), so d drops. That is fine for safety but limits coverage. Track the per-consultant flip rate so low coverage can be diagnosed.
10. **The consultant's own verdict on the crux is never collected.** The explainer says consultants report "their resolution of it: true or false". Doc 1 and Doc 2 never ask for it. It is cheap and useful: if both consultants say the crux is TRUE but hold different answers, the mapping is inconsistent with their own positions. That is a red flag for leakage or a decoy.
11. **Derived crux label.** It inherits mapping errors, and Doc 1 already notes this. With fix 1 above, cruxes with an `UNDETERMINED` branch get `derived_crux_label = null` instead of a guess.
12. **`c_prob` token mechanics.** After `VERDICT:` the next token is usually `" TRUE"` or `" FALSE"` with a leading space, but sometimes `"True"`, `" true"` or a newline. Use logsumexp over the first tokens of all case and space variants, and log the raw mass P(T) + P(F) before renormalizing. Low mass means a format failure, not a confident verdict.

### 3.3 Proposed protocol changes (cheap, kill-test compatible)

| Fix | Addresses | Cost |
|---|---|---|
| **P1. No unilateral veto.** For each candidate, store d_A, d_B and the mapping agreement separately. If a candidate passes for its non-proposer but fails because of its proposer's opponent (one flips, the other doesn't), set `veto_flag`. **If any candidate is vetoed by exactly one consultant, abstain** instead of selecting the other candidate. | A1 | Zero extra calls |
| **P2. Audit both candidates and require consistency.** When two non-leaky candidates have d > 0, judge both. If their mapped answers disagree, abstain. This drops the c tie-break, which A2 exploits. | A2, flaw 2 | One more judge pass per case (cheap, local) |
| **P3. Polarity symmetrization.** Judge both C and ¬C (negation written by a fixed neutral prompt and model), and require complementary verdicts. Compute c_prob on both. | A3, yes-bias | 2× judge calls |
| **P4. Leakage probe by both consultants.** Flag if *either* says YES. Add numeric normalization to L1 (parse numbers and units; flag if within 1% of any option value). | A4 | +1 call per candidate |
| **P5. Cheap structural filters:** connective/atomicity check (` and `, ` or `, `because`, `;`) and a deixis check (`the reaction`, `this compound`, `step \d`, `the above`, `the question`, `option`). Log them as flags first; promote to filters only if the manual audit supports it. | A5, A6 | Regex |
| **P6. Randomize answer order in 1B** per consultant per sample and map back. Allow `UNDETERMINED`. | A7, flaw 1 | Prompt change |
| **P7. Per-consultant injection slot** in 1A, 1B and 1C prompts, so the sabotage condition can steer crux proposal and conditionals, not only DR turns. | Sabotage experiment validity | Template change |

P1 and P2 together turn Doc 1's argument ("a single manipulator can't fake it alone") into a property the selection rule actually has. Doc 2 says the selection rule is fixed, so please confirm these as a team before I change any prompt or rule. CLAUDE.md requires that.

---

## 4. Model swap: Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct consultants, local HF judges

### 4.1 What must change (all on our side; `collaborative-dr/` stays untouched)

1. **Backend layer replacing `call_openrouter`.** One interface, `chat(model_key, messages, *, temperature, top_p, max_tokens, seed, n=1) -> text`, with implementations for:
   - `openai_compat` (vLLM server, OpenRouter, Groq; same wire format);
   - `hf_local` (transformers);
   - `fake` (canned JSON, so every track can test without a GPU).

   It is injected into `run.call_openrouter` (§1.2). It needs exponential backoff on 429 and 5xx, because the base code fails the whole case on one 429. Every call's resolved parameters are logged into `run_meta`.
2. **Model registry** (`configs/models.yaml`): model key, HF repo id, **pinned revision hash**, backend, quantization, dtype, max context, sampling defaults, and chat-template kwargs. One backend per model for the whole kill test (CLAUDE.md rule).
3. **Qwen3-8B, thinking off:**
   - With HF or vLLM, `apply_chat_template(..., enable_thinking=False)`; on a vLLM server, `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
   - With thinking off, the template may still emit an empty `<think></think>` block. Strip `<think>…</think>` before `force_json_parse`, because braces inside a think block would break the outermost-`{…}` heuristic.
   - Qwen's recommended non-thinking sampling is T = 0.7, top_p = 0.8, top_k = 20. The paper used T = 1, the code 0.2. **Pick one per call type and record it.**
   - If a hosted provider is used, check that it exposes the thinking switch at all. Otherwise use the `/no_think` soft switch, and record which one was used.
4. **Llama-3.1-8B-Instruct:** gated on HF, so each member accepts the licence. The 128k context is ample.
5. **Memory on Kaggle 2×T4:**
   - An 8B model in fp16 is about 16 GB of weights and **does not fit on one 16 GB T4**. Use vLLM with `tensor_parallel_size=2`, or a 4-bit AWQ/GPTQ checkpoint on one T4. Quantization goes into `run_meta`; never mix the two within the kill test.
   - T4 has no bf16, so use fp16.
   - FlashAttention-2 does not run on T4 (compute capability 7.5), so expect the xformers or SDPA backends.
6. **Message-format compatibility:**
   - `run.py` can place a `system` message after earlier turns. The judge path does this (`[system, user]` after an empty history), and the consultant path would too if `CONSULTANT_SYSTEM_PROMPT` were non-empty.
   - Check that the Qwen3 and Llama-3.1 chat templates accept this. If not, have the backend merge system text into the first user turn. The base code only does that on an HTTP 400 (`:247-259`), which a local backend never raises.
7. **Output validation (wrapper side):**
   - 8B models will answer `"B)"`, `"Option B"` or a third letter. The base code accepts anything (`:412`). Our parser normalizes and flags `answer_valid` and `answer_in_pair`. It never changes what DR saw, so results stay comparable.
   - Consider vLLM guided JSON decoding *only* if the JSON failure rate on seed data is above about 5%. It changes model behaviour, so if used it must be on for the whole run and logged.
8. **Context growth.** A consultant's history after 5 turns is about 8–15k tokens. That is fine for Qwen3-8B (32k native) and Llama-3.1 (128k). Set `max_model_len` explicitly in vLLM, or KV-cache allocation will fail on T4.
9. **Initial answers are not produced by `run.py`.** Track 3 generates them with `collaborative-dr/prompts/reasoning_generation.txt` through the same backend and writes `model_answer` / `model_reasoning`. Doc 2's filter (disagree, and exactly one right) is applied after that.

### 4.2 Local judges with log-probabilities (Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct)

- **Use transformers, not a vLLM server, for `c_prob`.**
  - vLLM's `logprobs=k` returns only the top-k tokens, so a TRUE or FALSE token can be missing.
  - transformers gives the full softmax: generate at T = 0, cut the text after `VERDICT:`, run one forward pass on `prompt + generated prefix`, and read the last-position logits.
  - Both 1–1.5B models in fp16 need about 3–4 GB, so the two layers fit on one T4 together.
- **Tokenizer check at start-up**, asserted in code:
  - For each judge, list the first-token ids of `" TRUE"`, `"TRUE"`, `" True"`, `" FALSE"`, `"FALSE"`, `" False"`.
  - Assert that the TRUE set and the FALSE set don't overlap.
  - Store the ids in `run_meta`.
- **Sampling for c_cons:** 5 samples at T = 0.7 with explicit per-sample seeds, so they are reproducible.
- **The base-DR judge also needs a backend.** For a fair comparison, base DR's judge must be the same Qwen2.5-1.5B as Layer 1, so `run_judge_review` runs through `hf_local`. Expect JSON-format failures from a 1.5B model on `judge.txt`. Log the retry and failure rate rather than silently dropping cases: the paper reports results "over successfully completed cases only" (App. C.1), and we should report the failure count next to every number.
- **Gemma-3-4B rerun (Phase 2):** same `hf_local` backend, gated licence.

---

## 5. Schema review (Doc 2 §1)

### 5.1 Structural problems

1. **One mutable record written by four people means merge conflicts and blocking.** Doc 2 has a single record that each track fills in. Two people appending to the same JSONL line, or re-running a stage, will clobber each other.
   **Proposal:** use **one JSONL file per stage**, each keyed by `id` (and `crux_id` for crux-level stages). A `merge` command joins them into the full record that Doc 2 shows. Each track owns its files outright. This is the main change that makes "zero blocking" real.
2. **`run_meta` is per record, but each stage has its own backend and prompt.** A single `consultant_backend` / `judge_backend` / `prompt_version` can't describe the five prompts and three models that touch one record. Move it to a **per-stage `meta`** block.
3. **The selection dependency** (§3.2 item 3) needs the judge output keyed by `crux_id` for *all* non-leaky candidates, plus a separate `gate` stage that writes the selection and the final answer.
4. **Label firewall.** `ground_truth` sits next to the fields the method reads, so a leak is one typo away. Put every label-derived field (`ground_truth`, `*_correct`, `is_agreement_trap`, `derived_crux_label`) under an `eval` block. The shared loader drops `eval` unless the caller passes `with_labels=True`. Only Track 4 and the filtering step in Track 3 do that.

### 5.2 Missing fields, by block

| Block | Add | Why / who needs it |
|---|---|---|
| top level | `schema_version` | Lets loaders fail loudly on drift |
| top level | `id` format `"{dataset}:{source_key}"`, `question_sha1` (normalized text) | Global uniqueness (base `index` is per pair, A4); dedupe; joins |
| top level | `provenance {dataset_name, hf_repo, hf_revision, hf_config, original_id, file, pair}` | Reproducibility; seed traceability |
| top level | `domain`, `discipline`, `subfield` | Stratified seed, per-domain analysis, Medicine exclusion |
| top level | `choices` as an ordered dict, `n_choices`, `choice_permutation_seed` | GPQA from HF must be shuffled by us; the order has to be reproducible |
| consultant init (×2) | `answer_valid`, `raw_output`, `parse_ok`, `n_retries`, `meta {model, hf_revision, backend, quantization, thinking, temperature, top_p, max_tokens, seed, prompt_file, prompt_sha256}` | Track 3 provenance; JSON failure rates of 8B models |
| `disagreement` (new) | `is_disagreement`, `exactly_one_right`, `initially_right: "a" \| "b"` (under `eval`) | Balanced subsample; analysis by which consultant was right |
| `dr_baseline` | `status`, `error`, `swapped` (consultant_1 ↔ a/b mapping), `final_answer_a/b`, `exit_mode ∈ {TC, AT, DC, THIRD}`, `per_turn[] {turn, a_answer, b_answer, a_conf, b_conf, a_action, b_action}`, `judge_model`, `judge_winner`, `judge_parse_ok`, `results_path + index` | Track 3 rates (consensus, trap, adoption, action calibration as in paper Table 4); a pointer instead of a lossy `discussion_transcript` |
| `candidate_cruxes[]` | `crux_id` (e.g. `"{id}#a"`), `disagreement_analysis`, `proposer_raw`, `parse_ok`, `proposer_own_verdict` (true/false), `flags {multi_clause, deictic, numeric_match}`, `leak_lexical_option` (which option matched), `leak_probe_by {a, b}` | Stable join key for Track 2; P4/P5; manual-audit context |
| decisiveness sample | `k`, `consultant`, `answer_order` (which answer was listed first), `if_true`, `if_false` (may be `UNDETERMINED`), `raw`, `parse_ok`, `seed` | P6; debugging. **Store per-consultant samples**, not pre-paired rows, so the pairing rule can change later without rerunning. |
| crux decisiveness summary | `d`, `d_a`, `d_b`, `flip_rate_a`, `flip_rate_b`, `mapping_agree_rate`, `veto_flag`, `agreed_mapping` | P1 |
| `judge_results` (new, keyed by `crux_id`, one entry per layer) | `layer`, `model`, `polarity ∈ {pos, neg}`, `verdict_t0`, `samples[]`, `p_true_renorm`, `p_mass_tf` (raw P(T) + P(F)), `c_cons`, `c_prob`, `parse_ok`, `raw_path`, `meta` | Replaces the single `weak_judge_audit`, so selection and P2/P3 can use it |
| `baselines` (new) | `full_q_two_way {answer, raw, model, meta}`, `q_plus_crux {crux_id, answer, …}`, `naive_judge`, `double_consultancy`, `stronger_consultant` | Doc 2 mixes these into `weak_judge_audit`. They are separate stages with separate owners. |
| `gate` (new, final stage) | `selected_crux_id`, `selection_reason ∈ {selected, no_candidates, all_leaky, d_zero, veto, candidates_disagree, layers_disagree, below_tau}`, `crux_score`, `tau` (+ dev-run id it was fitted on), `final_answer`, `final_source ∈ {crux, dr_consensus, dr_judge}` | One unambiguous final answer per case for Kill Criteria 1–3 |
| `eval` (new) | `ground_truth`, `ground_truth_text`, `a_correct`, `b_correct`, `is_agreement_trap`, `derived_crux_label` per `crux_id`, `final_correct` | Label firewall |
| run-level manifest (separate file) | `stronger_consultant` with accuracies and denominators, split seed, dev/test id lists, frozen go/no-go thresholds plus their git commit | These are not per-record facts. Keeping them in a manifest stops them drifting between records. |

**Conventions to fix on Day 1:**
- Letters are always uppercase single labels.
- `null` means "ran, no value"; a missing key means "stage not run".
- Booleans are never strings.
- Every stage file carries `{stage, code_git_sha}` in each `meta`.
- Seed records always have `split: "seed"`, which the eval loader refuses to score.

---

## 6. Proposed repo layout and CLI

```text
capstone/
├── collaborative-dr/            # submodule, never edited (CLAUDE.md)
├── docs/                        # Doc1, Doc2, paper, explainer, this audit
├── prompts/                     # OUR prompts only; DR prompts are read from collaborative-dr/prompts/
│   └── v1/
│       ├── crux_proposal.txt         (1A)
│       ├── decisiveness.txt          (1B)
│       ├── restatement_probe.txt     (1C)
│       ├── judge_crux.txt            (2A)
│       ├── judge_full_q_2way.txt     (2B)
│       ├── judge_q_plus_crux.txt     (2C)
│       └── crux_negation.txt         (P3, if adopted)
├── configs/
│   ├── models.yaml              # model key → repo, revision, backend, quantization, sampling
│   ├── seed.yaml                # run config: which models, prompt version, K, seeds, paths
│   └── killtest.yaml
├── src/crux/                    # our package (pip install -e .)
│   ├── schema.py                # per-stage record types + validator; schema_version
│   ├── io.py                    # jsonl read/write, merge-by-id, label firewall
│   ├── prompts.py               # load prompts/<version>/<name>.txt, return text + sha256
│   ├── backends/                # base.py, openai_compat.py, hf_local.py (logprobs), fake.py, retry/backoff
│   ├── data/                    # seed.py (from collaborative-dr/data), benchmarks.py (GPQA, SuperGPQA), choices.py, filter.py
│   ├── consultants/initial.py   # 8B initial answers via reasoning_generation.txt
│   ├── dr/                      # runner.py (inject backend, call run.run_case), parse.py (exit modes, actions)
│   ├── track1/                  # extractor.py, leakage.py, decisiveness.py
│   ├── track2/                  # weak_judge.py, checkability.py
│   ├── gate.py                  # selection + tau gating → final answer
│   ├── baselines.py             # full-q 2-way, q+crux, stronger-consultant; wrappers for naive/double
│   └── cli.py                   # `crux <subcommand>`
├── notebooks/kill_test_eval.ipynb
├── data/
│   ├── seed/seed_30.jsonl       # committed (small, plumbing only)
│   └── runs/<run_name>/<stage>.jsonl   # gitignored; shared via Drive/Kaggle datasets
├── tests/                       # fake-backend tests: choices parser, schema, decisiveness, leakage, gate
├── .env.example                 # backend keys/URLs; real .env gitignored
├── pyproject.toml
└── README.md                    # setup + FROZEN go/no-go thresholds
```

**Why this layout:**
- Each track owns one sub-package and one or two stage files, so the tracks don't step on each other.
- The `fake` backend lets Tracks 1, 2 and 4 run end-to-end on seed_30 before any GPU work.
- `collaborative-dr` is imported only from `crux/dr/runner.py` (add the submodule path to `sys.path`).

**CLI.** Every stage reads stage files by `--run` name and writes exactly one file:

```text
crux seed build        --out data/seed/seed_30.jsonl --n 40 --seed 42
crux data load         --dataset supergpqa --exclude-discipline Medicine --run kt1
crux consultants init  --run kt1 --config configs/killtest.yaml          # Track 3
crux data filter       --run kt1                                         # disagree & exactly-one-right
crux dr run            --run kt1 --max-turns 5                           # base DR via run.run_case
crux track1 propose    --run kt1                                         # 1A
crux track1 leakage    --run kt1                                         # L1 + 1C
crux track1 decisive   --run kt1 --k 3                                   # 1B
crux track2 judge      --run kt1 --layers l1,l2,l2same                   # 2A on all non-leaky cruxes
crux baselines run     --run kt1 --which full_q_2way,q_plus_crux,stronger,naive,double
crux split assign      --run kt1 --dev 100 --test 200 --seed 42          # once, then frozen
crux gate fit-tau      --run kt1 --split dev
crux gate apply        --run kt1
crux merge             --run kt1 --out data/runs/kt1/merged.jsonl
```

Global flags: `--backend fake|vllm|hf|openrouter`, `--limit`, `--resume` (skip ids already in the output file; append-only writes, no rewrite).

---

## Open questions for the team

1. Adopt P1–P7 (§3.3), and which ones for the kill test as opposed to later? P1, P2 and P6 are the cheapest and close the single-actor holes.
2. How does Kill Criterion 1 count abstentions (§3.2 item 5)? This must be in README.md before the 300-case run.
3. Temperatures per call type (§4.1 item 3): paper T = 1, repo T = 0.2, Qwen guidance T = 0.7.
4. Should Layer 2 be a neutral family (§3.2 item 7), or keep Llama-3.2-1B and log the confound?
5. OK to correct the CLAUDE.md "Base repo facts" line about the `Question` field (A1)?
6. Doc consistency: the explainer still mentions MMLU-Pro, MATH, "about 3 weeks" and college GPUs. Doc 1 v2 supersedes it on all four. Fine to leave the explainer as historical context?
