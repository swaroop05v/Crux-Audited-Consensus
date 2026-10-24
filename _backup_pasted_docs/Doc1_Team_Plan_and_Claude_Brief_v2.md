Crux-Audited Consensus for Weak-Judge Oversight
Project ID: PW26_BJD_09 | Guide: Dr. Bhaskarjyoti Das
Team: Swaroop Venkateshwar, Shreesh Athreya, Sruthi Vejju, Sunay Hegde
2-Week Kill-Test Plan & Architecture Validation Brief (v2, revised after Review-1)

1. Project Overview (In Simple Terms)
   Our base paper is Collaborative Disagreement Resolution (DR) by Jiang et al. (ICML 2026, arXiv 2607.01251). In DR, two strong consultant models answer a hard question independently, and the paper keeps only the questions where they disagree and one of them is right. The consultants then discuss for up to 5 turns. Each turn, a consultant either keeps its answer or adopts the other's, until they agree or run out of turns. A weaker judge model then gives the final answer. In the paper, the consultants are frontier models (GPT-4o vs. Claude Sonnet 4, and GLM-4.6 vs. Kimi K2 Thinking), the judges are GPT-4o-mini and Gemma-3-4B, and the benchmarks are GPQA, SuperGPQA and HLE-MC. DR averages 62.1% judge accuracy, against 49.2% for standard debate.
   The Flaw in Base DR (Agreement Traps)
   • Both consultant pairs in the paper reach consensus in at least 96% of cases (Figure 3).
   • When they agree, the judge (GPT-4o-mini) adopts the consensus in at least 99.5% of cases (Table 3: 363/365 and 242/242).
   • So when the consultant who was initially right gets talked into the wrong answer, both converge on it (an Agreement Trap). The paper's own accuracy decomposition (Remark 4) shows agreement traps contribute essentially nothing to accuracy, and nothing in the protocol catches them.
   • The paper assumes the consultants are honest, and says a dishonest consultant would break DR just as it breaks debate (Section 1). It also lists “what an active judge can contribute beyond acting as a passive mediator” as future work (Limitations). Our project targets both points.
   What DR Already Does, and What We Add
   DR already has a notion of a “crux”, but only as a by-product: when the consultants still disagree at the final turn, the remaining point of disagreement is isolated for the judge. That happens in at most about 4% of cases. Our change: we extract the crux at the start, while the disagreement is still live, and we audit it even when the consultants end up agreeing. The judge stops being a rubber stamp and checks one small, decisive claim.
   Our Solution: Crux-Audited Consensus
1. Propose cruxes at the disagreement stage: Right after the two consultants' initial answers disagree (before any discussion), each consultant proposes one crux: a single, self-contained TRUE/FALSE claim that the disagreement hinges on. This gives up to two candidate cruxes per question.
1. Filter out leaky cruxes: Drop any candidate that just restates an answer option (an automatic wording check plus a restatement probe; see Doc 2, Track 1). This step is needed because the decisiveness check in step 3 rewards leaky cruxes.
1. Check decisiveness (d): Each consultant is asked, independently and several times: “If this crux is TRUE, which of the two answers is correct? If it is FALSE?” A sample passes only if both consultants' answers flip with the crux and both give the same mapping. d is the share of passing samples. The surviving candidate with the highest d is selected.
1. Two-layer weak judge, commit-first, on the crux only: Layer 1 (Qwen2.5-1.5B-Instruct) sees ONLY the crux (not the question, the mapping, or the consultants' consensus), works it out, and commits to TRUE or FALSE. Layer 2 (Llama-3.2-1B-Instruct, a different model family) checks the same crux independently. If the two layers disagree, the audit abstains.
1. Measure checkability (c) without labels: Self-consistency over 5 samples, c_cons = |2p − 1|; token-probability margin, c_prob = |P(TRUE) − P(FALSE)|; and Layer-1/Layer-2 agreement. The kill test decides which of these signals to keep.
1. Gated verdict: Crux score = d × c. If the score clears a threshold τ and both judge layers agree, the final answer is the one the crux verdict maps to. This is how agreement traps get caught. Otherwise the audit abstains and DR's consensus stands.
   Our Setting vs. the Paper's
   • We run a scaled-down version: ~8B open-weight consultants (Qwen3-8B and Llama-3.1-8B-Instruct) and 1–1.5B judges. That is what we can run now, and open weights give us token probabilities. The paper's ≥96% and ≥99.5% figures were measured on frontier models, so we re-measure consensus, agreement-trap and judge-adoption rates in our own setting (Track 3) instead of quoting theirs.
   • Our two consultants are unequal in strength. A “trust the stronger consultant” baseline and a balanced subsample (as in the paper's Appendix C.2) make sure any gain isn't just picking the stronger model.
   • Once Phase-1 results are in, we rerun the key tables with the paper's weakest judge (Gemma-3-4B) to connect our numbers to theirs.
1. Answering Dr. Das's Review-1 Questions
   How exactly is the crux found?
   Each consultant reads both initial reasoning traces and writes one claim that (a) the two traces disagree on, (b) would settle which answer is right, and (c) can be checked by someone who never saw the question. Prompt 1A in Doc 2 sets out the rules. We never take a model's word that its claim is the crux: a candidate only counts if it passes the leakage filter and the decisiveness test, and both of those are checks we run on the models' behaviour.
   How does the model decide it, and who picks the final crux?
   The consultants propose; the protocol decides. Selection is mechanical: drop leaky candidates, keep the one with the highest d, break ties on c. No model gets to choose which crux is audited.
   What if a consultant manipulates the crux?
   • Decoy crux (true but not load-bearing): the honest consultant's conditional answers won't flip with it, so the pair fails and d ≈ 0. The decoy is rejected, and the honest consultant's own candidate is still in the pool.
   • False mapping (“if TRUE, the wrong answer”): decisiveness requires both consultants to give the same mapping, so one consultant cannot fake it alone.
   • Leaky crux (restates the answer so the judge just echoes it): caught by the leakage filter, and measured in the manual audit.
   • Both consultants colluding: out of scope. This is the same assumption the base paper makes, and we state it as a limitation.
   So far this is an argument, not evidence. Planned after the kill test: a sabotage condition where one consultant is prompted to steer toward the wrong answer. We measure how often its decoy cruxes survive selection, and compare against base DR's agreement-trap rate under the same sabotage.
   The two-layer weak judge
   A single weak judge can simply be wrong, and with one judge every mistake goes straight into the final verdict. Layer 2 is an independent judge from a different model family that checks the same crux; if the layers disagree, the audit abstains. Two copies of the same model tend to make the same mistakes, so the kill test also compares a same-family second layer (another Qwen2.5-1.5B run) against the cross-family one. If diversity beats duplication, that is a result in itself.
1. Goal for This Sprint (2-Week Phase-1 Kill Test)
   Before months of full-scale experiments or DPO training, we set up the codebase on top of DR's public repo (collaborative-dr-main) and run the kill test on ~300 disagreement cases (Consultant A and B initially disagree and exactly one is right). Two weeks instead of one leaves room for coursework and reruns; the order of work is unchanged.
   • Kill Criterion 1 (Go/No-Go, end-to-end): On these cases, does the crux route (Layer-1 verdict on the crux, then the agreed mapping, then the answer) pick the correct answer more often than the same judge answering the full question as a two-way choice? This needs no crux labels: the route's answer is compared directly with the ground truth. Paired McNemar test.
   • Kill Criterion 2 (Metric check): Which label-free signal (c_cons, c_prob, layer agreement, d × c) best predicts when the crux route is right? Is the judge ever confidently wrong?
   • Kill Criterion 3 (Sanity): Does the gain survive (a) removing leaky cruxes, (b) the balanced subsample, and (c) comparison with “trust the stronger consultant”?
   We agree the go/no-go thresholds for all three criteria as a team before running the 300 cases, and write them in the repo README, so the results can't shape the decision.
1. Key Operational Questions Answered
   Q1: We don't have college GPU access yet. How do we run 8B and 1.5B models?
   We don't need college GPUs for the kill test.
1. Day-1 plumbing data: According to the repo, data/ contains pre-computed initial responses from the paper's frontier consultants (claude.json, gpt_4o.json, glm.json, kimi.json) on GPQA, SuperGPQA and HLE. Step 0 confirms exactly what's there. Swaroop extracts 30–50 disagreement cases into seed_30.jsonl so every track can build and test from Day 1. These are frontier-model outputs, so they are for plumbing only and never enter kill-test numbers (source = "frontier_precomputed").
1. 8B consultants: Kaggle's free 2×T4 GPUs and free API tiers (Groq, OpenRouter, Cerebras, SambaNova). Rule: one backend per model for the whole kill test. Mixing providers for the same model changes quantization and sampling and makes results inconsistent. Check that the provider hosts the exact model, and never substitute a different one silently. Every record stores the provider, model revision, quantization and temperature.
1. 1–1.5B judges: Run locally with Hugging Face transformers or vLLM (Colab/Kaggle T4, about 3 GB VRAM in FP16), not through an API, because c_prob needs exact token log-probabilities. Ollama is fine for quick prompt testing only.
   Call budget: Assuming roughly 15% of questions give a usable disagreement (to be measured): ~2,000 questions × 2 consultants (≈4,000 calls) to find 300 cases, plus up to 5 DR turns × 2 consultants × 300 (≈3,000), plus crux proposals (600), plus decisiveness samples (300 × 2 candidates × 2 consultants × 3 ≈ 3,600). That is about 11,000 consultant calls. Check each free tier's current daily and per-minute limits against this before Day 1.
   Q2: How do all 4 members work in parallel without waiting for each other?
   We decouple the 4 stages with a shared JSONL data contract (Doc 2, Section 1) locked on Day 1. Swaroop builds seed_30.jsonl from the pre-computed files on Day 1. Everyone develops on seed_30.jsonl during Week 1 while Sruthi generates the 300 8B disagreement cases in the background. Week 2 runs the full batch through the pipeline.
   Q3: Do we need hand-labelled ground truth for every crux?
   No. The go/no-go (Kill Criterion 1) is end-to-end and only needs each question's ground-truth answer. For diagnosis we derive a crux label: if a crux is decisive, its true value is whichever branch leads to the correct answer. That label inherits any mapping mistakes the consultants make, so Sunay's 50-case manual audit checks it, and we report how often derived and manual labels agree.
1. Work Division Across the 4 Team Members
   Member Assigned Track Core Deliverables Tools / Compute
   Swaroop
   (Has Claude Pro + Claude Code) Track 1: Lead Architect, Crux Proposal, Leakage Filter & Decisiveness (d) 1. Run the Step-0 audit with Claude Code & scaffold the repo.
1. Create seed_30.jsonl from data/ (plumbing only).
1. Build extractor.py (each consultant proposes a self-contained TRUE/FALSE crux).
1. Build leakage.py (wording check + restatement probe).
1. Build decisiveness.py (K = 3 sampled If-TRUE / If-FALSE pairs per crux, compute d, select the crux). Claude Code, Claude Pro, the pinned consultant backend
   Shreesh Track 2: Two-Layer Weak Judge & Checkability (c) 1. Set up Qwen2.5-1.5B-Instruct (Layer 1) & Llama-3.2-1B-Instruct (Layer 2) locally with log-probs.
1. Build weak_judge.py (crux-only commit-first, full-question 2-way baseline, question+crux ablation).
1. Build checkability.py (c_cons, c_prob, layer agreement) + same-family vs. cross-family Layer-2 comparison. Colab / Kaggle T4 (transformers or vLLM)
   Sruthi Track 3: Benchmarks, 8B Consultant Runs & Base DR Replication 1. Load GPQA Diamond + SuperGPQA (multiple-choice).
1. Pin backends; run initial answers for Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct; filter ~300 cases where they disagree and exactly one is right.
1. Run base DR (run.py) and baselines; re-measure consensus, agreement-trap and adoption rates. Kaggle 2×T4, one API provider per model
   Sunay Track 4: Kill-Test Evaluation & Manual Crux Audit 1. Build kill_test_eval.ipynb (Kill Criteria 1–3 tables, McNemar tests, calibration/AUROC of c and d × c, balanced subsample).
1. Lead the 50-sample manual crux audit (self-containment, leakage, simpler than the question, derived-label check). Jupyter / Colab, Google Sheets

1. 2-Week Parallel Schedule
   Days Swaroop (Track 1) Shreesh (Track 2) Sruthi (Track 3) Sunay (Track 4)
   Days 1–2 Step-0 audit with Claude Code; push GitHub skeleton, schema and seed_30.jsonl. Set up both judges locally; confirm 5-sample runs and TRUE/FALSE log-prob extraction. Load GPQA Diamond + SuperGPQA; pin backends; smoke-test both 8B models. kill_test_eval.ipynb skeleton on seed_30; agree go/no-go thresholds with the team.
   Days 3–4 Build extractor.py; tune Prompt 1A until cruxes are self-contained and atomic on seed_30. Build weak_judge.py (crux-only commit-first, full-question 2-way, question+crux). Initial answers for both consultants; start filtering disagreements and track the yield rate. Manual-audit sheet and rubric; first audit pass on seed_30 cruxes.
   Days 5–7 Build leakage.py and decisiveness.py; end-to-end run on seed_30. Build checkability.py; Layer-2 judge and same-family comparison. Reach ~300 disagreement cases; start base DR (run.py) on the first 100. Test the full evaluation pipeline end-to-end on seed_30 outputs.
   Days 8–10 Crux proposal, leakage filter and decisiveness on all 300 cases. Both judge layers, checkability and ablations on all 300. Finish base DR on all 300; run naive_judge.py, double_consultancy.py and the stronger-consultant baseline; fix the dev/test split. 50-sample manual audit (12–13 per member); derived-vs-manual label agreement.
   Days 11–12 Gated verdict (d × c ≥ τ and layers agree → crux verdict, else DR consensus); choose τ on the 100-case dev split only. Commit-first vs. non-commit-first check on checkable cruxes. Build the balanced subsample; report consensus, trap and adoption rates for our setting. Kill Criteria 1–3 tables on the 200-case test split, McNemar tests, calibration and ROC curves.
   Days 13–14 Clean repo, README and 2-page Gate-1 summary for Dr. Bhaskarjyoti Das. Document judge and checkability results. Document the 8B DR replication and agreement-trap rates. Finalize the notebook and audit summary; go/no-go call against the pre-agreed thresholds.

1. STEP 0: Architectural Validation Prompt for Swaroop's Claude Code
   Before writing any implementation code, Swaroop unzips collaborative-dr-main.zip, puts 'Our Capstone Idea, Explained Simply.pdf' and a PDF export of Doc 2 (v2) in the folder, launches Claude Code in the terminal, and pastes this prompt:
   We are a 4-member capstone team building "Crux-Audited Consensus for Weak-Judge Oversight" on top of this repository (`collaborative-dr-main`), the code for Jiang et al., "Collaborative Disagreement Resolution for Scalable Oversight" (ICML 2026).

Before writing any code, read:

1. All files in this repository (`run.py`, `run_ablation.py`, `prompts/`, `baselines/`, and the contents of `data/`).
2. `Our Capstone Idea, Explained Simply.pdf` and the Doc 2 (v2) PDF with our prompts and data contract.

Do NOT write implementation code yet. First, perform a complete architectural audit and answer these 6 questions:

1. Base repo: How does `run.py` load initial consultant responses, run the multi-turn discussion, isolate a crux (if at all), and call the judge? Which exact functions/lines in `run.py` and `prompts/` do we hook into? Report anything that differs from what the paper describes.
2. Data on disk: List what is actually in `data/` (files, datasets, models, keys/fields). Do not assume: if a file we mention does not exist, say so. Can we extract 30-50 disagreement cases from it today as `seed_30.jsonl` for plumbing?
3. Protocol check: Each consultant proposes a self-contained TRUE/FALSE crux at the initial-disagreement stage; we drop leaky cruxes; we compute decisiveness d from K sampled conditional answers ("If C is TRUE -> answer?", "If C is FALSE -> answer?") from both consultants; then only the crux goes to a two-layer weak judge (commit-first). Find logical flaws, edge cases and leakage risks, especially ways a single manipulative consultant could get a bad crux selected.
4. Model swap: What must change to run the consultants as Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct, and the judges as local Hugging Face models that return token log-probabilities?
5. Schema: Review our shared JSONL data contract. Suggest missing fields so all 4 tracks can work with zero blocking dependencies.
6. Repo layout: Propose the cleanest folder/file layout and CLI commands so our code stays modular and integrates cleanly with `collaborative-dr-main`.

7. What Changed from v1
   • Base paper described correctly (frontier consultants; GPT-4o-mini and Gemma-3-4B judges). The 96% and 99.5% figures are now attributed to that setting, and our 8B / 1.5B setup is labelled as scaled-down, with its rates re-measured.
   • New Section 2 answers Dr. Das's Review-1 questions: how the crux is found, who decides, manipulation, and the two-layer judge.
   • Leakage filter added before crux selection, because decisiveness rewards cruxes that restate the answer.
   • Kill Criterion 1 is now end-to-end, so it no longer needs 300 hand-labelled cruxes; crux labels are derived and spot-checked.
   • Datasets: GPQA Diamond + SuperGPQA, matching the paper. MATH-500 and MMLU-Pro are dropped from the kill test.
   • New baselines: trust-the-stronger-consultant, and a balanced subsample. τ is chosen on a dev split and reported on a separate test split.
   • One backend per model; judges run locally for log-probs; Consultant A fixed as Qwen3-8B with thinking off.
   • Sprint extended from 1 to 2 weeks with the same order of work; data contract extended (see Doc 2).
