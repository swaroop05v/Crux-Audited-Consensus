# Track 1 on seed_30: first real runs (seed30_v1, seed30_v2)

**Date:** 30 Oct 2026. **Purpose:** evidence for the team's decisions on (1) the decisiveness definition and
(2) the L2 leakage rule. No protocol change has been made; the gate still uses two-way `d`.

## Runs

| | seed30_v1 | seed30_v2 |
|---|---|---|
| Commit | `a3fa532` | `dd649a9` |
| Proposal prompt (1A) | v2 | **v3** (asks for a two-way crux) |
| Decisiveness prompt (1B) | v2 (P6: random order, UNDETERMINED) | v2 |
| Decisiveness `max_tokens` | 768 | 1024 |
| Parser | before the LaTeX fix | with it |
| Setup | Kaggle 2×T4, vLLM 0.32.0, Qwen3-8B-AWQ (A) + Llama-3.1-8B AWQ (B), K = 3, 16 workers | same |
| Track 1 time | 11.3 min | 11.5 min |

Both runs: 80 cruxes (40 cases × 2 proposers), 0 call errors, smoke test passed (no `<think>`; replies identical
across runs for the same seeds).

**Caveat for every number below:** seed_30 uses the paper's GPT-4o / Claude answers and reasoning. Our 8B consultants
propose and judge cruxes about *someone else's* reasoning, defending answers that may not be their own. Expect
different behaviour once Track 3 produces the 8B consultants' own initial answers.

## 1. Decisiveness

Most cruxes are **one-way**: "if C is TRUE the answer is X; if C is FALSE, undetermined". P6's UNDETERMINED option
exposes this; the audit (§3.2 item 1) predicted it.

What each consultant answered (480 samples per run):

| | decisive (both branches, different answers) | TRUE-branch only | FALSE-branch only | both UNDETERMINED | same answer both ways | parse error | truncated |
|---|---|---|---|---|---|---|---|
| v1 Qwen (A) | 72 | 105 | 7 | 39 | 11 | 6 | 0 |
| v1 Llama (B) | 87 | 92 | 9 | 9 | 13 | 20 | 10 |
| v2 Qwen (A) | 77 | 87 | 13 | 49 | 13 | 1 | 0 |
| v2 Llama (B) | 106 | 95 | 10 | 4 | 8 | 11 | 6 |

Per crux (80 per run):

| | v1 | v2 |
|---|---|---|
| two-way `d`: mean / > 0 / ≥ 2/3 | 0.08 / 23 / 2 | **0.14 / 34 / 7** |
| `d_true`: mean / > 0 / ≥ 2/3 | 0.41 / 62 / 30 | 0.45 / 57 / 36 |
| `d_false`: mean / > 0 / ≥ 2/3 | 0.10 / 26 / 3 | 0.17 / 39 / 8 |
| either branch > 0 | 65 | 64 |
| comparison-style wording ("greater than", "rather than", ...) | 9 | 19 |

- **v3 helps, modestly.** The FALSE branch firms up and two-way `d` > 0 rises from 23 to 34 cruxes. Only 19 of 80
  v3 cruxes actually use comparison wording, so the consultants follow the two-way instruction only partly.
- **The LaTeX parser fix is not what moved d.** Re-parsing v1 with it changes mean `d` only from 0.082 to 0.085.
- **Parse errors and truncation fell below the handoff §9.2 limits:** 5.4% → 2.5%, and 2.1% → 1.2%.

## 2. Coverage proxy (per case, before judges, P2 and τ)

Cases (of 40) with at least one **non-leaky** crux meeting each condition. The README's minimum coverage is ≥ 30% of
cases on the gated crux route, after judge agreement and τ, so these proxies are upper bounds:

| Leakage rule | Run | cases with a non-leaky crux | two-way `d` > 0 | `d_true` > 0 | `d_true` ≥ 2/3 |
|---|---|---|---|---|---|
| current: L1, or **either** consultant's L2 says YES | v1 | 25 | 10 (25%) | 21 | 8 |
| | v2 | 25 | 11 (28%) | 18 | 10 |
| alternative: L1, or **both** say YES | v1 | 34 | 15 (38%) | 28 | 12 |
| | v2 | 36 | **19 (48%)** | 27 | 17 |

Under the current rules (two-way `d`, either-YES L2), the upper bound is already below the 30% minimum coverage.
Both decisions move coverage a lot.

## 3. L2 leakage rule

- **Llama says "restates the answer" on about half of all cruxes; Qwen on under 10%.** L2 verdicts (A says, B says):
  v1 {NO/NO 39, NO/YES 33, YES/YES 6}; v2 {NO/NO 40, NO/YES 31, YES/YES 8}. Every Qwen YES is also a Llama YES.
- **Under "either says YES" (P4), Llama alone decides:** 45 of 80 leaky in both runs. "Both say YES" or "Qwen only"
  gives about 23–25.
- **Llama's YES calls are mixed, not pure noise.** In v1 the L2-only flags were mostly plain intermediate steps
  (e.g. "The matrix P has eigenvalues 0, 1, and 1"). In v2 some look justified: "The SNP rs113993960 is positioned at
  the center of the sequences provided in **option D**" (a Prompt 1A rule-4 violation), and a meso-compound claim that
  nearly states the answer.
- **Naming an option outright is rare** (1 crux in v1, 3 in v2) and mostly already flagged.

Settling the L2 rule needs Sunay's manual audit on these cruxes: is_leaky against human judgement, for Llama-only YES
cases in particular.

## 4. P1 vetoes

| | v1 | v2 |
|---|---|---|
| vetoes | 34 (by proposer 20, by opponent 14) | 29 (by proposer 10, **by opponent 19**) |
| what the vetoing consultant answered, mostly | TRUE-branch only (52 of 102 samples), parse errors (13) | TRUE-branch only (36 of 87), both UNDETERMINED (16) |

Vetoes come from indecision (one-way or undetermined answers), not from a consultant contradicting a clear mapping.
P1 as a hard abstain rule would remove coverage without a protective benefit on this data. Keep flag mode
(decisions D1), and re-check on 8B-own-answer data, since v2's shift toward opponent vetoes has the A1 shape.

## Options for the team

1. **Decisiveness:** keep two-way `d` (v3 prompt), switch to branch scoring (gate uses the branch the judge's verdict
   lands on; abstain on an UNDETERMINED branch), or both. Branch scoring needs a change to `gate.py` (Shreesh) and
   D10.
   - *Caveat:* most cruxes are TRUE-sided, so under branch scoring the route answers mainly when the judge says TRUE,
     and small judges lean TRUE (audit attack A3). Its safety then depends on:
     - P2: when both candidates are TRUE-sided and both judged TRUE, abstain unless their mapped answers agree;
     - each judge's TRUE rate, which must be measured once Track 2 runs on these cruxes.

     P3 (judging both C and NOT-C) may need to come back.
2. **L2 rule:** keep "either YES", switch to "both YES" or "Qwen only", or decide after Sunay's audit of the Llama-only
   flags.
   - *Caveat:* "both YES" and "Qwen only" reopen audit attack A4.
     - Under "both YES", a manipulative consultant can answer NO on its own leaky crux and clear it.
     - Under "Qwen only", consultant A decides leakage for every crux, including its own.
   - *Middle option:* leaky = L1 OR both YES, while logging every Llama-only YES case. The final rule is then chosen
     against Sunay's audit of those logged cases.
3. **P1:** stay in flag mode.

Reproduce: `crux track1 report --run seed30_v1/seed30_v1` and `--run seed30_v2/seed30_v2` (run folders are gitignored;
the zips are with Swaroop).
