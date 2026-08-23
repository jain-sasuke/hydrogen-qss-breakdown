# Master Plan v2 — Thesis to Defense

**Replanned 22 August 2026**, against an audit of every project `.md` file and the
repo state reconstructed from `git log`, `stat`, and the summary files.

**Defense:** ~21 September 2026 · **Submission deadline: UNKNOWN — blocking, confirm first**
**Student:** Nikhil Jain, M.Tech ChemE, IIT Kanpur · **Supervisor:** Prof. R. G. S. Pala

This supersedes `master_plan.md` (14 Jul). That plan assumed continuous work at a
weekly cadence; the filesystem shows three working sessions in five weeks. This
one is planned against demonstrated throughput.

---

## PART 0 — Document audit (done 22 Aug)

Every `.md` in `outputs/` and in project knowledge was read. Result:

### 0.1 Authoritative — keep, one copy each

| File | Status | Note |
|---|---|---|
| `thesis_grade_results_backlog.md` (27 KB) | **authoritative** | Contains A5, A6, A7, C8–C11. The 11 KB copy in project knowledge is a strict subset — **delete it** |
| `derivation_01_rate_matrix.md` | closed ✅ | Conservation 2.59e-11; 8s ℓ-mixing argmax checked against n⁴ |
| `derivation_02_detailed_balance.md` | **Steps 4–6 open** | Stub, brutal test, write-up never done. Also an open action: confirm whether the 0.43% or 0.05% Klein–Rosseland figure comes from `qc_ccc.py` |
| `derivation_03_qss_steady_state.md` | analytic ✅, numeric open | The `n^ss` grid solve was never run |
| `derivation_04_two_timescales.md` | closed ✅ | The most load-bearing note. §9.3 eigenvector table is the cleanest evidence in the project |
| `derivation_04b_ell_mixing.md` | closed ✅ | Bug found, quantified, bounded |
| `derivation_04c_non_normality.md` | derived ✅, grid run open | µ(L) at one point only |
| `derivation_05_qss_breakdown.md` | Session B only | C–F never done; §5 numbers used a **stand-in source vector** |
| `SETUP.md` | authoritative for method | **§8.1 findings are unverified** — see 0.3 |

### 0.2 Superseded — delete from project knowledge

- The pasted "PROJECT CONTEXT" block: inverted M convention (M = τ_relax/τ_QSS),
  49 states, 696 grid points, Gate E as "find M ~ 1". Under its convention
  M = 9982 reads as *QSS fails* — the retracted conclusion.
- `session_status_and_forward_plan.md` (14 Jul): overlaps `master_plan.md`,
  quotes M = 9963 where the backlog says 9982, and dates the L_grid regeneration
  to 14 Jul when the filesystem says **21 Jul 20:44**.
- `thesis_grade_results_backlog.md` (11 KB copy).
- `CHAPTER5_CORRECTIONS_REPORT.md` — keep as a historical record of *what was
  corrected*, but its numbers are pre-21-July.

### 0.3 The claim with no provenance

`SETUP.md` §8.1 / backlog A7 report an all-states vs excited-only error table
(1.697/0.0555, 0.205/0.00724, 0.0276/0.00104, J = 1.780 / 0.333) from an LSODA
integration convergence-checked over three tolerances.

**No script in the repo produces these numbers.** `verify_ramp_vs_step.py` writes
seven columns — Te, ne, drive, τ_drive, τ_relax, De, suppression — with no
integration, no J, no ε, and `De` bit-identical to `suppression` in every row.

By method 6 (provenance on every number), A7 cannot enter the thesis until a
script that produces it exists and is understood. The 1%-vs-2% tolerance
inconsistency inside A7 is a *symptom*, not the disease.

---

## PART 1 — Corrections to the record

Recorded in place per method 7. Three are corrections to Claude's own claims made
during this audit; leaving them silent would let the same wrong path be re-derived.

| # | Claim | Correction | Source of truth |
|---|---|---|---|
| R1 | *"C11/A7 mislabel τ: τ_relax at Te = 1 eV, ne = 10¹² is 26.5 ns, not 38.9 ns"* | **Wrong. The notes were right.** τ_relax there is 38.879 ns | `timescale_verification.csv` (3.888e-08) and `derivation_04` §9.3, which gives 38.879 ns under **both** framings A and B |
| R2 | *"C7 is obsolete — no point has τ_QSS > 1 s"* | **Wrong. C7 is exactly right**: 19 points, max 67.2 s at Te = 1 eV, ne = 10¹² | `timescale_verification.csv` |
| R3 | *"There is a breakdown corner at Te = 1 eV, ne = 10¹² with M ≈ 1.5"* | **Wrong.** M = 1.73×10⁹ there — the *strongest* separation on the grid | same |
| R4 | Grid M range `1..101197289` (`qss_analysis_summary`) | Contaminated by the `eigs < -1.0` filter. Clean range **86.8 to 1.73×10⁹**, minimum at Te = 10 eV, ne = 10¹⁵ | same |

**R1–R3 share one cause:** numbers were read out of `qss_analysis_summary.txt`
without checking them against `timescale_verification.csv`. Both files sit in
`validation/` with no provenance header. **Fix: every output file gets a header
line — script, date, source matrix SHA.**

---

## PART 2 — The filter, and a sharpened prediction

`qss_analysis.py` filters `eigs < -1.0`, discarding every eigenvalue with
|λ| < 1 s⁻¹, i.e. every τ > 1 s. It does not fail loudly. It silently promotes
the next eigenvalue up the ladder.

**Prediction, recorded before the run (method 1):** at the 19 affected points the
filter shifts the *whole* ladder by one, so **both** timescales are wrong:

- reported τ_QSS = |1/λ₁| = the true τ_relax
- reported τ_relax = |1/λ₂| = the second fast mode

At Te = 1 eV, ne = 10¹²: clean λ₀ → 67.23 s, λ₁ → 38.88 ns. Predicted filtered
output: τ_QSS = 3.887944677886894e-08 (bit-identical to true τ_relax),
τ_relax = 2.65e-08. `qss_analysis_summary` reports exactly those two numbers.

**What would refute this:** if the filtered τ_relax is *also* 3.888e-08, the
filter only affects τ_QSS and `tau_relax_grid.npy` is clean.

**Consequence if confirmed:** `tau_relax_grid.npy` is contaminated at 19 points
too, not just `tau_QSS_grid.npy` and `M_grid.npy`.

**Unexplained anomaly — run this even if everything else is skipped.**
`qss_analysis` reports max M = 1.01197289×10⁸. That value appears nowhere in
`timescale_verification.csv`; the nearest entry is 1.0589×10⁸. Both files
postdate the 21 Jul matrix rebuild, so on unfiltered points they should agree to
machine precision. **If they don't, the filter is not the only difference between
these two scripts**, and that matters more than the filter does.

---

## PART 3 — Phase 0: repair the instrument (days 1–5)

Nothing is written, derived, or titled until the measuring apparatus is trusted.
Run via the `verifier` subagent. Report only — no repairs, no substitutions, no
tolerance adjustments. Missing data → stop.

| # | Task | Deliverable |
|---|---|---|
| **V0** | Commit `compute_lmix.py` (modified) + four untracked `verify_*.py`. Report hash. Also `git check-ignore -v` and `shasum -a 256` on `L_grid.npy` — `data/` is probably gitignored, so put the matrix hash in the commit message | known state |
| **V1** | Two files named `S_grid.npy` (`cr_matrix/` = recombination source; `sensitivity/` = S-criterion grid). Report shape, dtype, range, writing script. Don't rename yet | collision resolved |
| **V2** | Audit `verify_ramp_vs_step.py` line by line. Then grep repo-wide for `solve_ivp`, `odeint`, `LSODA`, `expm`. **Find the script that produced A7's numbers, or establish that none does** | A7 provenance settled |
| **V3** | Slow eigenvector v₀ at the benchmark point: ground component, all 42 excited components at full precision, ‖v₀,exc‖/\|v₀,gnd\|, and ‖v₀,exc‖/‖n^ss_exc‖ after scaling v₀ to n^ss_ground. **n_ion is undefined in the pipeline — set n_ion = 1 and say so; every requested quantity is a ratio and is invariant under rescaling b.** Report Im(λ₀) and the sign convention | the τ question, half of it |
| **V4** | The `eigs < -1.0` filter: line, expression, affected points with coordinates, filtered vs unconditional values, affected outputs | Part 2 prediction tested |
| **V5** | Cross-file consistency: `timescale_verification.csv` vs `M_grid.npy` + summary, point by point. Locate max M = 1.01197289e8 | anomaly resolved |
| **V6** | Figure staleness table: figure, generating script, figure mtime, data mtime, stale y/n. Flag everything older than 2026-07-21 20:44 | regeneration list |
| **V7** | `chapter4.tex` number audit: every τ_relax, τ_QSS, M with line number and sentence. Known: 876, 908, 993, 1493, 1556 | correction list |

**Then, and only then:** re-run `qss_analysis.py` with the filter repaired and
regenerate `M_grid`, `tau_QSS_grid`, `tau_relax_grid`, `breakdown_map`,
`epsilon_traces`, and the Ch. 5 maps. Re-run the validation gates — Gate D's 0%
pass is dated **28 March** and may not survive the corrected matrix.

**Skeptic pass** on the restated claim: *"M ≥ 86.8 at every one of 400 grid
points, so no timescale-separation breakdown regime exists in the ITER divertor
operating range."* This is now load-bearing for the title.

**ChatGPT, cold:** raw `timescale_verification.csv` + `qss_analysis_summary.txt`,
no interpretation from me. Ask what would have to be true for both files to be
measuring the same quantity.

---

## PART 4 — Phase 1: the one open quantity (days 6–12)

### Q8 — Which τ enters De?

This is the only genuinely open physics question left, and the entire breakdown
result turns on it. Three of your own documents disagree:

| Source | Law | De at ITER ref | ε̄ at ELM (100 µs) |
|---|---|---|---|
| `chapter4.tex` line 953 | ε̄ = ε_res·(τ_QSS/τ_d)(1 − e^{−τ_d/τ_QSS}) | 0.227 | **0.141 — breakdown** |
| `qss_analysis.py` | same | 0.227 | 0.1415 (reproduces) |
| `SETUP.md` §8.2 | De = τ_relax/τ_drive | 2.3×10⁻⁵ | 1.4×10⁻⁵ — invisible |

Four orders of magnitude apart. Follow the six-step rhythm; **you derive, I ask
and wait.**

1. **Physics.** ε is built on ratios r_p = n_p/n_1S. Excited states re-equilibrate
   to the *instantaneous* ground state in τ_relax — argues ns. But the QSS target
   r_QSS itself moves as the ion balance evolves — argues µs. Which object does
   the error actually track?
2. **Hand derivation.** From δ̇ = −δ/τ_r + J/τ_d, derive the suppression law and
   check both limits (τ_d → 0 and τ_d → ∞).
3. **Worked example.** Both candidate τ at the benchmark point, carried to a number.
4. **Brutal test.** Units. Signs. Does ε̄ → ε_res as τ_d → 0? Does it → 0 as
   τ_d → ∞? Does the answer survive using the L² norm instead of max_p?
5. **Code.** `qss_analysis.py`, against the derivation.
6. **Note** → `derivation_06_the_drive_timescale.md`.

**Predict before computing** (write it down before V3 returns): is
‖v₀,excited‖/\|v₀,ground\| order 10⁻³ or negligible? `derivation_04` §9.3 records
λ₀'s excited components as "all others < 10⁻³" — four decimals, so anything up to
~10⁻³ is invisible there. The excited manifold is 73× smaller than the ground
population, so a 10⁻³ leakage is an O(10%) contamination *of an excited-only
norm*, decaying on τ_QSS. **What would you conclude from each answer?**

**Empirical cross-check, independent of the derivation:** load
`epsilon_traces.npz` (regenerated, post-repair) and find where ε(t) falls to 1/e
of ε_step at the benchmark point. Near 2 ns → τ_relax governs. Near 20 µs → τ_QSS
governs. Predict which first.

### Then, in order of value

- **Q7 (S criterion).** Not "not started" — done in March and invalidated by the
  July matrix. Derive ε_step ≈ S·δT_e with S = ΔE/kT_e², then re-run. The
  `physics_tests` report already shows E_eff(2P) → 10.2 eV = ΔE(2p−1s) and
  E_eff(n15) → 13.55 eV ≈ I_H, which is exactly this criterion confirming itself
  — but on the old matrix.
- **Q6 (Hα).** The −46%/−48% must leave Ch. 5 regardless. This is a deletion plus
  a regeneration, not a new result.
- **Derivation 02 Steps 4–6.** Half a session; closes the oldest open note and
  settles which script produces the Klein–Rosseland number.

---

## PART 5 — Phase 2: the title decision gate (day ~13)

**Do not choose the title before Q8 closes.** The title is a claim, and right now
the claim has two incompatible values.

### The decision rule

| If Q8 gives | Grid evidence | Then the thesis is | Candidate title |
|---|---|---|---|
| τ_QSS governs | ε̄ = 14% at ELM timescales; breakdown map has structure | a **breakdown study**, as titled | *Quantifying Quasi-Steady-State Breakdown in Hydrogen Divertor Plasmas: A Time-Dependent Collisional-Radiative Study* |
| τ_relax governs | ε < 10⁻⁴ everywhere; M ≥ 86.8 everywhere | a **validity map with a derived criterion** | *When Does Quasi-Steady-State Hold? A Validity Criterion for Collisional-Radiative Models of Divertor Hydrogen Plasmas* |

The second is not the weaker thesis. Fujimoto and Capitelli *assert* that QSS
holds; a quantitative map of *where*, *by how much*, and *what would break it*,
with a criterion other groups can apply to their own CR models, is more
defensible and harder for an examiner to attack.

**Either way, three results are untouched and carry the thesis on their own:**
the one-gap spectral structure (A2), the boundary-level descent (A1), the
moving-bottleneck density scaling (A5), and the ℓ-mixing bug found, quantified,
and bounded (A4).

### The Pala conversation — highest-value hour available

Book it in Phase 0, not after. Bring one page:

1. The ℓ-mixing bug is found, fixed, and bounded (<0.85% on τ_relax anywhere).
2. The matrix was regenerated 21 July; the timescale results are clean.
3. A filter bug corrupted 19 of 400 grid points; it is being repaired.
4. Four months of April analysis (Mori-Zwanzig, S-criterion, Balmer sweeps,
   trapping) predates the fix and is being scoped **into the paper, not the thesis**.
5. One question decides the title, and it will be settled within the week.

Frame it as a status report with a decision attached, not as a crisis.

---

## PART 6 — Phase 3: writing (days 13–22)

### Style — non-negotiable, from the supervisor

**A first-year graduate student must follow it top to bottom.** Per chapter:

1. **The physical question** — why anyone should care, before any formalism.
2. **Every symbol defined at first use, with units.**
3. **Intuition** — ChemE analogies where they are genuine correspondences, not
   decoration: the rate matrix as a reaction network, **b** as a CSTR feed
   stream, QSS as Bodenstein applied to the whole excited manifold at once,
   the moving target as a chase problem.
4. **Mathematical rigour** — the derivation, complete, built from first principles.
5. **The result, with its caveats attached** — not in a footnote.
6. **Why it matters and what comes next** — the bridge to the following chapter.

**The test:** hand a section to someone with your starting background. If they
can't follow it unaided, intuition is missing — not detail.

**Leverage: Chapters 2 and 3 are substantially assembly, not composition.** The
derivation notes are already written in this shape.

### Chapter map

| Ch | Content | Fed by | Story spine |
|---|---|---|---|
| 1 | Introduction | everything | Why divertor spectroscopy needs CR models, and why the QSS shortcut is everywhere |
| 2 | Atomic data | Q2, Q4b | Where every rate comes from, and how each was checked |
| 3 | Theory | Q1, Q3, Q4, Q4c, Q8 | From the rate equation to the two timescales to the QSS criterion |
| 4 | Validation | Q2 Gate A, gates re-run, ℓ-mixing robustness | Why you should believe the matrix |
| 5 | Results | Q5, Q6, Q7, the breakdown/validity map | What the model says about QSS |
| 6 | Limitations | scope list, trapping bound | What this model cannot tell you |
| 7 | Conclusions | everything | What is now known that wasn't |

### Ch. 4 reconciliation (mechanical, but must not be skipped)

`chapter4.tex` is internally contradictory today: 22.7 µs at lines 876 and 993,
but M = 611 at line 908 and 15.3 µs / 25.0 ns / 611 in the Gate E table at 1493
and again at 1556. Those imply τ_relax = 37 ns, a number that appears nowhere.
An examiner reading pages 876 and 908 in sequence will find it.

Also: **Gate E as currently written** ("clear transition at ne ~ 10¹⁴ where M ~ 1")
tests for something the physics says does not exist. Reframe or retire it, with
an argument.

### Scope limitations for §1.5

1. Maxwellian electrons (Q2 — non-Maxwellian breaks detailed balance)
2. T_i = T_e in ℓ-mixing (Q4b — the relevant temperature is the proton one)
3. Optically thin — bounded, not assumed: Θ_P = 0.98 at Te = 3 eV attached, so
   the correction is 1.9%, inside atomic-data uncertainty. **State the honest
   version: τ_relax is insensitive to Ly-α trapping to within 1% even at
   Θ_P = 0.012.** The `summary_trapping.txt` sentence claiming trapping increases
   τ_relax "significantly, by a factor of 1.00" is self-refuting — do not paste it.
4. n_max = 15 truncation, n ≥ 9 bundled (C6 untested — state as a limitation)
5. Uniform plasma, no transport
6. Ground state tracked, not frozen (index 0 of **n**)
7. Open system: the ion is a fixed reservoir in **b**; there is no 44th state
   closing the manifold. **This is why Gate D compares badly to ADAS** — ADAS
   SCD/ACD are defined for a closed ionisation balance.

---

## PART 7 — Phase 4: defense preparation (days 23–30)

Absent from the previous plan entirely. On a **frozen document** — no edits after
day 22 except typos.

**Pass 1 — the framing question.** *"Your title says breakdown and your result
says QSS holds."* Rehearse until it is a 60-second answer with a number in it.

**Pass 2 — methodology.** Why M = τ_QSS/τ_relax and not the inverse. Why max_p on
ratios and not an L² norm. Why 43 states. Why the benchmark point is
illustrative and every general claim is carried by the grid.

**Pass 3 — the bugs.** You will be asked. The answer is a strength: *"I found an
error in my own ℓ-mixing implementation, traced it to Badnell 2021 Eq. 9,
quantified it at ×3–7 on the rates, and proved τ_relax moves by less than 0.85%
anywhere on the grid."* Same for the eigenvalue filter.

**Questions to have answers ready for:**
- Why not close the system with a 44th ion state? (Gate D)
- How do you know n_max = 15 is enough? (C6 — untested; answer honestly)
- Is the transient growth you report physical or an artifact of your norm?
  (Answer: norm- and perturbation-specific, and we state which — present in L²
  for random and 2P kicks, absent in L¹ and for a ground-state kick.)
- What would falsify your central claim?

---

## PART 8 — Cut list

**Into the paper, not the thesis.** All of the following is dated 8–11 April or
earlier, all predates the 21 July matrix, and none is regenerable alongside the
writing:

- Mori-Zwanzig (26 arrays, 7 figures, 4 scripts). `weekC_summary.txt` is
  internally inconsistent: it claims "M_MZ ~ 12× larger than thesis M" two lines
  below its own "M_MZ/M_thesis = 1.1×". The 12× only holds against the retracted
  25 ns τ_relax; against 2.277 ns, τ_K = 2.058 ns is **the same timescale to
  within 10%**. And τ_K is a property of L_FF, the fast excited block — the one
  place the A6 robustness argument does *not* protect, since that is exactly
  where the F(U_m) error lived. Everything MZ must be regenerated before it is
  quoted anywhere.
- S-criterion outputs (regenerate for Q7; the March figures do not go in).
- Balmer sensitivity / regime / robustness / timescale-audit families.
- Trapping analysis — survives as **one paragraph** in Ch. 6 citing the escape
  factor and the insensitivity bound.
- C1 (µ(L) grid map), C2, C3, C4 (pseudospectra), C5, C9, C10.
- C6 (truncation) — state as a scope limitation rather than testing it.

**Paper 2 scope,** once the thesis is submitted: non-normality + pseudospectra,
and the Mori-Zwanzig memory kernel as the formal route to the QSS criterion.
The MZ machinery is the right formalism for Q8 — K̃(0) = −L_SF·L_FF⁻¹·L_FS *is*
the QSS rate — but making that argument properly is a paper, not ten days.

---

## PART 9 — Standing decisions

| Decision | Settled as |
|---|---|
| M convention | M = τ_QSS/τ_relax. Large = the QSS *necessary* condition holds. **The inverse convention is retired** |
| τ_relax definition | Framing A (λ₁ of the full matrix); agreement with Framing B (<0.35% grid-wide, 0.0098% at the reference) reported as robustness |
| τ_QSS definition | Least-negative eigenvalue, **unconditionally** — no magnitude filter |
| τ in De | **OPEN — Q8. Blocks the title** |
| benchmark point | Illustrative anchor only. Prove with the grid, illustrate with the point. Te = 2.947 eV (idx 23), ne = 1.389×10¹⁴ cm⁻³ (idx 5) |
| Error measure | max_p on ratios r_p = n_p/n_1S. **Not** an L² norm over all 43 states — that is ground-dominated (73×) and measures the wrong thing |
| Spectrum framing | One gap of ~10⁴, then a 42-mode quasi-continuum. **Not** "three groups" |
| Boundary descent | Discrete staircase. **No fitted exponent** |
| Transient growth | Always state the norm and the perturbation |
| Cold corner | 19 points have τ_QSS > 1 s (max 67.2 s). Physical but operationally meaningless — state a principled exclusion rather than quoting the full M range |
| Journal | Decide after submission. PRE / JQSRT / JPP class |

---

## PART 10 — Working method

Unchanged, and it works. Recorded here so it survives a chat switch.

- **One quantity per session.** Physics → hand derivation → worked example →
  brutal physics test → code → written note.
- **The student derives. Claude asks and waits.** Recognition is not
  understanding, and recognition collapses under defense questioning. Stuck →
  next step, not the answer.
- **Ground truth: physics → math → code → documents.** Documents are never
  authorities — not this plan.
- **Nothing enters the thesis before ✅** (reproduced on your machine ·
  sensitivity-checked · caveats written · named thesis home · stated falsifier
  that did not occur).
- **Predict before you compute.** Write the expected answer and the refuting
  observation down first.
- **Provenance on every number**, and from now on a provenance header on every
  output file: script, date, source-matrix SHA.
- **Subagents:** `verifier` runs and reports, never repairs. `skeptic` before any
  promotion to ✅. `referee` only after Q8 closes — a referee pass on a claim
  whose central parameter is undefined produces noise.
- **ChatGPT gets raw output, cold**, never Claude's interpretation. Disagreement
  is the signal; agreement between two language models is weak evidence.

---

## PART 11 — This week

1. **Confirm the submission deadline.** If examiners need the thesis a week out,
   the writing window is 23 days, not 30. Everything below shifts.
2. **V0 — commit.** Two minutes. The ℓ-mixing fix currently exists only in your
   working tree.
3. **Book Prof. Pala.**
4. **Run V1–V7** via `verifier`. Report only.
5. **Write down your Q8 prediction** before V3 comes back.
