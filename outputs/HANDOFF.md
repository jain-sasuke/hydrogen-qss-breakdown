# Handoff: non_markovian_cr thesis

Written 11 September 2026, updated the same evening after the Round 2 session.
Everything a fresh session needs to continue without re-deriving the context.
Read this first, then `CLAUDE.md`, then `outputs/REMAINING.md`.

---

## 1. The project

M.Tech thesis, Chemical Engineering, IIT Kanpur. **Defence 15 October 2026**
(corrected in `CLAUDE.md` and here on 11 Sep).
Time-dependent collisional-radiative (CR) modelling of hydrogen, quantifying what
a Balmer-ratio divertor diagnostic costs when it assumes ionisation balance has
settled.

Repo `/Users/phi/Desktop/non_markovian_cr`, branch
`backup/verification-session-2026-09-10`, **pushed** and tracking
`origin/backup/verification-session-2026-09-10`.

**The remote was renamed.** `origin` still points at
`github.com/jain-sasuke/NonMarkovianCR`; GitHub now redirects to
`github.com/jain-sasuke/hydrogen-qss-breakdown`. Pushes work via the redirect
but will break if anyone claims the old name. Fix with
`git remote set-url origin https://github.com/jain-sasuke/hydrogen-qss-breakdown.git`.
It is still **PUBLIC**: `refs/` and `data/processed/**/*.npy` are gitignored and
must stay that way.

**Current state (18 Sep 2026): 218 pages, 0 LaTeX errors, 0 undefined
references, 0 undefined citations, BibTeX clean, 0 em dashes in live prose,
66 bib entries.**

**Page counts need a converged build.** A single `latexmk` invocation can stop
before the page numbers settle, and the count was reported as 202 for most of
16 Sep on exactly that mistake; the committed PDF from that day was under-run.
Run `latexmk` twice and confirm the last `.toc` entry's printed page is
consistent with `pypdf`'s count before quoting a number.
Markers: **1** `\todo` (the front-matter personalise note only), 0
`[UNVERIFIED]`, 0 `[SOURCE REQUIRED]`, 1 `[MECHANISM NOT ESTABLISHED]`
(`chapter5.tex`, the ridge-gain mechanism), 0 `[KEY REQUIRED]`.

Round 2 committed 11 Sep. **Round 4 committed 16 Sep as `5a56d67` and pushed** —
54 files, and it put `validation/` under version control for the first time.

---

## 2. Environment, and the traps in it

**Python is not the default one.** Use `/opt/anaconda3/envs/cr/bin/python`.
The system `python3` has no numpy/scipy/matplotlib. Nothing in `src/` will run
otherwise.

**Build the thesis with:**
```
latexmk -pdf -cd -interaction=nonstopmode thesis_tex/thesis_main.tex
```
The `-cd` matters: chapter files are `\input` by relative name. Check
`grep -c '^! ' thesis_tex/thesis_main.log` and
`grep -ci 'warning--\|error message' thesis_tex/thesis_main.blg`.

**The shell is zsh.** Unquoted `--include=*.py` and `set -- $var` fail. Quote
globs. `timeout` does not exist. Use heredoc + `git commit -F -` for commit
messages, never `-m` with quotes.

**`refs/` is gitignored** and must stay that way. It holds downloaded papers
under publisher copyright; the remote is public. `.gitignore` has a `!*.pdf`
un-ignore earlier in the file that `refs/` overrides.

**PDFs can be read directly.** `pypdf` is installed in the `cr` env. Papers in
`refs/` were read this way, including a 42-page accepted manuscript.

**Web search budget is exhausted** (200/200). `WebFetch` still works. Crossref
(`api.crossref.org`) and Semantic Scholar (`api.semanticscholar.org`) answer;
IOP and ScienceDirect are bot-walled; OSTI `servlets/purl/<id>` serves PDFs.

---

## 3. The argument, and the numbers that carry it

A Balmer ratio is inverted for `(Te, ne)` against a table built at
collisional-radiative equilibrium. Indexing on two axes requires the
neutral-to-ion ratio `u = n_1s/n_ion` to be a function of those two axes, which
holds only once ionisation balance has settled. A divertor during an ELM has not.

**The reversal.** The approximation everyone doubts holds; the one nobody
examines fails. The excited-state QSS closure is accurate to `8.66e-6` at the
benchmark and `6.73e-9` at the worst diagnostic-error point. What does not keep
up is the ground state.

**The exact factorisation.** `eps_plateau = |exp(Sbar · G · Dln Te) − 1|`, with
`Sbar` the mean sensitivity (belongs to the observable) and `G = Dln u/Dln Te`
the reservoir gain (belongs to the plasma). Both are **secants** over the step
taken, not derivatives.

**The bound.** `max|f_3 − f_4| = tanh(|Delta|/4) < 1`, with `Delta` evaluated on
the **post-step** operator.

Key values, all reproducible:

| quantity | value |
|---|---|
| benchmark | `[23,5]`, Te = 2.947 eV, ne = 1.389e14 cm^-3 |
| tau_slow / tau_relax at benchmark | 22.73 us / 2.277 ns, M = 9982 |
| grid | 50 Te (1-10 eV) x 8 ne (1e12-1e15), `dlnTe = ln(10)/49 = 0.0469915` |
| two-channel superposition residual | 3.075e-14 |
| benchmark eps_plateau | 0.063612 |
| census, Te >= 2 eV, 100 us | 45 of 448 on the estimate (44 on the line ratio), 43 propagated; worst 17.5% |
| census at ITER-extrapolated 506 us | 24 of 448 (23 on the line ratio, propagated), worst 15.3% |
| estimate against true average, 448 pairs | 0.979 to 1.022 (100 us), 0.967 to 1.038 (506 us) |
| line/shell eps ratio | 0.9482 to 0.9999, worst at heat [48,0] |
| \|Sbar\| range | 0.0649 to 0.4822 |
| \|G\| range | 2.64 to 14.52, negative everywhere |
| inversion at known n_e, plateau, Te >= 2 eV (K8, 16 Sep) | 448 = 166 off-table + 44 fold-crossed + 238 clean; clean median Te error 47.7 %, amplification 8.3; slope cancellation \|Sigma\|/\|P\| 0.068 |
| slope decomposition (K9, 16 Sep) | Sigma = P + SG (local, exact); Sigbar = Pbar + SGbar (secant, exact, 5e-15); Pbar > 0 and SGbar < 0 at 448/448, both MEASURED, not derived; P_a > 0 448/448, P_c > 0 426/448 |

**Sign convention (settled, do not re-litigate).** *Trap found 16 Sep:* the K8/K9 column `SbarG` is the reservoir secant with the *natural* sign of S = f₃ − f₄ > 0, i.e. −S̄Ḡ in this convention; §5.5.2 uses natural-sign S for the local theorem and says so. Do not write "S̄G < 0" in thesis notation; S̄Ḡ > 0.

**Original note:** `Dln u = ln(u+/u-)`,
post-step over pre-step. Heating: `Dln u < 0`, `Sbar < 0`, `G < 0`, product
positive. Cooling: `Dln u > 0`, `Sbar < 0`, `G < 0`, product negative. `eps` is
positive in both because the equation takes a modulus. Verified on all 2288 rows.

---

## 4. What was done in this work

**Structure.** Six figures inserted with generated captions; abstract,
certificate, declaration, acknowledgements and nomenclature written;
`tau_QSS` renamed `tau_slow` at 102 sites (QSS names the *fast* elimination, so
using it for the slow clock made "QSS holds because tau_QSS is long" a sentence
about two different subsystems).

**Eight new verification scripts**, all in `src/validation/`:
`verify_transport_selection.py`, `diagnose_gate_d.py`, `verify_joint_step_map.py`,
`verify_crest_subgrid.py`, `verify_2s_not_slow.py`,
`verify_cold_corner_pressure.py`, `verify_fault_injection.py`,
`verify_emissivity_generalisation.py`.

**Findings that changed the thesis:**

- **Transport selection effect.** Not one of the 45 defended breakdown pairs
  satisfies `tau_esc > tau_slow`, at 10 or 20 cm, while 27-36% of the other 635
  do. Structural: a pair enters the census because `tau_slow` is long, and a long
  `tau_slow` is exactly when transport sets `n_1s`. Chapter 6 had tested
  transport only at `[0,4]`, below the scope boundary, the one place it survives.
- **Gate D's failure is in the gate.** It mixes both supply channels against a
  table that separates them. Rebuilt on the ground-fed channel: 323 of 400 agree.
  **ACD, which the gate loads and never reads, agrees at 400 of 400**, median
  0.943. Strongest external check in the thesis, available all along.
- **ELM duration.** `tau_drive = 100 us` was unsourced and is the aggressive end.
  Loarte 2003 Sec. 5 gives 506 us for the ITER pedestal. The census nearly halves;
  the worst case barely moves.
- **Lomanowski 2015 reframed the thesis.** No published procedure eliminates
  `n_1s/n_i`; careful practice takes the cancellation route at high n. The thesis
  is now "the cost of the closure where the cancellation is unavailable", not a
  demonstration that Balmer diagnostics fail. This is in the abstract.
- **Emissivity generalisation.** The derivation holds for any non-negative linear
  functional of the fast-state vector, so it covers the real A-weighted
  `Halpha/Hbeta` ratio and not just the shell ratio.

---

## 5. Review history

**Professor review** (earlier): reframing demanded and now done. The remaining
unmet item was "show me the diagnostic that eliminates `n_1s/n_i`", closed by
reading Lomanowski 2015 and reframing.

**Round 1 (mathematical spine).** Verdict FAIL, core survives. All seven repairs
applied in `7e3ef17`. The headline "sign error" was real but misdiagnosed: the
root cause was that `u-` and `u+` were **never defined anywhere**.

**Round 2.** Reviewed by a subagent; I never saw the review text itself. Four
items verified and fixed in `48ac151`: the ELM bound demoted (it printed `>=`
while the thesis's own data has a ratio of 0.9999, a counterexample); the plateau
window sweep cited (it was on disk, uncited, and favourable: `k` sets census
membership, never magnitude); "M carries no information" withdrawn (contradicted
by a table 60 lines below); four caption sites saying "the quasi-steady-state
closure fails", the reverse of the central finding.

**Maths audit of the new work** (`dde2b9c`) found seven defects in same-day
edits, two changing numbers: the grid interval printed as `0.046984` instead of
`0.0469915`, and `|SbarG|` secants presented as the zero-step limit. Also
`eq:eps_plateau_def` defined `R_PE` with `b_1^-` where it must be `u^-`, a factor
of 24 if read literally.

---

## 5b. The Round 2 session (11 September, afternoon)

The Round 2 review text arrived. Its ten demands plus the ramp point were
inventoried against the text (0 closed, 6 partial, 5 open) and then worked
through with one script per demand, each sent to a hostile subagent before its
numbers entered the thesis. Backlog entries G8, K1 to K5 carry the detail.

New scripts, all in `src/validation/`, all stamped under `validation/`:

| script | artifact | what it settles |
|---|---|---|
| `verify_weighted_census.py` | `weighted_census/` | A-weighted Hα/Hβ against the shell ratio: eps ratio 0.9482 to 0.9999, below 1 everywhere; census 45 → 44; the thesis's 0.978 / 2.2 % was wrong (5.2 %) |
| `verify_trajectory_census.py` | `trajectory_census/` | every plateau cell propagated: QSS tracking ≤ 4.4e-5 at all 680 window pairs; true/estimate 0.979 to 1.022 (100 µs), 0.967 to 1.038 (506 µs); exposure ratio ∫j_α/∫j_β; census propagated 45 → 43 |
| `verify_ramp_plateau.py` | `ramp_plateau/` | ramp reaches De(1 − e^(−1/De)) of the plateau, De = τ_slow/t_ramp; benchmark overshoot 6.6 % of eps_plat before the window, a property of the ratio |
| `verify_window_sweep.py` | `window_sweep/`, `divertor_map_w10/20/50/`, `reservoir_gain_w10/50/` | census 45 at k = 10, 20, 30, 50; denominators 547/496/448/364 |
| `verify_m_rank_test.py` | `m_rank_test/` | at fixed Te, M is monotone in ne and eps unimodal: M cannot rank the error |
| `verify_partial_fe.py` | `partial_fe/` | the negative partial survives two-way fixed effects (−0.94 heating), jackknife and permutation |

**Findings that changed the thesis:**

- The single-slow-mode estimate (Eq. 5.10) is neither a bound nor accurate to
  0.7 %: it is within 2.2 % (100 µs) and 3.8 % (506 µs) over the 448 defended
  pairs, low for heating and high for cooling, because eps_CRE is a ratio whose
  denominator relaxes on the same τ_slow (asymptote (1+δ)ln(1+δ)/δ). Its
  τ_drive → 0 limit was stated wrongly. The propagated census is 43 of 448.
- The finite-exposure observable a detector records gives the same census.
- The line-versus-shell worst case is 5.2 % at the hot low-density corner, not
  2.2 %; the mechanism is the channel-dependent ℓ-distribution, with the Hα
  numerator contributing more than the 4f darkness.
- §5.9's negative partial correlation is real (fixed effects), but "at fixed
  (Te, ne)" had no referent, the extrema were mislabelled ([0,0] → cool [1,0]),
  and `make_ch5_figures.py` (~line 893) refuses to draw the figure unless the
  quadratic partial is negative: a results lock, reported to the author, not
  changed.
- The benchmark's step response overshoots the plateau by 6.6 % at 1.2 τ_relax
  (visible in the published trajectory figure, now captioned). Not
  non-normal growth: the deviation norm decays monotonically at all 784 pairs.

**Method notes that saved time:** eigen-propagation of L⁺ is fast and, gated
against `expm` on the state and on the observable with tolerance
max(1e-8, ε_mach‖L‖t), trustworthy; at the cold corner both float64 methods are
limited by ε_mach‖L‖t and the eigen path is the more accurate (checked against
a 30-digit mpmath exponential). `expm_multiply` hangs at ‖Lt‖ ~ 1e10; do not
use it there. Exposure integrals are exact through the augmented matrix
[[L, d0],[0, 0]]. The 100 samples/decade log trapezoid carries a +8e-5 bias;
the stamped artifact uses 400.

## 5c. The Round 4 session (16 September)

An external ChatGPT review ("Round 4", verdict FAIL) was adjudicated rather than
obeyed. Four read-only audit agents checked it against the tree; two of its
fifteen findings were wrong, three were already fixed, and one broke a thesis
claim.

**The result that changed a claim.** Chapter 4's l-closure explanation of the
Fujimoto r_1 deficit is withdrawn. Imposing the source's own statistical-l
closure, `L_bundle = P L R`, moves r_1(3) by **0.42 %** (8.29 -> 8.26) and makes
p=2 *worse* (10.8x -> 11.8x low); r_0(2) also degrades, 0.7392 -> 0.6459 against
a tabulated 0.730. The projector annihilates the l-mixing operator identically
(`max|P B R| = 8.5e-7` against a scale of 1.0e11), so removing l-mixing is the
opposite limit, not a proxy, and the old "bracket" argument was never a bracket.
Stamped by `src/validation/verify_fujimoto_bundle.py` ->
`validation/fujimoto_bundle/`, six gates including a **severity gate** (bundling
a deliberately non-statistical operator moves r_1(2) by 129x, so the null is a
measurement not a blind instrument) and a **calibration gate** (production
r_1(3) reproduces the value Chapter 4 already quoted, to 2.8e-5).
Status is now an open external disagreement with three eliminated causes.

**Ten numbers requoted** where a stamped artifact disagreed with the text; see
the commit message of `5a56d67` for the list. The sharpest: Chapter 5's
Lyman-alpha depths reproduced only under a *wrong* Doppler width `sqrt(kT/m)`,
and were credited to a script that never emits an optical depth.

**The eigenvalue filter** `eigs[eigs < -1.0]` was removed from
`src/rates/solve_cr.py:269` and `check_mz.py:10` after confirming nothing
imports either. It changes no number: identical at the benchmark, and at the
cold corner the old filter returned tau_relax as tau_slow, wrong by nine orders.
Chapters 2 and 4 now record the repair. **CLAUDE.md's known-issues entry is
stale** — it points at `qss_analysis.py`, which was fixed on 23 Aug.

**Appendices A, B, C written** from empty title pages. **Table 4.4 rebuilt**:
18 internal + 5 external rows and a **provenance column** separating stamped
results from working-note ones. Four checks graded *severe* rest on notes.
**New section 6.10** gathers the five open questions.

## 5d. The 22 September session

Read `CLAUDE.md`, this file and backlog K10 to K19 first; the state described in
§1 held. `origin` now points at `github.com/jain-sasuke/hydrogen-qss-breakdown.git`
(`git remote set-url` run; both branches answer). Nothing committed or pushed
this session; the author decides.

**State at the end of the session.** 209 pages (converged, two latexmk runs; up from 207 by the four requotes), 0 LaTeX errors, 0 BibTeX warnings, 0 undefined references, 0 em dashes outside comments. Nothing committed; the author decides. Files touched: chapters 1, 2, 4, 5, 6, 7, appendix C; new `src/validation/{build_L_at_Te,verify_l_blindness,verify_interior_operator,verify_maxwell_grid_convergence}.py`; new `validation/{l_blindness,interior_operator,maxwell_grid_convergence}/`; `outputs/{HANDOFF,claim_evidence_table,thesis_grade_results_backlog}.md`, new `outputs/claim_evidence_table_lineref_refresh_2026-09-22.md`, `outputs/round6_adjudication_2026-09-22.md`.

**Number guard.** Before any edit, every numeric token per `.tex` file was
extracted from HEAD (`3b613c1`); after editing, no token was lost from the
thesis, and every gained token traces to a stamped artifact named in the
backlog entry (K20, K21).

**`outputs/claim_evidence_table.md` line references refreshed** by an
evidence-auditor subagent against `3b613c1`: 348 occurrences over 291
(chapter, range) keys; 318 replaced (204 HIGH, 63 MEDIUM confidence keys), 25
left with a `[ref stale 22 Sep: ...]` tag; mapping in
`outputs/claim_evidence_table_lineref_refresh_2026-09-22.md`. Verdicts and
body text untouched. Residual inconsistencies the refresh found still live in
the chapters (report only): ch3 "three decades" for a 3.54-decade range;
86.8 (full matrix) versus 86.5 (excited block) presented as one quantity across
ch3, ch4, ch5; 54 versus 104 no-plateau exclusions in ch5; 2300 and 2332 for the
same ratio in ch6; ch2's "21 July" write date against CLAUDE.md's 14 July.

**K20, the ℓ-blindness argument (backlog).** `verify_l_blindness.py` →
`validation/l_blindness/`. The CCC tables resolve ℓ on both sides up to
n = 10, so the argument §6.8.3 called untested could be tested. The rates are
NOT ℓ-blind (adjacent upward spread grows with n to 2.5 to 2.8 at n = 9,
downward 5 to 6 at n = 7 to 10, largest at the yrast sublevel); the statistical
distribution at the 121 combinations is licensed instead by detailed balance
among shells that are Boltzmann with each other to 6e-5 and Saha with the
continuum to 1e-4 at all 34 grid points; the two criteria (mixing, LTE) cover
every grid point. The stake was up to 21 % in u_CRE had the assumption failed.
Hostile audit demanded six changes; all applied before the requote. ch6
sec:bundling_check closing paragraph and ch4's ladder summary requoted.

**`build_L_at_Te.py`, the operator at an arbitrary Te.** Every ingredient of L
and S is a closed-form function of Te or a Maxwell average of a stored
cross section, so the operator can be built at any Te from the pipeline's raw
inputs without editing a rate module: `compute_K_CCC.py` cannot be imported
(it writes at module scope) and its Maxwell average and the `assemble_K_exc.py`
merge rules are re-implemented; everything else is imported read-only. The
gate reproduces every stored table, `L_grid` and `S_grid` at all 50 x 8 nodes
with difference 0.0. Use `OperatorBuilder.load().operator(Te, ne)`; the
ℓ-mixing cutoff is frozen at 1e14 as in `L_grid` unless `lmix_ne` is passed.

**K21, K22 (backlog).** `verify_interior_operator.py` → `validation/interior_operator/`:
the quarter rule measured (0.25 to 0.27), the true ramp at [15,3] reaches
0.9956 / 0.9567 / 0.6548 of the plateau (Tₑ linear in t; 0.6571 for ln Tₑ
linear in t), inside the interpolant bracket; the stamped 0.0107 gap splits
0.0079 linear interpolation, 0.0005 log-linear, 0.0024 ramp shape. Its audit
found the pipeline's 5000-point Maxwell-average grid carries about 1 % in
τ_slow at the ramp point; `verify_maxwell_grid_convergence.py` →
`validation/maxwell_grid_convergence/` stamped it (100000 points: τ_slow up to
1.5 %, u_CRE 1.3 %, S 0.8 %, ε_plateau 1.0 % at Tₑ ≥ 2 eV; 3.3 % at the cold
corner; nothing repaired, `N_GRID` stays 5000). ch5 sec 5.8.2 and ch2
sec:maxwellian requoted. Reported to the author: `compute_K_CCC.py`'s
"validated: <2% error" comment does not hold for the weak transitions.

**K23, Round 6 adjudicated and applied** (`outputs/round6_adjudication_2026-09-22.md`):
35 claims, 33 correct in some degree (8 already applied, 3 understated), 1
wrong (stale: the ADAS reservoir check exists since K13), 0 fabricated. Chapter
1 (11 edits) and chapter 7 (21 edits, plus one appendix C sentence) requoted
by science-editor subagents from stamped artifacts. The one item left to the
author: re-weighting §7.3 toward error separation, the crest mechanism and
Σ = P + SG, which is a restructuring under the page budget.

**Master register (Rounds 0 to 10) adjudicated and applied, K24, K25**
(`outputs/master_register_adjudication_2026-09-22.md`): 27 of 36 MUST items
were already applied, 6 had residual sentences (applied), M19 stamped
(`verify_saha_balance.py`), M22 superseded by K21; M28 and M31 are yours.
Prose cleanup across all chapters after reading thesis-writing guidance (Lund,
Coventry, Thomson, the examiner study): draft-history narration, internal
file names, reader address, rhetorical words and 24 detective-style headings
gone from live prose; every number, caveat and provenance pointer kept (guard:
12 lost tokens, all superseded values with their stamped replacement in
place; the 100000-point grid sentence and the K23/K25 requotes are the gains).
Caption edits were made in both the `.tex` files and the generator
templates; a `--force` regeneration was not rerun this session. Final state:
**208 pages**, 0 errors, 0 BibTeX warnings, 0 undefined references, 0 em
dashes outside comments, `pdftitle`/`pdfauthor` set. Nothing committed.

**Round 7 adjudicated and applied, K26** (`outputs/round7_adjudication_2026-09-22.md`).
Six claims: Park 1972 correct and understated, the chapter 3 chord cancellation
correct, Fujimoto wording and the provenance defect partially correct, the
abstract claim wrong as stated, the console criterion already applied. Park was
read in full from `refs/` and is now cited: he assumes excited-state QSS, uses
the same two-coefficient decomposition and the same ground-state parameter, but
computes only in the two terminal limits where the sensitivity vanishes by his
own definition, and zero of the 400 grid points lie there. A chord-integration
limitation was added to chapter 6, which had none. Auditing a claim the reviewer
CLEARED found a real overstatement: chapter 6's step idealisation is safe by one
to two orders of magnitude at the median but only by a factor of two at the
least favourable pair, where the ramp reaches 0.78 of the step plateau.

**Open and formal: the abstract is about 734 words against the IIT Kanpur
M.Tech limit of 300.** No review in the series raised it. The cut is yours.

**Round P0 adjudicated and applied, K27** (`outputs/roundP0_adjudication_2026-09-22.md`).
The dual-writer contamination claim is refuted on all six results it named; the
arrays feed nothing printed and the two writers are byte-identical.
`verify_headline_provenance.py` now re-derives all 45 numbers that carried a
working-note citation from stamped artifacts, and `verify_ladder_counts.py`
makes Table 4.4's caption counts self-checking. Two real defects were found in
the process: tab:position_effect's u-ratio mixes temperature indices (stated in
the caption now), and **24 of 97 validation CSV files carry no provenance header
at all**, including artifacts behind abstract-level numbers. That last one is
the largest open provenance item and is yours, since writing headers means
re-running the producers.

**Round P0 re-check adjudicated and applied, K28** (`outputs/roundP0b_adjudication_2026-09-22.md`).
The review was right: active numbers still cited a working note. `verify_residual_claims.py`
now computes all of them from the canonical operator and **all 43 reproduce**, including the
five ell-weighting values of Delta to five decimals, the four eigenvalue condition numbers
exactly, and the ion-closure census (202 to 200, 38.68 to 38.56 per cent, 45 of 448). The
phrase is down from 23 live occurrences to 7: five labelled historical, one the Table 4.4
definition, one a structural claim with no measured number. One real error surfaced: the
sub-grid crest endpoints were wrong and the drift is not monotone (it reverses at 5.96 eV);
requoted. Two constructions could not be recovered (the matched heat/cool ratio 1.036 and the
linearisation median 1.031) and the text now says so rather than citing a note.

**The abstract is cut to the institute limit, K29.** 270 words as rendered
against the 300-word M.Tech limit, down from about 800. Every scope statement
the review rounds installed survives (known density, the "not a demonstration
that divertor spectroscopy fails" disclaimer, the ceiling attributed to the
linear system, the 166 + 44 + 238 split, the closed-parcel conditionality). The
number guard is unchanged at 15 lost and 157 gained, so nothing dropped from the
abstract left the thesis. 210 pages.

## 6. What is left

Read `outputs/REMAINING.md` for the full register. The substantive items:

1. ~~The emissivity-ratio reformulation not carried through the numerics.~~
   **Done 11 Sep** (`verify_weighted_census.py`, `verify_trajectory_census.py`).
   The figure scripts still draw the shell ratio; the chapters state the line
   numbers alongside. Do **not** use
   `Halpha_sensitivity.py:estimate_halpha_weights_from_L`, which self-documents
   as unsuitable.
2. **Three appendices are empty stubs**: state ordering, atomic data sources,
   numerical methods and convergence.
3. **A5 needs an independent re-run.** Demoted from verified because the May run
   was never repeated.
4. **The crest's sensitivity to n_max** (the one remaining `[UNVERIFIED]`, at
   `chapter5.tex:1242`) needs the rate pipeline rerun at n_max = 12 and 20.
5. **Eight older Ch3/Ch5 figures fail a colour-blind check** (`#2e7d32` against
   `#c0392b`, dE 4.2 under deuteranopia).
6. ~~Round 2's remaining items.~~ **Closed 11 Sep**; see §5b. The bridge test
   turned out to cost 9 s, not hours.
7. **The two-parameter inversion is not measured** (16 Sep). K8 measured the
   single-parameter one at known n_e; the 2x2 Jacobian of `eq:inversion_2x2`
   needs a second line ratio.
8. Four `\todo`s are author-supplied citations or named-as-impossible work
   (SOLPS-coupled calculation, molecular matrix, the separatrix quantity, the
   n=15 ground-fed provenance).

---

## 7. Working conventions that matter here

- **`CLAUDE.md` rules are real.** Report, do not repair. No hardcoded grids. Fail
  loudly. Provenance on every number. Ground truth is physics > math > code >
  documents. A failing check is information.
- **Every number in the thesis needs a producing script and a stamped artifact
  under `validation/`.** "Recorded in a markdown file on a date" is a provenance
  defect even when the value is right.
- **Check before asserting.** Several times in this work a plausible sentence was
  wrong and the data caught it: the cooling sign case, the "near-step-independent"
  product, the grid interval. Write the check, then the sentence.
- **The user edits files in parallel.** Commit with an explicit file list, never
  `git add -A`, or their work gets swept into your commit message. This happened
  once (`71691bf`).
- **Skills to use:** `scientific-prose-audit` as a *final* pass only, after the
  physics is frozen; `thesis-verification` for establishing numbers;
  `thesis-writing` for producing prose. Subagents: `math-auditor` earns its keep
  and found seven real defects; `Explore` is good for exhaustive inventories.
- **Prose rules:** zero em dashes, no filler transitions, claim strength matched
  to evidence, physical quantities named. The mechanical scan is in the
  `scientific-prose-audit` skill.
