# Claim-to-Evidence Table — Chapters 1 to 7

**GATE 5. Compiled 10 September 2026. Report only.** No chapter, script, data
file, register entry or `validation/` artifact was modified in producing this
document. Two report-only scripts (`verifych3_gb.py`, `verify_ch3_claims.py`)
were executed; both print "Nothing was written" and both were confirmed to
contain no write call. Two throwaway scripts were written to the session
scratchpad only.

**Canonical matrix for everything below:** `data/processed/cr_matrix/L_grid.npy`,
SHA-256 `2d92b58e…59224e`, companion `S_grid.npy` `7822f536…fb80a`.

---

## Sources cross-referenced

| Source | Role here |
|---|---|
| `thesis_tex/chapter1.tex` … `chapter7.tex` | the claims |
| `outputs/claim_hierarchy.md` (10 Sep) | claim architecture, validation ladder, 15 approximations |
| `outputs/thesis_ready.md` (23 Aug, corrected 10 Sep) | register A1–A12, blockers B1–B8 |
| `outputs/thesis_grade_results_backlog.md` §F–J | session register, C1/C2/C6 closures, G1–G7, H1–H11, I1–I10, J |
| `outputs/findings_10_four_agent_review.md`, `findings_09_*`, `CH5_EVIDENCE.md`, `CHANGE_REPORT.md`, `pivot_decision.md` | the numbers the chapters cite |
| `validation/**` (37 entries), `src/validation/*.py`, `src/rates/assemble_cr_matrix.py` | the artifacts, or their absence |

## What was extracted

| Chapter | lines | assertions extracted | own-computation numeric | rows given below |
|---|---|---|---|---|
| 1 Reading the Light | 849 | 62 | 6 | 8 |
| 2 The Atoms | 1335 | 157 | ~95 | 26 |
| 3 Two Clocks | 1641 | 160 | ~90 | 34 |
| 4 Does the Model Work? | **860, incomplete** | 80 | ~70 | 31 |
| 5 What the Model Says | 1546 | 134 | ~120 | 38 |
| 6 What This Model Cannot Say | 1077 | 99 | ~80 | 24 |
| 7 What It Means | 435 | 43 | ~30 | 14 |
| | | **735** | | **175** |

**Selection rule for the table.** Every claim that (a) states a number this
thesis computed, and (b) is load-bearing for Claims A–G of
`claim_hierarchy.md`, gets a row. Textbook-standard results (Bohr levels, the
Maxwellian average, Levy–Desplanques, Holstein's integral) and pure
definitions are aggregated, not itemised: they are not what an examiner
attacks. Literature-cited claims get a row only where the citation is missing,
wrong, or doing quantitative work.

## Verdict key

`SUPPORTED` · `SCOPE-RESTRICTED` (true over a narrower set than stated) ·
`UNSUPPORTED` (no artifact, or no test that could have failed) ·
`REFUTED` (measurement contradicts the sentence as written).

Graduation states are the project's own: 💡 idea · 📐 derived · 💻 implemented ·
▶️ run · ✅ verified with sensitivity check and written caveats.

---

## PART 0 — Headline counts

Counts are machine-counted from the tables in PART 4, not estimated. Each row
is classified by the **most severe** verdict it carries, so a row reading
"number SUPPORTED, provenance UNSUPPORTED" counts once, as UNSUPPORTED.

| | count |
|---|---|
| Rows adjudicated | **175** |
| SUPPORTED outright | 139 |
| SCOPE-RESTRICTED (true over a narrower set than stated) | 5 |
| UNSUPPORTED (no artifact, or no test that could have failed) | 20 |
| REFUTED (measurement contradicts the written sentence) | **11** |
| Claims whose only artifact is a markdown file | **57** |
| Claims the chapters already self-flag `[UNVERIFIED]` / `[SOURCE REQUIRED]` / `\todo` | 21 |
| Gates presented as validation that cannot fail on the fault class they guard | **3, all confirmed today** |

The SCOPE-RESTRICTED count of 5 understates the problem and should be read
with PART 3: seven extrema are audited there, four of them quoted globally
somewhere in the project, and three of those four have already been corrected
in the chapters but not in the registers.

**The single most important structural fact,** carried forward from
`backlog` §J and confirmed here against the written chapters: of thirteen
major results, **twelve have a prediction step with a named falsifier and one
does not** — R6, the magnitude, which is the number the abstract was planned
around. Everything in Part 3 below follows from that absence.

---

# PART 1 — The three gates that cannot fail

The brief named three checks currently reported as validation. **All three are
confirmed. All three are wiring checks, not validation.** Each was tested by
fault injection against the real data, in the `cr` conda environment, with
modified copies run from the scratchpad. Two independent runs (mine and a
verification agent's) agree on the conservation baseline to six figures.

---

## 1.1 The particle-conservation gate — CONFIRMED TAUTOLOGICAL

**Where:** `src/rates/assemble_cr_matrix.py:271-282`, "Check D", inside
`precompute_L_grid`. Prints:

```
Check D — Column sum = -K_ion*ne (particle conservation):
  Max relative error = {max_col_err:.2e}  PASS/FAIL   (threshold 1e-8)
```

**What it computes.** `col_sums = L_grid[i,j].sum(axis=0)` against
`expected = -K_ion[:,i]*ne[j]`, relative error, max over the sample.

**Coverage — confirmed by reading the strides.** `range(0, n_Te, 10)` gives 5
temperatures (0,10,20,30,40 of 50); `ne_grid[::2]` gives 4 densities (0,2,4,6
of 8). **20 of 400 points.** `derivation_01_rate_matrix.md:142` claims the
residual holds "across all 50×8 grid points" and `chapter3.tex:267-272` says
"evaluated at all 400 grid points". **The committed code evaluates 20.**

**Why it cannot fail on a rate error — the structural reason.** In `build_L`
every collisional and radiative block writes its diagonal as *minus the row
sum of the off-diagonal entries it has just written*
(`np.fill_diagonal(L, np.diag(L) - Ke.sum(axis=1) - Kd.sum(axis=1))`, and the
same pattern for the radiative and ℓ-mixing blocks). Only `K_ion_final` is a
pure diagonal sink with no off-diagonal partner — and that is precisely the
`-K_ion*ne` the check compares against. So the column-sum identity is true
**by construction** for every block whose diagonal is derived from its own
off-diagonals, whatever numbers those off-diagonals contain.

**Fault injection, run today against the real arrays:**

| injected fault | max abs ΔL (s⁻¹) | Check D residual | outcome |
|---|---|---|---|
| baseline | — | **3.607812×10⁻¹¹** | PASS |
| `K_exc_full[1s→2p]` × 10 | 3.58×10⁴ | 3.607812×10⁻¹¹ **unchanged** | PASS |
| all de-excitation deleted | 9.89×10⁹ | 3.607812×10⁻¹¹ **unchanged** | PASS |
| `K_exc_full` transposed | 1.42×10¹⁰ | 2.009664×10⁻¹¹ same order | PASS |
| `A` halved, `γ` **not** halved | 3.13×10⁸ | **7.22×10⁴** | **FAIL — caught** |

This reproduces `CHANGE_REPORT.md` §4.2 qualitatively and exactly in
structure. **The only fault class it detects is an inconsistency between a
term's diagonal and its own off-diagonals** — an `A`/γ wiring error.

**Consequence for written LaTeX.** `chapter3.tex:222-225`:

> "An error in a single off-diagonal element, a transposed index, or a missing
> back-reaction all break this immediately."

Three fault classes are named. **Two of the three were injected and did not
break it** (the transposed index, and the missing back-reaction — de-excitation
deleted entirely). The first breaks it only if the diagonal is not updated
consistently, which in this assembler it always is. **Verdict: REFUTED as
written.** `chapter4.tex:182-224` already reports the fault-injection table and
grades the check "tautological"; **Chapter 3 and Chapter 4 contradict each
other on the same gate**, and Chapter 3 is the one that is wrong.

**Persistence.** The residual is printed and never written. `L_meta.csv`
carries shape and grid metadata only, with no `max_col_err` field. It appears
in **no file under `validation/`**. Per `findings_10` §11 item 10, this is the
exact condition under which three retractions happened.

---

## 1.2 The tanh bound gate — CONFIRMED TAUTOLOGICAL, and the chapter's stated reason is wrong

**Where:** `src/analysis/make_ch3_figures.py:375-425` (closed form
`np.tanh(abs(Delta)/4.0)` against a numerically swept `argmax|f₃−f₄|`, raising
above 5×10⁻³) and `src/validation/verify_ridge_mechanism.py:838-845` (raise)
with the pass line at `:957`.

**Fault injection at the benchmark [23,5], real `a`, `c` from the matrix.**
Run twice, independently, by two agents using different code; the Δ values
agree to all six decimals shown. State ordering loaded through
`CRContext.load()` rather than hardcoded, per CLAUDE.md rule 1. Measured
channel coefficients at [23,5]:

```
a3 = 4.543572e-05   a4 = 1.073753e-05
c3 = 1.140562e-07   c4 = 1.886233e-07
min(a) = 1.082e-08   min(c) = 8.555e-09     -> a, c >= 0 confirmed
```

The positivity of every component of `a` and `c` is the premise the theorem
needs, and it is measured here rather than assumed. It follows from
`-L_FF` being a non-singular M-matrix, and independently confirms the
"0 negative entries at all 400 points" result behind Claim B.

| input | Δ | tanh(\|Δ\|/4) | numerical peak | rel dev | outcome |
|---|---|---|---|---|---|
| unperturbed | +1.945614 | 0.4513573 | 0.4513573 | 3.9e-10 | PASS |
| 3↔4 full shell swap | −1.945614 | 0.4513573 | 0.4513573 | 3.9e-10 | PASS |
| **c₃↔c₄ only** | **+0.939493** | **0.2306474** | 0.2306474 | 4.6e-10 | **PASS** |
| a₃↔a₄ only | −0.939493 | 0.2306474 | 0.2306474 | 4.6e-10 | PASS |
| a₃ × 7.3 | +3.933488 | 0.7545220 | 0.7545220 | 0.0 | PASS |
| `tanh(\|Δ\|/2)` instead of `/4` | +1.945614 | 0.7499352 | 0.4513573 | 3.98e-01 | **FAIL — caught** |

The unperturbed Δ = 1.945614 reproduces `chapter5.tex`'s 1.94561 to six
figures, so the corrupted rows are perturbations of the real quantity and not
of a mis-set-up calculation.

**Confirmed: the gate passes on every corrupted physical input and fails only
on an arithmetic bug in its own formula.** It is a theorem for any
`a, c ≥ 0`: both sides are recomputed from the same, possibly corrupted,
coefficients.

**A REFUTED sentence in Chapter 4.** `chapter4.tex:548-550`:

> "Swapping shells 3↔4 leaves both |Δ| and the logarithmic ratio unchanged;
> swapping c₃↔c₄ likewise."

Write `A = ln(a₃/a₄)`, `C = ln(c₃/c₄)`, so `Δ = A − C`. The full 3↔4 relabel
sends `Δ → −Δ`, leaving `|Δ|` fixed — correct. But `c₃↔c₄` alone sends
`C → −C`, hence `Δ → A + C`, which equals `±Δ` only if `A·C = 0`. **Measured:
|Δ| moves from 1.945614 to 0.939493, a factor 2.07.** The gate still passes,
but not for the reason the chapter gives. **Verdict on the conclusion:
SUPPORTED. Verdict on the stated justification: REFUTED.** The correct
sentence is that the bound survives a c₃↔c₄ swap because it is a theorem about
whatever non-negative pair is supplied, not because |Δ| is invariant.

**Docstring vs printed line.** `verify_ridge_mechanism.py:148` files it under
`HARD / WIRING CHECKS`. The line at `:957` prints
`"tanh bound honoured at all {n} points"` with no such label, while the
*adjacent* kernel check at `:899` does print `"WIRING CHECK …"`. A reader sees
the unlabelled line. `chapter4.tex:550-552` already says exactly this and
grades it tautological — **Chapter 4 is correct here and should not be
softened.**

---

## 1.3 Gate C, the Saha limit — CONFIRMED: it does not test the limit it is named for

**Where:** `src/validation/validate_gates.py:197-270`, `gate_C`.

**Pass flag, verbatim (line 250):**

```python
'passed':      monotone and below_saha,
```

with, at lines 230-239:

```python
monotone    = bool(np.all(np.diff(ratios_arr) >= 0))
below_saha  = bool(np.all(ratios_arr <= saha_2P * 1.05))   # 5% tolerance
approach    = ratios_arr[-1] / saha_2P if saha_2P > 0 else 0
approaching = approach > 1e-4
```

**`approaching` is computed, stored in the results dict at line 249, and
excluded from the pass flag.** Confirmed by reading the expression.

**Measured on disk.** `validation/gate_C.csv`, column `approach_frac`:
0.01548, 0.01535, 0.01522, 0.01509, 0.01497, 0.01485, 0.01474, 0.01462,
0.01451, 0.01441 … 0.01359. **Populations reach 1.2–1.5% of Saha at the top of
the grid**, `ne = 10¹⁵`. `validation/gate_summary.txt` reports
`Gate C: PASS (100% of points)`.

**Why `below_saha` cannot fail here.** The model is open: `S` is an externally
imposed source with the ion reservoir held fixed and no Saha tie between the
continuum and the bound states, and there is no radiation trapping. Nothing on
this grid drives the populations toward LTE. The tolerance band sits ~65×
above the highest measured ratio. An ordinary rate-coefficient fault cannot
move populations by the two orders of magnitude needed to trip it — the
Check-D injections above (transposed excitation array, de-excitation deleted)
do not come close.

**What Gate C actually tests:** that the CR-equilibrium 2P/1S ratio rises
monotonically with `ne` — a real and checkable trend — and that it stays below
Saha, which the regime guarantees. **Verdict: WIRING / REGIME CHECK, weaker
than the other two but not validation of the named limit.**
`chapter4.tex:348-364` already grades it "weak, and mis-named" and reports the
1.5% figure. Correct; keep it.

---

## 1.4 What must change, and what must not

| Item | Action |
|---|---|
| `chapter3.tex:222-225` | **REFUTED.** Rewrite: the check detects an `A`/γ inconsistency and nothing else. Do not delete the check. |
| `chapter4.tex:548-550` | **Justification REFUTED.** Replace "likewise" with the theorem argument. |
| `verify_ridge_mechanism.py:957` | Printed line should carry the `WIRING CHECK` prefix its own docstring uses, as line 899 already does. |
| `derivation_01:142`, `chapter3.tex:270` | Say "20 of 400 sampled" or widen Check D. Do not adjust the tolerance. |
| Check D residual | Not persisted anywhere. Stamp it. |
| Gate C | Keep the `approaching` column, and either put it in the flag or say in the text that it is excluded. |

**None of these three checks should be repaired to pass or removed.** All three
are correctly implemented for what they actually do. The defect is the label.

---

# PART 2 — Numbers whose only artifact is a markdown file

This is the largest single class of defect found. **57 numeric claims in
written LaTeX cite a markdown file as their source, or cite nothing at all
while no script computes them.**

## 2.1 Five verification scripts write nothing to disk

Confirmed by grepping every `to_csv`, `savez`, `np.save`, `savetxt` and file
`open(...,'w')` in `src/validation/`:

| script | lines | writes | what depends on it |
|---|---|---|---|
| `verify_ridge_mechanism.py` | 1577 | **nothing** | Claim F.3 — "the best-evidenced result in the thesis". Every number in `chapter5.tex` §mechanism: the 32/32 and 0/32 attribution counts, the 7.11× vs 1.01 amplitude argument, the Griem 7.2-vs-6.7 test, `tab:mechanism_profiles`, the 76%/19% temperature split |
| `verify_eps_gridmap.py` | 177 | **nothing** | `chapter5.tex:289-297` — the defended-scope maximum 18.07% at [15,3], the benchmark 6.36%, `tab:step_dependence` |
| `verify_plateau_bridge.py` | 2355 | **nothing** | `chapter5.tex:1079-1136` — the whole dynamic bridge: 1.94 τ_relax, 3.72 τ_relax, 3.33%/1.28% plateau flatness, the decay-law constants, the honestly-reported preregistration failure |
| `verify_ch3_claims.py` | 224 | **nothing** (by design) | its own audit, including the τ_QSS floor finding |
| `verifych3_gb.py` | 196 | **nothing** | `chapter3.tex:809-822` — the 1.118/1.151/1.526 ratio, `M_eff = 8659`, `M_eff ≥ 77.6` |

`verify_ridge_mechanism.py` is the producing script for Claim F.3, which
`claim_hierarchy.md` calls "the best-evidenced result in the thesis" and PART 3
grades **exemplary**. It leaves no artifact. The result is reproducible only by
re-running the script and reading the terminal.

**This is not an argument that the numbers are wrong.** I re-ran
`verifych3_gb.py` today and it reproduces `chapter3.tex`'s 1.118 / 1.151 /
1.526 / 8659 / 77.6 exactly, and reproduces the 0.0098% and 0.346% of
`chapter3.tex:589-590` (measured 0.0098% and 0.3461%). The defect is that
nothing on disk records it.

## 2.2 Scripts with no output for a claim that names 400 points

| claim | file:line | committed script | artifact |
|---|---|---|---|
| κ(L_FF) ∈ [1.48×10³, 1.74×10⁵] grid-wide | `chapter3.tex:719-722`, `chapter4.tex:572-576` | **none** | none |
| μ(L) > 0 at every grid point | `chapter3.tex:1559-1560` | none named | none |
| −L_FF⁻¹ has no negative entry at any of 400 points | `chapter3.tex:1186-1187` | none named | none |
| b₁ ∈ [76.6, 2.683×10⁵], 956 at benchmark | `chapter3.tex:945-948` | none named | none |
| η ∈ [1.16, 29.5], 3.12 at benchmark | `chapter3.tex:411-412` | none named | none |

**The condition-number case is worth stating precisely, because the register
and the chapter disagree.** `backlog` H11 says the range "now has a script".
`chapter4.tex:580-587` says, in the compiled text:

> **[UNVERIFIED: no committed script produces the grid-wide condition
> number.]** … `np.linalg.cond` appears at one site in `src/`, inside
> `verify_fujimoto_table41.py`:334, and covers four point-variants at
> Te = 10 eV only.

I confirmed the chapter, not the register: `np.linalg.cond` has exactly four
hits repo-wide — `verify_fujimoto_table41.py:334` and three `Balmer_*` scripts
that condition the *eigenvector* matrix, not `L_FF`. **`backlog` H11 is wrong;
`chapter4.tex` is right.** The number is correct and still has no producer.

## 2.3 Backlog §G — five results graduated ✅ Verified with no script and no artifact

| entry | headline numbers | producing script | artifact | reaches |
|---|---|---|---|---|
| **G3** quasi-neutrality | 68 of 392 need \|Δnₑ/nₑ\|>10%; 36 >100%; 235 Pa; 3280 Pa; +20.0 at [0,0] | **none found** | none | `chapter5.tex:1333-1408`, `chapter7.tex:346-350` |
| **G4** QSS partition | min spectral separation **86.5**; 8.229 s⁻¹; 553×; 7×10³ cm⁻³ | **none** (`grep two_photon\|8.229` over `src/` returns nothing) | none | `chapter5.tex:433-473` |
| **G5** detailed balance | 2457 pairs, max dev 7.31×10⁻⁹; Saha ratio 0.999997 for all 43 states | **none** | none | `chapter4.tex:300-313` |
| **G6** ℓ-mixing saturated | ×0.1–×10 moves f₃−f₄ <0.3% / <5%; Debye cutoff ≤0.42%; q spans ×3.7 (n=2) to ×50 (n=8) | **none** | none | not yet in a chapter |
| **G7** Balmer opacity | τ(Hα,10 cm) = 6.3×10⁻⁷ / 1.0×10⁻³ / **0.263**; Θ ≥ 0.85 | **none** | none | `chapter4.tex:711-717` |

All five are marked ✅ Verified in the backlog. **Under this project's own
graduation rule — "Nothing reaches ✅ without a sensitivity check and written
caveats" — and CLAUDE.md rule 4, provenance on every number, none of the five
can hold ✅ while its only artifact is a markdown paragraph.** G4's 86.5 is
promoted in `chapter5.tex:466-473` to "**the figure that describes the grid**",
displacing the benchmark 9982. That is a headline sentence resting on a number
with no producer.

To the chapters' credit, `chapter5.tex:475-479` already carries
`[SOURCE REQUIRED]` over exactly this block. `chapter4.tex:302, 716` does not —
it cites `findings_10 ADDENDUM D.6` and `D.8` as if a markdown section were an
artifact.

## 2.4 Chapter 3 names no producer for any number

**`chapter3.tex` contains ~90 own-computation numeric claims and names zero
scripts, zero `.npy` files, zero `validation/` paths.** The only producer
referenced anywhere is the phrase "the figure script", unnamed, at lines 570,
1285 and 1338. This is the chapter that establishes Claims B, C, D, F.1 and
F.2 — the conceptual core.

## 2.5 The `findings_10 ADDENDUM` citation pattern

Roughly 40 sites in Chapters 4, 5 and 7 cite `findings_10` or `thesis_ready`
sections in place of an artifact. A representative list:

`chapter4.tex`: 302 (D.6), 457 (B.1), 550 (B.3), 581 (C.1), 604 (ADDENDUM B),
620 (D.8), 716 (D.8), 772 (§3.1), 830 (§3.2), 384/696 (`thesis_ready` A1/A6).
`chapter5.tex`: 231, 244, 304, 343, 347, 357, 389, 510, 616, 628, 659, 698,
705, 720, 844, 858, 931, 952, 982, 1153, 1196, 1222, 1243, 1298.
`chapter7.tex`: 363, 367 (both D.4).

Per the auditor's standing rule and CLAUDE.md's ground-truth hierarchy
(physics → math → code → **documents**), a markdown section is a consistency
check, never an authority. **Each of these is a provenance defect regardless
of whether the number is right.**

## 2.6 Two things that are properly stamped, and should be the template

- **`validation/reservoir_gain/reservoir_gain.csv`** — 2288 rows, header
  `direction,k,i,j,Te,ne,dlnTe,lnx,G,Sbar,eps,tau_QSS,M,window_ok`, written by
  `verify_reservoir_gain.py`. This backs G1, the result the thesis now
  headlines. **But `chapter7.tex:149-152` correctly flags that no script
  performs the two aggregations** (the 6.3% spread and the 8.44 factor) — the
  rows exist, the summary statistic does not.
- **`make_ch5_figures.py:378-458`** recomputes every plotted quantity from the
  matrix and raises on a mismatch at rtol 10⁻⁹ on every run, and refuses to
  draw `fig:M_vs_eps` unless the quadratic partial comes out negative. That is
  a negative control wired into a figure script. It is the best-engineered
  file in the repo and the pattern the rest should copy.

---

# PART 3 — Extremum scope register

**Every extremum below is stated with the set it is an extremum over.** The
brief names four prior failures of this kind; I confirm all four and add
three.

## 3.1 The four known cases

### (i) The τ_QSS floor — CONFIRMED, and already fixed in Chapter 3

Reproduced today by running `verify_ch3_claims.py`:

```
[FAIL] tau_QSS min   thesis 1.18e-06 s   measured 7.53769e-08 s   rel 9.36e-01
  M at the tau_QSS-min point = 86.77
  window_ok requires M > 900 -> point is OUTSIDE the analysed set
  points with M > 900: 346 of 400
  tau_QSS min over M>900 subset = 1.177240e-06 s
```

- **1.18 µs** is the minimum over the **346-point `M > 900` subset**.
- **75.4 ns** is the minimum over **all 400 points**, at [49,7].
- **46 of 400 points lie below 1.18 µs.**

`chapter3.tex:519-543` **already carries the corrected value** (Eq. `M_range`
gives 75.4 ns) *and* explains the scope correction in the following paragraph.
**Verdict: SUPPORTED as now written.**

**But the script's FAIL is a false alarm, and this matters.**
`verify_ch3_claims.py` hardcodes `1.18e-6` as "the thesis value". The thesis no
longer says that. The script is checking a stale literal, and
`thesis_ready.md` A1 reports the resulting FAIL as though the chapter were
wrong. **A verification script that hardcodes the values it is checking will
silently invert its own verdict when the document is corrected** — which is
what has happened. This is also a CLAUDE.md rule-1 concern (hardcoding).

### (ii) Lyman-α insensitivity of τ_relax — CONFIRMED, fixed in Chapter 6, live in the register

| statement | scope it is true over |
|---|---|
| "τ_relax insensitive to Ly-α trapping to within 1%, even at Θ_P = 0.012" (`thesis_ready.md` PART C) | **the benchmark point [23,5] only** — measured +0.78%, 2.2769→2.2947 ns at D = 20 cm |
| "M falls 158×" (`derivation_07:299-300`) | **the cold corner** — at [0,3], τ_relax goes 8.8842×10⁻⁹ → 2.4289×10⁻⁷ s, **a factor 27.3** |

Both are correct; neither states its regime. **`chapter6.tex:393-399` states
both, with both numbers and both scopes, and is correct.** The unscoped form
survives only in `thesis_ready.md` PART C. **Verdict: SUPPORTED in the
chapters, register defect outstanding.**

### (iii) The M_eff floor — CONFIRMED, and the "13%" is the defect

Reproduced today by running `verifych3_gb.py`:

```
||L_FF^-1|| / (1/|lambda_min(L_FF)|):  min 1.118  median 1.151  max 1.526
worst at [49,0]  Te=10 eV  ne=1e+12
M_eff = tau_QSS / ||L_FF^-1||:
    benchmark 8659   (spectral M = 9982)
    grid min  77.6 at [49,7]   (spectral M min = 86.77)
```

`chapter3.tex:814-822`:

> M_eff = 8659 at the benchmark, M_eff ≥ 77.6 at all 400 grid points … so
> Eq. (qss_condition) **survives non-normality with a margin reduced by 13%**
> and not by orders of magnitude.

- `M_eff ≥ 77.6 over all 400 points` — **SUPPORTED**, genuine grid-wide floor,
  attained at [49,7], which is also where the spectral minimum 86.77 sits. The
  two extrema in the equation **are** co-located; that half is sound.
- **`13%` is a benchmark-point number presented as the grid conclusion.**
  1 − 8659/9982 = 13.3% at [23,5]. The grid-worst degradation is at **[49,0]**,
  where the ratio is 1.526 and the margin is reduced by **34.5%**, not 13%.
  The chapter itself states "at most 53%" two sentences earlier, from the same
  1.526, and then quotes 13% as the conclusion. **Verdict: SCOPE-RESTRICTED.**
  Correct sentence: *the margin is reduced by 13% at the benchmark and by at
  most 35% anywhere on the grid.*

### (iv) The controlled correlations — CONFIRMED basis-dependent

| quantity | value | basis / scope |
|---|---|---|
| raw corr(log M, log ε) | **+0.757** | all 680 pairs |
| bare `exp(13.6/Te)`, no dynamics | **+0.708** | all 680 pairs — the negative control |
| log M explained by (log Te, log ne) | **93%** | all 680 pairs |
| partial, linear control | **+0.274** (all pairs) / **+0.326** (heating only) | `make_ch5_figures.py` |
| partial, quadratic control | **−0.442** | `make_ch5_figures.py` |
| earlier recorded values | +0.33 / −0.16 | `findings_09` §3.1 — **do not reproduce** (`backlog` H8) |
| range over eight scope/basis combinations | **−0.22 to −0.70** | `thesis_ready` A11 — **no script computes eight** |

**The sign flip is robust across every basis tried. The magnitude is not.**
`chapter5.tex:1297-1302` already carries `[UNVERIFIED]` over the eight-basis
claim and states that `make_ch5_figures.py` computes four, not eight.
**Verdict: SUPPORTED for the sign, UNSUPPORTED for any quoted magnitude.**
Quote the sign, and name the basis or say nothing.

## 3.2 Three further extrema whose scope is not stated

### (v) `ε_plateau = 38.69%` — an extremum over a set the thesis then excludes

| scope | pairs | count > 10% | worst |
|---|---|---|---|
| all `window_ok` | 680 | 202 | **0.3868** at [0,4] |
| Te ≥ 2 eV — the defended range | 448 | **45** | **0.1748** at [15,3] |
| Te ≥ 2 eV and ne ≥ 10¹⁴ | 108 | **0** | 0.0717 |

The 38.69% maximum sits at **Te = 1.000 eV**, inside the region
`chapter3.tex:1017-1019` declares out of scope and `backlog` G3 shows is not
quasi-neutrally self-consistent. It is also **edge-truncated**:
`plateau_gridmap.txt` records `sits on a grid EDGE: True`, all eight density
columns peak at the lowest available Te, and d ln ε/d ln Te ≈ −1.27 and still
rising at the edge. The ne direction **is** genuinely interior.

**And the number is a rate, not a number.** ε_plateau is linear in the step:
38.7% / 87.4% / 190.4% at one, two and four grid intervals. "38.7%" means
"38.7% per 4.81% temperature step", and 4.81% exists only because someone chose
50 log-spaced points between 1 and 10 eV. **Verdict: SCOPE-RESTRICTED on three
axes simultaneously** — temperature range, grid edge, and step size.
`chapter5.tex:280-288, 398-408` states all three. Good.

### (vi) `86.5` vs `86.8` — two minima of two different quantities, quoted as one

- `86.77` = min over 400 points of `|λ₁|/|λ₀|` for the **full matrix**, at [49,7]. Reproduced today.
- `86.5` = min over 400 points of `|Re λ|` over the **excited block L_FF** to `|λ₀|`, at [49,7] (`backlog` G4, no artifact).

`chapter5.tex:466-473` says they are "the same stress point and essentially the
same quantity … a consistency check on each other and not two results", then
promotes **86.5** as "the figure that describes the grid".
`chapter4.tex:385` says "the smallest value anywhere is **86.8**".
**Verdict: SCOPE-RESTRICTED / notation defect.** Two chapters lead with
different numbers for what the reader will take to be one quantity. Decide
which, once, and say what it is a minimum of.

### (vii) Chapter 3's grid-wide extrema vs Chapter 3's own scope declaration

`chapter3.tex:1017-1019` declares: *"The results of this chapter are therefore
stated for Te ≳ 2 eV."* Roughly two dozen extremum claims in the same chapter
are stated "across all 400 grid points", and several attain their extremum
**inside the excluded region**:

| claim | line | extremum located at |
|---|---|---|
| b₁ from 76.6 to 2.683×10⁵ | 945-948 | **[0,7]** and [49,0] — [0,7] is Te = 1 eV |
| n_gs/n_ion = 40.4 at [0,0], 20.5 at [0,7] | 965-969 | **both at Te = 1 eV** |
| Θ_P falls to 3×10⁻⁵ | 1011-1015 | **Te = 1 eV edge** |
| κ(L_FF) minimum 1.4821×10³ | (ch4:574) | **[0,0]**, Te = 1.000 eV |

**Verdict: SCOPE-RESTRICTED.** Each number is correct over 400 points. The
chapter must not state a 400-point range and a Te ≥ 2 eV scope in the same
chapter without saying which claims the scope binds.

---

# PART 4 — The claim-to-evidence table

Columns: **claim · file:line · test · artifact · state · alternative
definitions · falsifier named / did it appear · verdict.**

---

## Chapter 1 — Reading the Light from a Divertor

Chapter 1 makes almost no own computations; its risk is citation, not
measurement.

| # | Claim | Line | Test → artifact | State | Alt-def | Falsifier / appeared? | Verdict |
|---|---|---|---|---|---|---|---|
| 1.1 | Target Te spans ~1 to a few tens of eV, majority below 15 eV | 121-124 | literature only; `\cite{Stangeby2023, Stangeby2023}` — **key duplicated** | 📐 | n/a | none named | SUPPORTED, cite defect |
| 1.2 | Te falls below 1 eV over half a metre in strongly detached solutions | 125-126 | `\cite{Lore2022}` | 📐 | n/a | none | SUPPORTED |
| 1.3 | Near-target ne reaches ~10¹⁵ cm⁻³ under partial detachment | 126-128 | `\cite{Krasheninnikov2017}` | 📐 | n/a | none | SUPPORTED |
| 1.4 | **The grid's lower density bound 10¹² cm⁻³ is not sourced** | 143-150 | open `\todo` in text | 💡 | — | — | **UNSUPPORTED, self-flagged.** Every cold-corner extremum in the thesis sits at this unsourced edge |
| 1.5 | ELM pulse 0.75–3 ms; ITER rise 250 µs, decay ~2×; 2–10% of edge stored energy | 554-559 | `\cite{Eich2017}` | 📐 | n/a | none | SUPPORTED |
| 1.6 | ITER ELM frequency ~1 Hz at 15 MA, ∝ I⁻²; the widely quoted 1–2 Hz traces only to a non-peer-reviewed conference paper and is not repeated | 563-572 | `\cite{Kirk2013}` + explicit negative | 📐 | n/a | — | SUPPORTED — **a correctly refused citation, worth advertising** |
| 1.7 | Sawada & Fujimoto (1994) established the boundary-level relaxation structure | 646-655 | `\cite{SawadaFujimoto1994}` | 📐 | n/a | — | SUPPORTED. `findings_10` calls the open `\todo` on this "the single largest unresolved publication risk in Chapter 1" |
| 1.8 | Greenland settled the general validity question 25 years ago | 668-676 | `\cite{Greenland2001a,Greenland2001b}`, direct quotation | 📐 | n/a | — | SUPPORTED — **this is the priority statement the thesis needs and it is present** |

**Chapter 1 verdict.** The framing is honest and the two priority citations
that most threaten the novelty claim are present and quoted, not buried. The
one substantive defect is 1.4: the grid's lower density edge is asserted with
no source, and it is where the headline extremum lives.

---

## Chapter 2 — The Atoms

Claim A. 157 assertions; 26 rows.

| # | Claim | Line | Test → artifact | State | Alt-def | Falsifier / appeared? | Verdict |
|---|---|---|---|---|---|---|---|
| 2.1 | 43 states; n≤8 ℓ-resolved (0–35), n=9–15 bundled (36–42); 50×8 = 400 grid | 174-178, 233-235 | `cr_context.py` state ordering; `L_grid.npy` shape (50,8,43,43) **confirmed today** | ✅ | n/a | shape mismatch would raise | SUPPORTED |
| 2.2 | Process-rate table at [23,5]: ℓ-mix 1.475×10¹¹ s⁻¹ … RR feed 1.095×10¹ s⁻¹; spread 14 decades | 142-148 | stored arrays | ▶️ | n/a | none named | SUPPORTED, no artifact stamp |
| 2.3 | Excitation coverage: 546 CCC + 36 + 36 + 180 + 21 = **819**; K_ion 36 + 1 + 6; RR 43 × Johnson1972 | 471-483 | `src/rates/*.py` headers, `data/**/*meta.csv`, `PROVENANCE.md`; tallies exact and reproducible | ✅ | n/a | a tally mismatch would show | **SUPPORTED — ladder rung 1, the strongest rung in the project** |
| 2.4 | 201 of 819 pairs are Vriens–Smeets at ~20% accuracy against CCC's ~5% | 1077-1079 | metadata | ✅ | n/a | — | SUPPORTED |
| 2.5 | Detailed balance: 1s→2p ratio 9.404800×10⁻² predicted and stored, identical to 7 figures; max deviation 2.48×10⁻⁹ over 819 pairs | 329-336 | Gate A, `validate_gates.py:90-122` | ✅ | n/a | — | **SUPPORTED but TAUTOLOGICAL**, and the chapter says so at 338-340: `K_deexc` is derived from `K_exc`. Correctly labelled |
| 2.6 | Raw CCC cross sections satisfy microscopic reversibility: mean 0.9995, sd 0.0032, 98.7% within 1% | 1062-1065 | `src/parsers/qc_ccc.py:87-168`; `ccc_qc_report.png` | ✅ | n/a | a systematic bias would show as a mean offset; **did not appear** | **SUPPORTED — this is the real detailed-balance evidence and it is external.** Chapter 2 gives it one line at 0.05%; Chapter 4 leads with it correctly |
| 2.7 | 3115 CCC files on disk; the acquisition report records 3117 | 402-407 | direct count | ▶️ | n/a | — | **UNSUPPORTED discrepancy, self-flagged, unreconciled** |
| 2.8 | Δn = 0 transitions excluded on the provider's explicit instruction | 422-430 | provider quotation | ✅ | n/a | — | SUPPORTED — a stated choice with its source |
| 2.9 | With Δn=0 excluded there is no electron route between 2s and 2p | 436-439 | structural | 📐 | n/a | — | SUPPORTED |
| 2.10 | **Proton ℓ-mixing carries ≥ 99.92% of the 2s loss rate over all 400 points**; 99.972% at benchmark; 99.994% (1 eV) to 99.921% (10 eV); density-independent | 944-956 | decomposition of the 2s column | ▶️ | share is a ratio, no norm choice | if the excluded Δn=0 rates were comparable, n=2 would be wrong; **structural argument only, not tested numerically** | SUPPORTED — extremum correctly stated as a floor over 400 points |
| 2.11 | Without ℓ-mixing 2s empties at 6.17×10⁵ s⁻¹; with it 7.79×10⁸ s⁻¹, factor 1262 | 957-959 | hot edge | ▶️ | n/a | — | SUPPORTED |
| 2.12 | 4f fraction of n=4 runs 0.4361–0.4375 vs statistical 14/32; largest departure 0.3% at ne = 10¹² | 209-213 | the solve; grep for `(2l+1)`, `statistical`, `stat_weight` returns **zero hits** | ✅ | five ℓ-weightings move Δ by 0.07% | the 4f dipole-dark lever should have broken it; **did not** | **SUPPORTED — measured, not assumed** |
| 2.13 | The bundled block carries no ℓ-mixing and is *assumed* statistical, untested | 214-216 | — | 💡 | — | — | **UNSUPPORTED, correctly declared.** `verify_bundling_psm20.py` never run |
| 2.14 | Rydberg 13.605693122994 eV vs ionisation potential 13.598434599702 eV; `CHI_H` bound to the former, 0.0534% too large where an IP is meant | 358-385 | source audit | ▶️ | n/a | — | **SUPPORTED — a live defect in four Saha–Boltzmann exponents** (`verifych3_gb.py:57`, `verify_ridge_mechanism.py:224`, `make_ch3_figures.py:66`, `verify_fujimoto_table41.py:205`) |
| 2.15 | Lyman A-values are exact non-relativistic infinite-mass hydrogenic, +0.056–0.060% above Wiese–Fuhr; 6.2684×10⁸ stored vs 6.2649×10⁸ recommended | 680-693 | inversion of the repo's own A's, 1–4 parts in 10⁵ | ✅ | n/a | — | **SUPPORTED, with the right consequence stated: do not cite NIST as the source** |
| 2.16 | Placing the ℓ-resolved A(2p→1s) on an n-resolved element would overestimate Ly-α by 33.3%, and no conservation or sign check would notice | 704-718 | arithmetic; audit finds A[1s,2s] = 0 exactly, A[1s,2p] = 6.2684×10⁸, and no non-zero \|Δℓ\|≠1 entry in 36×36 | ▶️ | n/a | — | SUPPORTED. `claim_hierarchy` calls this "the single highest-risk unaudited item in Claim A"; the 36×36 audit at 721-726 **substantially closes it** |
| 2.17 | ADAS214 eq. 3.14.14 is the isotropic/sphere-centre case, not a slab; a true slab is smaller by a factor approaching 2 (0.529 at τ=1, 0.477–0.484 at τ=100–1000) | 772-781 | read directly from the primary source; integral evaluated | ▶️ | n/a | — | **SUPPORTED — an unrepaired defect, correctly declared.** Affects only Te < 2 eV numbers |
| 2.18 | e = 4.80326×10⁻¹⁰ esu vs CODATA 4.803204713×10⁻¹⁰, entering σ₀ as e², so 0.0023% | 795-798 | audit | ✅ | n/a | — | SUPPORTED, immaterial |
| 2.19 | Doppler width uses the **proton** mass 1.67262×10⁻²⁴ g where the neutral atom 1.6735328407×10⁻²⁴ g is meant; line 0.027% too wide | 798-802 | audit | ✅ | n/a | — | SUPPORTED — wrong object, immaterial number |
| 2.20 | µ = m_H/m_e = 918.08 in eq. `psm20` | 890 | — | ▶️ | — | — | **Numeral SUPPORTED, label REFUTED.** 918.08 = m_p/2m_e is the correct reduced mass for proton–hydrogen; the text's own line 834 says "1836 times heavier". The *definition text* is wrong, not the value. A reader checking it will think there is a factor-2 error under a square root |
| 2.21 | Debye length in U_m frozen at 10¹⁴ cm⁻³; "F varies by about 30% across the grid" | 921-924 | — | ▶️ | — | — | **REFUTED.** `backlog` G6: evaluating the module's own functions, q spans **×3.7 for n=2 and ×50 for n=8**. The documented justification is wrong by up to a factor 50. The result is insensitive anyway (f₃−f₄ moves <0.3%), so the headline is unaffected — but the sentence is false |
| 2.22 | Anderson RMPS: 340 comparisons, 42.4% within 20%, mean \|err\| 29.0%; n_up ≤ 4 → 82.1% / 12.7%; n=5 only → 14.5% / 40.4% | 1018-1032 | `collisions/anderson_benchmark_full.csv` | ✅ | n/a | if the disagreement were spread across all n the data would be unusable; **it is confined to n=5, 200 of 340** | **SUPPORTED — an honest external benchmark with the failure localised** |
| 2.23 | Anderson could not include n=6 (would need a 140 a₀ box); the direction of the n=5 error is not established and is not claimed | 1048-1052 | literature | ✅ | n/a | — | SUPPORTED — correct refusal |
| 2.24 | ℓ-mixing F(U_m) error: old code set F = 1 justifying it by an inverted limit; correct F = 6.95 at U_m = 1.05×10⁻³; old code under-counted every rate by 3–7×; effect on τ_relax bounded at 0.85% | 1098-1128 | stored q vs independent evaluation, agreeing to 8 parts in 10⁶ | ✅ | n/a | a >1% shift in τ_relax would have mattered; **did not appear** | **SUPPORTED — the model example of this project's standard: bug found, traced to Badnell 2021 Eq. 9, sized, headline proven insensitive** |
| 2.25 | Eigenvalue filter `eigs[eigs < -1.0]`: 19 of 400 points, Te ≤ 1.389 eV; at each, τ_relax^new = τ_QSS^old **exactly**; M range 1.34–1.01×10⁸ → 86.77–1.72928×10⁹; benchmark bit-identical | 1143-1164 | `*_FILTERED_20260721` artifacts preserved | ✅ | n/a | if the shift were not exactly one rung the diagnosis would be wrong; **it was exact at all 19** | **SUPPORTED — the falsifier was sharp and absent** |
| 2.26 | The literal is still live in `solve_cr.py:269` and `check_mz.py:10` | 1172-1174 | grep | ✅ | n/a | — | SUPPORTED. **Note: CLAUDE.md's known-issue table names `qss_analysis.py`, which now reads `eigs < 0.0`. Fix the table, not the code** |

**Chapter 2 verdict.** The strongest chapter in the thesis for provenance.
Rung 1 of the validation ladder passes outright. Six separate self-found
defects (2.7, 2.14, 2.17, 2.19, 2.20, 2.21, 2.24, 2.25) are declared in the
text rather than hidden. Two REFUTED items (2.20 label, 2.21 justification),
both immaterial to any result but both false as written. One date conflict:
`chapter2.tex:1113-1115` says `L_grid.npy` was written 21 July 2026; CLAUDE.md
says regenerated 14 July 2026.

---

## Chapter 3 — Two Clocks

Claims B, C, D, F.1, F.2. **No number in this chapter names its producer.**
160 assertions; 34 rows.

| # | Claim | Line | Test → artifact | State | Alt-def | Falsifier / appeared? | Verdict |
|---|---|---|---|---|---|---|---|
| 3.1 | Column sums obey Σ_p L_pq = −Q_q n_e | 218 | derivation | 📐 | n/a | — | SUPPORTED |
| 3.2 | **"An error in a single off-diagonal element, a transposed index, or a missing back-reaction all break this immediately"** | 222-225 | fault injection, Part 1.1 | — | — | — | **REFUTED.** Transposed array and deleted back-reaction both left the residual unchanged |
| 3.3 | Largest relative column-sum residual = 3.61×10⁻¹¹, **"evaluated at all 400 grid points"** | 267-272 | **I recomputed this today over all 400 points: 3.6078×10⁻¹¹ at [0,2].** A verification agent independently got 3.607812×10⁻¹¹ | ▶️ | n/a | — | **Number SUPPORTED (now twice, independently). Scope UNSUPPORTED:** the only committed check samples 20 of 400. The claim is true; nothing in the repo computed it |
| 3.4 | ℓ-mixing cancels exactly in the column sum and is invisible to this test | 282-284 | structural | 📐 | n/a | — | **SUPPORTED — and this is the sentence that should have prevented 3.2** |
| 3.5 | Spectrum at benchmark: λ₀ = −4.3999×10⁴, λ₁ = −4.3919×10⁸; gap 9982; largest of the other 41 gaps 2.80; **isolation 3567** | 342-356 | eigendecomposition | ▶️ | Framing A vs B agree to 0.0098% | an intermediate mode would destroy the picture; **none: one gap then a continuum** | SUPPORTED |
| 3.6 | Isolation minimum over 400 points = **24.3** at [49,7], where the first gap is 86.8 and its competitor 3.57; 147× less margin than the figure displays | 552-560 | reproduced today: `isolation grid minimum 24.3286` | ▶️ | n/a | — | **SUPPORTED — and the chapter states the scope correctly.** Note the guard is `if iso.min() < 10.0`, a threshold **2.4× slacker** than the measured minimum, with no source for 10.0 |
| 3.7 | Slow eigenvector: 1s component 1.0000, every excited component < 10⁻³; PR(v₀) = 1.00 | 388-396 | eigenvector | ▶️ | **two norms in play** | — | **SCOPE-RESTRICTED.** `backlog` C2: v₁'s ground weight is 1.0000 raw and 1.9×10⁻⁴ population-scaled; adjacent sentences use different norms without naming either |
| 3.8 | v₁ is "distributed across the excited manifold" | 444-446 | — | — | — | — | **REFUTED.** `backlog` C2 measures **PR(v₁) = 2.64** at the benchmark and 1.88 at [49,7]. PR ≈ 2 is not "distributed". The chapter quotes PR for v₀ only, where it carries no information |
| 3.9 | η ∈ [1.16, 29.5] across the grid, 3.12 at benchmark, nowhere below unity | 411-412 | none named | ▶️ | n/a | η < 1 would contradict the M-matrix argument; absent | SUPPORTED, **no artifact** |
| 3.10 | τ_QSS = 22.73 µs, τ_relax = 2.277 ns, M = 9982 at [23,5] | 441-454 | three independent implementations bit-for-bit; reproduced today (`M = 9981.93`) | ✅ | Framing A vs B 0.0098% | — | **SUPPORTED — reproduces CLAUDE.md to four digits** |
| 3.11 | M⁻ = 9982, M⁺ = 8243 one grid step up; d ln M/d ln Te ≈ −4, so a 4.8% step moves M by 17% | 493-497 | reproduced today: 9981.93 / 8242.74 / −4.074 | ✅ | n/a | — | SUPPORTED |
| 3.12 | The "+0.6 eV" step is four intervals, Te[23]→Te[27], factor 1.2068, M⁺ = 4856 | 507-509 | reproduced: 4856.3 | ✅ | n/a | — | **SUPPORTED, but note `verify_ch3_claims.py` reports the +0.6 eV point is *not* on the grid** (nearest Te[27] is +0.266% away); the chapter's four-interval reading resolves this and should be the one used everywhere |
| 3.13 | Eq. `M_range`: 0.87 ns ≤ τ_relax ≤ 38.9 ns; **75.4 ns** ≤ τ_QSS ≤ 67.2 s; 86.8 ≤ M ≤ 1.73×10⁹ | 519-526 | reproduced today, 5 of 6 bounds PASS | ▶️ | n/a | — | **SUPPORTED — Part 3.1(i)** |
| 3.14 | "τ_QSS varies by **seven** orders of magnitude" | 528-529 | — | — | — | — | **REFUTED, and it is a stale-value artifact.** 67.2 s / 75.4 ns = 8.9×10⁸, i.e. **nine** orders. "Seven" is consistent with the superseded 1.18 µs floor that the very next paragraph retires |
| 3.15 | M = 86.8 is the minimum over 400; that point is excluded by `window_ok`; the floor over the analysed set is M ≥ 902 and **is imposed by the criterion, not measured** | 534-539 | `window_ok ≡ M > 900` | ✅ | n/a | — | **SUPPORTED — an exemplary scope statement. This is how every extremum in the thesis should read** |
| 3.16 | The 1.18 µs floor is over 346 points, not 400; over 400 it is 75.4 ns at the same [49,7] corner | 539-543 | reproduced: 1.177240×10⁻⁶ over the M>900 subset | ✅ | n/a | — | SUPPORTED |
| 3.17 | Minimum τ_relax, τ_QSS, M and isolation all occur at [49,7] | 562-565 | reproduced: `all coincide at one point: True` | ✅ | n/a | — | **SUPPORTED — verified, not assumed** |
| 3.18 | d ln τ_relax/d ln ne between −0.46 and −0.53, not −1 | 573-576 | fit | ▶️ | fitted slope, range given | — | SUPPORTED — supersedes the stale `ne^-1.00` in CLAUDE.md's known-issues |
| 3.19 | λ₁(L) and λ_min(L_FF) agree to 0.0098% at benchmark and better than 0.346% grid-wide | 589-590 | reproduced today: 0.0098% and 0.3461% | ✅ | n/a | — | SUPPORTED |
| 3.20 | M × (relative disagreement) runs 0.074 to 7.94, median 0.494 — so the agreement **measures the gap** rather than evidencing separability independently | 605-608 | reproduced: min 0.07418, median 0.4937, max 7.937 | ✅ | n/a | if the product were not O(1) the agreement would be independent evidence; **it is O(1)** | **SUPPORTED — a self-administered demotion of the project's own evidence. Exemplary** |
| 3.21 | Radiative decay ~6.3×10⁸ s⁻¹ vs collisional excitation ~10⁴ s⁻¹, **"some five orders of magnitude"** | 630-633 | arithmetic | ✅ | n/a | — | SUPPORTED (6.3×10⁴ ≈ 4.8 decades) |
| 3.22 | The same two rates differ by **"some nine orders of magnitude"** | 1535-1539 | arithmetic | — | — | — | **REFUTED, and it contradicts 3.21 in the same chapter with the identical two numbers** |
| 3.23 | κ(L_FF) ∈ [1.48×10³, 1.74×10⁵] across the grid, "so the inversion is numerically well behaved" | 719-722 | **no producing script anywhere in the repo** | 💡 | n/a | — | **UNSUPPORTED. Correct — `findings_10` C.1 measures 1.4821×10³ at [0,0], 1.7436×10⁵ at [33,7], median 2.1044×10⁴ — but no committed code produces it.** No grid point is named for either endpoint, and "well behaved" is an unquantified judgement attached to κ = 1.7×10⁵ |
| 3.24 | The ill-conditioning "lives in the fast subspace … and barely touches the slowest mode" | 822-827 | **none** | 💡 | — | — | **UNSUPPORTED.** A mechanism asserted as "the answer is", resting on 3.23, with no computation attached |
| 3.25 | ‖L_FF⁻¹‖₂·\|λ_min\| ∈ [1.118, 1.526], median 1.151; spectral estimate understates by at most 53% | 809-812 | reproduced today, exactly | ✅ | 2-norm; other norms differ | ratio < 1 would contradict σ_min ≤ min\|λ\|; **did not occur** | SUPPORTED |
| 3.26 | M_eff = 8659 at benchmark, ≥ 77.6 at all 400 points; "margin reduced by **13%**" | 814-822 | reproduced: 8659, 77.6 at [49,7] | ▶️ | n/a | — | **Floor SUPPORTED; 13% SCOPE-RESTRICTED — Part 3.1(iii). Grid-worst is 34.5% at [49,0]** |
| 3.27 | Direct integration gives a residual ~10⁻⁸ against the scaling argument's 1.21×10⁻⁴ — four orders better than required | 841-847 | integration | ✅ | n/a | if the closure were the culprit the residual would be O(ε); **it is 10⁻⁸** | **SUPPORTED — this is the thesis** |
| 3.28 | b₁ runs 76.6 at [0,7] to 2.683×10⁵ at [49,0], 956 at benchmark; ground over-populated by 2–5 decades; **"varies over three decades"** | 945-955 | none named | ▶️ | Saha normalisation | — | Values SUPPORTED, **no artifact**; "three decades" is **REFUTED** — 76.6 → 2.683×10⁵ is **3.54** decades |
| 3.29 | Solving directly, n_gs/n_ion falls from 40.4 at [0,0] and 20.5 at [0,7] — 95–98% neutral — to 5.3×10⁻⁶ at [49,7]; a large b₁ does not mean many neutrals, and the two peak at opposite corners | 965-971 | none named | ▶️ | n/a | — | **SUPPORTED and important — no artifact** |
| 3.30 | −L_FF⁻¹ has no negative entry at any of the 400 points | 1186-1187 | none named | ▶️ | n/a | a negative entry would break the M-matrix argument; **none at 400 points, min component +6.019×10⁻¹³** | SUPPORTED, no artifact |
| 3.31 | **d ln R/d ln b₁ = f₃ − f₄** — the mathematical core | 1191-1215 | superposition to 3.075×10⁻¹⁴ over 784 pairs; the ∫ identity to 3.3×10⁻¹⁶ | ✅ | photon vs energy weighting 6.7×10⁻¹⁶; shell vs A-weighted 0.06% | a 3↔4 swap must be caught; **it is** | **SUPPORTED. The chapter correctly notes the identity check validates implementation, not physics** |
| 3.32 | max\|f₃−f₄\| = tanh(\|Δ\|/4), attained at the geometric mean of the switching points; hence \|d ln R/d ln b₁\| < 1 always | 1268-1298 | closed form; Part 1.2 | ✅ | Δ invariant under normalisation (exact under 10¹⁰ and 10⁻⁷) and to 0.07% across five ℓ-weightings | — | **SUPPORTED as a theorem. Its gate is a wiring check — Part 1.2** |
| 3.33 | At the benchmark f₃ = 0.260, f₄ = 0.048; sensitivity 0.212, 47% of the maximum 0.4514, attained nearly a decade higher at b₁ = 7.2×10³ | 1255-1259, 1320-1323 | none named; arithmetic self-consistent (Δ = 1.950 vs the 1.945614 I measured) | ▶️ | n/a | — | SUPPORTED, no artifact |
| 3.34 | Gate D disagrees at every grid point by a systematic factor between 9 and 65 peaking near Te ≈ 2 eV; only the ionisation half is implemented; **no agreement with ADAS is claimed anywhere** | 1396-1410 | `validation/gate_summary.txt` | ▶️ | n/a | — | **SUPPORTED — an honest failure reported as failing. Do not repair.** This supersedes the ADAS promise `claim_hierarchy` flagged at `chapter3.tex:1377-1387`, which has been rewritten |

**Chapter 3 verdict.** The mathematics is the strongest part of the thesis and
several scope statements (3.15, 3.17, 3.20) are model examples. Against that:
**four REFUTED sentences** (3.2, 3.8, 3.14, 3.22, 3.28's "three decades"), two
of which are internal arithmetic contradictions with the chapter's own numbers,
and **not one number in 1641 lines names its producing script.**

---

## Chapter 4 — Does the Model Work?

Claim A (A.3, A.4, A.6–A.9) and the validation ladder.

> **Chapter 4 is incomplete.** It runs to line 860 and stops mid-sentence
> ("…the spectrum says which is which"), with no summary and no closing
> section. Four labels are forward-referenced and never defined:
> `sec:fujimoto`, `sec:errors_found`, `sec:validation_summary`, `sec:gate_d`.
> **The chapter contains zero `\cite{}` keys** while making external
> comparisons to RMPS, Bray's CCC data, Fujimoto Table 4.1 and "published
> values". The Fujimoto external gate — `thesis_ready` A2, the strongest
> external evidence in the project — is promised at line 8 and never written.

| # | Claim | Line | Test → artifact | State | Alt-def | Falsifier / appeared? | Verdict |
|---|---|---|---|---|---|---|---|
| 4.1 | Conservation residual 3.61×10⁻¹¹ | 161-164 | Part 1.1; reproduced twice today | ▶️ | n/a | — | SUPPORTED as a number; **contradicts its own Table at line 192, which gives a 2.238×10⁻¹² baseline. Neither is wrong — they are different scopes (grid max at [0,2] vs one unnamed point) — and the chapter does not say so** |
| 4.2 | "That claim is testable, and it is false" (of Chapter 3's conservation claim) | 168 | fault injection | ✅ | n/a | — | **SUPPORTED. Chapter 4 refutes Chapter 3 and is right** |
| 4.3 | Fault-injection table: three faults up to ΔL = 2.7×10¹¹ s⁻¹ leave the residual unchanged; A halved with γ unchanged → 36.3, caught | 182-224 | `CHANGE_REPORT.md` §4.2 — **a markdown file**; independently reproduced today | ✅ | n/a | — | **SUPPORTED. Confirmed by two independent runs. Artifact is markdown only** |
| 4.4 | The grid point of the fault-injection residuals is not recorded | 234-237 | — | — | — | — | **UNSUPPORTED, self-flagged `\todo`. This is what makes 4.1's apparent contradiction unresolvable** |
| 4.5 | Gate A: max relative error < 0.01% over 819 pairs × 3 temperatures. **Grade: tautological** | 266-275 | `validate_gates.py:118`; `compute_K_CCC.py` `detailed_balance` | ✅ | n/a | can only catch a wiring error | **SUPPORTED and correctly graded** |
| 4.6 | Raw CCC reversibility: mean 0.9995, sd 0.0032, 98.7% within 1% — a 0.05% bias with 0.32% scatter on external data | 280-285 | `qc_ccc.py:87-168`, `ccc_qc_report.png` | ✅ | n/a | a systematic bias would appear as a mean offset; **did not** | **SUPPORTED — genuine external check. Chapter 4 correctly leads with this rather than Gate A. This closes `derivation_02:206`'s open "~0.43% (or 0.05%?)" — it is 0.05% mean / 0.32% scatter, and neither reading was 0.43%** |
| 4.7 | Saha ratio 0.999997 for every one of 43 states at every temperature, **to the same eight digits**; state-independent across Boltzmann factors spanning e^{13.6/kT} to e^{0.06/kT}. **Grade: severe** | 300-309 | `findings_10` ADDENDUM D.6 — **markdown only, no script** | ▶️ | n/a | two energy ladders would show as a drift across the ladder; **there is none** | **Reasoning SUPPORTED and genuinely severe — the state-independence is the evidence, not the closeness to unity. Provenance UNSUPPORTED** |
| 4.8 | 2457 excitation-pair comparisons, max deviation 7.31×10⁻⁹, median 7.5×10⁻¹⁰ | 311-313 | `findings_10` D.6 — markdown only | ▶️ | n/a | — | UNSUPPORTED provenance; **the chapter's own comment at 314-316 flags that 819 × 3 = 2457 pairs vs "2457 pairs × 3 temperatures" = 7371 comparisons is unresolved** |
| 4.9 | Gate B: 50/50 temperatures pass, worst convergence 0.0034 against a 50% tolerance — a factor 150 in hand | 327-334 | `validate_gates.py:175,181` | ✅ | n/a | — | **SUPPORTED, and correctly described as rejecting only a gross error** |
| 4.10 | Gate C pass flag uses monotone and below_saha and **excludes** `approaching`; approach fraction 0.0123–0.0155, so populations reach ~1.5% of Saha; LTE never approached. **Grade: weak, and mis-named** | 348-364 | `validate_gates.py:250`; `validation/gate_C.csv` — **confirmed today, values read from the file** | ✅ | n/a | — | **SUPPORTED and correctly graded — Part 1.3** |
| 4.11 | Gate E: M ranges 86.77 at [49,7] to 1.72928×10⁹; τ_QSS 75.4 ns to 67.2 s; τ_relax 0.87–38.9 ns. **Grade: tautological as posed** | 379-390 | `thesis_ready` A1; reproduced today | ✅ | n/a | a 43-state stiff system with A ~ 10⁸ and ionisation ~ 10⁴ cannot have M ≤ 1 | **SUPPORTED, correctly graded** |
| 4.12 | **The script returns exit code zero whether or not Gate D fails. No gate raises on failure** | 408-412 | code audit | ✅ | n/a | — | **SUPPORTED — a serious and correctly disclosed defect** |
| 4.13 | Superposition residual 3.075×10⁻¹⁴ over all 784 pairs. **Grade: severe** — one of only two checks known to catch a 3↔4 swap | 437-446 | `validation/plateau_gridmap/plateau_gridmap.txt` line 19 — **a real stamped artifact** | ✅ | n/a | a 3↔4 swap must be caught; **it is** | **SUPPORTED. Exact by construction, so it validates implementation — but severe because it catches the swap. The chapter says both** |
| 4.14 | Reduced-vs-full R: an independent re-implementation got 0.777768 by three routes against `chapter3.tex`'s 0.7778 | 453-457 | `findings_10` B.1 — markdown | ▶️ | n/a | a 3↔4 swap breaks the agreement | Reasoning SUPPORTED, provenance UNSUPPORTED |
| 4.15 | `L_grid.npy` → `2d92b58e1693107d`, `S_grid.npy` → `7822f536590c76ba`, both matching `preflight.py:37-38` | 470-480 | pinned hashes | ✅ | n/a | a stale array is the failure mode with three retractions behind it | **SUPPORTED — the right defence against this repo's characteristic failure** |
| 4.16 | `preflight.py` names six scripts, two of which do not exist on disk; the miss is downgraded to a non-blocking advisory while the summary prints "All checks passed" | 483-489 | `preflight.py:258-262` | ✅ | n/a | — | **SUPPORTED. Per CLAUDE.md rule 2 this must raise** |
| 4.17 | `make_ch5_figures.py` rebuilds every plotted quantity from L and S and raises at rtol 10⁻⁹ with the offending point, column and hash | 493-498 | `make_ch5_figures.py:393-432` | ✅ | n/a | — | **SUPPORTED — the template** |
| 4.18 | It refuses to draw the figure unless the controlled correlation changes sign | 504-508 | same | ✅ | n/a | — | **SUPPORTED — a negative control enforced in code** |
| 4.19 | The σ₀ cross-check **failed at 1156% on first run** and located a 4π CGS/SI error; after repair the two agree to 0.059% against a 1% raising tolerance | 517-522 | `escape_factor.py` | ✅ | n/a | — | **SUPPORTED — a gate that has actually fired. Worth more at a defence than ten that never have** |
| 4.20 | With proton ℓ-mixing removed, `verify_fujimoto_table41.py` returns r₀(2) = 104.5 and flags it itself as unphysical; production gives 0.7392 | 524-529 | the script | ✅ | n/a | — | **SUPPORTED — a check that fires on a known-bad input is the definition of severe** |
| 4.21 | The semigroup propagator check returns 0.000×10⁰ by construction and was **withdrawn by the author** | 531-536 | — | ✅ | n/a | — | **SUPPORTED — withdrawing your own check is evidence of judgement; report it as such** |
| 4.22 | The tanh bound "reads as validation. It is not." Fed corrupted input it still passes. **Grade: tautological** | 540-552 | Part 1.2 | ✅ | n/a | — | **Conclusion SUPPORTED. The c₃↔c₄ justification at 548-550 is REFUTED — measured \|Δ\| moves 1.945614 → 0.939493** |
| 4.23 | κ(L_FF): min 1.4821×10³ at [0,0], max 1.7436×10⁵ at [33,7], median 2.1044×10⁴, over 400 points | 572-576 | `findings_10` C.1 — markdown | ▶️ | n/a | — | **UNSUPPORTED provenance — Part 2.2** |
| 4.24 | **[UNVERIFIED: no committed script produces the grid-wide condition number]** | 580-587 | — | — | — | — | **SUPPORTED, and the chapter is right where `backlog` H11 is wrong** |
| 4.25 | Eigenvalue condition numbers are O(1) — 1.71/1.98 at benchmark, 3.26/2.59 at the cold corner — despite ‖L‖₁ = 2.0×10¹⁴; single precision moves the channels by 1.3×10⁻⁵; rtol 10⁻⁹ vs 10⁻¹² shifts by 1.20×10⁻⁸; `expm` vs Krylov by 1.13×10⁻¹⁰ | 591-601 | `findings_10` ADDENDUM B — markdown | ▶️ | n/a | — | **SCOPE-RESTRICTED and correctly declared at 603-606: "three of 400 points; the tolerance comparisons cover two. They are point measurements, not grid scans"** |
| 4.26 | Truncation increments decay geometrically with ratio 0.75; extrapolated error +0.9% at benchmark, −1.4% at cold corner, −0.55% at ridge | 615-620 | `findings_10` D.8 — markdown | ▶️ | n/a | — | **SUPPORTED as a number. Directly contradicts `chapter5.tex:969-975`, which says the n_max sensitivity "has not been tested"** |
| 4.27 | A truncation test that drops states without removing their loss channels produces a spurious 22–37% jump that looks exactly like a headline finding | 626-635 | — | ✅ | n/a | — | **SUPPORTED — a self-caught artifact trap, and the most valuable methodological sentence in the chapter** |
| 4.28 | Terminal-shell excess measured internally: 9.4× at p=8, 10.7× at p=9, 6.3× at p=10; the external 4.9–6.2× at p=15 falls inside that band | 637-643 | none named | ▶️ | n/a | — | **SUPPORTED — an internal measurement is stronger than the external inference it replaces.** `chapter6.tex:916-918` explicitly flags that "their run artifact is not stamped in `validation/`" |
| 4.29 | 4f fraction 0.4361–0.4375 vs 14/32; ε(line)/ε(shell) runs 0.978–0.9999, **worst case 2.2% at the lowest density**; at [0,4] the two agree to 0.06% | 688-701 | `thesis_ready` A6 | ✅ | shell vs A-weighted; photon vs energy weighting 6.7×10⁻¹⁶ | the 4f dipole-dark lever should break it; **did not** | **SUPPORTED, and the chapter states the scope explicitly: "That 0.06% is a single-point figure and must not be quoted as a grid-wide one." Exemplary** |
| 4.30 | Closure attack: falsifier τ_QSS/τ_drive ≈ 2300 named **before** the test; measured factor 15; count 202→200 of 680, worst 0.38682→0.38563; above 2 eV, 45 before and 45 after | 742-798 | `findings_10` §3.1 — markdown | ✅ | closure factor min 1.00, median 1.00, max 41.4 | **2300 named in advance; measured 15. The falsifier did not appear** | **SUPPORTED — a preregistered falsifier, survived by two orders of magnitude. Caveat correctly carried: the closure adds an ion reservoir, does not close the electron balance, and adds no transport** |
| 4.31 | The ELM lower bound reproduces the true average to better than 0.7% everywhere tested; the sub-unity value at [0,4] is a trapezoid artifact on a 25 ns mesh | 826-847 | `findings_10` §3.2 — markdown; table lists **5** points | ✅ | n/a | W3 asserted it is not a bound; §3.2 then showed near-exactness | **SUPPORTED — say both: it is not a *guaranteed* bound, and it is empirically accurate to 0.7%.** `chapter5.tex:1152` says "seven points"; the Chapter 4 table lists five |

**Chapter 4 verdict.** Where written, this is the most rigorous chapter in the
thesis: it grades its own gates *tautological*, *weak*, and *severe*, discloses
that no gate raises on failure, and refutes Chapter 3. Its defects are
structural rather than argumentative — **it is unfinished**, it cites nothing,
and the external Fujimoto comparison that is the project's strongest evidence
is promised and absent.

---

## Chapter 5 — What the Model Says

Claims E, F.3, F.4, G, ¬2. 134 assertions; 38 rows.

| # | Claim | Line | Test → artifact | State | Alt-def | Falsifier / appeared? | Verdict |
|---|---|---|---|---|---|---|---|
| 5.1 | **ε_QSS = 8.66×10⁻⁶ at benchmark and 6.73×10⁻⁹ at [0,4] against ε_CRE = 6.34×10⁻² and 3.869×10⁻¹** | 227-238 | `findings_09` §1, direct integration | ✅ | measured along the trajectory, not at endpoints | if the closure were the culprit the residual would be O(ε); **it is 10⁻⁸ at the worst point on the grid** | **SUPPORTED. This four-to-eight-decade gap is the thesis and it is the best-evidenced claim in it** |
| 5.2 | Spectral gap at L[1,4]: −4.29, −1.60×10⁸, −3.44×10⁸ s⁻¹, gap 3.7×10⁷ | 240-245 | `findings_10` §3.2 | ✅ | n/a | an intermediate mode would spoil the closure; **none** | SUPPORTED |
| 5.3 | ε_plateau > ε_step at **680 of 680** pairs, median ratio ~12, minimum 1.45 | 192-195 | `verify_plateau_gridmap.py` → `plateau_gridmap.csv/.txt` — **stamped** | ▶️ | n/a | one counterexample would refute it; **none in 680** | SUPPORTED |
| 5.4 | Do not quote the 1271× amplification maximum: it sits at the point of smallest ε_step, where ε_plateau is below the median, and log(amp) = log ε_plateau − log ε_step identically | 199-206 | same | ✅ | n/a | — | **SUPPORTED — a correctly refused extremum** |
| 5.5 | Superposition residual 3.075×10⁻¹⁴ grid-wide | 262-266 | `plateau_gridmap.txt` line 19 | ✅ | n/a | 3↔4 swap caught | SUPPORTED |
| 5.6 | **Global maximum ε_plateau = 38.69% per +4.81% step at [0,4], Te = 1.000 eV, ne = 5.18×10¹³ — and it sits on a grid edge** | 280-288 | `plateau_gridmap.txt` lines 41-43, `sits on a grid EDGE: True` | ▶️ | linear in step size | **no prediction was ever stated for this quantity** | **SCOPE-RESTRICTED on three axes — Part 3.2(v). This is R6, the one broken chain in the thesis** |
| 5.7 | Defended-scope maximum 18.07% at [15,3], Te = 2.02 eV | 289-294 | `verify_eps_gridmap.py` — **writes nothing** | ▶️ | n/a | — | SUPPORTED, **no artifact** |
| 5.8 | Heat/cool asymmetry is a step-size artifact; matched and normalised the median ratio is 1.036 | 300-304 | `thesis_ready` A8 | ✅ | n/a | — | **SUPPORTED — an apparent asymmetry correctly dissolved** |
| 5.9 | Crest width: across the peak column and its neighbours, a factor 7.2 in density, ε varies 6–22%; over three decades the row falls 2.2–6.6× | 325-329 | `make_ch5_figures.py`, injected into the caption | ▶️ | n/a | **if it did not look like a ridge the word must not be used** (`plan_to_submission` T1.3) | SUPPORTED |
| 5.10 | Therefore quote the density of maximum error as the **range 7×10¹² to 5×10¹³ cm⁻³** | 334-336 | grid spacing 2.683 per column | ✅ | parabolic vertex shifts 15% | — | **SUPPORTED — a correctly widened extremum** |
| 5.11 | The 5.18×10¹³ column carries the row maximum in **3 of 49** heating rows; per-row argmax over rows 0-15 is [4,4,4,3,3,…] | 336-342 | `findings_10` §1.2 | ✅ | cooling moves it to [1,3] | "at every temperature" is false at 3 of 49 rows | **SUPPORTED — this is `backlog` H2, a correction to a previously published table, correctly carried into the chapter** |
| 5.12 | Parabolic vertex at 1.65×10¹³, a 15% shift — **[UNVERIFIED: prose only, no script, no artifact]** | 343-351 | — | 💡 | — | — | **UNSUPPORTED, self-flagged. And it conflicts with 5.29 below** |
| 5.13 | An Hα/Hγ (3,5) diagnostic has its worst density at 7.2×10¹², a factor 2.68 lower | 354-357 | `findings_10` §7.3 | ▶️ | n/a | — | **SUPPORTED — the ridge belongs to the (3,4) pair, not "the Balmer diagnostic"** |
| 5.14 | Step-dependence: 38.7% / 87.4% / 190.4% at 1, 2, 4 intervals; a factor 4.3 in step moves the error by 4.9 | 398-406 | `verify_eps_gridmap.py` via `CH5_EVIDENCE.md` §1 | ▶️ | — | — | SUPPORTED, no artifact |
| 5.15 | **"the QSS error reaches 39%" is not falsifiable because it does not say what it is 39% of** | 406-408 | — | ✅ | — | — | **SUPPORTED — the chapter diagnoses its own headline. This is `backlog` §J** |
| 5.16 | 2s: two-photon 8.229 s⁻¹ is still 553× faster than \|λ₀\| at the cold corner; 2s would be a reservoir only below ne ≈ 7×10³ cm⁻³ | 452-459 | **no script, no artifact** | ▶️ | n/a | this was the one open question that could have changed the shape of the two-channel decomposition; **it did not** | SUPPORTED reasoning, **UNSUPPORTED provenance** |
| 5.17 | Minimum ratio of \|Re λ\| over L_FF to \|λ₀\| is **86.5** over 400 points; "**that is the figure that describes the grid**" | 461-473 | **no script, no artifact** | ▶️ | n/a | — | **SCOPE-RESTRICTED — Part 3.2(vi). Conflicts with `chapter4.tex:385`'s 86.8** |
| 5.18 | **[SOURCE REQUIRED: the four measurements in this subsection … the script and its stamped artifact must be named before submission]** | 475-479 | — | — | — | — | **SUPPORTED self-flag — the correct response to 5.16 and 5.17** |
| 5.19 | Freezing n_ion costs at most 0.09%: total bound population per unit n_ion is 8.925×10⁻⁴ | 507-510 | `findings_10` §B.4 | ▶️ | n/a | — | SUPPORTED, bounded |
| 5.20 | ln(R_PE/R_CRE) = S̄ Δln u, exactly | 518-525 | derivation | ✅ | n/a | — | **SUPPORTED, and the chapter states at 662-669 that comparing the two sides of an identity tests nothing** |
| 5.21 | Gain table: −5.736/−5.634/−5.439 at benchmark, spreads 5–6% — "**no script in the repository computes them and no artifact stores them**" | 592-610 | `pivot_decision.md` §4 | ▶️ | — | — | **UNSUPPORTED, self-flagged** |
| 5.22 | **[UNVERIFIED: the k=2 and k=4 columns, which are the entire basis of the step-independence claim, have no artifact behind them]** | 617-623 | — | — | — | — | **UNSUPPORTED, self-flagged — and this is the basis of G1, the result the thesis now headlines** |
| 5.23 | Worked arithmetic: 0.2288 × 5.736 × 0.046984 = 0.06166, e^0.06166 − 1 = 0.0636 against 0.063612 | 627-640 | `findings_10` B.2; "**the arithmetic is performed here in the text and is not itself a run**" | ▶️ | n/a | — | **SUPPORTED, with an unusually honest disclosure** |
| 5.24 | The linearisation understates in only **219 of 392** heating steps and **over**states at the benchmark (0.9459); ratio runs 0.8417–1.3891, median 1.0310 | 653-659 | `verify_plateau_gridmap.py:212` | ✅ | endpoint vs kernel form named | the previously published "understates one-sidedly" claim | **SUPPORTED — a prior claim of this project correctly retracted in the text** |
| 5.25 | \|S̄\| ≤ tanh(\|Δ\|/4) < 1; width unity because the populations are exactly affine in n_gs | 686-699 | `findings_10` B.1 | ✅ | Δ invariant under normalisation and to 0.07% over five ℓ-weightings; **I measured Δ = 1.945614, matching 1.94561** | — | **SUPPORTED** |
| 5.26 | Bound tightness: 51% of the cap at the benchmark, **97% at [23,3]** | 716-721 | `findings_10` B.2 | ▶️ | n/a | — | SUPPORTED |
| 5.27 | ε_step passes through a minimum of 4.9×10⁻⁵ near 6.87 eV: **the n₃/n₄ ratio is locally non-invertible for temperature at that density** | 738-753 | `verify_plateau_gridmap.py` | ▶️ | n/a | — | **SUPPORTED — and this is a genuinely useful result that is currently undersold** |
| 5.28 | Attribution: the column maximising ε is the column maximising \|S̄\| in **32 of 32** rows (k=1) and 31 of 31 (k=4); the column maximising \|Δln u\| coincides in **0 of 32** and **0 of 31**. \|S̄\| varies 7.11×, \|Δln u\| varies 1.01× | 783-796 | `verify_ridge_mechanism.py` — **writes nothing** | ▶️ | k=1 and k=4 agree | a quantity changing by 1% cannot produce a 7× crest | **SUPPORTED and severe. No artifact** |
| 5.29 | **The preregistered hypothesis is FALSIFIED**: Δ is nearly flat (0.9655 → 1.9403 → 1.9368), the cap spans only 0.394–0.450, a 14% range. **The maximum sits where ln(u_CRE/u_peak) changes sign** | 836-882 | `findings_10` B.2 | ✅ | n/a | **predicted f₃−f₄ → 0 at LTE; falsified, and the falsification produced a better result** | **SUPPORTED — exemplary. A prediction written in advance, refuted, and the refutation improved the claim** |
| 5.30 | Griem test: moving from the (3,4) pair to (4,5) should move the crest by (5/4)^8.5 = **6.7 predicted before measurement**; measured **7.2** | 898-905 | `verify_ridge_mechanism.py` — no artifact | ▶️ | argmax by shell pair: 3.73×10¹⁴ / 1.93×10¹³ / 2.68×10¹² | **the ridge failing to move with the shell pair, or moving the wrong way. Neither occurred** | **SUPPORTED — the best-evidenced result in the thesis, and it has no artifact** |
| 5.31 | Griem's LTE criterion at n=4, Te = 2 eV gives 2.05×10¹³ against a measured 1.93×10¹³, 6% | 911-914 | `\cite{Griem1963,Griem1997}` | ▶️ | n/a | — | SUPPORTED |
| 5.32 | Independent diagnostic: boundary level n̄ = 3.08–3.25 at the crest density at every temperature | 916-922 | `verify_boundary_descent.py` → **`validation/boundary_descent.csv`, a real artifact** | ▶️ | n/a | the boundary falling outside (3,4) would refute it; **did not** | **SUPPORTED — the one leg of the three-route mechanism that has a stamped artifact** |
| 5.33 | The competing explanation is refuted: at ne = 10¹⁵ both f's fall toward zero (0.071, 0.011) — both shells become **continuum**-fed, not ground-fed | 924-931 | `findings_10` §2 | ✅ | n/a | — | **SUPPORTED — a competing hypothesis tested and refuted** |
| 5.34 | Sub-grid drift: parabolic peak 3.66×10¹³ (1 eV) → 1.51×10¹³ (9.5 eV), range 2.56×, drifting **downward** while Griem predicts **upward** as √Te by 3.2× | 947-954 | `findings_10` §7.2 | ▶️ | drift is inside one grid interval (2.68×), so unresolvable | — | **SUPPORTED with the right conclusion: "constant to within a factor 2.68 over 1 to 10 eV, never as fixed". Conflicts with 5.12's single vertex** |
| 5.35 | Filter mismatch: `verify_plateau_gridmap.py` excludes the 54 points with no plateau window; `verify_ridge_mechanism.py` does not; all 54 lie at j ≥ 4 | 977-984 | `findings_10` §B.3 | ▶️ | — | — | **SUPPORTED disclosure. But 54 here vs 104 at line 1207 — only 104 is consistent with 680 + 104 = 784** |
| 5.36 | ELM census: 680/202/0.3868; **448/45/0.1748**; 108/0/0.0717. "**The middle row is the result**" | 1176-1193 | `validation/divertor_map/divertor_map.csv` — **a real artifact** | ▶️ | thresholds 5/10/20% give 348/202/61 — **the count is not robust, the worst case is** | — | **SUPPORTED. The scope discipline here is correct and should be the model for the rest of the thesis** |
| 5.37 | Timescale separation does not predict: maxima 52× apart in density, ε = 12.0% (75th percentile) at the M maximum, a factor 3.2 below the worst; raw corr +0.76, Arrhenius control +0.71, 93% explained by (Te,ne), quadratic partial −0.44 | 1268-1291 | `make_ch5_figures.py`, which refuses to draw the figure unless the sign flips | ✅ | **eight bases; sign robust, magnitude not** | if M predicted, controlling for (Te,ne) would leave the correlation; **it flips sign** | **SUPPORTED for the sign — Part 3.1(iv). Greenland priority correctly cited at 1304-1308** |
| 5.38 | Quasi-neutrality: 68 of 392 operators need \|Δne/ne\| > 10%, 36 need > 100%; +20.0 at [0,0]; 235 Pa vs an ITER design range of 1–20 Pa; 3280 Pa at [0,7] | 1367-1408 | **[SOURCE REQUIRED]** at 1403-1408, and the 1–20 Pa range needs a citation | ▶️ | n/a | — | **UNSUPPORTED provenance, self-flagged — and this is `backlog` G3, "the hardest new constraint", which independently forces the same Te ≥ 2 eV boundary that trapping does** |

**Chapter 5 verdict.** The scope discipline is the best in the thesis: the
grid-edge admission (5.6), the widened ridge range (5.10), the refused
amplification maximum (5.4), the retracted one-sided linearisation (5.24), the
falsified LTE prediction (5.29), and eight self-flagged `[UNVERIFIED]` brackets.
Against that, **the mechanism chapter's producing script writes nothing**, and
the single result the thesis now headlines (G1, step-independence) has its two
critical columns flagged as having no artifact by the chapter itself.

---

## Chapter 6 — What This Model Cannot Say

Claim G, ¬4, the approximation register. 99 assertions; 24 rows.

| # | Claim | Line | Test → artifact | State | Alt-def | Falsifier / appeared? | Verdict |
|---|---|---|---|---|---|---|---|
| 6.1 | σ₀ = 7.74×10⁻¹⁴ cm² at 1 eV | 142-144 | — | — | — | — | **REFUTED. I evaluated the repo's own `escape_factor.lyman_alpha_sigma0(1.0)` today: 5.474×10⁻¹⁴ cm². The chapter's value is high by exactly √2 (5.474 × 1.41421 = 7.741).** `backlog` G2 records the same. The script used the right value throughout; only the write-up is wrong |
| 6.2 | Optical depth 114 per cm at 1.00 eV, ne = 5.18×10¹³ | 146-159, 161, 303 | — | — | — | — | **REFUTED by inheritance from 6.1: 114/√2 = 80.6, and `backlog` G2 gives 80.4.** The τ = 7900 over 5 cm at line 165 "follows from no convention" |
| 6.3 | Escape factor 2.7×10⁻⁵ over 5 cm, independently reproducing §3.5.3's 3×10⁻⁵ | 164-168 | two routes | ▶️ | n/a | — | SUPPORTED as an order of magnitude; the prefactor inherits 6.1 |
| 6.4 | The effective A(2p→1s) in L is wrong by 4–5 orders of magnitude at the cold edge | 169-170 | — | ▶️ | n/a | — | **SUPPORTED — and this is why Te ≥ 2 eV is a real boundary, not a hedge** |
| 6.5 | Θ_P at the benchmark is 0.986–0.9999 depending on D; at [23,5] the plasma is thin to better than 1 part in 10³ | 173-175, 280-283 | reproduced today via `verifych3_gb.py` B3: Θ_P(1 cm) = 0.9986, Θ_P(10 cm) = 0.9863, Θ_P(100 cm) = 0.8725 at [23,5] | ✅ | slab vs isotropic ≈ 2.1× | — | **SUPPORTED** |
| 6.6 | Of the 202 points above 10%, **157 lie below 2 eV**, 105 have τ_Lyα(5 cm) > 1, and 26 above 100 | 184-188 | `divertor_map.csv` | ▶️ | n/a | — | SUPPORTED |
| 6.7 | Self-consistent Θ_P ↔ n(1s) fixed point converged at all 400 points, all 14 Lyman channels, D swept 1–20 cm | 204-209, 236-237 | `verify_lyman_trapping.py` → **`validation/lyman_trapping/lyman_trapping.csv/.txt`, a real stamped artifact** | ✅ | D swept rather than fixed | — | **SUPPORTED — the strongest single piece of Claim-A evidence, and it has an artifact** |
| 6.8 | Gate 1: f = 0.4162, 0.0791, 0.0290, 0.0139 for Ly-α to δ, the literature values to four figures, none imported | 215-219 | derived from the repo's own A's | ✅ | n/a | — | **SUPPORTED — no `\cite{}` for "accepted literature values"** |
| 6.9 | Gate 2: σ₀ **failed at 1156% on first run**, catching a 4π error; then agrees to 0.059% | 220-225 | `escape_factor.py` | ✅ | n/a | — | **SUPPORTED — the result exists because that gate fired** |
| 6.10 | Gate 3: the untrapped rebuild reproduces the canonical matrix with max\|rebuilt − canonical\| = **0**, exactly | 226-229 | — | ✅ | n/a | any non-zero difference would invalidate every trapped number | **SUPPORTED — every trapped number is a difference against the real matrix** |
| 6.11 | **Table 6.1: above 2 eV the count does not move at all — 45/448 in every run — and the worst case moves 0.5%, 0.1748 → 0.1740, across a twentyfold range in D** | 241-297 | `lyman_trapping.txt` — **stamped and quoted in the caption** | ✅ | D = 0, 1, 5, 20 cm; T_at = Franck-Condon 3 eV gives 173/680 vs 170/680 | **the prior claim was that trapping makes breakdown *worse*. FALSIFIED** | **SUPPORTED — Claim G, and the strongest robustness statement in the thesis. The scope is stated on every row** |
| 6.12 | Below 2 eV the 38.7% becomes 11.6–15.7%, a factor 2.5–3.3, but still exceeds 10% at every thickness | 336-339 | Table 6.2 | ✅ | n/a | — | **SUPPORTED — the point still breaks down; the magnitude is not quotable** |
| 6.13 | The fall is not loss of sensitivity: \|f₃−f₄\| moves only 6% at D = 1 cm; what collapses is the displacement — n_gs falls 2.99×10¹⁴ → 7.26×10¹³, τ_QSS shortens 8×, \|Δln x\| roughly halves | 347-361 | — | ▶️ | n/a | — | **SUPPORTED — the mechanism is identified, not just the effect** |
| 6.14 | At Te ≥ 1.6 eV the maximum falls on 1.93×10¹³ in every run; **below 1.15 eV it wanders across three columns, a factor 7, so the two coldest rows cannot support a ridge claim at all** | 369-377 | — | ✅ | D = 0–20 cm | — | **SUPPORTED — a correctly withdrawn claim** |
| 6.15 | τ_relax at benchmark 2.2769 → 2.2947 ns (+0.78%) at D = 20 cm; at [0,3] 8.8842×10⁻⁹ → 2.4289×10⁻⁷ s, **a factor 27.3** | 393-399 | "this project's records" | ▶️ | n/a | — | **SUPPORTED, and this is where the register's unscoped "within 1%" is correctly resolved — Part 3.1(ii)** |
| 6.16 | Transport: τ_QSS = 233 ms at the worst point; a 1–3 eV D atom crosses 10 cm in 6–10 µs; a renewal time of ~26 µs suffices to drop the worst point below 10% | 448-474 | — | ▶️ | free-streaming; **`backlog` I6 gives 72 µs with CX trapping, against the 26 µs threshold — the result survives by 2.8×** | — | **SUPPORTED as a conditional. NOT TESTED and untestable in 0-D. This, not closure, is the real threat to the cold-corner magnitude** |
| 6.17 | No H₂, D₂⁺, D⁻, MAR or dissociative excitation anywhere in the rate assembly | 514-518 | grep | ✅ | n/a | — | SUPPORTED |
| 6.18 | At the worst point the model's own equilibrium is **96.6% neutral**; literature attributes up to 60–70% of D_α and 10–20% of D_γ to molecular channels | 541-554 | `\cite{Wijkamp2023,Verhaegh2021}` | ✅ | n/a | — | **SUPPORTED — exactly where the headline lives** |
| 6.19 | "**this work offers no bound on how much, and none can be constructed from the data in this repository**" | 565-567 | — | — | — | — | **REFUTED by `chapter7.tex:362-366`, which constructs one (factor 3–5) from `findings_10` D.4. Chapter 7 flags the contradiction itself. `backlog` I5 says the bound *is* constructible from references Chapter 6 already cites** |
| 6.20 | Ridge scaling: ×0.1 → 2.28×10¹², ×1 → 1.68×10¹³, ×10 → 2.48×10¹⁴ — one decade in n_gs moves the ridge one decade in ne | 620-639 | — | ▶️ | n/a | — | **SUPPORTED — and it is why the ridge location is conditional. Note the table's ×1 row gives 1.68×10¹³ while line 641 quotes the measured crest as 1.93×10¹³; the text does not distinguish them** |
| 6.21 | The ridge sits at 1.93×10¹⁹ m⁻³, a factor **5.2 below** the detached band and **1.6 below** the modelled upstream separatrix | 688-692 | `\cite{Guillemaut2011}`, `\cite{Stangeby2023,Pitts2019}` | ✅ | n/a | — | **SUPPORTED — ¬3, the detachment retraction, correctly executed** |
| 6.22 | The repo records `Stangeby2023` as "Stangeby 2022, Nucl. Fusion 62" in `grid.py` and three `outputs/` documents; the DOI is right, the volume and year wrong | 707-711 | audit | ✅ | n/a | — | **SUPPORTED — a self-disclosed citation defect** |
| 6.23 | Closure table: cold corner 67.23 → 1.6226 s (41.4×); worst-case operator 0.23324 → 0.015173 s (15.4×); ridge 1.01×; benchmark 1.00×. Grid min 1.00, median 1.00, max 41.4 | 782-796 | `findings_10` §3.1 | ✅ | n/a | 2300 named in advance; measured 15 | **SUPPORTED. The chapter itself flags at 822-825 that the source record labels the worst-case operator [1,4] in one place and [0,4] in another for the same τ_QSS** |
| 6.24 | Threshold sensitivity: 348 / 202 / 61 at 5 / 10 / 20% over 680. "**The count is not robust to the threshold; the worst case is**" | 1006-1009 | — | ✅ | n/a | — | **SUPPORTED — the correct reading, stated in bold** |

**Chapter 6 verdict.** The best-scoped chapter in the thesis. Claim G — that
the Te ≥ 2 eV restriction is a *measured boundary* rather than a hedge — is
established with a stamped artifact, three gates passed before any result was
read, and one gate that actually fired. Two REFUTED numbers (6.1, 6.2), both
√2 errors in the write-up rather than the code, both propagating into three
places. One live contradiction with Chapter 7 (6.19).

---

## Chapter 7 — What It Means

43 assertions; 14 rows. **Structurally complete**, with three self-flagged
brackets.

| # | Claim | Line | Test → artifact | State | Alt-def | Falsifier / appeared? | Verdict |
|---|---|---|---|---|---|---|---|
| 7.1 | ε_QSS = 8.66×10⁻⁶ at benchmark and 6.73×10⁻⁹ at the grid point carrying the largest diagnostic error anywhere | 46-49 | restates 5.1 | ✅ | n/a | — | **SUPPORTED — the reversal, correctly headlined** |
| 7.2 | A QSS solve returns an answer already correct to eight decimal places | 66-68 | restates 5.1 | ✅ | n/a | — | SUPPORTED |
| 7.3 | The recipe: ε = \|exp(S̄ G Δln Te) − 1\| | 93-99 | derivation | ✅ | exact form, not linearised | — | **SUPPORTED — this is the pivot from a number to a rate, and it is the right one** |
| 7.4 | G is negative everywhere on the grid | 112-116 | none named | ▶️ | n/a | — | SUPPORTED, no artifact |
| 7.5 | \|G\| moves by at most **6.3%** across step sizes of 1, 2, 4 intervals over 736 triples while ε changes by up to **8.44**× | 142-146 | `validation/reservoir_gain/reservoir_gain.csv`, 2288 rows, stamped 10 Sep — **but [UNVERIFIED: no script performs these two aggregations]** | ▶️ | **k = 1, 2, 4** | **if G varied across k as much as ε does, there would be no step-independent coefficient and the reframing would fail. It did not appear** | **SUPPORTED with a named falsifier that did not appear; aggregation UNSUPPORTED, self-flagged. This is G1 — the result the thesis now headlines** |
| 7.6 | Over 982 heating steps \|S̄\| runs 0.0649–0.4822; over 2288 rows \|G\| runs 2.640–14.519; restricted to Te ≥ 2 eV, 0.0649–0.4552 and 2.640–7.822, product 0.362–3.535 | 154-160 | `reservoir_gain.csv` | ▶️ | **scopes explicitly stated for each range** | — | **SUPPORTED — every extremum here names its set. This is the standard the rest of the thesis should meet** |
| 7.7 | In the worst defended conditions a 1% temperature excursion costs the Balmer inversion ~3.5%; at the benchmark ~1.3% | 160-162 | — | ▶️ | n/a | — | SUPPORTED |
| 7.8 | \|d ln R/d ln b₁\| < 1 everywhere, so a reservoir stale by a factor of two cannot put the ratio out by more than a factor of two — "**for every shell pair, every (Te,ne) and every atomic dataset**" | 173-178 | theorem | ✅ | n/a | — | **SUPPORTED as stated, but see 7.14: the universal quantifier is correct only for a two-channel matrix** |
| 7.9 | At [23,3] the measured sensitivity is at **97% of the ceiling**, so no refinement of the rate coefficients can make it worse there | 178-182 | `findings_10` B.2 | ▶️ | n/a | — | **SUPPORTED — a genuinely useful transferable statement** |
| 7.10 | The novelty claim: "**it has not previously been stated in this form in the collisional–radiative literature**" — the absence is weak evidence; the search reached Crossref and OpenAlex | 269-278 | literature search | ▶️ | n/a | — | **SUPPORTED — correctly hedged, exactly as `claim_hierarchy` requires. Never "genuinely new"** |
| 7.11 | **[SOURCE REQUIRED: Fujimoto Ch. 4 and the JPSJ series I–IV have not been read against this question and are not in `references.bib`]** | 284-292 | — | — | — | — | **UNSUPPORTED, self-flagged. This is the largest remaining novelty risk** |
| 7.12 | Above 2 eV, conservation of nuclei demands \|δne/ne\| < 10⁻³; below it, 68 of 392 operators need > 10% and 36 need > 100%. **Te ≥ 2 eV is the only self-consistent region of the grid** | 346-353 | `backlog` G3 — no script, no artifact | ▶️ | n/a | — | **SUPPORTED reasoning, UNSUPPORTED provenance. An independent second route to the same Te ≥ 2 eV boundary that trapping gives — worth advertising once it has an artifact** |
| 7.13 | Molecular emissivity fractions of 0.60 and 0.30 at n=3 and n=4 reduce the cold-corner error by a factor 3–5 — **[UNVERIFIED: Section `sec:molecules` says no such bound can be constructed. The two statements are inconsistent]** | 362-371 | `findings_10` D.4 | 💡 | — | — | **UNSUPPORTED, self-flagged, and it contradicts 6.19** |
| 7.14 | With a molecular channel, n_p = a_p n_gs + c_p n_ion + m_p n_H₂, and "**there is no reason to expect [the bound] to survive in that form**" | 419-427 | structural | ✅ | n/a | — | **SUPPORTED, and it correctly qualifies 7.8 and 6.19's "any matrix whatever". The framework is two-channel *by construction*; a molecular channel is a third and destroys the functional form rather than perturbing it** |

**Chapter 7 verdict.** The pivot to `dε/d ln Te` is the correct architectural
repair for the one broken chain (R6), and Chapter 7 executes it. The novelty
statement is properly hedged. The chapter self-flags its three weakest points.
Its one substantive defect is the unreconciled molecular-bound contradiction
with Chapter 6.

---

# PART 5 — Internal contradictions

Each of these is two files, or two chapters, describing one quantity — the
condition `findings_10` §11 item 10 names as the mechanism behind three
retractions.

| # | Quantity | Values in conflict | Adjudication |
|---|---|---|---|
| 5.1 | Conservation residual | **2.59×10⁻¹¹** (`claim_hierarchy`:170, `derivation_01`:142) · **3.61×10⁻¹¹** (`chapter3.tex`:272, `chapter4.tex`:164) · **2.238×10⁻¹²** (`chapter4.tex`:192, `CHANGE_REPORT` §4.2) · **3.6×10⁻¹¹** (`chapter2.tex`:996) | **Settled today by direct computation: 3.6078×10⁻¹¹ is the max over all 400 points, at [0,2].** 2.59×10⁻¹¹ is stale (pre-correction; `HANDOVER.md`:116 records the change). 2.238×10⁻¹² is a *single unnamed point*, not a grid max — which is why `chapter4.tex:234-237` flags the missing grid point. Three of the four are reconcilable; none of the four says which scope it is |
| 5.2 | Worst-case separation | **86.8** (`chapter4.tex`:385, `chapter5.tex`:1255) vs **86.5** (`chapter5.tex`:464, 472) | Different quantities — full matrix vs excited block — presented as interchangeable. Part 3.2(vi) |
| 5.3 | Conservation gate severity | `chapter3.tex`:222-225 says three fault classes break it; `chapter4.tex`:168 says "that claim is testable, and it is false" | **Chapter 4 is right.** Part 1.1 |
| 5.4 | n_max convergence | `chapter4.tex`:615-620 reports a completed scan with extrapolated errors; `chapter5.tex`:969-975 says `[UNVERIFIED: has not been tested]` | Both cite the same session. `backlog` C6 closes it. Chapter 5's bracket is stale |
| 5.5 | No-plateau exclusion count | **54** (`chapter5.tex`:978) vs **104** (`chapter5.tex`:1207) | Only 104 is consistent with 680 + 104 = 784, the superposition pair count |
| 5.6 | Sub-grid ridge vertex | one vertex **1.65×10¹³** (`chapter5.tex`:346) vs temperature-dependent **3.66×10¹³ → 1.51×10¹³** (`chapter5.tex`:947) | Both cite `findings_10`, §B.3 and §7.2 |
| 5.7 | Molecular bound | `chapter6.tex`:565 "none can be constructed" vs `chapter7.tex`:362 constructs one | Chapter 7 flags it; `backlog` I5 sides with Chapter 7 |
| 5.8 | κ(L_FF) provenance | `backlog` H11 "now has a script" vs `chapter4.tex`:580 "[UNVERIFIED: no committed script]" | **The chapter is right.** Confirmed: four `np.linalg.cond` hits, none on the full grid |
| 5.9 | ELM count | **202** (`findings_10` §4.1, figure captions) vs **201** (`thesis_ready` A11, which decomposes it as 105 + 97 in the same sentence) | 105 + 97 = **202**. `backlog` H9 settles it; `chapter5.tex`:1195-1201 carries the flag |
| 5.10 | Order-of-magnitude of the radiative/collisional asymmetry | "five orders" (`chapter3.tex`:633) vs "nine orders" (`chapter3.tex`:1538), from the identical two rates | 6.3×10⁸/10⁴ = 6.3×10⁴. **Line 633 is right; 1538 is wrong by four decades** |
| 5.11 | τ_QSS dynamic range | "seven orders" (`chapter3.tex`:528) | 67.2 s / 75.4 ns ≈ **nine**. "Seven" matches the retired 1.18 µs floor |
| 5.12 | b₁ dynamic range | "three decades" (`chapter3.tex`:954) | 76.6 → 2.683×10⁵ is **3.54** decades |
| 5.13 | τ_QSS/τ_drive falsifier | **2300** (`chapter4.tex`:743, `chapter6.tex`:759) vs **2332** (`chapter5.tex`:1234, `chapter6.tex`:978) | Same quantity, rounded two ways in one document |
| 5.14 | r₁ deficit at p=3 | **8.3** (`chapter6.tex`:855) vs "factor-8" (`chapter6.tex`:1039, `chapter7.tex`:400) | Rounding, but in a number that gates the paper |
| 5.15 | Lower-bound test points | 5 rows in `tab:lowerbound` (`chapter4.tex`:836) vs "seven points" (`chapter5.tex`:1152) | Unreconciled |
| 5.16 | Temperature-decay attribution | 76% + 19% = 95%; 72% + 13% = 85% (`chapter5.tex`:997-999) | Neither sums to 100; the remainder is unexplained |
| 5.17 | `L_grid.npy` write date | 21 July 2026 (`chapter2.tex`:1114) vs 14 July 2026 (CLAUDE.md) | Unreconciled |
| 5.18 | Ridge at unit scaling | 1.68×10¹³ (`chapter6.tex`:629 table) vs 1.93×10¹³ (`chapter6.tex`:641 text) | Possibly different quantities; the text does not distinguish them |

---

# PART 6 — Falsifiers named in advance, and whether they appeared

This is the strongest section of the thesis's evidentiary record and it should
be presented as a table in Chapter 4.

| Result | Falsifier named **before** the test | Appeared? | Where |
|---|---|---|---|
| **R2 QSS closure is exact** | if the closure were the culprit, the residual would be O(ε) | **No** — 8.66×10⁻⁶ / 6.73×10⁻⁹ against a reported 10⁻²–10⁻¹ | ch5:227-238 |
| **R7 Ridge mechanism** | **(5/4)^8.5 = 6.7 predicted before measurement**; refuters named (ridge fails to move with the shell pair, or moves the wrong way) | **No** — measured 7.2; both refuters absent | ch5:898-905 |
| **R8 Ridge is a position effect** | predicted f₃−f₄ → 0 at LTE | **YES — FALSIFIED, productively.** Δ is flat; the falsification produced a better claim | ch5:836-882 |
| **R11 Trapping invariance** | prior claim: trapping makes breakdown *worse* | **YES — FALSIFIED.** 45/448 unchanged over a twentyfold slab range | ch6:241-297 |
| **R12 Open/closed boundary** | a factor ≈ **2300** would refute the ELM census | **No** — measured 15. Survived by two orders of magnitude | ch4:742-798 |
| **R13 Observable robustness** | the 4F dipole-dark lever should break the shell/line equivalence | **No** — f(4S) = 0.0608 vs f(4F) = 0.0606 | ch4:688-701 |
| **G1 Reservoir gain** | if G varied across k as much as ε does, the reframing fails | **No** — \|G\| 6.5%, ε 8.44× | ch7:142-146 |
| **A12 Eigenvalue filter** | if the shift were not exactly one rung the diagnosis is wrong | **No** — exact at all 19 points | ch2:1155-1164 |
| **σ₀ cross-check** | agreement within 1% | **YES — FIRED at 1156%**, catching a 4π CGS/SI error | ch6:220-225, ch4:517-522 |
| **Fujimoto no-ℓ-mixing variant** | a level cannot exceed Saha equilibrium | **YES — FIRED**, r₀(2) = 104.5 flagged by the script itself | ch4:524-529 |
| **A7 truncation confound** | max_p attained at n=15 at 346 of 400 points ⇒ suspected artifact | **No** — resolved-only recomputation moved the mean 0.517 → 0.515 | `thesis_ready` A7 |
| **Ridge competing hypothesis** | "both shells become ground-fed at high ne" | **YES — REFUTED**; both f's fall toward zero (0.071, 0.011) | ch5:924-931 |
| **R6 the magnitude (38.7%)** | **NONE EVER STATED** | — | ch5:280-288 |

**Twelve of thirteen results carry a named falsifier. Three of those falsifiers
actually fired, and two of the three fired on a *prediction of this project's
own*, forcing a better claim.** That record is unusually strong and is
currently spread across four documents rather than presented once.

**R6 remains the exception,** and it is the number the abstract was planned
around. `chapter5.tex:406-408` diagnoses this itself: *"the sentence 'the
quasi-steady-state error reaches 39%' is not falsifiable because it does not
say what it is 39% of."* Chapter 7's pivot to `dε/d ln Te = |S̄·G|` is the
repair, and both factors are independently measurable, so the product is a
prediction and the map is its test.

---

# PART 7 — What to fix, ranked by cost of getting it wrong

**Report only. Nothing below was acted on.**

1. **`chapter3.tex:222-225`** — REFUTED by fault injection. The conservation
   check detects an `A`/γ inconsistency and nothing else. Chapter 4 already
   says so; Chapter 3 must stop contradicting it.
2. **`chapter4.tex:548-550`** — the c₃↔c₄ justification is algebraically wrong.
   Measured: |Δ| moves 1.945614 → 0.939493. The conclusion stands; the reason
   does not.
3. **Five scripts write nothing** (`verify_ridge_mechanism`, `verify_eps_gridmap`,
   `verify_plateau_bridge`, `verify_ch3_claims`, `verifych3_gb`). The first
   produces Claim F.3, the best-evidenced result in the thesis. Add a stamped
   CSV to each.
4. **Backlog §G3–G7 hold ✅ with no script and no artifact.** Five results, one
   of which (86.5) is promoted in Chapter 5 to "the figure that describes the
   grid". Either produce the scripts or demote to ▶️.
5. **κ(L_FF) still has no producer.** `backlog` H11 says otherwise and is
   wrong; `chapter4.tex:580-587` is right and already carries the bracket.
6. **`chapter6.tex:142-159` σ₀ and τ are high by √2.** Correct σ₀ = 5.478×10⁻¹⁴,
   τ per cm at [0,4] = 80.4 not 114. Three sites. The code is right.
7. **`chapter3.tex` names no producing script for any of ~90 numbers.**
8. **Chapter 4 is unfinished** — four dangling labels, zero citations, and the
   Fujimoto external gate (`thesis_ready` A2, the project's strongest external
   evidence) promised and absent.
9. **`chapter3.tex:1538` "nine orders"** contradicts line 633's "five orders"
   from the identical two rates. Also line 528 "seven orders" (should be nine)
   and line 954 "three decades" (should be 3.54).
10. **`chapter3.tex:444-446`** describes v₁ as "distributed across the excited
    manifold"; PR(v₁) = 2.64. REFUTED by `backlog` C2.
11. **The M_eff "13%"** is a benchmark number in a grid-wide sentence. Grid
    worst is 34.5% at [49,0].
12. **`verify_ch3_claims.py` hardcodes the thesis values it checks**, so its
    sole FAIL now fires against a chapter that has already been corrected.
    A checker that goes stale silently inverts its own verdict.
13. **`preflight.py` prints "All checks passed" while two of six named scripts
    do not exist on disk**, downgraded to a non-blocking note. Per CLAUDE.md
    rule 2 this must raise.
14. **No gate raises on failure and `validate_gates.py` exits zero even when
    Gate D fails at 100% of points.**
15. **The Chapter 6 / Chapter 7 molecular-bound contradiction** (6.19 vs 7.13).
16. **`audit_writers.py` must not be cited as single-writer evidence** — four
    live CSV writers are invisible to it, including the producer of
    `plateau_gridmap.csv`, the source of A8/A9/A11.
17. **The isolation gate is 2.4× slacker than the quantity it guards**
    (`if iso.min() < 10.0` against a measured minimum of 24.33), with no source
    for 10.0.
18. **Two "Gate D"s** — the ADAS external gate and the assembler's conservation
    Check D — in a repo where two-names-one-quantity has caused three
    retractions.
19. **CLAUDE.md's known-issue table points at the wrong file:**
    `qss_analysis.py:137` now reads `eigs < 0.0`; the `< -1.0` literal is live
    at `solve_cr.py:269` and `check_mz.py:10`. **Fix the table, not the code.**
20. **`chapter2.tex:890`** labels 918.08 as m_H/m_e when it is the correct
    reduced mass m_p/2m_e. The number is right; the definition is wrong, and it
    sits under a square root where a reader will assume a factor-2 error.

---

## Closing note on method

Nothing in this audit was made to pass, no tolerance was touched, and no
missing value was stood in for. Three checks were confirmed unable to fail and
are labelled wiring checks rather than removed.

**What was run.** Two report-only scripts from the repo
(`verifych3_gb.py`, `verify_ch3_claims.py`, both confirmed to contain no write
call) and two throwaway scripts in the session scratchpad, all against the
canonical `L_grid.npy`. `git status` confirms nothing under `data/`,
`validation/`, `src/` or `thesis_tex/` was modified.

**Reproduced and agreed** — 3.6078×10⁻¹¹ at [0,2] (twice, independently);
`M_eff` 8659 at benchmark and 77.6 at [49,7]; the 1.118/1.151/1.526 norm
ratio; the 0.0098% and 0.3461% framing agreement; Δ = 1.945614 (twice,
independently, matching `chapter5.tex` to six figures); M = 9981.93,
M⁺ = 8242.74, isolation 3566.96 and 24.3286.

**Reproduced and disagreed with a document** — the τ_QSS floor against a stale
literal hardcoded in `verify_ch3_claims.py`; σ₀ = 5.474×10⁻¹⁴ against
`chapter6.tex`'s 7.74×10⁻¹⁴; |Δ| under a c₃↔c₄ swap (1.945614 → 0.939493)
against `chapter4.tex:548-550`'s "likewise"; the conservation sampling scope
(20 of 400) against `derivation_01:142`'s "all 50×8"; and κ(L_FF)'s provenance
against `backlog` H11's "now has a script". **Every disagreement is reported
and left standing.** Three of the five are cases where the chapter is right
and a register or a script is wrong, which is the opposite of the direction
this project has had to correct before.

**The thesis's evidentiary record is stronger than its provenance record.**
Twelve of thirteen results carry a falsifier named in advance and three of
those falsifiers actually fired — a standard most computational theses do not
meet. What is missing is not evidence but *stamps*: 57 numbers whose only
artifact is a paragraph, and five verification scripts that print their results
and forget them.
