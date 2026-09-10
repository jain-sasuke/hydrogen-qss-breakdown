# Findings 10 — Four-Agent Review of PART A

**Written 9 September 2026**, from four independent passes over
`outputs/thesis_ready.md` PART A: a physics attack on A10/A11, a reproduction
run against the canonical matrix, a line-by-line audit of the nine uncommitted
verification scripts, and a publication assessment against the external
literature. The four agents did not see each other's output.

**Status:** the *scripts* are sound and the *mechanism* is stronger than
claimed. `thesis_ready.md` PART A is not: it carries ✅ marks on statements
`findings_09` withdrew, quotes one extremum on a different scope from the two
beside it, and prints a table whose omitted column contradicts its own caption.
Two attacks that were expected to be fatal were survived and quantified, and
belong in the defence rather than in a footnote.

**Provenance for everything below:** `L_grid.npy` SHA-256
`2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`
(2026-07-21 20:44), `S_grid.npy` `7822f536590c…fb80a`, state index
`23eb18538c1…979a65` — all three verified byte-wise at the start of the
reproduction run. Analysis in `validation/plateau_gridmap/`,
`validation/divertor_map/`, `validation/boundary_descent.csv`; scripts under
`src/validation/`. Interpreter `/opt/anaconda3/envs/cr/bin/python` (the one
named in `run_pipeline.sh:24`), numpy 2.3.5, scipy 1.16.3.

**Nothing in the repository was modified by the review itself** except an
append to `thesis_grade_results_backlog.md` at ▶️ Run. No tolerance, filter,
data file or script was touched. The ADDENDUM at the end of this document
reports a later computation which added one new script and one new output
directory; it likewise did not modify any existing file or any data under
`data/`.

---

## 1. Convergent findings — reached by more than one agent, independently

These are the ones to act on first. Independent routes to the same defect is
the strongest evidence this review produced.

### 1.1 A1's τ_QSS floor is a subset minimum quoted as a grid-wide range

Found three ways: measured by the reproduction run, derived arithmetically by
the code audit, and located as a chapter contradiction by the publication pass.

`verify_ch3_claims.py` already **FAILS** this on the canonical matrix:

```
[FAIL] tau_QSS   min      thesis 1.18e-06 s   measured 7.53769e-08 s   rel 9.36e-01
  tau_QSS min at [49,7]  Te=10 eV  ne=1e+15
  M at the tau_QSS-min point = 86.77
  window_ok requires M > 900 -> point is OUTSIDE the analysed set
  tau_QSS min over M>900 subset = 1.177240e-06 s
```

A1 states *"M ranges 86.77 to 1.72928×10⁹ … τ_QSS spans 1.18 µs – 67.2 s;
τ_relax spans 0.87 ns – 38.9 ns"* — all in one sentence, over "all 400 points".
But τ_relax_min and M_min both come from **[49,7]**, where τ_QSS is **75.4 ns**.

The arithmetic that makes this undeniable:

$$75.4\ {\rm ns} / 0.8687\ {\rm ns} = 86.8 = M_{\min}\ \checkmark \qquad {\rm but} \qquad 1.18\ \mu{\rm s} / 0.87\ {\rm ns} = 1356 \neq 86.8$$

**46 of 400 points lie below the stated floor.** 1.18 µs is the minimum over the
346-point `M > 900` subset (`verify_plateau_gridmap.py:177,93-94`, win_lo =
win_hi = 30), reproduced to three figures as 1.177240×10⁻⁶ s.

`chapter3.tex` Eq. `\eqref{eq:M_range}` already says 75.4 ns and is correct.
`thesis_ready.md` A1 is stale.

**Action:** quote 75.4 ns as the unrestricted floor, or state the M > 900 scope
on all six bounds of the box. As written, an examiner who divides the two
quoted minima gets 1356 and asks why M_min is 86.8.

### 1.2 A10's table omits the column that contradicts its caption

Found independently by the code audit (full 8-column recomputation) and the
physics pass (sub-grid parabolic fit).

All 16 numbers in A10's table reproduce exactly. The defect is column
selection: the table shows ne = 10¹², 1.93×10¹³, 1.39×10¹⁴, 10¹⁵ — four of
eight — and the omitted **j = 4 (5.18×10¹³)** is the actual row maximum at the
three coldest rows:

| Te (eV) | 1.93e13 (shown, bolded) | **5.18e13 (omitted)** |
|---|---|---|
| 1.000 | 37.093 | **38.690** |
| 1.048 | 35.541 | **36.452** |
| 1.099 | 34.012 | **34.322** |
| 1.600 | **23.220** | 21.054 |
| 2.947 | **12.166** | 9.984 |

Per-row argmax, rows 0–15: `[4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3]`.

**A10 and A11 therefore contradict each other in adjacent sections.** A11's
worst case — *"38.7% at Te = 1.0 eV, ne = 5.18×10¹³"* — is at j = 4, the column
A10's table does not print. Same grid, same quantity, two ridge locations.

`verify_eps_gridmap.py:149` contradicts itself on the same point: its headline
`j_ridge = argmax(nanmax(eps_p, axis=0))` prints "ridge column ne = 5.179e13
(j=4)" while its own per-row table prints j = 3.
`verify_ridge_mechanism.py` is correctly scoped (`TE_LO = 2.0`, line 229) and
gets j = 3 for all 32 rows — **the script is right; the register
over-generalised it.**

**What refutes A10 and did occur:** a temperature row whose density argmax is
not j = 3. It is in the data and was masked by showing half the columns.

### 1.3 The detachment identification is retracted in one file and live in four

| file:line | says |
|---|---|
| `chapter5_C5D.tex:284` | **"It is not detachment."** |
| `thesis_ready.md:294` | "In divertor terms that crossover is **detachment**" |
| `chapter3.tex:1349` | "…what a divertor passes through as it detaches" |
| `thesis_main.tex:336` | "The crossover is detachment" |
| `HANDOVER.md` PART 2 | "…which is detachment" |

One withdrawal, four surviving assertions. Chapter 3 currently promises a
Chapter 5 result that Chapter 5 explicitly retracts.

Arithmetic confirmed independently: 1.93×10¹⁹ m⁻³ is 5.2× below Guillemaut's
detached band (10²⁰–10²¹ m⁻³) and 1.6× below Stangeby's upstream separatrix
band. At the citable detached band (10¹⁴–10¹⁵ cm⁻³) with Te ≥ 2 eV the map
gives ε_plateau ≤ 10.4% and an ELM-averaged lower bound that **never reaches
10%** (max 7.2%). **The ridge and the detached band do not overlap.**

This is the condition under which the −46% Hα number propagated: two files in
`outputs/` giving opposite instructions about the same sentence.

### 1.4 PART A carries ✅ on statements `findings_09` withdrew

`findings_09_central_quantity_misnamed.md:97-102` withdraws W1–W6. Five of them
are load-bearing sentences in A8 and A11, which are still marked ✅. Most
serious: *"M reaches 1.7×10⁹ in the same corner carrying the 38.7% error"* is
false by arithmetic — M_max and ε_max are **52× apart in density**, and at the
M maximum ε = 0.12, the 75th percentile. (Grid index per
`findings_09_index_correction.md`: [0,0], not the [1,0] in the uncorrected W2
text.)

Per CLAUDE.md, ✅ requires a sensitivity check and written caveats. **A8, A9,
A10 and A11 do not meet that bar and should be demoted to ▶️.**

---

## 2. The mechanism — attacked hardest, came out stronger than claimed

The single most valuable result of this review. The two-channel/Griem-boundary
mechanism behind A10 survived three independent severe tests:

| test | prediction | measured |
|---|---|---|
| ridge moves with shell pair, Griem $n^{-17/2}$ | $(5/4)^{8.5} = 6.7$ | (3,4)→(4,5) ratio **7.2** |
| absolute density, Griem LTE criterion at $n=4$, Te = 2 eV | $2.05\times10^{13}$ | ridge at **1.93×10¹³** |
| independent diagnostic, `boundary_descent.csv` | boundary between 3 and 4 at the ridge | $\bar n = 3.08$–$3.25$, **all Te** |

Argmax of $|f_n - f_{n+1}|$ by pair at Te = 2.02 eV: (2,3) → 3.73×10¹⁴;
**(3,4) → 1.93×10¹³**; (4,5) → 2.68×10¹²; (5,6) and above fall off the grid
bottom.

**The competing hypothesis was refuted.** "Both shells become ground-fed
because collisional coupling dominates" is wrong: at high n_e both f's fall
toward *zero* (continuum-fed), not toward one — at Te = 2.95 eV, ne = 10¹⁵,
f₃ = 0.071 and f₄ = 0.011.

**What would have refuted the mechanism:** the ridge failing to move with the
shell pair, or moving the wrong way. Neither occurred.

A10 undersells this. It should be the headline of the section.

---

## 3. Two attacks survived, quantified — these belong in the defence

### 3.1 Closure does not destroy A11

The 44-state closure was rebuilt independently as
`Lc = [[L, S], [−colsum(L), −ΣS]]` — exactly particle-conserving (column sums
zero to machine precision; all column sums of L verified ≤ 0). It reproduces
the project's own recorded 1.6226 s at [0,0] to **five digits**, confirming it
is the same construction.

| point | τ_QSS open | τ_QSS closed | factor |
|---|---|---|---|
| [0,0] Te 1.0, ne 1e12 | 67.23 s | 1.6226 s | 41.4 |
| **[1,4] — post-step operator at the worst point** | **0.23324 s** | **0.015173 s** | **15.4** |
| [15,3] ridge | 2.027 ms | 1.999 ms | 1.01 |
| [23,5] benchmark | 22.728 µs | 22.708 µs | 1.00 |

**Index note, added 10 Sep 2026.** Two labels appear in this document for the
same worst case; both are correct, for different objects. The grid point where
the error is evaluated is **[0,4]** (Te = 1.000 eV, ne = 5.18e13), whose own
operator has tau_QSS = **0.42101 s**. The value 0.23324 s belongs to **L[1,4]**
(Te = 1.0481 eV), the **post-step** operator reached by the heating step from
[0,4] -- verified by direct recomputation from the canonical L_grid. This is
the same pre/post-step ambiguity recorded as trap 2 in section 8, which gives
M = 9982 at L[23,5] against 8243 at L[24,5]. **Every tau_QSS, tau_relax and M
quoted in the thesis must name which operator it belongs to.**

Grid-wide: min 1.00, median 1.00, max 41.4. **The factor is large only where
τ_QSS already exceeds the event duration by ~10³, and unity everywhere τ_QSS
approaches τ_d.** Recomputing the entire ELM map with closed τ_QSS:

- breakdown count 202 → **200** of 680
- worst case 0.38682 → **0.38563**
- restricted (Te ≥ 2 eV): 45 → 45, unchanged

**What would have refuted A11:** a closure factor of order τ_QSS/τ_d ≈ 2300 at
[0,4]. The actual factor is 15.

Caveat not to soften: this closure adds an ion reservoir but does not close the
electron balance (n_e held fixed while the ion population moves) and adds no
transport. See §4.2.

### 3.2 The lower bound is not merely conservative — it is near-exact

Direct eigen-propagation of the full 43-state system under the step, ε(t)
integrated over 0–100 µs:

| point | ε_plateau | true 100 µs average | claimed lower bound | true/lower |
|---|---|---|---|---|
| [0,4] worst case | 0.386903 | 0.386785 | 0.386820 | 0.9999 |
| [15,3] ridge | 0.180728 | 0.175272 | 0.174812 | 1.0026 |
| [15,5] | 0.103550 | 0.072175 | 0.071722 | 1.0063 |
| [23,5] benchmark | 0.063612 | 0.011790 | 0.011712 | 1.0066 |
| [0,0] | 0.083004 | 0.083065 | 0.083004 | 1.0007 |

The 0.9999 at [0,4] is a trapezoid artifact — the ns-scale rise is
under-resolved on a uniform 25 ns mesh, which *under*-weights it.

**Why it is this tight: the spectrum has no intermediate modes.** At [1,4] the
eigenvalues are −4.29, −1.60×10⁸, −3.44×10⁸ s⁻¹ — a gap of 3.7×10⁷ between λ₀
and λ₁. Nothing can make the observable decay faster than τ_QSS. That was the
failure mode being hunted, and `derivation_07:301-304` documents it occurring
for the *other* observable ε_res (1/e time 2.58 s against τ_QSS = 67.2 s).

**The bound language is understated.** For this observable the "lower bound" is
the answer to better than 0.7% at every point tested.

---

## 4. What actually kills the divertor claim

Not closure, and not the bound. Two things neither was a proxy for.

### 4.1 78% of the breakdown points lie inside the region §3.5.3 excludes

`verify_eps_gridmap.py:5-13,58` records that §3.5.3 restricts quantitative
results to Te ≳ 2 eV because the Lyman-α escape factor falls to 3×10⁻⁵ on the
Te = 1 eV edge. A11 leads with **38.7% at Te = 1.0 eV** — inside the excluded
region.

The escape factor was recomputed independently rather than taken from the
document. From the model's own CRE solution, n(1s)/n_ion = 28.3 at [0,4], so
with quasineutrality n(1s) = 1.47×10¹⁵ cm⁻³; Doppler line-centre
σ₀ = 2.654×10⁻² f/(√π Δν_D) = 7.74×10⁻¹⁴ cm² at 1 eV:

| Te (eV) | τ_Lyα per cm at ne = 5.18×10¹³ |
|---|---|
| 1.00 | **114** |
| 1.15 | 15.0 |
| 1.60 | 0.32 |
| 2.02 | 0.036 |
| 2.95 | 0.0023 |

Over 5 cm, τ = 7900 at the worst point → escape factor **2.7×10⁻⁵**,
independently reproducing §3.5.3's 3×10⁻⁵. **The Einstein coefficients in
`L_grid.npy` are wrong by 4–5 orders of magnitude in effective decay rate
there.** `derivation_07:299-300` records that at Λ = 10⁻³ — thirty times *less*
trapped than [0,4] — τ_relax goes to 2.05 µs and M falls 158×. Nobody has
computed what it does to f₃, f₄ or ε_plateau.

Recounted from `validation/divertor_map/divertor_map.csv` (ELM lower bound >
10%):

| scope | points | breakdown | worst lower bound |
|---|---|---|---|
| all window_ok (**as published**) | 680 | **202** | 0.3868 |
| Te ≥ 2 eV (**§3.5.3**) | 448 | **45** | 0.1748 at [15,3] |
| Te ≥ 2 eV **and** ne ≥ 10¹⁴ (citable divertor density) | 108 | **0** | 0.0717 at [15,5] |

157 of the 202 are at Te < 2 eV; 105 of the 202 have τ_Lyα(5 cm) > 1; 26 have
> 100.

**A11 cannot be written as "202 of 680, worst 38.7%" and also keep §3.5.3.**
Inside the defensible range: *"45 of 448 points above 2 eV; worst 17.5% at
Te = 2.02 eV, ne = 1.93×10¹³; at citable divertor densities the ELM-averaged
lower bound peaks at 7.2%."* That is still a thesis result. It is not the
abstract sentence currently planned at `thesis_ready.md:328`.

### 4.2 Transport, which cannot be settled inside a 0-D model

τ_QSS at [0,4] is the CR ionisation time of a *stationary* neutral. In an ITER
divertor the ground-state neutral population at a fluid element is not set by
ionisation: a 1–3 eV D atom crosses a 10 cm plasma in **6–10 µs**, against the
model's **233 ms**. For the [0,4] lower bound to fall below 10% an effective
ground-state renewal time of ≈ 26 µs suffices. Neutral transit is faster than
that.

`thesis_ready.md:386-392` states "uniform plasma, no transport". That assumption
is load-bearing for the headline and it is not a small correction — it is the
difference between 233 ms and ~10 µs. **This, not closure, is the real threat
to the 38.7%.**

What would settle it: a two-region or SOLPS-coupled calculation in which n(1s)
is set by recycling influx rather than local CR balance. Absent that, the honest
claim is conditional: *if* the ground-state neutral density is frozen for the
duration of the event, the error is 38.7%.

### 4.3 Molecular channels, flagged independently by two agents

At [0,4] the model's own equilibrium is **96.6% neutral**
(n(1s) = 1.47×10¹⁵ against n_e = 5.18×10¹³). At those conditions D₂, D₂⁺ and
molecular-assisted recombination contribute a large fraction of measured Balmer
emission in real detached divertors — the literature attributes up to 60–70% of
Dα and 10–20% of Dγ at detachment onset. `grep -riE "H2|molecul|MAR|dissociat"
src/rates/` returns nothing. Unverified, unquantified, and it lives directly
under the headline number.

### 4.4 The ridge location is proportional to the assumed neutral density

Scaling n(1s) by a factor, exact within the two-channel split since
$f_p = a_1 n_g/(a_0 + a_1 n_g)$:

| n(1s) scale | ridge density of \|f₃−f₄\| at Te = 2.02 eV |
|---|---|
| ×0.01 | ≤10¹² (off grid) |
| ×0.1 | 2.28×10¹² |
| **×1 (CRE)** | **1.68×10¹³** |
| ×10 | 2.48×10¹⁴ |
| ×100 | ≥10¹⁵ (off grid) |

**One decade in n(1s) moves the ridge by roughly one decade in n_e.** The
agreement with Griem at ×1 is therefore conditional on the model's own local
CRE ionisation balance, which has no transport and no recycling — in a divertor
the neutral density is set by recycling, which is the entire physics of the
regime being invoked.

"The ridge sits at n_e ≈ 2×10¹³" is a statement about *this model's ionisation
balance*, not about a divertor. Quote it with that condition or not at all.

This is the quantified form of the open r₁ deficit in `findings_09` §9 item 4:
the model's low-p r₁ values sit a factor 8.3 (p=3) and 4.4 (p=4) below
Fujimoto Table 4.1(b) at ne = 10¹², one grid interval from the ridge.

---

## 5. Script-level defects that do not change a PART A number

Bounded, quantified, and none of them affects a headline today.

| # | file:line | defect | impact |
|---|---|---|---|
| 5.1 | `verify_plateau_bridge.py:282` | plateau duration is **linear** in undocumented `TOL_PLATEAU = 2e-3` (verified to 0.6%); no scan performed | do not quote "3.4% of τ_QSS"; the plateau *start* is log-sensitive and is safe |
| 5.2 | `verify_plateau_bridge.py:1788-1827` | `k = 0.94033` is a log-spaced-sampling artifact; max residual = 96% of fitted range; expected slope from the script's own τ_eff/τ_QSS = 1.0102 is 0.990 | quote τ_eff/τ_QSS = 1.010 instead |
| 5.3 | `verify_plateau_bridge.py:872-937` | prints the **post-step** spectrum (`L[24,5]`, M = 8242.74) under a header reading "benchmark [23,5]" | a reader copying M = 8243 is 21% low against CLAUDE.md's 9982; relabel |
| 5.4 | `grid.py:73` | `NE_RIDGE_M3 = 1.931e19 # measured, verify_eps_gridmap.py` — that script reports 5.179e19 | the "factor 5.2 below the detached band" becomes 1.9 if j=4 is used; qualitative conclusion survives, quoted factor does not |
| 5.5 | `audit_writers.py:111-119` | reads mode from `n.args[1]`, correct for `open(p,"w")` but wrong for `p.open("w")`; `.write`/`.tofile` absent from `WRITE_CALLS` | **four live CSV writers invisible**, including `verify_plateau_gridmap.py:362`, the producer of `plateau_gridmap.csv` (source of A8/A9/A11). Do not cite this script as single-writer evidence |
| 5.6 | `preflight.py:40-47` | `SCRIPTS` names `verify_ch3_groupB.py` and `verify_grid_coverage.py`; on disk they are `verifych3_gb.py` and `grid.py`. Miss downgraded to non-blocking `NOTE`, then prints "All checks passed" | two of six scripts unchecked; per CLAUDE.md rule 2 this must be `bad()`, not `note()` |
| 5.7 | `verify_plateau_bridge.py:519-556`, `make_ch3_figures.py:160-162` | hardcoded `ib, jb = 23, 5` behind `assert`, which `python -O` deletes | zero impact today (Te[23] = 2.9471 verified); `verify_ch3_claims.py:94-98` does it correctly by argmin |
| 5.8 | `verifych3_gb.py:86-87` | hardcodes `g = 0` instead of `ctx.ground_index` | zero impact today (`ground_index == 0` verified); silent nonsense if state ordering changes |
| 5.9 | `make_ch3_figures.py:155` | isolation gate `if iso_gr.min() < 10.0` against a measured minimum of 24.33 | gate is 2.4× slacker than the quantity it guards; no source for 10.0 |
| 5.10 | `make_ch3_figures.py:66` vs others | `CHI_H = 13.605693` vs `13.605693122994` | 1.2×10⁻⁷ relative — 7th significant figure, immaterial. Hygiene only |

### 5.11 The ε_step census is ten, not six

B6 says "at least six structurally distinct definitions". The full census is
**ten**, and two of them are mutually inconsistent hardcoded fits:

| # | file:line | formula | matches PART C? |
|---|---|---|---|
| 1 | `verify_eps_gridmap.py:123` | `abs(R_old/R_new − 1)`, shell ratio | **yes** |
| 2 | `verify_plateau_gridmap.py:208` | `R_old/Rq − 1` | **yes — identical to #1** |
| 3 | `qss_analysis.py:194-197` | `max_p abs(r_act−r_qss)/(r_qss+1e-60)`, r = n_p/n_1S | no — this is the A7 metric |
| 4 | `test_scaling.py:183` | same shape as #3, interpolated L | no |
| 5 | `S_criterion.py:158` | `max_p abs(n_old−n_new)/n_new` — **raw populations** | no; docstring:127 claims ratios, code uses populations |
| 6 | `S_criterion_fixed.py:~392` | `max_p abs(r_old−r_new)/r_new` | no |
| 7 | `S_criterion_3P.py:149` | `abs(r_old−r_new)/r_old`, single state | no — **denominator flipped** |
| 8 | `unified_scaling.py:339` | Hα emissivity | no |
| 9 | **`verify_partition.py:134`** | **`1.53*exp(−0.37*Te) + 0.01`** | **hardcoded fit, no data** |
| 10 | **`plot_results.py:459`** | **`1.52*exp(−0.42*Te) + 0.04`** | **a second, different hardcoded fit** |

At Te = 3 eV, #9 and #10 give 0.5163 and 0.4711 — a **9.6% spread**, neither
traceable to any run.

**The nine new scripts are clean on this axis.** Grep for `np.exp(-<number>`,
`polyfit`, `curve_fit` across all nine returns zero hits; the
`verify_partition.py:134` pattern does not recur.

---

## 6. What survived — attacked hard, did not break

1. **The ε definitions do not drift between scripts.** `verify_eps_gridmap.py:114-123`
   and `verify_plateau_gridmap.py:200-209` build R^PE by different-looking code
   (`a·u_old + c` vs `n⁰ + n¹`), verified algebraically identical; numbers agree
   to 5 digits at both test points (0.180728 vs 18.073%; 0.063612 vs 6.361%).
2. **The `tanh(|Δ|/4)` bound is real mathematics, not a fitted constraint.** It
   follows exactly from each f_m being a unit-width logistic in ln u shifted by
   `Δ = ln[(a₃/a₄)/(c₃/c₄)]`. `make_ch3_figures.py:386-392` checks peak *and*
   area against closed form to 5×10⁻³ and `:419` checks peak *location* to two
   log-grid steps — which together make the hardcoded `bb = logspace(-2,8,600)`
   range non-load-bearing, since a truncated range would fail the area check.
   A severe check that passes.
3. **All residual gates raise, none warn.** `res_a`, `res_c` > 1e-10 →
   `RuntimeError` in ridge, bridge and figures. Superposition, reduced-vs-full
   R, non-negativity, sign of ground displacement, non-decaying modes, complex
   eigenvalues — all raise. No `try/except` swallows anything in the four
   numeric verification scripts.
4. **Zero occurrences of `np.nan_to_num`, tolerance-widening or outlier-dropping**
   in all nine audited files.
5. **No magnitude filter is re-implemented.** None of the nine contains
   `eigs[eigs < -1.0]` or an equivalent. Checked by grep on `eigs[`, `< -1.0`
   and equivalent slicing.
6. **Numerical conditioning.** Eigenvalue condition numbers 1/|yᴴx| are
   1.71/1.98 at [23,5], 3.26/2.59 at the cold corner, 1.16/1.27 at [49,7] — all
   O(1) despite ‖L‖₁ = 2.0×10¹⁴. Under 20 random 10⁻¹³ relative perturbations of
   every entry, τ_QSS moves 3.3×10⁻⁸ relative at the benchmark and 4.6×10⁻⁴ at
   the cold corner. The caveat against quoting "67.2 s" to three figures is
   *physics* (d ln τ_QSS/d ln Te = −13.3), not roundoff.
7. **A8, A9, A11 reproduce bit-for-bit.** `verify_plateau_gridmap.py` and
   `verify_divertor_map.py` reproduced their committed CSVs byte-for-byte
   (680/680, min amplification 1.44738, medians 12.70/10.77; A9 ratios
   0.7659–1.3891, medians 1.0310/0.9851). A12 reproduces exactly: 19/400 points,
   19/19 with τ_relax^new == τ_QSS^old.
8. **The 104 excluded no-window points contain zero breakdowns** — all at
   Te ≥ 3.24 eV, ne ≥ 5.18×10¹³, max lower bound 0.0011. Including them changes
   202/680 (29.7%) to 202/784 (25.8%). Report the fraction with its denominator
   named; the count is safe.
9. **Dimensional and limit checks.** ε dimensionless throughout; lower bound →
   ε_plateau as τ_d → 0 and → 0 as τ_d → ∞; ε(t=0) = ε_step exactly at all seven
   integrated points; zero step gives exactly zero. n(1s)/n_ion × n_e at [0,7] =
   2.05×10¹⁶ against `CH5_EVIDENCE.md:282`'s independently measured 2.0×10¹⁶ —
   **the s⁻¹ reading of `S_grid` (B8) is confirmed by physical cross-check, not
   just inspection.**
10. **A9's validity window is respected by the published map.** Measured |ln x|
    over all 400 points at a one-interval step: **0.1240 to 0.6823**, matching
    A9's stated window to four digits. Neither A10 nor A11 uses the linearisation
    to compute anything — ε_plateau comes from exact linear solves (superposition
    residual 3.1×10⁻¹⁴). The linearisation appears only in the interpretation,
    inside its window.

---

## 7. Three framing statements that must change

### 7.1 "Its height falls with Te" is 85% a step-convention artifact

At the ridge column (j = 3, heating, `plateau_gridmap.csv`):

| Te | \|ln x\| | \|f₃−f₄\| | ε_plateau |
|---|---|---|---|
| 1.000 | 0.678 | 0.427 | 0.371 |
| 1.600 | 0.446 | 0.463 | 0.232 |
| 2.947 | 0.276 | 0.437 | 0.111 |
| 5.179 | 0.179 | 0.390 | 0.071 |
| 9.541 | 0.124 | 0.353 | 0.044 |

ε falls **8.4×**; |ln x| falls **5.5×**; the intrinsic sensitivity |f₃−f₄| is
**flat to ±13% and non-monotonic** (it peaks at 1.6 eV). "The error decreases
monotonically in temperature" is a statement about how far a fixed 4.81% Te step
moves the ground state, not about the diagnostic.

**The physically interesting statement is the opposite of the one written:** at
the ridge density the diagnostic's sensitivity to ground-state lag is
essentially temperature-independent over the whole decade. `CH5_EVIDENCE.md:55-58`
already carries this ("76% displacement"); `thesis_ready.md:288-290` does not.

### 7.2 "Roughly fixed density" needs its resolution stated

Sub-grid (parabolic in log–log) the ε peak runs 3.66×10¹³ at 1 eV → 1.43×10¹³ at
6.5 eV → 1.51×10¹³ at 9.5 eV (range 2.56×); the |f₃−f₄| peak runs 5.35×10¹³ →
1.54×10¹³ (range 3.48×). Both drift **downward** with Te, while Griem's criterion
predicts the crossing density ∝ √Te, i.e. **upward** by 3.2×.

The drift is inside one grid interval (2.68×) so it cannot be resolved. Do not
write "fixed"; write "constant to within one grid interval, a factor 2.68, over
1–10 eV" — which is what `CH5_EVIDENCE.md:63-65` already says. The reason the
drift runs the "wrong" way is §4.4: n(1s)/n_ion collapses from 40 to 1.1×10⁻⁵
across the grid, dragging the supply-channel crossover down in density faster
than the Griem boundary drags it up.

### 7.3 The ridge belongs to the (3,4) pair, not to "the Balmer diagnostic"

An Hα/Hγ diagnostic (n=3/n=5) has its worst density at **7.2×10¹²**, a factor
2.68 lower. Write "the Hα/Hβ ratio is least reliable at…", never "the Balmer
diagnostic is".

---

## 8. Traps — numbers that are correct but will be misread

1. **`ramp_vs_step.csv`'s De = τ_relax/τ_d ≈ 10⁻⁵ and "suppression ≈ 10⁻⁵".**
   This is the suppression of the *fast* excited-manifold step error. The
   plateau error is governed by the slow ground-state response,
   De_slow = τ_QSS/τ_d = **2332** at [0,4] — a true step. Any finite ramp with
   τ_relax ≪ t_ramp ≪ τ_QSS reaches the same plateau, so the step idealisation
   *is* safe — but not for the reason that file suggests. **Never quote it for
   or against A11.**
2. **Three values of M at "the benchmark point":** 9982 (`CLAUDE.md`, from
   L[23,5]), 8243 (`divertor_map.txt`, post +4.81% step, L[24,5]), 4856
   (`plateau_slowmode.txt`, post +0.6 eV step). Each individually correct; none
   distinguished outside Chapter 3. This is what makes A11's benchmark row
   (τ_QSS = 1.8494×10⁻⁵ s, not A1's 2.2728×10⁻⁵ s) look like an error when it
   is not.
3. **The ±5% step is not an ELM.** |ln x| ∈ 0.124–0.682 for one interval; a
   four-interval step gives 0.516–2.563 and ε_plateau at [15,3] rises from 18.1%
   to 81.1%. On step size alone **A11 understates**. But an ELM also raises n_e,
   and the map steps Te at fixed n_e — joint (+1 Te, +1 ne) steps give
   comparable ε_plateau but a *time-averaged* bound that can fall 4× (at [23,5],
   0.0117 → 0.0030) because higher n_e shortens τ_QSS. **A joint-step map has
   not been run** and an examiner who knows ELM physics will ask for one.
4. **The 10% threshold is doing real work.** At 5% / 10% / 20% the count is
   **348 / 202 / 61**. The count is not robust to the threshold; the worst case
   is.
5. **`CH5_EVIDENCE.md`'s four-interval numbers (58.5%, 81.1%, 190.4%) are exact
   solves but sit 3.8× beyond A9's validated ceiling** (|ln x| median 1.04
   against the 0.682 window). The values are not wrong; any sentence explaining
   them via |f₃−f₄|·|ln x| is outside where that was checked. Keep the
   "do not use the linearised form" instruction attached wherever they travel.

---

## 9. Publication assessment

**Not a divertor paper.** No molecules, no transport, optically thin where
τ_Lyα = 114 cm⁻¹, single species. Never quote 38.7% and "divertor" in one
sentence. PPCF / Nuclear Fusion / NME referees are drawn from the
Verhaegh–Lipschultz–Wijkamp community and will ask about molecules within two
paragraphs.

**Not a QSS-breakdown paper** — `HANDOVER.md` PART 2 already says so, and
`findings_09` §1 shows the QSS closure is exact to 10⁻⁸ on the plateau at the
very point carrying the 38.7%.

**Not a paper about M.** Greenland, *On the validity of collisional–radiative
models*, J. Nucl. Mater. **290–293**, 615 (2001) concludes that validity criteria
"are not related to the equilibrium time-scales" and that "the eigenvalues have
secondary importance", in general form, for arbitrary CR systems. That is A11's
headline, asserted 25 years earlier. The thesis can still contribute by
*quantifying* rather than discovering.

**What is genuinely new**, as far as the literature search found:

1. The exact logistic form $f_m(x) = 1/(1+e^{-(x-x_m)})$, so the two ground-fed
   fractions are the **same curve displaced**, with $x_m = \ln(c_m/a_m)$.
2. The closed-form bound $\max|f_3-f_4| = \tanh(|\Delta|/4)$ and its corollary
   $|d\ln R/d\ln b_1| < 1$ — a line ratio can never respond to the ground-state
   reservoir faster, in relative terms, than the reservoir itself moves.
3. The exact factorisation and its two-axis attribution: across density the
   structure is in $\bar S$ (varies 7.11×) and not in $\Delta\ln u$ (1.01×);
   across temperature it is 76% in $\Delta\ln u$.

**Target: Journal of Physics B or JQSRT**, contingent on resolving the r₁
benchmark (§4.4). Title along the lines of "A closed-form bound on the
ground-state sensitivity of hydrogen line ratios." One paper, not two.

**Two citations that must be added.** `Sawada & Fujimoto, Phys. Rev. E 49, 5565
(1994)` — same system, same abrupt-step experiment, same question, and the same
answer structure (response set by the relaxation time of Griem's boundary level,
which is exactly what `verify_boundary_descent.py` measures) — **appears nowhere
in the repository.** Not citing the single most directly overlapping prior work
reads as avoidance at review. And `Verhaegh et al., PPCF 61, 125018 (2019)`,
which already treats the neutral fraction (this work's `u`) as a free
Monte-Carlo'd parameter, so the contribution is a closed form and a map rather
than a discovery.

---

## 10. Open items

1. **Optical thickness on ε_plateau.** The escape factor is bounded (§4.1) but
   its effect on f₃, f₄, ε is not computed. Needed: rerun the matrix build with a
   Holstein/Doppler escape factor on the Lyman series —
   `src/analysis/escape_factor.py` already exists — and recompute the map for
   Te ≤ 2 eV. **Until then Te ≤ 1.6 eV is not quotable.**
2. **The Ly-α contradiction.** `thesis_ready.md:388` says τ_relax is "insensitive
   to Ly-α trapping to within 1% even at Θ_P = 0.012"; `derivation_07:299-300`
   says at Λ = 10⁻³ τ_relax → 2.05 µs and M falls **158×**. Both are in
   `outputs/`. One will be quoted at the defence and they cannot both stand.
3. **The r₁ deficit vs Fujimoto Table 4.1(b)** — factor 8.3 at p=3, 4.4 at p=4,
   still open (`findings_09` §9 item 4). §4.4 gives it a sharper form: report
   ∂(ridge location)/∂(r₁ scaling) explicitly. If a factor-3 r₁ change moves the
   ridge by more than one grid interval, the ridge *location* must be withdrawn.
   The *mechanism* (§2) is independent of this and survives either way.
4. **Joint (Te, ne) ELM step map.** Costs the same as the existing one and
   closes trap §8.3.
5. **Transport** (§4.2) and **molecular channels** (§4.3) — cannot be settled
   inside this model. State as conditionals.
6. **`verify_bundling_psm20.py` has still never been run**, and
   `chapter5_C5D.tex §sec:ridge_alternatives` names truncation as an untested
   alternative explanation. One command closes it.
7. **CLAUDE.md's known-issue table is stale.** It warns that `qss_analysis.py`
   contains `eigs[eigs < -1.0]`. It does not — line 137 now reads
   `eigs[eigs < 0.0]`, and `validate_gates.py:403` likewise. The filter is live
   at **`src/rates/solve_cr.py:269`** and **`src/rates/check_mz.py:10`**; the
   table should name those. Related: commit `ffe1768`'s message claims the
   removal was done in both files, but its `solve_cr.py` diff changes only a
   units comment and `check_mz.py` is not in the commit at all.
8. **`qss_analysis.py:139` and `validate_gates.py:404`** both do
   `if len(neg) >= 2:` with **no else**, silently leaving τ_QSS = τ_relax =
   M = **0** at any point with fewer than two negative eigenvalues. A wrong
   result there looks like zeros and nothing raises.
9. **B7 is unresolved.** Three grid paths, two writers each, "D.4 NOT
   IMPLEMENTED", six downstream readers. `qss_analysis.py` also bypasses
   `cr_context.py`, hardcoding cwd-relative paths at lines 71–77 — outside the
   provenance gate CLAUDE.md rule 1 mandates.

---

## 11. Actions, in order

1. **Demote A8, A9, A10, A11 from ✅ to ▶️** and strike the W1–W6 sentences. A
   register carrying retracted claims under a verified mark is worse than no
   register.
2. **Fix A1's τ_QSS floor** — 75.4 ns unrestricted, or state the M > 900 scope
   on all six bounds.
3. **Add column j = 4 to A10's table**, or state the migration below 1.15 eV.
   Reconcile with A11, which already sits at j = 4.
4. **Delete the detachment sentence** from A10, `chapter3.tex:1349` and
   `thesis_main.tex:336`; adopt the `CH5_EVIDENCE.md:85-87` wording plus the
   n(1s)-proportionality caveat from §4.4.
5. **Restate A11 inside §3.5.3:** 45 of 448 above 2 eV, worst 17.5%, zero
   breakdown at citable divertor densities where the worst is 7.2%. Report 38.7%
   separately as the asymptotic cold-edge value with the τ_Lyα > 100 flag.
6. **Promote the mechanism (§2) to the headline of A10** — three independent
   routes, two refuters that did not occur.
7. **Add §3.1 and §3.2 as positive evidence.** Two attacks survived, quantified,
   belong in the defence.
8. **Replace "height falls with Te"** with the |ln x| decomposition and the
   flat-|f₃−f₄| statement.
9. **Fix the traps:** relabel `verify_plateau_bridge.py`'s spectrum block as
   post-step L[24,5]; stop quoting plateau duration and k = 0.94; retrace
   `grid.py:73`.
10. **Persist and stamp** the outputs of `verify_eps_gridmap.py`,
    `verify_ridge_mechanism.py` and `verify_plateau_bridge.py` into `validation/`
    — none of the three currently writes an artifact, so every number they
    produce exists only in a `.md`. `HANDOVER.md` PART 4: *"Two files describing
    the same quantity, neither stamped, is how three retractions happened."*
11. **Add the Sawada–Fujimoto (1994) and Greenland (2001) citations.**

---

## 12. Defense one-liners

- *"Why is your worst case at 1 eV when your own scope statement stops at 2?"* —
  It is not the result; it is the asymptotic edge value. The result is 17.5% at
  2.02 eV, and zero breakdown at citable divertor densities. The 1 eV point has
  τ_Lyα = 114 per cm and the model is optically thin, so it is quoted as a bound,
  flagged.
- *"Doesn't the open boundary invent your long τ_QSS?"* — Tested. The closure is
  exactly particle-conserving and reproduces the recorded 1.6226 s at [0,0] to
  five digits. It changes the breakdown count from 202 to 200 and the worst case
  from 38.68% to 38.56%. The factor is 41 only where τ_QSS already exceeds the
  event by 10³, and 1.00 wherever it would matter.
- *"Is your lower bound actually a bound?"* — Better than that. Direct 43-state
  eigen-propagation gives true/lower between 0.9999 and 1.0066 at seven points
  including the worst. The spectrum has no intermediate mode — the gap between λ₀
  and λ₁ is 3.7×10⁷ — so nothing can decay faster than τ_QSS.
- *"Why should I believe the ridge is physics and not a grid artifact?"* — It
  moves with the shell pair as Griem's n^(−17/2) predicts: measured 7.2 against
  predicted 6.7. Its absolute density matches Griem's LTE criterion for n = 4 to
  6%. And `boundary_descent.csv`, an entirely independent diagnostic, puts the
  boundary level at n̄ = 3.1 at exactly that density, at every temperature.
- *"Isn't 'M doesn't predict validity' just Greenland 2001?"* — Yes, as a
  criterion. This work quantifies it for a specific diagnostic and maps where it
  bites. That is the contribution, and the citation is in the introduction.

---

# ADDENDUM — Open items 1 and 2 closed: Lyman trapping computed

**Run 9–10 September 2026.** Script `src/validation/verify_lyman_trapping.py`,
new, written for this purpose. Outputs `validation/lyman_trapping/`
(`.txt` run log, `.csv` 5440 rows). Canonical `data/processed/cr_matrix/` was
**not touched**; trapped matrices are built in memory from the same rate arrays
via `assemble_cr_matrix.build_L` with a modified rate dict.

## A.1 Three validation gates, passed before any result was read

1. **Oscillator strengths derived from the repo's own A-values reproduce the
   literature.** Inverting the Gaussian-CGS Einstein relation
   $A_{ul} = (8\pi^2 e^2 \nu^2/m_e c^3)(g_l/g_u) f_{lu}$ over
   `A_resolved[1S, :]` gives $f$ = **0.4162, 0.0791, 0.0290, 0.0139** for
   Ly-α, β, γ, δ — the accepted values, to four figures, with no oscillator
   strength imported.
2. **Cross-check against an independent implementation.** The derived Ly-α
   $\sigma_0$ agrees with `escape_factor.lyman_alpha_sigma0` (which reads
   $f_{12} = 0.4162$ from literature) to **0.059%**. This gate is severe: at
   first run it **FAILED at 1156%**, catching a $4\pi$ error from using the SI
   coefficient $2\pi$ with CGS constants. The result below exists because that
   check fired.
3. **The untrapped rebuild reproduces the canonical matrix exactly.**
   `max|rebuilt − canonical| = 0.000000e+00`. Every trapped number is therefore
   a difference against the real matrix, not against a re-derivation of it.

Trapping is applied **self-consistently**: $\Theta_P$ depends on $n(1s)$, which
depends on $\Theta_P$ through the CR balance. The fixed point converged at all
400 points for every slab thickness (median 16–21 iterations); non-convergence
raises rather than returning the last iterate. All 14 Lyman channels are
included (7 resolved nP→1S, 7 bundled n9–n15→1S), not Ly-α alone.

The slab thickness $D$ is a new free parameter the 0-D model does not contain.
It is swept, never fixed.

## A.2 The result: the defensible claim is untouched, the headline is not

ELM recount, $\tau_d = 100\ \mu$s, lower bound > 10%:

| run | all window_ok | Te ≥ 2 eV | Te ≥ 2 eV & ne ≥ 10¹⁴ |
|---|---|---|---|
| untrapped (canonical) | 202/680, worst **0.3868** | 45/448, worst **0.1748** | 0/108, worst 0.0717 |
| trapped D = 1 cm | 176/680, worst 0.2925 | **45/448, worst 0.1748** | 0/108, worst 0.0712 |
| trapped D = 5 cm | 170/680, worst 0.2550 | **45/448, worst 0.1746** | 0/108, worst 0.0694 |
| trapped D = 20 cm | 160/678, worst 0.2237 | **45/446, worst 0.1740** | 0/106, worst 0.0627 |

**Above 2 eV the count does not move at all and the worst case moves 0.5% over
a twentyfold range in slab thickness.** The Te ≥ 2 eV restriction in §3.5.3 was
not a hedge — it is exactly the boundary beyond which trapping stops mattering.
Θ_P(Ly-α) at Te = 2.947 eV runs 0.9999 (D = 1 cm) to 0.973 (D = 20 cm): the
plasma is optically thin there in the sense that matters.

Sensitivity to the neutral temperature is not load-bearing: at D = 5 cm, fixing
T_at = 3 eV (Franck-Condon) instead of T_at = Te gives 173/680 and worst 0.2653
against 170/680 and 0.2550.

## A.3 The 38.7% falls by a factor of 2.5 to 3.3

Tracking the A11 worst point [0,4] (Te = 1.0 eV, ne = 5.18×10¹³, heating):

| D (cm) | ε_plateau | ELM lower bound | τ_QSS (s) | M | f₃ | f₄ | \|f₃−f₄\| |
|---|---|---|---|---|---|---|---|
| **0 (untrapped)** | **0.38690** | **0.38682** | 2.3324e-01 | 3.74e7 | 0.7728 | 0.2862 | 0.4866 |
| 1 | 0.15748 | 0.15739 | 9.0651e-02 | 2.75e6 | 0.6643 | 0.2055 | 0.4589 |
| 5 | 0.13274 | 0.13260 | 4.6940e-02 | 6.13e5 | 0.5967 | 0.1793 | 0.4174 |
| 20 | 0.11570 | 0.11550 | 2.8500e-02 | 2.14e5 | 0.5139 | 0.1501 | 0.3639 |

**The 38.7% becomes 11.6–15.7%.** It stays above the 10% threshold at every
slab thickness, so the *point* still breaks down — but the number quoted in
`thesis_ready.md` A11 and planned for the abstract is a factor 2.5–3.3 too
large, and its size is set by an assumed slab thickness the model does not
contain.

**The mechanism of the fall is not loss of sensitivity.** |f₃−f₄| moves only
6% at D = 1 cm and 25% at D = 20 cm. What collapses is the *displacement*:
trapping lengthens the n = 2 lifetime, which raises stepwise ionisation, which
drops the neutral fraction (n(1s) at [0,3] falls 2.99×10¹⁴ → 7.26×10¹³ from
D = 1 to D = 20 cm) and shortens τ_QSS by 8× at [0,4]. A ground state that
turns over faster is less stale, so |ln x| roughly halves. This is consistent
with the two-axis attribution in `CH5_EVIDENCE.md` §2 — the density axis lives
in $\bar S$, the temperature axis in $\Delta\ln u$, and trapping acts on
$\Delta\ln u$.

## A.4 The cold-row ridge location is not robust; the warm-row ridge is

ε_plateau argmax over the density axis, heating:

| Te (eV) | untrapped | D = 1 cm | D = 5 cm | D = 20 cm |
|---|---|---|---|---|
| 1.000 | **j = 4** (5.18e13) | **j = 2** (7.20e12) | **j = 3** (1.93e13) | **j = 3** (1.93e13) |
| 1.099 | **j = 4** | j = 3 | **j = 2** | **j = 2** |
| 1.600 | j = 3 | j = 3 | j = 3 | j = 3 |
| 2.947 | j = 3 | j = 3 | j = 3 | j = 3 |
| 5.179 | j = 3 | j = 3 | j = 3 | j = 3 |

Below 1.15 eV the argmax wanders across three columns — a factor 7 in density —
as a function of an assumed parameter. **The two coldest rows cannot support a
ridge-location claim at all.** At Te ≥ 1.6 eV the ridge is at j = 3 in every
run, and the values barely move: at Te = 2.947 eV, 0.12166 → 0.12166 → 0.12165
→ 0.12163, and |f₃−f₄| = 0.43391 → 0.43389.

This *strengthens* §2 of this document. The ridge mechanism was verified in
`verify_ridge_mechanism.py` over Te ≥ 2 eV, and that is precisely the range
where trapping is now shown to be irrelevant. The Griem-boundary agreement
(7.2 vs 6.7; 1.93×10¹³ vs 2.05×10¹³) is not an artifact of neglected trapping.

## A.5 Open item 2 resolved: the Ly-α contradiction is a scope collision

`thesis_ready.md:388` ("τ_relax insensitive to Ly-α trapping to within 1%") and
`derivation_07:299-300` ("τ_relax → 2.05 µs, M falls 158×") are **both correct,
in different regimes, and neither states its regime.** Measured τ_relax:

| point | untrapped | D = 20 cm | change |
|---|---|---|---|
| [23,5] benchmark, Te = 2.947 eV | 2.2769e-09 | 2.2947e-09 | **+0.78%** |
| [0,3] cold, Te = 1.0, ne = 1.93e13 | 8.8842e-09 | 2.4289e-07 | **×27.3** |

The "within 1%" claim holds at the benchmark and fails by more than an order of
magnitude at the cold corner. This is the same defect as §1.1: a
regime-restricted number quoted as a global one. **Both sentences must carry
their scope.**

## A.6 What this changes

1. **§10 item 1 is closed.** Te ≤ 1.6 eV is now quantified, not merely
   unquotable: ε_plateau there is inflated by a factor 2.5–3.3 in the untrapped
   matrix, and the ridge location below 1.15 eV is not determined.
2. **§10 item 2 is closed** — see A.5.
3. **The Te ≥ 2 eV result is now robustness-tested against the one physical
   process most likely to invalidate it, and survives with the count unchanged.**
   That is a stronger statement than the thesis currently makes anywhere, and it
   converts §3.5.3 from a disclaimer into a measured boundary.
4. **Action 5 of §11 tightens.** The restated A11 — 45 of 448 above 2 eV, worst
   17.5%, zero at citable divertor densities — is now known to be *invariant*
   under trapping over a twentyfold slab range. Report it as such.
5. **The 38.7% must not appear in the abstract**, and where it appears as an
   asymptotic cold-edge value it must carry both the τ_Lyα > 100 flag and the
   trapped range 11.6–15.7%.

**Caveats.** The slab thickness and the escape-factor geometry (homogeneous
slab, constant emission, isotropic field, Doppler profile, evaluated at the
plasma centre) are assumptions this 0-D model cannot check. Continuum lowering,
opacity in the Balmer series itself, and the transport and molecular objections
of §4.2–4.3 are untouched by this calculation and remain open.

---

# ADDENDUM B — First-principles re-derivation of the Chapter 3 core

**Run 10 September 2026.** An independent agent derived all six central results
from the linear CR system alone, implemented them from scratch, and wrote down
predictions BEFORE opening any project file. Only then was the code read.
Nothing in the repository was modified.

## B.1 The verdict: the mathematics is correct

All six reproduce. Every number in `chapter3.tex` §sec:ground_fed_fraction
reproduces from an independent solve:

| chapter3.tex | independent |
|---|---|
| b₁^CRE = 956 | 956.509 |
| switching points 2.73×10³, 1.91×10⁴ | 2727.3, 19085.5 |
| peak b₁ = 7.2×10³ | 7214.7 |
| f₃ = 0.260, f₄ = 0.048, difference 0.212 | 0.259652, 0.047725, 0.211927 |
| max = 0.4514, benchmark 47% of max | 0.451357, 46.95% |
| R = 0.7778 | 0.777768, by three independent routes |

τ_QSS = 22.73 µs, τ_relax = 2.277 ns, M = 9981.9 — CLAUDE.md's reference values
to four digits. cond(L_EE) = 4.198×10⁴, solve residuals ~10⁻¹⁴; float32 moves
the channels by 1.3×10⁻⁵, so double precision is nowhere near a conditioning
cliff. **Tightening tolerances would change no conclusion.**

Two results are strengthened by the re-derivation:

- **The unit logistic width is not incidental — it is equivalent to result 1.**
  A general Hill function x^m/(K+x^m) maps to σ(m(X−X₀)) with width 1/m. Here
  m = 1 *because* the excited populations are affine in n_g. Width 1 ⟺ Hill
  coefficient 1 ⟺ exact linearity in the reservoir. The whole 42-state network
  enters only through the location x_p.
- **Non-negativity is proved cleanly.** −L_EE is a Z-matrix with strictly
  positive column sums (from Σᵢ L_ij = −S_j n_e), hence a non-singular
  M-matrix, hence (−L_EE)⁻¹ ≥ 0 elementwise. Equivalently L_EE is Metzler and
  Hurwitz, so (−L_EE)⁻¹ = ∫₀^∞ exp(L_EE t) dt ≥ 0 — entry (p,q) is the expected
  time an atom created in q spends in p before leaving the manifold. Measured:
  **0 negative entries at all 400 points**, minimum component +6.019×10⁻¹³.

## B.2 The productive falsified prediction

The agent predicted f₃−f₄ → 0 at high n_e for a structural reason: in LTE both
channels give the same Boltzmann shape, so a₃/a₄ = c₃/c₄ and Δ → 0.
**Falsified.** Δ is nearly *flat*: 0.979 at 10¹², 1.945 at 1.4×10¹⁴, 1.939 at
10¹⁵. The grid never reaches LTE, and because the model is **open** — S_E is an
externally imposed source not tied to n_i by Saha — the two channels can never
merge by construction.

**Consequence, and it sharpens the thesis's central claim.** Because Δ is flat
across n_e above j ≈ 2 (tanh spans only 0.394 → 0.450, a 14% span) while the
operating point ln(u_CRE/u_peak) sweeps from +1.29 to −3.29, **the density
maximum is a *position* effect, not a *span* effect.** The tanh cap is not what
locates it. At Te = 2.947 eV:

| j | ne | Δ | tanh(Δ/4) | ln(u_old/u_peak) | \|S̄\| | ε |
|---|---|---|---|---|---|---|
| 0 | 1.00e12 | 0.9655 | 0.2368 | **+1.2877** | 0.1754 | 4.873% |
| 2 | 7.20e12 | 1.6662 | 0.3940 | **+0.3842** | 0.3884 | 11.053% |
| 3 | 1.93e13 | 1.8857 | 0.4394 | **−0.2487** | 0.4261 | 12.166% |
| 5 | 1.39e14 | 1.9403 | 0.4503 | −1.7695 | 0.2288 | 6.361% |
| 7 | 1.00e15 | 1.9368 | 0.4496 | −3.2910 | 0.0668 | 1.831% |

**The maximum sits where ln(u_CRE/u_peak) changes sign.** This is more specific
than "sensitivity-dominated" and should replace it in Chapter 5. It also
forbids the inference that the effect vanishes at high n_e — it does not, on
this grid.

## B.3 Six required changes

1. **The "linearisation understates" claim is false as one-sided.** It holds for
   the *interval-mean* form and only on a rise. For the *endpoint* form actually
   computed at `verify_plateau_gridmap.py:212` it understates in only **219 of
   392** heat steps, and at the benchmark it **over**states (ratio 0.9459). The
   gridmap's own recorded output already contradicts a one-sided claim:
   `min 0.8417 median 1.0310 max 1.3891`. **Name which linearisation is meant.**
2. **The tanh gate is not a severe check.** Fed corrupted input it still
   passes: swapping shells 3↔4 leaves |Δ| and |ln r| unchanged; swapping c₃↔c₄
   likewise. It is a theorem given a, c ≥ 0, so it can only fail on an
   arithmetic bug. The docstring correctly calls it a wiring check; the printed
   line "tanh bound honoured at all 248 points" reads as validation and should
   not. **The severe checks are the superposition residual (3.075×10⁻¹⁴) and
   the reduced-vs-full R test** — those *do* catch the 3↔4 swap.
3. **Quote the density maximum as a range.** 1.931×10¹³ is a grid node; a
   parabolic fit in ln n_e puts the vertex at **1.65×10¹³**, a 15% shift, and
   the 8-point grid resolves only 0.43 decades per column. Write
   **7×10¹² – 5×10¹³ cm⁻³**. This is the second independent agent to reach this
   conclusion — see the figure builder's finding that ε varies only 6–22%
   across a factor 7.2 in density.
4. **The two scripts use different point sets.** `verify_plateau_gridmap.py`
   excludes points with no plateau window (M ≤ 900); `verify_ridge_mechanism.py`
   applies no such filter. **54 of 400 points have M ≤ 900**, all at j ≥ 4. The
   ridge script's j=6 and j=7 statistics — including the modal-argmax |Δln u|
   figure the verdict leans on — include points where the plateau state being
   computed does not physically exist.
5. **State that Δ is flat and the maximum is a position effect** (B.2).
6. **Downgrade the novelty wording.** See B.5.

## B.4 The opacity chain — state it where the logistic is derived

This is the most defence-dangerous item and it is currently only in Chapter 6:

$$\text{radiation trapping} \Rightarrow A = A(n_g) \Rightarrow \text{linearity in } n_g \text{ broken} \Rightarrow \text{Hill coefficient} \neq 1 \Rightarrow \text{width} \neq 1 \Rightarrow \text{bound becomes } \tanh(m\Delta/4),\ m \text{ unknown}$$

The exactness of the two-channel split requires L_EE, L_Eg and S_E to contain
no n_g. Trapping makes the effective A coefficients depend on n(1s) and breaks
that. With τ_Lyα = 114 cm⁻¹ at the cold corner this is not hypothetical. **It
belongs beside the derivation, not only in the limitations chapter.**

Similarly, **molecular channels would add a *third* channel**, which does not
perturb the two-channel split — it destroys its functional form. Chapter 6
should say the framework is two-channel *by construction*.

The fixed-ion-reservoir assumption, by contrast, is bounded: total bound
population per unit n_i at the benchmark is 8.925×10⁻⁴, so freezing n_i costs
at most **0.09%**. Quote that bound.

## B.5 Normalisation dependence, and novelty

**Δ and the tanh bound are invariant under rescaling the n_g/n_i normalisation
(verified exactly under factors of 10¹⁰ and 10⁻⁷); x₃, x₄ and f_p individually
are not.** Say so next to the 0.212. Δ is also insensitive to ℓ-weighting:
1.94561 (population sum), 1.94474 (p-states), 1.94447 (d), 1.94471 (s),
1.94574 (g-weighted) — a **0.07% spread**, so the bound is not a weighting
artifact.

**Novelty, revised.** The two-channel split is correctly credited to
Bates/Kingston/McWhirter and Fujimoto/McWhirter in `chapter3.tex:1372-1377`.
The **logistic form is a one-line rewriting** of that published formula, is the
same object as the Michaelis–Menten saturation fraction whose elasticity bound
is textbook, and is structurally the Rasch/1PL curve family. Write "not
previously stated in this form in the CR literature", **not** "genuinely new".
For the tanh bound the agent found no prior statement — but searched only
Crossref and OpenAlex, which retrieve equations buried in textbook chapters
poorly. **Before claiming novelty, read Fujimoto (2004) Ch. 4 and the Fujimoto
JPSJ *Kinetics of Ionization-Recombination* series I–IV (1979–1985) directly**;
that series develops the r₀/r₁ crossover in most detail and is the most likely
place for the switching point c_m/a_m to already appear.

The defensible claim is the **application**: that this bounds the diagnostic
error of a hydrogen line ratio with respect to the ground-state reservoir,
uniformly over atomic data.

Sawada & Fujimoto (1994) is confirmed to carry *"Validity range of the
quasi-steady-state solution of coupled rate equations"* **in its title** — the
thesis's exact question, 32 years earlier. It is now cited in Chapter 1 but
with an open `\todo` demanding the precise statement of what it established.
**That `\todo` is the single largest unresolved publication risk in Chapter 1.**

---

# ADDENDUM C — Gate 1 architecture pass, and three register corrections

**10 September 2026.** The Gate 1 architect pass produced
`outputs/claim_hierarchy.md` (thesis claim, claims A–F, a seven-chapter map with
per-chapter forbidden claims, four-step chains for thirteen results, a ten-rung
validation ladder, and fifteen approximations each with its expected failure
mode). Three items from that pass are settled here.

## C.1 The condition-number claim is correct, and now has provenance

`chapter3.tex:719-721` states that cond(L_EE) lies between 1.48×10³ and
1.74×10⁵ across the grid. No script computed it, which made it the most
quotable unsupported number in written LaTeX. Computed directly from the
canonical matrix over all 400 points:

| | measured | claimed | agreement |
|---|---|---|---|
| minimum | **1.4821×10³** at [0,0], Te = 1.000 eV | 1.48×10³ | +0.1% |
| maximum | **1.7436×10⁵** at [33,7], Te = 4.715 eV | 1.74×10⁵ | +0.2% |
| median | 2.1044×10⁴ | — | — |

**The claim was never wrong; it was unsupported.** It now needs a producing
script so the number is reproducible rather than merely correct.

## C.2 B8 is resolved and should leave PART B

The `S_grid` units mislabel is gone. `assemble_cr_matrix.py:58,217` now read
`source per unit n_ion [s^-1]`, and no cm³/s label on `S_grid` survives
anywhere. Fixed by commit `ffe1768` — the same commit whose message was found
to be inaccurate about `solve_cr.py`. A commit can carry a wrong message and a
right diff.

## C.3 A5 was marked verified on evidence that says it is not

`thesis_ready.md` A5 was marked ✅ and called "the strongest single check".
`derivation_04_two_timescales.md:375-381` says of the same numbers: *"This is a
reported result, not independently re-run in this session ... Treat it as
strong documented evidence, not as self-verified."* The May run has never been
reproduced against the canonical matrix. **Demoted to ▶️.** A cross-validation
whose earlier half cannot be re-executed is documented evidence, not a check.

## C.4 The structural finding: one result was never falsifiable

Of thirteen major results audited for the four-step chain (physical picture →
mathematics → prediction → numerical test), twelve have all four. Two are
exemplary: the Griem shell-pair test predicted 6.7 and measured 7.2, and the
LTE prediction was falsified in a way that improved the claim.

**R6, the magnitude, is the only result with no prediction step.** Nothing was
ever written down saying how large the error should be, or what size would
refute the picture. That is precisely why it turned out to be linear in an
arbitrary grid step: a number produced with no prior expectation can only be
rationalised, never falsified. The repair is the one in
`outputs/pivot_decision.md`: headline dε/d ln Te, predicted as |f₃−f₄| times
the CRE gain d ln n_g/d ln Te, both independently measurable. That converts the
weakest chain into one of the strongest.

## C.5 Two further items to act on

- **Rung 6 of the validation ladder, state-space convergence, is NOT DONE.**
- **`verify_bundling_psm20.py` is not the one-command win it is billed as.** It
  synthesises grids silently when files are missing and may read zeros for the
  bundled indices, returning a false verdict from missing data. Fix the silent
  fallback before running it.
- Three checks currently printed as validation are theorems or true by
  construction: the tanh gate, Gate A's detailed balance, and Gate C's Saha
  limit whose `approaching` column is computed and then excluded (populations
  reach only 1.5% of Saha).

---

# ADDENDUM D — Gate 3 CR physics audit

**10 September 2026.** Read-only against the canonical matrix. Provenance
anchors reproduced first: τ_QSS[23,5] = 22.728 µs, τ_relax = 2.2769 ns,
M = 9981.9, cold corner 67.233 s, and Chapter 6's "96.6% neutral" at [0,4]
(0.96590, n_g = 1.4669×10¹⁵).

**Nothing in this audit changes f₃, f₄, τ_QSS, τ_relax, M, or the structural
theorems.** What changes is which claims the model is entitled to make about
where those numbers apply, and one claim it should never have made about
itself.

## D.1 The finding that reshapes the scope: quasi-neutrality fails at the cold end

Holding n_e fixed while n_g varies is self-consistent only where the ionisation
degree is high. At Te = 1 eV the plasma is 95 to 98 percent neutral, so a
factor-2 fall in n_g releases about 20 n_e worth of electrons. Imposing nuclei
conservation (Δn_e = −Δn_g) across one 4.81% temperature step:

| Te step | ne | x = n_g,new/n_g,old | required Δn_e/n_e |
|---|---|---|---|
| 1.000 → 1.048 eV | 1.0×10¹² | 0.506 | **+20.0** |
| 1.000 → 1.048 eV | 5.18×10¹³ | 0.508 | **+13.9** |
| 1.000 → 1.048 eV | 1.39×10¹⁴ | 0.507 | **+12.6** |
| 2.947 → 3.089 eV | 1.39×10¹⁴ | 0.764 | +0.0002 |

**68 of 392 one-step operators (17%) require |Δn_e/n_e| > 10%; 36 (9%) require
> 100%.** The model evaluates the new operator at the old n_e at all of them.

**The dilemma the thesis currently presents as two separate survivable
objections.** If the parcel is closed, quasi-neutrality forces n_e up 10 to 20
times at the cold end and every rate in the new operator is wrong by that
factor. If it is transport-fed so n_e stays fixed, the ground state is being
resupplied and is not stale, which is the entire error mechanism. **Both cannot
hold.** §sec:transport and §sec:open_system must be merged into one dilemma.

Above Te ≈ 2 eV, Δn_e/n_e < 10⁻³. **The Te ≥ 2 eV restriction is therefore not
merely defensible; it is the only self-consistent region of the grid.** This
strengthens the pivot in `pivot_decision.md` rather than weakening it.

Supporting: the cold corner is not an ITER divertor state. n_g = 1.47×10¹⁵ at
[0,4] is a neutral pressure of **235 Pa (1.8 Torr)** at T_n = 1 eV, twelve
times the upper end of the ITER divertor design range of roughly 1 to 20 Pa. At
[0,7] it is 3280 Pa, 164 times over. These are fixed points of a 0-D box.

## D.2 The QSS partition is right everywhere. 2s is not a second reservoir.

This was flagged as the potentially most damaging question. It is not damaging,
and the margin is large.

- 2s loss is 99.99% proton ℓ-mixing: |L[2s,2s]| = 1.2564×10⁹ s⁻¹ at the
  thinnest, coldest point against λ₀ = 0.0149 s⁻¹. Separation 8×10¹⁰.
- **A bound independent of the ℓ-mixing rate:** with proton ℓ-mixing deleted
  entirely, the two-photon rate 8.229 s⁻¹ alone is **553× faster** than |λ₀| at
  the cold corner.
- Spectral test over all 400 points: min |Re λ(L_EE)|/|λ₀| = **86.5**, at
  Te = 10 eV, ne = 10¹⁵. No near-degenerate second slow mode, no slow
  eigenvector with dominant 2s weight.
- 2s would become a second reservoir only below ne ≈ 7×10³ cm⁻³, nine orders
  below the grid.

**{1s} is the correct slow subspace. The two-channel decomposition does not
change shape.** Quote the worst-case separation **86.5**, not a benchmark-point
figure.

## D.3 The Fujimoto self-indictment is probably a transcription error

`chapter6.tex:849-868` reports r₁ low by 8.3× at p=3 and publishes it as a
model failure on the two shells the mechanism is built from. The model was
attacked first and did not break; the target was attacked and did.

- **The model's r₁ reproduces analytically.** Coronal r₁(2) from C(1s→n=2) =
  9.2796×10⁻⁹ cm³/s gives 1.37×10⁻⁵ against the full 43-state solve's
  1.66×10⁻⁵. The atomic stack passes three external checks: α_RR(1s) within
  2% of the scaled standard value, Σα_RR(n≥2) = 2.08×10⁻¹³ against α_B ≈
  2.31×10⁻¹³ (the 10% shortfall is exactly the n>15 truncation), and
  S_ion(1s) = 4.864×10⁻⁹ against Janev's ~5×10⁻⁹.
- **The tabulated target demands impossible atomic data.** Fujimoto's quoted
  coronal asymptote requires C(1s→n=2) = 1.44×10⁻⁷ cm³/s at 11.03 eV,
  **fifteen times the accepted value**.
- **Truncation eliminated:** r₁(3) moves 1.2% from n_max = 10 to 15.
- **Lyman trapping eliminated:** Θ = 0.1 brings r₁(2) to 0.92× the table but
  leaves r₁(3) at 0.26× and drives r₀(2) to 8.2, a level above Saha.
- **A single-row density offset reconciles it.** The table's rows run at
  lg n_e = 12, 14, 16, 17, 18, …, non-uniform spacing, the classic setup for an
  off-by-one read. Against the model at 10¹³ instead of 10¹², the p = 2 to 7
  ratios are 0.817, 0.888, 0.836, 0.819, 0.977: **flat to ±10%, against a
  factor 50 spread as published.**

`verify_fujimoto_table41.py`'s own docstring says *"Table values … Re-check
against the book before anything enters the thesis."* It was not done, and it
has entered the thesis. **Withdraw or heavily qualify §sec:r1_deficit until the
table is re-read with the density-row labels verified.**

## D.4 The molecular bound is constructible, and Chapter 6 says it is not

`chapter6.tex:566` states "this work offers no bound on how much, and none can
be constructed from the data in this repository." A bound follows from the two
references cited in the same paragraph. For a given upper level the molecular
emissivity fraction equals the population fraction, and MAR bypasses the ground
state, so f_m → f_m(1−φ_m):

| point | atomic f₃−f₄ | φ₃=0.60, φ₄=0.30 | φ₃=0.70, φ₄=0.45 |
|---|---|---|---|
| cold corner [0,0] | +0.20109 | +0.05411 (×0.27) | +0.03839 (×0.19) |
| ridge [0,4] | +0.05294 | +0.01894 (×0.36) | +0.01402 (×0.27) |

ε_plateau is linear in f₃−f₄, so this is a **factor 3 to 5 reduction of the
cold-corner ε_plateau.** Replace "no bound can be constructed" with that.

## D.5 Charge exchange is correctly absent, and it corrects a number in the thesis's favour

Symmetric resonant CX is the identity operator on the population vector, so
there is nothing to include. CX into excited states is adiabatically dead:
Massey parameter ξ = 26 for a 10 eV deuteron, giving ~e⁻²⁶.

**But CX changes §sec:transport by an order of magnitude, helpfully.** That
section uses free-streaming transit, "6 to 10 µs across a 10 cm plasma". With
CX at n_p = 5.18×10¹³ the neutral mean free path is **1.72 cm**, escape is
diffusive with D = 5.6×10⁵ cm²/s, and the slab escape time is **72 µs**, not
6 to 10. Chapter 6's own threshold for the worst point to fall below 10% is
26 µs. **The result survives, by a factor 2.8, and only because CX traps the
neutrals.** The free-streaming number understates the thesis's own case at
exactly the point that matters.

## D.6 Detailed balance passes, and the two-energy trap is not sprung

Over 2457 excitation pairs × 3 temperatures: max deviation from detailed
balance **7.31×10⁻⁹**, median 7.5×10⁻¹⁰, consistent with the state table's
8-figure rounding. Ionisation against three-body via Saha: **0.999997 for every
one of the 43 states at every temperature, to eight identical digits.** Because
the ratio is state-independent across levels whose exp(I_p/kT) spans e^13.6 to
e^0.06, the same energy is provably used on both sides.

## D.7 Corrections to Chapter 2

- **`chapter2.tex:924`**: the claim that the frozen Debye density varies F "by
  about 30 percent across the grid" is wrong. Evaluating the module's own
  functions, q spans **×3.7 for n=2 and ×50 for n=8** across the grid. But
  rebuilding L with a density-consistent cutoff moves f₃−f₄ by **≤ 0.42%**, and
  scaling all ℓ-mixing rates over ×0.1 to ×10 moves it < 0.3% at the benchmark
  and ridge. **ℓ-mixing is saturated.** Correct the statement, keep the result,
  and report it as the CLAUDE.md pattern: error found, traced, sized, headline
  insensitive.
- **`chapter2.tex:940`**: "It cannot radiate" about 2s is wrong. Two-photon at
  8.229 s⁻¹. Line 192 concedes it two paragraphs earlier. Say instead that its
  only radiative exit is two-photon at 8.2 s⁻¹, ten orders below the
  collisional rates, which makes the argument stronger by supplying the
  n_e-independent floor.
- The "×141 ℓ-mixing effect" in `verify_fujimoto_table41.py` is a comparison
  against a non-physical reference: with ℓ-mixing removed, 2s has no radiative
  exit in the dataset and r₀(2) diverges as n_e → 0. Do not quote 141. The
  genuine measurement is the saturation result.

## D.8 Items closed cheaply

- **Balmer opacity**: τ(Hα, 10 cm) = 6.3×10⁻⁷ at [0,0], 1.0×10⁻³ at [0,4],
  **0.263** at the single worst cell [0,7]. Escape factor ≥ 0.85 there, moving
  the observed ratio ≲ 10% in one cell and nowhere else. Closes
  `chapter6.tex:414`.
- **n_max = 15 convergence**: f₃−f₄ increments decay geometrically with ratio
  ≈ 0.75. Extrapolated error **+0.9% at the benchmark, −1.4% at the cold
  corner, −0.55% at the ridge.** Truncation makes f₃−f₄ slightly too large at
  cold points and too small at the benchmark. This is rung 6 of the validation
  ladder, previously NOT DONE.
- **Terminal-shell over-population measured internally**: comparing each shell
  as terminal against interior gives 9.4× at p=8, 10.7× at p=9, 6.3× at p=10.
  The observed 4.9 to 6.2× excess at p=15 sits inside this band. This closes an
  open `\todo` with an internal measurement rather than an inference from the
  external table.

**A methodological warning recorded by the auditor:** an initial truncation test
that dropped states without removing their loss channels from the diagonal
produced a spurious 22 to 37% jump that looked like a headline finding. It was
an artifact. Anyone repeating the test must add back Σ_dropped L[k,i] to L[i,i].
The production matrix is self-consistently truncated.
