# Thesis-Ready Register

**Compiled 23 August 2026.** What has passed verification and may be written,
what has not, and what each result needs before it is quoted.

**Canonical matrix:** `data/processed/cr_matrix/L_grid.npy`, written
2026-07-21 20:44, SHA-256
`2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`.
Companion `S_grid.npy` SHA-256 `7822f536590c…fb80a`; state index
`23eb18538c1…979a65`.

Nothing enters the thesis without: a named producing script · reproduction on
the student's machine · a sensitivity test against the definitional choices it
depends on · written caveats · a named thesis home · a stated falsifier that
did not occur.

---

## PART A — Ready to write

### A1. The two-timescale structure ✅

At the ITER reference (Te = 2.947 eV, ne = 1.389×10¹⁴ cm⁻³):

$$\tau_{\rm QSS} = 2.2728\times10^{-5}\ {\rm s}, \qquad \tau_{\rm relax} = 2.2769\times10^{-9}\ {\rm s}, \qquad M = 9982$$

Grid-wide: **M ranges 86.77 to 1.72928×10⁹** over all 400 points.
τ_QSS spans 1.18 µs – 67.2 s; τ_relax spans 0.87 ns – 38.9 ns.

**Why it is trusted:** three independent implementations agree bit-for-bit —
`qss_analysis.py`, `validate_gates.py` Gate E, and an unconditional
recomputation stored in `timescales_unfiltered_CHECK.npz`. Framing A (λ₁ of the
full matrix) agrees with Framing B to 0.0098% at the reference and <0.35%
grid-wide.

**Caveats to carry:** τ_QSS is boundary-dependent. Closing the manifold with a
44th ion state changes it from 67.2 s to 1.62 s at the cold corner — a factor
of **41**. τ_relax is unchanged by closure. λ₀ = 2.286 × K_ion(1S)·ne: it is the
CR ionisation time of ground-state hydrogen, **not an equilibration time**.
d ln τ_QSS/d ln Te = −13.3, so "67.2 s" to three figures is not defensible;
"tens of seconds" is.

**Thesis home:** Ch. 3 (definition), Ch. 4 (Gate E), Ch. 5 (map).

### A2. External validation against Fujimoto ✅

Fujimoto, *Plasma Spectroscopy* (2004), Appendix 4B works an example at
Te = 10 eV, ne = 10¹⁸ m⁻³ = **10¹² cm⁻³** — a point that lies exactly on this
grid.

| | Fujimoto App. 4B | this model, grid [49,0] | ratio |
|---|---|---|---|
| excited-state response | ~1×10⁻⁷ s | 3.4057×10⁻⁸ s | 0.34 |
| ground-state depletion | ~1×10⁻⁴ s | 1.7657×10⁻⁴ s | 1.77 |

**Why this matters more than Gates A–E:** every other gate is an internal
consistency check. This is the only comparison against published timescales
from an independent code with independent atomic data.

**Caveat:** Fujimoto's response time is max_p t_tr(p), the 63% rise time of the
slowest-responding level (= t_rl at Griem's boundary level p_G), **not**
1/|λ₁|. Agreement to a factor of a few is the meaningful comparison; exact
agreement would be suspicious. His p_G ≈ 4–5 at that density also matches the
boundary-descent result (5G at low density), by a second independent route.

**Thesis home:** Ch. 4, as a new external gate.

### A3. Fujimoto's two-channel decomposition, realised in this matrix ✅

Fujimoto eq. (4.20):

$$n(p) = r_0(p)Z(p)\,n_e n_z + r_1(p)\frac{Z(p)}{Z(1)}n(1)$$

Under the post-step operator, with the ground density scaled by x relative to
its old value, the excited block splits **exactly**:

$$\mathbf n_E(x) = \mathbf n^{(0)} + x\,\mathbf n^{(1)}, \qquad \mathbf n^{(0)} = -L_{EE}^{-1}\mathbf S_E, \quad \mathbf n^{(1)} = -L_{EE}^{-1}L_{Eg}n_g^{\rm old}$$

with x = 1 the partial-equilibrium (plateau) state and x = n_g^new/n_g^old the
new QSS target.

**Verified:** superposition holds to **3.075×10⁻¹⁴** across all 784
grid-point/direction pairs. The identity ∫(f₃−f₄) d ln x = ln[R(x_new)/R(1)]
holds to **3.3×10⁻¹⁶** across |ln x| spanning 0.124–5.682, confirmed by two
independent quadratures.

With f_p the ground-fed fraction of shell p,

$$\frac{d\ln R}{d\ln x} = f_3 - f_4, \qquad R = \frac{n_3}{n_4}$$

**Note:** the identity check validates *implementation*, not physics — it is
exact by construction. What it establishes is that the coded f₃ − f₄ really is
d ln R/d ln x for the coded family.

**Thesis home:** Ch. 3, as the analytic core.

### A4. The plateau is the partial-equilibrium state ✅

After a temperature step the excited manifold equilibrates within τ_relax to
the pattern slaved to the **old, unmoved** ground state, and sits there until
τ_QSS. That state is computable in one linear solve:

$$\mathbf n_E^{\rm PE} = -L_{EE}^{-1}\left(\mathbf S_E + L_{Eg}n_g^{\rm old}\right)$$

| | analytic | measured (LSODA) | agreement |
|---|---|---|---|
| cold corner [0,0] | 0.240751 | 0.240747 | **6 digits** |
| ITER ref [23,5] | 0.329030 | 0.324170 | 1.5%, explained below |

The ITER gap is window sampling, not model error: extrapolating the decay back
to t = 0 recovers 0.329045 (ratio to analytic **1.000047**); the fitted decay
time is **1.079 τ_QSS** without being told τ_QSS; R² = 0.9999999; and the
observed window deficit 0.98527 matches the predicted 0.98406 to 1.2×10⁻³.

**Window sensitivity (method 4.4):** repeated at window factors 10, 30 and 100.
τ_fit/τ_QSS = 1.0774 / 1.0791 at the reference; every ε value identical. The
factor 30 is a default, not a load-bearing choice — but **30 is the right
default**: at 100 the guard (M > 10⁴) skips the ITER point itself.

**Consequence, practical:** the grid map needs no stiff integration. Three
linear solves per point.

**Thesis home:** Ch. 3 (derivation), Ch. 5 (result).

### A5. May/August cross-validation ✅ — the strongest single check

Two runs, four months apart, different matrices, different observables,
different numerical methods:

| | May (10 May) | August (23 Aug) |
|---|---|---|
| matrix | pre-correction | post-correction |
| observable | A-weighted Hα/Hβ line ratio | n=3/n=4 shell population ratio |
| method | eigen-propagation of the 43-state system | one 42×42 linear solve |
| script | `Balmer_transient_ratio.py` | `verify_plateau_slowmode.py` |

At the ITER reference, ΔTe ≈ 0.609 eV:

| quantity | May | August | difference |
|---|---|---|---|
| τ_slow | 10.470692 µs | 10.47737 µs | **0.064%** |
| τ_fast | 2.161809 ns | 2.157494 ns | **0.200%** |
| M | 4843.49 | 4856.27 | **0.264%** |
| ε_step | 0.03210869 | 0.03210413 | **0.014%** |
| peak / analytic plateau | 0.327019 | 0.329030 | 0.611% |

**The duration test.** Predict how long the error stays above 10%, assuming the
plateau-then-decay picture:

$$t = \tau_{\rm QSS}\ln\!\left(\varepsilon_{\rm plateau}/0.10\right) = 12.478\ \mu{\rm s}$$

May's measured value, by full ODE integration: **12.4794 µs**.

**Report this as agreement at the ~0.6% level, not 0.008%.** The inputs agree
only to 0.06–0.6%, so the apparent 0.008% is cancellation luck. Using May's own
internal numbers throughout gives 12.406 µs, 0.59% off. 0.6% is the honest and
still-strong figure.

**Two things this establishes that neither run could show alone:**
1. The ℓ-mixing correction is immaterial for these quantities — τ_slow moved
   0.064%, ε_step 0.014%, both well inside the <0.85% bound, measured rather
   than argued.
2. The shell ratio and the A-weighted Balmer line ratio agree to 0.014% on
   ε_step, tighter than the 2.2% worst case from the 4F test (A6).

**Why peak < plateau, and it is not a discrepancy:** the plateau is an
asymptote the trajectory approaches on τ_relax while already decaying on τ_QSS,
so the true maximum falls just short. Both numbers are correct; they are
different quantities and the thesis must say which it quotes.

**Thesis home:** Ch. 4, as an internal reproducibility gate.

### A6. The ℓ-distribution is statistical — C8 closed ✅

`Balmer_ratio_sensitivity.py` and `Balmer_transient_ratio.py` build line
intensities as A-weighted sums over resolved channels — Hα from 3S→2P, 3P→2S,
3D→2P; Hβ from 4S→2P, 4P→2S, 4D→2P. A-values are filtered on
`type == "res_to_res"` plus the full (n,ℓ)→(n,ℓ) quadruple, raising unless
exactly one row matches. **Populations come from the solve**; grep for
`(2l+1)`, `statistical`, `stat_weight`, `degenerac*` returns zero hits in both
files. Nothing is assumed.

**4F is correctly absent from Hβ** — 4F→2P is Δℓ = 2, forbidden for E1; 4F
decays only to 3D and is dark in Hβ. The n=4 shell sum used in the analytic work
therefore contains a state the line cannot see.

Measured across the grid: the 4F fraction of the n=4 shell is **0.4361–0.4375**
against the statistical value 14/32 = 0.4375. Statistical to four significant
figures almost everywhere; the 0.3% departure sits at ne = 10¹², exactly where
ℓ-mixing weakens and radiative decay competes.

Consequence: ε(Balmer denominator)/ε(shell denominator) = **0.978 to 0.9999**.
Worst case 2.2%, at the lowest density.

**This is the C8 answer, and it is stronger than "the A-coefficients cancel":**
proton-impact ℓ-mixing drives n = 4 to statistical equilibrium to <0.3%
everywhere on the grid.

**Thesis home:** Ch. 2 (ℓ-mixing), Ch. 3 (observable definition).

### A7. The published error metric measures ground-state lag ✅

The metric behind Chapter 5's breakdown fractions is

$$\varepsilon = \max_p \frac{|r_p^{\rm act} - r_p^{\rm QSS}|}{r_p^{\rm QSS}}, \qquad r_p = n_p/n_{1S}$$

Its denominator moves on τ_QSS while its numerator moves on τ_relax, so every
excited state inherits the same error. Measured at the ITER reference: 0.521
over all states, 0.517 over resolved excited states, 0.484 for n=3, 0.500 for
n=4 — and **0.032** for n₃/n₄, which cancels n_1S. **94% of the reported error
is the common ground-state factor.**

The cancellation arithmetic closes: (1−0.484)/(1−0.500) = 1.032 against a
measured ratio error of 0.032, to two figures.

**A competing hypothesis was tested and refuted.** max_p is attained at n=15
(index 42) at 346 of 400 points, so the metric was suspected of being a
truncation artifact. Recomputing over resolved states only moves the grid mean
from 0.517 to 0.515, and the fraction above 10% **rises** from 88.3% to 91.8%.
Truncation ruled out; ground-state contamination stands.

**Consequence:** Chapter 5's breakdown fractions must be restated on a
ground-free observable, and the old metric goes into the corrections chapter
beside the −46% Hα artifact.

**Thesis home:** Ch. 4 (corrections), Ch. 3 (why the metric was changed).

### A8. The QSS error is not bounded by the step error ✅

Under a controlled ±5% fractional temperature step, over the 680 grid-point/
direction pairs that possess a timescale-separated plateau:

$$\varepsilon_{\rm plateau} > \varepsilon_{\rm step} \quad \text{at 680 of 680 points}$$

Minimum ratio 1.44, median ~12. Comparing endpoints understates the error at
every single point on the grid.

**The sharpest case:** at Te = 6.866 eV, ne = 5.18×10¹³, ε_step = 4.9×10⁻⁵ —
the two QSS targets very nearly coincide — while ε_plateau = 0.045. QSS would be
exactly right if only endpoints were compared; the system still spends four
decades in time 4.5% away from the answer.

**Do NOT quote the amplification maximum.** ε_plateau/ε_step is unstable
wherever ε_step passes through zero; the apparent grid maximum of 1271× occurs
at a point whose ε_step is the grid *minimum* and whose ε_plateau is *below* the
median. Report ε_plateau as the primary quantity and amplification only as a
distribution. Note also that log(amp) = log ε_plateau − log ε_step identically,
so any correlation with ε_step is partly tautological.

**Heat/cool asymmetry is a step-size artifact:** the achieved steps differ
(+4.81% vs −4.59% on a log grid). Matched per grid point and normalised, the
median heat/cool ratio is **1.036**.

**Thesis home:** Ch. 5, central result.

### A9. The mechanism, quantitatively ✅

$$\varepsilon_{\rm plateau} \approx |f_3 - f_4|\cdot|\ln x_{\rm new}|$$

derived from A3 with no fitting. Ratio of measured to predicted, over 680
points: median **1.031** (heating) and **0.985** (cooling), range 0.766–1.389.

**Read the ratio, not the correlation.** corr(log prediction, log ε_plateau) =
+0.989/+0.992 is near-tautological — a good predictor correlates with what it
predicts by construction. The spread of the ratio is the evidence.
|f₃ − f₄| alone gives only +0.66: sensitivity without displacement predicts
nothing.

**Bound:** in this map |ln x_new| spans only 0.124–0.682, so the ±30% claim is
established in the mildly-nonlinear regime only. At |ln x| ≈ 5.7 (reachable with
larger steps) the linearisation is expected to fail — and does: at the cold
corner under a 0.6 eV step, the linear estimate misses by 46×.

**Thesis home:** Ch. 3 (derivation), Ch. 5 (verification).

### A10. The ridge — where the diagnostic is least reliable ✅

ε_plateau has an **interior maximum in density near ne ≈ 2×10¹³ cm⁻³, at every
temperature tested**:

| Te (eV) | 10¹² | 1.93×10¹³ | 1.39×10¹⁴ | 10¹⁵ |
|---|---|---|---|---|
| 1.000 | 0.083 | **0.371** | 0.303 | 0.131 |
| 1.600 | 0.071 | **0.232** | 0.145 | 0.050 |
| 2.947 | 0.049 | **0.122** | 0.064 | 0.018 |
| 5.179 | 0.032 | **0.071** | 0.036 | 0.009 |

Rising and falling by factors of 4–5 either side. The ridge sits at roughly
**fixed density** while its height falls with Te.

**This is the physics of A3 made visible.** ε_plateau tracks |f₃ − f₄|, which
vanishes in both of Fujimoto's limits — fully ionizing (f₃ = f₄ → 1) and fully
recombining (f₃ = f₄ → 0) — and is largest where the two supply channels
compete. In divertor terms that crossover is **detachment**: the Balmer
diagnostic is least trustworthy at the transition it is used to characterise.

**Caveats:** the Te-direction maximum sits on the grid edge (Te = 1 eV), so the
temperature range is truncated and the error may keep rising below 1 eV — which
is detached-divertor territory. The ne direction is not truncated.

**Thesis home:** Ch. 5, the headline figure.

### A11. Divertor-relevant magnitudes ✅ (bounds, not point estimates)

Because the error decays on τ_QSS at some points and is pinned at others, the
time-average over an event of duration τ_d is bracketed rather than estimated:

- **lower (fast-recovery):** ε_plateau·(τ_QSS/τ_d)(1 − e^{−τ_d/τ_QSS})
- **upper (pinned):** ε_plateau

The lower bound is **conservative** — it understates the error wherever the
observable decays more slowly than τ_QSS, as at the cold corner where
τ_fit = 207 τ_QSS. Any breakdown reported at the lower bound is therefore robust.

At ELM timescales (τ_d = 100 µs), lower bound above 10% at **202 of 680**
points; 105 heating points confined to **Te ≤ 2.947 eV, ne ≥ 2.68×10¹²**.
Worst case **38.7%** at Te = 1.0 eV, ne = 5.18×10¹³, where τ_QSS = 233 ms
against a 100 µs event — the ground state cannot recover within the event, and
the two bounds coincide (0.3868 vs 0.3869).

**The ITER reference does not break down:** 1.2% at ELM timescales.

**The claim M does not predict:** M ≥ 86.8 everywhere and reaches 1.7×10⁹ in
the cold corner — the same corner carrying the 38.7% error. Timescale
separation is *largest* where QSS is *worst*. Measured across 680 points, not
argued.

**Thesis home:** Ch. 5, and the abstract.

### A12. The eigenvalue filter — a correction found by audit ✅

`eigs[eigs < -1.0]` in **both** `qss_analysis.py` (~137) and
`validate_gates.py` (~402) discarded every eigenvalue with |λ| < 1 s⁻¹ and
silently promoted the eigenvalue ladder one step, so τ_QSS, τ_relax **and** M
were all wrong at the affected points.

- **19 of 400 points**, Te ≤ 1.389 eV, ne ∈ [10¹², 1.93×10¹³]
- at every one, τ_relax^new = τ_QSS^old **exactly** — the ladder shift confirmed
- M range 1.341–1.01×10⁸ → **86.77–1.72928×10⁹**
- ε_step **bit-identical** at all 400 points; the S-criterion and step-error
  maps were never affected
- ITER reference bit-identical; the correction is local
- regenerated grids bit-for-bit identical to an independent recomputation

Filtered outputs preserved as `*_FILTERED_20260721` as evidence.

**Still live and unfixed:** the same literal appears in `solve_cr.py:269` and
`check_mz.py:10`.

**Thesis home:** Ch. 4, corrections. This is a strength at the defence, not a
weakness — as is the ℓ-mixing F(U_m) error, found, traced to Badnell 2021 Eq. 9,
and bounded at <0.85% on τ_relax anywhere.

---

## PART B — Not ready

| # | Item | What blocks it |
|---|---|---|
| B1 | **Gate D** | Fails at 0% of points, post-correction, with a systematic factor 9–65 peaking near Te ≈ 2 eV. Hypothesis: `SCD_model` sums ionization over the **full** steady state, but ADAS SCD is the *ionizing* coefficient — ionization out of recombination-fed Rydberg states belongs under ACD. Test: recompute using the ground-fed channel n^(1) only. Also: `acd_adas` is loaded and never used, so the ACD half of the gate is unimplemented; and the docstring names files the code does not read |
| B2 | **Figures** | Of ~141 files in `figures/`, only `fig_boundary_descent` postdates the corrected matrix — and it has a known Te label mismatch (3.24 vs 2.947 eV). 18 `Balmer_*` figures and 6 `figures/paper/*` have **no producer in `src/`**. `mz_fig3_M_comparison` is the uncorrected version despite a corrective script existing |
| B3 | **`chapter4.tex`** | Internally contradictory: 22.7 µs at lines 876/993, M = 611 and 15.3 µs / 25.0 ns at 908/1493/1556. Implies τ_relax = 37 ns, a number appearing nowhere |
| B4 | **Mori-Zwanzig** | April, pre-correction. `weekC_summary.txt` claims "M_MZ ~ 12× larger" two lines below its own "M_MZ/M_thesis = 1.1×"; the 12× only holds against the retracted 25 ns. τ_K lives in L_FF, the one block the robustness argument does not protect. Full regeneration required before anything is quoted |
| B5 | **Title and abstract** | Gated on B1 and on the skeptic pass for A10/A11 |
| B6 | **`eps_step` name collision** | At least six structurally distinct definitions share the name across the repo, including a hardcoded fit in `verify_partition.py:134` that depends on no data. Must be resolved in Ch. 3 notation before any number is written |
| B7 | **Two writers, one path** | `qss_analysis.py` and `validate_gates.py` both write `M_grid.npy`, `tau_QSS_grid.npy`, `tau_relax_grid.npy`. Last runner wins; six downstream readers cannot tell whose numbers they hold |
| B8 | **Units mislabel** | `assemble_cr_matrix.py:58,217,347` and `solve_cr.py:131` label `S_grid` as cm³/s. It is **s⁻¹**: S = ne(α_RR + ne·α_3BR), verified numerically, and b = S·n_ion |

---

## PART C — Standing definitions for Chapter 3

Fix these once, in notation, before writing.

| Symbol | Definition | Not to be confused with |
|---|---|---|
| τ_relax | 1/\|λ₁\| of the full matrix (Framing A) | Fujimoto's t_res = t_rl(p_G), a 63% rise time |
| τ_QSS | 1/\|λ₀\|, least-negative eigenvalue, **no magnitude filter** | any value from a filtered spectrum |
| M | τ_QSS/τ_relax. Large = the QSS *necessary* condition holds | the retired inverse convention |
| R | n₃/n₄, or the A-weighted Hα/Hβ line ratio — agreeing to 0.014% at the reference | either taken as the other without the A6 statement |
| ε_step | \|R(old QSS)/R(new QSS) − 1\| — one scalar observable, two QSS states | the max-over-42-states ground-normalised ε in `qss_analysis.py` |
| ε_plateau | same, for the partial-equilibrium state | ε_step; ε_plateau > ε_step at every point |
| f_p | ground-fed fraction of shell p, from the exact two-channel split | anything statistical or assumed |
| b₁ | n(1)/[Z(1) n_e n_z], the ground-state departure coefficient | x = n_g^new/n_g^old, its ratio across the step |

**Scope statements for §1.5:** Maxwellian electrons · T_i = T_e in ℓ-mixing ·
optically thin, with the bound that τ_relax is insensitive to Ly-α trapping to
within 1% even at Θ_P = 0.012 · n_max = 15, n ≥ 9 bundled, with no ℓ-mixing in
the bundled block (`verify_bundling_psm20.py` has never been run) · uniform
plasma, no transport · ground state tracked · **open system: the ion is a fixed
reservoir in b, there is no 44th state, and this is why Gate D cannot compare
directly to ADAS**.

---

## PART D — Next three actions

1. **Gate D**, using the ground-fed channel. It is the only gate comparing
   against external atomic data and an examiner will ask.
2. **Skeptic pass** on A10 and A11 in final form. Named attacks: is the 10%
   threshold doing the work (test 5% and 20%); is the lower bound really
   conservative where the observable is pinned; do the 104 excluded no-window
   points change the counts; is the Te-edge maximum a truncation of the range.
3. **Then the title**, and Prof. Pala.
