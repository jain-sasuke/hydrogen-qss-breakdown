# Findings 09 — The Central Quantity Was Misnamed

**Written 23 August 2026**, after a three-agent skeptic pass returned **REJECT**
on the headline claim and a read of the existing Chapter 2/3/4 LaTeX located the
error in the thesis text itself.

**Status:** the physics is intact and reproduces everywhere. The *name* of the
central quantity is wrong, and the sentence built on it inverts what the work
shows. This note records the diagnosis, the six numbers that must be withdrawn,
what survives, and the corrected claim.

**Provenance for everything below:** `L_grid.npy` SHA-256
`2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`
(2026-07-21 20:44); analysis in `validation/plateau_gridmap/`,
`validation/divertor_map/`, `validation/plateau_slowmode/`; existing thesis text
in `chapter4.tex` (which holds Ch. 2 Atomic Data, Ch. 3 Theory, Ch. 4
Validation).

---

## 1. The finding, in one line

$$\text{What we measured} = \big\|\,\text{QSS answer} - \text{CR-equilibrium answer}\,\big\|, \qquad \text{not} \qquad \big\|\,\text{truth} - \text{QSS answer}\,\big\|$$

Direct integration of the full 43-state system, measuring both errors along the
same trajectory:

| point | error vs CRE (**what was reported**) | error vs QSS at instantaneous $n_g$ |
|---|---|---|
| benchmark [23,5] | $6.34\times10^{-2}$ | $8.66\times10^{-6}$ |
| worst [0,4] | $3.869\times10^{-1}$ | $6.73\times10^{-9}$ |

**The QSS closure is exact to eight significant figures on the plateau, at the
very point carrying the 38.7%.** None of the headline is a failure of
quasi-steady state.

---

## 2. Where the error entered — it is in the thesis text

`chapter4.tex` §`sec:qss_ratio` defines

$$r_p^{\rm QSS}(T_e, n_e) = \frac{\big[\mathbf L_{FF}^{-1}(\mathbf L_{FS}\mathbf n_S + \mathbf S_F n_e)\big]_p}{n_{1s}}$$

and then states:

> *"These ratios depend on $(T_e, n_e)$ but not on the absolute normalisation
> of $\mathbf n$."*

**That sentence is false as written, and it is the whole error.**

The numerator contains $\mathbf L_{FS}\mathbf n_S$ (proportional to the ground
density) and $\mathbf S_F n_e$ (proportional to the ion density). Their *ratio*
sets $r_p$. So $r_p$ is a function of $(T_e, n_e)$ alone **only if**
$n_{1s}/(n_e n_{\rm ion})$ is itself a function of $(T_e, n_e)$ — which is to
say, only under **equilibrium ionisation balance**.

Once $r_p^{\rm QSS}(T_e, n_e)$ is evaluated from a table, CRE has been silently
imposed. The metric

$$\varepsilon(t) = \max_p \frac{|r_p(t) - r_p^{\rm QSS}(T_e,n_e)|}{r_p^{\rm QSS}(T_e,n_e)+\delta}$$

therefore measures the distance to the **CRE** state, not to the QSS state.

**The chapter already contains the honest reading**, two subsections earlier
(§`sec:algebraic_elim`):

> *"$\boldsymbol\Omega_{\rm QSS}$ is precisely the effective rate matrix that a
> plasma modeller using QSS would construct from scratch — it is the algebraic
> form of the ADAS effective ionisation coefficient framework."*

That is the corrected thesis. What has been measured is the cost of the
ADAS-style two-parameter $(T_e, n_e)$ inversion, not the cost of adiabatic
elimination.

**And §`sec:validity_condition` is right**, which is why the closure comes out
exact: $M \gg 1$ everywhere, so the reduced equation is an accurate description
of the long-time dynamics. The thesis tested its own validity condition, passed
it, and then reported a different approximation's error under the same name.

### 2.1 My own note contained the correct statement and I did not act on it

`derivation_07` §5: *"after Stage 1 the excited manifold is in equilibrium with
a **stale reservoir**."* That sentence **is** the statement that the QSS closure
holds and the reservoir is the thing that lags. It was written on 23 August and
then, for four days, its consequence was called a QSS error.

---

## 3. The six numbers to withdraw

Each is wrong for a different reason. Recorded per method 7 rather than silently
corrected.

| # | Claim | Why it fails |
|---|---|---|
| W1 | *"M ≥ 86.8 everywhere"* | $M = 86.77$ is at heat[48,7], which has `window_ok = False` and **is not in the analysis**. The floor over the analysed set is $M \ge 902$ — and that floor is **imposed**, since `window_ok` ≡ M > 900 (win_lo × win_hi = 30×30). Quoting the floor of a set that excludes it, and calling a constructed cut a measurement |
| W2 | *"reaches 1.7×10⁹ in the same cold corner that carries the largest error"* | **Factually wrong.** $M_{\max} = 1.73\times10^9$ at [1,0], $n_e = 10^{12}$; $\varepsilon_{\max}$ at [0,4], $n_e = 5.18\times10^{13}$. **52× apart in density.** At the M maximum, $\varepsilon = 0.12$ — the 75th percentile, 3.2× below the actual maximum |
| W3 | *"conservative lower bound"* | **Not a bound, on either side.** Direct integration shows it *overstates* the true time-average at 330 of 680 pairs (319 of them cooling), because the observable decays **faster** than $\tau_{\rm QSS}$ for every cooling step ($\tau_{\rm fit}/\tau_{\rm QSS}$ down to 0.61). And $\varepsilon_{\rm plateau}$ is not an upper bound either — transiently exceeded at 358 of 680 pairs, by up to **3.45×**. Count is **201**, not 202 |
| W4 | *"confined to $T_e \le 2.947$ eV"* | A **level set of two arbitrary constants**, monotone and knee-free: 10.0 eV at a 1% threshold, 4.72 at 5%, 2.947 at 10%, 1.76 at 20%. At a 20% Te step it covers the whole 1–9.5 eV grid. Not confined in $n_e$ at all. Only 53.7% of the $T_e \le 2.947$ box exceeds. That it lands on benchmark index 23 is coincidence and will not read as one |
| W5 | *"202 of 680 grid points"* | **Category error.** These are (point, direction) pairs: 105 heat + 97 cool over **113 distinct grid points of 400** |
| W6 | *"$\varepsilon_{\rm plateau} > \varepsilon_{\rm step}$ at 100%"* | Specific to the **n=3/n=4 pair**: n=4/n=5 gives 91%, n=7/n=8 gives 53%, n=3 alone gives 73%. A statement about differential ground-feeding between two strongly-differing shells, not a general plateau property. **Also specific to the small step**: at ΔTe = +0.6 eV the same cold corner gives $\varepsilon_{\rm step} = 0.2439 > \varepsilon_{\rm plateau} = 0.2408$ — amplification 0.987, **below 1** |

### 3.1 And the headline I proposed yesterday must also be withdrawn

I recommended $\text{corr}(\log M, \log\varepsilon) = +0.76$ as a stronger
headline than the negation. **Wrong.** Controlling for $(T_e, n_e)$ it falls to
$+0.33$; under quadratic control it **flips sign to $-0.16$**. $M$ is 93%
explained by $(T_e, n_e)$ alone, $\text{corr}(\log M, \log\tau_{\rm QSS}) =
+0.97$, and a bare Arrhenius factor $e^{13.6/T_e}$ — containing no dynamics
whatever — correlates at $+0.71$, essentially matching $M$'s $+0.77$.

**The correlation is a temperature proxy, not physics.**

### 3.2 The M sentence was a strawman, retired months ago in this project

No cited source claims $M$ bounds this distance, and the dossier says the
opposite **twice**:

- `derivation_03` §5: *"$M$ captures only #1… $M \gg 1$ does not guarantee small
  transient error."*
- `derivation_05` §2: *"independent quantities — knowing one tells you nothing
  about the other."*

Settled in July; re-litigated in August as news.

---

## 4. Three framing defects, independent of the numbers

**4.1 The ±5% step is not a controlled perturbation of the driver.**
$\varepsilon_{\rm plateau} \approx |f_3-f_4|\cdot|\ln x|$ with $x = n_g^{\rm
new}/n_g^{\rm old}$, and the gain $d\ln n_g/d\ln T_e$ is not constant:

| Te range | median gain | ground swing for a 4.8% Te step |
|---|---|---|
| 1–1.5 eV | 12.7 | 50–68% |
| 1.5–3 eV | 7.8 | ~30% |
| 3–6 eV | 4.6 | ~20% |
| 6–11 eV | 3.1 | ~14% |

**The cold corner is driven ~4× harder in the ground reservoir.** Part of the
Te-dependence of the error map is this gain, not $|f_3-f_4|$ — consistent with
the script's own diagnostic ($+0.66$ with $|f_3-f_4|$ alone, $+0.99$ with the
product).

**4.2 The ELM time-average is close to a no-op on the counted set.**
LOWER/UPPER > 0.99 at 127 of 202 points, median **0.9971**. Median
$\tau_{\rm QSS}/\tau_{\rm ELM}$ among the 202 is **173** (vs 0.83 among the 478
non-exceeding) — the 202 are effectively *selected by* $\tau_{\rm QSS} >
\tau_{\rm ELM}$. Presenting 38.7% as "time-averaged over a 100 µs ELM" implies
the averaging does work; it contributes a 0.02% correction at that point.

**4.3 The worst case is edge-truncated, and the "interior maximum" print is a
bug.** All eight $n_e$ columns peak at the lowest available $T_e$ index, in both
directions. `verify_plateau_gridmap.py:287` tests `i in (0, len(Te)-1)`, but for
cooling row $i=0$ is dropped (a −5% step from Te = 1.0 rounds back to itself), so
[1,3] is the first available row and is **exactly as truncated as [0,4]**.
$d\ln\varepsilon/d\ln T_e \approx -1.27$ and still rising at the edge. The grid
stops at 1 eV by **hardcoded convention** (`np.logspace(log10(1), log10(10), 50)`
in nine files), not by a data limit — CCC cross sections run to ~960 eV. The
defensible reason to stop is that the model has **no molecular channels**, which
is also a hard limit on what the claim can say. Must be *"at least 38.7%,
truncated by the low-Te edge."* The $n_e$ direction **is** genuinely interior.

---

## 5. Two further defects in the existing thesis text

**5.1 §`sec:eps_bar` assumes what has now been refuted.** It models the decay as
$\varepsilon_{\rm res}e^{-t/\tau_{\rm QSS}}$ and calls the result a
time-average. Direct integration says the observable decays *faster* than
$\tau_{\rm QSS}$ for every cooling step. Note also that the text labels it *"the
Stage 1 QSS error decay"* while using $\tau_{\rm QSS}$, the **Stage 2**
timescale — the label and the constant disagree within one sentence.

**5.2 The 10% threshold does have a stated rationale**, and it should be kept:
matched to combined ADAS PEC (~5%) and atomic-data (~5%) uncertainty, so that
errors below it are buried in irreducible atomic physics. That justifies the
*threshold*; it does not rescue the *contour*, which has no knee (W4).

**5.3 Gate E in the current text is entirely stale.** It reports $M$ from 44 to
309,326, $\tau_{\rm relax}$ 1.65 ns – 9.13 µs, $\tau_{\rm QSS}$ 75 ns – 730 ms,
and Table `tab:M_Te3` carries the retracted 15.3 µs / 25.0 ns / 611 row. Current
values: $M$ from **86.8 to 1.73×10⁹**, $\tau_{\rm QSS}$ up to **67.2 s**,
benchmark row 22.7 µs / 2.28 ns / 9982. Every number in that section changes.

**5.4 `CHAPTER5_CORRECTIONS_REPORT.md` is itself stale.** It reports post-fix
breakdown fractions of 70.9 / 48.0 / 29.1 / 14.3%; `qss_analysis` on the current
matrix gives **60.75 / 42.75 / 27.00 / 14.75%**. Anything quoting 70.9% is a
March number.

---

## 6. What survived — attacked hard, did not break

| Sub-claim | Status |
|---|---|
| Amplification $\varepsilon_{\rm plateau} > \varepsilon_{\rm step}$ | **784/784** if the excluded points are admitted; invariant for any M cut in $[1, 3\times10^4]$; 680/680 under the A-weighted Hα/Hβ substitution |
| Worst case ≈ 38.7% at [0,4] | Holds to 3 s.f. — 0.386903 analytic, 0.386810 direct LSODA, 0.38660 under Hα/Hβ; same grid point under every observable and every ℓ-mixing scale from 0.377× to 2.65× |
| Two-channel mechanism $d\ln R/d\ln x = f_3-f_4$ | Superposition error $3.075\times10^{-14}$ grid-wide, reproduced independently |
| File integrity | All 784 rows reproduced from `L_grid.npy`/`S_grid.npy` at 0.000e+00 relative deviation, by two agents independently |
| **Observable robustness** | **The most robust part of the whole project** — see 6.1 |

### 6.1 C8 can be retired with a number

The Balmer substitution attack **failed to break anything**:

- Hα/Hβ gives **0.386683** vs shell **0.386903** at the same point — **0.06%**
- The 4F dipole-dark lever fails because the ℓ-populations move as a **rigid
  body**: $f(4S) = 0.0608$ vs $f(4F) = 0.0606$
- Photon- vs energy-weighting: **$6.7\times10^{-16}$**
- $n_{\max}$ truncation: **0.04% per top shell**
- Counts move 202 → 200; worst case unchanged; same Te range

**C8 is closed.** The caveat in `divertor_map.txt` should be deleted and replaced
with these numbers.

---

## 7. The corrected claim

> In this optically-thin, atomic-only model (ℓ-resolved to $n=8$, $n$-bundled
> 9–15), a ±5% $T_e$ step leaves the $n{=}3/n{=}4$ ratio at partial equilibrium
> with a stale ground reservoir, differing from the full CR-equilibrium ratio by
> a median 7.0% and up to at least 38.7% at the $T_e = 1$ eV grid edge. It equals
> $|f_3-f_4|\cdot|d\ln n_g/d\ln T_e|\cdot\delta$ to within a factor 1.35. It
> persists for $\tau_{\rm QSS}$ and is therefore not averaged away over a 100 µs
> ELM. **It is not a failure of the QSS closure, which is exact to $10^{-8}$ on
> the plateau — it is the error of assuming ionisation balance, which is what a
> two-parameter $(T_e, n_e)$ CR inversion table actually assumes.**

**Why this is still a thesis.** Every ADAS-style Balmer inversion table is
indexed on $(T_e, n_e)$ and assumes equilibrium ionisation balance. The work
quantifies the error that assumption carries in a transient, derives a criterion
for it from Fujimoto's two-channel decomposition, and shows the result survives
the observable substitution to 0.06%. That is a real, publishable, and narrower
result about line-ratio inversion.

**Structurally almost nothing is wasted.** §`sec:subspace`,
§`sec:algebraic_elim`, the Schur complement, the eigenspectrum, the two-stage
picture — all survive verbatim. What changes: the sentence in §`sec:qss_ratio`
becomes an explicit stated assumption rather than an aside; $\varepsilon$ is
renamed and re-referenced; §`sec:eps_bar` becomes an estimate with measured
deviation; Ch. 5's framing moves from *breakdown* to *the cost of the CRE
assumption*.

---

## 8. Two caveats that cannot be bounded from repo data

Both attach to any $T_e \le 1.5$ eV number, and both belong in §1.5.

**8.1 Optical thickness.** At the worst point [0,4], $n(1s)/n_{\rm ion} = 28.3
\Rightarrow n(1s) \approx 1.5\times10^{15}$ cm⁻³ — **15× denser** than the
thickest case in `summary_trapping.txt`. `assemble_cr_matrix.py` contains no
escape factor. A uniform Ly-series probe moves $\varepsilon$ by **−13% to −77%**.
The trapping document's claim that trapping makes breakdown *worse* is asserted,
not derived, and was run against a matrix whose $L[2P,2P]$ differs from the
current one by 75×.

**8.2 No molecular channels.** `grep -riE "H2|molecul|MAR|dissociat" src/rates/`
returns nothing, in a **96.6%-neutral** gas at 1 eV where molecular-assisted
recombination feeds $n=3$ directly.

---

## 9. Open items

1. **Rename and re-reference $\varepsilon$** throughout. Patch §`sec:qss_ratio`
   first — three sentences, and everything downstream inherits from it.
2. **The pre/post-step $M$ ambiguity.** Both scripts take eigenvalues of the
   *post-step* operator while labelling $T_e$ as *pre-step*. $M$ at "the
   benchmark point" now has three values in three files — 9982 (`CLAUDE.md`,
   from $L[23,5]$), 8243 (`divertor_map.txt`, post +4.81% step → index 24), 4856
   (`plateau_slowmode.txt`, post +0.6 eV step → index 25). Each is individually
   correct; none is distinguished. A reader sees a 21% disagreement with the
   project's own benchmark that is entirely a labelling artifact.
3. **Two scripts quoted in one sentence from different regimes.**
   `divertor_map.txt` lines 12–14 justify the bound using
   `plateau_slowmode`'s $\tau_{\rm fit} = 207\tau_{\rm QSS}$ — measured at a
   **+60% absolute step**, at grid [0,0], $n_e = 10^{12}$, **52× lower in
   density** than the point whose bound it justifies. At that step the
   linearisation is invalid there (error $2.08\times10^{2}$; linearised estimate
   0.00528 vs actual 0.24075, a factor 46). The two scripts are in different
   regimes and their results cannot be quoted in one sentence.
4. **Low-$p$ $r_1$ deficit vs Fujimoto** (factor 8–11) — still open. B1 ruled
   out the excitation data: $\Delta\ell = 1$ agrees at median −5.0% over 120
   comparisons. A different suspect is required.
5. **Title.** The registered title names an approximation this work has now
   shown to be *exact* in the relevant limit. Candidates:
   *"The Cost of Assuming Ionisation Balance: Transient Error in Balmer-Ratio
   Inversion of Divertor Hydrogen Plasmas"*, or *"When Balmer Diagnostics Lag:
   Quantifying Ionisation-Balance Error in Time-Dependent Collisional-Radiative
   Modelling"*. This is a supervisor conversation.

---

## 10. Defense one-liners

- *Does QSS break down?* No — and demonstrating that is part of the result. The
  QSS closure is exact to $10^{-8}$ on the plateau at the worst point on the
  grid. What breaks is the assumption of equilibrium ionisation balance, which
  is what a $(T_e, n_e)$-indexed inversion table silently imposes.
- *Then what did you measure?* The distance between the QSS answer and the
  CR-equilibrium answer, in the Hα/Hβ ratio: median 7.0%, at least 38.7% at
  1 eV, persisting for $\tau_{\rm QSS}$ and therefore not averaged away over an
  ELM.
- *How did you find the misnaming?* By integrating the full system and measuring
  both errors along the same trajectory. The QSS residual came out at
  $10^{-8}$–$10^{-6}$ while the reported error was $10^{-2}$–$10^{-1}$. A
  four-order-of-magnitude gap is not a numerical question.
- *Does $M$ predict the error?* No, and the correlation that appears to say
  otherwise is a temperature proxy: $+0.76$ raw, $+0.33$ controlling for
  $(T_e, n_e)$, $-0.16$ under quadratic control, while a bare $e^{13.6/T_e}$
  with no dynamics in it correlates at $+0.71$.
