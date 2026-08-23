# Derivation 07 — What the QSS Error Metric Measures, and the Transient Diagnostic Error

**Quantity:** the error measure $\varepsilon$ used throughout Chapter 5 — what it
actually measures, and the time-resolved error in the Balmer $H_\alpha/H_\beta$
ratio after a temperature step.

**Status:** **Run, not yet ✅.** Every number below was produced on 22 Aug 2026
against `L_grid.npy` SHA-256
`2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`
(regenerated 2026-07-21 20:44), on the student's machine. Convergence-checked.
**Not yet skeptic-passed in its final form; grid map not done; one index
definition unverified — see §8.**

**Feeds:** Ch. 3 §3.5 (definition of the error metric), Ch. 5 (central result),
Ch. 4 (corrections). Answers pending item 1 of `derivation_04c` §7.

---

## 1. Core results

**(a) The published error metric is ~94% a measurement of ground-state lag.**

$$\varepsilon = \max_p \frac{|r_p^{\rm act} - r_p^{\rm QSS}|}{r_p^{\rm QSS}}, \qquad r_p = n_p/n_{1S}$$

The denominator $n_{1S}$ lags the step. Every excited state inherits the same
error. At the benchmark point the metric reads 0.521; the ratio $n_3/n_4$, which
cancels $n_{1S}$, reads **0.032**.

**(b) Timescale separation bounds neither the transient nor the plateau.**

At the benchmark point ($M = 9982$), after a $+0.6$ eV step the Balmer ratio error
rises to $10.2\times$ its step value and **stays there for three decades in
time**:

$$\boxed{\;\varepsilon^{\rm ratio}_{\rm peak} / \varepsilon^{\rm ratio}_{\rm step} = 10.23, \qquad \varepsilon^{\rm ratio}_{\rm plateau} / \varepsilon^{\rm ratio}_{\rm step} = 10.10\;}$$

**(c) A filter bug corrupted 19 of 400 grid points.** Found, fixed, bounded.

---

## 2. The metric problem (physics first)

### 2.1 What QSS actually claims

QSS says the *excited manifold* equilibrates instantly while the ground state and
$n_e$ evolve freely. It makes no claim about $n_{1S}$.

### 2.2 Why $r_p = n_p/n_{1S}$ measures the wrong thing

After a step, the excited states re-equilibrate within $\tau_{\rm relax}$ to the
pattern slaved to the **instantaneous** ground state. The ground state itself
moves only on $\tau_{\rm QSS}$, four orders of magnitude later. So for a time
window spanning those four decades:

- numerator $n_p$ — has moved
- denominator $n_{1S}$ — has not

and every $r_p$ carries the **same** common ground-state factor.

### 2.3 The signature: uniformity across the manifold

A truncation artifact would be localised at high $n$. A state-specific physical
effect would vary with $n$. A common denominator error gives the *same* value
everywhere. Measured, benchmark point:

| $\varepsilon$ over | value |
|---|---|
| all 43 states | 0.521 |
| resolved excited only ($n \le 8$) | 0.517 |
| $n=3$ shell | 0.484 |
| $n=4$ shell | 0.500 |
| **$n_3/n_4$ ratio** | **0.032** |

Uniform from $n=2$ to $n=15$, then collapsing by 16× the moment the denominator
is cancelled.

### 2.4 The arithmetic closes

If a common factor $f$ multiplies both shells, then
$\varepsilon_{n_3} = 1 - f\,a$ and $\varepsilon_{n_4} = 1 - f\,b$, and the ratio
error retains only $a/b$:

$$\frac{1-\varepsilon_{n_3}}{1-\varepsilon_{n_4}} = \frac{0.516}{0.500} = 1.032$$

Measured $\varepsilon^{\rm ratio}_{\rm step} = 0.032$. **Agreement to two
figures.** 94% of the published error is the common ground-state factor.

### 2.5 This is the same disease as the $L^2$ norm

`SETUP.md` §8.1 found the 43-state $L^2$ norm to be ground-dominated (ground
population $73\times$ the entire excited manifold) and concluded it did not
measure QSS. The `max_p` ratio metric has the same defect through a different
route. It also explains why the two disagreed so violently — $L^2$ *averages*
the common factor away among 42 states and reports 0.7%; `max_p` *selects* it
and reports 52%. Neither was measuring QSS.

It also retro-explains Q5: $\varepsilon^{\rm ratio}$ was $31\times$ smaller than
$\varepsilon^{L^2}$ not mainly through common-mode cancellation between adjacent
shells, but because the ratio removes the contaminating denominator.

---

## 3. A hypothesis that was tested and refuted

**Hypothesis (Claude's, 22 Aug):** the metric is a truncation artifact, because
$\max_p$ is attained at $n=15$ (index 42) at 346 of 400 grid points (88.3%), and
`validation/physics_tests/report.md` independently reports the same argmax below
$T_e \approx 4$ eV.

**Test:** recompute $\varepsilon$ over resolved states only ($n \le 8$).

**Result: REFUTED.** Grid-mean $\varepsilon$ moves 0.517 → 0.515. At the ITER
reference, 0.521 → 0.517. The fraction of grid points exceeding the 10% threshold
**rises** from 88.3% to 91.8%.

**Why the refutation matters:** it rules out truncation and leaves the
ground-state explanation standing. The argmax sits at $n=15$ because that state
is furthest from the ground state in excitation energy, so the common factor bites
hardest there — not because the bundling is wrong.

Recorded per method 7. The hypothesis was reasonable, the test was severe, the
answer was no.

---

## 4. The time-resolved result

### 4.1 Setup

At $t=0$, $T_e \to T_e + 0.6$ eV at fixed $n_e$. Integrate the full 43-state
system $\dot{\mathbf n} = L_{\rm new}\mathbf n + \mathbf b_{\rm new}$ from
$\mathbf n(0) = \mathbf n^{ss}_{\rm old}$, LSODA. Report

$$\varepsilon^{\rm ratio}(t) = \left|\frac{n_3(t)/n_4(t)}{n_3^{ss,\rm new}/n_4^{ss,\rm new}} - 1\right|$$

**Note on which operator:** $\tau_{\rm relax}$ and $\tau_{\rm QSS}$ below are
eigenvalues of $L_{\rm new}$ (the post-step operator), not of the grid-point
operator. They therefore differ from the grid tables and any thesis sentence must
say which operator it used.

### 4.2 Measured

| | ITER ref [23,5] | cold corner [0,0] |
|---|---|---|
| $\tau_{\rm relax}$ | $2.157\times10^{-9}$ s | $3.472\times10^{-8}$ s |
| $\tau_{\rm QSS}$ | $1.048\times10^{-5}$ s | $3.946\times10^{-1}$ s |
| $\varepsilon_{\rm step}$ (steady states) | 0.032104 | 0.243940 |
| $\varepsilon(t=0)$ (integrator) | 0.032104 | 0.243940 |
| **peak** | **0.328485** at $7.8\,\tau_{\rm relax}$ | **1.862135** at $0.074\,\tau_{\rm relax}$ |
| peak / step | **10.232** | **7.634** |
| plateau | 0.324170 | 0.240747 |
| **plateau / step** | **10.098** | **0.987** |

$\varepsilon(t=0)$ from the integrator matches the independent steady-state solve
to six digits at both points — the initial condition is right.

### 4.3 Three stages

1. **$\tau_{\rm relax}$ (ns).** Excited manifold rearranges. The ratio error
   *grows* — by $10\times$ at the reference, $7.6\times$ in the cold corner.
2. **Plateau.** ITER holds ~0.32 for three decades; the cold corner holds 0.2407
   through $10^4\,\tau_{\rm relax}$ ($= 9\times10^{-4}\,\tau_{\rm QSS}$ — it has
   not begun to decay).
3. **$\tau_{\rm QSS}$.** Ground state catches up, error collapses. Visible at
   ITER (0.046 by $2\,\tau_{\rm QSS}$); off the window in the cold corner.

### 4.4 The physics of the plateau — and why the two points differ

After Stage 1 the excited manifold is in equilibrium with a **stale reservoir**:
it has relaxed to the pattern slaved to the *old, unmoved* ground state. QSS
assumes the whole system has equilibrated. The plateau is the gap between those
two statements.

**Cold corner: plateau/step = 0.987.** The stale-reservoir state and the new QSS
target nearly coincide. The transient is a genuine excursion that heals.

**benchmark point: plateau/step = 10.1.** They do not coincide, and the error sits
$10\times$ above the step value until $\tau_{\rm QSS}$.

**Why the reference is the worse case — counterintuitive, and worth stating.**
At the reference, common-mode cancellation is excellent at $t=0$ (94% cancelled:
0.48 → 0.032). But cancellation is exact only while $n_3$ and $n_4$ move
*together*. They are fed differently — $n_3$ more by cascade and excitation,
$n_4$ more by recombination from the ion reservoir — so during Stage 1 they
redistribute at different rates. The very cancellation that makes the diagnostic
look robust at steady state leaves the transient residual unprotected.

### 4.5 This is non-normality, in the observable

`derivation_04c` established $\mu(L) = +1.28\times10^{11}$ s⁻¹ $> 0$, which
*guarantees* transient growth for some perturbation, and flagged that the $L^2$
result does not transfer to the Balmer observable (§7 pending item 1).

**It transfers.** Mechanically this is the un-cancellation of §4.1: opposing modal
amplitudes on non-orthogonal eigenvectors cancel at $t=0$ and stop cancelling as
the modes decay at different rates. Q4c and Q5 join into one result.

### 4.6 Convergence — physics, not numerics

Identical to five digits across `rtol` $10^{-10}$, $10^{-12}$, and a capped
`max_step` ($\tau_{\rm QSS}/50$). Method 9 satisfied.

---

## 5. Contribution to Q8 (which $\tau$ enters $De$)

Measured, not argued. 1/e decay of $\varepsilon_{\max}$:

| | vs $\tau_{\rm relax}$ | vs $\tau_{\rm QSS}$ |
|---|---|---|
| ITER ref | $9.8\times10^{3}$ | **2.0** |
| cold corner | $7.3\times10^{7}$ | **6.4** |

**The ground-dominated metric relaxes on $\tau_{\rm QSS}$.** That settles which
$\tau$ belongs in $De$ *for that metric* and reconciles the three-way documentary
disagreement: `chapter4.tex` line 953 and `qss_analysis.py` use $\tau_{\rm QSS}$
and are correct **for the metric they use**; `SETUP.md` §8.2's $\tau_{\rm relax}$
was for the excited-only measure.

**Refuted along the way (Claude's hypothesis):** that
$\varepsilon^{\rm ratio}$ relaxes on $\tau_{\rm relax}$ in one regime and
$\tau_{\rm QSS}$ in the other. A 1/e time is meaningless for a signal that grows
first; both points overshoot. **A decay time may not be quoted for
$\varepsilon^{\rm ratio}$ at all** — the correct descriptors are peak,
peak time, plateau, and plateau duration.

---

## 6. The filter correction (Ch. 4 corrections chapter)

`eigs[eigs < -1.0]` in **both** `src/validation/qss_analysis.py` (~137) and
`src/validation/validate_gates.py` (~402) discarded every eigenvalue with
$|\lambda| < 1$ s⁻¹ and silently promoted the eigenvalue ladder by one step, so
$\tau_{\rm QSS}$, $\tau_{\rm relax}$ **and** $M$ were all wrong at the affected
points. Replaced with `eigs < 0.0` after confirming
$\max \operatorname{Re}\lambda = -1.487\times10^{-2} < 0$ over all 400 points.

| | filtered | corrected |
|---|---|---|
| $M$ range | 1.341 – $1.01197\times10^{8}$ | **86.7678 – $1.72928\times10^{9}$** |
| $\tau_{\rm QSS}$ max | 0.9251 s | **67.2333 s** |
| $\tau_{\rm relax}$ max | $3.536\times10^{-8}$ s | **$3.888\times10^{-8}$ s** |

- **19 of 400 points affected**, $T_e \le 1.389$ eV, $n_e \in [10^{12},
  1.93\times10^{13}]$. Set-identical to prediction; zero points outside it moved.
- At every one of the 19, $\tau_{\rm relax}^{\rm new} = \tau_{\rm QSS}^{\rm old}$
  **exactly** — the ladder promotion, confirmed.
- $\varepsilon_{\rm step}$ **bit-identical** at all 400 points. The S-criterion
  and step-error maps were never affected.
- $\varepsilon_{\rm res}$ moves by $\le 5\times10^{-7}$ relative, only at the 19,
  and only because it is sampled at $t = 100\,\tau_{\rm relax}$ and
  $\tau_{\rm relax}$ moved. The steady states did not change.
- Breakdown counts $+19$ at every drive: ELM crash 243 → 262 (60.75% → 65.50%).
- **benchmark point bit-identical** on every quantity.
- Regenerated grids **bit-for-bit identical** to an independent unconditional
  recomputation (`timescales_unfiltered_CHECK.npz`) at all 400 points.

Filtered outputs preserved as `*_FILTERED_20260721` — evidence for Ch. 4.

---

## 7. Checks passed

1. $\varepsilon(t=0)$ from LSODA matches the independent linear solve to 6 digits,
   both points. ✓
2. Convergence across three tolerance settings, identical to 5 digits. ✓
3. Truncation hypothesis tested severely and refuted (resolved-only $\varepsilon$
   moves $<0.5\%$). ✓
4. Cancellation arithmetic $(1-\varepsilon_{n_3})/(1-\varepsilon_{n_4}) = 1.032$
   vs measured 0.032. ✓
5. Filter fix: two independent code paths agree bit-for-bit. ✓
6. Filter fix leaves the benchmark point bit-identical — the correction is local. ✓
7. Dimensions: all $\varepsilon$ dimensionless. ✓
8. $n_{\rm ion}$ invariance: $\mathbf b \propto n_{\rm ion}$ exactly and every
   $\varepsilon$ is built from ratios, so all results are invariant under it. ✓

---

## 8. Honest limitations — none of this is ✅ yet

1. **The step is not a controlled variable.** $+0.6$ eV is a 60% step at
   $T_e = 1$ eV and 6% at 10 eV. Under a fixed 5% fractional step the cold-corner
   $\varepsilon_{\rm step}$ is 0.471, not 0.993. **Every contour in the Ch. 5 map
   is confounded with this.** The controlled version must be recomputed.
2. **Two grid points, one step direction.** Heating only. Cooling untested. Grid
   map of peak/step and plateau/step not done.
3. **$N_3 = [3,4,5]$, $N_4 = [6,7,8,9]$ were inferred, not read from
   `cr_context.py`.** Must be verified before any number here is quoted. This
   violates the no-hardcoding rule and is the single most likely source of a
   silent error in §4.
4. **Open-system boundary.** The ion is a fixed reservoir in $\mathbf b$; there is
   no 44th state. Closing the manifold changes $\tau_{\rm QSS}$ at [0,0] from
   67.2 s to 1.6226 s — a factor of **41**. $\tau_{\rm relax}$ is unchanged. Any
   $\tau_{\rm QSS}$ statement is boundary-dependent and must say so.
   $\lambda_0 = 2.286 \times K_{\rm ion}(1S)\,n_e$ — it is the CR ionisation time
   of ground-state hydrogen, **not an equilibration time**.
5. **$T_e$-fragility.** $d\ln\tau_{\rm QSS}/d\ln T_e = -13.3$. "67.2 s" to three
   figures is not defensible; "tens of seconds" is.
6. **Optically thin only.** Under $\Lambda = 10^{-3}$, $\tau_{\rm relax} \to 2.05$
   µs and $M$ falls by $158\times$.
7. **No $\ell$-mixing in the bundled block.** `K_lmix.npy` has non-zero entries
   only for $n = 2$–8; `verify_bundling_psm20.py` has never been run.
8. **The $\bar\varepsilon$ decay law in `qss_analysis.py` is wrong in shape.** The
   docstring asserts $\varepsilon_{\rm res}e^{-t/\tau_{\rm QSS}}$; direct
   integration at [0,0] gives a 1/e time of 2.58 s, not 67.2 s. Reported, not
   touched.
9. **Skeptic returned MODIFY** on the cold-corner claim. Its verdict: do not write
   $\varepsilon_{\rm res} = 0.9886$, "fails completely", and
   $\tau_{\rm QSS} = 67.2$ s in the same sentence as $M$.

---

## 9. What must happen before ✅

1. Verify $N_3$/$N_4$ against `cr_context.py`. **Blocking.**
2. Grid map: peak/step and plateau/step over all 400 points, heating and cooling.
3. Recompute under a **fixed fractional step** — that is the controlled experiment.
4. Test the open question: is plateau/step $\approx 10$ a property of the
   reference point, or does it appear wherever $t=0$ cancellation is strong?
   These predict opposite things about where diagnostics are safe.
5. Skeptic pass on the §1(b) claim in final form.
6. Decide D.4: `validate_gates.py` is fixed but not re-run, and still writes the
   same three `.npy` paths as `qss_analysis.py`.

---

## 10. Defense one-liners

- *What does your error metric measure?* As originally defined, mostly
  ground-state lag — the ratio $n_p/n_{1S}$ has a denominator that moves on
  $\tau_{\rm QSS}$ while the numerator moves on $\tau_{\rm relax}$. We measured
  the contamination at 94% at the benchmark point and report the Balmer ratio
  instead, which cancels it.
- *How can $M = 9982$ and the diagnostic still be wrong by 32%?* Because $M$ is a
  ratio of rates and the error is a distance that *grows*. After the step the
  excited manifold equilibrates — correctly, on $\tau_{\rm relax}$ — to a ground
  state that has not yet moved. It sits there, $10\times$ its step error, for
  three decades in time.
- *Is the transient growth physical or a norm artifact?* We report it in the
  Balmer $H_\alpha/H_\beta$ ratio, the quantity an experimentalist measures, with
  a stated step size and direction. $\mu(L) > 0$ guarantees growth for some
  perturbation; this is the perturbation.
- *You found a bug in your own analysis?* Two. An $\ell$-mixing coefficient error,
  quantified and shown to move $\tau_{\rm relax}$ by $<0.85\%$ anywhere; and an
  eigenvalue filter that corrupted 19 of 400 grid points, found by recomputing the
  spectrum unconditionally and comparing bit-for-bit.
