# Derivation 05 — QSS Breakdown: Why $M \gg 1$ Is Not Enough

**Quantity:** the step error $\delta\mathbf n(0^+)$ and its two scalar measures,
$\varepsilon^{L^2}_{\rm step}$ (mathematical) and $\varepsilon^{\rm ratio}_{\rm step}$
(diagnostic, Balmer $H_\alpha/H_\beta$).

**Status:** **Session B complete.** The vector result and both scalarisations are
derived; the $A$-coefficient cancellation is established; a prediction was made
and confirmed numerically. **Sessions C–F pending** (brutal physics test,
$\tau_{\rm QSS}$ reconciliation, non-normality applied, code review).

**Grounded in:** Q3 (the moving QSS target), Q4 (the two timescales), Q4c
(non-normality). Not grounded in any thesis text.

**Feeds:** §3.3, §3.5 (metrics) — **the paper's central argument.**

---

## 1. The question

$M = \tau_{\rm QSS}/\tau_{\rm relax} = 9982$ at the ITER reference. The excited
manifold relaxes ten thousand times faster than the ionisation balance moves. By
the textbook argument — the same Bodenstein reasoning used in reaction kinetics
— QSS should be flawless.

**Yet QSS still breaks down.** How?

---

## 2. Core result — the step error is a distance, not a rate

At $t=0$ the plasma conditions step. Because rate coefficients depend on $T_e$
and $n_e$, **both** operators change:

$$L_{\rm old} \to L_{\rm new}, \qquad \mathbf b_{\rm old} \to \mathbf b_{\rm new}$$

The QSS target follows *instantly*, because it is algebra, not evolution:

$$\mathbf n^{ss}_{\rm old} = -L_{\rm old}^{-1}\mathbf b_{\rm old} \qquad\longrightarrow\qquad \mathbf n^{ss}_{\rm new} = -L_{\rm new}^{-1}\mathbf b_{\rm new}$$

But the real populations **cannot jump**. All rates in $L_{\rm new}$ are finite,
so $\dot{\mathbf n}$ is finite and $\mathbf n(t)$ is continuous. Immediately
after the step the system is still where it was:

$$\mathbf n(0^+) = \mathbf n^{ss}_{\rm old}$$

QSS *claims* the populations are at $\mathbf n^{ss}_{\rm new}$. Reality has them
at $\mathbf n^{ss}_{\rm old}$. Hence:

$$\boxed{\;\delta\mathbf n(0^+) \;=\; \mathbf n^{ss}_{\rm old} - \mathbf n^{ss}_{\rm new} \;=\; -L_{\rm old}^{-1}\mathbf b_{\rm old} + L_{\rm new}^{-1}\mathbf b_{\rm new}\;}$$

### Why this answers the paradox

The right-hand side contains **only** $L_{\rm old}, \mathbf b_{\rm old},
L_{\rm new}, \mathbf b_{\rm new}$. There is no $\tau_{\rm relax}$, no
$\tau_{\rm QSS}$, no $\lambda$, no $M$.

**The initial QSS error is a geometric distance between two steady states.
$M$ is a ratio of rates. These are independent quantities — knowing one tells
you nothing about the other.**

$M=9982$ means the system *recovers* from the error in ~2.28 ns. It does **not**
mean the error is small. That is the whole argument.

### The physical picture

A sprinter chasing a bus. The bus moves at 30 km/h, the sprinter runs at
30,000 km/h — $M = 1000$, excellent separation. Then the bus is teleported 10 km
ahead. At $t=0^+$ the sprinter is 10 km behind, and the ratio
(sprinter speed)/(bus speed) says nothing whatever about that 10 km. It only says
how fast the gap will close.

**QSS is the assumption that the sprinter is always beside the bus.** After a
teleport, it is wrong by 10 km — however fast the sprinter runs.

---

## 3. Scalarisation I — the norm

$$\varepsilon^{L^2}_{\rm step} = \frac{\|\mathbf n^{ss}_{\rm old} - \mathbf n^{ss}_{\rm new}\|}{\|\mathbf n^{ss}_{\rm new}\|}$$

**Why normalise at all.** Populations at $n_e=10^{15}$ are ~1000× those at
$10^{12}$, so a raw $\|\delta\mathbf n\|$ would be dominated by density and would
be plotting scale rather than error. Normalisation makes it a *fractional* error,
comparable across the grid.

**Why by the new target.** QSS *claims* the system is at
$\mathbf n^{ss}_{\rm new}$, so the natural question is "by what fraction is that
claim wrong?"

**The objection, which must be stated in the thesis.** This makes $\varepsilon$
**asymmetric and direction-dependent**: a step A→B gives a different value from
B→A, because the denominators differ. For a $T_e$ increase,
$\mathbf n^{ss}_{\rm new}$ has larger excited populations, so the same absolute
mismatch reads as a *smaller* fractional error. **Heating and cooling steps are
therefore not directly comparable**, and any map must state the step direction.

**Units.** Numerator and denominator are both cm⁻³ — $\varepsilon^{L^2}_{\rm step}$
is **dimensionless**. (A units check worth running every time: if $\varepsilon$
carried units, the formula would be wrong.)

---

## 4. Scalarisation II — the Balmer ratio

### Why a ratio, not a single line

An experimentalist measures line intensities and inverts them for $T_e$ and
$n_e$. Absolute intensity depends on total neutral density and viewing path
length, neither known well. **In a ratio those cancel.** The workhorse is

$$R = \frac{I_{H\alpha}}{I_{H\beta}}, \qquad I_{H\alpha}\propto A_{3\to2}\,n_3, \qquad I_{H\beta}\propto A_{4\to2}\,n_4$$

Because the model is $\ell$-resolved, the shell populations are sums:

$$n_3 = n_{3s}+n_{3p}+n_{3d}, \qquad n_4 = n_{4s}+n_{4p}+n_{4d}+n_{4f}$$

### The error measure, and the cancellation

$$\varepsilon^{\rm ratio}_{\rm step} = \frac{R(\mathbf n^{ss}_{\rm old}) - R(\mathbf n^{ss}_{\rm new})}{R(\mathbf n^{ss}_{\rm new})} = \frac{R(\mathbf n^{ss}_{\rm old})}{R(\mathbf n^{ss}_{\rm new})} - 1$$

Writing $R$ out, the $A$-coefficients appear identically in old and new states
and **cancel completely**:

$$\boxed{\;\varepsilon^{\rm ratio}_{\rm step} = \frac{n^{ss}_{3,\rm old}\,/\,n^{ss}_{4,\rm old}}{n^{ss}_{3,\rm new}\,/\,n^{ss}_{4,\rm new}} - 1\;}$$

**This is a genuinely valuable robustness property and belongs in the thesis:
the diagnostic error depends only on the shell population ratios, not on the
atomic transition rates.** The $A$-coefficients could be wrong by 20% and this
number would not move.

### The caveat on that cancellation (backlog C8)

$H_\alpha$ is $3\to2$ with three dipole-allowed channels ($3s\to2p$, $3p\to2s$,
$3d\to2p$), each carrying its own $A$. So "$A_{3\to2}$" is an $\ell$-weighted
average. **If the $\ell$-distribution within $n=3$ differs between the endpoints,
the effective $A_{3\to2}$ is not identical old-to-new and the cancellation is
only approximate.**

Since ℓ-mixing runs at $10^{11}$–$10^{12}$ s⁻¹ (Q4b) — ~1000× faster than
anything else — both endpoints *should* be statistically populated and the
cancellation should be excellent. **But this is to be checked, not assumed.**

---

## 5. Prediction and confirmation — common-mode cancellation

**Prediction, made before computing:** $\varepsilon^{L^2}_{\rm step} >
|\varepsilon^{\rm ratio}_{\rm step}|$.

*Reasoning.* The $L^2$ norm registers the whole 43-dimensional displacement. The
Balmer ratio compares two **adjacent** excited shells that respond to a $T_e$ or
$n_e$ change in broadly similar ways, so much of their absolute change is
**common mode and cancels in the ratio**. The ratio should therefore be far less
sensitive, unless a step specifically alters the relative $n{=}3$/$n{=}4$
excitation balance more than it alters the overall population vector.

**Confirmed numerically** (ITER reference, $T_e$ step $2.947\to3.556$ eV;
preliminary run with a stand-in source vector):

| Quantity | Value |
|---|---|
| $\varepsilon^{L^2}_{\rm step}$ | 1.17 |
| $\lvert\varepsilon^{\rm ratio}_{\rm step}\rvert$ | 0.038 |
| **ratio** | **31×** |

And the mechanism is visible directly in the numbers:

| | fractional change |
|---|---|
| $n_3$ | $+18.7\%$ |
| $n_4$ | $+14.4\%$ |
| **difference — all the ratio sees** | $+4.3\%$ |

Both shells move together; the Balmer ratio registers only the small
differential.

**Caveat.** The magnitudes used a stand-in $\mathbf b$ (uniform excited feed,
ground-dominated) and must be recomputed with the real `S_grid.npy`. The
*structural* result is robust to that choice, since it depends on how $n_3$ and
$n_4$ respond similarly to a $T_e$ step, not on how they are fed.

---

## 6. What this does to the thesis argument — read honestly

**Against the simple diagnostic-bias story.** A 117% error in the population
vector produces a 3.8% error in the measured ratio. **"QSS-based Balmer
diagnostics are badly biased" is not supported at this step size, and must not
be claimed.**

**For a better and more defensible result.** We now have a quantitative account
of *why* the standard diagnostic is robust, plus a framework for locating where
it is not. The ratio fails when $n_3$ and $n_4$ stop moving in common mode —
presumably for steps large or fast enough that the shells decouple.

**The sharper question, and the better paper:** *where on the $(T_e,n_e)$ grid,
and for what step sizes, does common-mode cancellation break down?* Finding a
regime boundary is stronger than asserting a bias, and much harder for a referee
to attack. (Backlog C9, now the highest-priority open idea.)

**The central argument is unaffected.** $M\gg1$ yet $\varepsilon^{L^2}=1.17$
stands on its own — timescale separation does not bound the error. What changes
is only which *observable* carries the consequence.

---

## 7. Checks passed

1. **No timescale in $\delta\mathbf n(0^+)$** — verified by inspection; this is
   the algebraic content of the central claim. ✓
2. **Dimensions** — both $\varepsilon$ measures dimensionless. ✓
3. **$A$-coefficient cancellation** — derived; makes the diagnostic measure
   independent of atomic transition rates. ✓
4. **Prediction before computation** — common-mode reasoning stated first, then
   confirmed at 31×, with $n_3$/$n_4$ fractional changes exposing the mechanism. ✓
5. **Asymmetry of the norm measure** — identified and flagged rather than
   discovered later. ✓

---

## 8. Pending (Sessions C–F)

| Session | Content |
|---|---|
| **C** | Brutal physics test: small-step limit, large-step limit, $M\to\infty$, signs, dimensions |
| **D** | Reconcile the two $\tau_{\rm QSS}$ definitions (eigenvalue vs target-motion); introduce the drive timescale and $De=\tau_{\rm relax}/\tau_{\rm drive}$ |
| **E** | Non-normality applied: $P_{\rm slow}$ via **left** eigenvectors (Q4c); explain $\varepsilon_{\rm res}>\varepsilon_{\rm step}$; **test transient growth in the Balmer observable, not $L^2$** |
| **F** | Code review of `qss_analysis.py` and `Balmer_transient_ratio.py`; finalise this note |

Also outstanding: recompute §5 with the real source vector; C8 (ℓ-distribution
check); C9 (the cancellation-failure map).

---

## 9. Defense one-liners

- *How can $M=9982$ and QSS still fail?* Because $M$ is a ratio of **rates** and
  the step error is a **distance**. $\delta\mathbf n(0^+) = \mathbf n^{ss}_{\rm old}
  - \mathbf n^{ss}_{\rm new}$ contains no timescale at all. $M$ tells you how fast
  the system recovers, not how far it was displaced.
- *Why can't the populations just follow the target?* Because populations are
  continuous — every rate in $L$ is finite. The plasma conditions can step
  discontinuously; the atoms cannot.
- *Why normalise by the new target?* QSS claims the system is at
  $\mathbf n^{ss}_{\rm new}$, so we measure how wrong that claim is. It does make
  the measure asymmetric between heating and cooling steps, which we state
  explicitly.
- *Doesn't the Balmer ratio show a large bias?* No — and that is itself a result.
  Common-mode cancellation between $n{=}3$ and $n{=}4$ suppresses the ratio error
  by roughly 30× relative to the population-vector error. We report where that
  cancellation holds and where it fails, rather than claiming a bias that the
  data does not support.
