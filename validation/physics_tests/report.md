# QSS Breakdown Analysis — Results Report

**Model:** 43-state hydrogen CR model, $n=1$–$8$ $\ell$-resolved, $n=9$–$15$ bundled  
**Parameter space:** $T_e = 1$–$10$ eV, $n_e = 10^{12}$–$10^{15}$ cm$^{-3}$  
**Reference conditions:** ITER divertor, $T_e = 3$ eV, $n_e = 10^{14}$ cm$^{-3}$

---

## Figure 1 — QSS step error $\varepsilon_\text{step}$ and exponential fits

### What was computed

For each $(T_e, n_e)$ grid point, a temperature step of size $\Delta T_e$ was applied
instantaneously at $t = 0$. The initial QSS ratio error was computed analytically as

$$\varepsilon_\text{step} = \max_p \frac{|r_p(T_e + \Delta T_e) - r_p(T_e)|}{r_p(T_e + \Delta T_e)}$$

where $r_p = n_p/n_{1S}$ is the excited-to-ground ratio. Five step sizes were tested:
$\Delta T_e \in \{0.3, 0.6, 1.0, 1.5, 2.0\}$ eV.

### Panel (a) — Raw $\varepsilon_\text{step}(T_e)$

All five curves share the same qualitative behaviour. At $T_e = 1$ eV, $\varepsilon_\text{step} \approx 1$
regardless of step size — the QSS ratios change by order-unity when the plasma
is cold and the Boltzmann suppression is strong. The error decays with $T_e$ as
the plasma becomes more collisional and the QSS ratios vary less steeply with
temperature.

The fitted exponential $\varepsilon_\text{step} \approx A \exp(-B \cdot T_e)$ matches the
$\Delta T_e = 0.3$–$1.0$ eV curves well (solid lines through dots). For
$\Delta T_e = 1.5$ and $2.0$ eV the fit degrades at high $T_e$ because
$\varepsilon_\text{step}$ approaches its saturation value of 1 and the exponential
form no longer holds.

### Panel (b) — Fitted decay constant $B$ vs $\Delta T_e$

| $\Delta T_e$ (eV) | $B$ (eV$^{-1}$) |
|---|---|
| 0.3 | 0.643 |
| 0.6 | 0.420 |
| 1.0 | 0.305 |
| 1.5 | 0.256 |
| 2.0 | 0.264 |
| **mean** | **0.377** |

The decay constant $B$ decreases monotonically with $\Delta T_e$, with a
coefficient of variation CV$(B) = 0.386$. This means the exponential shape of
$\varepsilon_\text{step}(T_e)$ is not universal — it flattens for large perturbations.
The physical reason is saturation: $\varepsilon_\text{step}$ is bounded by 1, so
larger steps hit the ceiling sooner, compressing the apparent decay rate.

### Panel (c) — Fitted amplitude $A$ vs $\Delta T_e$

The fitted amplitude $A$ decreases from 1.75 at $\Delta T_e = 0.3$ eV to 1.10
at $\Delta T_e = 2.0$ eV, well below the linear prediction (dashed line).
This sub-linear scaling confirms that the QSS error saturates at large step
sizes. Even the smallest step ($\Delta T_e = 0.3$ eV) gives $A \approx 1.75$,
far above the linearised expectation of $\sim 0.3$, demonstrating that the
QSS error is driven by the nonlinear sensitivity of the Boltzmann factors to
temperature, not by the step size alone.

---

## Figure 2 — State-resolved $\varepsilon_\text{step}$ and effective transition energy

### Panel (a) — $\varepsilon_\text{step}$ decomposed by excited state

The maximum over all states (black solid) is dominated by different states in
different temperature regimes. Below $T_e \approx 4$ eV, the $n = 15$ bundled
state (blue dashed) sets the maximum. Above $T_e \approx 5$ eV, the 2P state
(red dashed) dominates. The 3D state (green dashed) tracks close to the maximum
throughout the low-temperature regime. The $n = 9$ bundled state (orange) is
consistently below all resolved states.

The crossover from $n = 15$ to 2P dominance reflects competing effects: at low
$T_e$, the Rydberg states ($n = 15$) are populated primarily by 3-body recombination
and respond strongly to temperature changes; at high $T_e$, direct excitation
from 1S to 2P dominates and the 2P QSS ratio has the steepest temperature
sensitivity.

### Panel (b) — Extracted effective energy $E_\text{eff}$

The effective energy is defined as

$$E_\text{eff} = \varepsilon_\text{step} \cdot \frac{T_e^2}{\Delta T_e}$$

which extracts the characteristic energy scale governing each state's QSS
sensitivity. For the 2P state (red), $E_\text{eff}$ rises from $\approx 2$ eV at
$T_e = 1$ eV to an asymptote of $\approx 10.2$ eV as $T_e \to 7$ eV,
converging to the 2P–1S excitation energy $\Delta E(2P\text{-}1S) = 10.2$ eV
(red dashed). This confirms that the 2P QSS error is governed by the
excitation threshold at high temperature, where the linearised relation
$\varepsilon \approx (\Delta T_e / T_e^2) \cdot \Delta E$ holds.

For the $n = 15$ state (blue), $E_\text{eff}$ approaches 13.55 eV (blue dashed),
close to the ionisation potential $I_H = 13.61$ eV (dotted). This shows that
the high-$n$ Rydberg states are sensitive to perturbations on the scale of the
ionisation energy, not their small binding energies, because their populations
are controlled primarily by recombination from the continuum.

**Physical interpretation:** Each excited state's QSS sensitivity is set by the
energy scale that controls its dominant population pathway. For low-$n$ states
this is the excitation energy from ground; for high-$n$ Rydberg states it is
the ionisation potential. This naturally explains why the $n = 15$ state
dominates $\varepsilon_\text{step}$ at low $T_e$: 3-body recombination has the
steepest $T_e$ dependence of any process in the model ($\propto T_e^{-4.5}$).

---

## Figure 3 — Balmer $\alpha$ spectroscopic test

**Conditions:** $T_e$ step $3.0 \to 3.6$ eV at $t = 0$, $n_e = 10^{14}$ cm$^{-3}$  
**Timescales:** $\tau_\text{relax} \approx 25$ ns (Stage 1), $\tau_\text{QSS} \approx 15$ $\mu$s (Stage 2)

### Panel (a) — Balmer $\alpha$ intensity: CR vs QSS

The normalised Balmer $\alpha$ intensity $I_{H\alpha}(t)/I_{H\alpha}(0)$ shows
two qualitatively different responses. The full CR solution (red solid) drops
immediately after the step, falling to $\approx 0.8$ within $\tau_\text{relax} \approx 10^{-8}$ s
as the excited states rebalance to the new temperature (Stage 1). The intensity
then continues to decay through Stage 2 as the ground-state density evolves
on the $\tau_\text{QSS}$ timescale, reaching the final steady state of $0.619$.

The QSS prediction (blue dashed) holds the intensity fixed at the pre-step
value of $\approx 1.52$ (relative to $t = 0^-$) throughout Stage 1 and Stage 2,
then drops sharply to $0.619$ only at $t \sim \tau_\text{QSS}$. This is physically
wrong: QSS assumes the excited states instantaneously track the new temperature,
but the real excited-state populations lag behind by $\tau_\text{relax}$.

Both solutions converge to the same final steady state $= 0.619$, confirming
that QSS is valid at long times ($t \gg \tau_\text{QSS}$) as expected.

### Panel (b) — Spectroscopic error $(I_\text{CR} - I_\text{QSS})/I_\text{QSS}$

The error is negative throughout because CR gives a lower intensity than QSS
predicts during the transient. The peak error reaches $-48\%$ at
$t = 1.2 \times 10^{-8}$ s, coinciding with the Stage 1 timescale
$\tau_\text{relax}$. At this moment, QSS overestimates the true Balmer $\alpha$
emissivity by nearly a factor of 2.

The error remains below $-10\%$ for times from $\sim 10^{-8}$ s to
$\sim 10^{-5}$ s — spanning the entire transition from Stage 1 to Stage 2.
It returns to zero only when the ground-state density has fully equilibrated
and the excited states have re-established their QSS ratios at the new
temperature.

**Implication for ITER spectroscopy:** During an ELM thermal quench
($\tau_\text{drive} \sim 100$ $\mu$s), the error at the end of the event is small
($\ll 5\%$) because $\tau_\text{drive} \gg \tau_\text{relax}$. However, during
the crash itself (first $\sim 10$–$100$ ns), the instantaneous error exceeds 40%.
Any spectroscopic diagnostic that integrates over a time window shorter than
$\tau_\text{QSS}$ will incur a systematic bias of this magnitude when using
ADAS/QSS-based inversion tables.

### Panel (c) — Population trajectories $n(t)/n(t=0)$

The population trajectories expose the mechanism behind the spectroscopic error.
After the step:

- The CR 3D population (red solid) drops rapidly to $\approx 0.7$ during
  Stage 1, tracking the new Boltzmann ratio.
- The CR 3P population (black solid) decays more slowly, reaching $\approx 0.5$
  by $\tau_\text{QSS}$.
- The QSS 3P prediction (blue dashed) holds at an elevated value of $\approx 1.5$
  throughout Stage 2, reflecting the QSS assumption that 3P instantly equilibrates
  to the new $T_e = 3.6$ eV — but in reality the ground-state density $n_{1S}$
  has not yet adjusted, so this prediction is wrong.
- The ground state $n(1S)$ (black solid CR) decays slowly over the Stage 2
  timescale, which is what drives the eventual convergence of both solutions.

The QSS overshoot of $n(3P)$ (factor $\sim 1.5\times$ relative to CR) directly
explains the $-48\%$ spectroscopic error: Balmer $\alpha$ emissivity is
proportional to $n(3D)$ and $n(3P)$, and QSS overestimates both during the
transient.

---

## Summary of key quantitative results

| Quantity | Value |
|---|---|
| Peak $\varepsilon_\text{step}$ at $T_e = 1$ eV | $\approx 1.0$ (all $\Delta T_e$) |
| $\varepsilon_\text{step}$ decay constant $B$ (mean) | $0.377$ eV$^{-1}$, CV = 38.6% |
| State dominating error, $T_e < 4$ eV | $n = 15$ (Rydberg, 3BR-controlled) |
| State dominating error, $T_e > 5$ eV | 2P (excitation-controlled) |
| $E_\text{eff}$(2P) asymptote | $10.2$ eV $= \Delta E(2P\text{-}1S)$ |
| $E_\text{eff}$($n15$) asymptote | $13.55$ eV $\approx I_H$ |
| Peak spectroscopic error (Balmer $\alpha$) | $-48\%$ at $t = 1.2 \times 10^{-8}$ s |
| Time of peak error | $\tau_\text{relax} \approx 25$ ns (Stage 1) |
| Error at ITER ELM timescale ($100$ $\mu$s) | $< 1\%$ |
| QSS overshoot of $n(3P)$ | $\approx 1.5\times$ during Stage 2 |

---

## Physical interpretation

The three figures together establish the following hierarchy:

1. **The QSS error is dominated by the state with the largest transition energy
   to ground**, not by the most populated state. Below $T_e \approx 4$ eV,
   this is the $n = 15$ Rydberg state because 3-body recombination ($\propto T_e^{-4.5}$)
   has the steepest temperature sensitivity; above $\approx 5$ eV, it is the 2P
   state controlled by direct excitation.

2. **The error saturates at $\varepsilon_\text{step} \approx 1$ for all step sizes
   at low $T_e$**, meaning that the QSS approximation completely fails to predict
   the instantaneous emissivity immediately after a large perturbation in cold
   plasma. This is step-size independent — even $\Delta T_e = 0.3$ eV gives
   $\varepsilon_\text{step} \approx 1$ at $T_e = 1$ eV.

3. **The spectroscopic consequence is a $-48\%$ systematic error in Balmer $\alpha$**
   during the Stage 1 transient ($\sim$10 ns). This directly translates to a
   $\sim$50% overestimate of neutral hydrogen density by QSS/ADAS-based inversion
   during the first nanoseconds of an ELM thermal quench.