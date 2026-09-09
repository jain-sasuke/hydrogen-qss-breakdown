# Chapter 5 — evidence ledger

**24 August 2026.** Everything below is verified against
`L_grid.npy` SHA-256 `2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`
and `S_grid.npy` `7822f536…`, by the script named in each row. Numbers not in
this table are not yet verified and must not be written.

Write from this file. If a number is needed that is not here, verify it first
or leave the sentence out.

---

## 1. The headline, with scopes attached

`verify_eps_gridmap.py`. **Never quote any of these bare.**

| scope | +4.81% step | +20.68% step | note |
|---|---|---|---|
| $T_e\ge2$ eV **and** $n_e\ge10^{14}$ | **10.4%** | **58.5%** | at $[15,5]$ — the **corner of the restriction**, still rising outside it. A lower bound, not a maximum |
| $T_e \ge 2$ eV | 18.1% | 81.1% | at $[15,3]$, the defensible range (Sec 3.5.3) |
| all 400 points | 38.7% | 190.4% | at $[0,4]$ and $[0,5]$ — optically thick corner. Asymptotic only |

**Two further scope traps:**

- The **38.7% maximum sits on the $T_e = 1$ eV grid edge.**
  `verify_plateau_gridmap.py` flags it: *"range may be truncated, not a true
  ridge."* Carry that flag wherever the number appears. The cooling maximum,
  0.2858 at $[1,3]$, is interior.
- The thesis's **"median 7%"** is `0.0741608`, the median over the **338
  `window_ok` heat points**, not over the grid. The all-points median is
  **6.425%**. Name the set, exactly as §3.3.4 does for $\tau_{\rm QSS}$.

**Amplification:** $\varepsilon_{\rm plateau} > \varepsilon_{\rm step}$ at
392/392 and 368/368 — for fixed **fractional** steps of one and four grid
intervals. **Name the convention.** findings_09 W6's counterexample used a
fixed $+0.6$ eV, which at $T_e = 1$ eV is *ten* intervals, not four.

---

## 2. The mechanism — two axes, two different answers

`verify_ridge_mechanism.py` v3, both step sizes.

**Across density: sensitivity.** $\arg\max_j\varepsilon = \arg\max_j|\bar S|$
at **32/32** and **31/31** rows; $\arg\max_j|\Delta\ln u|$ agrees at **0/32**
and **0/31**. The decisive number is not the count:

$$|\bar S| \text{ varies } 7.11\times \text{ across density}; \quad
|\Delta\ln u| \text{ varies } \mathbf{1.01\times}$$

Normalised profiles of $\varepsilon$ and $|\bar S|$ agree to **2.17%** at every
column; $|\Delta\ln u|$ is flat. A factor varying by 1% cannot shape a ridge in
which $\varepsilon$ varies by 7.25×.

**Across temperature: displacement.** On $j=3$, $\varepsilon$ falls 3.72× from
2.02 to 8.29 eV, of which $|\Delta\ln u|$ contributes 2.72× and $|\bar S|$ only
1.29× — **76% displacement**. Same at the four-interval step (72%).

**Finite-step formula.** $\varepsilon = |e^{\bar S\,\Delta\ln u} - 1|$, exact to
five figures. The first-order product gives 59.4% against a true 81.1% at the
four-interval step — a **27% understatement**. Do not use the linearised form.

**Ridge location.** $n_e = 1.93\times10^{13}$ cm⁻³ ($j=3$), identical at both
step amplitudes, 31/31 rows. Resolved to **one grid interval — a factor
2.68** — state that. Below 2 eV the ridge migrates to $j=4$.

---

## 3. What must NOT be claimed

- ~~"the ridge sits at the ionising–recombining crossover, which is
  detachment"~~ — $1.93\times10^{19}$ m⁻³ is **5.2× below** the citable detached
  band (Guillemaut 2011: $10^{20}$–$10^{21}$ m⁻³) and **1.6× below** the
  upstream separatrix band (Stangeby 2022: $3$–$6\times10^{19}$). It is in
  neither.
- ~~"the error peaks at $T_e \approx 2$ eV"~~ — there is **no interior maximum
  in temperature**. $\varepsilon$ falls monotonically 18.07% → 4.55%. The
  apparent peak is the floor of the Sec 3.5.3 restriction.
- ~~"cold, thin plasma along the divertor leg"~~ — no source; the model has no
  geometry.
- ~~"the ridge is where $b_1^{\rm CRE} \approx b_1^{\rm peak}$"~~ — on the
  four-interval step that ratio passes through unity at $T_e = 4.7$ eV where
  $\varepsilon$ is 33% and falling.

Replacement wording: *an interior maximum in density at
$n_e \approx 2\times10^{19}$ m⁻³, temperature-independent over 2–9 eV, with the
error decreasing monotonically in temperature. Resolved to one grid interval.*

---

## 4. The causal chain — first arrow DEMONSTRATED

`verify_plateau_bridge.py` v2, full 43-state integration, `scipy.linalg.expm`.
**Dynamic bridge observed at both representative points.** Scope: two
conditions, not grid-wide — say so.

### 4.1 The persistence result — quote this one

$$\tau_{\rm eff}/\tau_{\rm QSS} = 1.0102 \text{ (benchmark)}, \qquad 1.0656 \text{ (ridge)}$$

Half-decay at $0.7002$ and $0.7386\,\tau_{\rm QSS}$ against $\ln 2 = 0.6931$ for
a pure exponential. **The CRE diagnostic error decays as a single exponential
with time constant $\tau_{\rm QSS}$**, to 1% and 6.6%. In physical units,
$\Delta t_{1/2} = 12.95\ \mu$s and $1.10$ ms.

`thesis_architecture` says the error "persists for $\tau_{\rm QSS}$, longer than
an ELM." That was an order-of-magnitude assertion; it is now a measurement.
Unlike the plateau duration it carries **no threshold dependence** — it is the
same number for any tolerance.

### 4.2 The plateau exists, and is flat

| | benchmark | ridge |
|---|---|---|
| QSS tracking onset | $1.94\,\tau_{\rm relax}$ | $3.72\,\tau_{\rm relax}$ |
| plateau duration | $283\,\tau_{\rm relax}$ | $2890\,\tau_{\rm relax}$ |
| $\varepsilon^{\rm CRE}$ departure from analytic $\varepsilon_{\rm plateau}$ | **3.33%** | **1.28%** |

The flatness is what licenses quoting a single $\varepsilon_{\rm plateau}$
rather than a time-dependent error. Measured, not assumed.

**TRAP: the duration is threshold-set.**
$\text{duration}/\tau_{\rm QSS} = \text{TOL}/(k\,\varepsilon_{\rm plateau})$ —
predicted 0.0334 and 0.0132 against measured 0.0343 and 0.0143, i.e. 3% and 9%.
So 283 and 2890 $\tau_{\rm relax}$ are statements about `TOL_PLATEAU = 2e-3`,
not about the physics. Never quote a duration without naming what sets it.

### 4.3 TRAP: settled values, not entry maxima

The v2 plateau begins at 1.9–3.7 $\tau_{\rm relax}$, so its **maxima are the
tail of the fast transient**, not the plateau state:

| quantity | max in plateau | settled | ratio |
|---|---|---|---|
| state QSS deviation, bench | $5.9\times10^{-4}$ | $1.3\times10^{-6}$ | 471× |
| state QSS deviation, ridge | $1.4\times10^{-4}$ | $8.1\times10^{-8}$ | 1721× |
| $\varepsilon_{\rm track}$, bench | $1.9\times10^{-3}$ | $8.7\times10^{-6}$ | 221× |
| $\varepsilon_{\rm track}$, ridge | $1.9\times10^{-3}$ | $6.9\times10^{-7}$ | 2688× |

**Quote the settled column.** "The excited manifold has reached its
instantaneous QSS state" is a claim about $10^{-6}$, not $10^{-3}$. An external
review quoted the maxima; so did an earlier draft of this file.

Settled scaling: $\varepsilon_{\rm track}\times M = 0.071$ and $0.139$ — the
coefficient moves 2× while $M$ moves 24×. Report it as an observation at two
points, **not** as a law: $\boldsymbol\delta \simeq L_{FF}^{-1}\dot{\mathbf n}_F$
and $\lVert L_{FF}^{-1}\rVert \ne 1/|\lambda_1|$ for a non-normal operator
(§3.4).

### 4.4 The early slow response has a closed form

$$\eta_R \equiv \frac{\ln(R/R^{\rm PE})}{\ln(R^{\rm CRE}/R^{\rm PE})}
\simeq k\,\frac{t}{\tau_{\rm QSS}}, \qquad
k = \frac{1-e^{-|\Delta\ln u|}}{|\Delta\ln u|}$$

Fitted $k = 0.94033$ (bench) and $0.84103$ (ridge). At the ridge the closed form
predicts **0.83786** from $|\Delta\ln u| = 0.3649$ measured independently by
`verify_ridge_mechanism` — **0.38%**, from a fit that never saw that input.

**Verified at the ridge, consistent at the benchmark** — the benchmark $k$ was
only inverted to $|\Delta\ln u| = 0.124$, with no independent measurement. Say
that.

### 4.5 Propagator

`expm` vs `expm_multiply`: $1.13\times10^{-10}$ and $3.69\times10^{-12}$. Two
independent algorithms, no differencing. **This is the only real propagator
evidence.**

Two checks that are NOT evidence, both mine, both withdrawn:
- the **ODE residual** is identically zero by construction ($\mathrm dn/\mathrm dt$
  and $L^+n+S^+$ are the same expression), so a finite difference measures only
  differencing noise. It returned $2.7\times10^{-3}$, implying
  $\varepsilon_{\rm expm}\sim4\times10^{-11}$ — normal, and orders below anything
  reported.
- the **semigroup** check is near-tautological for scipy: `expm` scales and
  squares, so $e^{At/2}$ squared once more traverses the same operations. It
  returned exactly `0.000e+00` at both points for that reason.

### 4.6 Scope, and the chain as it may be stated

Two points establish the **dynamic** arrows only:

$$\text{full transient} \to R^{\rm QSS}(u(t)) \to R^{\rm PE}$$

The density part comes from `verify_ridge_mechanism` over all eight columns.
Combined:

$$\text{full transient} \to R^{\rm PE} \to f_3-f_4 \to \text{density ridge}$$

A grid-wide bridge test would be needed for a grid-wide dynamic claim.

### 4.7 Preregistration record

The v1 criterion — $\varepsilon_{\rm PE} < 2\times10^{-3}$ throughout
$20\tau_{\rm relax}$ to $0.02\,\tau_{\rm QSS}$ — **failed at the ridge**
($2.778\times10^{-3}$, at the window end only). It is retained unchanged and
`WIN_HI` was not shortened. It failed because the ground state had begun its
expected motion while $\varepsilon_{\rm track}$ stayed at $6.8\times10^{-7}$:
the system had not left the QSS manifold, it had moved **along** it. The v2
interval definition is **not** preregistered and must not be described as such.

---

## 5. QSS validity is not diagnostic validity — the sharpest result

| point | $M$ | $\varepsilon_{\rm plateau}$ |
|---|---|---|
| benchmark $[23,5]$ | 8 243 | 6.36% |
| ridge $[15,3]$ | **201 494** | **18.1%** |

**24× better timescale separation, 3× the diagnostic error.** $M$ and
$\varepsilon$ move in opposite directions between the only two points measured
by full integration. $M$ controls the *width of the plateau window*, not the
*magnitude of the error*.

This is the thesis's central distinction, from two numbers in one run, and it
independently reinforces findings_09 W2.

---

## 6. The inversion Jacobian — §3.5.4 promises it, and here it is

`verify_plateau_gridmap.py`, $T_e$ scan at $n_e = 5.18\times10^{13}$:

```
Te   6.251   eps_step 0.000718
Te   6.551   eps_step 0.000379
Te   6.866   eps_step 0.000049   <- minimum
Te   7.197   eps_step 0.000274
Te   7.543   eps_step 0.000589
```

$\varepsilon_{\rm step} = |R^{\rm CRE}_{\rm old}/R^{\rm CRE}_{\rm new} - 1|$, so
a zero means **$R^{\rm CRE}$ is stationary in temperature**:
$\partial\ln R/\partial\ln T_e = 0$ near 6.9 eV.

**The Balmer ratio is locally non-invertible for $T_e$ there**, and §3.5.4's
bound — which constrains only the numerator of

$$\Delta\ln T_e = \frac{\mathrm d\ln R/\mathrm d\ln b_1}
{\mathrm d\ln R/\mathrm d\ln T_e}\,\Delta\ln b_1$$

— says nothing about the inferred temperature there. This discharges the
promise §3.5.4 makes, and it is arguably a stronger result than the ridge.

---

## 7. Inherited from Chapter 3, verified, quote freely

| quantity | value | scope |
|---|---|---|
| $M^{\rm eff} = \tau_{\rm QSS}/\lVert L_{FF}^{-1}\rVert$ | $\ge 77.6$ grid-wide, 8659 at benchmark | **rigorous bound**, not a spectral estimate |
| $\lvert f_3-f_4\rvert$ | $\le\tanh(\lvert\Delta\rvert/4) < 1$ | bounds the **observable**, not the inferred $T_e$ |
| benchmark sensitivity | $0.212$ = 47% of peak | on the rising flank, not the worst case |
| isolation | 3567× at benchmark, **24.3× grid minimum** at $[49,7]$ | quote 24.3, not 24 |
| $[49,7]$ | weakest $\tau_{\rm relax}$, $\tau_{\rm QSS}$, $M$, isolation — all at one point | **excluded from the analysed set**, so the weakest separation on the grid is never tested against a transient. Say this in Ch. 5 |
| $b_1^{\rm CRE}$ | 76.6 at $[0,7]$ cold/dense → $2.683\times10^5$ at $[49,0]$ hot/thin | a large departure coefficient does **not** mean many neutrals |

---

## 8. Cross-validations that have held

- $\varepsilon_{\rm plateau}$ at $[15,3]$: **0.180728** from
  `verify_eps_gridmap`, **0.180728** from `verify_plateau_bridge`, **18.073%**
  from `verify_plateau_gridmap` — three routes.
- Grid maximum: **38.690%** vs **0.386903** at $[0,4]$ — two scripts, 0.0008%.
- Whole $T_e$ column agrees to better than **0.05%** between two scripts.
- $M^+$ at the benchmark: **8242.74** computed from scratch vs **8243** in
  `notation_and_definitions.md`.
- $\tanh(\Delta/4) = 0.451357$ vs numerical $0.451356$.
- Kernel wiring check $1.9\times10^{-9}$ over 256/256 points.

---

## 9. Still open

| item | status |
|---|---|
| **`verify_plateau_bridge` v2** | one run. Intersection-based plateau interval, data-determined $t_*$, $\varepsilon^{\rm CRE}$ half-life as its own output, ODE residual on the propagator |
| propagator verification | both checks return `0.000e+00`; `expm(0)=I` makes the first vacuous. Unverified inside the window |
| Chapter 6 $n(1s)$ | planned $1.5\times10^{15}$ cm⁻³ is low by **13.7×**; measured worst is $2.0\times10^{16}$ at $[0,7]$ |
| `findings_09` W1 | `heat[48,7]` vs measured `[49,7]` — needs the script that built the heat/cool lists |
| D.4 | violated but **inert**: no Chapter 5 script reads the three arrays. Fix for Paper 2 |
| `Balmer_transient_ratio` ×3 | two superseded copies still write live paths. Rename `.py.bak` |
| submission deadline | **still unconfirmed** |
