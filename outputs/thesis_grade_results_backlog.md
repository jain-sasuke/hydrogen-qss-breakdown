# Thesis-Grade Results — Backlog and Verification Register

**Purpose.** A running register of results that are worth putting in the thesis.
Ideas get added here the moment they surface. Nothing is executed from this list
until the underlying physics is understood — the order is always
**physics → derivation → code we understand line by line → run on our own
system → verify → document.**

**How an entry graduates.** Each result moves through five states:

| State | Meaning |
|---|---|
| 💡 **Idea** | Worth doing; physics not yet derived |
| 📐 **Derived** | Physics and mathematics understood; ready for code |
| 💻 **Coded** | Script written and understood line by line (input, computation, output) |
| ▶️ **Run** | Executed on Nikhil's own system, output in hand |
| ✅ **Verified** | Cross-checked, sensitivity-tested, caveats known, ready for the thesis |

**Rule.** No entry is written into the thesis before reaching ✅. A result that
runs but is not understood is a black box, and black boxes are what produced the
−46% Hα artifact.

---

## A. RESULTS ALREADY OBTAINED (need write-up, not new work)

### A1. Boundary-level descent of the relaxation eigenmode ▶️ Run

**Claim.** The relaxation eigenmode $\lambda_1$ migrates to progressively lower
principal quantum number as density rises — a direct observation of the
descending Griem/Fujimoto radiative–collisional boundary.

**Evidence obtained** (`verify_boundary_descent.py`, run 14 Jul):

| $T_e$ | $10^{12}$ | $3\times10^{12}$ | $7\times10^{12}$ | $2\times10^{13}$ | $5\times10^{13}$ | $10^{14}$ | $4\times10^{14}$ | $10^{15}$ |
|---|---|---|---|---|---|---|---|---|
| 1.0 eV | 5G | 4F | 4F | 3D | 3D | 3D | 2P | 2P |
| 10.0 eV | n15 | 4F | 3D | 3D | 3D | 2P | 2P | 2P |

$\langle n\rangle$ falls from $\sim$6.7–11.2 at $n_e=10^{12}$ to $2.0$ at $10^{15}$.
Negative slope under **all three** weightings tested ($v^2$/ground-excluded,
$|v|$/ground-excluded, $v^2$/ground-included), so the descent is not an artifact
of the definition.

**What must be stated carefully:**
- The migration is a **discrete staircase** (5G/n9 → 4F → 3D → 2P), not a
  continuous power law. $R^2$ of the log-log fit is only 0.91–0.94, and dropping
  the two saturated points shifts the slope by 19.5%. **Do not quote a fitted
  exponent.**
- Name the quantity **"eigenmode-weighted mean principal quantum number"**, not
  "the boundary level" — it is a dynamical proxy for the Griem $n_{cr}$, not the
  same object.
- The $\langle n\rangle = 2$ floor is imposed by excluding the ground state.
- Griem's own caution applies: the analytic $n_e^{-2/17}$ estimate is crude for
  hydrogen by his own admission.

**Remaining to reach ✅:**
- Fix the figure: consistent $T_e$ between panels (currently 3.24 eV in panel (a)
  vs the 2.947 eV reference); plot shell weights $W_n=\sum_\ell |v_{n\ell}|^2$
  vs $n$ instead of raw components vs state index, to remove the
  resolved/bundled visual bias ($n=8$ spreads over 8 points, $n=9$ over 1);
  relabel the dashed line "slope guide $\propto n_e^{-2/17}$" (it is anchored
  arbitrarily at 9.17, not at the analytic 3.84 or 5.46).
- Add a sign panel or note — $|v|$ hides that this is an *exchange* mode.

**Thesis home:** Ch. 4 Gate E (timescale hierarchy validation).

---

### A2. Two-timescale structure: one gap, then a continuum ▶️ Run

**Claim.** The CR spectrum has exactly **one** large spectral gap — between the
ionisation mode and everything else — followed by a quasi-continuum, not three
separated groups.

**Evidence** (`verify_timescales.py` Test 2, benchmark point):

| Between | Ratio |
|---|---|
| $\tau_0$ and $\tau_1$ | **9981.9** |
| $\tau_1$ and $\tau_2$ | 1.23 |
| $\tau_2$ and $\tau_3$ | 2.80 |
| remaining 39 pairs | 1.01 – 2.71 |

**Exactly one gap exceeds 10×.** Below $\lambda_1$ the spectrum runs smoothly
from 2.28 ns to 0.068 ps — four decades, no internal break.

**What is NOT retracted — important.** The two-timescale *separation* stands,
confirmed by two independent routes: (i) the single $\sim10^4$ spectral gap, and
(ii) direct ODE integration by matrix exponential (Test 5, no eigenvalues used),
which shows the error norm dropping, sitting on a **plateau spanning ~three
decades in time** (~1 ns to ~1 µs), then dropping again. A plateau of that width
is the signature of genuine separation; a pure continuum would decline
monotonically. **The headline — $M=9982$, one dominant gap, clean two-stage
relaxation — is intact.** Only the description of the fast group's internal
structure changes.

**Correction this forces (B4).** The "three groups: picosecond ℓ-mixing /
nanosecond relaxation / microsecond ionisation" framing is **not supported** and
must be dropped. The ps and ns "groups" are one distribution.

**Thesis sentence:**
> The CR spectrum exhibits one dominant gap, of order $10^4$, separating the
> ionisation mode from a quasi-continuum of 42 fast modes. $\tau_{\rm relax}$ is
> the slowest member of that continuum rather than an isolated mode; the modes
> immediately below it lie within a factor of two, so Stage-1 relaxation is
> genuinely multi-modal even though the two-stage separation is clean.

This is *stronger* than the earlier claim — it survives an examiner plotting the
eigenvalues.

**The near-degeneracy is a mechanism, not an embarrassment.** $\tau_1=2.277$ ns
and $\tau_2=1.852$ ns differ by 1.23, and
$t_{\max}=\ln(|\lambda_2|/|\lambda_1|)/(|\lambda_2|-|\lambda_1|)\approx2.05$ ns —
matching the observed transient bump at 1–4.6 ns. Closely spaced rates give a
broad, late transient rather than a sharp spike, so the multi-modal Stage 1
directly explains the transient behaviour Q5 rests on.

**Also established:**
- Framing A (full matrix $\lambda_1$) vs Framing B (excited block, ground removed)
  agree to **0.0098%** at the reference point, <0.35% across the grid. **Why:**
  $\lambda_0$'s eigenvector has ground-state weight *exactly* 1.0000, so deleting
  the ground state removes that mode and nothing else. **Why not exact:**
  $\lambda_1$ retains 0.5567 ground weight (it is a ground↔excited exchange
  mode), so removing the ground state perturbs it slightly.
- Mode identity: $\lambda_0$ PR 1.00, pure 1S (ionisation drain); $\lambda_1$
  PR 2.64 spanning $n=1$–3, ground weight 0.56 (ground↔excited exchange);
  mode 3 ground weight exactly 0.0000, spans $n=3$–9 (intra-excited, ℓ-mixing).

**Thesis home:** Ch. 3 §3.4 (two timescales), Ch. 4 Gate E.

---

### A5. $\tau_{\rm relax}$ density scaling — a moving bottleneck ▶️ Run

**This is a physical reinterpretation, not merely a corrected exponent.**

**Measured** (`verify_timescales.py` Test 6): $d\ln\tau_{\rm relax}/d\ln n_e$ runs
from $-0.456$ ($T_e=1$ eV) through $-0.476$ (2.95 eV) to $-0.529$ (10 eV).

**Limits for comparison:** a fixed, purely *collisional* bottleneck gives $-1.00$
(every collisional rate carries a factor $n_e$); a purely *radiative* one gives
$0$ ($A$ coefficients are density-independent).

**Interpretation.** The measured $\approx-1/2$, together with its systematic
drift with $T_e$, is **not** a fixed level with a 50/50 loss split. It is a
**moving bottleneck**: as $n_e$ rises, (i) collisional rates increase $\propto
n_e$, and (ii) the controlling mode migrates to lower $n$ (5G → 4F → 3D → 2P,
see A1) where radiative rates are faster. The two effects carry different
density dependences and partly offset. A fixed level could not produce a
$T_e$-drifting exponent.

**Thesis sentence:**
> $\tau_{\rm relax}$ scales as roughly $n_e^{-1/2}$ rather than $n_e^{-1}$
> because the relaxation bottleneck is not a fixed level: as density rises the
> controlling mode migrates to lower principal quantum number, so the increase in
> collisional rate is partly offset by the changing identity of the controlling
> level.

**Supersedes** the Ch. 3 §3.4.3 claim $\tau_{\rm relax}\propto n_e^{-1.00}$
(correction B1). Directly coupled to A1 — the same migration explains both.

**Thesis home:** Ch. 3 §3.4.3, Ch. 5.

---

### A6. Timescale separation protects slow eigenvalues ✅ Verified

**General principle**, established while bounding the ℓ-mixing bug (A4):

> **Timescale separation protects slow eigenvalues from errors in fast rates.**

**Evidence.** Correcting $F(U_m)$ changed ℓ-mixing rates by ×3–7 yet moved
$\tau_{\rm relax}$ by 0.12% at the reference point and <0.85% anywhere on the
grid. ℓ-mixing operates at $10^{11}$–$10^{12}$ s⁻¹ (picoseconds) — three orders
of magnitude faster than $\tau_{\rm relax}=2.28$ ns. On the nanosecond
timescale the $\ell$-sublevels are *already fully equilibrated*, having reached
their statistical distribution within the first picosecond. Speeding up an
equilibration that has already finished changes nothing.

**The converse matters and must be stated.** The same separation means the error
was **not** harmless to absolute ℓ-resolved populations — which is exactly why
the fix still had to be made before any line-ratio work (Q5, Q6).

**Thesis home:** Ch. 4 (robustness argument), and as a defense one-liner.

---

### A3. Non-normality of the CR operator 📐 Derived → needs grid run

Full derivation in `derivation_04c_non_normality.md`. Numerical abscissa at the
benchmark point: $\mu(L) = +1.28\times10^{11}$ s⁻¹ vs spectral abscissa
$-4.40\times10^{4}$ s⁻¹ — a fifteen-order gap. $\mu(L)>0$ *guarantees* transient
growth for some perturbation.

**Remaining to reach ✅:** map $\mu(L)$ across the full $(T_e,n_e)$ grid; test
transient growth in the **Balmer-ratio observable** rather than $L^2$ (the
$L^2$ result does not transfer automatically — see A3 caveat below).

**Essential caveat already established:** transient growth is norm- and
perturbation-dependent — present for random and 2P-localised kicks in $L^2$,
**absent** in $L^1$ and for a ground-state kick. Every claim must state the norm
and the perturbation.

**Thesis home:** Ch. 3 §3.5 (metrics), Ch. 5 (mechanism behind the central
argument).

---

### A4. ℓ-mixing implementation error, quantified and bounded ✅ Verified

Found the $F(U_m)=1$ error in `compute_lmix.py`, traced it to Badnell (2021)
Eq. 9 read from the source, quantified it (rates under-counted ×3–7), corrected
the code, and proved $\tau_{\rm relax}$ is insensitive (0.12% at the reference
point, <0.85% anywhere on the grid). Regenerated `L_grid.npy`; new headline
values $\tau_{\rm QSS}=22.73$ µs, $\tau_{\rm relax}=2.277$ ns, $M=9982$.

**Thesis home:** Ch. 2 (atomic data), Ch. 4 (as a robustness result — finding and
bounding one's own bug is evidence of rigour).

---

### A7. The error measure must exclude the ground state — and QSS is then robust ▶️ Run (needs independent re-verification)

**This changes the central claim of the thesis. Treat as provisional until
re-verified in a fresh session.**

**The measurement problem.** At the benchmark point, ground population is
$8.80\times10^{-4}$ against a summed excited population of $1.21\times10^{-5}$ —
the ground state is **73× the entire excited manifold**. An $L^2$ norm over all
43 states is therefore effectively an error measure on $n_{1s}$.

**Why that is the wrong quantity.** QSS is an approximation about the **excited
manifold only**: excited states are assumed to equilibrate instantly while the
ground state and $n_e$ evolve freely. The ground state was never supposed to be
QSS-approximated, so including it in the norm measures something QSS never
claimed.

**Measured consequence** (full 43-state integration under a finite ramp, LSODA,
convergence-checked across rtol $10^{-6}$–$10^{-11}$ — identical to five digits,
so physics rather than numerics):

| Drive $\tau_d$ | ALL states | EXCITED only | ratio |
|---|---|---|---|
| 1 µs | 1.697 | 0.0555 | 31× |
| 100 µs (ELM crash) | 0.205 | **0.00724** | 28× |
| 1 ms (fast detachment) | 0.0276 | **0.00104** | 27× |

Step errors: $J = 1.780$ (all states), $J = 0.333$ (excited only).

**With the correct measure, QSS holds to better than 1% at every
divertor-relevant timescale.**

**Two wrong turns recorded deliberately** — both were Claude's, both survived
plausible reasoning, and both died on contact with the student's real data:

1. *"Scalar suppression kills the step error, so the breakdown claim is
   unsupported."* Based on the single-mode scalar model before the full system
   was integrated.
2. *"Non-normality defeats the suppression by 5000×, so the breakdown claim
   survives."* The 5000× came from comparing the full-system error against a
   scalar prediction built on $\tau_{\rm relax}$, when the ground-dominated norm
   was in fact relaxing on $\tau_{\rm QSS}=22.7$ µs. With the correct timescale
   the scalar criterion agrees with the full system to within a factor of two.

**Consequence for the thesis claim.**

*Not supported:* "QSS breaks down in ITER divertor conditions."

*Supported, and stronger:*
> QSS validity is governed by $De=\tau_{\rm relax}/\tau_{\rm drive}$, not by the
> timescale ratio $M$. We derive the suppression law
> $\varepsilon_{\rm peak}/J = De\,(1-e^{-1/De})$, verify it against full 43-state
> integration, and establish that QSS remains valid to better than 1% throughout
> the ITER divertor operating range — with a quantified failure threshold at
> $\tau_{\rm drive}\sim40$ ns, roughly 2500× faster than an ELM crash.

A validity map with a derived criterion, where the literature offers assertion.

**Untouched by this:** A1 (boundary descent), A2 (two-timescale structure), A3
($\mu(L)>0$, transient growth — still true, simply not what breaks QSS), A4
(ℓ-mixing bug), A5 (moving bottleneck).

**Open — the question that decides everything (see C11).** The 0.7% figure is
*one grid point*. $\tau_{\rm relax}$ reaches 38.9 ns at $T_e=1$ eV,
$n_e=10^{12}$ cm⁻³ — 45× slower than at the reference. Whether a breakdown
regime exists anywhere on the grid is **not yet known**.

**Also open:** which $\tau$ enters $De$ for the excited-only measure. The
all-states norm relaxes on $\tau_{\rm QSS}$ (ground-dominated); the excited-only
norm *should* relax on $\tau_{\rm relax}$. **Verify rather than assume** —
mislabelling exactly this produced wrong turn 2.

**Thesis home:** Ch. 3 §3.5 (definition of the error metric), Ch. 5 (the central
result), Ch. 1 and 7 (the claim itself).

---

## B. CORRECTIONS FORCED BY THESE RESULTS

Results that **contradict** what the current thesis text says. Each must be
fixed in Phase 2.

| # | Current thesis claim | Measured value | Status |
|---|---|---|---|
| B1 | $\tau_{\rm relax}\propto n_e^{-1.00}$ (Ch. 3 §3.4.3) | $n_e^{-0.46}$ to $n_e^{-0.53}$, drifting with $T_e$ | **Physical reinterpretation, not just a number.** A fixed collisional bottleneck gives $-1$; the measured $\approx-1/2$ with $T_e$-drift means a *moving* bottleneck. See A5 |
| B2 | $\tau_{\rm relax}=25$ ns, $\tau_{\rm QSS}=15.3$ µs, $M=611$ (Table 4.6) | 2.277 ns, 22.73 µs, 9982 | Stale — spurious intermediate eigenmode in the pre-correction matrix |
| B3 | Hα transient $-46\%$ dip at 19 ns (Ch. 5) | $+43.7\%/-30.3\%$ monotonic | Artifact; same root cause as B2 |
| B4 | "Three timescale groups" (ps / ns / µs) | One gap ($\sim10^4$), then a 42-mode quasi-continuum | **Reframe, do not retract.** The two-stage separation is independently confirmed by the ODE plateau; only the fast group's internal structure changes. See A2 |
| B5 | Title and central claim: *"Quantifying QSS **Breakdown**"* | Excited-manifold error <1% at all divertor drive timescales | **Provisional, pending C11.** If no breakdown regime exists on the grid, the claim must become a *validity map with a derived criterion*. **Discuss with Prof. Pala early** — this is a framing change, not a number fix. See A7 |

---

## C. IDEAS NOT YET DERIVED 💡

Add freely. Nothing here is executed until the physics is understood.

### C1. Map $\mu(L)$ across the grid
Is the non-normality uniformly extreme, or does it peak where QSS breaks down?
If $\mu(L)$ correlates with the observed QSS error better than $M$ does, that is
a genuine result: a *predictive* non-normality criterion.

### C2. Participation ratio across the spectrum
PR vs mode index, across the grid. Would quantify "collective vs single-level"
rigorously rather than by eyeballing components, and would sharpen the
"$\tau_{\rm relax}\neq1/A_{21}$" argument into a number.

### C3. Where does the fast continuum come from?
Modes below $\lambda_1$ form a continuum spanning four decades. Preliminary
inspection suggests they are ℓ-mixing modes ordered by shell (higher $n$ →
faster, following $n^4$). A systematic classification (which $n$, how many sign
changes) would turn the "quasi-continuum" statement into a structural one.

### C4. Pseudospectra
The rigorous tool for non-normal operators. Would strengthen A3 considerably,
but may belong in the paper rather than the thesis. **Decide before spending
time on it.**

### C5. Does the eigenvector localisation predict the Hα behaviour?
$\lambda_1$ involves 2P and 3D strongly — the Balmer-α upper level is $n=3$. Is
the Hα transient magnitude predictable from $|v_{\lambda_1,3D}|$? If so, that
links the eigenstructure directly to the observable, which is exactly the
diagnostic-bias story.

### C6. Truncation sensitivity
The model bundles $n=9$–15 and truncates at 15. At low density the relaxation
mode has weight on the bundled states (A1 shows $\langle n\rangle\approx11$ at
$n_e=10^{12}$, $T_e=10$ eV) — so the truncation may be affecting
$\tau_{\rm relax}$ there. Test by re-running with $n_{\max}=12$ and $n_{\max}=20$
and comparing. **This is a defensible-scope question an examiner may well ask.**

### C11. Does a QSS breakdown regime exist ANYWHERE on the grid? (excited-only measure, full grid)

**The single most important open question in the project.** It decides whether
the thesis is a breakdown demonstration or a validity map.

A7 established that with the ground state excluded, the excited-manifold QSS
error is **0.7% at ELM timescales** — at *one grid point*. But
$\tau_{\rm relax}$ varies by 45× across the grid: 0.87 ns at $T_e=10$ eV,
$n_e=10^{15}$, rising to **38.9 ns** at $T_e=1$ eV, $n_e=10^{12}$. Since
$De=\tau_{\rm relax}/\tau_{\rm drive}$, the low-density corner is 45× closer to
breakdown than the reference point.

**Test:** integrate the full 43-state system under a finite ramp at every
$(T_e,n_e)$ grid point, for each divertor drive timescale, and map
$\varepsilon^{\rm excited}_{\rm peak}$. Report the maximum and where it occurs.

**Two possible outcomes, both publishable:**
- *Error stays <1% everywhere* → QSS is robust across the entire ITER divertor
  operating range. The thesis is a validity map with a derived criterion.
- *Error becomes large in some corner* → that corner is the breakdown regime,
  and the thesis can characterise it specifically rather than claiming breakdown
  generally.

**Cost note:** this is 400 grid points × 4 drives × a stiff 43-state
integration. The single-point run at $\tau_d=10^{-3}$ s was already expensive
and one earlier full run was killed by the OS. Budget for it, or subsample the
grid first to locate the worst corner before running it densely.

**Do this before Q6 and Q7** — both were designed to characterise a breakdown
that may not occur.

### C7. Cold-corner exclusion
19 grid points have $\tau_{\rm QSS}>1$ s (max 67 s at $T_e=1$ eV,
$n_e=10^{12}$). Physical but operationally meaningless, and they inflate the
quoted $M$ range by orders of magnitude. Decide on a principled exclusion
criterion and state it, rather than quietly quoting the full range.

### C8. Verify the ℓ-distribution within $n=3$ and $n=4$ is statistical at both step endpoints

**Why it matters.** The Balmer error measure
$\varepsilon^{\rm ratio}_{\rm step} = \dfrac{n_{3,\rm old}/n_{4,\rm old}}{n_{3,\rm new}/n_{4,\rm new}} - 1$
was derived assuming the $A$-coefficients cancel exactly between old and new
states. That cancellation is what makes the result **independent of atomic
transition rates** — a genuinely valuable robustness property, worth stating in
the thesis.

But $H_\alpha$ is $3\to2$ with three dipole-allowed channels ($3s\to2p$,
$3p\to2s$, $3d\to2p$), each carrying its own $A$. So "$A_{3\to2}$" is really an
$\ell$-weighted average. **If the $\ell$-distribution within $n=3$ differs
between the old and new states, the effective $A_{3\to2}$ is not identical
old-to-new and the cancellation is only approximate.**

**Expected outcome:** ℓ-mixing runs at $10^{11}$–$10^{12}$ s⁻¹ (Q4b), roughly
1000× faster than anything else, so both endpoints should be statistically
populated and the cancellation should be excellent. **But this is an assumption
to check, not to assume** — it is exactly the kind of thing that produces a
factor-of-two surprise late in a defense.

**Test:** compute $n_{3\ell}/n_3$ and $n_{4\ell}/n_4$ at both endpoints and
compare against the statistical weights $(2\ell+1)/n^2$. Quantify the deviation
and propagate it into $\varepsilon^{\rm ratio}_{\rm step}$.

### C9. Map $\varepsilon^{\rm ratio}_{\rm step}$ against $\varepsilon^{L^2}_{\rm step}$ — where does common-mode cancellation fail?

**The finding that motivates this.** At the benchmark point with a $+0.6$ eV step
(preliminary run, stand-in source vector):

| Quantity | Value |
|---|---|
| $\varepsilon^{L^2}_{\rm step}$ | 1.17 |
| $\lvert\varepsilon^{\rm ratio}_{\rm step}\rvert$ | 0.038 |
| ratio | **31×** |

Mechanism, visible directly in the numbers: $n_3$ changes by $+18.7\%$, $n_4$ by
$+14.4\%$ — they move **together**. The Balmer ratio sees only the $4.3\%$
*difference*. This is **common-mode cancellation**, and it means the ratio
diagnostic is far more robust to QSS error than the population vector is.

**This cuts both ways, and the honest reading matters:**

- *Against the simple diagnostic-bias story:* a 117% error in the population
  vector produces a 3.8% error in the measured ratio. "QSS-based Balmer
  diagnostics are badly biased" is **not supported** at this step size. Do not
  claim it.
- *For a better result:* we now have a quantitative account of *why* the standard
  diagnostic is trustworthy, plus a framework for locating where it is not. The
  ratio fails when $n_3$ and $n_4$ stop moving in common mode — presumably for
  large enough or fast enough steps, where the shells decouple.

**The sharper question, and the better paper:** *where on the $(T_e,n_e)$ grid,
and for what step sizes, does the common-mode cancellation break down?*

**Test:** map both error measures across the grid for a range of $\Delta T_e$;
plot their ratio; identify the regime boundary where the ratio error becomes
comparable to the norm error.

**Caveat on the preliminary numbers.** The run above used a stand-in source
vector (uniform excited feed, ground-dominated), not the real `S_grid.npy`. The
*magnitudes* must be recomputed with the real recombination source. The
*structural* result — common-mode cancellation suppressing the ratio error — is
robust to the source choice, since it depends on how $n_3$ and $n_4$ respond
similarly to a $T_e$ step, not on how they are fed.

---

### C10. Is the excited manifold excitation-fed or recombination-fed at the benchmark point? Which shell is more $T_e$-sensitive?

**Why this is a real physics question, not bookkeeping.** It decides how the sign
of $\varepsilon^{\rm ratio}_{\rm step}$ is *explained* in the thesis, and the
obvious explanation appears to be wrong.

**The naive (Boltzmann) expectation.** $n=4$ sits at higher excitation energy
than $n=3$, so its population should be more $T_e$-sensitive. Heating should
raise $n_4$ fractionally more than $n_3$, lowering $R=n_3/n_4$, giving
$\varepsilon^{\rm ratio}_{\rm step}>0$ for a heating step.

**What the preliminary run actually shows** (benchmark point, stand-in source
vector — magnitudes provisional):

| Step | $\Delta n_3/n_3$ | $\Delta n_4/n_4$ | More sensitive | $\varepsilon^{\rm ratio}_{\rm step}$ |
|---|---|---|---|---|
| Heating $+0.78$ eV | $-19.1\%$ | $-15.3\%$ | **$n_3$** | $+0.047$ |
| Cooling $-0.62$ eV | $+26.5\%$ | $+20.7\%$ | **$n_3$** | $-0.046$ |

**The predicted sign is confirmed and the flip with step direction is real** —
but the mechanism is *not* the Boltzmann one. Two things contradict it:

1. **$n_3$ is more $T_e$-sensitive than $n_4$**, the opposite of the naive
   expectation.
2. **Heating *depletes* both shells** rather than populating them.

**Likely explanation, to be verified.** If the excited manifold is fed
predominantly by **recombination** rather than by excitation from the ground
state, then heating reduces the feed (recombination rate falls with rising
$T_e$) and depletes the excited states. That is a *recombining-plasma* regime,
and Boltzmann reasoning — which presumes excitation-driven population — does not
apply to it.

**Essential caveat.** The preliminary run used a stand-in $\mathbf b$ (uniform
excited feed, ground-dominated), which may itself be forcing the recombining
regime. With the real `S_grid.npy` the balance between recombination feed and
ground-state excitation may differ, **and the sign could flip.**

**Test:**
1. Recompute with the real source vector.
2. Decompose the steady-state population of $n=3$ and $n=4$ into contributions
   from (a) recombination feed via $\mathbf b$ and (b) collisional excitation
   from the ground state — i.e. which term dominates $\mathbf n^{ss}$.
3. Map the regime across the $(T_e,n_e)$ grid: where is the divertor plasma
   ionising, and where recombining?
4. Only then write the explanation of the $\varepsilon^{\rm ratio}$ sign.

**Do not write the Boltzmann explanation into the thesis** — it does not match
even the preliminary data.

**Connects to:** C9 (the same $n_3$/$n_4$ response controls where common-mode
cancellation fails), and to the ITER-divertor detachment literature, where the
ionising→recombining transition is exactly the physics of interest.

---

## D. CODE TO REVIEW LINE BY LINE

Per operating rule 7 (no black boxes) and §5.1 of the session plan. Order
reflects how load-bearing each is for the thesis.

| Script | Why it matters | Status |
|---|---|---|
| `cr_context.py` | Loads grids and state ordering; everything else depends on it | ⏳ |
| `verify_timescales.py` | Produces A1, A2 | ⏳ |
| `verify_boundary_descent.py` | Produces A1 | ⏳ |
| `qss_analysis.py` | Produces the Q5 headline figures; **known issue**: `eigs < -1.0` filter silently truncates $\tau_{\rm QSS}$ at $\sim1$ s | ⏳ |
| `Balmer_transient_ratio.py` | The diagnostic observable for Q5 | ⏳ |
| `assemble_cr_matrix.py` | Builds $L$; partially audited (conservation verified) | 🔄 partial |
| `compute_lmix.py` | Audited and corrected | ✅ |

---

## E. HOW TO USE THIS DOCUMENT

1. **Add an idea the moment it appears** — a line under §C is enough. Do not
   stop to derive it; the point is not to lose it.
2. **When the learning sequence reaches the relevant physics**, promote the entry
   from 💡 to 📐 by deriving it.
3. **Then write the code together, understood line by line** — what goes in, how
   it computes, what comes out. No script is run before it is understood.
4. **Run on Nikhil's own system.** Results printed here are not evidence until
   reproduced there.
5. **Verify**: sensitivity-test the definitions, look for the result that would
   *refute* the claim, and write down the caveats explicitly.
6. **Only then** promote to ✅ and write it into the thesis.

---

### Verification note — `cr_context.py` state-ordering gate (2026-08-22)

Ran ad hoc verification (no repo files modified; scratch script only):
`CRContext.load()` from `src/validation/cr_context.py`, plus direct
`stat`/`shasum`/`cmp`/`diff` on both files named in
`REL_STATE_INDEX_CANDIDATES`.

- Both candidate files exist:
  `data/processed/collisions/K_exc_full/state_index.csv` (1288 B, SHA-256
  `23eb18538c19165b6872dcae783944cd28f7e12dd3c741c7325645a623979a65`,
  mtime 2026-03-23 14:54) and
  `data/processed/Radiative/state_index.csv` (1388 B, SHA-256
  `02a19cfc9b9326a05b01758946ea290fe4efdbcad3e8d364f29ad4fd5960afe1`,
  mtime 2026-03-22 01:31).
- They are **not identical**: `cmp` differs at char 5; parsed headers differ
  (`idx,label,n,l,bundled,g,I_eV` vs `idx,n,l,label,type,gamma_rad_s-1`) — a
  genuine schema/content difference (different auxiliary physical columns),
  not a line-ending artifact. However the `idx,n,l,label` values themselves
  are identical row-for-row across both files (same 43 states, same ordering).
- The loader (`CRContext.load`, picks first existing candidate in
  `REL_STATE_INDEX_CANDIDATES`) resolves to `K_exc_full/state_index.csv`
  today, since it is listed first and exists. Confirmed via
  `ctx.state_index_path`.
- Resulting ordering: 43 states total; N3 = indices [3,4,5] (count 3); N4 =
  indices [6,7,8,9] (count 4). Matches the expected mapping exactly. Ground
  index = 0 (`1S`).
- `L_grid.npy` SHA-256 confirmed matching the recorded reference
  (`2d92b58e...059224e`).

**Graduation:** ▶️ (executed, current behavior verified against both
candidate files and the actual loader resolution). Not ✅: this check
confirms the *current* state of two files and the loader's *current*
tie-break behavior (first-existing-candidate wins); it does not by itself
guarantee no other script in the repo reads `Radiative/state_index.csv`
directly under its own (different) column assumptions, nor that the
candidate-list order in `cr_context.py` won't matter if `K_exc_full`'s file
is ever absent or regenerated with a different row order. Caveat: the two
files carry genuinely different auxiliary columns (weights/ionization energy
vs. radiative rates) under nominally the same idx/n/l/label ordering — worth
tracing to source before treating either as fully redundant with the other.

---

### Verification note — thesis_ready.md PART A results check (2026-09-09)

Independent re-run of the named producers against the canonical matrix. No repo
file was edited; no tolerance, filter or data file was touched. Scripts that
write into `validation/` were run either with `--out` to a scratch directory or
from an isolated sandbox cwd, so the committed evidence files were preserved.
Interpreter: `/opt/anaconda3/envs/cr/bin/python` (the one named in
`run_pipeline.sh`), numpy 2.3.5 / scipy 1.16.3.

**Integrity — all three canonical hashes match thesis_ready.md exactly:**
`L_grid.npy` `2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`
(mtime 21 Jul 20:44), `S_grid.npy`
`7822f536590c76bae8355bddef023fc53e0f3595d2f4687313792e8f277fb80a`,
`data/processed/collisions/K_exc_full/state_index.csv`
`23eb18538c19165b6872dcae783944cd28f7e12dd3c741c7325645a623979a65`.

**A1 (thesis_ready) / backlog A2 — reproduced, with one documented failure.**
`verify_timescales.py` (no args) at grid `[23,5]` (Te = 2.9471 eV,
ne = 1.3895e14): τ_QSS = 2.272799e-05 s, τ_relax = 2.276913e-09 s,
M = 9981.93, largest spectral gap 9981.9× at k=0 (exactly one gap > 10×),
Framing A vs B 0.0098%. Matches the recorded values to all quoted digits.
`qss_analysis.py`, run in a sandbox cwd, wrote `M_grid`, `tau_QSS_grid`,
`tau_relax_grid` **bit-identical** to the committed `validation/*.npy`, and
`breakdown_map.csv` bit-identical column-for-column including the LSODA-derived
`eps_res`. `timescales_unfiltered_CHECK.npz` is bit-identical to both. The
"three independent implementations agree bit-for-bit" claim holds today.
Grid-wide over all 400 points: M ∈ [86.7678, 1.729277e9] ✓,
τ_relax ∈ [8.687198e-10, 3.887945e-08] s ✓ (0.87–38.9 ns, a 44.8× spread),
τ_QSS max 6.723334e+01 s ✓. **τ_QSS min does NOT reproduce**: measured
7.537688e-08 s at `[49,7]`, not 1.18 µs — a factor 15.6, and 46 of 400 points
lie below the stated floor. `verify_ch3_claims.py` reports this as its only
FAIL (rel 9.36e-01) and traces it: 1.177240e-06 s is the τ_QSS minimum over the
**M > 900 window_ok subset (346 of 400)**. Eq. (3.14) as written in
thesis_ready.md line 27–28 mixes scopes — M and τ_relax over all 400 points,
τ_QSS over the analysed subset — and needs an explicit clause. Everything else
in that script passed: grid endpoints, benchmark indices, isolation 3566.96×
(chapter 3567), grid-min isolation 24.33 (chapter 24), M⁺ = 8242.74.

**Sensitivity (numerical conditioning of A1).** Eigenvalue condition numbers
1/|yᴴx| at [23,5]: λ₀ 1.71, λ₁ 1.98; at the cold corner [0,0]: 3.26 / 2.59; at
[49,7]: 1.16 / 1.27. All O(1) — the eigenvalues are well conditioned despite
‖L‖₁ reaching 2.0e14 and λ₀ falling to −1.49e−02 s⁻¹. `scipy.linalg.eig` agrees
with `numpy.linalg.eigvals` to 0 ulp at all three points. Under 20 random
relative perturbations of every entry at 1e−13, τ_QSS moves by 3.3e−08 relative
at the benchmark and 4.6e−04 relative at the cold corner; τ_relax by ≤1.4e−10
everywhere. So "67.2 s" is numerically sound to ~4 digits — the existing caveat
against quoting three figures rests on the physics sensitivity
(d ln τ_QSS/d ln Te = −13.3), not on roundoff. **Note the A1 numbers involve no
stiff integration**; the convergence-table requirement bites on A4/A5, not here.

**A8 / A9 — reproduced bit-for-bit.** `verify_plateau_gridmap.py --out <scratch>`
reproduced the committed `plateau_gridmap.csv`/`.txt` byte-for-byte apart from
the generation-timestamp line. ε_plateau > ε_step at 338/338 heat + 342/342 cool
= **680/680** ✓; amplification min 1.44738 (cool) ✓, median 12.70 heat /
10.77 cool ✓; sharpest case Te = 6.866 eV, ne = 5.1795e13, ε_step = 0.000049,
ε_plateau = 0.044974 ✓; step-normalised heat/cool median 1.0360 ✓; amplification
max 1271.36 ✓ (correctly flagged do-not-quote). A9: prediction ratio median
1.0310 heat / 0.9851 cool, range 0.7659–1.3891 ✓; corr +0.9893/+0.9920 ✓;
|f₃−f₄| alone +0.6587/+0.6938 ✓; |ln x_new| 0.124039–0.68228 ✓. The committed
file was already git-dirty on the timestamp line only — i.e. it had also
reproduced exactly on 24 Aug.

**A10 — all 16 ridge numbers reproduce**, at grid rows i = 0, 10, 23, 35 and
columns j = 0, 3, 5, 7, from `plateau_gridmap.csv`: 0.08300 0.37093 0.30333
0.13138 / 0.07095 0.23220 0.14538 0.05019 / 0.04873 0.12166 0.06361 0.01831 /
0.03195 0.07084 0.03555 0.00949. Independently confirmed by
`verify_eps_gridmap.py`, a separate implementation with no window guard, which
returns 12.166% at [23,3] and 18.073% as the Te ≥ 2 eV restricted maximum.
**Two caveats the table does not carry.** (i) It is the **heating** direction
only; the cooling table is materially different (0.28585 vs 0.37093 at [0,3],
0.18044 vs 0.30333 at [0,5]) and the ridge maximum moves from [0,4] to [1,3].
The direction must be stated. (ii) The bottom-right entry 0.009 (Te = 5.179,
ne = 1e15) has `window_ok = False` in both directions — it is excluded from
every statistic in the same file it is quoted from.

**A11 — reproduced bit-for-bit.** `verify_divertor_map.py --out <scratch>`
reproduced the committed `divertor_map.csv` byte-for-byte and the `.txt` apart
from the timestamp. Lower bound above 10% at τ_d = 1e-4 s: **202/680** ✓
(upper bound 225/680); 105 heating points, Te 1.000–2.947 eV,
ne 2.683e12–1.000e15 ✓; worst Te = 1.0000, ne = 5.1795e13, lower = 0.3868,
ε_plateau = 0.3869, τ_QSS = 2.332e-01 s ✓ (233 ms, and the two bounds do
coincide); benchmark point ELM lower bound 0.011712 = 1.2% ✓. Note the
benchmark row of that file carries τ_QSS = 1.8494e-05 s and M = 8243, i.e. the
**post-step** operator L[24,5], not the 2.2728e-05 s / 9982 of A1 — the two
numbers are different operators and the thesis must say which it quotes.

**A11 sensitivity — plateau-window factor k (11 Sep 2026, report-only, 💻→▶️).**
Reviewer question: is A11's census (45/448, worst 0.1748 at heat[15,3], 24/448
at 506 µs) stable if the window `k*tau_relax < t < tau_slow/k` uses k = 10, 20,
50 instead of the thesis's k = 30? Ran
`verify_divertor_map.py --win-lo K --win-hi K --out validation/divertor_map_wK`
for K = 10, 20, 50 (each 0.3 s wall time; canonical `validation/divertor_map/`
confirmed byte-identical before and after by sha256, so the sweep did not
touch it), then recounted from all four CSVs with a new script,
`src/validation/verify_window_sweep.py`, written for this check and saved for
reuse; output `validation/window_sweep/window_sweep_summary.csv` and `.txt`.

K = 30 recount reproduces A11/A1-adjacent figures exactly: n_window_ok = 680,
n_warm (Te ≥ 2 eV) = 448, n_dense (also ne ≥ 1e14) = 108, census@100µs = 45,
worst = 0.174812 at heat[15,3] (Te = 2.0236 eV, ne = 1.9307e13 cm⁻³),
census@506µs = 24 (worst 0.153301) — all match the values already on record.

**What changes with K, what does not.** n_window_ok falls monotonically as K
grows (779 → 728 → 680 → 596 for K = 10, 20, 30, 50) because the gate is
`M > K²` and M is fixed per row; n_warm and n_dense fall with it (547→496→
448→364 warm; 202→151→108→66 dense). **The census counts do not move at
all**: census@100µs = 45/45/45/45, worst = 0.174812 at heat[15,3] in every
case; census@506µs = 24/24/24/24, worst = 0.153301 in every case; thr = 5%
count 141 and thr = 20% count 0 at every K. `eps_plateau` (the frozen-reservoir
algebraic solve) is bit-identical across K for all 784 rows, max|Δ| = 0.000e+00
exactly — confirmed both by direct row-by-row comparison and inside the new
script. Same identity check run on `verify_reservoir_gain.py` (which does
accept `--win-lo/--win-hi/--out`; run at K=10, K=50 with `--write`, canonical
untouched, confirmed by sha256): G and Sbar bit-identical to the K=30 file for
all 2288 rows, only the `window_ok` flag flips (285/2288 rows at K=10,
245/2288 at K=50). The reported `|Sbar|`/`|G|` *ranges* over "k=1 heating,
window_ok" narrow or widen with K (canonical 0.0649–0.4822 / 2.64–14.52;
K=10 0.0491–0.4822 / 2.64–14.52; K=50 0.0786–0.4822 / 2.647–14.52) but this is
membership at the edge of the distribution, not a changed value — the row
that used to set the K=30 minimum G (2.6396) simply exits the window at K=50
and a different row (2.647) becomes the new extremum among the smaller set.

**One thing that does change, and was not anticipated going in: the crest.**
For 9 of 34 warm (Te ≥ 2 eV) heating rows — the hottest, Te = 6.55–9.54 eV —
the density index j at which `eps_plateau` is largest **among window_ok
columns** moves from j=3 (K ≤ 30) to j=2 (K=50, Te = 6.55–8.69 eV) or j=1
(K=50, Te = 9.10–9.54 eV). This is purely a membership effect: `eps_plateau`
at j=3 is unchanged (e.g. 0.05815 at i=40 for every K) but at K=50 that column
fails `M > 2500` (M = 2457.6 there) and drops out of window_ok, so the
argmax among the *surviving* columns falls back to smaller ne. The true,
window-unrestricted maximum over ne never moves. This qualifies the "k sets
census membership, never magnitude" claim: it is exactly true for the 100 µs
and 506 µs census reported in the thesis (their worst-case row, heat[15,3] at
Te ≈ 2 eV, stays comfortably inside the window at every K tested), but a
crest-style "which density is worst at fixed Te" statement is not
K-invariant at the hot edge of the grid and should not be extended there
without saying so.

**What would have refuted "stable":** a change in census@100µs or
census@506µs count, or in the worst-case value/location, across K = 10, 20,
30, 50. None appeared. A change in eps_plateau, G, or Sbar with K would have
meant the window gate was leaking into the "physics" quantities instead of
only gating which rows are counted; none appeared (max|Δ| = 0 exactly, not a
small residual). The crest shift at the hot edge is the one caveat that did
appear and must be carried alongside the "stable" headline.

**Runtimes.** `verify_divertor_map.py`: ≈0.3 s wall per K (10, 20, 50).
`verify_reservoir_gain.py --write`: ≈0.6 s wall per K (10, 50).
`verify_window_sweep.py`: <1 s. No failures, warnings, or kills.

**Graduation: 💻 → ▶️.** Sensitivity check run as requested; caveats above are
written. Not ✅ — this entry inherits A11's own un-promoted status (the
[23,5]-vs-[24,5] operator-identity caveat above) and adds one of its own (the
crest is not K-invariant at Te ≳ 6.5 eV). Artifacts:
`validation/divertor_map_w10/`, `validation/divertor_map_w20/`,
`validation/divertor_map_w50/`, `validation/reservoir_gain_w10/`,
`validation/reservoir_gain_w50/`, `validation/window_sweep/`. Script:
`src/validation/verify_window_sweep.py`.

**A12 — reproduced exactly, from the preserved `*_FILTERED_20260721` evidence.**
Comparing `tau_QSS_grid.npy`/`tau_relax_grid.npy`/`M_grid.npy` against their
`_FILTERED_20260721` counterparts: **19 of 400** points differ ✓, at Te indices
0–7 (1.0000–1.3895 eV) and ne indices 0–3 (1.0000e12–1.9307e13) ✓; at **19/19**
of them τ_relax^new == τ_QSS^old **exactly** (`np.array_equal`), the ladder
shift confirmed ✓; M range 1.34092–1.011973e8 → 86.7678–1.729277e9 ✓;
`eps_step` bit-identical at all 400 points in `breakdown_map.csv` while
`eps_res` and every `eps_bar`/`eps_end` column changed ✓; benchmark [23,5]
bit-identical ✓; 0 points had τ_QSS > 1 s under the filter versus 19 after ✓.
**"Still live and unfixed" is correct today**: `src/rates/solve_cr.py:269`
(`eigs_neg = eigs[eigs < -1.0]   # exclude near-zero numerical noise`) and
`src/rates/check_mz.py:10` (`neg = eigs[eigs < -1.0]`), verified by grep at
those exact line numbers. Not touched.

**Two things found that are not in thesis_ready.md.**
1. `CLAUDE.md`'s known-issue table is **stale**: `qss_analysis.py` no longer
   contains `eigs[eigs < -1.0]`. Line 137 now reads `neg = eigs[eigs < 0.0]`,
   and `validate_gates.py:403` likewise. The table should point at
   `solve_cr.py:269` and `check_mz.py:10` instead.
2. Commit `ffe1768` has a **false commit message**: it states "eigs<-1.0
   removed from solve_cr.py:269 and check_mz.py:10", but its diff for
   `solve_cr.py` changes only a units comment (`cm^3/s` → `s^-1`), and
   `check_mz.py` is not in the commit at all. The filter survives in both.
3. `qss_analysis.py` does **not** use `cr_context.py`; it hardcodes
   cwd-relative `PATHS` and `OUT_DIR = 'validation'` (lines 71–77). It happens
   to resolve correctly when run from the repo root, but it is outside the
   provenance gate CLAUDE.md rule 1 mandates, and it is one of the two writers
   in B7. `audit_writers.py` confirms B7 is still unresolved: all three of
   `M_grid.npy`, `tau_QSS_grid.npy`, `tau_relax_grid.npy` have 2 writers,
   "D.4 NOT IMPLEMENTED", with 6 downstream readers.

**Graduation: ▶️ Run.** Not ✅. A1 cannot be promoted while its stated τ_QSS
lower bound is a subset minimum presented as a grid-wide range; A10 cannot be
promoted while the table omits its direction and includes one `window_ok=False`
point. A8, A11 and A12 reproduced without exception and carry written caveats
already; they are the closest to ✅ but inherit the A1 scope wording.

**What would have refuted these claims, and did not appear:** a hash mismatch on
any of the three canonical files; a τ_QSS/τ_relax/M grid differing from the
committed one at any of 400 points; ε_plateau ≤ ε_step at any of 680 pairs; a
non-negative eigenvalue anywhere (max Re λ = −1.487e−02); complex eigenvalues
(max |Im|/|λ| = 0); a filtered/unfiltered ε_step difference; an eigenvalue
condition number large enough to put λ₀ at the roundoff floor.

---

# SESSION REGISTER — 9 to 10 September 2026

Ten review passes: four on `thesis_ready.md` PART A, three verification agents,
and Gates 1, 3 and 4 of the ten-gate thesis system. All ran read-only against
`L_grid.npy` SHA-256 `2d92b58e…59224e`. Full accounts in
`findings_10_four_agent_review.md` (four addenda), `claim_hierarchy.md`,
`pivot_decision.md` and `CHANGE_REPORT.md`.

## F. IDEAS CLOSED BY THIS SESSION

### C1. Map μ(L) across the grid — ✅ Verified, closed, artifact `validation/operator_conditioning/`

Measured over all 400 points: **min 3.396×10⁸, max 1.513×10¹², benchmark
1.2841×10¹¹ s⁻¹**, reproducing CLAUDE.md's +1.28×10¹¹. μ(L) > 0 everywhere, so
the operator is non-normal at every grid point.

**The question C1 actually asked was whether μ(L) predicts the QSS error better
than M does. It does not, and neither does M.** ADDENDUM D and the correlation
work show M and ε are uncorrelated once (Te, ne) is controlled. There is no
predictive non-normality criterion here. Recording the negative result closes
the idea; a *predictive* criterion was the only thing that would have made it a
thesis result.

### C2. Participation ratio across the spectrum — ✅ Verified, closed, artifact `validation/operator_conditioning/`

PR(v₀) = 1.00, trivially, because λ₀ is the ground-state mode. **PR(v₁) = 2.64
at the benchmark and 1.88 at [49,7].**

This closes C2 and simultaneously **refutes a sentence in Chapter 3**:
`chapter3.tex:444-446` describes v₁ as "distributed across the excited
manifold", which PR ≈ 2 contradicts. The chapter also quotes PR for v₀ only,
where it carries no information, and describes v₁ verbally. Quote both or
neither. See also the norm collision at W3 in ADDENDUM to Gate 4: v₁'s
ground-state weight is 1.0000 in the raw measure and 1.9×10⁻⁴ in the
population-scaled one, and two adjacent sentences use different norms without
naming either.

### C6. Truncation sensitivity — ✅ Verified, closed. This is validation ladder rung 6.

f₃−f₄ against n_max, increments decaying geometrically with ratio ≈ 0.75:

| point | n=10 | 12 | 14 | **15** | extrapolated n→∞ |
|---|---|---|---|---|---|
| benchmark [23,5] | 1.3870e-2 | 1.4494e-2 | 1.4701e-2 | **1.4759e-2** | ≈1.490e-2, **+0.9%** |
| cold corner [0,0] | 2.1168e-1 | 2.0498e-1 | 2.0203e-1 | **2.0109e-1** | ≈1.98e-1, **−1.4%** |
| ridge [0,4] | 5.3830e-2 | 5.3288e-2 | 5.3024e-2 | **5.2936e-2** | ≈5.265e-2, **−0.55%** |

Truncation makes f₃−f₄ slightly **too large** at cold points and **too small**
at the benchmark. It is a 1% effect, not a mechanism. Independently consistent
with Σα_RR(n≥2,≤15) capturing 90% of α_B.

**A methodological warning that must travel with this result.** A truncation
test that drops states without removing their loss channels from the diagonal
produces a spurious 22 to 37 percent jump between n_max = 14 and 15 that looks
exactly like a headline finding. It is an artifact. The correct sub-model adds
back Σ_{k dropped} L[k,i] to L[i,i]. The production matrix is self-consistently
truncated (column sums = −K_ion·n_e to 2.2×10⁻¹²).

**Bonus result, closing an open `\todo` in Chapter 6.** Terminal-shell
over-population measured internally, comparing each shell as terminal against
interior: **9.4× at p=8, 10.7× at p=9, 6.3× at p=10, 4.4× at p=11, 3.1× at
p=12.** The 4.9 to 6.2× excess observed at p=15 against Fujimoto sits inside
this band. This is an internal measurement, which is stronger than the
inference from the external table it replaces.

## G. NEW RESULTS FROM THIS SESSION

### G1. The reservoir gain G is step-independent where ε is not ✅ Verified

**The result the thesis now headlines.** With ε = |exp(S̄·G·Δln Te) − 1| and
G ≡ d ln u/d ln Te, measured over all 368 points carrying k = 1, 2, 4
(`verify_reservoir_gain.py`, `validation/reservoir_gain/reservoir_gain.csv`,
2288 rows):

| | median spread across k | maximum |
|---|---|---|
| \|G\| | 1.0549 | **1.0652** |
| ε | 4.49 | **8.44** |

**|G| varies by at most 6.5% where ε varies by up to a factor 8.4.** Grid-wide
|S̄| runs 0.065 to 0.482 and |G| runs 2.64 to 14.52; the invariant
dε/d ln Te = |S̄·G| runs 0.362 to 6.960, median 1.522.

**Falsifier stated in advance and did not appear:** if G varied across k by as
much as ε does, there would be no step-independent coefficient and the
reframing would fail. Sensitivity: the two-channel identity is enforced as a
raise and the superposition residual is checked at every point.

**Caveat:** G is *stable*, not invariant. Quote 6.5% with it.

### G2. Lyman trapping does not touch the defended range ✅ Verified

`verify_lyman_trapping.py`, self-consistent Θ_P ↔ n(1s) fixed point converged
at all 400 points, all 14 Lyman channels, slab thickness swept. **Above 2 eV
the ELM breakdown count does not move (45/448) and the worst case moves 0.5%
across a twentyfold slab range.** Below 2 eV the 38.7% becomes 11.6 to 15.7%
and the density maximum wanders across three columns.

Gate: oscillator strengths derived from the repo's own A-values give 0.4162,
0.0791, 0.0290, 0.0139 for Ly-α to δ, the literature values to four figures,
with none imported. The σ₀ cross-check against an independent implementation
**failed at 1156% on first run**, catching a 4π error from using the SI Einstein
coefficient with CGS constants. The result exists because that gate fired.

**Open defect in the write-up, not the script:** ADDENDUM A and
`chapter6.tex:143` quote σ₀ = 7.74×10⁻¹⁴ cm², high by exactly √2. Correct value
5.478×10⁻¹⁴. τ per cm at [0,4] is 80.4, not 114, and 7900 follows from no
convention. The script used the right value throughout.

### G3. Quasi-neutrality fails at the cold end ▶️ Run — the hardest new constraint

Holding n_e fixed while n_g varies is self-consistent only where the ionisation
degree is high. **68 of 392 one-step operators require |Δn_e/n_e| > 10%, and 36
require > 100%.** At [0,0] the requirement is +20.0.

The closed-parcel and transport-fed pictures cannot both hold: closed forces
n_e up tenfold to twentyfold at the cold end, transport-fed resupplies the
reservoir so it is not stale, which is the entire mechanism. Above Te ≈ 2 eV,
Δn_e/n_e < 10⁻³. **Te ≥ 2 eV is the only self-consistent region of the grid.**

Supporting: n_g at [0,4] corresponds to 235 Pa neutral pressure, twelve times
the upper end of the ITER divertor design range; at [0,7], 3280 Pa, 164 times.

### G4. The QSS partition is correct everywhere ✅ Verified

2s is **not** a second reservoir. Loss is 99.99% proton ℓ-mixing. With ℓ-mixing
deleted entirely the two-photon rate 8.229 s⁻¹ alone is still **553× faster**
than |λ₀| at the cold corner. Minimum spectral separation over all 400 points
**86.5**, at Te = 10 eV, ne = 10¹⁵. 2s would become a reservoir only below
ne ≈ 7×10³ cm⁻³, nine orders below the grid.

**Quote 86.5, not the benchmark 9982.** This was the one open question that
could have changed the shape of the two-channel decomposition. It does not.

### G5. Detailed balance passes, and the two-energy trap is not sprung ✅ Verified

2457 excitation pairs × 3 temperatures: max deviation **7.31×10⁻⁹**, median
7.5×10⁻¹⁰, consistent with the state table's 8-figure rounding. Ionisation
against three-body via Saha: **0.999997 for every one of the 43 states at every
temperature, to eight identical digits**. Because the ratio is state-independent
across levels whose exp(I_p/kT) spans e^13.6 to e^0.06, the same energy ladder
is provably used on both sides.

### G6. ℓ-mixing is saturated ✅ Verified

Scaling all ℓ-mixing rates over ×0.1 to ×10 moves f₃−f₄ by **< 0.3%** at the
benchmark and ridge and **< 5%** at the cold corner. Only s = 0 changes
anything. A density-consistent Debye cutoff instead of the frozen one moves it
**≤ 0.42%**.

**This also refutes `chapter2.tex:924`,** which justifies the frozen cutoff with
"F varies by about 30% across the grid". Evaluating the module's own functions,
q spans **×3.7 for n=2 and ×50 for n=8**. The documented justification is wrong
by up to a factor 50 and the result is insensitive anyway. Report it as the
CLAUDE.md pattern: error found, traced, sized, headline unaffected.

### G7. Balmer opacity bounded ✅ Verified — closes an open item cheaply

τ(Hα, 10 cm) = 6.3×10⁻⁷ at [0,0], 1.0×10⁻³ at [0,4], **0.263 at the single
worst cell [0,7]**, where the escape factor is still ≥ 0.85. The observed ratio
would move ≲ 10% in that one cell and nowhere else.

### G8. The A-weighted line ratio changes no defended number, and the thesis's 2.2 percent is wrong ✅ Verified (11 Sep 2026)

**Script:** `src/validation/verify_weighted_census.py`. **Artifact:**
`validation/weighted_census/` (784 rows, summary CSV, text log; sha256 of
L_grid, S_grid, state_index and radiative_rates.csv in the header).
Re-executes the `verify_divertor_map.py` construction and contracts the same
`n0`, `n1`, `n_old`, `n_new` with the Einstein-A weights from
`Balmer_transient_ratio.load_radiative_weights` (Hα: 3s, 3p, 3d → n=2;
Hβ: 4s, 4p, 4d → n=2; 4f carries zero). Shell half reproduces
`divertor_map.csv` bitwise on `eps_plateau`, `tau_QSS`, `f3`, `f4`, `eps_step`.

**Predictions written before the run, and outcomes.**
- P3, that eps_line/eps_shell runs 0.978 to 0.9999 as chapter 3:923 and
  chapter 4:771 state: **refuted.** It runs **0.9482 to 0.9999** (all 784 rows,
  identical over the 680 window rows), minimum at heat [48,0], the step
  9.54 → 10.0 eV at ne = 1e12. The 0.978 is reproduced only by restricting to
  Te < 2 eV, or by substituting the Hβ denominator alone (0.9810 to 1.0000),
  which is what `thesis_ready.md` A6 literally describes. Chapter 4's
  single-point [0,4] pair (0.386683 against 0.386903) reproduces exactly.
- P4, that the census moves by at most a few pairs: **held.** 100 µs:
  45 → 44 of 448 (cool [22,3] leaves, 0.1001 → 0.0999); worst 0.1748 → 0.1746
  at heat [15,3], same point. 250 µs: 33 → 33. 506 µs: 24 → 24, 0.1533 → 0.1531.
  750 µs: 20 → 19. Window scope: 202 → 200 of 680. Dense: 0 → 0 of 108.
- Benchmark heat [23,5]: 0.063612 → 0.063571 (ratio 0.99935).
- `ratio_plateau < 1` at every one of 784 rows: the shell census is
  conservative for the line observable.

**Skeptic pass (subagent, 11 Sep 2026).** Numbers reproduced at 60 digits in
mpmath and by a from-scratch label-based rebuild of the weights that shares no
code with the script (ratio_plateau to 1.2e-13 over 784 rows). The
`emissivity_generalisation` artifact independently puts the Δ disagreement at
5.4 percent at the same corner. Two of the script's printed checks (the
factorisation identity and the photon/energy invariance) are algebraic
identities of zero severity and are now labelled as such. The
`eps_step` ratio range (0.04 to 1.36) is a zero-crossing artifact, now
captioned.

**Mechanism.** Not the 4f darkness alone. Substituting one side at a time at
the corner: 4f removal 0.971, Hβ denominator 0.981, **Hα numerator 0.967**,
full 0.948. The ℓ-distribution is channel-dependent (4F fraction of n=4 is
0.432 ground-fed against 0.442 recombination-fed at the corner); A6's
0.4361 to 0.4375 is the total-QSS 4F fraction, which is not the driver.

**Sensitivity and caveats.** The 5.2 percent scales as 1/(ℓ-mixing rate):
10.0 percent at half the PSM20 rate, 2.6 percent at double, 1.1 percent at
five times. The sign survives the full range. The benchmark moves 0.13 to
0.03 percent over the same range. The extremum is on the grid boundary with
both trends monotone. The census insensitivity is a fact about where the
census points sit (ratio 0.992 to 0.999 there), not about the observable;
5 of 448 lower bounds lie within 0.002 of the 0.10 threshold.

**Thesis consequences, not yet applied (author's call):**
1. `chapter3.tex:921-925`, `chapter4.tex:769-777`: 0.978 / 2.2 percent →
   0.948 / 5.2 percent, cite `validation/weighted_census/`.
2. `chapter3.tex:925` "the residue is the 4f level": replace with the
   channel-dependent ℓ-distribution statement.
3. `chapter3.tex:922` "agrees to 0.014 percent at the benchmark": not
   reproduced by any quantity in this script (eps_plateau 0.065 percent,
   eps_step 0.0009 percent).
4. Whether chapter 5's headline numbers switch to the line observable
   (44 of 448, worst 0.1746) or stay on the shell with this bound stated.

## H. CORRECTIONS FORCED BY THIS SESSION

| # | Item | Correction |
|---|---|---|
| H1 | A1 τ_QSS floor | 1.18 µs → **75.4 ns**. The old value is the minimum over the 346-point M>900 subset quoted as a 400-point range; 46 of 400 points lie below it |
| H2 | A10 table | Column j=4 (5.18×10¹³) restored; it is the row maximum at the three coldest rows and is where A11's worst case lives |
| H3 | A10 detachment | Retracted. 5.2× below the detached band, and the location moves a decade per decade in assumed n(1s) |
| H4 | A11 scope | Restated inside Te ≥ 2 eV: 45 of 448, worst 17.5%, zero at citable divertor densities |
| H5 | W2 colocation | "M is largest where the error is worst" is false; the maxima are 52× apart |
| H6 | A5 | **Demoted ✅ → ▶️.** `derivation_04:375-381` says the May run was never independently re-run, while A5 called it "the strongest single check" |
| H7 | B8 | **Resolved.** The S_grid cm³/s mislabel is gone |
| H8 | Controlled correlations | findings_09 §3.1's +0.33 and −0.16 do not reproduce. Measured +0.274 (all pairs), +0.326 (heating only, which is where +0.33 came from), and **−0.442** quadratic. The **sign flip is robust across eight bases**; the magnitude is basis-dependent and must carry its basis |
| H9 | ELM count | **202, not 201.** findings_09 W5 asserted "201, not 202" while decomposing it as 105 heat + 97 cool in the same sentence. Recounted: 105 + 97 = 202 over 113 distinct points |
| H10 | ε at [23,3] | **0.12166.** findings_10 A10's 0.122 is right; its own §7.1 figure of 0.111 is wrong |
| H11 | cond(L_EE) | `chapter3.tex:719-721`'s 1.48×10³ to 1.74×10⁵ is **correct**, verified over all 400 points (1.4821e3 at [0,0], 1.7436e5 at [33,7], median 2.1044e4). **Correction, 10 Sep:** an earlier version of this row said it "now has a script". It did not. The value had been computed once in an ad-hoc shell session, which is the same provenance defect the row was recording. `verify_operator_conditioning.py` now produces it, together with μ(L), the spectral abscissa, the Levy–Desplanques premise and both participation-ratio norms, stamped to `validation/operator_conditioning/` |
| H12 | Claim 5.3 falsifier | **The 680/680 is an identity, not a test.** With $\ln A=\ln(R_{\rm pe}/R_q)$ and $\ln D=\ln(R_{\rm old}/R_{\rm pe})$, $\varepsilon_{\rm plateau}>\varepsilon_{\rm step}\iff D/A\in(-2,0)$. Measured over the 680: $D/A$ min $-1.0752$, median $-0.9142$, max $-0.3105$, inside $(-2,0)$ at 680/680 and never closer than a factor $1.86$ to the boundary. It is forced by three facts each holding 680/680 with no exception: heating depletes the ground ($\ln x<0$, 338/338), $f_3>f_4$ everywhere, and $\mathrm{sign}(\ln D)=-\mathrm{sign}(\ln A)$ (680/680). `claim_evidence_table.md:745` names the falsifier "one counterexample would refute it; none in 680" and marks it SUPPORTED; no counterexample can exist, so the named falsifier is not one. This is §J's missing-prediction defect applied to claim 5.3. **Replacement statement, which does have a falsifier:** the two supply channels cancel to a median $|1+D/A|=0.0858$ in the CRE $T_e$-response of $n_3/n_4$ — an 8× cancellation, measurable and refutable. Verified 10 Sep 2026 from `L_grid.npy` (`2d92b58e…`) / `S_grid.npy` (`7822f536…`) via `cr_context.py`; the same run reproduces the stored 680 pairs, 680/680, median amplification 11.653, min 1.447. |
| H13 | `eq:gain_def` defined G as a derivative | **G is a secant, and the thesis said derivative.** `verify_reservoir_gain.py` computes `G = ln x / dlnTe`, a secant over the step taken, and `tab:gain` lists three values of G at one grid point (k = 1, 2, 4) — which no derivative can do. The boxed `eps = |exp(Sbar*G*dlnTe)-1|` was therefore first-order, not exact as printed. Definition changed to the secant with the partial derivative as its small-step limit; the boxed form is now exact. **The 5–6% step-spread is fully explained:** |G| is linear in k, and fitting through k = 1, 4 predicts the withheld k = 2 to median 0.057%, worst 0.066%, at **314 of 314** heating points (`verify_secant_linearity.py`, reducing the stamped `validation/reservoir_gain/reservoir_gain.csv`, stamped to `validation/secant_linearity/`). Tangent intercepts 5.835, 7.910, 14.729 at benchmark, crest, cold edge. Corrected 11 Sep 2026. |
| H14 | Fujimoto status not propagated | Backlog I.1 closed this 10 Sep and recorded "Chapters 4 and 6 rewritten", but **four sites kept the withdrawn self-indictment**: the §6.7 section title ("and has not explained"), §6.10.1 ("unresolved … for which no explanation has been found", in the same chapter as §6.7.4 "The source resolves it"), the Ch.4 ladder row ("open / target unverified") and its caption, and the `sec:further_work` item ("Until that is read, the benchmark adjudicates nothing"). All four brought into line 11 Sep 2026. |
| H15 | `sec:further_work` carried a retracted result | It listed the joint (Te,ne) step map as still to do and repeated the **benchmark spot check** (bound 0.0117 → 0.0030, "a factor of 4" in the model's favour) that §5.8.3 explicitly retracts as "not representative" and of the **wrong sign** — the worst case rises 26%, to 0.2206, under a +1 density step. Paragraph replaced by the one item that genuinely remains (a density grid fine enough to name an ELM). The 22.1% joint-step worst case now also appears in the abstract and §7.2, which had quoted only the fixed-density 17.5%. Corrected 11 Sep 2026. |

## I. OPEN, RANKED BY DAMAGE

1. ~~**The Fujimoto benchmark target is unverified.**~~ **CLOSED 10 Sep 2026.**
   The printed table was read. The question was "is Table 4.1(b) bundled-n or
   ℓ-resolved?" and the answer is **bundled**: the columns are p = 2, 3, 4, 5,
   7, 10, 15, principal quantum numbers, with no ℓ label anywhere. The
   transcription is exact and the row labels are log₁₀(n_e/m⁻³), which the
   table establishes internally. The one-decade offset speculation is
   withdrawn. The r₁ deficit measures this model's ℓ-closure against a bundled
   tabulation, not a model defect: removing proton ℓ-mixing moves r₁(2) by
   118× at that density. **Chapters 4 and 6 rewritten to withdraw the
   self-indictment and report the r₀ agreement as the pass it is.** See
   `findings_10` ADDENDUM I. **Follow-up 11 Sep 2026:** the rewrite reached the
   section *bodies* only. Four sites still carried the withdrawn self-indictment
   and are now fixed — see H14. A closed backlog entry is not evidence that the
   claim was propagated; grep the titles, summaries, ladder rows and further-work
   lists too.
2. **The conservation gate is tautological.** Three injected faults (a tenfold rate error, all de-excitation deleted, the whole input array transposed) left the residual unchanged. It detects only an A/γ inconsistency. `chapter3.tex:222-225` claims otherwise.
3. **Chapters 3 and 6 contradict each other about the same operator.** Chapter 3 says the split and the tanh bound hold for a trapped matrix; Chapter 6 correctly says Θ_P depends on n(1s).
4. **Gate D fails at 100% of points**, ACD half unimplemented.
5. **The molecular bound is constructible and Chapter 6 says it is not.** Factor 3 to 5 on cold-corner ε_plateau, from references it already cites.
6. **Transport uses the wrong timescale, in the thesis's disfavour.** Free-streaming gives 6 to 10 µs; with CX trapping the neutrals it is **72 µs** against a 26 µs threshold. The result survives by 2.8×.
7. **Three-body recombination at 10¹².** `chapter3.tex:190-193` says "under 10% of the feed into any level"; 30 of 36 levels exceed 10%.
8. **261 em dashes** across Chapters 1, 2, 3 and 6, against a house rule of zero.
9. `verify_bundling_psm20.py` synthesises grids silently when files are missing. Fix the fallback before running it.
10. **Validation ladder rung 9** contains the one honest FAIL (Gate D); rung 6 is now closed by C6 above.

## J. THE STRUCTURAL FINDING

Of thirteen major results audited against the chain physical picture →
mathematics → prediction → numerical test, twelve have all four. **R6, the
magnitude, is the only one with no prediction step.** Nothing was written down
saying how large the error should be or what would refute it, which is exactly
why it turned out to be linear in an arbitrary grid step. A number produced with
no prior expectation can only be rationalised, never falsified.

The repair is G1: headline dε/d ln Te = |S̄·G|, both factors independently
measurable. That converts the weakest chain into one of the strongest, and it is
what `pivot_decision.md` implements.

---

## SESSION REGISTER — 11 September 2026 (report-only, verification agent)

### K1. Reviewer question on §5.9: does M(ne) rank ε_plateau(ne) at fixed T_e? 💻 → ▶️

**The question.** A reviewer of Chapter 5 §5.9 ("Timescale separation does not
predict it") called the polynomial partial-correlation machinery
(Table `tab:partial_sweep`, residualising log M and log ε_plateau on a
quadratic/cubic basis in log T_e, log n_e, reported negative in all twelve
scope/basis combinations, median −0.53) over-engineered, and asked the more
direct question that removes the T_e confound by construction instead of by
regression: **at each fixed T_e row, does M(n_e) correctly rank
ε_plateau(n_e) across the 8-point density sweep?** Nobody had run this before
this session.

**Script.** `src/validation/verify_m_rank_test.py` (new, written for this
check). Run: `/opt/anaconda3/envs/cr/bin/python src/validation/verify_m_rank_test.py --write`
from `/Users/phi/Desktop/non_markovian_cr`. Loads
`validation/divertor_map/divertor_map.csv` (784 rows, sha256
`5a20bbfcf05e258d9e4a9ad459cc0d27de7130ca2403d029914f1fd265d80490`), and
labels the CSV's `i`, `j` columns against `Te_grid_L.npy` (50 pts,
1.000–10.000 eV) / `ne_grid_L.npy` (8 pts, 1e12–1e15 cm⁻³) through
`cr_context.py`; verified `Te_grid[i]`, `ne_grid[j]` reproduce the CSV's own
`Te`, `ne` columns to 1e-9 relative before trusting any index label. Outputs:
`validation/m_rank_test/m_rank_test.csv` (536 detail rows, one per
direction×scope×quantity×i), `m_rank_test_summary.csv` (64 rows, all six
sub-tests), `m_rank_test.txt`, each headed with the script name, date,
interpreter path, and the four `#` provenance lines copied verbatim from
`divertor_map.csv`'s own header (L_grid/S_grid/state_index sha256, fractional
step 0.05, threshold 0.10).

**Prediction, written in the script's docstring before computing anything.**
If Chapter 5's negative partial correlation is real physics and not an
artifact of the polynomial control basis, the same sign should appear in the
much more elementary within-row rank test: a **majority** of fixed-T_e rows
should have ρ_i(M, ε_plateau) < 0 across n_e. Refutation: a majority with
ρ_i > 0.

**The refutation appeared.** Pooled (heat+cool merged at common T_e index),
scope = all 50 T_e rows, quantity = ε_plateau: only **36.0%** of rows have
ρ_i < 0 (18/50), median ρ_i = **+0.228**. **PREDICTION REFUTED** by the
script's own stated criterion.

**Per-direction/scope breakdown (part 1 of the six sub-tests), n_rows always
= n_rows qualified (every row had ≥4 window_ok columns; the ≥4-column
minimum never actually binds except in scope (c) below):**

| direction | scope | quantity | n rows | median ρ_i | frac ρ_i<0 | frac ρ_i>0.5 | frac ρ_i<−0.5 |
|---|---|---|---|---|---|---|---|
| heat | all | ε_plateau | 49 | +0.191 | 0.469 | 0.000 | 0.184 |
| heat | Te≥2 | ε_plateau | 34 | −0.257 | 0.529 | 0.000 | 0.265 |
| cool | all | ε_plateau | 49 | +0.476 | 0.163 | 0.102 | 0.163 |
| cool | Te≥2 | ε_plateau | 35 | +0.393 | 0.229 | 0.143 | 0.229 |
| pooled | all | ε_plateau | 50 | +0.228 | 0.360 | 0.000 | 0.160 |
| pooled | Te≥2 | ε_plateau | 35 | +0.071 | 0.486 | 0.000 | 0.229 |

**Heat/Te≥2 is the one cell that narrowly supports the prediction (52.9%
negative) — and it does not survive inspection.** Printing ρ_i row by row
(saved in `m_rank_test.csv`) shows the negative rows are concentrated
entirely at the hot edge, T_e ≥ 4.29 eV, exactly where the plateau-window gate
(A11/A11-sensitivity, §K above line ~675) starts stripping columns:
n_cols = 8 for T_e = 2.02–3.09 eV (ρ_i = +0.33 to +0.48, all positive),
n_cols = 7 for 3.24–4.09 eV (ρ_i = +0.214, still positive), then n_cols drops
to 6, 5, 4 for T_e = 4.29–9.54 eV where ρ_i = −0.257, −0.700, and exactly
−1.000 respectively. At n_cols = 4 (the script's own floor) a single
adjacent-pair swap forces ρ_i to ±1 — three of the nine "negative" heat/Te≥2
rows are driven by 4-point estimates that can only take the values
{−1, −0.4, +0.4, +1, ...}. **None of the individual row p-values are
significant** (p = 0.12–0.96 for every heat/Te≥2 row printed); the per-row
test has essentially no power at n=4–8, and the one scope where the sign
looks "right" is carried by the sparsest, least reliable rows at the grid
edge, not by the well-populated interior (T_e = 2.02–4.09 eV, n_cols=7–8,
ρ_i uniformly **positive**).

**Part 2 — lo_ELM_crash (the 100 µs time-averaged quantity the census
actually counts) is the sharpest result.** Row-wise ρ_i(M, lo_ELM_crash) is
POSITIVE almost everywhere: pooled/all median ρ_i = **+0.740**,
frac(ρ_i<0) = 0.020; pooled/Te≥2 median ρ_i = **+0.797**, frac(ρ_i<0) = 0.000
(35/35 rows positive, 35/35 with ρ_i>0.5). Every direction/scope cell for
lo_ELM_crash has frac(ρ_i<0) ≤ 0.082. This is the **opposite sign** from the
Chapter 5 headline, for the exact quantity §5.9's census is built on.
**Caveat (mechanism, not "fixed"):** lo_ELM_crash = ε_plateau · f(τ_QSS/τ_d)
with f(x)=x(1−e^{−1/x}) monotonically increasing in τ_QSS, and
τ_QSS = M·τ_relax; at fixed T_e the τ_QSS factor is close to proportional to
M (τ_relax varies far less across n_e at fixed T_e than across the whole
grid — A5 records a 45× range grid-wide). So part of lo_ELM_crash's strong
positive row-wise correlation with M is close to definitional, not purely an
independent physics finding about the "error". Reported, not adjusted.

**Part 3 — argmax/argmin agreement.** argmax_j M never equals argmax_j
ε_plateau, in **any** direction or scope (0/49, 0/34, 0/49, 0/35, 0/50, 0/35
— exactly 0.000 every time). This is structural, not a coincidence: M(n_e)
is monotonically decreasing in n_e at every T_e row examined (argmax always
at j=0, the lowest density), while ε_plateau(n_e) is unimodal, peaking at an
interior j (≈3–4); a monotone function and a unimodal function cannot share
an argmax except by construction. argmin agreement is partial and direction-
dependent: heat 0.510 (all)/0.471 (Te≥2), cool 0.837/0.771, pooled
0.820/0.771 — because M's argmin is always at j=7 (highest density) and
ε_plateau is often, but not always, also smallest there.

**Part 4 — the reverse question is close to deterministic.** At FIXED n_e,
across T_e rows with T_e≥2 eV, ρ_j(M, ε_plateau) = **+1.0000** for every one
of the 8 density columns, in BOTH the heat direction (p ≈ 0–6.7e-64) and the
cool direction (p ≈ 0–3.4e-197). Pooled (heat+cool merged per column,
n=22–69) softens only slightly, ρ_j = +0.749 to +0.999, still overwhelmingly
positive and significant. This directly confirms Chapter 5's own diagnosis:
the raw, uncontrolled correlation (+0.76 over 680 pairs, reproduced here
independently as pearson(log M, log ε_plateau) = **+0.7574** over the same
680 window_ok rows — a standalone sanity check outside the six numbered
sub-tests) is almost entirely the shared temperature trend, since M and
ε_plateau are essentially perfectly co-monotonic in T_e at every fixed n_e.

**Part 5 — extrema restricted to T_e≥2 eV & window_ok (the range Chapter 5
can defend; the current text quotes [0,0]/[0,4], both T_e<2 eV).** In every
direction, the largest-M point and the largest-ε_plateau point sit on the
**same T_e row**, [15,·], T_e=2.0236 eV — structurally consistent with
Chapter 5's own [0,0]/[0,4] comparison (also same row, i=0) — but materially
smaller in magnitude:
- heat: max M at [15,0] (n_e=1.000e12, M=1.387e6, ε=0.0624); max ε at [15,3]
  (n_e=1.931e13, ε=0.1807, M=2.015e5). n_e ratio 0.0518 (≈**19.3×**, not the
  52× quoted for the whole-grid [0,0]/[0,4] pair); M ratio ≈**6.9×** (not the
  factor-46 quoted whole-grid); ε at the M-maximum is 0.0624 vs 0.1807 at the
  worst point (factor ≈**2.9**, not 3.2 as separately reported for the
  whole-grid pair, and the absolute worst error in this restricted range is
  **18.1%, not 38.7%**).
- cool: max M at [15,0] (M=2.633e6, ε=0.0735); max ε at [15,3]
  (ε=0.1548, M=3.690e5); same n_e ratio 0.0518; M ratio ≈7.1×.
- pooled: max M at cool[15,0], max ε at heat[15,3]; same n_e ratio 0.0518.
**If Chapter 5 restricts this comparison to the T_e≥2 eV range it already
defends elsewhere in the same section, the extremum-separation story shrinks
by roughly a factor of 2.5–3 in every metric (density ratio, M ratio, and
worst-case ε) relative to the whole-grid numbers currently quoted.**

**Part 6 — plain (unresidualised) correlations, T_e≥2 eV, window_ok, for
comparison with the row-wise numbers and with
`validation/partial_correlation/partial_correlation_sweep.csv`'s own
linear-basis partial (+0.416720, n=448, exact match on n confirms the same
row selection):** pearson(log M, log ε_plateau): heat +0.574 (n=218), cool
+0.687 (n=230), pooled +0.625 (n=448); spearman: heat +0.596, cool +0.705,
pooled +0.645. **All positive**, same sign as the raw uncontrolled
correlation and the linear-basis partial, opposite sign from the
quadratic/cubic-controlled partial (−0.53) that is the section's headline.

**What would have refuted the prediction, and whether it appeared.** Stated
in advance: a majority of rows with ρ_i>0 for ε_plateau. It appeared (64.0%
of pooled/all rows; every scope/direction cell except the small, edge-driven
heat/Te≥2 case). For the operationally relevant quantity (lo_ELM_crash) the
refutation is not narrow — it is close to unanimous (98% of rows positive).

**What failed/warned/was killed.** Nothing. The script ran clean, wrote all
three output files, no exceptions, no timeouts. The one internal consistency
check the script performs (CSV `Te`,`ne` columns vs `Te_grid[i]`,
`ne_grid[j]` to 1e-9 relative) passed. The 0.000 argmax-match rate looked at
first like a bug; traced to the structural mismatch between a monotone
function (M) and a unimodal one (ε_plateau) — not a defect in the script.

**One-sentence conclusion the data support.** At fixed T_e, M does not
reliably rank ε_plateau in reverse — the row-wise test is close to a coin
flip and trends positive in the well-populated interior of the grid, and for
the quantity the census actually counts (the 100 µs time-averaged bound) M
ranks it in the SAME direction almost everywhere, so the section's negative
partial-correlation headline is a property of the quadratic/cubic polynomial
control basis specifically, not something visible in the raw data, the
linear partial, the reverse (fixed-n_e) test, or this simpler no-basis-choice
rank test that the reviewer asked for instead.

**Not ✅.** This is a report-only run: one script, one pass, no sensitivity
sweep of the script's own MIN_COLS=4 threshold or of the plateau-window k
(inherited from A11-sensitivity, not re-tested here), and the mechanism
caveat on lo_ELM_crash (built from τ_QSS, which is nearly proportional to M
at fixed T_e) has not been quantified. Graduation: **💻 → ▶️**. Whether this
overturns or merely qualifies §5.9's headline sign claim is an editorial
decision for the chapter, not this script's to make.

Artifacts: `src/validation/verify_m_rank_test.py`,
`validation/m_rank_test/m_rank_test.csv`,
`validation/m_rank_test/m_rank_test_summary.csv`,
`validation/m_rank_test/m_rank_test.txt`.

**SUPERSEDED IN PART by the skeptic pass of 11 Sep 2026 (see K2).** The
arithmetic above reproduces independently to 4 to 6 digits. The conclusion in
the paragraph above it does not stand: the row-wise test controls for T_e
only and gives +0.4 to +0.6; the partial controls for T_e *and* n_e and gives
−0.53 (quadratic) to −0.94 (saturated two-way fixed effects, heating). The two
are different conditionings and the row-wise test does not bear on the
partial. The sign flip is produced by removing the density trend, not by the
polynomial. Also: the headline cell (+0.228, 36 percent negative) is the
pooled all-T_e scope; at the chapter's own T_e ≥ 2 eV scope it is +0.071 with
49 percent of rows negative, and heating alone is −0.257. The lo_ELM_crash
result is definitional (ρ(M, f(τ_slow/τ_d)) = +1 exactly in 34 of 50 rows; the
null is +1, not 0, and the observed +0.74 to +0.80 sits *below* it). The
"argmax never coincides" and "ρ_j ≈ +1 at fixed n_e" results are arithmetic
identities (argmax M is always j = 0; both quantities are monotone in T_e) with
no refuting outcome. The row-wise statistic is fragile to MIN_COLS (heating
T_e ≥ 2 median goes −0.257 → +0.214 → +0.476 at MIN_COLS = 4, 5, 7) and to
T_e decimation. What survives: at fixed T_e, M falls monotonically with n_e in
98 of 98 series while ε_plateau peaks at the crest in 95 of 98, so M cannot
rank the error across density by construction. Chapter 5 defects found in the
same pass: the largest M (1.73e9) is at cool [1,0], not [0,0]; "52-fold apart
in density" holds only for the coldest rows (19.3-fold at any T_e cut ≥ 1.5 eV);
"86.8" is the minimum over 784 rows, 902 over the 680 window rows; "at fixed
(T_e, n_e)" has no referent; and the `make_ch5_figures.py` guard near line 887
raises unless the quadratic partial is negative, which is a results lock, not a
test. Graduation stays ▶️.

### K2. Stamping the skeptic pass: is §5.9's negative partial correlation a polynomial-basis artifact? 💡📐 (scratch) → 💻 → ▶️

**The question K1 left open.** K1 found the row-wise rank test does not
support §5.9's headline sign, and flagged (without resolving) that the
headline itself is "a property of the quadratic/cubic polynomial control
basis." A skeptic pass run earlier the same day, in scratch scripts only,
tested that specific claim directly by residualising on a fully saturated,
non-parametric two-way (T_e-row x n_e-column) fixed-effects control instead
of a polynomial, and found the negative partial survives — this session
turns that scratch result into a stamped artifact.

**Script.** `src/validation/verify_partial_fe.py` (new). Run:
`/opt/anaconda3/envs/cr/bin/python src/validation/verify_partial_fe.py --write`
from `/Users/phi/Desktop/non_markovian_cr`. **Loaded:**
`validation/divertor_map/divertor_map.csv` (784 rows, sha256
`5a20bbfcf05e258d9e4a9ad459cc0d27de7130ca2403d029914f1fd265d80490`), with
`Te_grid_L.npy`/`ne_grid_L.npy` via `cr_context.py` (never redefined); the
CSV's `Te`,`ne` columns were checked against `Te_grid[i]`,`ne_grid[j]` to
1e-9 relative before anything downstream trusted the index labels — passed.
The `±5%` step fraction was **parsed from `divertor_map.csv`'s own header**
("fractional step 0.05"), not hardcoded, then asserted to equal the task's
0.05. Outputs: `validation/partial_fe/partial_fe.csv` (352 detail rows: the
88-row main table over 8 controls × 2 scopes × 3 directions × 2 T_e-step
variants, plus every leave-one-out run), `partial_fe_summary.csv` (30 rows:
reference reproduction, raw Pearson/Spearman, permutation summaries, M
range), `partial_fe.txt`, each headed with script name, date, interpreter
path, and the **three sha256 lines copied verbatim** from
`divertor_map.csv`'s own header (L_grid/S_grid/state_index) plus the CSV's
own sha256 and the parsed step fraction. Deterministic: re-run with
`--write` reproduced `partial_fe_summary.csv` byte-for-byte except the
timestamp line. Runtime 1.6 s, no warnings, no exceptions, no failed
stages (`failures` list empty in every run).

**Every design was solved twice** (centred-and-scaled `numpy.linalg.lstsq`
and QR/triangular-solve) and the two coefficient vectors were required to
agree to 1e-8 before either was trusted — they did, in every one of the
main-table, leave-one-out, and permutation-baseline fits; the largest
lstsq-vs-QR discrepancy printed anywhere was `5.57e-11` (the 9-parameter
additive-quartic design, condition number `1.8e5`). Design condition
numbers ranged `1.3` (linear) to `1.8e5` (additive quartic); the two-way FE
design (41-57 parameters) had condition number `6-11`, better conditioned
than the polynomial bases despite far more parameters.

**Reference reproduction (target vs. observed, `Te≥2 eV`, `window_ok`,
pooled unless stated, pre-step T_e):**

| quantity | target | observed | \|diff\| |
|---|---|---|---|
| quadratic, pooled | −0.531 | **−0.531359** | 0.0004 |
| linear, pooled | +0.417 | **+0.416720** | 0.0003 |
| two-way FE, pooled | −0.617 | **−0.617409** | 0.0004 |
| two-way FE, heat | −0.939 | **−0.938964** | 0.0000 |
| two-way FE, cool | −0.770 | **−0.770033** | 0.0000 |
| T_e-row FE only, band [0.38, 0.59] | — | heat **+0.376690**, cool **+0.587553**, pooled **+0.474593** | heat **narrowly outside the band** (by 0.0033); cool and pooled inside |

The quadratic value reproduces `partial_correlation_sweep.csv`'s own stored
`-0.531359` to the digit, confirming the same 448-row selection and basis as
Chapter 5's own script (`make_ch5_figures.py:partial()`). All five numeric
targets reproduced to ≤0.0004 absolute; the one qualitative target (T_e-FE
band) reproduced in 2 of 3 directions and missed the heat direction by a
margin smaller than the band width itself (0.0033 vs. a 0.21-wide band) —
reported as a miss, not rounded into the band.

**The refuting observation did not appear.** Stated in advance: a positive
two-way FE partial in any direction, or a permutation p above 0.01. Neither
appeared:

- **Two-way FE is at least as negative as quadratic in all three
  directions** (heat −0.939 ≤ −0.590; cool −0.770 ≤ −0.620; pooled −0.617 ≤
  −0.531) — prediction 1 held everywhere.
- **T_e-row FE is positive in all three directions** (+0.377, +0.588,
  +0.475) — prediction 2 held everywhere, consistent with K1 Part 4's
  finding that M and eps_plateau are co-monotonic in T_e.
- **Permutation test** (shuffle ln eps_plateau within each T_e row, 2000
  draws, `numpy.random.default_rng(20260911)`, `Te≥2 & window_ok`): every
  one of the 6 direction×control cells gave `frac(null ≤ observed) = 0.00000`
  (0 of 2000 draws at or below the observed value), i.e. p < 1/2000, well
  under the 0.001 threshold. Null means clustered near zero (+0.036 to
  +0.044 for quadratic, −0.003 to +0.0004 for two-way FE) — the *un*controlled
  T_e-row shuffle produces an essentially zero partial, as expected, and the
  observed strongly negative values sit many standard deviations below that
  (null sd 0.057-0.079).

**Sensitivity, `Te≥2` scope, quadratic and two-way FE, both directions and
pooled — every leave-one-out value is in the detail CSV:**

- **Leave-one-ne-column-out (8 runs each):** sign never flips. Quadratic
  pooled ranges −0.410 to −0.699 across the 8 drops (observed −0.531); two-way
  FE pooled ranges −0.549 to −0.783 (observed −0.617). Heat and cool show the
  same pattern (quadratic heat: −0.380 to −0.688; two-way FE heat: −0.924 to
  −0.954). Dropping the benchmark-adjacent column (`j=5`, n_e≈1.39e14) gives
  the *weakest* quadratic partial in every direction (pooled −0.410, heat
  −0.380, cool −0.395) — flagged, not explained here; the two-way FE partial
  at the same drop is far less moved (pooled −0.549, heat −0.929, cool
  −0.742), consistent with the polynomial basis being more sensitive to
  which columns are present than the FE basis is.
- **Leave-one-T_e-row-out (34-35 runs each):** far tighter than the
  density-column leave-out. Quadratic pooled: −0.536 to −0.527 (observed
  −0.531); two-way FE pooled: −0.620 to −0.609 (observed −0.617). No sign
  changes, no outlier rows.

**a-variant (pre-step vs. post-step T_e) — reported, not resolved.** The
two-way FE and T_e-row FE controls are index-based (dummies on grid row/
column `i`,`j`) and are therefore identical under both variants by
construction. The **continuous** controls are not: at `Te≥2, window_ok,
pooled`, the quadratic partial is −0.531 under the pre-step T_e (the
chapter's own variable) but only **−0.335** under the post-step T_e, and at
the wider `window_ok` scope (no `Te≥2` cut) the quadratic partial is **−0.442
pre-step vs. −0.172 post-step** — same sign, roughly 2.5× smaller in
magnitude. Not every case even keeps sign this cleanly resolved from the
printed table alone; the full 88-row main table in `partial_fe.csv` carries
every combination for inspection. **This was not asked to be adjudicated,
only reported: which T_e (pre- or post-step) belongs in the control matters
to the size of the effect, not obviously to its sign in the cases checked
here, but has not been swept as thoroughly as the fixed-effects question.**

**Raw (unresidualised) Pearson/Spearman of (ln M, ln eps_plateau), `Te≥2 &
window_ok`:** pooled Pearson +0.6251, Spearman +0.6449 (n=448) — matches K1
Part 6's independently-computed +0.625/+0.645 to 3-4 digits, a second
independent confirmation of the same raw positive correlation the partial
analysis reverses. Heat +0.5743/+0.5958 (n=218), cool +0.6867/+0.7045
(n=230). At the wider `window_ok` scope (no T_e cut): pooled +0.7574/+0.7909
(n=680), matching `make_ch5_figures.py`'s stored `+0.76` and K1's
independently-recomputed `+0.7574`.

**M range:** window_ok (n=680): **9.021e2 to 1.729e9**. All rows (n=784):
**8.677e1 to 1.729e9**. These are **whole-grid** extrema (50 T_e × 8 n_e, 1-10
eV, 1e12-1e15 cm⁻³, both step directions) — not the ITER-benchmark-point M of
9982 quoted elsewhere, and not restricted to `Te≥2`. The ratio of max to min
window_ok M is ≈1.9×10^6, consistent with the earlier-recorded statement
that τ_relax alone varies 45× across the grid — M varies far more than
τ_relax alone because τ_QSS also varies strongly and non-uniformly.

**What would have refuted the claim, and whether it appeared.** Stated in
the script's docstring before any computation: a positive two-way FE partial
in any direction, or a permutation p above 0.01. **Neither appeared** — the
two-way FE partial was negative and *more* negative than the quadratic in
every direction (−0.617 to −0.939), and every permutation p was effectively
0 (0/2000).

**What failed, warned, or was killed.** Nothing. `failures` (the script's
own explicit collection of any stage that raised) was empty on every run;
every lstsq/QR agreement check passed at 1e-8; every design was full column
rank (checked before solving). The one near-miss is not a failure of the
script but a genuine numeric result: the T_e-row-FE-only partial for the
heat direction (+0.3767) falls 0.0033 short of the task's stated [0.38,0.59]
reference band — reported as an out-of-band miss above, not rounded in.

**One-sentence conclusion the data support.** Under the least
assumption-laden control available on this grid (saturated two-way fixed
effects, which removes any additive function of T_e and any additive
function of n_e, not just a quadratic one), the partial correlation between
ln M and ln eps_plateau is **negative and at least as strong** as under
Chapter 5's quadratic control, in all three directions, survives dropping
any single density column or any single T_e row, and is far outside a
2000-draw permutation null — so K1's open question is answered in the
opposite direction from what K1's own row-wise test suggested: the sign is
not a polynomial-basis artifact, though K1's row-wise-rank and reverse-
question findings (both real, both reproducible) still show that the
*mechanism* is dominated by the shared T_e trend and is not simply "M ranks
eps_plateau in reverse at fixed T_e."

**Not ✅.** This report-only run adds the sensitivity checks (leave-one-out,
permutation) that K1 lacked, but: (a) the pre-/post-step T_e sensitivity of
the continuous-basis controls (−0.531 vs. −0.335 at the same scope) is
reported, not resolved or swept as thoroughly as the FE question; (b) the
T_e-row-FE heat-direction reference band was missed by a small margin,
unexplained; (c) how this stamped result should be read against K1's
row-wise finding (same data, opposite-seeming implication) is an editorial/
physical-interpretation question for the chapter, not this script's to
settle. Graduation: **💡📐 (scratch) → 💻 → ▶️**.

Artifacts: `src/validation/verify_partial_fe.py`,
`validation/partial_fe/partial_fe.csv`,
`validation/partial_fe/partial_fe_summary.csv`,
`validation/partial_fe/partial_fe.txt`.

### K3. A finite ramp reaches the step's plateau: De(1 − e^(−1/De)), De = τ_slow/t_ramp ▶️ Run (11 Sep 2026)

**Script:** `src/validation/verify_ramp_plateau.py`. **Artifact:**
`validation/ramp_plateau/` (28 rows, sha256 of L_grid, S_grid, state_index,
radiative_rates). Full 43-state integration (Radau, rtol 1e-10, hold segments
ending exactly at the readout times, expm cross-check on every ramp case to
better than 3e-8 on the state) through L(t) = (1 − f)L⁻ + fL⁺, f = min(t/t_ramp, 1),
one grid interval heating, at [23,5] and [15,3].

**Result.** Plateau column eps(t_ramp + 30 τ_relax)/eps_step(30 τ_relax) at
[15,3]: 0.9995, 0.9954, 0.9555, 0.6469 for t_ramp/τ_slow = 1e-3, 1e-2, 1e-1, 1,
against the single-pole ramp response 0.9995, 0.9950, 0.9516, 0.6321. Accurate
to 0.5 percent for De ≥ 10; at De = 1 the formula is 2.3 percent low and the
measured value moves a further 1.7 percent between linear and log-linear
interpolation of the operator (0.6469 → 0.6576), so it is 0.65 ± 0.01. Ground
reservoir drift during the ramp is (1/2 − 1/(6De))(t_ramp/τ_slow) of the total
excursion, reproduced to 0.15 percent at [15,3]. The benchmark's De = 1000 ramp
is only 8.2 τ_relax long and is outside the law's domain. Convergence (rtol
1e-8 vs 1e-10): plateau ratio 1.6e-9, eps_max 4.1e-7.

**By-product.** At the benchmark the pure step overshoots: n₃/n₄ exceeds R_PE
by 0.40 percent at 1.16 τ_relax (2.6 ns), carrying ε_CRE to 0.0678 against
ε_plat = 0.0636 (6.6 percent), settled by 5 τ_relax. It is a property of the
ratio (n = 4 relaxes faster than n = 3; relative deviations cross at 0.72 and
4.66 τ_relax), not of the state: the deviation norm over the excited block
decays monotonically at all 784 pairs (worst growth factor 0.998), so it is
not transient non-normal growth. It is in both the shell and the line ratio
(1.0661 / 1.0661), observable-dependent (n = 3 population 1.020, n₃/n₅ 1.132,
n₄/n₅ 2.92), present at 456 of 784 pairs (median excess 1.35, max 5.91 at
[49,7] cool, largest where ε_plat is smallest), absent at [0,4], and it changes
the 100 µs average by −4e-5 relative at the benchmark. No census, estimate or
map moves.

**Skeptic pass (11 Sep 2026):** reproduced with expm alone to six digits;
found and fixed a dense-output readout error in the fifth digit; struck the
word "non-normal". Caveats: two points, heating only, one grid interval,
linear-in-time operator interpolation; a real ELM's factor-of-several excursion
is not tested.

### K4. Every plateau cell propagated: the bridge, the estimate and the exposure observable, grid-wide ✅ Verified (11 Sep 2026)

**Script:** `src/validation/verify_trajectory_census.py`. **Artifact:**
`validation/trajectory_census/` (784 rows, 400 samples/decade, sha256 of
L_grid, S_grid, state_index, radiative_rates). Exact post-step solution
n(t) = n_new + exp(L⁺t)(n_old − n_new) by eigen-propagation, gated at 12 times
per row against `scipy.linalg.expm` on the state (tolerance max(1e-8,
ε_mach‖L‖₂t), the intrinsic float64 limit, established against a 30-digit
mpmath exponential) and on the observable; exposure integrals by the augmented
matrix exponential, checked against direct quadrature to 1e-9. Convergence:
true averages change by ≤ 8e-5 between 100 and 400 samples/decade; the [0,4]
ratio by 4e-9. Closes Round 2 items 2, 4 and 5.

**A. Bridge, grid-wide (Round 2 item 2).** Over the k = 30 window at all 680
window pairs, max eps_track = |R(t)/R_QSS⁺(u(t)) − 1| ≤ 4.4e-5, median 1.1e-6,
0 pairs above the 2e-3 tolerance; sustained tracking within 4.9 τ_relax
(median 2.9); max |R(t)/R_PE − 1| median 0.0022, max 0.0154 at cool [1,3].
max_W eps_track · M has median 0.06 (correlation with 1/M +0.945): the
singular-perturbation O(1/M) departure of the slow eigenvector from the QSS
manifold, resolved three to five orders above the 1.5e-10 numerical floor.
Benchmark and [15,3]: t_track 1.948 and 3.734 τ_relax (bridge script 1.94,
3.72); flatness max|ε_CRE/ε_plat − 1| on the k = 30 window 0.0323 and 0.0301
(chapter 5's 3.33 % and 1.28 % were on the bridge script's measured plateau,
a different interval at [15,3]).

**B. The single-slow-mode estimate against the true average (item 4).** Over
the 448 pairs above 2 eV, true/estimate runs 0.9791 to 1.0222 at 100 µs
(median 0.9998, below 1 at 226) and 0.9671 to 1.0377 at 506 µs. tab:lowerbound
reproduces with its 25 ns mesh bias removed: 0.99998, 1.0027, 1.0065, 1.0072,
1.0010. The [0,4] value below unity is real: 2.19e-5, a rise deficit of
−5.21e-5 plus +3.0e-5 curvature, residual 1.5e-9. **Mechanism (skeptic pass):**
not the reviewer's weak-projection or multi-mode cancellation (slow-mode
projection 0.03 to 0.45, residual 1e-9), but that ε_CRE is a ratio whose
denominator relaxes on the same τ_slow: with δ the slow-mode amplitude in the
Hβ channel relative to CRE, true/estimate → (1+δ)ln(1+δ)/δ ≈ 1 + δ/2; heating
δ > 0 (estimate low), cooling δ < 0 (estimate high); δ runs −0.22 to +0.30 on
the warm pairs, asymptote 0.88 to 1.14. The τ_d → 0 limit of the estimate is
wrong (it gives ε_plat; the true average gives ε_step); valid only for
τ_d ≫ τ_relax.

**C. Census (items 4 and 5).** 100 µs, window_ok ∧ Te ≥ 2, > 0.10: estimate
shell 45, estimate line 44, true average 43 (shell and line), exposure ratio
∫j_α/∫j_β 43 (shell and line); worst 0.1753 at heat [15,3]. Leavers cool
[22,3] (0.09949) and cool [23,2] (0.09999, 1.5e-4 below threshold: quote 43,
44 within the numerics). 506 µs: 24/24/24/23/24/23. Exposure ratio against
time average pointwise: 0.990 to 1.011 (100 µs), 0.981 to 1.022 (506 µs).
Caveat: at 98 of 784 pairs the signed error changes sign during the rise, so
the exposure integral can cancel in principle; none is a census member and all
have ε_plat ≤ 0.064.

**Skeptic pass (11 Sep 2026):** numbers reproduced by Gauss–Legendre panels on
expm to 1e-9; found and corrected in place: the P3 arithmetic (dropped
division), the invalid 3.33 % comparison, the false docstring accuracy
sentence, the state-normalised gate (now also on the observable), and census
bookkeeping for the line rows. Definition-dependent: t_track (threshold,
sampling), max_W ε_PE (window), the count itself (45/44/43 by definition).
Inherits every upstream caveat of L_grid and the fixed-ion closure; the
12.7 % rate uncertainty of chapter 4 dwarfs the 1.5e-4 that decides 43 vs 44.

### K5. Round 2 review, status after this session

| item | state |
|---|---|
| 1 captions | closed (generator and .tex) |
| 2 grid-wide bridge | closed, K4 |
| 3 window factor k | closed, `validation/window_sweep/` (census 45 at k = 10, 20, 30, 50; denominators 547/496/448/364) |
| 4 estimate not a bound | closed, K4; chapter 4 and 5 text updated |
| 5 finite-exposure observable | closed, K4 |
| 6 100 µs naming | closed (text) |
| 7 joint step heading, secant gains | closed (text) |
| 8 §5.9 | closed on K2 (FE partial) and K1-as-amended; `make_ch5_figures.py` results lock at ~line 893 reported, not changed |
| 9 [UNVERIFIED] sweep | artifact exists; provenance header still missing on `partial_correlation_sweep.csv` (writer in `make_ch5_figures.py`) |
| 10 Table 5.1 | closed (same-point series from `reservoir_gain.csv`) |
| 11 ramp | closed, K3 |

### K6. Addenda after the maths audit of the day's edits (11 Sep 2026, evening)

A `math-auditor` pass over the ~400 changed thesis lines found one false
sentence (the deviation norm "decays monotonically": it never exceeds its
initial value over 10 τ_relax, then rises with the reservoir drift), a lag law
quoted beyond its tested range (the truncated 1/2 − 1/(6De) is 9 % off at
De = 1; the exact form is 1 − De(1 − e^(−1/De))), three stale "0.7 percent"
sites, three leftover "bound" wordings, and eight numbers that traced only to
script docstrings. All eight are now stamped rather than deleted:

| number | now in |
|---|---|
| growth factor of the excited-block deviation ≤ 0.9998 (784 pairs, L2, t ≤ 10 τ_relax) | `trajectory_census/` §A |
| early-transient effect on the 100 µs average, −4.0e-5 at the benchmark | `trajectory_census/` §A and column `early_effect_*` |
| convergence at 1600 samples/decade, ≤ 4.7e-6 at the five table points | `trajectory_census/` §B |
| M · max_W eps_track median 0.060, correlation +0.945 | `trajectory_census/` §A |
| Hα-numerator-only 0.9672, Hβ-denominator-only 0.9810 at the corner | `weighted_census/` P3 block, columns `ratio_numerator_only`, `ratio_denominator_only` |
| ℓ-mixing scan 0.9000 / 0.9482 / 0.9736 / 0.9893 at s = 0.5, 1, 2, 5 | `weighted_census/weighted_census_lmix.csv` |
| two-way FE R² 0.9980/0.9919 (heat), 0.9871/0.9829 (pooled); full quartic −0.524/−0.281/−0.335 | `partial_fe/` |
| log-linear interpolation rows, exact lag law | `ramp_plateau/` (`interp` column) |

Not stamped and therefore removed from the text: the −5.2e-5 / +3.0e-5
decomposition of the [0,4] deficit (the net 2.2e-5 is stamped).

### K7. Round 3 review (sections 5.10, 6.1 to 6.6) checked, 11 Sep 2026 evening 📐 Checked, not yet applied

Three read-only subagent passes (skeptic on the optical depth, cr-physicist on the
five physics arguments, evidence-auditor on the sentences and numbers). The
reviewer is right on every substantive point; the passes found eight further
defects. Nothing changed in the thesis yet.

**Confirmed, with the mechanism located.**
- √2 in the Lyman optical depth: chapter 5 (2150 to 2152) and chapter 6 (178 to
  179) carry σ₀ from a hand calculation with √(kT/m) (findings_10:250-260); the
  code (`escape_factor.py:167`, `verify_lyman_trapping.py:192`) and chapter 6's
  table use the correct √(2kT/m). The citation `(verify_lyman_trapping.py)` at
  chapter5:2153 is false: that script never emits an optical depth. The boundary
  1.13 to 1.98 eV and the 45/448 census already use the correct σ₀; nothing
  headline moves (the √2 is D = 5 → 7.07 cm, inside the factor-20 sweep).
- "Four to five orders of magnitude" (ch5:2155, ch6:175, 199): Θ_P = 1.2e-3 is
  2.9 orders one-shot; the self-consistent value at [0,4], D = 5 is 1.44e-2 (1.8
  orders); the self-consistent minimum anywhere is 7.8e-4 (3.1 orders). The
  descendant of the bogus τ = 7900 in findings_10.
- §6.3 molecular sentence is false as written: with independent reservoirs R is
  a ratio of affine forms in (u, v, ...); the tanh ceiling applies per reservoir
  with Δ(v) (0.46 → 0.25 at the crest as the molecular fraction goes 0 → 1, sign
  change in between) and bounds nothing about the v-term; H₂ at 2 eV is a 10 to
  100 µs reservoir, a second slow mode.
- "Factor 3 to 5": φ₄ = 0.30 is interpolated (Dγ is n = 5); the 0.201/0.054/
  0.053/0.019 sensitivities match no point in any artifact; done properly with a
  third channel the factor is 6 to 9 at the cold corner and the diluted
  sensitivity changes sign near φ₃ ≈ 0.75. Emissivity = shell fraction IS
  defensible here: intrashell ℓ-mixing exceeds radiative decay by ≥ 12 at every
  grid point (checked from L_grid).
- Transport "upper estimates" (ch6:593-595) does not survive: a recycling-fed
  reservoir with fixed source turns the 0.18 transient at [15,3] into a 0.18
  permanent offset; source ×2 gives 0.60. τ_esc < τ_slow means u relaxes fast to
  the recycling-set value, not to local CRE.
- Quasineutrality (ch5:2039) conflates two laws; the inferred Δn_e/n_e = 13.9 at
  [0,4] is not a closed-parcel number: the self-consistent closed-nuclei solve
  (L exactly linear in n_e, S exactly quadratic, so off-grid evaluation is
  exact) gives n_e ×2.05 and Δln n_g = −0.038. Above 2 eV the correction is
  ≤ 5.3e-3 in n_e and < 0.2 percent in ε_plat.
- "Coincide" / "not a choice" (ch5:2193, 2221) and "It is not detachment"
  (ch6:810): overstated as the reviewer says; the ridge comparison is local
  divertor n_e against an upstream separatrix band.

**Found beyond the review.**
1. ch6:178-179 quotes 0.0023 "at the benchmark point (\benchTe, \benchne)"; that
   is the n_e = 5.18e13 column with the wrong σ₀; the benchmark value is 0.0039.
2. ch5:1551-1552 and ch6:192-193: "105 of 202 have τ(5 cm) > 1, 26 above 100"
   reproduces only with the wrong σ₀ and full-path τ; correct σ₀ gives 100/23
   (full path) or 85/15 (the repo's own half-slab definition).
3. ch6:456 argues transport with "a deuterium atom" while the opacity uses m_H;
   the isotope changes σ₀ by exactly √2 (the deuterium value IS 114).
4. T_n = T_e never tested (`--t-at` flag exists, unused); T_n = 3 eV gives 46 per
   cm, a 42 percent swing larger than the √2.
5. ch6:286 "0.9999 at D = 1 cm" against the artifact's 0.99862.
6. No artifact exists for §6.3, §6.4 (the n_g scaling 2.28e12/1.68e13/2.48e14) or
   §6.6 (the ion-closure factors 41.4/15.4); all are markdown-only, though the
   §6.6 numbers reproduce from L_grid/S_grid directly.
7. ch6:843 "So the quantity is settled" sits above a `\todo` (859) asking to
   confirm the same quantity.
8. §6.5's "chiefly used to identify detachment" carries no citation.

**Reviewer's one miss:** there is no `[1,4]` versus `[0,4]` TODO; ch6:972-983
already resolves the labels.

### K8. The inversion, measured: what the table returns for T_e on the plateau ✅ Verified (16 Sep 2026)

**Why.** Chapter 1 asked for the error in an inversion; Chapter 5 measured the error in the observable. `eq:inversion_jacobian` (Ch. 3) defines the single-parameter inferred-temperature error at known n_e and says Chapter 5 uses "the exact finite-step form wherever a number is quoted". No script computed it. The Round-0 reviewer costed it as "another project"; it is one root-find per pair on the three solves the thesis already does.

**Prediction, written before the run** (from a chained-`signed_step` reconstruction of the same solve, 11 Sep): scope 448 → ~166 off-table, ~282 solvable, median |Δln Te| ≈ 0.445, 90th ≈ 0.90, max ≈ 1.42, amplification ≈ 10. **Refuter:** median amplification of order 1.

**Run.** `src/validation/verify_inversion_error.py` → `validation/inversion_error/{inversion_error.txt, .csv, _summary.csv}`; L_grid `2d92b58e…`, S_grid `7822f536…`; same step rule (nearest index to ±5 %), window rule (30/30) and three solves as `verify_plateau_gridmap.py`. Inversion = every Te* on the density column with R_cre(Te*) = R_pe, bracketed between nodes and interpolated; error = ln Te* − ln Te_new.

| check | result |
|---|---|
| reproduces `plateau_gridmap.csv` signed_plateau and signed_step at all 784 pairs | worst diff 0.000e+00 |
| sign identity on the two monotone columns, sign(err) = sign(d_pe)·sign(dlnR/dlnTe) | 0 bad of 180 |
| interpolation in (ln Te, ln R) vs (Te, R), ratio of clean medians | 0.9999 |
| refuter | median amplification 8.30; not refuted |

**Result, Te_old ≥ 2 eV, window_ok, one grid interval (+4.81 % / −4.59 %).** 448 = **166 off-table** (all cooling, all below the column's minimum) + **44 fold-crossed** (all heating) + **238 clean**. Clean: median |Δln Te*| 0.3899 → **47.7 % in Te**, 90th 0.674 (96 %), max 0.892 (144 %); amplification median **8.30**, 90th 14.3, max 19.0. Solvable incl. fold-crossed at nearest root: 0.4449 (56.0 %), amp 9.47. Heating clean 174: 0.439 (55 %); cooling 64: 0.321 (38 %). Monotone columns (j = 6, 7; n_e ≥ 3.73e14): 56 pairs, median 0.150 (16.2 %), max 0.177 (19.4 %), amp 3.19 to 3.8. Multi-root in scope 2 (both cooling; 9 over all 680 window pairs). Prediction reproduced to four digits.

**Derivation, then checked against the artifact.** Σ ≡ dlnR_cre/dlnTe = P + S̄G, P ≡ (∂lnR/∂lnTe)_u. As secants over the step from the same solves: S̄G = [ln R_pe − ln R_cre(new)]/Δ, Σ̄ = [ln R_cre(new) − ln R_cre(old)]/Δ, P̄ = Σ̄ − S̄G. Signs in scope: P̄ > 0 at 448/448 (measured, not derived from the rate structure), S̄G < 0 at 448/448 (= f₃ > f₄ times G < 0, both established), Σ̄ < 0 at 359/448. On the plateau only P acts; the table reads the offset back at Σ. **Theorem:** where Σ < 0, heating → colder and cooling → hotter: 172/172 and 64/64. Where Σ > 0 (46 heating pairs above the fold) a hotter root is predicted and the table does not climb enough before 10 eV at 44 of them: those are exactly the fold-crossed pairs. **Amplification = |S̄G|/|Σ̄| = 1/(residual cancellation):** medians |P̄| 1.068, |S̄G| 1.123, |Σ̄| 0.0907, |Σ̄|/|P̄| 0.0676; secant prediction vs exact over the 238 clean pairs: rank corr 0.925, pred/exact median 1.257. **The fold is Σ = 0, the cancellation completed**, so the columns with the fold are the columns with the amplification. Off-table cooling: |P̄|/|Σ̄| median 19.7 there vs 4.8 where a root exists. This is H12's |1 + D/A| cancellation (0.0858 over 680) seen from the inversion side; it was not in Chapter 5 before.

**Sensitivity to definitions.** Interpolation variable 0.01 %. Fold masking: 0.445 → 0.390, the clean value is quoted. Nearest vs far root: 2 pairs. Linearised Jacobian quotient (central difference at Te_new) vs exact: overstates the clean median by 17.7 %, heating by 48.8 %, understates cooling by 16.9 %; the exact form is quoted.

**Caveats, written.** Single-parameter, n_e known; the 2×2 Jacobian of `eq:inversion_2x2` is defined and not measured. Plateau values, not the ELM time averages of `sec:persistence`. Same frozen-reservoir conditional as ε_plateau; Ch. 6's transport selection applies verbatim. The table's 10 eV edge is a boundary condition on the 44 and 166 counts.

**Thesis home.** New §5.5.2 `sec:inversion_measured` with `eq:slope_two_channels`; §5.5.1 tail ("reported and not pursued" withdrawn); §5.11; §1.7 and §1.8 reworded from "(T_e, n_e) inversion" to "temperature inferred at known density"; Ch. 3 pointer at `eq:inversion_jacobian`; §7.2.1 retitled "Five things", fold paragraph retitled from "unrelated to any of this" to the same cancellation at its limit, new paragraph; abstract, one sentence.

**Addendum, 16 Sep 2026, after the reviewer's read of the draft §5.5.2.** Five corrections, all applied.
(i) The direction rule is a *local* result of Eq. `eq:inversion_local` (Σ < 0 ⇒ inferred temperature on the pre-step side; 172/172, 64/64) plus an *empirical* boundary statement for the 46 heating pairs with Σ(Te⁺) > 0: 44 cross the fold to a colder root before the 10 eV edge, 2 find a colder root inside the step interval. The draft's "a theorem on 236 and the table's upper edge on the rest" overclaimed the 46 and is withdrawn; a local slope predicts local direction, not root existence over a finite interval.
(ii) "Amplification is the inverse of a cancellation" was imprecise: it is |SG/Σ|, the ratio of the reservoir term to the residual slope, which becomes an inverse-cancellation measure only because |SG| varies far less than the near-zero denominator. Reworded in §5.5.2, §5.11, §7.2.
(iii) "Returns no temperature at all" → "has no solution within the table's 1 to 10 eV domain" at every count (abstract, §5.5.2, §5.11, §7.2). The edges are part of the result.
(iv) **Sign convention.** The K8 CSV column `SbarG` is ln(R_cre⁺/R_pe)/D, the reservoir secant with the *natural* sign of S = f₃ − f₄ > 0, hence negative. The thesis's S̄ (Eq. `Sbar_def`) integrates from u⁺ to u⁻ and is *negative*, so in thesis notation the column is −S̄Ḡ, and the draft's "S̄G < 0" was wrong in the thesis's own convention (S̄Ḡ > 0, both factors negative; only S̄Ḡ·Δln Te changes sign with direction, per HANDOFF §3). §5.5.2 now states the local theorem with natural-sign S, writes the secant as \overline{SG}, and reconciles the two in one sentence; the CSV header carries a SIGN NOTE; K9 check 3 verifies \overline{SG} = −S̄Ḡ to 4e-15.
(v) Local versus secant: `eq:slope_two_channels` had mixed a local Σ with a barred S̄. It is now the unbarred local chain rule; the finite-step identity Σ̄ = P̄ + \overline{SG} is a separate equation `eq:slope_secants`, exact by telescoping, and the reviewer's caution about multiplying separately averaged factors is met by construction: by `eq:exact_decomposition` and `eq:gain_def` the thesis *integrates* the reservoir term, so S̄Ḡ D = ln(R_pe/R_cre⁺) exactly (K9 check 3).
The scratch claim "both channels raise the ratio for the single reason f₃ > f₄" was never in the thesis and is refuted by K9. **P > 0 stays a measured property.**

### K9. Why P > 0: the operator-slope decomposition by feed channel ✅ Verified (16 Sep 2026)

**Why.** §5.5.2 rests on P > 0 and SG < 0 (both measured). The reviewer asked for a stamped script recording, pair by pair, why each feed channel has the sign it has, with the sufficient ratios, and for the additive split's residual to be measured rather than assumed. He also refuted the scratch conclusion from its own numbers.

**Algebra.** P = f₃A₃ + (1−f₃)C₃ − f₄A₄ − (1−f₄)C₄ exactly (differential form); P_a = f₃A₃ − f₄A₄ > 0 ⟺ (f₃/f₄)/(A₄/A₃) > 1 when A > 0; P_c = (1−f₃)C₃ − (1−f₄)C₄ > 0 ⟺ ((1−f₄)/(1−f₃))/(B₃/B₄) > 1 when B = −C > 0. Over the finite step the exact form is the log-mixture P̄D = ln[f₃e^{A₃D} + (1−f₃)e^{C₃D}] − ln[f₄e^{A₄D} + (1−f₄)e^{C₄D}] with secant A, C and f at (Te⁻, u⁻); the additive split is its first-order part.

**Prediction, written before the run** (scratch, 16 Sep): P_a > 0 at 448/448, P_c > 0 at ~426/448, A > 0 and C < 0 at 448/448. **Refuter:** P_a or P_c negative at a large fraction, or an additive residual comparable to the terms.

**Run.** `src/validation/verify_operator_slope_decomposition.py` → `validation/operator_slope_decomposition/`; L_grid `2d92b58e…`, S_grid `7822f536…`. Same step, window and solves as K8.

| check | result |
|---|---|
| two-channel superposition at all 400 nodes | 3.06e-14 |
| reproduces K8's P̄, \overline{SG}, Σ̄ | 5.5e-12 |
| Σ̄ − P̄ − \overline{SG} | 5.1e-15 |
| \overline{SG} + S̄Ḡ (thesis sign convention) | 4.0e-15 |
| log-mixture identity for P̄ | 1.1e-14 |
| sign(P_a) ⟺ ratio_a > 1, sign(P_c) ⟺ ratio_c > 1 | 0 bad of 784, both |

**Result, defended scope (448).** A₃, A₄ > 0 and C₃, C₄ < 0 at 448/448; A₄ > A₃ at 448/448; |C₄| > |C₃| at 340. f₃ > f₄ at 448 (medians 0.611, 0.190). **P_a > 0 at 448/448** (ratio_a median 2.87, min 1.23). **P_c > 0 at 426/448** (ratio_c median 2.00, min 0.955). **P_a + P_c > 0 at 448/448.** Medians A₃ +2.79, A₄ +2.90, C₃ −1.45, C₄ −1.60, P_a +0.62, P_c +0.48, P̄ +1.07. Additive residual |P̄ − P_a − P_c|/|P̄|: median 3.7 %, 90th 12.8 %, max 35.4 %. Cancellation |Σ̄|/|P̄| median 0.0676 (|P̄| 1.068, |\overline{SG}| 1.123, |Σ̄| 0.0907).

**The 22 exceptions** (P_c < 0, sum still positive): all at nₑ = 10¹⁵ cm⁻³, Te 2.02 to 3.39 eV, 10 heating and 12 cooling. There f₃ = 0.077, f₄ = 0.012: both shells almost entirely recombination-fed, the share factor (1−f₄)/(1−f₃) ≈ 1.07 cannot compensate B₃/B₄ ≈ 1.11, and the channel sign is decided by |C₃| > |C₄|. Over all 784 pairs P_c < 0 at 80, all on the same column.

**What may be said.** P > 0 (448/448) is an empirical property of this operator over the defended range, associated with the unequal supply fractions; the ground-fed contribution is positive everywhere, the recombination-fed one at 426, the sum everywhere. It is *not* a consequence of f₃ > f₄ alone: the sufficient conditions involve A₄/A₃ and B₃/B₄ as well as the shares, and the 22 pairs show the shares losing. Not written as a theorem.

**Sensitivity.** The first-order additive split reproduces the exact secant P̄ to 3.7 % median, 35 % worst; the sign conclusions do not depend on it (P̄ > 0 and P_a + P_c > 0 agree at 448/448). Secant A, C over one grid interval; a finer grid would tighten the residual, not move the counts.

**Thesis home.** §5.5.2 paragraph "Why P > 0: a decomposition, not a proof", Eq. `eq:P_channels`.


### K10. Chapter 2 figures audited against their CSVs: the n=5 Rydberg test, the propagation figure, and the subset table ✅ Verified (17 Sep 2026)

**Why.** The author reported the Chapter 2 figures wrong. Every number the three CCC/RMPS figures and their tables draw was rebuilt from the CSVs the figure script reads, and each claim the text makes about them was tested against the same files.

**What was wrong.** (1) `fig2_3` and §2.x claimed the RMPS kink at n=5 "at both 1 eV and 10 eV". At 10 eV there is none: RMPS is 3 to 7 points flatter than CCC at every step and the step into n=5 is where the two differ *least* (RMPS/CCC along 1s→np 1.06, 1.19, 1.29, 1.34). The tabulated quantity K n³ was also dominated at 1 eV by the shared threshold factor e^(−ΔE/T): −26, −15, −10 % at 4→5, 5→6, 6→7 out of CCC's "smooth" −41, −16, −12. (2) `fig2_4(a)` drew the 8 density rows with arithmetic midpoints on a log axis: 1.06 decades for the 10¹² row, 0.28 for 10¹⁵. (3) `fig2_4(b)` plotted the defended range pooled over k = 1, 2, 4 while ch5 quoted Δ 7.1 / cap 6.5 / S̄ 7.6 / G 0.95 %, which are means over all 2288 rows; the ion-swap "−7.6 / −8.3 / −1.4 %" were unlabelled medians over 338 heating pairs whose means are −1.7 / −3.1 / −1.5. (4) ch2 said "all ten comparisons beyond a factor of two" have CCC larger; the plotted file has 105 above 2 and 275 below 0.5. (5) `tab:atomic_sensitivity` cited a CSV with no subset rows; the rows lived only in `ccc_anderson.md` §7.5. The 140 a₀ attribution to Anderson 2000 was in live text with a `%`-comment flag.

**Prediction, written before the runs.** Subset substitution: n=5-only ≈ −0.9 %, n≤4 ≈ −10.8 %, ground-state ≈ −12.4 %, all ≈ −11.3 % at [23,5]. **Refuter:** n=5-only of several percent, or ground-state-only far below the full set.

**Runs.** `anderson_validity_range.py` extended with `R_CCC_bz`, `R_And_bz` = K n³ e^(+ΔE/T) (IH = 13.6058 eV, the threshold the Maxwell average uses) and rerun. New `verify_anderson_subset_impact.py --write` → `validation/anderson_subset_impact/`; L_grid `2d92b58e…`, ratio from `ccc_vs_anderson2002_thesis_Te_grid.csv`; P0 (22.73 µs, 2.277 ns, 9982) asserted.

| check | result |
|---|---|
| K_CCC_stored ≡ recomputed K_CCC (170 joined rows) | ratio 1.0000 |
| tab:anderson from the CSV, all six rows | exact |
| Υ→K with g = 2(2ℓ+1) at ℓ = 0, 1, 2 | 0.9999 |
| corrigendum 1.44 / 13.1 / 33.1 / 86.2 % on 1s→3p | reproduced |
| L = R + nₑC | 3.1e-16 |
| six recorded subset rows | 6/6 reproduced to the digit |
| Fig 2.1: 2s→2p share of 2s loss 99.972 % / min 99.921 %; A(2p→1s) 6.2684e8 | reproduced |

**Result.** In the corrected quantity the RMPS step into n=5 at 1 eV is *positive* in all three series (+28.4, +12.4, +12.0 %) while every other step of either code is negative; CCC is not smooth there either (−20.6 % on 1s→np between −9 and −1). The kink is a 1 eV feature, which points to a near-threshold basis artefact rather than the high-energy flux-absorption mechanism the text gave. n=5's −0.94 % is a net: ground-fed −3.27 %, excited-fed +2.37 %. Unified scope for every substitution figure: the 448 defended pairs (k = 1): exc Δ 9.1, cap 8.4, S̄ 9.1, G 1.1, ε 9.8 (max 25.7); ion ε 16.4; both ε 15.5, max 25.4. Under the ionisation swap the per-pair headroom moves 17.7 % in absolute terms, 62 % downward, median −8.3 %, while the distribution median is unchanged (80.4 → 80.5 %): it scrambles the operating point, not shifts it.

**What may be said.** The localisation to RMPS survives on the sign of the 1 eV step; the 10 eV claim is withdrawn; the temperature dependence is now stated as evidence about mechanism, held as an argument not a computation. Table 2.x's numbers were right and now have a producing script.

**Sensitivity.** Pooling k = 1, 2, 4 moves the fig 2.4(b) bars by < 0.3 points; 13.598 vs 13.6058 eV moves the R_bz steps by < 0.1 point. The Lotz range quotes 1.16 / 2.13 where the pooled n<9 file gives 1.17 / 2.11 (2.13 is the Tₑ = 2.95 eV column); left as is.

**Thesis home.** §2.x `sec:atomic_uncertainty`, `sec:n5_anomaly`, `sec:atomic_propagation`; `tab:rydberg`, `tab:atomic_sensitivity`; `fig2_3`, `fig2_4`; ch5 `sec:bound` and summary; ch7 atomic-data paragraph.

**Addendum, same day: Fig 2.2(a) redrawn against ΔE/T.** K_RMPS on the x-axis mixes strength, threshold and temperature and had no single meaning; the "no dependence on the rate" it showed (rank corr 0.02 with log K at n′=5) was two opposing trends cancelling. **Predicted before computing:** |err| grows toward threshold for ground-state 1s→5ℓ, shrinks for 4→5, pooled n′=5 small. **Found:** ground-state splits 3:2 (5s, 5p, 5d grow; 5f, 5g do not), 4→5 splits 10:7:3, and, not predicted, all 25 transitions into n=5 from n=2 and n=3 grow toward threshold without exception (pooled +0.65, +0.73) while 17 of 24 into n=4 shrink toward it (−0.3 to −0.5). Opposite signs in neighbouring shells rule out a shared averaging defect, on the right variable; "rules out a near-threshold artefact" as previously written was wrong, since the 2,3→5 class is exactly near-threshold behaviour, consistent with the 1 eV RMPS kink of Fig 2.3. Thresholds joined from `ccc_vs_anderson2002_benchmark.csv`; text, caption and figure script updated.
