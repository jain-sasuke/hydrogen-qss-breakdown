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
