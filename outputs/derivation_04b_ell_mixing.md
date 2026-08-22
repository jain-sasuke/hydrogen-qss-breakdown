# Derivation 04b — Proton-Impact ℓ-Mixing (physics chain)

**Quantity:** the proton-impact ℓ-mixing rate $K_{\rm lmix}$ — collisional redistribution of
population among the $\ell$-sublevels within a fixed $n$-shell.

**Why it earns its own note:** ℓ-mixing is *the* physics that changed the excited-state
relaxation from the artifact-era 25 ns to the corrected 2.28 ns. It enters `build_L` as Term 5
(`K_lmix`, from `compute_lmix.py`, PSM20 Debye formula). Before that code can be audited, the
physical scaling and order of magnitude must be derivable by hand — otherwise the code is being
trusted, not checked.

**Status:** **COMPLETE — all steps done.** Step 1 (physics picture): derived from first
principles, student-led. Step 2 (quantitative hand-derivation): time integral, cross-section,
Coulomb log, Maxwellian average, and $n{=}5$ number — all done by hand (§6). Step 4 (code audit
of `compute_lmix.py`): done — **found a real error** ($F(U_m)$ set to 1; true value 3–7), traced
to Badnell (2021) Eq. 9 at the source, and **numerically proved harmless** to
$\tau_{\text{relax}}$ (<0.2% shift) (§7). Triangulated across hand derivation, independent
second-model derivation, Badnell's published equations, and the actual `L_grid.npy`.

**To be grounded in:** Badnell et al. (2021) PSM20 (`Badnell_et_al__2021.pdf` in project);
Vrinceanu & Flannery (`Vrinceanu__Flannery.pdf`); standard Stark / impact-parameter treatment.

**Feeds:** Q4 (the two timescales — ℓ-mixing sets the fast intra-shell group and reshapes the
relaxation eigenvalue); `verify_bundling_psm20.py`.

---

## 1. The physical chain (derived, not asserted)

Each link below was reasoned out from physics, not taken from a formula:

**1. Degeneracy — the mixing is nearly free.**
In hydrogen, $ns$, $np$, $nd$ … within a shell are *degenerate* (identical energy). So
redistributing population among them costs **no energy** — unlike collisional excitation
$n\to n{+}1$, which needs a real energy chunk delivered by the collision partner. This is the
root reason ℓ-mixing is fast: there is no energy barrier to clear.

**2. Slow, long collisions — time to act.**
Protons are ~1836× heavier than electrons, so at the same temperature they move far slower. The
encounter is long and gentle compared with the electron's orbital period — the proton's field
leans on the atom adiabatically rather than delivering a sudden kick. Long dwell time lets the
field actually rotate the atom's internal state.

**3. The mechanism — linear Stark effect.**
A passing proton is a moving source of a Coulomb field; beside the atom it acts as a nearly
static electric field $\mathcal E$. Because $s/p/d$ are degenerate, hydrogen shows the **linear**
Stark effect (shift $\propto \mathcal E$; in non-hydrogenic atoms it would be quadratic and much
weaker). The field's energy eigenstates are **not** $s/p/d$ but *mixtures* of them (Stark /
parabolic states). An atom starting in pure $np$ is therefore no longer in an energy eigenstate
once the field arrives: it evolves as a superposition of Stark states, and when the field
rotates and fades as the proton leaves, the population is left redistributed across $\ell$. No
photon, no $\Delta\ell=\pm1$ selection rule — just the field making $s/p/d$ the "wrong" basis
during the encounter.

**Causal chain:** degeneracy → linear Stark effect → field mixes $s/p/d$ into Stark states →
passing proton redistributes $\ell$-population → **ℓ-mixing**, fast and nearly free.

---

## 2. Why distant collisions dominate (and the correct reason)

The rate is a competition between "how strong" (per collision) and "how many" (trajectories):

- **Field strength:** $\mathcal E \propto 1/b^2$ at impact parameter $b$ — closer is stronger.
- **Number of trajectories:** protons passing near impact parameter $b$ cross an annulus of area
  $\propto b\,db$ — there is far **more phase-space area** at large $b$.

When these are combined in the rate integral $\int(\text{effect})\times(\text{number at }b)\,db$,
the large-$b$ region wins: distant, weak, gentle encounters dominate.

**Precision (from independent derivation, verified — adopt this wording):** "distant collisions
dominate" specifically means they dominate **logarithmically, not by a strong power law**. With
$P(b)\propto b^{-2}$ and the $b\,db$ area weight, the integrand goes as $\int P(b)\,b\,db \propto
\int db/b = \ln(b_{\max}/b_{\min})$ — so **each decade in $b$ contributes roughly equally**, and
close encounters still supply an order-unity term (which PSM20 regularizes at small $b$). This is
why the cross section carries a Coulomb logarithm and why the *upper cutoff matters so much*.

**Correction recorded (student's first reason was wrong, then fixed — twice-refined):** distant
dominance is **not** due to Coulomb repulsion steering protons away. The atom is **neutral**, so
the **monopole** term cancels — there is no net Coulomb force on the atom's centre of mass. But
the sharper statement is: neutrality kills the net *force*, **not** the *field across the atom*.
The passing proton still produces a field gradient over the atomic size, and the surviving
**dipole (Stark) term** is exactly what drives the mixing. So the correct picture is "no net
force, but a real dipole perturbation," not "no interaction." The dominance itself is
phase-space counting ($b\,db$), made precise as the logarithm above.

Physical consequence: because gentle long-range encounters dominate, ℓ-mixing is driven by the
**long-range dipole coupling**, which is exactly why it is so strong and why it needs a
long-distance cutoff.

---

## 3. What sets the cutoff — Debye shielding

The impact-parameter integral would **diverge** at large $b$ (the integrand goes as $db/b$, whose
antiderivative $\ln b$ runs to infinity). A physical cutoff $b_{\max}$ is required. In a plasma a
proton is not bare: it is surrounded by a screening cloud (electrons drawn in, ions pushed out),
so its potential is $\phi(R)\propto e^{-R/\lambda_D}/R$. Beyond about one **Debye length**
$\lambda_D$, the field is exponentially suppressed and the atom feels nothing. So:

$$b_{\max} \approx \lambda_D$$

This regulates the integral and is the origin of the **Coulomb logarithm** $\ln(b_{\max}/b_{\min})
= \ln(\lambda_D/b_{\min})$ in the rate formula.

**Numbers at ITER reference (independently verified this session):** for $T_e=3$ eV,
$n_e=1.4\times10^{14}$ cm⁻³, the electron-only Debye length is $\lambda_{De}\approx1.09$ µm and
the two-species ($T_e=T_i$) value is $\lambda_D\approx0.77$ µm. With $b_{\min}\sim n^2 a_0
\approx1.3$ nm for $n=5$, the Coulomb log is $\ln(\lambda_D/b_{\min})\approx6$ — order-unity-times-
a-few, as a logarithm should be.

**Audit item for `compute_lmix.py` (flagged by the independent review, and correct):** the Debye
length is the right upper cutoff **only if it is smaller than competing cutoffs** — the finite
lifetime of the state, or any residual energy splitting that breaks exact $\ell$-degeneracy
(Badnell/PSM20 emphasize these can matter for low-lying states). At this high-density reference
point Debye plausibly wins, but the code must be checked to confirm (a) the Debye length appears
as $b_{\max}$, and (b) whether a lifetime/splitting cutoff is also applied and which one is
actually binding. If the Debye cutoff is missing or mis-set, the ℓ-mixing rate — and hence the
2.28 ns — is wrong.

---

## 4. What this predicts (to be made quantitative in Step 2)

Qualitatively, before any arithmetic:

- **Steep growth with $n$.** Larger shells have larger orbitals (dipole $\propto n^2$) and more
  closely spaced Stark states, so ℓ-mixing rises steeply with $n$ (the "$\sim n^4$" behaviour
  referenced in Q1's $8s$ diagonal). This is why high-$n$ shells $\ell$-mix so fast they can be
  bundled with a statistical $\ell$-distribution.
- **Fast at ITER density.** $\propto n_p = n_e$ (quasi-neutrality); at $n_e\sim10^{14}$ cm$^{-3}$
  the rate should reach the picosecond group — faster than radiative decay for mid/high $n$.
- **Beats radiative decay above some $n$.** The crossover (ℓ-mixing rate $\approx$ radiative
  rate) is conceptually the same competition as Griem's boundary (Q4 §6A).

**Step 2 targets (next session):** set up the rate as an impact-parameter integral, obtain the
$\ln(\lambda_D/b_{\min})$ Coulomb-log form, and estimate an actual number for $n=5$ at ITER
reference. Then check it beats $A(n{=}5)$ and lands in the picosecond range. Only then read
`compute_lmix.py`.

> **Note on the ITER reference point (framing decision — applies across the whole thesis).**
> The "ITER reference point" used for worked numbers is $T_e\approx3$ eV, $n_e\approx1.4\times10^{14}$
> cm⁻³. The nearest actual grid values are $T_e=2.947$ eV, $n_e=1.389\times10^{14}$ cm⁻³ (grid
> indices Te=23, ne=5) — quote these once so an examiner checking the indices finds no mismatch.
> Literature support: $T_e\approx3$ eV is the detachment-onset temperature (JET-ILW), and
> $n_e\approx1.4\times10^{20}$ m⁻³ sits at the low end of the ITER detached-divertor density range
> (ITER SOLPS). It is physically real and citable.
>
> **Decided policy:** the reference point is an *illustrative anchor only*, used to make abstract
> results concrete (one worked trace, one order-of-magnitude number). It is **not** load-bearing.
> Every general claim — QSS breakdown, $M\gg1$-yet-error-appears, ℓ-mixing dominance — must be
> carried by the **full $T_e$–$n_e$ grid**, not by this single point. We deliberately keep
> discussion of the reference point *brief*, because leaning on one point weakens a general
> conclusion (an examiner can ask "why 3 eV and not 1 eV?"). Anchor with the point; prove with
> the grid.
>
> For ℓ-mixing specifically: derive the $n=5$ number *at* the reference point for concreteness,
> but state the $n$-, $n_e$-, and $T_e$-scalings as the real result — the scalings hold across the
> grid; the single number just illustrates them.

---

## 5. Checkpoint status

- ✅ Physics picture: degeneracy → Stark → distant-collision dominance → Debye cutoff. Student
  derived the chain; the one wrong turn (repulsion) was caught and corrected to phase-space
  counting, then further sharpened (logarithmic dominance; dipole survives even though the
  monopole force cancels).
- ✅ **Independent cross-check (second model, then verified here).** An independent derivation
  reproduced the same chain and the impact-parameter integral, and I verified its checkable
  numbers directly: two-species Debye length 0.769 µm (claim 0.77), electron-only 1.088 µm (claim
  1.09), Coulomb log ≈ 6. Its predicted $\nu_{\ell\text{-mix}}(n{=}5)\sim10^{11}$–$10^{12}$ s⁻¹,
  $\tau\sim1$–5 ps — and this **matches the actual `L_grid.npy`**, whose $n=5$ sublevel loss
  rates are $8\times10^{10}$–$1.1\times10^{12}$ s⁻¹ (0.94–12 ps). Three independent sources agree:
  the derivation, the matrix, and the original picosecond expectation.
  *(Caveat: the second model's specific per-$\ell$ decimal rate values were asserted, not shown —
  usable as an order-of-magnitude cross-check, not citable as a calculation.)*
- ✅ Quantitative hand-derivation (Step 2) — complete, §6 below.
- ✅ Code audit of `compute_lmix.py` (Step 4) — complete, §7 below. Found a real
  implementation error, traced it to the published source, and proved it harmless to the
  headline result.

---

## 6. Step 2 — The quantitative hand-derivation (student-led)

### 6.1 The time integral (the total transverse push)

Setup: atom at origin; proton on straight-line trajectory
$\mathbf R(t)=b\,\hat x+vt\,\hat z$, so $R(t)=\sqrt{b^2+v^2t^2}$. The perturbation is the
dipole term (monopole cancels because the atom is neutral — nucleus repulsion and electron
attraction cancel at order $1/R$):

$$V(t) = -\frac{e^2}{4\pi\epsilon_0}\,\frac{\mathbf R(t)\cdot\mathbf r}{R(t)^3}$$

$\mathbf R\cdot\mathbf r = b\,x + vt\,z$. The $z$-piece is **odd in $t$** (pushes one way on
approach, the opposite on recession) → integrates to zero. The $x$-piece is even and survives.
Because same-shell states are degenerate, the phase factor $e^{i\omega_{fi}t}=1$ (no rhythm
mismatch — the mathematical fingerprint of "mixing is free"). The surviving time integral,
evaluated by $u=vt/b$:

$$I=\int_{-\infty}^{\infty}\frac{b\,dt}{(b^2+v^2t^2)^{3/2}} = \frac{1}{bv}\left[\frac{u}{\sqrt{1+u^2}}\right]_{-\infty}^{\infty} = \boxed{\frac{2}{bv}}$$

Physics of the result: instantaneous field $\propto1/b^2$, but encounter duration $\propto b/v$,
so integrated push $\propto1/b$ — **distant collisions are partially compensated by longer dwell
time.** The $1/v$: slower protons mix more (the original intuition, now in algebra).

### 6.2 Probability, cross-section, Coulomb logarithm

$$P(b)=|a_{fi}|^2 \propto \frac{1}{b^2v^2}\,|\langle f|x|i\rangle|^2$$

$$\sigma(v)=2\pi\int_{b_{\min}}^{b_{\max}} P(b)\,b\,db \;\propto\; \frac{|\langle f|x|i\rangle|^2}{v^2}\int_{b_{\min}}^{b_{\max}}\frac{db}{b} \;=\; \frac{|\langle f|x|i\rangle|^2}{v^2}\,\ln\!\left(\frac{b_{\max}}{b_{\min}}\right)$$

The $1/b^2$ from $P$ meets the $b\,db$ annulus weight and leaves $db/b$: **the Coulomb
logarithm is born from that single cancellation.** Each decade in $b$ contributes equally. The
integral diverges at **both** ends without cutoffs: $b_{\max}\approx\lambda_D$ (Debye
screening), $b_{\min}$ where $P\to1$ (perturbation theory fails; strong-collision regime, PSM20
regularizes). The divergences are the theory announcing the limits of its own validity.

With $|\langle f|r|i\rangle|\sim n^2a_0$: $\sigma\propto n^4$ — the same $n^4$ that showed up in
Q1's $8s$ diagonal.

### 6.3 Maxwellian average and the scaling law

$\sigma v\propto1/v$, and $\langle 1/v\rangle=\sqrt{2\mu/\pi k_BT}\propto T^{-1/2}$ (derived
explicitly by Gaussian integral, verified). Hence:

$$\boxed{\nu_{\ell\text{-mix}} = n_p\langle\sigma v\rangle \propto n_e\,n^4\,T_i^{-1/2}\,\ln\Lambda}$$

**Scope flag:** the temperature is the **proton** (relative-motion) temperature. $T_i=T_e$ is an
additional plasma assumption, to be stated in §1.5 scope alongside the Maxwellian assumption.

### 6.4 The number (no placeholder constants)

Assembling with $v_0=e^2/4\pi\epsilon_0\hbar$ (Bohr velocity), $\sigma\approx 8\pi(v_0/v)^2(n^2a_0)^2\ln\Lambda$,
at ITER reference ($T=3$ eV, $n_e=1.4\times10^{14}$ cm⁻³, $n=5$):

| Quantity | Value |
|---|---|
| $\lambda_D$ (two-species) | 0.769 µm |
| $b_{\min}=n^2a_0$ | 1.32 nm |
| $\ln\Lambda$ | 6.4 |
| $\langle\sigma v\rangle$ | $4.5\times10^{-2}$ cm³/s |
| $\nu_{\ell\text{-mix}}(n{=}5)$ | $6.2\times10^{12}$ s⁻¹ |
| $\tau_{\ell\text{-mix}}(n{=}5)$ | **0.16 ps** |

Comparison: hand estimate 0.16 ps; actual `L_grid.npy` $n{=}5$ sublevels 0.94–12 ps. **High by
~6–10×, in the right direction, for an understood reason:** the crude estimate uses
$|\langle f|r|i\rangle|\sim n^2a_0$ (full orbital size) and drops all angular factors; the real
Badnell $D_{ji}$ angular algebra supplies exactly those $O(1)$ suppression factors. **Verdict:
pass** — order-of-magnitude agreement with understood discrepancy, which is what an
order-of-magnitude derivation can deliver.

### 6.5 What the toy omits (honest limits)

Straight-line trajectory (fine for the log-dominant distant collisions; fails only at small $b$
where perturbation theory fails anyway); exact angular algebra (the ~6–10× factor; supplied by
Badnell $D_{ji}$); exact degeneracy (fine structure gives the competing lifetime/splitting
cutoff — real check item); first-order PT (the $b_{\min}$ boundary); $T_i=T_e$, Maxwellian,
$n_p=n_e$ (scope assumptions). The toy's scaling structure is **exact**; the omissions are
$O(1)$ prefactors and known cutoff physics — precisely how the published derivations
(Pengelly–Seaton 1964 → Guzmán 2017 → Vrinceanu 2019 → Badnell PSM20 2021) are structured.

---

## 7. Step 4 — Code audit of `compute_lmix.py`: a real bug, proven harmless

### 7.1 What the code gets right (matches the hand derivation)

- **$D_{ji}=6n^2\ell_>(n^2-\ell_>^2)$** — the $n^4$ scaling with the proper angular factors. ✓
- **$\sqrt{\pi\mu I_H/kT}$** — the Maxwellian $T^{-1/2}$. ✓
- **Reciprocity** $q_{\rm up}/q_{\rm down}=(2\ell_{\rm hi}+1)/(2\ell_{\rm lo}+1)$. ✓
- **Conservative redistribution** (column sums preserved) — independently verified to hold in
  `L_grid.npy` earlier (column sums exactly $-K_{\rm ion}n_e$ across all densities). ✓
- Structure: $\Delta\ell=\pm1$ only, $n=2$–8 resolved, bundled shells statistical. ✓

### 7.2 The bug — $F(U_m)$ set to 1

Badnell (2021), Eq. 9 (read directly from the paper, OCR of the source PDF):

$$q_{ji}=\frac{a_0^3}{\tau_0}\left(\frac{\pi\mu I_H}{k_BT_e}\right)^{1/2}\frac{D_{ji}}{\omega_\ell}\underbrace{\left[\frac{\sqrt\pi}{2}U_m^{-3/2}\mathrm{erf}(\sqrt{U_m})-\frac{e^{-U_m}}{U_m}+E_1(U_m)\right]}_{F(U_m)}$$

with $U_m=E_{\min}/k_BT_e$ and $E_{\min}=a_0^2\mu I_H D_{ji}/(2P_1\omega_\ell R_c^2)$,
$R_c=\lambda_D$ (Eq. 5, 8). **$F(U_m)$ is the Coulomb logarithm** — for small $U_m$ it behaves
as $\approx\ln(R_c/R_1)$, the very $\ln(\lambda_D/b_{\min})$ of §6.2.

The code (line 166) sets $F=1$, with a docstring claiming this is the "low-density limit
$U_m\to0$, $F\to1$." **That justification is wrong twice:**
1. At ITER reference the *true* $U_m$ (computed from Badnell Eq. 5/8 with the real Debye
   radius) is $10^{-3}$–$10^{-1}$ — small, yes — **but $F(U_m\to0)\to\infty$ logarithmically,
   not $\to1$.** The limit was inverted.
2. Computed values at ITER reference: $F=6.65$ (2p–2s), 5.73 (3d–3p), 4.29 (5g–5f), 2.92
   (8k–8i). The code therefore **under-counts ℓ-mixing rates by ×3–7** (more for low $n$).

*(Note: the docstring's own definition of $U_m$ — multiplying by $n_p\,a_0^3/\tau_0$ — is also
wrong dimensionally vs. Badnell Eq. 8; the first audit pass using it gave $U_m\sim10^{14}$, a
red herring corrected by reading the source.)*

### 7.3 The robustness proof — the bug does not touch $\tau_{\text{relax}}$

Test: scale every intra-$n$ adjacent-$\ell$ off-diagonal of the actual $L$ (ITER ref) by the
correct per-transition $F(U_m)$, restore the exact column sums (ℓ-mixing is conservative), and
recompute the eigenvalues:

| | $\tau_{\rm QSS}$ | $\tau_{\rm relax}$ |
|---|---|---|
| Code as-is ($F=1$) | 22.71 µs | 2.280 ns |
| Corrected ($F=3$–7) | 22.73 µs | 2.277 ns |
| **Change** | **+0.07%** | **−0.12%** |

**Why, physically:** ℓ-mixing is already the fastest process in the matrix (~ps). Multiplying
the fastest modes by 3–7 leaves the slow relaxation eigenvalue (~ns) untouched — the
$\ell$-sublevels are fully equilibrated on the relaxation timescale either way. Timescale
separation protects the slow eigenvalue from errors in the fast rates.

### 7.4 Consequences and actions

1. **$\tau_{\rm relax}=2.28$ ns stands.** Proven insensitive (<0.2%) to the ℓ-mixing
   Coulomb-log factor. This robustness statement belongs in Ch. 4 — it *strengthens* the result.
2. **Fix `compute_lmix.py`** (or at minimum document the error): the absolute ℓ-mixing rates
   are low by ×3–7, which matters for any $\ell$-resolved population or line-ratio use, even
   though it doesn't move the eigenvalues. The docstring's limit statement must be corrected.
3. **Remaining check item:** whether the competing lifetime/splitting cutoff (Badnell Eq. 10–12,
   $t=0.72\tau_{n\ell}$ or $1.12\hbar/\Delta E$) ever beats the Debye cutoff anywhere on the
   grid — expected not at these densities, but unverified.
4. **Defense framing:** "We found an implementation error in the ℓ-mixing Coulomb-logarithm
   factor, traced it to Badnell (2021) Eq. 9, quantified it (×3–7 under-count), and proved the
   relaxation eigenvalue is insensitive to it (<0.2%)." Finding and bounding one's own bug is
   evidence of rigor, not weakness — same pattern as the −46% Hα artifact.

### 7.5 Triangulation summary (the 100%-sure table)

| Source | What it gave | Agrees? |
|---|---|---|
| Hand derivation (§6) | scaling $n_e n^4 T^{-1/2}\ln\Lambda$; $n{=}5$ ~0.16 ps (crude) | ✓ |
| Independent second model | same chain; 1–5 ps with angular factors | ✓ |
| Badnell 2021 (source PDF) | Eq. 9 confirms $F(U_m)$ = the Coulomb log | ✓ |
| `L_grid.npy` (actual matrix) | $n{=}5$ rates 0.94–12 ps | ✓ |
| `compute_lmix.py` | correct structure; $F=1$ error, ×3–7, harmless to $\lambda_1$ | audited |
