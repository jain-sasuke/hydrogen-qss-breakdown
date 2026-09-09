# Notation and Definitions

**Draft of Chapter 3 §3.1 and §3.6, and the project-wide convention.**
Written 23 August 2026. Everything downstream inherits from this document; fix
it here and nowhere else.

Written in the target voice — first principles, every symbol defined at first
use with units, intuition before formalism — so it can go into the thesis
directly rather than being translated later.

---

## 1. The state space

The model tracks the populations of 43 bound levels of atomic hydrogen.

For $n \le 8$ the levels are resolved in orbital angular momentum $\ell$, giving
$\sum_{n=1}^{8} n = 36$ states: $1s$, $2s$, $2p$, $3s$, $3p$, $3d$, and so on to
$8k$. For $n = 9$ to $15$ the $\ell$-sublevels are **bundled** into a single
state per shell, on the assumption that they are populated in proportion to
their statistical weights $g_{n\ell} = 2(2\ell+1)$. That gives 7 more states, 43
in total.

**Why the split.** Resolving $\ell$ matters where the sublevels of a shell are
*not* interchangeable — where one has a fast radiative exit and another does not.
That is the case at low $n$: $2p$ decays to the ground state at
$6.27\times10^{8}$ s⁻¹ while $2s$ cannot decay at all by electric-dipole
radiation. At high $n$ the sublevels are collisionally mixed far faster than
anything else happens to them, so distinguishing them buys nothing. Chapter 2
shows this assumption measured rather than asserted: at $n = 4$ the $4f$ fraction
of the shell is 0.4361–0.4375 against the statistical 0.4375 across the whole
grid.

Throughout, $p$ indexes a level, $n$ its principal quantum number, and the index
$g = 0$ denotes the ground state $1s$. The set $\mathcal{E} = \{1, \dots, 42\}$
denotes the excited levels.

**Ordering is loaded from `state_index.csv`**, never assumed, and is quoted in
Appendix A.

---

## 2. The rate equation

Each level gains atoms from every process that populates it and loses atoms to
every process that depopulates it. Writing that balance for every level at once
gives

$$\frac{d\mathbf n}{dt} = \mathbf L\,\mathbf n + \mathbf S\,n_{\rm ion}$$

| symbol | meaning | units |
|---|---|---|
| $\mathbf n$ | population vector, 43 components | cm⁻³ |
| $\mathbf L$ | rate matrix, a function of $T_e$ and $n_e$ | s⁻¹ |
| $\mathbf S$ | recombination source per unit ion density | s⁻¹ |
| $n_{\rm ion}$ | proton density | cm⁻³ |

$L_{pq}$ for $p \ne q$ is the rate of the transition $q \to p$; the diagonal
$L_{pp}$ is minus the total loss rate from $p$.

**A chemical engineer will recognise the structure.** $\mathbf L$ is the rate
matrix of a reaction network among 43 species, and $\mathbf S n_{\rm ion}$ is a
feed stream entering from outside it. Recombination cannot be an element of
$\mathbf L$ because $\mathbf L$ multiplies $\mathbf n$: recombination is
proportional to the *ion* density, a species outside the tracked manifold, so it
enters as a constant feed exactly as a CSTR inlet does.

**The source is quadratic in density:**

$$\mathbf S = n_e\left(\boldsymbol\alpha_{\rm RR} + n_e\boldsymbol\alpha_{\rm 3BR}\right)$$

with $\boldsymbol\alpha_{\rm RR}$ the radiative and $\boldsymbol\alpha_{\rm 3BR}$
the three-body recombination coefficients. Verified numerically to machine
precision. Note the consequence for units: $\mathbf S$ is in s⁻¹, **not** cm³
s⁻¹ — an error in the code comments that propagated for months and is recorded
in Chapter 4.

**Conservation.** Every column of $\mathbf L$ sums to $-S_{q,\rm ion}n_e$, the
ionisation leak from level $q$: an atom excited out of $q$ arrives somewhere else
in the manifold, so internal transfers cancel and only ionisation removes an atom
from the bound states. This is the primary check on the matrix assembly and holds
to $2.6\times10^{-11}$ across the grid.

---

## 3. Two clocks

Diagonalising $\mathbf L$ at any grid point gives 43 eigenvalues, all with
negative real part. Two of them matter.

$$\tau_{\rm QSS} = \frac{1}{|\lambda_0|}, \qquad \tau_{\rm relax} = \frac{1}{|\lambda_1|}, \qquad M = \frac{\tau_{\rm QSS}}{\tau_{\rm relax}}$$

where $\lambda_0$ is the least-negative eigenvalue and $\lambda_1$ the next.
$\tau_{\rm QSS}$ is the time for the **ionisation balance** to settle;
$\tau_{\rm relax}$ is the time for the **excited manifold** to settle.

At the benchmark point these are 22.7 µs and 2.28 ns — a separation of
**$M = 9982$**, four orders of magnitude. Across the grid $M$ never falls below
86.8 and reaches $1.73\times10^{9}$.

**Convention, and it matters.** $M$ is defined as $\tau_{\rm QSS}/\tau_{\rm
relax}$, so large $M$ means the excited states are fast compared with the
ionisation balance. The inverse convention appears in some early project notes
and is **retired**.

**Which operator.** Where a temperature step is involved, the eigenvalues are
taken from the **post-step** operator $\mathbf L(T_e + \Delta T_e, n_e)$, since
that is the operator governing the relaxation. Any table quoting $M$ must state
the operator and the step, because $M$ at the benchmark point takes three
different values — 9982 (unstepped), 8243 (after $+4.8\%$), 4856 (after
$+0.6$ eV) — all individually correct.

---

## 4. The quasi-steady-state approximation

Partition the levels into the **slow** subspace $\mathcal{S} = \{1s\}$ and the
**fast** subspace $\mathcal{F} = \mathcal{E}$, and split $\mathbf L$ conformably.
The QSS approximation sets $d\mathbf n_F/dt = 0$, giving the algebraic system

$$\mathbf L_{FF}\,\mathbf n_F^{\rm QSS} + \mathbf L_{FS}\,\mathbf n_S + \mathbf S_F n_{\rm ion} = 0$$

$$\boxed{\ \mathbf n_F^{\rm QSS} = -\mathbf L_{FF}^{-1}\left(\mathbf L_{FS}\,\mathbf n_S + \mathbf S_F n_{\rm ion}\right)\ }$$

**This is the Bodenstein steady-state approximation applied to 42 species at
once.** For a reactive intermediate $A \to I \to P$ one sets $d[I]/dt = 0$ and
solves for $[I] = (\text{production})/(\text{loss})$; the intermediate stops
being a differential unknown and becomes algebraically slaved to whatever $[A]$
currently is. QSS is the identical move, with $-\mathbf L_{FF}^{-1}$ playing the
role of "divide by the loss network" — a *networked* division, because every
level's balance couples to every other's.

Substituting back into the slow equation gives the reduced dynamics

$$\frac{d\mathbf n_S}{dt} = \underbrace{\left(\mathbf L_{SS} - \mathbf L_{SF}\mathbf L_{FF}^{-1}\mathbf L_{FS}\right)}_{\boldsymbol\Omega_{\rm QSS}}\mathbf n_S + \left(\mathbf S_S - \mathbf L_{SF}\mathbf L_{FF}^{-1}\mathbf S_F\right)n_{\rm ion}$$

$\boldsymbol\Omega_{\rm QSS}$ is the **Schur complement** of $\mathbf L_{FF}$ in
$\mathbf L$: the effective rate matrix for the slow subspace after adiabatically
eliminating the fast one. It is the algebraic form of the effective ionisation
coefficient that frameworks such as ADAS tabulate.

---

## 5. Two reference states — the distinction this thesis rests on

Spectroscopic practice works with **ratios** to the ground state,
$r_p = n_p/n_{1s}$, because absolute populations require a neutral density and a
viewing path length that are rarely known. Dividing the QSS solution by $n_{1s}$
gives

$$r_p^{\rm QSS} = \frac{\left[-\mathbf L_{FF}^{-1}\left(\mathbf L_{FS}\mathbf n_S + \mathbf S_F n_{\rm ion}\right)\right]_p}{n_{1s}}$$

**This is not a function of $(T_e, n_e)$ alone**, and the point is easy to miss.
The numerator carries $\mathbf L_{FS}\mathbf n_S$, proportional to the ground
density, and $\mathbf S_F n_{\rm ion}$, proportional to the ion density. What
sets $r_p$ is the *ratio* of the two supplies, so $r_p^{\rm QSS}$ depends on the
**reservoir ratio**

$$b_1 \equiv \frac{n_{1s}}{Z_1\,n_e\,n_{\rm ion}}$$

the ground-state **departure coefficient** — the factor by which the ground
population differs from its Saha–Boltzmann value $Z_1 n_e n_{\rm ion}$. It is
dimensionless, it is a *dynamical variable*, and it is not determined by the
local plasma conditions.

Two reference states follow, and they are not the same:

**The QSS reference.** $r_p^{\rm QSS}(T_e, n_e;\,b_1)$, evaluated at the
**instantaneous** reservoir. This is what adiabatic elimination actually claims:
the excited manifold is in steady state with whatever reservoir is present at
that moment.

**The CR-equilibrium reference.** $r_p^{\rm CRE}(T_e, n_e) \equiv r_p^{\rm
QSS}(T_e, n_e;\,b_1^{\rm eq}(T_e,n_e))$, obtained by additionally setting
$d\mathbf n_S/dt = 0$ — that is, by letting the ionisation balance equilibrate
too. **This is the quantity tabulated against $(T_e, n_e)$, and it is what a
two-parameter line-ratio inversion silently assumes.**

They coincide only when the ionisation balance has finished settling. Chapter 5
shows the first is accurate to $\mathcal O(10^{-8})$ throughout the transient,
while the second departs by tens of per cent and stays wrong for $\tau_{\rm
QSS}$.

---

## 6. The observable

$$\boxed{\ R \equiv \frac{n_3}{n_4} = \frac{n_{3s}+n_{3p}+n_{3d}}{n_{4s}+n_{4p}+n_{4d}+n_{4f}}\ }$$

**Why a ratio of these two shells and not something else.** The measure must
satisfy two criteria, stated here and used to justify every subsequent choice:

1. **It must contain no quantity the approximation does not claim to predict.**
   A norm over all 43 states fails this: the ground state is 73× the entire
   excited manifold, so such a norm is effectively an error measure on $n_{1s}$,
   which QSS never claimed to model.
2. **It must be reconstructible from a measurement.** Absolute line intensities
   need the neutral density and the path length; a ratio needs neither.

$R$ satisfies both. It is also, up to Einstein coefficients, the Balmer
$H_\alpha/H_\beta$ ratio — the standard divertor diagnostic — since
$H_\alpha$ is $3\to2$ and $H_\beta$ is $4\to2$.

**Shell ratio versus line ratio.** The A-weighted line ratio and the bare shell
ratio agree to **0.06%** at the benchmark point (0.386683 vs 0.386903), because
the $\ell$-populations move as a rigid body. Photon- versus energy-weighting
changes the result by $6.7\times10^{-16}$. Note that $4f$ is dipole-dark in
$H_\beta$ — it can only decay to $3d$ — and yet excluding its 44% share of the
$n=4$ shell moves the answer by under 2.2% anywhere on the grid. **The results
are reported for the shell ratio; the line-ratio equivalence is established in
Chapter 4 with these numbers.**

---

## 7. The two error measures

$$\varepsilon^{\rm QSS}(t) = \left|\frac{R(t)}{R^{\rm QSS}\!\left(T_e, n_e;\,b_1(t)\right)} - 1\right|, \qquad \varepsilon^{\rm CRE}(t) = \left|\frac{R(t)}{R^{\rm CRE}(T_e, n_e)} - 1\right|$$

$\varepsilon^{\rm QSS}$ asks: *have the excited states kept up with the reservoir
they actually see?* $\varepsilon^{\rm CRE}$ asks: *is the tabulated answer
right?* The first is the textbook worry. The second is what a diagnostician
actually suffers.

Two derived quantities are used in Chapter 5:

$$\varepsilon_{\rm step} = \left|\frac{R^{\rm CRE}_{\rm old}}{R^{\rm CRE}_{\rm new}} - 1\right|, \qquad \varepsilon_{\rm plateau} = \left|\frac{R^{\rm PE}}{R^{\rm CRE}_{\rm new}} - 1\right|$$

where $\mathbf n^{\rm PE}$ is the **partial-equilibrium state**: the excited block
solved under the new operator with the ground density frozen at its old value,

$$\mathbf n_E^{\rm PE} = -\mathbf L_{EE}^{-1}\left(\mathbf S_E n_{\rm ion} + \mathbf L_{Eg}\,n_g^{\rm old}\right)$$

$\varepsilon_{\rm step}$ compares the two endpoints of the transition;
$\varepsilon_{\rm plateau}$ compares the state the system actually occupies for
most of the transient against the tabulated answer.

**Naming discipline.** Six structurally different quantities have shared the name
`eps_step` across this project's code — max-over-states on raw populations,
max-over-states on ground-normalised ratios, single-state, a hardcoded fit, a
transient sample, and the shell-ratio quantity above. Only the definitions in
this section appear in the thesis. Where an earlier result is quoted, the measure
is named explicitly.

---

## 8. The two supply channels

Because $\mathbf L_{EE}$ is linear, the excited block splits exactly into what
arrives from below and what arrives from above:

$$\mathbf n_E = \underbrace{-\mathbf L_{EE}^{-1}\mathbf S_E n_{\rm ion}}_{\mathbf n^{(0)},\ \text{recombination-fed}} + \underbrace{-\mathbf L_{EE}^{-1}\mathbf L_{Eg}n_g}_{\mathbf n^{(1)},\ \text{ground-fed}}$$

This is Fujimoto's population-coefficient decomposition realised in the model's
own matrix; Chapter 4 verifies the correspondence against his published $r_0(p)$
and $r_1(p)$. Superposition holds to $3.1\times10^{-14}$ across 784 test cases.

Define the **ground-fed fraction** of shell $p$,

$$f_p = \frac{n_p^{(1)}}{n_p^{(0)} + n_p^{(1)}}$$

Scaling the ground density by a factor $x$ relative to its old value gives
$\mathbf n_E(x) = \mathbf n^{(0)} + x\,\mathbf n^{(1)}$, an exact one-parameter
family with $x = 1$ the partial-equilibrium state and $x = n_g^{\rm new}/n_g^{\rm
old}$ the new CR equilibrium. Differentiating $R$ along it:

$$\boxed{\ \frac{d\ln R}{d\ln x} = f_3 - f_4\ }$$

**The sensitivity of the diagnostic to a stale reservoir is the difference in
ground-fed fraction between the two shells.** It vanishes when either supply
dominates completely — when both shells are entirely ground-fed or entirely
recombination-fed — and is largest where the two compete. Chapter 5 shows that
this is the ionizing–recombining crossover, and that the crossover is detachment.

---

## 9. The benchmark point

$$T_e = 2.947052\ {\rm eV}, \qquad n_e = 1.389495\times10^{14}\ {\rm cm^{-3}}$$

grid indices $[23, 5]$ — the grid point nearest $T_e = 3$ eV,
$n_e = 10^{14}$ cm⁻³, chosen as a representative **attached** divertor condition.

**It is an illustrative anchor, not a published operating point.** The digits are
an artifact of a 50-point logarithmic temperature grid. Where a claim is general
it is proved on all 400 grid points and illustrated at this one; where a
quantity is expensive to compute it is reported here and its representativeness
stated. In particular the benchmark point lies *below* the region where the
diagnostic error becomes large, and off the ridge of Chapter 5 — which is stated
where it matters rather than left for a reader to notice.

---

## 10. Grids

$T_e$: 50 points, logarithmically spaced from 1 to 10 eV.
$n_e$: 8 points, logarithmically spaced from $10^{12}$ to $10^{15}$ cm⁻³.
400 grid points; both grids are loaded from the pipeline's own files.

The lower temperature bound is not a data limit — the cross sections extend far
higher — but a physics limit: below about 1 eV, molecular channels omitted from
this model become significant in a gas that is 96.6% neutral. Chapter 6 states
what that costs.

---

## 11. Symbol table

| symbol | meaning | units |
|---|---|---|
| $n_p$ | population of level $p$ | cm⁻³ |
| $n_e$, $n_{\rm ion}$ | electron, proton density | cm⁻³ |
| $T_e$ | electron temperature | eV |
| $\mathbf L$ | CR rate matrix | s⁻¹ |
| $\mathbf S$ | recombination source per unit $n_{\rm ion}$ | s⁻¹ |
| $\mathcal S, \mathcal F$ | slow $\{1s\}$ and fast (42 excited) subspaces | — |
| $\boldsymbol\Omega_{\rm QSS}$ | Schur complement; effective slow rate matrix | s⁻¹ |
| $Z_p$ | Saha–Boltzmann coefficient of level $p$ | cm³ |
| $b_1$ | ground-state departure coefficient | — |
| $r_p$ | $n_p/n_{1s}$ | — |
| $R$ | $n_3/n_4$, the observable | — |
| $\mathbf n^{(0)}, \mathbf n^{(1)}$ | recombination-fed, ground-fed channels | cm⁻³ |
| $f_p$ | ground-fed fraction of shell $p$ | — |
| $x$ | ground-density scale factor along the channel family | — |
| $\tau_{\rm relax}, \tau_{\rm QSS}$ | excited-manifold, ionisation-balance times | s |
| $M$ | $\tau_{\rm QSS}/\tau_{\rm relax}$ | — |
| $\varepsilon^{\rm QSS}, \varepsilon^{\rm CRE}$ | error against each reference | — |
| $\varepsilon_{\rm step}, \varepsilon_{\rm plateau}$ | endpoint and plateau errors | — |
| $\tau_{\rm drive}$ | duration of the driving event | s |
