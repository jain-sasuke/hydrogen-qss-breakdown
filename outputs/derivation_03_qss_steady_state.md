# Derivation 03 — The QSS Steady State (the target $\mathbf n^{ss}$)

**Quantity:** the quasi-steady-state excited populations $\mathbf n^{ss}$ — the target the
system relaxes *toward* — and why that target moves in time.

**Status:** Analytic derivation **complete and verified** by hand ($\dot{\mathbf n}=0 \Rightarrow
L\mathbf n^{ss}=-\mathbf b$). The numerical solve on the real grid and the confirmation of which
reservoirs sit in $\mathbf b$ (vs. tracked in $\mathbf n$) are **PENDING** the code audit of
`assemble_cr_matrix.py`.

**Grounded in:** Capitelli (2016) §6.5 (QSS = freeze excited-state derivatives, let reservoirs
evolve); Fujimoto, *Plasma Spectroscopy* (2004) §4 (QSS populations). **Not** grounded in the
thesis text or earlier dossier numbers — those are the things under test.

**Feeds thesis sections:** §3.2 (QSS as adiabatic elimination / Schur complement).

**Relation to neighbours:** Q3 defines *where* the system settles ($\mathbf n^{ss}$); Q4 defines
*how fast* it gets there ($\tau_{\text{relax}}$); Q5 asks whether it actually keeps up when the
target moves (the breakdown).

---

## 1. Core result

Start from the full time-dependent CR equation (the master equation the whole thesis solves):

$$\frac{d\mathbf n}{dt} = L\mathbf n + \mathbf b$$

Impose the quasi-steady-state condition — the excited-state time derivatives vanish:

$$\frac{d\mathbf n}{dt} = 0$$

This gives the QSS balance and its formal solution:

$$\boxed{L\,\mathbf n^{ss} + \mathbf b = 0 \quad\Longrightarrow\quad \mathbf n^{ss} = -L^{-1}\mathbf b}$$

Because the reservoirs feeding $\mathbf b$ (and the rates in $L$) depend on plasma conditions,
the target is in general **time-dependent**:

$$\boxed{\mathbf n^{ss}(t) = -L(t)^{-1}\,\mathbf b(t)}$$

$\mathbf n^{ss}$ is not a fixed point reached once — it is a **moving target** that the real
populations chase, always trailing slightly behind.

---

## 2. The physical picture (story-first)

**It is the Bodenstein steady-state approximation, applied to the whole excited manifold at
once.** For a reactive intermediate $A \to I \to P$, one sets $d[I]/dt = 0$ and *solves* for
$[I]_{ss} = (\text{production})/(\text{loss})$ — the intermediate stops being a differential
unknown and becomes algebraically slaved to whatever $[A]$ currently is.

QSS is the identical move for all excited levels simultaneously. Setting $\dot{\mathbf n}=0$
turns the excited populations from differential unknowns into algebraic functions of the
reservoirs (ground-state density, ion density, $n_e$, $T_e$) that live in $\mathbf b$. The
operator $-L^{-1}$ plays the role of "divide by the loss network" — but a *networked* division,
because every level's balance couples to every other's.

**Why this is the right approximation here:** Capitelli (§6.5) frames QSS as freezing the
excited-state derivatives to zero while the ground state and electron density relax in time,
"justified on the basis of the much shorter relaxation times of the excited states compared to
the ground state." That timescale separation is exactly the $M \gg 1$ condition quantified in
Q4. QSS is valid *because* the excited manifold settles far faster than the reservoirs move.

---

## 3. What goes in $L$ vs. $\mathbf b$ — the partition rule

Every term in the per-level balance
$\dot n_p = \sum_q[K_{qp}n_q - K_{pq}n_p] - S_p n_p + \alpha_p n_e n_{\text{ion}} + \sum_q A_{qp}n_q$
sorts into exactly one container by a single rule:

**Depends on a tracked population $n_q$ → goes in $L$ (multiplies $\mathbf n$).
Comes from a frozen reservoir outside the tracked manifold → goes in $\mathbf b$ (a constant
feed).**

| Process | Container | Why |
|---|---|---|
| Excitation / de-excitation between tracked levels | $L$ | $\propto$ tracked $n_q$ |
| Radiative decay between tracked levels | $L$ | $\propto$ tracked $n_q$ |
| Proton-impact $\ell$-mixing within shells | $L$ | $\propto$ tracked $n_q$ |
| Ionization *out* of a tracked level | $L$ (diagonal loss) | $\propto$ tracked $n_p$ |
| Recombination *from the ion pool* | $\mathbf b$ | $\propto n_{\text{ion}}n_e$, a frozen reservoir |
| Excitation *from a frozen ground state* (if ground is a reservoir) | $\mathbf b$ | $\propto$ reservoir density |

**Structural consequence:** $L$ is *linear* in the tracked populations; $\mathbf b$ is
*constant* given the reservoirs. That linear/constant split is what makes the equation an
inhomogeneous linear ODE with the clean solution "particular ($\mathbf n^{ss}$) + decaying
eigenmodes (Q4)."

**Recombination check (consistent with Q1):** recombination is $\propto n_{\text{ion}}n_e$,
outside the neutral manifold — a feed stream (the CSTR-feed analogy), so it belongs in
$\mathbf b$, never as a matrix element of $L$. This is exactly the term correctly deleted from
the 2-state toy in Q4.

**OPEN ITEM (code audit):** whether the ground state $n_1$ is *tracked* (inside $\mathbf n$) or
*frozen* (folded into $\mathbf b$) is a modelling choice that changes what $L$ and $\mathbf b$
contain — and therefore changes the eigenstructure of Q4. Must be read off
`assemble_cr_matrix.py`, not assumed.

---

## 4. Numerical note — never form $L^{-1}$

$\mathbf n^{ss} = -L^{-1}\mathbf b$ is the correct *statement*, but the inverse should **not** be
formed explicitly in code. Solve the linear system instead:

$$L\,\mathbf n^{ss} = -\mathbf b \qquad\text{(e.g. \texttt{np.linalg.solve}, or an LU factorization)}$$

Reason: $L$ is **stiff / ill-conditioned** — from Q1 its entries span ionization ($\sim10^4$
s$^{-1}$) to $\ell$-mixing ($\sim10^{12}$ s$^{-1}$), roughly eight orders of magnitude.
Explicitly inverting an ill-conditioned matrix amplifies round-off; a direct solve is faster and
numerically stable. On paper: $-L^{-1}\mathbf b$. In code: `solve(L, -b)`.

---

## 5. The bridge to Q5 — a moving target, chased with a lag

Subtract the (now time-dependent) target. With $\delta\mathbf n \equiv \mathbf n - \mathbf n^{ss}$,
differentiating $\mathbf n = \mathbf n^{ss} + \delta\mathbf n$ and using
$L\mathbf n^{ss}+\mathbf b=0$ gives the deviation equation with an extra forcing term that
survives *only because the target moves*:

$$\frac{d\,\delta\mathbf n}{dt} = L\,\delta\mathbf n \;-\; \dot{\mathbf n}^{ss}(t)$$

Read as a race:

- **Chaser:** $L\,\delta\mathbf n$ closes the gap at rate $\sim 1/\tau_{\text{relax}}$ (fast).
- **Runaway target:** $-\dot{\mathbf n}^{ss}$ is the forcing, at rate $\sim 1/\tau_{\text{QSS}}$
  (slow).

If the chaser is much faster ($\tau_{\text{relax}}\ll\tau_{\text{QSS}}$, i.e.
$M=\tau_{\text{QSS}}/\tau_{\text{relax}}\gg1$), the lag is negligible and
$\mathbf n(t)\approx\mathbf n^{ss}(t)$ — **QSS holds**. When the target moves comparably fast
($M\sim1$), the lag grows and QSS breaks. The QSS approximation *is* the assumption that this
lag is zero; the thesis measures when that assumption fails.

**The subtlety that is Q5 (flagged here, proved there):** "trailing behind" has two independent
causes — (1) the target moving fast relative to the chaser (a *rate* mismatch, measured by $M$),
and (2) the target jumping a long *distance* in one step (measured by the step mismatch $J$).
$M$ captures only #1. So $M\gg1$ everywhere does **not** guarantee small transient error — a
sudden large step in $T_e$ can leave the population far from the new target regardless of how
fast it then recovers. That $M\gg1$-yet-error-appears gap is the paper's central argument.

---

## 6. Checks

1. **Reduces to Bodenstein.** For a single intermediate, $L\to -(\text{loss rate})$ and
   $\mathbf b\to(\text{production})$, giving $n^{ss}=\text{production}/\text{loss}$. ✓
2. **Dimensions.** $[L]=\text{s}^{-1}$, $[\mathbf b]=\text{cm}^{-3}\text{s}^{-1}$, so
   $[\mathbf n^{ss}]=[L^{-1}\mathbf b]=\text{cm}^{-3}$ — a population density. ✓
3. **Existence.** $\mathbf n^{ss}=-L^{-1}\mathbf b$ requires $L$ invertible; with ionization
   present, $\det L\neq0$ (no zero eigenvalue — cf. Q4's conservation insight, where removing
   ionization sends one eigenvalue to zero and the inverse blows up). ✓
4. **Consistency with Q4.** The Q4 general solution
   $\mathbf n(t)=\mathbf n^{ss}+\sum_k c_k e^{\lambda_k t}\mathbf v_k$ has $\mathbf n^{ss}$ as its
   particular (steady) part; as $t\to\infty$ with fixed reservoirs, the eigenmodes decay and
   $\mathbf n\to\mathbf n^{ss}$. ✓

---

## 7. PENDING — next steps (code audit)

1. Read `assemble_cr_matrix.py`: confirm the $L$/$\mathbf b$ partition, and specifically whether
   the ground state is tracked or frozen.
2. Solve $L\mathbf n^{ss}=-\mathbf b$ on the real grid (direct solve, not inverse); sanity-check
   $\mathbf n^{ss}$ against coronal (low-$n_e$) and Saha-approach (high-$n_e$) limits.
3. Confirm $\mathbf n^{ss}>0$ componentwise (a physical population must be non-negative).

---

## 8. Defense one-liners

- *What is $\mathbf n^{ss}$?* The excited populations for which every level's gains balance its
  losses, so $\dot{\mathbf n}=0$: $\mathbf n^{ss}=-L^{-1}\mathbf b$. It is the Bodenstein
  steady-state approximation applied to the whole excited manifold at once.
- *Is it constant?* No — it tracks the reservoirs ($T_e,n_e$, ground, ions), so
  $\mathbf n^{ss}(t)=-L(t)^{-1}\mathbf b(t)$ is a moving target the real populations chase.
- *Why not invert $L$?* $L$ is stiff (≈8 orders of magnitude in its entries); a direct linear
  solve is stable and fast, an explicit inverse amplifies round-off.
- *Where does QSS come from mathematically?* Adiabatic elimination of the fast excited-state
  variables (Schur complement): freeze $\dot{\mathbf n}=0$, solve the algebra, valid when the
  excited manifold relaxes far faster than the reservoirs move ($M\gg1$).
