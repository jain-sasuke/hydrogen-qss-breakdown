# Derivation 04 — The Two Timescales (Eigenvalues of the CR Matrix)

**Quantity:** the two relaxation timescales $\tau_{\text{relax}}$ and $\tau_{\text{QSS}}$, and the
memory metric $M = \tau_{\text{QSS}}/\tau_{\text{relax}}$, extracted from the eigenspectrum of the
rate matrix $L$.

**Status:** **COMPLETE — Session A closed; exam Parts 2–3 taken 14 Jul.**
Analytic 2-state derivation verified (hand derivation, worked example, sign / units / limit
checks). §6A boundary-level physics corrected (arithmetic error + two overclaims). The
**~25 ns vs ~2 ns dispute is resolved**: live computation on the regenerated
`L_grid.npy` at benchmark point ($T_e{=}2.947$ eV, $n_e{=}1.389\times10^{14}$ cm$^{-3}$) gives
$\tau_{\text{relax}}=2.277$ ns, $\tau_{\text{QSS}}=22.73\,\mu$s, $M=9982$ — reproduced
independently on the student's own machine. The 25 ns value was an artifact of a stale
pre-March-2026 matrix carrying a spurious intermediate eigenmode (same root cause as the
retracted $-46\%$ Hα artifact). §9.3 resolves the Framing A vs B ambiguity and identifies the
eigenvectors. **§10 records the spectrum reframing (one gap, not three groups) and the exam
outcome.**

**Grounded in:** Fujimoto, *Plasma Spectroscopy* (2004), §4 (bucket model) and App. 4B
(boundary level, QSS validity); Capitelli (2016), §6.5 (QSS as freezing the excited-state
derivatives). **Not** grounded in the thesis text or earlier dossier numbers — those are the
things under test.

**Feeds thesis sections:** §3.4 (two timescales), §4 Gate E (timescale hierarchy).

---

## 1. Core results

General solution of $\dot{\mathbf n} = L\mathbf n$ as a sum of modes:
$$\mathbf n(t) = \sum_k c_k\, e^{\lambda_k t}\, \mathbf v_k,$$
where $\mathbf v_k$ (eigenvector) is a fixed *pattern* of populations and $\lambda_k$
(eigenvalue) is that pattern's decay rate.

Two-state eigenvalues in the physical limit $s_1, s_2 \ll \alpha, \beta$:

$$\boxed{\lambda_- \approx -(\alpha+\beta) \quad \text{(fast, relaxation)}}$$

$$\boxed{\lambda_+ \approx -\frac{\alpha s_2 + \beta s_1}{\alpha+\beta} \quad \text{(slow, QSS)}}$$

Timescales (the short time is relaxation, because relaxation is the *fast* process):

$$\boxed{\tau_{\text{relax}} = \frac{1}{|\lambda_-|} = \frac{1}{\alpha+\beta}}$$

$$\boxed{\tau_{\text{QSS}} = \frac{1}{|\lambda_+|} = \frac{\alpha+\beta}{\alpha s_2 + \beta s_1}}$$

Memory metric — a **dimensionless** ratio of two times:

$$\boxed{M = \frac{\tau_{\text{QSS}}}{\tau_{\text{relax}}} = \frac{|\lambda_-|}{|\lambda_+|} = \frac{(\alpha+\beta)^2}{\alpha s_2 + \beta s_1} \approx \frac{\beta}{s_1} = \frac{\text{radiative rate}}{\text{ionization rate}} \gg 1}$$

**Symbols** (all rates, units s$^{-1}$): $\alpha = C_{12}n_e$ (excitation $1\!\to\!2$);
$\beta = C_{21}n_e + A_{21}$ (de-excitation + radiative decay $2\!\to\!1$);
$s_1 = S_{1+}n_e$, $s_2 = S_{2+}n_e$ (ionization out of states 1 and 2).

---

## 2. The physical picture (story-first, fresh-grad readable)

**It is the chemical-kinetics steady-state approximation in disguise.** For a reactive
intermediate, $A \xrightarrow{} I \xrightarrow{} P$, one sets $d[I]/dt \approx 0$ when $I$ is
consumed much faster than it is produced — it never piles up, it jumps to a small steady value
and then just *tracks* the slowly-draining $A$. In the plasma the excited states play the role
of $I$ (fast, short-lived), and ionization is the slow $A \to P$ drain. QSS is the same
approximation: freeze the excited-state derivatives, let only the ground state and $n_e$ evolve.

**Three groups of timescales** (the full system; the 2-state toy captures only the middle and
slow ones):

| Group | Process | Timescale |
|---|---|---|
| Fastest | proton-impact $\ell$-mixing *within* a shell | picoseconds |
| Fast | excited states relaxing to their QSS pattern ($\tau_{\text{relax}}$) | nanoseconds |
| Slow | ionization balance of the whole gas ($\tau_{\text{QSS}}$) | µs – ms |

**Meaning of $M$.** $M \gg 1$ means the excited states settle *far* faster than the ionization
balance moves, so at any instant they look already-equilibrated to the current ground-state
population. That is exactly the condition under which QSS is self-consistent. $M$ is the ratio
across the big middle gap (fast vs slow); the picosecond group is irrelevant to it.

---

## 3. The 2-state derivation

**3.1 Rate equations.** Two bound states, 1 (lower) and 2 (upper):

$$\dot{n}_2 = C_{12}n_1 n_e - C_{21}n_2 n_e - A_{21}n_2 - S_{2+}n_2 n_e$$

$$\dot{n}_1 = -C_{12}n_1 n_e + C_{21}n_2 n_e + A_{21}n_2 - S_{1+}n_1 n_e$$

*Key modelling point:* recombination ($\propto n_+ n_e$) is **not** included — it lives outside the neutral manifold and enters as a feed term in $b$ (a CSTR feed), never as an element of $L$.

**3.2 Matrix form.** With $\alpha, \beta, s_1, s_2$ as defined above:

$$L = \begin{pmatrix} -\alpha - s_1 & \beta \\ \alpha & -\beta - s_2 \end{pmatrix}$$
*Checks:* diagonal negative (states lose population), off-diagonal positive (states gain from
each other); column sums $= (-s_1, -s_2)$ — non-zero only because ionization permanently
removes atoms. Without ionization the columns would sum to zero (population conserved).

**3.3 Characteristic equation.** $\det(L-\lambda I)=0$ gives

$$\lambda^2 + (\alpha+\beta+s_1+s_2)\,\lambda + (s_1 s_2 + \alpha s_2 + \beta s_1) = 0$$

where $-\operatorname{tr}L = \alpha+\beta+s_1+s_2$ and $\det L = s_1 s_2 + \alpha s_2 + \beta s_1$. For any matrix, eigenvalues sum to the trace and multiply to the determinant.

**3.4 Conservation insight.** If $s_1=s_2=0$, the constant term ($=\det L$) vanishes, so one
root is exactly $\lambda = 0$. A zero eigenvalue is a mode that never decays — conserved total
population. Ionization is precisely what tips that root slightly negative, turning "conserved
forever" into "slowly leaks." **The slow eigenvalue is born from the ionization terms.**

**3.5 Solving (perturbation).** With $A \equiv \alpha+\beta+s_1+s_2$ and $B \equiv s_1 s_2 + \alpha s_2 + \beta s_1$, and using $A^2 \gg B$:

$$\sqrt{A^2 - 4B} = A\sqrt{1 - 4B/A^2} \approx A - \frac{2B}{A}$$

$$\lambda_+ = \frac{-A + (A - 2B/A)}{2} = -\frac{B}{A} \approx -\frac{\alpha s_2 + \beta s_1}{\alpha+\beta}$$

$$\lambda_- = \frac{-A - (A - 2B/A)}{2} \approx -A \approx -(\alpha+\beta)$$

**3.6 Modes (why $\mathbf n(t)=\sum_k c_k e^{\lambda_k t}\mathbf v_k$).** Put the eigenvectors as
columns of $V$ and the eigenvalues on the diagonal of $\Lambda$, so $V^{-1}LV = \Lambda$. In
modal coordinates $\mathbf a = V^{-1}\mathbf n$ the system **decouples**:
$\dot{\mathbf a} = \Lambda \mathbf a$, i.e. $\dot a_k = \lambda_k a_k$ with solution
$a_k(t)=a_k(0)e^{\lambda_k t}$. Transforming back gives the modal sum. Each eigenvalue is a
mode's personal decay rate: $\lambda<0$ decays (time constant $1/|\lambda|$), $\lambda=0$ is
sustained (conserved), $\lambda>0$ would blow up (forbidden — why a valid $L$ has all
$\lambda \le 0$).

---

## 4. Worked example (toy numbers chosen for clean arithmetic, not physical)

$\alpha=1,\ \beta=10,\ s_1=0.1,\ s_2=0.1$

$$L = \begin{pmatrix} -1.1 & 10 \\ 1 & -10.1 \end{pmatrix}$$

- Characteristic eq.: $\lambda^2 + 11.2\lambda + 1.11 = 0$, discriminant $=121$,
  $\lambda = \tfrac{-11.2 \pm 11}{2} \Rightarrow \lambda_- = -11.1,\ \lambda_+ = -0.1.$
- Eigenvectors: $\mathbf v_- = (1,-1)$ (fast), $\mathbf v_+ = (10,1)$ (slow).
- $\tau_{\text{relax}} = 1/11.1 \approx 0.09$; $\tau_{\text{QSS}} = 1/0.1 = 10$;
  $M = 10/0.09 \approx 110 \approx |\lambda_-|/|\lambda_+| = 111.$

**Physical reading of the eigenvectors:**
- Fast mode $(1,-1)$: components opposite-sign, sum to 0 → carries *no net population*, pure
  sloshing between the two levels. This is excited-state **relaxation**.
- Slow mode $(10,1)$: both same sign → carries the population, drains as a whole in the fixed
  ratio $10:1$. That ratio **is the QSS population ratio**; the draining is ionization.

**Two-stage relaxation.** Starting from $\mathbf{n}(0)=(1,0)$:

$$\mathbf{n}(t) = \frac{1}{11}e^{-0.1t}\begin{pmatrix}10\\1\end{pmatrix} + \frac{1}{11}e^{-11.1t}\begin{pmatrix}1\\-1\end{pmatrix}$$

Stage 1 ($t \lesssim 1/11.1$): the fast term dies, populations snap to the $10:1$ QSS ratio.
Stage 2 ($t \gtrsim 1/11.1$): only the slow term survives; the gas drains via ionization while
*holding* the $10:1$ ratio. This reproduces the Chapter-3 two-stage picture from first
principles.

---

## 5. Checks passed

1. **Signs.** Sum of roots $<0$ and product $>0$ ⇒ both eigenvalues negative ⇒ both modes
   decay, nothing grows. Stable relaxation. ✓
2. **Units.** $M$ must be dimensionless. $(\alpha+\beta)^2/(\alpha s_2+\beta s_1)$ has units
   $(\text{s}^{-1})^2/(\text{s}^{-1})^2 = 1$. ✓ *(Guards against the slip
   $\tau_{\text{QSS}}\cdot\tau_{\text{relax}}$, which has units of s$^2$ and cannot be $M$.)*
3. **Limit.** $s_1=s_2=0 \Rightarrow \lambda_+ \to 0$: the conserved-population mode. ✓
4. **Magnitude / reliable source.** $\tau_{\text{relax}} = 1/(\alpha+\beta) \approx 1/\beta
   \approx 1/A_{21} \approx 1.6$ ns, matching Fujimoto's bucket-model $\tau_{\text{rel}}(n=2)
   \approx 1.4$ ns. The fast clock *is* the Lyman-$\alpha$ radiative lifetime. ✓
5. **Label discipline.** Relaxation = fast process = short time = *large* $|\lambda|$.
   Swapping the labels would invert $M$ and (falsely) say QSS fails everywhere. ✓

---

## 6. Honest limits of the 2-state toy

- The toy uses $n=2$, whose relaxation is the lightning-fast Lyman-$\alpha$. The real system
  relaxes at the rate of its **slowest excited level** (Fujimoto's *boundary level*, a higher-$n$
  state that radiates more slowly), so the real $\tau_{\text{relax}}$ is **longer** and the real
  $M$ is **more modest** (hundreds–thousands, not the millions $\beta/s_1$ would give for $n=2$).
- The toy gives the right *structure* (two-mode split, QSS-ratio slow mode, conservation in the
  zero-ionization limit) and the right *direction* ($M \gg 1$). It does **not** give the real
  number — that needs the full $L$.

---

## 6A. What controls $\tau_{\text{relax}}$ in the real 43-state system
*(Grounded in: Fujimoto App. 4B; Griem §7.6; van der Mullen 1990)*

### The key principle: the slowest level sets the pace

When 43 levels are coupled, the whole manifold can only finish equilibrating as fast as its
**slowest** member. Fujimoto (App. 4B) states this directly: "the relaxation of these
populations as a whole is given by the relaxation time of a certain critical level, which has
the **longest** relaxation time among the excited levels." This critical level is called
**Griem's boundary level**, $p_G$.

Mathematically, the slowest excited level *is* the second eigenvalue: $\tau_{\text{relax}} =
1/|\lambda_1|$. The eigenvector of $\lambda_1$ tells you which level $p_G$ is — its
population weight is concentrated there.

**This is why the toy is wrong in the dangerous direction.** The toy uses $n=2$, which
relaxes fastest (Lyman-$\alpha$, $\sim$1.4 ns, Fujimoto App. 4B). The manifold is held up by
a slower, higher-$n$ level. The real $\tau_{\text{relax}}$ is always *longer* than the toy says.

---

### Why higher-$n$ levels are slower: radiative lifetime scaling

Radiative transition rates fall steeply with $n$. From Fujimoto's approximation (eq. 4.14a)
and the standard hydrogen Einstein-$A$ scaling:

$$A(n) \propto n^{-4.5}$$

$$\tau_{\text{rad}}(n) \propto n^{4.5}$$

Concretely (hydrogen, $\ell$-averaged):

| Level $n$ | $\tau_{\text{rad}}$ (approximate) |
|---|---|
| 2 | $\sim$ 1.6 ns (Lyman-$\alpha$ dominated) |
| 3 | $\sim$ 15 ns |
| 4 | $\sim$ 70 ns |
| 5 | $\sim$ 230 ns |

So the "slowest level" is not the highest-$n$ state either — it is wherever the boundary sits.

---

### Griem's boundary: where radiation and collisions are equally fast

**Correction applied:** an earlier draft of this note used a one-channel comparison
($n_e C(p,p{+}1) \approx A(p)$) and made an arithmetic error evaluating it. Both are fixed here.
The boundary must compare **total** depopulation, not one channel — van der Mullen (1990), eq.
7.21, and Griem §7.6 both define it this way. For level $p$:

$$D_p^{\text{rad}} = \sum_{q<p} A_{p\to q} \qquad\qquad D_p^{\text{coll}} = n_e\Big[\sum_{q\neq p} C_{p\to q} + S_p^{\text{ion}}\Big]$$

(radiative depopulation; collisional depopulation, including de-excitation, excitation to other
bound levels, *and* ionization). The boundary ratio is

$$R_G(p) = \frac{D_p^{\text{coll}}}{D_p^{\text{rad}}}$$

with $R_G(p)\sim1$ a simple crossing, or Griem's stricter LTE-type criterion $R_G(p)\sim10$.
Below the crossing: radiation dominates (levels are radiative/coronal-type). Above it:
collisions dominate (levels are collisional/near-saturation).

**The analytic estimate — and why it's a rough guide, not a number to trust.** Two published
closed-form estimates exist for the critical $n_{cr}$ where $R_G\sim10$:

$$n_{cr} \approx 141\,N_e^{-2/17} \ \text{(Griem 1963, eq. 7.77)}, \qquad n_{cr} \approx 99\,N_e^{-2/17} \ \text{(van der Mullen 1990)}$$

**Corrected arithmetic** (an earlier draft had an error here): at benchmark point $N_e=10^{14}$
cm$^{-3}$,

$$(10^{14})^{-2/17} = 10^{-1.647} \approx 0.0225 \quad\Rightarrow\quad n_{cr} \approx 99\times0.0225 \approx 2.2 \ \text{(van der Mullen)}, \ \approx 3.2\ \text{(Griem)}$$

So the boundary sits at $n\approx2$–3.5, **not** $n\approx17$ as an earlier draft claimed — that
was an arithmetic slip, and the physical lean it produced ("boundary is deep in the bundled
shells, so 25 ns is favored") does not survive the correction and is **retracted**.

**Griem's own caveat, which matters more than the number.** Griem states explicitly that for
hydrogen ($z=1$) this formula "gives a substantial underestimate," is "particularly crude," and
"do[es] not agree very well with CR model calculations" — CR models typically give $n_{cr}$
values ~20% larger. So treat $n_{cr}\approx2$–3.5 as a rough guide, not a target to match.
**Neither 2 ns nor 25 ns is favored by this estimate.** The honest position is: this crude
formula cannot adjudicate the dispute either way.

**What actually locates the slow mode.** The eigenvector of $\lambda_1$ shows *where the slow
relaxation mode is localized* — that is the real diagnostic, to be *compared against* $R_G(p)$,
not derived from it. (An earlier draft overstated this as "the eigenvector tells you $p_G$";
the two are separate diagnostics that should agree, not the same object by definition.)

---

### Why $\tau_{\text{relax}}$ changes with density and temperature — direction only, not a fixed power law

**Correction applied:** an earlier draft claimed $\tau_{\text{relax}}\propto n_e^{-1}$ as a
"purely collisional scaling." This overclaims. If the controlling level itself shifts with
density (boundary descending as $n_{cr}\propto n_e^{-2/17}$) while its radiative floor scales as
$\tau_{\text{rad}}(n)\propto n^{4.5}$, combining the two gives a *different*, weaker net power
($\tau_{\text{rad}}(n_{cr})\propto n_e^{-9/17}$, not $n_e^{-1}$) — and that combination is itself
only as trustworthy as the crude $n_{cr}$ formula above. **The correct statement is directional,
not a fixed exponent:**

- **Density:** collisional depopulation rates scale $\propto n_e$, so $\tau_{\text{relax}}$ is
  expected to *decrease* with $n_e$ when the slow mode is collision-controlled — but the actual
  scaling must be *measured* from $\lambda_1(n_e)$ across your grid, not assumed analytically.
- **Temperature:** $T_e$ enters the collisional rate coefficients through the Maxwellian average,
  but the sign of the trend is not universal — de-excitation coefficients roughly follow a
  $T_e^{-1/2}$-type trend, while excitation and ionization coefficients have threshold/exponential
  structure that can push the other way. **Don't assume "higher $T_e$ → shorter
  $\tau_{\text{relax}}$" — measure it.**
- At very high $T_e$ ($\gtrsim 20\,z^2$ eV, van der Mullen's CRC regime) the whole manifold
  approaches LTE even at low density and the boundary-level distinction disappears. Your range
  ($T_e=1$–10 eV) is well below this, so the boundary-level structure is relevant — but its
  precise scaling is an empirical question for the eigenvalue sweep, not a derived law.

**A caution on "below $p_G$ = slow to relax."** An earlier draft stated this as if the boundary
directly ranks relaxation *speed*. That's not quite right: $R_G(p)$ classifies which
*mechanism* depopulates level $p$ (radiative vs. collisional), not how fast the whole manifold
equilibrates. Fujimoto's transient examples show low levels (strongly radiative) can reach their
final value *first* — i.e. fast — while intermediate levels lag. The correct statement is:
**below $p_G$, radiative depopulation dominates; above $p_G$, collisional depopulation
dominates. The identity of the slowest-relaxing level must come from $\tau_{\rm tr}(p)$ or the
slow eigenmode itself — not from the $R_G$ classification alone.**

---

### RESOLVED: ~2.28 ns is the real $\tau_{\text{relax}}$; ~25 ns was a stale-matrix artifact

A documented prior run (`CHAPTER5_CORRECTIONS_REPORT.md`, Session 10 May 2026) computed
$\mathrm{eigvals}(L\_{\text{grid}}[23,5])$ at benchmark point directly:

$$\lambda_1 = 4.387\times10^8\ \text{s}^{-1} \;\Rightarrow\; \tau_{\text{relax}} = 2.28\ \text{ns}$$

with a second fast eigenvalue $\lambda_2 = 5.381\times10^8\ \text{s}^{-1}$ ($\tau=1.86$ ns) close
behind — a clean two-timescale spectrum with **no** intermediate mode. The report's own
diagnosis: an older (pre-March-2026) matrix carried a **spurious intermediate eigenmode near
25 ns**, which does not exist in the corrected matrix. That stale mode is also the documented
root cause of the retracted $-46\%$ Hα artifact (both trace to the same old file) — this is not
a coincidence, it's the same bug surfacing in two different derivations.

**This corroborates, rather than contradicts, the corrected boundary-level physics above.**
Once the $n_{cr}\approx2$–3.5 arithmetic was fixed, the boundary-level argument already predicted
a *low*, collision-dominated boundary — meaning the slowest relaxing level should be low-$n$ and
therefore fast (few ns, not tens of ns). The independently-computed eigenvalues (2.28 ns, 1.86
ns) land exactly in that range. Textbook boundary physics and a documented code run agree.

**RESOLVED (Session A, verified live from `L_grid.npy`).** The eigenvectors have now been
computed and the picture is confirmed and sharpened — see §9.3 below. The slow mode $\lambda_1$
*is* localized at low $n$ (1s against 2p, 3d, 3p, 2s), consistent with the corrected
boundary-level prediction.

---

### How to calculate $\tau_{\text{relax}}$ — three levels

| Method | What you compute | Rigor |
|---|---|---|
| **Exact** | $1/|\lambda_1|$ from $\mathrm{eig}(L\_{\text{grid}})$ | Full — this is the definition |
| **Physical estimate** | $\tau_{\text{rad}}(p_G)$ at boundary from Griem/van der Mullen | Order of magnitude |
| **Per-level bucket** | $1/[\sum_q A(p_G,q) + n_e\sum_q C(p_G,q)]$ (Fujimoto eq. 4.14) | Good approximation |

The eigenvector of $\lambda_1$ shows *where the slow relaxation mode is localized* — this
should be compared against the $R_G(p)$ crossing, not treated as automatically defining $p_G$
(they are two separate diagnostics that ought to agree, not one and the same by definition). If
the eigenvector weight and the $R_G(p)\sim1$ crossing land in the same region, that is good
evidence the slow mode is associated with the Griem/Fujimoto boundary.

---

## 7. PENDING — next steps (code audit, step 3 of the rhythm)

1. ~~Compute $\mathrm{eig}(L\_{\text{grid}})$...~~ **DONE — see §9.**
2. ~~Eigenvector of $\lambda_1$...~~ **DONE (Session A) — see §9.3.**
3. **Still open:** the $n_e$ and $T_e$ scaling of $\tau_{\text{relax}}$, measured directly across
   the grid (§6A already warns against assuming a fixed power law). Partial data: at $T_e=2.95$
   eV, $\tau_{\text{relax}}$ runs 32.3 ns ($n_e=10^{12}$) → 2.28 ns ($1.4\times10^{14}$) → 1.27 ns
   ($10^{15}$), i.e. roughly $n_e^{-0.47}$ overall, **not** a clean $n_e^{-1}$ power law.
4. Stub: a short script reproducing the toy eigenvalues $(-0.1, -11.1)$ to machine precision —
   independent of the real-grid question, still a clean sanity check to keep.
5. **$\tau_{\text{QSS}}$ double-definition — PARTIALLY RESOLVED (Session A).** The
   *eigenvalue-based* ambiguity (full-matrix $\lambda_1$ vs excited-block least-negative) is
   settled in §9.3: they agree to <0.35%. **What remains open** is the different question of
   eigenvalue-$\tau_{\text{QSS}}$ ($1/|\lambda_0|$, the intrinsic ionization drain) versus
   *target-motion* $\tau_{\text{QSS}}$ ($\|\mathbf n^{ss}\|/\|\dot{\mathbf n}^{ss}\|$, set by
   external driving). These diverge whenever the plasma is driven faster than its own ionization
   timescale — e.g. an ELM crash (100 µs) versus $\tau_{\text{QSS}}=22.7$ µs. **This is a Q5
   item**, not a Q4 defect.

---

## 9. CONFIRMED — real numbers from a documented prior code run

**Source:** `CHAPTER5_CORRECTIONS_REPORT.md` (Session 10, May 2026), reporting
`np.linalg.eigvals(L_grid[23, 5])` at benchmark point ($T_e=2.947$ eV, $n_e=1.389\times10^{14}$
cm$^{-3}$, matrix indices from the 50×8 $(T_e,n_e)$ grid defined in `assemble_cr_matrix.py`).
**This is a reported result, not independently re-run in this session** — the underlying
`L_grid.npy` and its upstream dependencies (`K_exc_full.npy`, `A_resolved.npy`, etc.) are not
present in this project, so this note cannot verify it from scratch. Treat it as strong
documented evidence, not as self-verified.

$$\lambda_0 = 4.403\times10^4\ \text{s}^{-1} \;\Rightarrow\; \tau_{\text{QSS}} = 22.71\,\mu\text{s}$$

$$\lambda_1 = 4.387\times10^8\ \text{s}^{-1} \;\Rightarrow\; \tau_{\text{relax}} = 2.28\ \text{ns}$$

$$\lambda_2 = 5.381\times10^8\ \text{s}^{-1} \;\Rightarrow\; \tau = 1.86\ \text{ns} \ \text{(next-fastest mode)}$$

$$M = \tau_{\text{QSS}}/\tau_{\text{relax}} = 9963$$

**Which $\tau_{\text{QSS}}$ definition this uses:** $\lambda_0$ here is the least-negative
eigenvalue of the full 43-state $L$ matrix itself (ground state + excited block + ionization
loss, no time-dependence) — this is **definition (A)** from §7 item 5 (the reservoir
eigenvalue), *not* the target-motion definition (B). If the thesis reports $M$ elsewhere using
definition (B), the two must be reconciled or explicitly distinguished — still an open item.

**Root cause of the old 25 ns value:** a pre-March-2026 version of `L_grid.npy` carried a
spurious intermediate eigenmode near 25 ns that the corrected matrix does not have — documented
in the same report as the mechanism behind the now-retracted $-46\%$ Hα artifact. Both trace to
the same stale file; this is one bug with two symptoms, not two separate errors.

**Consistency check against §6A boundary physics.** The corrected Griem-boundary estimate
($n_{cr}\approx2$–3.5, from van der Mullen 1990 / Griem 1963, corrected arithmetic) predicted a
*low*, collision-dominated boundary — hence a fast (few-ns) slowest excited level. The
independently-sourced $\lambda_1,\lambda_2$ (2.28 ns, 1.86 ns) fall exactly in that range.
Textbook boundary physics and this documented code run corroborate each other. **This does not
yet confirm *which* level the mode lives on** — that requires the eigenvector, still pending.

**One documented red flag to carry forward, not swept under the rug:** the same report notes
$M_{\min}=1$ at low-$T_e$/low-$n_e$ is *suspected to be a code bug* — `qss_analysis.py` filters
`eigs < -1.0` and can silently discard the true slow eigenvalue at very low rates, returning the
wrong one instead. This is exactly the kind of hostile-physics-test flag this dossier is meant
to catch; it should be checked, not assumed fixed.

---

## 8. Defense one-liners

- *What are the two timescales?* Both come from the eigenspectrum of $L$.
  $\tau_{\text{QSS}} = 1/|\lambda_0|$ where $\lambda_0$ is the **least-negative** eigenvalue of
  the full 43×43 $L$; $\tau_{\text{relax}} = 1/|\lambda_1|$ where $\lambda_1$ is the
  **second-least-negative**. Equivalently (and verified numerically, §9.3),
  $\tau_{\text{relax}}$ is the least-negative eigenvalue of the 42×42 excited-state block with
  the ground state removed — the two definitions agree to <0.35% everywhere on the grid,
  because $\lambda_0$ is a pure ground-state mode. $\tau_{\text{relax}}$ is a *collective* mode
  of the coupled levels, not any single radiative lifetime.
- *Why is $M$ so large (in the 2-state toy)?* There, $M$ reduces to essentially the radiative
  rate over the ionization rate. In the real 43-state system, a documented eigenvalue computation
  gives $\tau_{\text{relax}}=2.28$ ns, $\tau_{\text{QSS}}=22.7\,\mu$s, $M=9963$ — smaller than the
  $n{=}2$ toy estimate would suggest (as expected, since the toy uses the fastest level), but
  still $\gg1$.
- *Does large $M$ prove QSS is fine?* No — it is *necessary*, not *sufficient*. $M\gg1$
  guarantees smooth tracking of a slowly-moving target; it says nothing about the size of a
  finite step error after a sudden change in plasma conditions. That is $J$ and $P_{\text{slow}}$
  (Q5), a separate axis from $M$.
- *Why not just use $1/A_{21}$ for $\tau_{\text{relax}}$?* Because $\tau_{\text{relax}}$ is the
  slowest collective relaxation time of the full excited-state CR matrix, not the lifetime of
  any single transition. $1/A_{21}$ is a lower bound on the fast end of the spectrum, not the
  number that controls how long the manifold as a whole takes to settle.
- *What is the Griem/Fujimoto boundary, precisely?* A depopulation classification,
  $R_G(p)=D_p^{\text{coll}}/D_p^{\text{rad}}$, comparing *total* collisional depopulation
  (de-excitation + excitation + ionization) against total radiative depopulation for level $p$ —
  not the full CR gain/loss balance. It is a rough physical guide (unreliable to better than
  ~20–30% for hydrogen, by Griem's own admission); the actual controlling level is identified
  from the eigenstructure of $L$, then checked against $R_G(p)$ for physical interpretation.

---

## 9.3 Session A — eigenvector identification and framing reconciliation

**Computed live from the post-ℓ-mixing-fix `L_grid.npy` at benchmark point
($T_e=2.947$ eV, $n_e=1.389\times10^{14}$ cm⁻³).**

### The three slowest modes and where they live

| Mode | $\lambda$ [s⁻¹] | $\tau$ | Localization (normalized eigenvector) | Ground-state weight |
|---|---|---|---|---|
| $\lambda_0$ | $-4.400\times10^{4}$ | 22.73 µs | 1s = +1.000; all others $<10^{-3}$ | **1.0000** |
| $\lambda_1$ | $-4.392\times10^{8}$ | 2.277 ns | 1s = +1.000, 2p = −0.613, 3d = −0.511, 3p = −0.307, 2s = −0.205 | 0.557 |
| $\lambda_2$ | $-5.399\times10^{8}$ | 1.852 ns | 1s = −1.000, 2p = +0.898, 2s = +0.300, 3d = −0.225 | 0.508 |

### What this settles

**1. $\lambda_0$ is a pure ground-state mode.** Weight fraction 1.0000 — the eigenvector is
entirely the ground state, with every excited component below $10^{-3}$. Physically: the neutral
population, essentially all of which sits in 1s, draining away by ionization. The rate is set by
the ground-state ionization rate, which is the smallest-magnitude diagonal in the whole matrix
($L[0,0]=-1.47\times10^5$ s⁻¹) because ionizing from 13.6 eV at $T_e\approx3$ eV is slow.

**2. $\lambda_1$ is the genuine excited-state relaxation mode.** Note the **opposite signs**:
ground state (+1.000) against the excited manifold (2p, 3d, 3p, 2s all negative). This is
population sloshing *between* ground and excited states — exactly what "excited-state
relaxation" means. It is **not** a single-level lifetime: four levels contribute at the >0.2
level. This confirms §11's claim that $\tau_{\text{relax}}\neq1/A_{21}$ — it is a collective mode.

**3. The localization is low-$n$ (2p, 3d, 3p, 2s), confirming the corrected boundary-level
prediction.** §6A's corrected arithmetic ($n_{cr}\approx2$–3.5, after fixing the $p_G\approx17$
error) predicted a *low*, collision-dominated boundary → low-$n$ slow mode → few-ns relaxation.
The eigenvector confirms it directly. **The earlier "plausible inference" is now a verified
fact.**

### Framing A vs Framing B — reconciled

Two definitions of $\tau_{\text{relax}}$ were in circulation:

- **Framing A (full matrix):** $\tau_{\text{relax}}=1/|\lambda_1|$, the second-least-negative
  eigenvalue of the full 43×43 $L$.
- **Framing B (excited block):** delete the ground-state row and column; take the least-negative
  eigenvalue of the remaining 42×42 block. This matches the textbook QSS framing (Fujimoto,
  Capitelli): freeze the ground state as a reservoir, ask how fast the excited manifold settles.

**Numerical test across the grid** ($T_e\in[1,10]$ eV, $n_e\in[10^{12},10^{15}]$ cm⁻³):

| $T_e$ [eV] | $n_e$ [cm⁻³] | A: $\tau_{\text{relax}}$ | B: $\tau_{\text{relax}}$ | ratio B/A |
|---|---|---|---|---|
| 1.00 | $10^{12}$ | 38.879 ns | 38.879 ns | 1.000000 |
| 1.00 | $1.39\times10^{14}$ | 3.8185 ns | 3.8185 ns | 1.000000 |
| 2.95 | $1.39\times10^{14}$ | 2.2769 ns | 2.2771 ns | 1.000098 |
| 10.00 | $1.39\times10^{14}$ | 1.8748 ns | 1.8784 ns | 1.001944 |
| 10.00 | $10^{15}$ | 0.8687 ns | 0.8717 ns | 1.003473 |

**Worst disagreement anywhere on the sampled grid: 0.35%** (at $T_e=10$ eV, $n_e=10^{15}$).

**Why they agree:** because $\lambda_0$ is a *pure* ground-state mode (weight 1.0000), deleting
the ground state removes exactly that mode and nothing else — leaving $\lambda_1$ as the
excited block's slowest. The residual 0.35% discrepancy at high $T_e$/high $n_e$ arises because
$\lambda_1$ itself retains ~56% ground-state weight, so removing the ground state perturbs it
slightly.

**Decision:** use **Framing A** ($\lambda_1$ of the full matrix) as the thesis definition — it
requires no arbitrary partitioning and is what the code computes. State in Ch. 3 that it agrees
with the textbook excited-block definition to <0.35% across the operating range, with the
above table as evidence. This turns a potential examiner question into a demonstrated
robustness result.

### One caution to carry forward

$\lambda_1$ (2.277 ns) and $\lambda_2$ (1.852 ns) are **close** — only a factor 1.23 apart.
This is *not* a clean spectral gap between "the" relaxation mode and everything else. Any
statement of the form "the system relaxes with a single time constant $\tau_{\text{relax}}$" is
an approximation; in reality two comparable modes contribute to Stage-1 relaxation. **This
near-degeneracy is a likely contributor to the $\varepsilon_{\text{res}}>\varepsilon_{\text{step}}$
transient growth observed in the step-response data — a Q5 item.**

---

## 10. Spectrum reframing and exam outcome (14 Jul 2026)

### 10.1 The correction: one gap, not three groups

Earlier drafts of this note described **three timescale groups** — picosecond
ℓ-mixing, nanosecond relaxation, microsecond ionisation. **That framing is not
supported by the spectrum and is retracted.**

Full spectrum at benchmark point, consecutive timescale ratios
(`verify_timescales.py` Test 2):

| Between | Ratio |
|---|---|
| $\tau_0$ and $\tau_1$ | **9981.9** |
| $\tau_1$ and $\tau_2$ | 1.23 |
| $\tau_2$ and $\tau_3$ | 2.80 |
| $\tau_3$ and $\tau_4$ | 2.71 |
| … all remaining pairs … | 1.01 – 1.93 |

**Exactly one gap exceeds 10×.** Below $\lambda_1$ the spectrum is a smooth
continuum running from 2.28 ns down to 0.068 ps — four decades with no internal
break. The "picosecond" and "nanosecond" groups are one distribution, not two.

**Correct structure:**

$$\underbrace{\tau_0 = 22.7\ \mu\mathrm{s}}_{\text{ionisation mode, alone}} \quad\Big|\quad \underbrace{2.28\ \mathrm{ns}\ \longrightarrow\ 0.068\ \mathrm{ps}}_{\text{quasi-continuum, 42 modes}}$$

### 10.2 What is NOT retracted

The **two-timescale separation itself is confirmed**, by two independent routes:

1. **Spectral:** the single gap of $\sim10^4$ between $\lambda_0$ and $\lambda_1$.
2. **Dynamical, independent of eigenvalues:** direct integration by matrix
   exponential (`verify_timescales.py` Test 5) shows the error norm dropping,
   then sitting on a **plateau spanning roughly three decades in time**
   (~1 ns to ~1 µs), then dropping again. A plateau of that width is the
   signature of genuine timescale separation; a continuum would show a smooth
   monotone decline.

So the headline result — $M = 9982$, one dominant gap, clean two-stage
relaxation — **stands**. What changes is the description of the *internal
structure of the fast group*.

### 10.3 The defensible sentence for the thesis

> The CR spectrum exhibits one dominant gap, of order $10^4$, separating the
> ionisation mode from a quasi-continuum of 42 fast modes. $\tau_{\text{relax}}$
> is the slowest member of that continuum rather than an isolated mode; the modes
> immediately below it lie within a factor of two, so Stage-1 relaxation is
> genuinely multi-modal even though the two-stage separation is clean.

This is **stronger** than the earlier claim because it survives an examiner
plotting the eigenvalues.

### 10.4 Why the near-degeneracy is a mechanism, not an embarrassment

$\tau_1 = 2.277$ ns and $\tau_2 = 1.852$ ns differ by only 1.23. From the
transient-growth analysis (`derivation_04c`),

$$t_{\max} = \frac{\ln(|\lambda_2|/|\lambda_1|)}{|\lambda_2| - |\lambda_1|} = \frac{\ln 1.23}{1.01\times10^{8}\ \mathrm{s^{-1}}} \approx 2.05\ \mathrm{ns}$$

matching the observed transient bump at 1–4.6 ns. **Closely spaced rates produce
a broad, late transient rather than a sharp spike** — so the multi-modal
structure of Stage 1 directly explains the transient behaviour that Q5 depends
on. The near-degeneracy and the transient growth are the same fact viewed twice.

### 10.5 Two results consolidated from this session

**(a) $\tau_{\text{relax}}$ density scaling — reinterpretation, not just a number
change.** Measured slopes $d\ln\tau_{\text{relax}}/d\ln n_e$: $-0.456$ ($T_e=1$
eV) through $-0.476$ (2.95 eV) to $-0.529$ (10 eV). A fixed, purely collisional
bottleneck would give $-1.00$; a purely radiative one would give $0$.

The measured $\approx-1/2$, and its systematic drift with $T_e$, is **not** a
fixed level with a 50/50 loss split. It is a **moving bottleneck**: as $n_e$
rises, (i) collisional rates increase $\propto n_e$, and (ii) the controlling
mode migrates to lower $n$ (5G → 4F → 3D → 2P, §9.4), where radiative rates are
faster. The two effects have different density dependences and partly offset.

Thesis sentence: *"$\tau_{\text{relax}}$ scales as roughly $n_e^{-1/2}$ rather
than $n_e^{-1}$ because the relaxation bottleneck is not a fixed level: as
density rises the controlling mode migrates to lower principal quantum number,
so the increase in collisional rate is partly offset by the changing identity of
the controlling level."*

This supersedes the Ch. 3 §3.4.3 claim of $\tau_{\text{relax}}\propto n_e^{-1.00}$
(backlog item B1) — a physical reinterpretation, not merely a corrected exponent.

**(b) Why the ℓ-mixing correction was harmless — the general principle.**
The $F(U_m)$ fix changed ℓ-mixing rates by ×3–7 yet moved $\tau_{\text{relax}}$
by 0.12%. Reason: ℓ-mixing operates at $\sim10^{11}$–$10^{12}$ s⁻¹ (picoseconds),
three orders of magnitude faster than $\tau_{\text{relax}} = 2.28$ ns. On the
nanosecond timescale the $\ell$-sublevels are *already fully equilibrated* —
they reached their statistical distribution within the first picosecond.
Speeding up an equilibration that has already finished changes nothing.

> **Timescale separation protects slow eigenvalues from errors in fast rates.**

The converse also holds and matters: the same separation means the error was
**not** harmless to absolute ℓ-resolved populations, which is why the fix still
had to be made for line-ratio work.

### 10.6 Exam outcome (Parts 2–3, 14 Jul)

| Question | Result |
|---|---|
| Q2.1 general solution | Wrote the per-level rate equation (all terms correct) instead of its solution $\mathbf n(t)=\mathbf n^{ss}+\sum_k c_k e^{\lambda_k t}\mathbf v_k$ |
| Q2.2 slowest mode sets relaxation | ✅ Correct |
| Q2.3 Framing A vs B | Half — had "ground→ionisation is slowest because the energy gap is largest"; missed the eigenvector argument (§9.3): $\lambda_0$ has ground weight exactly 1.0000 so it deletes cleanly, while $\lambda_1$ retains 0.5567 ground weight so it shifts slightly — hence agreement, but not exact |
| Q3.1 density scaling | Right instinct (both channels contribute); needed the moving-bottleneck refinement of §10.5(a) |
| Q3.2 near-degeneracy | Not known — see §10.1, §10.4 |
| Q3.3 why the ℓ-mixing fix was harmless | Not known — see §10.5(b) |

**Diagnosed pattern:** strong on physical mechanism, weaker on *reading stored
results back to answer a new question*. Q3.2 and Q3.3 were both answerable from
numbers already in these notes (the spectrum table; the ℓ-mixing timescales).
The gap is retrieval and connection, not understanding — which is precisely what
defense rehearsal trains. Parts 4 (arithmetic) and 5 (code) deferred: Part 4 is
already-demonstrated skill, Part 5 belongs in the code-review session.
