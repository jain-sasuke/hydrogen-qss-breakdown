# Derivation 04c — Non-Normality and Transient Growth of the CR Operator

**Quantity:** the numerical abscissa $\mu(L)$, and the demonstration that the CR
rate matrix $L$ is strongly non-normal, so that relaxation errors can *grow*
transiently even though every eigenvalue is negative.

**Status:** **Derived and verified.** Mechanism derived from first principles;
the criterion $\mu(L)=\lambda_{\max}\!\big((L+L^T)/2\big)$ derived from
$\frac{d}{dt}\|\mathbf x\|^2$; numerical value computed directly from
`L_grid.npy`; independently corroborated by direct ODE integration
(matrix exponential, no eigenvalues used).

**Grounded in:** standard non-normal operator theory (Trefethen & Embree,
*Spectra and Pseudospectra*); the physical asymmetry of $L$ follows from
detailed balance (Q2) and the radiative/collisional rate hierarchy (Q1, Q4b).

**Feeds thesis sections:** §3.5 (metrics), §4 Gate E, §5 (the central
QSS-breakdown argument). **This is the mechanism behind Q5.**

---

## 1. Core results

**The CR operator is non-normal:**

$$L^T L \neq L L^T$$

because $L$ is strongly asymmetric — a physical fact, not a modelling artifact.

**Numerical abscissa (the rigorous criterion):**

$$\mu(L) \;=\; \lambda_{\max}\!\left(\frac{L + L^T}{2}\right)$$

**Measured at ITER reference ($T_e = 2.947$ eV, $n_e = 1.389\times10^{14}$ cm⁻³):**

$$\mu(L) = +1.28\times10^{11}\ \mathrm{s^{-1}} \qquad\text{vs}\qquad \max_k \mathrm{Re}\,\lambda_k = -4.40\times10^{4}\ \mathrm{s^{-1}}$$

$\mu(L) > 0$ **guarantees** that some initial perturbation grows at $t=0^+$,
despite every eigenvalue being negative. The gap between the numerical and
spectral abscissae spans **fifteen orders of magnitude**: this is extreme
non-normality, not a marginal effect.

---

## 2. Why $L$ is asymmetric — the physics

A matrix has orthogonal eigenvectors if and only if it is normal; for real
matrices the familiar sufficient case is symmetry, $L[i,j] = L[j,i]$. The CR
operator badly violates this. Two entries at the ITER reference make the point:

| Entry | Transition | Process | Value |
|---|---|---|---|
| $L[0,2]$ | 2p → 1s | spontaneous radiative decay | $\approx 6.3\times10^{8}$ s⁻¹ |
| $L[2,0]$ | 1s → 2p | electron-impact excitation | $\approx 0.7$ s⁻¹ (low density) |

**Nine orders of magnitude.** The asymmetry is not numerical sloppiness — it is
thermodynamics. Falling down requires nothing (spontaneous emission); climbing
up requires a collision partner delivering $\Delta E = 10.2$ eV, which at
$kT_e = 3$ eV is exponentially rare. Detailed balance (Q2) fixes the *ratio* of
forward and reverse rate coefficients at $(g_2/g_1)e^{-\Delta E/kT_e}$ — which
at these conditions is $\approx 0.1$, not $1$.

**Consequence:** $L$ cannot be symmetrised, its eigenvectors are not
orthogonal, and every tool that assumes orthogonality fails.

---

## 3. Consequence I — modal amplitudes require left eigenvectors

Decompose a perturbation into modes:

$$\delta\mathbf n(0) = \sum_k c_k \mathbf v_k \qquad\Longrightarrow\qquad \delta\mathbf n(t) = \sum_k c_k\,e^{\lambda_k t}\,\mathbf v_k$$

The standard orthogonal projection $c_k = (\mathbf v_k\cdot\delta\mathbf n)/(\mathbf v_k\cdot\mathbf v_k)$
**is invalid here**, because it relies on cross terms $\mathbf v_j\cdot\mathbf v_k$
vanishing for $j\neq k$, which they do not.

**Worked demonstration (2-state toy, $\alpha=1,\beta=10,s_1=s_2=0.1$):**
right eigenvectors $\mathbf v_+=(10,1)$ and $\mathbf v_-=(1,-1)$ give
$\mathbf v_+\cdot\mathbf v_- = 9 \neq 0$. For $\delta\mathbf n(0)=(11,0)$:

| Method | $c_+$ | $c_-$ | Reconstructs $(11,0)$? |
|---|---|---|---|
| Exact (solve the linear system) | 1 | 1 | ✅ yes |
| Orthogonal projection | 1.0891 | 5.5 | ❌ gives $(16.39,-4.41)$ |

The projection formula does not merely give slightly wrong amplitudes — it
fails to reproduce the original vector at all, i.e. it is not a valid
decomposition. Any downstream prediction built on it is wrong.

**The correct construction.** Introduce the *left* eigenvectors,

$$\mathbf w_k^T L = \lambda_k \mathbf w_k^T \qquad\text{(equivalently } L^T\mathbf w_k = \lambda_k\mathbf w_k\text{)}$$

They share the eigenvalues of $L$ but are different vectors, and they satisfy
**biorthogonality**: $\mathbf w_j^T\mathbf v_k = 0$ for $\lambda_j\neq\lambda_k$.

*Proof.* Evaluate $\mathbf w_j^T L\mathbf v_k$ two ways:
$\mathbf w_j^T(L\mathbf v_k)=\lambda_k\,\mathbf w_j^T\mathbf v_k$ and
$(\mathbf w_j^TL)\mathbf v_k=\lambda_j\,\mathbf w_j^T\mathbf v_k$. Subtracting,
$(\lambda_k-\lambda_j)\,\mathbf w_j^T\mathbf v_k=0$, so the product vanishes
whenever the eigenvalues differ. ∎

Hence, with normalisation $\mathbf w_k^T\mathbf v_k = 1$:

$$\boxed{c_k = \mathbf w_k^T\,\delta\mathbf n(0)}$$

**Interpretation:** $\mathbf v_k$ is the *shape* of mode $k$; $\mathbf w_k$ is
its *detector* — the vector one contracts with an initial condition to ask how
much of mode $k$ is present. For a symmetric operator the two coincide; for the
CR operator they are very different. In the 2-state toy, $\mathbf v_+=(10,1)$
while $\mathbf w_+=(1,1)$; verified numerically against `eig(L.T)`, with
biorthogonality holding to $\sim10^{-17}$ (machine precision) and exact
reconstruction of $(11,0)$.

**Thesis consequence:** the slow-mode projection $P_{\rm slow}$ used in the QSS
breakdown analysis must be computed with $\mathbf w_{\rm slow}$, not
$\mathbf v_{\rm slow}$.

---

## 4. Consequence II — transient growth

### 4.1 The mechanism

Non-orthogonal eigenvectors force **large, opposing modal amplitudes**. A
minimal example makes this concrete. Take two nearly parallel modes,

$$\mathbf v_1 = (1,\,0), \qquad \mathbf v_2 = (1,\,0.01)$$

and a perturbation perpendicular to both, $\delta\mathbf n(0) = (0,1)$. The only
decomposition is

$$c_1 = -100, \qquad c_2 = +100$$

The amplitudes are **100× larger than the perturbation itself**, and they
nearly cancel. With $\lambda_1=-1$, $\lambda_2=-2$ the first component evolves as

$$\delta n_1(t) = 100\left(e^{-2t} - e^{-t}\right)$$

which is $0$ at $t=0$ but $-23.9$ at $t=0.5$. It grew from nothing, although
both modes are decaying throughout.

**Transient growth is the un-cancellation of large opposing amplitudes.** At
$t=0$ they cancel exactly; because they decay at different rates the
cancellation breaks, leaving the slower (large) amplitude exposed.

For an orthogonal (normal) operator the amplitudes are bounded by the size of
the perturbation, there is nothing large to un-cancel, and the norm decays
monotonically — a theorem. Non-normality is therefore *necessary* for transient
growth.

### 4.2 The rigorous criterion — derivation

Rather than sampling initial conditions, the effect is captured exactly by one
number. For $\dot{\mathbf x} = L\mathbf x$,

$$\frac{d}{dt}\|\mathbf x\|^2 = \frac{d}{dt}\big(\mathbf x^T\mathbf x\big) = \dot{\mathbf x}^T\mathbf x + \mathbf x^T\dot{\mathbf x} = \mathbf x^T L^T\mathbf x + \mathbf x^T L\mathbf x = \mathbf x^T\big(L + L^T\big)\mathbf x$$

Only the **symmetric part** of $L$ survives; the antisymmetric part contributes
nothing to the norm's rate of change. Writing
$\|\mathbf x\|\frac{d\|\mathbf x\|}{dt} = \tfrac12\frac{d}{dt}\|\mathbf x\|^2$
and maximising the Rayleigh quotient of the symmetric part gives

$$\left.\frac{1}{\|\mathbf x\|}\frac{d\|\mathbf x\|}{dt}\right|_{t=0} \le \lambda_{\max}\!\left(\frac{L+L^T}{2}\right) \equiv \mu(L)$$

with equality attained by the corresponding eigenvector of the symmetric part.
Hence:

- $\mu(L) < 0$ ⟹ the norm decreases immediately for **every** initial condition;
  transient growth is impossible.
- $\mu(L) > 0$ ⟹ **some** initial condition grows. Guaranteed, not merely
  possible.

### 4.3 Measured values

At the ITER reference point:

| Quantity | Symbol | Value |
|---|---|---|
| Numerical abscissa | $\mu(L)$ | $+1.28\times10^{11}$ s⁻¹ |
| Spectral abscissa | $\max_k \mathrm{Re}\,\lambda_k$ | $-4.40\times10^{4}$ s⁻¹ |
| Ratio | — | $\sim10^{15}$ |

$\mu(L)>0$, so transient amplification is guaranteed. The fifteen-order gap
quantifies how far the operator is from normal.

### 4.4 Independent confirmation, and an essential caveat

Direct integration by matrix exponential (`verify_timescales.py` Test 5 — no
eigenvalue decomposition used, hence independent of §3) reproduces the effect:
$\|\delta\mathbf n(t)\|/\|\delta\mathbf n(0)\|$ falls, then **rises** around the
nanosecond scale, then decays. Detected in 3/3 random trials.

**However, the effect is norm- and perturbation-dependent:**

| Perturbation / norm | Growth? | Max step gain |
|---|---|---|
| random, $L^2$ | yes | 1.087 |
| random, $L^1$ | **no** | 0.9996 |
| ground-state kick, $L^2$ | **no** | 1.000 |
| 2P kick, $L^2$ | yes | 1.107 |

This is not a contradiction: $\mu(L)>0$ guarantees growth for *some* direction,
not for *every* direction, and the criterion above is specific to the Euclidean
norm. **Any thesis claim of transient growth must state the norm and the
perturbation.** In particular, the Q5 claim must be tested in the observable
actually used there (the Balmer $H_\alpha/H_\beta$ ratio), not asserted from an
$L^2$ result.

---

## 5. Why this matters for the thesis

The central QSS argument is that timescale separation is *necessary but not
sufficient*. Non-normality supplies the mechanism, and it is a stronger
statement than the timescale argument alone:

1. **Eigenvalues alone do not bound the error.** $M = \tau_{\rm QSS}/\tau_{\rm relax}
   \approx 10^4$ describes the *asymptotic* decay rates. It says nothing about
   the finite-time behaviour of a perturbation, because that is governed by the
   modal amplitudes $c_k$ and their interference, not by the $\lambda_k$.
2. **$\tau_{\rm relax}=1/|\lambda_1|$ is an asymptotic slow-mode time, not a
   guarantee of monotone error decay.** For a non-normal operator the error can
   exceed its initial value before settling.
3. **Two independent failure routes.** QSS error can be large because the target
   jumped a long way (the step distance $J$, Q5) *or* because non-normal
   amplification magnified a modest initial mismatch. Both are invisible to $M$.

**Defense statement:** *"A large spectral gap does not by itself guarantee a
small finite-time QSS error, because the CR operator is strongly non-normal
($\mu(L) = +1.3\times10^{11}\,\mathrm{s^{-1}}$ against a spectral abscissa of
$-4.4\times10^{4}\,\mathrm{s^{-1}}$). Modal amplitudes and their transient
interference matter as much as the decay rates."*

---

## 6. Checks passed

1. **Asymmetry is physical, not numerical.** Traced to detailed balance and the
   radiative/collisional hierarchy; $L[0,2]/L[2,0] \sim 10^9$. ✓
2. **Biorthogonality verified numerically**: $\mathbf w_j^T\mathbf v_k \sim
   10^{-17}$ for $j\neq k$ (machine precision). ✓
3. **Left-eigenvector decomposition reconstructs exactly**; orthogonal
   projection does not. ✓
4. **Hand-derived $\mathbf w_+=(1,1)$ matches `eig(L.T)`** for the 2-state toy. ✓
5. **$\mu(L)$ derivation**: only the symmetric part survives in
   $\frac{d}{dt}\|\mathbf x\|^2$; verified algebraically. ✓
6. **Independent ODE confirmation** by matrix exponential, no eigenvalues used. ✓
7. **Norm-dependence tested** rather than assumed — growth present in $L^2$,
   absent in $L^1$ and for a ground-state perturbation. ✓

---

## 7. Pending

1. Test transient growth in the **Balmer-ratio observable**, which is the
   quantity Q5 actually reports. The $L^2$ result does not transfer
   automatically.
2. Compute $P_{\rm slow} = c_{\rm slow}$ using $\mathbf w_{\rm slow}$ across the
   grid (Q5 Session D).
3. Map $\mu(L)$ over the full $(T_e, n_e)$ grid — is the non-normality uniformly
   extreme, or does it peak somewhere physically meaningful?
4. Consider whether pseudospectra add anything beyond $\mu(L)$ for the thesis, or
   whether they belong in the paper rather than the thesis.

---

## 8. Defense one-liners

- *Why do you use left eigenvectors?* Because $L$ is asymmetric — radiative
  decay downward is $\sim10^9$ times faster than collisional excitation upward —
  so its right eigenvectors are not orthogonal and ordinary projection does not
  even reconstruct the initial condition. The left eigenvectors are biorthogonal
  to the right ones by construction and give the correct modal amplitudes.
- *All eigenvalues are negative, so why worry about transients?* Negative
  eigenvalues guarantee the error vanishes eventually; they say nothing about
  its size in between. With $\mu(L)>0$ the error is guaranteed to grow initially
  for some perturbations, and $\mu(L)$ exceeds the spectral abscissa by fifteen
  orders of magnitude here.
- *What is the numerical abscissa?* The largest eigenvalue of the symmetric part
  of $L$. It appears because $\frac{d}{dt}\|\mathbf x\|^2 = \mathbf x^T(L+L^T)\mathbf x$
  — only the symmetric part affects the norm's growth rate. Its sign decides
  whether transient growth is possible at all.
- *Is the growth physical or an artifact of your norm?* Both matter, and we
  state which: $\mu(L)>0$ is norm-specific ($L^2$) and guarantees growth for
  some direction. We observe growth for random and 2P-localised perturbations in
  $L^2$, and no growth in $L^1$ or for a ground-state perturbation. Claims are
  always reported with the norm and perturbation specified.
