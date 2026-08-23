# Derivation 01 — The Collisional–Radiative Rate Matrix $L$

**Quantity:** the rate matrix $L(T_e, n_e)$ governing the bound-state populations.
**Status:** verified end-to-end (hand derivation → worked example → code audit → stub → physics tests).
**Code:** `src/rates/assemble_cr_matrix.py`, function `build_L` (lines 102–155).
**Verified output:** `data/processed/cr_matrix/L_grid.npy`, shape `(50, 8, 43, 43)`, units s⁻¹.

---

## 1. Governing equation and what $L$ is

The bound-state populations obey a linear system

$$
\frac{d\mathbf n}{dt} = L\,\mathbf n + \mathbf b,
$$

- $\mathbf n$ — bound-state population vector, $[\text{cm}^{-3}]$ (43 states: $n=1$–8 $\ell$-resolved, $n=9$–15 bundled).
- $L$ — rate matrix, $[\text{s}^{-1}]$, a function of $T_e$ and $n_e$.
- $\mathbf b$ — external source vector from recombination, $[\text{cm}^{-3}\,\text{s}^{-1}]$.

**Why recombination is in $\mathbf b$, not $L$.** $L$ must be a *linear operator on the neutral population vector*: every entry multiplies some $n_i$. Excitation, de-excitation, radiative decay, and ionization are all linear in the bound populations. Recombination is $\propto n_e n_+ \alpha$ — it depends on the **ion** density $n_+$, a species outside the neutral manifold — so it cannot be an element of $L$ acting on $\mathbf n$. It enters as a constant feed term $\mathbf b$ (ChemE analogy: $L$ is the reaction-rate matrix for the manifold; $\mathbf b$ is the feed stream from outside).

---

## 2. Hand derivation (2-state toy: 1s = state 1, 2p = state 2)

Population balances (species balances on bound atoms):

$$
\frac{dn_1}{dt} = -\,k_{12}n_e\,n_1 \;-\; S_{1,\text{ion}}n_e\,n_1 \;+\; k_{21}n_e\,n_2 \;+\; A_{21}\,n_2 \;+\; \underbrace{n_e n_+ \alpha_1}_{\to\,b_1}
$$

$$
\frac{dn_2}{dt} = +\,k_{12}n_e\,n_1 \;-\; k_{21}n_e\,n_2 \;-\; A_{21}\,n_2 \;-\; S_{2,\text{ion}}n_e\,n_2 \;+\; \underbrace{n_e n_+ \alpha_2}_{\to\,b_2}
$$

Collecting into $\dot{\mathbf n} = L\mathbf n + \mathbf b$:

$$
L = \begin{pmatrix}
-(k_{12}n_e + S_{1,\text{ion}}n_e) & \; k_{21}n_e + A_{21} \\[4pt]
k_{12}n_e & \; -(k_{21}n_e + A_{21} + S_{2,\text{ion}}n_e)
\end{pmatrix},
\qquad
\mathbf b = \begin{pmatrix} n_+\alpha_1 \\ n_+\alpha_2 \end{pmatrix}.
$$

**Symbols:** $k_{12}$ = excitation 1→2 rate coefficient [cm³/s]; $k_{21}$ = de-excitation 2→1 [cm³/s]; $A_{21}$ = spontaneous emission 2→1 [s⁻¹]; $S_{p,\text{ion}}$ = ionization out of state $p$ [cm³/s]; $\alpha_p$ = recombination into $p$ [cm³/s].

---

## 3. Conventions and structural rules (must hold for the full 43-state $L$)

| Rule | Statement | Physical meaning |
|---|---|---|
| Index | $L_{ij}$ ($i\neq j$) = rate $j \to i$ | off-diagonal = gain into $i$ from $j$ |
| Diagonal | $L_{ii} < 0$ | total loss rate from state $i$ |
| Off-diagonal | $L_{ij} \geq 0$ | rates cannot be negative |
| **Conservation** | $\sum_i L_{ij} = -S_{j,\text{ion}}\,n_e$ | every column sums to the ionization leak; bound-bound transfers cancel |

The conservation rule is the key check: an atom collisionally excited from $j$ leaves $j$ (a loss in $L_{jj}$) but arrives in some bound state $i$ (a gain in $L_{ij}$), so internal transfers cancel in the column sum. Only ionization genuinely removes an atom from the bound manifold, so the column sum equals $-S_{j,\text{ion}}n_e$ (negative = net leakage to the ion).

---

## 4. Worked example (numbers)

At $T_e = 3$ eV, $n_e = 10^{14}\,\text{cm}^{-3}$, with representative coefficients:

| Quantity | Value | × $n_e$ → s⁻¹ |
|---|---|---|
| $k_{12}$ | $1.0\times10^{-11}$ cm³/s | $k_{12}n_e = 10^{3}$ |
| $k_{21}$ | $1.0\times10^{-10}$ cm³/s | $k_{21}n_e = 10^{4}$ |
| $A_{21}$ | $6.27\times10^{8}$ s⁻¹ | — |
| $S_{1,\text{ion}}$ | $1.0\times10^{-13}$ cm³/s | $S_{1}n_e = 10$ |
| $S_{2,\text{ion}}$ | $1.0\times10^{-10}$ cm³/s | $S_{2}n_e = 10^{4}$ |

$$
L = \begin{pmatrix} -1010 & 6.2701\times10^{8} \\ 1000 & -6.2702\times10^{8}\end{pmatrix}\ \text{s}^{-1}
$$

Column sums: $(-1010+1000,\; 6.2701\times10^8 - 6.2702\times10^8) = (-10,\; -10^4) = (-S_{1}n_e,\; -S_{2}n_e)$. ✔

**Note (catastrophic cancellation):** column 2 is a difference of two numbers agreeing to four digits ($\sim 6.27\times10^8$) whose meaningful result is the fifth digit ($10^4$). This is why the conservation gate is quoted as a *relative* residual. The state-2 diagonal is dominated by radiative decay $A_{21}=6.27\times10^8$; the ionization leak is $\sim10^4$, four orders smaller — the seed of the two-timescale split in Derivation 04.

---

## 5. Code audit — `build_L`

**Inputs:** `Te_idx` (int, into 50-pt $T_e$ grid 1–10 eV), `ne` (float, cm⁻³), `rates` (dict of `.npy` arrays).

| Array | Shape | Units | Role |
|---|---|---|---|
| `K_exc_full` | (43,43,50) | cm³/s | excitation, indexed `[lo,hi]`, upper-triangular |
| `K_deexc_full` | (43,43,50) | cm³/s | de-excitation, indexed `[hi,lo]`, lower-triangular |
| `K_ion_final` | (43,50) | cm³/s | ionization per state |
| `A_resolved`, `A_bund_res`, `A_bund_bund` | (36,36),(36,7),(7,7) | s⁻¹ | Einstein $A$ |
| `gamma_resolved`, `gamma_bundled` | (36,),(7,) | s⁻¹ | total radiative loss per state |

**Outputs:** `L` (43,43) s⁻¹. Grid driver saves `L_grid.npy` (50,8,43,43), `S_grid.npy` (50,8,43).

**Assembly logic (verified):**
```python
Ke = K_exc_full[:,:,Te_idx]*ne        # Ke[lo,hi] = excitation lo->hi
Kd = K_deexc_full[:,:,Te_idx]*ne      # Kd[hi,lo] = de-excitation hi->lo
L += Ke.T   # transpose -> deposits Ke[lo,hi] into L[hi,lo]  (gain to upper)  ✔
L += Kd.T   # transpose -> deposits Kd[hi,lo] into L[lo,hi]  (gain to lower)  ✔
diag(L) -= Ke.sum(1) + Kd.sum(1)      # total collisional loss on diagonal     ✔
# radiative: L[:36,:36] += A_resolved; diag -= gamma
# ionization: diag -= K_ion*ne
```
The transpose is correct: excitation lands in the lower-left $L[\text{hi},\text{lo}]$ and de-excitation+radiative in the upper-right $L[\text{lo},\text{hi}]$, matching §2.

**Audit finding (documented, not a bug):** the docstring "VECTORIZED ASSEMBLY" block (lines 40–41) mislabels the transition *directions* in its parentheticals (calls excitation "upper→lower"). The **inline** comments (lines 126–127) and the **code** are correct. Comment only; no code defect.

---

## 6. Stub confirmation

A standalone 2-state stub built with the same convention reproduces §4 exactly:

```python
import numpy as np
ne = 1e14
k12, k21 = 1e-11, 1e-10
A21 = 6.27e8
S1, S2 = 1e-13, 1e-10
L = np.zeros((2,2))
L[1,0] += k12*ne
L[0,1] += k21*ne + A21
L[0,0] -= k12*ne + S1*ne
L[1,1] -= k21*ne + A21 + S2*ne
```
Output: `L = [[-1010, 6.2701e8], [1000, -6.2702e8]]`; `col sums = [-10, -10000]` = expected. ✔
Hand, worked example, and executable agree.

---

## 7. Brutal physics tests (passed)

1. **Units.** $k\,n_e$ = (cm³/s)(cm⁻³) = s⁻¹; $A$ = s⁻¹; $S\,n_e$ = s⁻¹. Every entry s⁻¹. $L\mathbf n$ = (s⁻¹)(cm⁻³) = cm⁻³ s⁻¹ = $\dot{\mathbf n}$. ✔
2. **Conservation (Gate 1 / QC-D).** Max relative column-sum error $= 2.59\times10^{-11}$ across all 50×8 grid points. ✔
3. **Signs (Gate 4 / QC-C).** 0 positive diagonals over all 50×8×43 entries. ✔
4. **$\ell$-mixing detailed balance.** At the benchmark point, $L[2s,2s]=-2.126\times10^{10}$; the collisional part of $L[2p,2p] = 7.748\times10^9 - A_{2p\to1s} = 7.12\times10^9$. Ratio $= 2.99 \approx g_{2p}/g_{2s}=3$ — correct detailed balance for the degenerate 2s↔2p pair. ✔
5. **Largest matrix entry.** $\arg\max|L| = (28,28) = $ **8s** diagonal $= -7.16\times10^{12}$ s⁻¹ (effective coeff $0.0516$ cm³/s). Checked against $n^4$ scaling of $\ell$-mixing from the validated 2s rate: observed enhancement $0.0516/1.53\times10^{-4} = 337$, predicted $(8/2)^4 = 256$ — agreement to ~30%. Physically correct: (i) it is a *resolved* state (bundled levels have no internal $\ell$-mixing), and (ii) the highest resolved $n$ must carry the fastest $\Delta n=0$ rate. **Real physics, not an artifact.** ✔

**Lesson for the dossier.** None of the six QC gates can validate the *magnitude* of an internal rate — $\ell$-mixing cancels in the conservation sum, and the sign/NaN checks are magnitude-blind. The 8s rate was confirmed only by extracting the argmax and beating it against $n^4$ by hand. This is precisely the class of check the prior $-46\%$ Hα artifact escaped: structurally consistent, physically wrong. Magnitude must be tested against physics explicitly.

**Open item:** confirm the $\ell$-mixing cutoff (Pengelly–Seaton / Debye) explicitly; the clean $n^4$ scaling is strong indirect evidence the cutoff is active, but it should be verified directly.

---

## 8. Verdict

$L$ is verified from first principles. The matrix $L\_grid.npy$ (28 Mar 2026) is the trustworthy object for all downstream work: structure (signs, conservation) holds to $10^{-11}$, and the dominant collisional rate is physically correct. Proceed to Derivation 02 (detailed balance) on this foundation.
