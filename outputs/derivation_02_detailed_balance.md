# Derivation 02 — Detailed Balance

**Quantity:** the relation between excitation and de-excitation rate coefficients $C_{12}$ and $C_{21}$.
**Status:** Steps 1–3 complete. Steps 4–6 (stub, brutal test, write-up) pending.
**Code:** `src/rates/compute_K_CCC.py` — functions `maxwell_average` and `detailed_balance`.
**Feeds thesis sections:** §2.2 (rate coefficients), §4 Gate A (validation).

---

## 1. The core result

For a Maxwellian electron distribution at temperature $T_e$, the excitation and
de-excitation rate coefficients are related by

$$
\boxed{
\frac{C_{12}}{C_{21}} = \frac{g_2}{g_1}\,\exp\!\left(-\frac{\Delta E}{kT_e}\right)
}
$$

where:
- $C_{12}$ [cm³/s] = excitation rate coefficient, lower→upper
- $C_{21}$ [cm³/s] = de-excitation rate coefficient, upper→lower
- $g_1, g_2$ = statistical weights (degeneracies) of lower and upper states
- $\Delta E = E_2 - E_1 > 0$ = excitation energy [eV]
- $kT_e$ in same units as $\Delta E$

Equivalently, de-excitation from the excitation rate:

$$
C_{21} = C_{12} \cdot \frac{g_1}{g_2}\,\exp\!\left(+\frac{\Delta E}{kT_e}\right).
$$

---

## 2. Step 1 — Hand derivation

**Thought experiment:** place the plasma in complete thermodynamic equilibrium (CTE).
In CTE, every microscopic process is exactly balanced by its reverse (principle of
detailed balance). For excitation/de-excitation:

$$
n_1\,n_e\,C_{12} = n_2\,n_e\,C_{21}
\quad\Longrightarrow\quad
\frac{C_{12}}{C_{21}} = \frac{n_2}{n_1}\bigg|_{\text{CTE}}.
$$

The $n_e$ cancels — the same electron is the collision partner for both processes.

In CTE, populations follow Boltzmann:

$$
\frac{n_2}{n_1}\bigg|_{\text{CTE}} = \frac{g_2}{g_1}\,\exp\!\left(-\frac{\Delta E}{kT_e}\right).
$$

Substituting gives the result. $\square$

**Why it holds out of equilibrium:** $C_{12}$ and $C_{21}$ are Maxwellian averages

$$
C_{ij}(T_e) = \sqrt{\frac{8}{\pi m_e}}\,(kT_e)^{-3/2}
              \int_{\Delta E}^{\infty} \sigma_{ij}(E)\,E\,e^{-E/kT_e}\,dE
$$

These depend only on $T_e$ and the atomic cross section $\sigma_{ij}(E)$ — not on the
level populations. So the ratio $C_{12}/C_{21}$ is a property of the electron
distribution and atomic physics alone. It holds for any Maxwellian plasma at $T_e$,
regardless of whether atomic populations are at Boltzmann ratios or far from
equilibrium. The derivation used CTE only as a consistency constraint to fix the
ratio; the ratio itself is population-independent.

**Critical condition: Maxwellian electrons.** If $f_e(v)$ is non-Maxwellian (e.g.
suprathermal tails during ELMs), the simple ratio breaks and the general form

$$
C_{ij} = \int \sigma_{ij}(v)\,v\,f_e(v)\,d^3v
$$

must be evaluated with the actual distribution. The thesis assumes Maxwellian
electrons throughout (§1.5, scope); the DB relation therefore holds exactly.

---

## 3. Step 2 — Worked example

**System:** $1s \leftrightarrow 2p$ transition.
$g_1 = 2$ (1s), $g_2 = 6$ (2p), $\Delta E = 10.2$ eV, $T_e = 3$ eV.

$$
\frac{C_{12}}{C_{21}}
= \frac{6}{2}\,e^{-10.2/3}
= 3\,e^{-3.4}
= 3 \times 0.03337
= \boxed{0.100}
$$

Given $C_{21} = 1.0\times10^{-10}$ cm³/s:

$$
C_{12} = 0.100 \times 1.0\times10^{-10}
= \boxed{1.0\times10^{-11}\ \text{cm}^3\text{s}^{-1}}
$$

**Internal consistency check:** $C_{12} = 1.0\times10^{-11}$ cm³/s is exactly $k_{12}$
used in the Q1 worked example. The Q1 toy numbers were physically consistent with
detailed balance at $T_e = 3$ eV.

**Units:** $C_{ij} = \langle\sigma v\rangle$: $[\sigma] = \text{cm}^2$, $[v] = \text{cm/s}$,
so $[C_{ij}] = \text{cm}^3\text{s}^{-1}$. Then
$[n_1\,n_e\,C_{12}] = \text{cm}^{-3}\cdot\text{cm}^{-3}\cdot\text{cm}^3\text{s}^{-1}
= \text{cm}^{-3}\text{s}^{-1} = [\dot{n}]$. Dimensionally correct.

**At $T_e = 3$ eV, de-excitation enhancement:**

$$
\frac{C_{21}}{C_{12}} = \frac{g_1}{g_2}\,e^{+\Delta E/kT_e}
= \frac{2}{6}\,e^{+10.2/3} = \frac{1}{3}\times 30.0 = 10.0
$$

De-excitation is 10× faster than excitation — physically correct: exciting up requires
a rare fast electron (Boltzmann-suppressed), coming down has no barrier.

**At $T_e = 10$ eV**, the enhancement factor = $(2/6)e^{+10.2/10} = 0.925 < 1$: at
high $T_e$ the threshold barrier is negligible and the degeneracy ratio $g_1/g_2 = 1/3$
wins. The crossover from $C_{21} > C_{12}$ to $C_{21} < C_{12}$ is real physics,
correctly captured.

---

## 4. Step 3 — Code audit: `compute_K_CCC.py`

**Architecture: excitation by Maxwellian averaging, de-excitation by DB construction.**

In the main loop:
```python
K_exc   = maxwell_average(E_raw, sig_raw, dE, TE_GRID)
K_deexc = detailed_balance(K_exc, l_i, l_f, dE, TE_GRID)
```

De-excitation is **derived**, not independently computed from CCC de-excitation cross
sections.

**`maxwell_average` audit:**
```python
def maxwell_average(E_raw, sig_raw, dE, Te_arr, n_grid=5000):
    E_grid   = np.linspace(dE + 1e-4, E_raw.max(), n_grid)
    sig_grid = np.interp(E_grid, E_raw, sig_raw, left=0.0, right=0.0)
    for k, Te in enumerate(Te_arr):
        integrand    = sig_grid * E_grid * np.exp(-E_grid / Te)   # a0^2 * eV
        integral_raw = np.trapezoid(integrand, E_grid)             # a0^2 * eV^2
        integral_SI  = integral_raw * A0_M**2 * KB**2             # m^2 * J^2
        K_arr[k]     = prefactor(Te) * integral_SI * 1e6          # cm^3/s
```

Unit chain verified:
- integrand: $[a_0^2]\cdot[\text{eV}]\cdot[1] = a_0^2\cdot\text{eV}$
- trapz over $dE$ [eV]: integral\_raw in $a_0^2\cdot\text{eV}^2$
- `* A0_M**2 * KB**2`: converts to m²·J²
- prefactor: $\sqrt{8/\pi m_e}\,(kT_e)^{-3/2}$ in SI → product is m³/s
- `* 1e6`: m³/s → cm³/s ✓

**`prefactor` audit:**
```python
def prefactor(Te_eV):
    return np.sqrt(8.0 / np.pi / ME) * (Te_eV * KB)**(-1.5)
```
Derived from first principles: $\sqrt{8/\pi m_e}\,(kT_e)^{-3/2}$.
Power is $-3/2 = -1.5$. **Correct.** Argument `Te_eV * KB` = $kT_e$ in joules. **Correct.**

*Important:* the $-3/2$ is the prefactor power only. $K(T_e)$ does not scale as
$T_e^{-3/2}$ overall — the integral also grows with $T_e$ (more electrons above
threshold). Common misreading; must be stated clearly in §2.2.

**`detailed_balance` audit:**
```python
def detailed_balance(K_exc_arr, l_i, l_f, dE_eV, Te_arr):
    omega_i = stat_weight(l_i)    # lower state weight = g_1
    omega_f = stat_weight(l_f)    # upper state weight = g_2
    return K_exc_arr * (omega_i / omega_f) * np.exp(dE_eV / Te_arr)
```

Mapping to hand formula $C_{21} = C_{12}(g_1/g_2)e^{+\Delta E/kT_e}$:
- `K_exc_arr` = $C_{12}$ ✓
- `omega_i/omega_f` = $g_1/g_2$ (lower/upper) ✓
- `np.exp(dE_eV / Te_arr)` = $e^{+\Delta E/kT_e}$ ✓

Ratio direction **correct**. Numerical cross-check: $(2/6)\,e^{+10.2/3} = 10.0$ =
$(C_{12}/C_{21})^{-1}$ = $(0.100)^{-1}$ ✓.

**Upper-limit truncation (open physics question):**
Integration runs to `E_raw.max()` (~100–968 eV), not $\infty$. At $T_e = 10$ eV,
$e^{-E/kT_e} = e^{-100/10} = e^{-10} \approx 4.5\times10^{-5}$. The Boltzmann
factor kills the integrand at $\sim10\,kT_e$, so truncation error at 100 eV is
$<0.01\%$ for the thesis $T_e$ range. **Not a problem in practice** — but should be
stated in §2.2 with the $e^{-E/kT_e}$ argument.

---

## 5. Critical audit finding: two distinct "detailed balance" numbers

**This distinction MUST appear in the thesis and at the defense.**

| Check | Where | What it measures | Expected value |
|---|---|---|---|
| DB by construction | `compute_K_CCC.py` Check B | $K_{\text{exc}}/K_{\text{deexc}}$ vs formula | Machine precision ($<10^{-10}$%) |
| DB of raw CCC data | `qc_ccc.py` | CCC forward vs reverse cross sections | ~0.43% (or 0.05%?) |

The 0.43% from the verified-facts list comes from the **second** check — a physics
statement that Igor's CCC cross sections themselves satisfy microscopic reversibility.
The first check is enforced by construction and has nothing to do with 0.43%.

**At defense question: "Is your detailed balance enforced or verified?"**
Answer: both, in different places, measuring different things. The CR matrix uses
DB-enforced rate coefficients (guaranteeing the Saha-Boltzmann limit at high $n_e$,
Gate C). The 0.43% is an independent physics verification of the raw CCC data.

**Action item:** read `src/parsers/qc_ccc.py` to confirm which number it produces and
exactly what comparison it makes. Pin down whether the 0.43% comes from there or
from the L-matrix Gate 2 check.

---

## 6. Thesis writing note for §2.2

Write §2.2 in this order:
1. Physical definition: $C_{ij} = \langle\sigma_{ij}v\rangle$, units cm³/s, why not s⁻¹.
2. Maxwellian averaging formula (write it out in full with all symbols defined).
3. Detailed balance derivation (CTE thought experiment → Boltzmann → ratio).
4. Why it holds out of equilibrium (rate coefficients are population-independent).
5. Maxwellian assumption stated explicitly — scope limitation, non-Maxwellian breaks it.
6. Code implementation: excitation by Maxwell averaging, de-excitation by DB construction.
7. Upper-limit truncation: negligible because $e^{-E/kT_e}$ at $E\sim100$ eV, $T_e\leq10$ eV.
8. Two distinct DB checks and what each measures.

---

## 7. Steps 4–6 — Pending

**Step 4 — Stub confirmation** (write and run):
```python
import numpy as np
# 1s->2p at Te=3 eV
g1, g2, dE, Te = 2, 6, 10.2, 3.0
C21 = 1e-10   # cm^3/s, given
C12 = C21 * (g2/g1) * np.exp(-dE/Te)
print(f"C12 = {C12:.3e}  (expect 1.000e-11)")
print(f"ratio = {C12/C21:.4f}  (expect 0.1000)")
# Reverse check
C21_recovered = C12 * (g1/g2) * np.exp(+dE/Te)
print(f"C21 recovered = {C21_recovered:.3e}  (expect 1.000e-10)")
```

**Step 5 — Brutal physics tests** (to do):
- Sign: $+\Delta E$ in the exponent for de-excitation. Verify direction.
- Low-$T_e$ limit: as $T_e\to0$, $C_{12}/C_{21}\to0$ (excitation vanishes, de-excitation
  survives — correct, the atom falls back to ground state).
- High-$T_e$ limit: ratio → $g_2/g_1$ (Boltzmann factor → 1, pure degeneracy).
- Monotonicity: ratio $C_{12}/C_{21}$ increases with $T_e$ (as expected).
- Load real `K_exc_full.npy` and `K_deexc_full.npy`, compute ratio for all $\Delta n\neq0$
  pairs, compare to $(g_2/g_1)e^{-\Delta E/kT_e}$ — confirm machine-precision agreement.

**Step 6 — Write-up:** finalise this note after Steps 4–5 complete.
