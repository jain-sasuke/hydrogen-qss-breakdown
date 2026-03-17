# SESSION REPORT — WEEK 3 COMPLETION + WEEK 4 PREPARATION
## Recombination Rates, Coverage Audit, V&S f_pn Derivation

**Project:** Hydrogen Collisional-Radiative Model — QSS Breakdown in ITER Divertor  
**Session Date:** 2026-03-15  
**Status:** Week 3 COMPLETE. Week 4 (`compute_K_VS.py`) ready to implement.

---

## 1. WHAT WAS COMPLETED TODAY

### 1.1 `recombination_rates.py` — DONE ✅

Computed and validated radiative recombination (RR) and three-body recombination
(3BR) rate coefficients for all 43 states (n=1..15).

### 1.2 Coverage Audit — DONE ✅

Confirmed exactly what is complete and what V&S must fill.

### 1.3 f_pn Derivation — DONE ✅

Resolved the oscillator strength source question for V&S. Verified formula and
file against NIST. Ready to write `compute_K_VS.py`.

---

## 2. RECOMBINATION RATES

### 2.1 Radiative Recombination (RR)

**Formula — Seaton (1959) / Johnson (1972) single-term:**

```
alpha_RR(n, Te) = D * sqrt(I_n/kTe) * exp(I_n/kTe) * E1(I_n/kTe)

D = 5.197e-14 cm³/s  (Johnson 1972, Eq.6)
I_n = 13.6058/n²  eV
E1(x) = integral_x^inf exp(-t)/t dt  (scipy.special.exp1)
```

**Why single-term, not Johnson three-term:**
Johnson's three-term Gaunt sum uses coefficients g₁(n) ~ −n, which grows
large negative for n ≥ 5. At low Te where x = I_n/kTe < 1 (e.g. n=5 at
Te=1 eV, x=0.54), the g₁/x term dominates and makes alpha_RR **negative** —
unphysical. The single-term formula is always positive and is what Capitelli
(2016), Hartgers (2001), and ADAS all use. Accuracy within factor 2, sufficient
since 3BR >> RR for high-n states at ITER conditions.

**l-distribution:**
```
alpha_RR(n, l, Te) = alpha_RR(n, Te) * (2l+1) / n²
```
Valid for ne ≥ ~1e10 cm⁻³ (Fujimoto 2004, Appendix 4A, Fig.4A.2). Parameter
space starts at ne = 1e12 cm⁻³ — two orders above threshold. ✓

**Blueprint correction:** Blueprint had coefficient 5.2e-14 (Seaton simplified).
Correct D = 5.197e-14 from Johnson (1972) Eq.(6). Difference is 0.06%.

**Non-monotonic Te behaviour is correct physics:**
alpha_RR peaks near Te ~ I_n. For n=1: peak at 13.6 eV (above range, so
decreasing in 1..10 eV). For n=2: peak at 3.4 eV (inside range, non-monotone).
For n ≥ 3: peak below 1.5 eV (below range, increasing). All physically correct.

### 2.2 Three-Body Recombination (3BR)

**Derivation from Saha equation (Griem 1997, Eq.6.23-6.25; V&S 1980, Sec.III.B):**

In LTE:  K_ion × N_nl × ne = alpha_3BR × ne² × n_ion

Saha equation:
```
n_e × n_ion / N_nl = (2/g_nl) × (2πme×kTe/h²)^(3/2) × exp(-I_n/kTe)
```

Rearranging:
```
alpha_3BR(n, l, Te) = K_ion(n, l, Te)
                     × (g_nl / 2)
                     × (h² / 2π×me×kTe)^(3/2)
                     × exp(I_n / kTe)
```

Units: K_ion [cm³/s] × (h²/...)^(3/2) [m³ → ×1e6 = cm³] × exp() = **cm⁶/s**

**CRITICAL BLUEPRINT ERROR CORRECTED:**
Blueprint had exponent **3** (i.e. (h²/...)³). Correct exponent is **3/2**,
which follows directly from the Saha equation derivation. Exponent 3 is
dimensionally wrong — gives cm⁹/s, not cm⁶/s. This would make 3BR rates
~10³⁰ times too large at ITER conditions. Fixed.

**Statistical weight:**
```
g_nl = 2(2l+1)  for resolved states
g_n  = 2n²      for bundled shells
```
Factor of 2 from electron spin (hydrogen doublets). This is the
omega = 2(2l+1) convention used throughout the model.

**3BR dominance regime:**
3BR >> RR only for HIGH-n states (n ≥ 6), not for n=1,2.
At ne=1e14, Te=3 eV:
- n=9 bundled: 3BR/RR rate ratio = **127×** ✓
- n=2P:        3BR/RR rate ratio = **0.17×** (RR comparable — correct physics)

The effective CR recombination (sum over all n) is dominated by high-n 3BR
capture followed by radiative cascade. This is consistent with Alcator C-Mod
observations.

### 2.3 QC Results (all pass)

```
Check A — No negatives/NaN:         PASS (neg=0, nan=0)
Check B — RR magnitudes at Te=1 eV:
  n=1: 1.318e-14 cm³/s  (Seaton 1959: ~1.3e-14) ✓
  n=9: 3.447e-14 cm³/s  (shell-total, expected ~3e-14) ✓
Check C — l-distribution sums to shell:  0.00e+00% error ✓
Check D — 3BR >> RR at n=9, ne=1e14:     127x ✓
Check E — 3BR/K_ion = (g/2)×lambda³:    err < 1e-14% (machine precision) ✓
```

### 2.4 Output Files

```
data/processed/recombination/
  alpha_RR_resolved.npy    (36, 50)  [cm³/s]  — RR for n=1..8 l-resolved
  alpha_RR_bundled.npy     ( 7, 50)  [cm³/s]  — RR for n=9..15 shell-total
  alpha_3BR_resolved.npy   (36, 50)  [cm⁶/s]  — 3BR for n=1..8 l-resolved
  alpha_3BR_bundled.npy    ( 7, 50)  [cm⁶/s]  — 3BR for n=9..15 bundled
  Te_grid_recomb.npy       (50,)     [eV]
  recombination_meta.csv   43 states — labels, I_n, g_nl, sources
```

**In rate equations:**
- RR:  alpha_RR  [cm³/s] × ne × n_ion → [cm⁻³s⁻¹]
- 3BR: alpha_3BR [cm⁶/s] × ne² × n_ion → [cm⁻³s⁻¹]
- ne and n_ion NOT baked in — pure rate coefficients.

---

## 3. COVERAGE AUDIT

### 3.1 State Space (confirmed)

```
States 0..35:  n=1..8 l-resolved   (36 states)
               n=1: 1S
               n=2: 2S, 2P
               n=3: 3S, 3P, 3D
               ...
               n=8: 8S, 8P, 8D, 8F, 8G, 8H, 8I, 8J  (8 states)

States 36..42: n=9..15 l-bundled   (7 states)
               Statistical: N(n,l) = N(n) × (2l+1)/n²

State 43:      H⁺ ion

Total: 44 states
```

**Te grid:** 50 points, log-spaced, 1.0..10.0 eV
```python
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)
```

### 3.2 Complete Rate Table Inventory

| Rate | Shape | Source | Status |
|------|-------|--------|--------|
| K_exc res↔res (n=1..8) | (870, 50) | CCC | ✅ |
| K_exc res→n10_bund | (36, 50) | CCC summed | ✅ |
| K_deexc n10→res | (36, 50) | DB from above | ✅ |
| K_ion resolved n=1..8 | (36, 50) | CCC TICS | ✅ |
| K_ion n=9 bundled | (1, 50) | CCC TICS.9 | ✅ |
| K_ion n=10..15 bundled | (6, 50) | Lotz 1968 | ✅ |
| alpha_RR resolved n=1..8 | (36, 50) | Seaton/Johnson | ✅ |
| alpha_RR bundled n=9..15 | (7, 50) | Seaton/Johnson | ✅ |
| alpha_3BR resolved n=1..8 | (36, 50) | DB(CCC TICS) | ✅ |
| alpha_3BR bundled n=9..15 | (7, 50) | DB(Lotz) | ✅ |
| A res→res n=1..8 | (36, 36) | Hoang Binh | ✅ |
| A bund→res n=9..15→1..8 | (36, 7) | Hoang Binh | ✅ |
| A bund→bund n=9..15 | (7, 7) | Hoang Binh | ✅ |
| gamma_resolved | (36,) | Hoang Binh | ✅ |
| gamma_bundled | (7,) | Hoang Binh | ✅ |
| **K_exc res→n9_bund** | **(36, 50)** | **V&S — MISSING** | ❌ |
| **K_exc res→n11..15_bund** | **(180, 50)** | **V&S — MISSING** | ❌ |
| **K_exc n9→n10** | **(1, 50)** | **V&S — MISSING** | ❌ |
| **K_exc n9→n11..15** | **(5, 50)** | **V&S — MISSING** | ❌ |
| **K_exc n10→n11..15** | **(5, 50)** | **V&S — MISSING** | ❌ |
| **K_exc n11..15↔n11..15** | **(20, 50)** | **V&S — MISSING** | ❌ |

**15/21 complete. 6/21 missing — all involve bundled shells. V&S fills all 6.**

Total V&S excitation arrays needed: 36 + 180 + 1 + 5 + 5 + 20 = **247**
De-excitation for all 247 via detailed balance (no extra computation).

### 3.3 A-matrix convention

A[i, j] means Einstein A from state j (initial, upper) to state i (final, lower).
Each **column** j sums to gamma_j (total radiative decay rate of state j).
Shapes confirmed:
- A_resolved:  (36, 36) — resolved → resolved
- A_bund_res:  (36,  7) — bundled  → resolved  (cascade down)
- A_bund_bund: ( 7,  7) — bundled  → bundled   (cascade within high-n)

Radiative: only downward (spontaneous emission). No A for resolved→bundled
(photons go down in energy, not up).

---

## 4. V&S f_pn DERIVATION (KEY RESULT)

### 4.1 Why not Johnson (1972) Gaunt formula

Earlier claim that "Johnson gives f_pn to 0.5%" was wrong. Tested:

| Transition | Johnson formula | Exact (NIST) | Error |
|---|---|---|---|
| f(1→2) | 0.269 | 0.4162 | −35% |
| f(1→3) | 0.340 | 0.0791 | +330% |
| f(9→15) | **negative** | 0.0184 | unphysical |

Root cause: Johnson Table 1 coefficients g_i(n) are fitted for the
**bound-free** Gaunt factor where x = E_photon/I_n ∈ [0,1]. For adjacent
high-n shells, x = 1−(p/n)² is small (e.g. x=0.19 for 9→10), causing
g₁/x and g₂/x² to diverge. The formula is simply not valid for bound-bound
f_pn, regardless of what V&S claim.

V&S cite "Johnson 1972" for f_pn — they likely mean his 1967 paper
(Phys. Rev. 115) which has a different formula, or their claim of 0.5%
accuracy applies only to the high-n regime (p ≥ 20) where x → 0 slowly.

### 4.2 Correct formula — from Hoang Binh A coefficients

**Derivation (CGS, from Osterbrock 1989 Eq.3.17):**

Level oscillator strength f_{pl → nu,lu} (absorption, lower=p, upper=n):
```
f_{pl→nl'} = (m_e × c) / (8π²e²) × λ_pn² × (2l'+1)/(2l+1) × A_{nl'→pl}
```

Shell-to-shell (summed over all l, l' pairs):
```
f_pn = (1/p²) × Σ_{l=0}^{p-1} Σ_{l'=l±1} (2l+1) × f_{pl→nl'}
     = prefactor × λ_pn² / p² × Σ_{l,l'} (2l'+1) × A_{nl'→pl}
```

**Prefactor:**
```
m_e × c / (8π²e²) in CGS = 1.4992 s/cm²
```

**Simpler using Hoang Binh f_abs column directly:**
```
f_pn = (1/p²) × Σ_{rows: nl=p, nu=n, |lu-ll|=1} (2×ll+1) × f_abs
```

The `f_abs` column in `H_A_E1_LS_n1_15_physical.csv` already contains
the level oscillator strength — no A conversion needed.

### 4.3 Verification against NIST

| p→n | f_pn (Hoang Binh) | f_pn (NIST) | Error |
|---|---|---|---|
| 1→2 | 0.41620 | 0.41620 | 0.00% |
| 1→3 | 0.07910 | 0.07910 | 0.00% |
| 1→4 | 0.02899 | 0.02899 | 0.00% |
| 2→3 | 0.64075 | 0.64070 | +0.01% |
| 2→4 | 0.11932 | 0.11930 | +0.02% |
| 3→4 | 0.84209 | 0.84210 | 0.00% |

High-n (all nonzero, physically sensible):
- f(1→9)   = 0.002216  (15 transitions summed: 1 l-pair)
- f(8→9)   = 1.807     (15 l-pairs)
- f(9→10)  = 1.999     (17 l-pairs, expected ~2 for adjacent large-n shells)
- f(9→15)  = 0.018     (17 l-pairs)

**Why Hoang Binh is A-tier:**
- Errors 0.00–0.02% vs NIST (Johnson gave −35% to +330%)
- Quantum mechanical — not a polynomial fit
- Self-consistent with A matrix already in the repo
- E1 transitions carry >99.9% of oscillator strength sum
- Covers all p=1..15, n=p+1..15 needed for V&S

**Thesis defense statement:**
*"Shell oscillator strengths f_pn computed from Hoang Binh hydrogenic
Einstein A coefficients via the standard f–A relation (Osterbrock 1989,
Eq.3.17), exact for E1 transitions which dominate the oscillator strength sum."*

---

## 5. FULL REPO STRUCTURE (current state)

```
non_markovian_cr/
├── src/
│   ├── parsers/
│   │   ├── parse_adas.py          # ADAS SCD/ACD parser (Week 1)
│   │   ├── parse_ccc.py           # CCC cross section parser (Week 2)
│   │   │   PATCHED: L_MAP extended to L=9 (n=10 L-orbital)
│   │   │   PATCHED: regex [SPDFGHIJKL], parse_state_bray(), n=10 support
│   │   └── parse_tics.py          # TICS ionization cross sections (Week 3)
│   └── rates/
│       ├── compute_K_CCC.py       # CCC Maxwell averaging (Week 2-3)
│       │   PATCHED: 50-pt Te grid, SPDFGHIJKL, n=10 collapse step
│       ├── compute_K_TICS.py      # TICS Maxwell averaging + Lotz (Week 3)
│       │   PATCHED: Lotz Eq.(5) correct form with exp1, no hard threshold
│       ├── ionization_rates.py    # Assembles K_ion_final.npy (Week 3)
│       ├── radiative_rates.py     # Hoang Binh A matrix assembly (Week 3)
│       ├── recombination_rates.py # RR + 3BR — NEW THIS SESSION (Week 3)
│       └── compute_K_VS.py        # V&S excitation — NEXT (Week 4)
│
├── data/
│   ├── raw/
│   │   ├── adas/                  # SCD96/ACD96 raw files
│   │   ├── ccc/                   # 3,117 CCC files from Prof. Bray
│   │   ├── tics/                  # 54 TICS files (n=1..9)
│   │   └── Radiative/             # H_A_E1_LS_n1_15_physical.csv
│   └── processed/
│       ├── adas/
│       │   ├── scd96_h_long.csv   (696, 4)
│       │   └── acd96_h_long.csv   (696, 4)
│       ├── collisions/
│       │   ├── ccc/
│       │   │   ├── ccc_crosssections.csv      (112,064 rows, n=1..10)
│       │   │   ├── K_CCC_exc_table.npy        (1320, 50) [cm³/s]
│       │   │   ├── K_CCC_deexc_table.npy      (1320, 50) [cm³/s]
│       │   │   ├── K_CCC_metadata.csv         (1320, 12)
│       │   │   ├── K_exc_to_n10_bundled.npy   (36, 50)   [cm³/s]
│       │   │   └── Te_grid.npy                (50,)      [eV]
│       │   └── tics/
│       │       ├── tics_crosssections.csv     (2484 rows)
│       │       ├── K_ion_resolved.npy         (36, 50)   [cm³/s]
│       │       ├── K_ion_n9_bundled.npy       (1, 50)    [cm³/s]
│       │       ├── K_ion_final.npy            (43, 50)   [cm³/s]
│       │       ├── K_ion_final_meta.csv       (43, 8)
│       │       └── Te_grid_ion.npy            (50,)      [eV]
│       ├── Radiative/
│       │   ├── H_A_E1_LS_n1_15_physical.csv  (1015 rows, input)
│       │   ├── A_resolved.npy                 (36, 36)   [s⁻¹]
│       │   ├── A_bund_res.npy                 (36, 7)    [s⁻¹]
│       │   ├── A_bund_bund.npy                (7, 7)     [s⁻¹]
│       │   ├── gamma_resolved.npy             (36,)      [s⁻¹]
│       │   └── gamma_bundled.npy              (7,)       [s⁻¹]
│       └── recombination/
│           ├── alpha_RR_resolved.npy          (36, 50)   [cm³/s]
│           ├── alpha_RR_bundled.npy           (7, 50)    [cm³/s]
│           ├── alpha_3BR_resolved.npy         (36, 50)   [cm⁶/s]
│           ├── alpha_3BR_bundled.npy          (7, 50)    [cm⁶/s]
│           ├── Te_grid_recomb.npy             (50,)      [eV]
│           └── recombination_meta.csv         (43 states)
│
├── WEEK1_COMPLETION_REPORT.md
├── WEEK2_COMPLETION_REPORT.md
└── SESSION_REPORT_WEEK3_4.md      ← THIS FILE
```

---

## 6. KEY PHYSICS DECISIONS (permanent record)

All decisions below are locked and must not be changed without explicit derivation.

| Decision | Value | Source | Reason |
|---|---|---|---|
| Statistical weight | ω = 2(2l+1) | hydrogen doublets | spin included throughout |
| De-excitation | always via DB, never direct Maxwell-average | Milne relation | cross-section level DB verified <0.02% |
| dE_eV storage | float64 | — | precision for Boltzmann factor |
| n=10 treatment | bundled | model decision | CCC summed over lf |
| Lotz formula | 1968 Eq.(5) with exp1, no hard threshold | Lotz (1968) | correct form verified vs TICS |
| n_max | 15 | model decision | Griem continuum lowering: n_max=15 at ne=1e15 |
| Te grid | 50 pts log-spaced 1..10 eV | — | direct computation, no interpolation |
| TICS.9 | exact statistical sum of TICS.9S..9K | Bray (personal comm.) | ratio=1.0000 confirmed |
| 3BR exponent | **(h²/2πmekTe)^(3/2)** | Saha equation derivation | blueprint had ^3, wrong |
| RR formula | single-term E1 | Johnson (1972) single term | three-term gives negative for n≥5 at low Te |
| f_pn source | Hoang Binh f_abs column | Osterbrock (1989) Eq.3.17 | Johnson Gaunt formula gives −35% to +330% error |
| Anderson benchmark | use ω=(2S+1)(2L+1)=2(2L+1) | Anderson (2000) | wrong stat weight causes factor-of-2 error |

---

## 7. ANCHOR VALUES (for regression testing)

These values must reproduce exactly on any re-run.

### K_CCC (excitation, cm³/s)

| Transition | Te=1 eV | Te=3 eV | Te=10 eV |
|---|---|---|---|
| 1S→2P | 6.796e-13 | 5.532e-10 | 7.817e-09 |
| 1S→2S | 3.960e-13 | — | — |

### alpha_RR (cm³/s)

| State | Te=1 eV | Te=3 eV | Te=5 eV | Te=10 eV |
|---|---|---|---|---|
| 1S | 1.318e-14 | 2.039e-14 | 2.423e-14 | 2.916e-14 |
| 2P | 1.699e-14 | 2.263e-14 | 2.460e-14 | 2.602e-14 |
| 8J | 8.148e-15 | 7.464e-15 | 6.888e-15 | 5.977e-15 |
| n9 (shell) | 3.447e-14 | 3.079e-14 | 2.813e-14 | 2.415e-14 |

### alpha_3BR (cm⁶/s)

| State | Te=1 eV | Te=3 eV |
|---|---|---|
| 1S | 1.748e-30 | 6.717e-31 |
| 2P | 1.294e-28 | 3.873e-29 |
| n9 (shell) | 2.603e-25 | 3.918e-26 |

### Detailed balance check (Te=3 eV)
```
3BR/K_ion ratio = (g_nl/2) × lambda_th³ × exp(I_n/Te)
1S:  expected=6.6244e-21  computed=6.6244e-21  err=0.00e+00%
2S:  expected=2.0767e-22  computed=2.0767e-22  err=0.00e+00%
2P:  expected=6.2301e-22  computed=6.2301e-22  err=1.11e-14%
```

### f_pn (Hoang Binh, for V&S)

| p→n | f_pn | Error vs NIST |
|---|---|---|
| 1→2 | 0.41620 | 0.00% |
| 1→3 | 0.07910 | 0.00% |
| 2→3 | 0.64075 | +0.01% |
| 8→9 | 1.80730 | — |
| 9→10 | 1.99869 | — |

---

## 8. WHAT'S NEXT — `compute_K_VS.py`

### 8.1 V&S formula (Table II, Eq.17 and Eq.24)

**Excitation K_pn (V&S Eq.17):**
```
K_pn(Te) = 1.6e-7 × (kTe)^0.5 × g_p/g_n
           / (kTe + Γ_pn)
           × [A_pn × ln(0.3×kTe/R + Δ_pn) + B_pn]   cm³/s

where kTe in eV, R = 13.6058 eV
```

**Parameters:**
```
A_pn = (2R/ΔE_pn) × f_pn                     (V&S Eq.11)
B_pn = (from V&S Eq.12, using Johnson b_pn)
Δ_pn = exp(-B_pn/A_pn) + 0.1×E_pn/R         (V&S Eq.22, analogue)
Γ_pn = R×[8+23(s/n)²]×(1+E_pn/R)⁻¹
       ×(8+1.1ps + 0.8/s² + 0.4p^1.5/s^0.5 × |s-1|)⁻¹   (V&S Eq.23)
s = |n - p|
```

**De-excitation K_np (V&S Eq.24):**
```
K_np(Te) = detailed balance:
K_np = K_pn × (g_p/g_n) × exp(ΔE_pn/kTe)
```

**f_pn input:**
```python
# From H_A_E1_LS_n1_15_physical.csv
# f_pn = (1/p²) × Σ_{nl=p, nu=n, |lu-ll|=1} (2×ll+1) × f_abs
```

### 8.2 Transitions V&S must cover

```
res→n9_bund       : 36 rates  (n=1..8 → n=9)
res→n11..15_bund  : 180 rates (n=1..8 → n=11..15, 5 targets)
n9→n10            : 1 rate
n9→n11..15        : 5 rates
n10→n11..15       : 5 rates
n11..15↔n11..15   : 20 rates (10 pairs, both directions)
Total: 247 excitation arrays (de-excitation via DB)
```

### 8.3 Output arrays planned

```
data/processed/collisions/vs/
  K_VS_res_to_n9.npy          (36, 50)   res → n9 bundled
  K_VS_res_to_bund.npy        (180, 50)  res → n11..15 bundled
  K_VS_bund_to_bund_exc.npy   (31, 50)   all bund → bund excitation pairs
  K_VS_metadata.csv           (247, cols)
```

### 8.4 QC plan for V&S

1. All rates positive, no NaN
2. Detailed balance: K_np/K_pn = (g_p/g_n)×exp(ΔE/kTe) to <0.01%
3. Magnitude check: K_VS(8→9) comparable to K_CCC(7→8) (similar ΔE)
4. Te-dependence: K_pn peaks near Te ~ ΔE_pn (physically expected)

---

## 9. REFERENCES USED THIS SESSION

| Paper | Used for |
|---|---|
| Johnson L.C. (1972) ApJ 174, 227 | D constant for RR; established f_pn formula is NOT from here |
| Seaton M.J. (1959) MNRAS 119, 81 | Original RR treatment |
| Griem H.R. (1997) Principles of Plasma Spectroscopy, Eq.6.23-6.25 | 3BR derivation from Saha |
| Vriens & Smeets (1980) Phys.Rev.A 22, 940 | 3BR Eq.9; excitation Table II Eq.17,24 |
| Fujimoto T. (2004) Plasma Spectroscopy, App.4A | l-distribution validity condition |
| Mao J. & Kaastra J. (2016) A&A 587, A84 | l-distribution formula confirmation |
| Osterbrock D.E. (1989) Astrophysics of Gaseous Nebulae, Eq.3.17 | f-A relation |
| Capitelli M. et al. (2016) | RR single-term usage in plasma CR codes |
| Hartgers A. et al. (2001) CPC 135 | RR single-term usage; V&S usage in CR code |

---

*Report generated: 2026-03-15*
*Next session: implement compute_K_VS.py*
