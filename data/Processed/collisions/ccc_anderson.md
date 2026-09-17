# CCC CROSS SECTIONS — PARSING, MAXWELL AVERAGING, AND THE ANDERSON BENCHMARK

**Project:** Hydrogen collisional-radiative modelling for the ITER divertor
**Original report:** 2026-03-14 (Week 2)
**Revised:** 2026-09-16 — re-benchmarked against the **Anderson et al. (2002) corrigendum**,
extended to the full Te range and the full (Te, nₑ) grid.

> **What changed in this revision.** The Week-2 benchmark used table 2 of
> **Anderson et al. (2000)**. That table was **superseded** by the 2002
> corrigendum, which is the version now used. Several pipeline facts recorded
> in the original report (table shapes, Te grid, transition counts) had also
> gone stale as the pipeline grew; they are corrected below and marked
> **[corrected]**. The superseded Week-2 original is the previous committed
> revision of this file; recover it with `git log --follow -p --
> data/Processed/collisions/ccc_anderson.md`.

---

## EXECUTIVE SUMMARY

CCC electron-impact excitation cross sections (Prof. Igor Bray, personal
communication 2026) were parsed, validated, Maxwell-averaged to rate
coefficients K(Te), and benchmarked against the R-matrix-with-pseudostates
(RMPS) effective collision strengths of Anderson et al.

Against the **2002 corrigendum**, over the thesis Te grid (50 points, 1–10 eV):

| Subset | comparisons | within 20% | mean \|err\| |
|---|---|---|---|
| n_upper ≤ 3 | 550 | **97.5%** | **8.9%** |
| n_upper ≤ 4 | 1750 | **81.4%** | **13.1%** |
| n_upper = 5 | 2500 | 12.3% | 41.1% |

The pipeline itself is clean: the stored K table reproduces an independent
re-average of the same cross sections to **0.000%**, and detailed balance
closes to **2.0×10⁻¹³ %**. The CCC-vs-RMPS spread is a genuine method
difference, not a processing error.

**The headline correction to the Week-2 conclusion.** Week 2 argued that the
n=5 disagreement "does not affect the thesis" because n=5 channels are weak.
The first part of that is right and the reasoning is wrong. Propagating the
substitution CCC → Anderson through the CR matrix over the whole (Te, nₑ)
grid (§7) shows the timescale observables move by **~11%**, and that shift is
driven almost entirely by **ground-state excitation at n ≤ 4**, not by n=5.
n=5 alone moves M by −0.9% at the benchmark point; ground-state transitions
alone move it by −12.4%. The conclusion "n=5 does not matter" survives; the
claim "therefore the benchmark spread does not matter" does not.

---

## 1. WHICH ANDERSON PAPER

Two papers, and they are not interchangeable:

| | |
|---|---|
| **Anderson, Ballance, Badnell & Summers (2000)**, *J. Phys. B* **33** 1255 | the RMPS calculation |
| **Anderson, Ballance, Badnell & Summers (2002)**, *J. Phys. B* **35** 1613 | **CORRIGENDUM** — table 1 here *replaces* table 2 of the 2000 paper |

Dere & Mason pointed out that the 2000 dipole effective collision strengths do
not approach the correct high-energy asymptotic form. The underlying collision
strengths were sound; the fault was in the Maxwell averaging. A least-squares
fit was made to the high-energy collision strengths using the infinite-energy
limit point, but RMPS collision strengths oscillate shallowly at all energies
because of high-lying pseudo-thresholds, so the fit degraded progressively
beyond the last computed energy. The 2002 table recomputes Υ using simple
linear interpolation of the reduced collision strength between the highest
finite energy and the infinite-energy limit — the Born limit for non-dipole
transitions, which was not available in 2000.

**Use the 2002 values and cite the corrigendum.** The 2000 numbers are
formally superseded.

**Provenance note.** `refs/` holds the 2002 corrigendum only. The 2000 paper is
not in the repository. Any statement sourced to the 2000 paper — including the
"n=6 orbitals would require a 140 a₀ R-matrix box" remark that appeared in the
Week-2 version of this document — is therefore **unverified in-repo** and must
not enter the thesis until the 2000 paper is obtained. The n=5 argument in §6
below is built from repository evidence instead and does not depend on it.

### 1.1 Impact of the corrigendum

Υ(2002) vs Υ(2000), all 85 transitions, parsed from the PDF and compared
against the table hardcoded in `anderson_benchmark_qc.py`:

| Te [eV] | mean \|Δ\| | max \|Δ\| | worst transition |
|---|---|---|---|
| 0.5 | 3.58% | 6.88% | 1S→5G |
| 1.0 | 1.51% | 4.40% | 1S→5G |
| 3.0 | 0.48% | 2.15% | 1S→5G |
| 5.0 | 0.58% | 1.93% | 2S→5S |
| 10.0 | 3.19% | 13.10% | 1S→3P |
| 15.0 | 6.27% | 33.13% | 1S→3P |
| 20.0 | 9.28% | 58.44% | 1S→3P |
| 25.0 | 12.09% | 86.18% | 1S→3P |

Over the thesis range (1–10 eV) the mean change is **1.44%** and 99.4% of
points move by <10%, consistent with the corrigendum's own statement. The
benchmark verdict barely moves: n_upper ≤ 4 goes from 82.1% → **79.3%** within
20% (mean 12.7% → **13.3%**) on the four Anderson temperatures.

**The corrigendum matters enormously above 15 eV** (up to 86% at 25 eV, on
dipole transitions). This work stays below 10 eV, so it is unaffected — but
any future extension of this model to attached-divertor or SOL temperatures
must use the 2002 table.

---

## 2. CCC DATA: CORRECTIONS FROM IGOR BRAY

Three corrections from Prof. Bray (email 2026-02-22), all still in force.

### 2.1 Filename convention: FINAL.INITIAL

File `1S.2P` holds the **2p→1s de-excitation** cross section, not 1s→2p
excitation. The filename reads right-to-left: final state first, initial
second. Excitation data are obtained by detailed balance from the
de-excitation files.

### 2.2 Same-n transitions excluded

Transitions with n_final = n_initial (2S↔2P, 3S↔3P, …) have unphysically large
cross sections from non-relativistic degeneracy in the CCC calculation. All
Δn=0 transitions are excluded. **Verified 2026-09-16:** zero Δn=0 keys remain
in `ccc_crosssections.csv`.

### 2.3 Data quality

For Δn≠0 transitions CCC accuracy is ~5% (Bray, personal communication). The
Anderson benchmark is consistent with this for n_upper ≤ 3 (§5).

### 2.4 Current parse state **[corrected]**

| Quantity | Week-2 report | Verified 2026-09-16 |
|---|---|---|
| Raw Bray files | 3,117 | **3,115** |
| Rows in `ccc_crosssections.csv` | 90,914 | **112,064** |
| Unique (nℓ→n'ℓ') blocks | 1,740 | **2,190** |
| — excitation direction | 870 | **1,320** |
| — de-excitation direction | 870 | **870** |
| n coverage | n ≤ 8 | **n_i 1–9, n_f 2–10** |

The parse was evidently re-run and extended after Week 2. The numbers above
are what the files now contain.

---

## 3. MAXWELL AVERAGING: PHYSICS AND IMPLEMENTATION

### 3.1 Formula

$$K(T_e) = \sqrt{\frac{8}{\pi m_e}}\,(kT_e)^{-3/2} \int_{\Delta E}^{\infty} \sigma(E)\, E\, e^{-E/kT_e}\, dE$$

E is the incident electron kinetic energy, ΔE the excitation threshold, σ(E)
the CCC cross section in m².

**K(Te) has no nₑ dependence.** It is a property of the cross section and the
electron distribution alone. Density enters only when these rates are
assembled into the CR matrix — see §7.

### 3.2 Implementation decisions

| Decision | Choice | Reason |
|---|---|---|
| Integration grid | 5,000-point uniform linspace [ΔE+10⁻⁴, E_max] | <2% error; 500 points gives ~17% at Te=1 eV |
| Threshold ΔE | Bohr formula, 13.6058×(1/n_i²−1/n_f²) eV | not from data; avoids noisy near-threshold sampling |
| Interpolation | `np.interp`, left=0.0, right=0.0 | zero outside the CCC data range |
| Units | SI throughout; convert to cm³/s at the end | avoids unit errors |
| De-excitation | detailed balance only; never average σ_deexc directly | guarantees the CR matrix satisfies detailed balance |

### 3.3 Te grid **[corrected]**

The Week-2 report recorded a 12-point grid. The pipeline now uses a **50-point
logarithmic grid**, and it is identical to `Te_grid_L.npy` used by the CR
matrix — verified by `np.allclose` in `ccc_anderson_grid_impact.py`:

```python
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)   # eV
# Te[23] = 2.9470517 eV  <- the CLAUDE.md benchmark point
```

Do not re-hardcode a 12-point grid anywhere.

### 3.4 De-excitation via detailed balance

$$K_{j\to i}(T_e) = K_{i\to j}(T_e)\,\frac{\omega_i}{\omega_j}\,\exp\!\left(\frac{+\Delta E_{ij}}{kT_e}\right)$$

with ω = 2(2ℓ+1) for hydrogen doublets and ΔE > 0.

Why not average σ_deexc directly: (1) it guarantees populations converge to
Saha-Boltzmann at high density; (2) it avoids near-threshold noise in σ_deexc;
(3) the CCC cross sections are self-consistent at the cross-section level.

---

## 4. INTERNAL QC — RE-RUN 2026-09-16

| Check | Description | Result |
|---|---|---|
| A | All K_exc > 0 | PASS |
| B | Detailed balance, max roundtrip error | PASS (**1.998×10⁻¹³ %**) |
| C | 1S→2P at Te = 2.947 eV | **5.5316×10⁻¹⁰ cm³/s** |
| D | No NaN/Inf in either table | PASS |
| E | Stored K table vs independent re-average of the same σ | **max \|diff\| 0.000%** (n=170, Te = 1 and 10 eV) |

Check E is new and is the important one: it separates *pipeline error* from
*method difference*. The stored `K_CCC_exc_table.npy` reproduces a fresh
Maxwell average computed from `ccc_crosssections.csv` by a completely separate
script, to the printed precision. Everything in §5–§7 is therefore physics.

**Table shapes [corrected]:** `K_CCC_exc_table.npy` and
`K_CCC_deexc_table.npy` are **(1320, 50)**, not (870, 12).

**Bug fixed during Week-2 development:** ΔE stored rounded to 5 decimal places
caused a spurious 4.44×10⁻⁴% detailed-balance error. Fixed by storing ΔE at
full float64 precision. Do not round `dE_eV`.

---

## 5. THE ANDERSON 2002 BENCHMARK

### 5.1 Method

Anderson's Υ are converted to excitation rate coefficients by their Eq. (3):

$$q_{\text{exc}} = \frac{2\sqrt{\pi}\,\alpha c a_0^2}{\omega_{\text{lower}}}\sqrt{\frac{I_H}{kT_e}}\exp\!\left(\frac{-\Delta E}{kT_e}\right)\Upsilon$$

The prefactor evaluates to **2.17166×10⁻⁸ cm³/s** (paper quotes 2.1716×10⁻⁸),
and ω_lower = (2S+1)(2L+1) = 2(2L+1) is the statistical weight of the **lower**
state.

**Table convention.** Rows are listed `i j` with i = upper, j = lower. This is
now *verified at runtime* rather than assumed: row `3 1` carries
A_ij = 6.270×10⁸ s⁻¹ = A(2p→1s) (NIST: 6.2649×10⁸). The script raises if this
check fails.

**Two traps, both live:**

1. **ω = 2L+1 instead of (2S+1)(2L+1)** introduces a systematic factor of 2.
   Caught in Week 2; still the first thing to check if results look doubled.
2. **Te-grid mismatch.** Anderson tabulates at Te = 0.5, 1, 3, 5, 10, 15, 20,
   25 eV. Of these, only **1 and 10 eV** lie on the 50-point CCC grid; 3 and 5
   eV do not (nearest grid points 2.947 and 4.942 eV). The Week-2 script
   matched by `argmin`, i.e. it compared K at 2.947 eV against Υ at 3.0 eV.
   The current scripts either restrict to exactly-matching temperatures or
   log-log interpolate Υ in Te, and flag which was used.

Υ values are **parsed from the PDF**, not retyped, with integrity checks
(85 rows, no Δn=0 pairs, i>j throughout).

### 5.2 Anchor spot checks

| Transition | Te | Υ (2002) | K_Anderson | Expected | |
|---|---|---|---|---|---|
| 1s→2p | 1.0 eV | 0.5290 | 7.841×10⁻¹³ cm³/s | [6×10⁻¹³, 9×10⁻¹³] | OK |
| 1s→2s | 1.0 eV | 0.2960 | 4.388×10⁻¹³ cm³/s | [3×10⁻¹³, 5×10⁻¹³] | OK |
| 2s→3p | 1.0 eV | 3.0500 | 1.846×10⁻⁸ cm³/s | [1×10⁻⁸, 3×10⁻⁸] | OK |

### 5.3 Results on the four Anderson temperatures (Te = 1, 3, 5, 10 eV)

340 comparisons, 85 transitions.

| Subset | n | within 20% | mean \|err\| | median signed | verdict |
|---|---|---|---|---|---|
| All matched | 340 | 40.6% | 29.4% | −21.2% | FAIL |
| n_upper ≤ 4 | 140 | 79.3% | 13.3% | −8.1% | PARTIAL |
| n_upper ≤ 4, excited-state only | 104 | 85.6% | 11.1% | −5.1% | PASS |
| n_upper = 5 | 200 | 13.5% | 40.6% | −32.6% | FAIL |

Resolved by upper shell — the degradation is **graded, not a cliff at n=5**:

| n_upper | n | within 20% | mean \|err\| | median signed |
|---|---|---|---|---|
| 2 | 8 | **100.0%** | 10.5% | −10.4% |
| 3 | 36 | **97.2%** | 8.2% | −5.1% |
| 4 | 96 | 70.8% | 15.5% | −9.9% |
| 5 | 200 | 13.5% | 40.6% | −32.6% |

Within n_upper = 4, the spread is concentrated in Δn = 1 and Δn = 3:

| Δn | n | within 20% | mean \|err\| |
|---|---|---|---|
| 1 (3ℓ→4ℓ′) | 48 | 68.8% | 15.2% |
| 2 (2ℓ→4ℓ′) | 32 | **100.0%** | 10.2% |
| 3 (1s→4ℓ′) | 16 | 18.8% | 26.9% |

Worst points at n_upper ≤ 4: 3S→4P **+65.7%** (10 eV), 1S→4F **−52.5%**
(10 eV), 3P→4D **+48.2%** (10 eV).

### 5.4 Full Anderson Te range, 0.5 – 25 eV

CCC cross sections re-averaged at all eight Anderson temperatures.

| Te [eV] | all: within 20% | all: mean \|err\| | n≤4: within 20% | n≤4: mean \|err\| |
|---|---|---|---|---|
| 0.5 | 42.4% | 28.4% | **97.1%** | **7.1%** |
| 1.0 | 42.4% | 27.5% | **97.1%** | **7.2%** |
| 3.0 | 37.6% | 30.3% | 82.9% | 12.7% |
| 5.0 | 38.8% | 30.6% | 71.4% | 15.6% |
| 10.0 | 43.5% | 29.2% | 65.7% | 17.9% |
| 15.0 | 49.4% | 27.4% | 68.6% | 17.8% |
| 20.0 | 52.9% | 25.9% | 68.6% | 17.5% |
| 25.0 | 52.9% | 24.9% | 65.7% | 17.3% |

Agreement is **best at the cold end** and degrades monotonically to ~10 eV,
then flattens. At Te ≤ 1 eV the n≤4 agreement is excellent (97% within 20%,
mean 7%) — the coldest part of the divertor regime is the best-validated part.

Anchor transitions across the full range (% error, CCC vs Anderson):

| Transition | 0.5 | 1 | 3 | 5 | 10 | 15 | 20 | 25 eV |
|---|---|---|---|---|---|---|---|---|
| 1s→2p | −15.1 | −13.3 | −10.4 | −8.8 | −5.4 | −1.5 | +1.9 | +5.2 |
| 1s→3p | −11.3 | −13.2 | −18.0 | −19.5 | −16.0 | −10.7 | −5.7 | −1.3 |
| 2p→3d | −8.4 | −5.1 | −3.3 | −2.4 | +3.2 | +8.7 | +12.6 | +15.6 |
| 2s→3p | −2.9 | −3.8 | −1.7 | +1.4 | +8.9 | +13.6 | +17.3 | +20.1 |

All four stay inside the combined CCC (~5%) ⊕ RMPS (~10%) uncertainty band
across the thesis range.

### 5.5 Full 50-point thesis Te grid

Υ log-log interpolated onto `Te_grid_L` (interpolation only — 1–10 eV lies
inside Anderson's 0.5–25 eV range), compared against the stored K table on its
own native grid. 85 × 50 = 4,250 comparisons.

| Subset | n | within 20% | mean \|err\| | median signed |
|---|---|---|---|---|
| all | 4250 | 40.7% | 29.6% | −21.5% |
| n_upper ≤ 3 | 550 | **97.5%** | **8.9%** | −6.9% |
| n_upper ≤ 4 | 1750 | **81.4%** | **13.1%** | −8.8% |
| n_upper = 5 | 2500 | 12.3% | 41.1% | −34.2% |

Te-stability across the grid:

- n_upper ≤ 4: mean \|err\| ranges **7.16% (Te = 1.00 eV) → 17.87% (Te = 10.00 eV)**
- n_upper = 5: mean \|err\| ranges 37.20% (10 eV) → 42.60% (2.68 eV)

The n≤4 agreement is not uniform — it roughly doubles across the grid. Quote
13% as a grid-average, not as a bound.

---

## 6. HOW FAR UP IN n IS THE BENCHMARK USABLE?

### 6.1 Coverage

The corrigendum table contains **15 ℓ-resolved levels, n = 1…5 (1s…5g)**, all
85 Δn≠0 pairs. The CCC database reaches n = 10. **Transitions with n_upper ≥ 6
have no Anderson counterpart and are simply unvalidated by this benchmark.**

### 6.2 Which dataset is anomalous at n = 5?

CCC can continue a Rydberg series past n=5; Anderson cannot. That asymmetry
gives a discriminator. For a dipole series n_lo → n_up, the hydrogenic
oscillator strength obeys f ~ C/n_up³ exactly as n_up → ∞, and at fixed Te
above threshold the Bethe term dominates, so

$$R(n_{up}) = K(n_{lo} \to n_{up})\cdot n_{up}^3$$

must approach a constant. A dataset whose R(n_up) is smooth through the series
is internally consistent; one that kinks at a single n is anomalous *at that n*.

Successive percentage changes in R(n_up), Te = 1 eV:

| series | 2→3 | 3→4 | **4→5** | 5→6 | 6→7 |
|---|---|---|---|---|---|
| 1s→np CCC | −87.8 | −53.1 | **−41.5** | −16.3 | −11.8 |
| 1s→np **Anderson** | −87.9 | −53.5 | **−5.4** | — | — |
| 2p→nd CCC | — | −70.4 | **−46.5** | −28.0 | −17.0 |
| 2p→nd **Anderson** | — | −70.4 | **−17.3** | — | — |
| 2s→np CCC | — | −63.1 | **−44.2** | −24.0 | −18.6 |
| 2s→np **Anderson** | — | −63.8 | **−17.5** | — | — |

CCC tracks a smooth monotone approach to the hydrogenic limit through n = 10.
Anderson tracks it identically until the step **into n = 5**, where it kinks —
Υ(n=5) sits high against its own series, in three independent series (and in
3d→nf), at both 1 and 10 eV. The kink lands on the top physical shell of the
RMPS target basis, the expected signature of the topmost shell absorbing flux
belonging to n ≥ 6 and the continuum. This is consistent with the bulk sign of
the n=5 disagreement: median signed error −32.6%, i.e. CCC *below* Anderson.

**Caveat, stated deliberately.** This localizes where a dataset stops behaving
hydrogenically; it is not absolute proof that CCC is right. The n³ argument is
asymptotic and is weakest at Te = 1 eV. What it does establish is that the
anomaly sits at Anderson's n = 5, not spread through CCC.

### 6.3 The opposite mode

Ten points disagree by more than 100%, and **all ten are Δn = 1 into n = 5**,
with CCC *above* Anderson:

| Transition | worst \|err\| |
|---|---|
| 4S→5P | +193.6% (5 eV) |
| 4P→5D | +145.7% (10 eV) |
| 4D→5F | +124.0% (10 eV) |
| 4P→5S | +111.2% (10 eV) |

These are dipole-allowed transitions between near-degenerate shells
(ΔE ≈ 0.31 eV), where near-threshold long-range dipole coupling dominates and
the two methods genuinely differ. Both failure modes are real and they have
opposite signs; neither method is established as correct here.

### 6.4 Working statement

| n_upper | status |
|---|---|
| ≤ 3 | **tightly validated** — 97.5% within 20%, mean 8.9% |
| 4 | **validated with a caveat** — 70.8% within 20%, mean 15.5%; Δn=1 and Δn=3 carry the spread |
| 5 | **method disagreement**, evidence points at Anderson's basis top; CCC not refuted |
| ≥ 6 | **unvalidated** — no Anderson data exists |

---

## 7. (Te, nₑ) GRID IMPACT ON THE CR MODEL

This is where nₑ enters. A discrepancy in a rate coefficient matters only
insofar as it moves an observable.

### 7.1 Method

`L_grid.npy` is **linear in nₑ to 3.1×10⁻¹⁶ (relative)** — verified, not
assumed — so it decomposes exactly as

$$L(T_e, n_e) = R(T_e) + n_e\,C(T_e)$$

with R radiative and C collisional. Anderson's rate coefficients were
substituted for CCC's in C for **all 85 transitions Anderson covers**, L was
rebuilt, and the timescales re-solved over all 50 × 8 grid points.
Excitation and de-excitation elements of each transition are scaled by the
**same** factor, so detailed balance is preserved; diagonals are compensated so
column sums are unchanged (drift 1.1×10⁻¹⁷), leaving the continuum sink intact.
`L_grid.npy` is read only — the perturbed matrix exists in memory.

**Baseline validation.** The eigenvalue extraction reproduces the CLAUDE.md
reference values exactly:

| Quantity | recorded | this script |
|---|---|---|
| τ_QSS | 22.73 µs | **22.728 µs** |
| τ_relax | 2.277 ns | **2.2769 ns** |
| M | 9982 | **9981.9** |

### 7.2 Result

Substituting CCC → Anderson wherever Anderson has data, over the whole grid:

| Observable | mean \|Δ\| | max \|Δ\| | where max occurs |
|---|---|---|---|
| τ_QSS | **11.42%** | 18.75% | Te = 1.00 eV, nₑ = 5.2×10¹³ |
| τ_relax | 2.39% | 7.97% | Te = 1.05 eV, nₑ = 1.4×10¹⁴ |
| M = τ_QSS/τ_relax | **11.08%** | 16.59% | Te = 1.33 eV, nₑ = 3.7×10¹⁴ |

At the CLAUDE.md benchmark point (Te = 2.947 eV, nₑ = 1.389×10¹⁴):

| | CCC | Anderson-substituted | Δ |
|---|---|---|---|
| τ_QSS | 22.728 µs | 19.627 µs | **−13.64%** |
| τ_relax | 2.2769 ns | 2.2166 ns | −2.65% |
| M | 9981.9 | 8854.8 | **−11.29%** |

### 7.3 nₑ dependence

| nₑ [cm⁻³] | mean \|Δτ_QSS\| | mean \|Δτ_relax\| | mean \|ΔM\| |
|---|---|---|---|
| 1.00×10¹² | 6.31% | 0.14% | 6.18% |
| 2.68×10¹² | 8.52% | 2.19% | 10.06% |
| 7.20×10¹² | 10.42% | 4.00% | 12.72% |
| 1.93×10¹³ | 11.97% | 2.22% | 12.03% |
| 5.18×10¹³ | 13.03% | 3.43% | 12.11% |
| 1.39×10¹⁴ | 13.59% | 3.53% | 11.24% |
| 3.73×10¹⁴ | 13.84% | 1.33% | 12.68% |
| 1.00×10¹⁵ | 13.65% | 2.28% | 11.64% |

Sensitivity **rises with density and saturates above ~10¹³ cm⁻³**, as expected:
at low nₑ radiative terms dominate L and the collisional rates matter less; at
high nₑ the collisional block dominates and the rate uncertainty passes
straight through. The ITER partially-detached regime sits in the saturated
part of this curve.

### 7.4 Te dependence

| Te [eV] | mean \|Δτ_QSS\| | mean \|Δτ_relax\| | mean \|ΔM\| |
|---|---|---|---|
| 1.000 | 17.02% | 3.39% | 14.09% |
| 1.389 | 15.92% | 2.84% | 13.45% |
| 1.931 | 14.20% | 2.08% | 12.42% |
| 2.683 | 12.01% | 1.88% | 11.11% |
| 3.728 | 10.35% | 1.62% | 10.44% |
| 5.179 | 8.83% | 2.05% | 9.76% |
| 7.197 | 7.26% | 2.83% | 9.19% |
| 10.000 | 5.61% | 3.20% | 8.19% |

Sensitivity is **largest at the cold end** — the opposite of the rate-level
agreement, which was *best* at Te ≤ 1 eV (§5.4). At low Te the excitation rates
sit deep in the Boltzmann exponential tail, so the CR eigenvalues are far more
responsive to a given fractional change in K.

### 7.5 Which transitions carry the impact

Substituting only selected subsets:

| Scenario | transitions | mean \|Δτ_QSS\| | ΔM at benchmark point |
|---|---|---|---|
| all 85 (n_upper ≤ 5) | 85 | 11.42% | −11.29% |
| only n_upper ≤ 3 | 11 | 4.54% | −6.97% |
| only n_upper = 4 | 24 | 2.97% | −4.37% |
| only n_upper ≤ 4 | 35 | 7.25% | **−10.82%** |
| only n_upper = 5 | 50 | 4.72% | **−0.94%** |
| only ground-state (1s→nℓ) | 14 | **10.45%** | **−12.36%** |

**This inverts the Week-2 conclusion's reasoning.** Fifty n=5 transitions move
M by −0.94% at the benchmark point; fourteen ground-state transitions move it
by −12.36%. τ_QSS is set by the slowest CR mode, which is ground-state
depletion, so 1s→nℓ rates control it almost exclusively — and those are exactly
the transitions the benchmark validates *least* well among n ≤ 4 (1s→4F is
−52.5%, Δn=3 out of the ground state is 18.8% within 20%).

The practical consequence: **the n=5 disagreement is genuinely irrelevant to
the thesis observables, but not for the reason originally given, and the n≤4
agreement is not good enough to treat M as better than ~±11%.**

---

## 8. KNOWN LIMITATIONS

| Limitation | Scope | Impact |
|---|---|---|
| n_upper=5: CCC vs RMPS differ by factor 2–5 | 50 transitions | Moves M by <1% at the benchmark point — genuinely negligible |
| n_upper≤4 agreement is 13% (grid mean), 18% at 10 eV | 35 transitions | **Propagates to ~11% on τ_QSS and M** — must be quoted as a systematic |
| Ground-state Δn=3 (1s→4ℓ) poorly matched | 4 transitions | Largest lever on τ_QSS per §7.5 |
| n_upper ≥ 6 has no Anderson counterpart | all n≥6 CCC data | Unvalidated; no independent check exists in-repo |
| No Δn=0 transitions (CCC limitation) | ℓ-mixing | Handled by the statistical-equilibrium assumption |
| CCC ends at n=10; n=11–15 bundled | high-n | Bundled states, Vriens-Smeets scaling |
| Anderson 2000 paper not in `refs/` | provenance | Claims sourced to it are unverified |

---

## 9. THESIS STATEMENT (Chapter 3)

> Electron-impact excitation cross sections for all ℓ-resolved transitions with
> Δn≠0 were obtained from the Convergent Close Coupling (CCC) database (Bray,
> personal communication 2026) and Maxwell-averaged to rate coefficients K(Tₑ)
> on the 50-point logarithmic temperature grid spanning 1–10 eV used throughout
> this work; de-excitation coefficients follow by detailed balance, which closes
> to 2×10⁻¹³ %. The data were benchmarked against the R-matrix-with-pseudostates
> effective collision strengths of Anderson et al. (2000), as revised by the
> 2002 corrigendum. Agreement is excellent for transitions into n ≤ 3 (97.5% of
> 550 comparisons within 20%, mean absolute deviation 8.9%) and acceptable into
> n = 4 (81.4% within 20%, mean 13.1%), degrading from 7.2% at Tₑ = 1 eV to
> 17.9% at Tₑ = 10 eV. Transitions into n = 5 disagree by factors of 2–5; a
> Rydberg-series test using CCC data at n = 6–10, which the RMPS calculation
> does not extend to, localizes the anomaly at the top shell of the RMPS target
> basis rather than in the CCC data. Propagating the substitution of Anderson's
> coefficients for CCC's through the collisional-radiative matrix over the full
> (Tₑ, nₑ) grid shifts τ_QSS and M by 11% on average, a shift carried almost
> entirely by ground-state excitation at n ≤ 4 rather than by the n = 5
> discrepancy. The quasi-steady-state breakdown metric M is therefore reported
> with an atomic-data systematic of ±11%; since M exceeds 10³ throughout the
> regime studied, the timescale-separation conclusions are unaffected in sign
> or order of magnitude.

**Do not** reuse the Week-2 sentence claiming the n=5 discrepancy is the reason
the benchmark does not affect the thesis. §7.5 shows that reasoning is wrong
even though the conclusion holds.

---

## 10. FILES

### Scripts

```
src/parsers/parse_ccc.py                     raw Bray file parser
src/parsers/qc_ccc.py                        QC checks 1-4
src/rates/compute_K_CCC.py                   Maxwell averaging + detailed balance
src/validation/anderson_benchmark_qc.py      Week-2 benchmark, Anderson 2000  [SUPERSEDED]
src/validation/anderson2002_benchmark.py     2002 corrigendum benchmark        <- use this
src/validation/anderson_validity_range.py    coverage + Rydberg-series n-scaling test
src/validation/ccc_anderson_grid_impact.py   full Te range, 50-pt grid, (Te,ne) impact
```

`anderson_benchmark_qc.py` is retained for traceability. It hardcodes the 2000
Υ table and matches temperatures by nearest neighbour; do not use it for new
results.

### Data

```
data/processed/collisions/ccc/
    ccc_crosssections.csv                    112,064 rows, 2,190 blocks
    K_CCC_exc_table.npy                      (1320, 50) float64 [cm^3/s]
    K_CCC_deexc_table.npy                    (1320, 50) float64 [cm^3/s]
    K_CCC_metadata.csv                       (1320, 12) transition labels
    Te_grid.npy                              (50,) eV, == Te_grid_L.npy
data/processed/collisions/
    anderson2002_upsilon_parsed.csv          Upsilon parsed from the PDF
    anderson_2000_vs_2002_upsilon.csv        corrigendum impact
    ccc_vs_anderson2002_benchmark.csv        340 rows, Te = 1,3,5,10 eV
    ccc_vs_anderson2002_full_Te_range.csv    680 rows, Te = 0.5-25 eV
    ccc_vs_anderson2002_thesis_Te_grid.csv   4,250 rows, 50-point grid
    anderson_validity_dipole_series.csv      Rydberg-series n-scaling test
    ccc_anderson_grid_impact.csv             400 (Te,ne) points, timescale shifts
figures/
    ccc_vs_anderson2002_benchmark.png
    ccc_anderson_grid_impact.png
```

### Loading

```python
import numpy as np, pandas as pd

K_exc   = np.load('data/processed/collisions/ccc/K_CCC_exc_table.npy')    # (1320, 50)
K_deexc = np.load('data/processed/collisions/ccc/K_CCC_deexc_table.npy')  # (1320, 50)
Te_grid = np.load('data/processed/collisions/ccc/Te_grid.npy')            # (50,) eV
meta    = pd.read_csv('data/processed/collisions/ccc/K_CCC_metadata.csv')

row = meta[(meta.n_i==1)&(meta.l_i==0)&(meta.n_f==2)&(meta.l_f==1)]
K_1S2P = K_exc[row.idx.values[0], :]      # (50,) cm^3/s
```

Metadata columns: `idx, n_i, l_i, l_i_char, n_f, l_f, l_f_char, omega_i,
omega_f, dE_eV, E_max_eV, n_raw_points`. **`dE_eV` is full float64 — do not
round it.**

### Reference

`refs/H_Anderson_2002_J._Phys._B__At._Mol._Opt._Phys._35_1613.pdf`
— H Anderson, C P Ballance, N R Badnell, H P Summers, *Corrigendum*,
J. Phys. B: At. Mol. Opt. Phys. **35** (2002) 1613–1615.
Original: *ibid.* **33** (2000) 1255–62 — **not in `refs/`**.

---

## 11. STATUS

| Item | Status |
|---|---|
| CCC parsing with Bray corrections | verified 2026-09-16 |
| QC checks A–D | PASS |
| Pipeline self-consistency (stored K vs re-average) | PASS, 0.000% |
| Benchmark vs Anderson **2002** | PASS for n_upper ≤ 4; n_upper ≤ 3 tight |
| Full Te range 0.5–25 eV | complete |
| Full 50-point thesis Te grid | complete |
| (Te, nₑ) propagation to CR observables | complete — ±11% systematic on M |
| n = 5 anomaly localized | evidence points to RMPS basis top |
| n ≥ 6 validation | **not possible** — no reference data |
| Anderson 2000 paper in `refs/` | **outstanding** |

**Revised:** 2026-09-16
**Author:** Nikhil (M.Tech Chemical Engineering, IIT Kanpur)
**Project:** Quasi-Steady-State Approximation Breakdown in Hydrogen Plasma:
A Time-Dependent Collisional-Radiative Model for ITER Divertor Conditions
