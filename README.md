# hydrogen-cr-qss

**Time-Dependent Collisional-Radiative Modelling of Hydrogen Plasmas:
Quantifying Quasi-Steady-State Breakdown in ITER Divertor Conditions**

M.Tech Thesis — Nikhil Jain
Department of Chemical Engineering, IIT Kanpur (2026)
Supervisor: Prof. Raj Ganesh S. Pala

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What this repository contains

A complete 43-state ℓ-resolved time-dependent collisional-radiative (CR)
model for hydrogen plasmas, built to quantify when and by how much the
quasi-steady-state (QSS) approximation fails in ITER divertor conditions.

The QSS approximation — that excited atomic states equilibrate
instantaneously with the local plasma — underpins nearly all
spectroscopic diagnostics used in fusion research. This code systematically
maps where it breaks down across the ITER divertor parameter space
(T_e = 1–10 eV, n_e = 10¹²–10¹⁵ cm⁻³).

---

## Key results

| Result | Value |
|---|---|
| Memory metric M = τ_QSS / τ_relax | 44–309,326 (M ≥ 44 everywhere) |
| ITER reference: T_e = 3 eV, n_e = 10¹⁴ cm⁻³ | M = 611, τ_QSS = 15.3 μs, τ_relax = 25 ns |
| QSS breakdown during ELM crash (100 μs) | 25.1% of ITER divertor regime |
| QSS breakdown during slow detachment (10 ms) | 59.4% of ITER divertor regime |
| Peak Hα spectroscopic error after ΔT_e = 0.6 eV step | −46% at t = 19 ns |
| Error > 10% duration | 22.3 μs = 1.45 τ_QSS |
| Step error scaling law | ε_step ≈ 1.53 exp(−0.37 T_e) + 0.01 (R² = 0.9947) |

**Main finding:** QSS is self-consistent at steady state (M ≫ 1 everywhere)
but fails significantly during transients. The breakdown is controlled by
electron temperature, not by M. The conventional criterion M ≫ 1 is
necessary but not sufficient for QSS validity during fast events.

---

## Model description

**State space:** 43 bound states
- n = 1–8: 36 fully ℓ-resolved states (each (n,ℓ) pair independent)
- n = 9–15: 7 bundled shells (statistical ℓ-equilibrium assumed)

**Atomic data sources:**

| Process | Source |
|---|---|
| Electron-impact excitation (n=1–8) | CCC database (Bray, Curtin Univ., 2026) |
| Electron-impact excitation (n≥9) | Vriens & Smeets (1980) |
| Electron-impact ionisation | CCC total ionisation cross sections |
| Radiative recombination | Johnson (1972) extending Seaton (1959) |
| Three-body recombination | Detailed balance with ionisation |
| Proton-impact ℓ-mixing (n=2–8) | Badnell et al. (2021) PSM20 |
| Einstein A-coefficients | Hoang Binh (1993) |

**Solver:** Radau implicit Runge-Kutta (scipy), L-stable, 5th order,
supplied with exact Jacobian. Stiffness ratio |λ_max|/|λ_0| ≳ 10⁶.

**Parameter grid:** 50 × 8 = 400 points, log-spaced.

---

## Repository structure

```
src/
├── analysis/
│   ├── assemble_cr_matrix.py   — builds L(Te,ne) rate matrix (43×43)
│   ├── compute_lmix.py         — PSM20 proton ℓ-mixing rates
│   ├── qss_analysis.py         — QSS error metrics (ε_step, ε_res, ε̄, M)
│   ├── plot_results.py         — all thesis figures
│   └── physics_tests.py        — Balmer α spectroscopic test
├── parsers/
│   ├── parse_ccc.py            — CCC cross section parser
│   ├── parse_adas.py           — ADAS SCD96/ACD96 parser
│   └── maxwellian_avg.py       — σ(E) → K(Te) integration
└── rates/
    ├── ionisation.py           — CCC TICS + Lotz validation
    ├── recombination.py        — Johnson RR + 3BR
    └── radiative.py            — Hoang Binh A-coefficients

data/
├── raw/
│   ├── ccc/                    — CCC .dat files (not tracked, see below)
│   └── adas/                   — SCD96/ACD96 files
└── processed/
    ├── cr_matrix/L_grid.npy    — (50,8,43,43) pre-computed matrices
    ├── collisions/             — K_exc, K_deexc, K_ion arrays
    ├── recombination/          — α_RR, α_3BR arrays
    ├── Radiative/              — A_resolved, gamma arrays
    └── lmix/K_lmix.npy        — PSM20 ℓ-mixing rates (43,43,50)

figures/                        — all thesis figures (PNG)
results/                        — breakdown maps, scaling law data
validation/                     — Gate A–E validation scripts
```

---

## Validation gates

All gates run automatically via `validation/validate_gates.py`:

| Gate | Test | Result |
|---|---|---|
| A | Collisional detailed balance | PASS (max error 5×10⁻⁷ %) |
| B | Approach to coronal scaling | PASS (50/50 T_e points, <2% convergence) |
| C | Boltzmann ratio approach | PASS (50/50 monotone, strictly sub-LTE) |
| D | ADAS effective coefficient | DOCUMENTED (valid T_e ≥ 3 eV only; see thesis §4.4) |
| E | Timescale hierarchy M > 1 | PASS (400/400 grid points) |

---

## Quickstart

```bash
git clone https://github.com/jain-sasuke/hydrogen-cr-qss
cd hydrogen-cr-qss
pip install -r requirements.txt

# Run the full pipeline (requires pre-computed data in data/processed/)
bash run_pipeline.sh

# Or step by step:
python src/analysis/compute_lmix.py          # → data/processed/lmix/K_lmix.npy
python src/analysis/assemble_cr_matrix.py    # → data/processed/cr_matrix/L_grid.npy
python validation/validate_gates.py          # → Gate A–E results
python src/analysis/qss_analysis.py          # → breakdown maps
python src/analysis/plot_results.py          # → figures/
python src/analysis/physics_tests.py         # → Balmer α test
```

**Runtime** (Apple M2 Pro): ~30 seconds end-to-end from assembled
L_grid.npy to all figures.

---

## Data availability

**CCC cross sections** (3,117 files, ~800 MB) were provided by
Prof. Igor Bray (Curtin University) as a personal communication (2026)
and are not redistributed here. Contact Prof. Bray directly to request
the e-H_XSEC_LS database.

**ADAS SCD96/ACD96** files are publicly available at
[open.adas.ac.uk](https://open.adas.ac.uk).

All processed NumPy arrays (L_grid.npy, K_exc_full.npy, etc.) are
tracked via Git LFS or available on request.

---

## Dependencies

```
numpy >= 1.26
scipy >= 1.11
matplotlib >= 3.8
pandas >= 2.0
```

No PyTorch or neural network dependencies. Pure NumPy/SciPy scientific
computing.

Install:
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

---

## Citation

If you use this code, please cite the accompanying thesis:

```
Jain, N. (2026). Time-Dependent Collisional-Radiative Modelling of
Hydrogen Plasmas: Quantifying Quasi-Steady-State Breakdown in ITER
Divertor Conditions. M.Tech Thesis, IIT Kanpur.
```

---

## References

1. Bray, I. (2026). CCC electron-hydrogen scattering cross sections.
   Personal communication, Curtin University.
2. Vriens, L. & Smeets, A. H. M. (1980). Phys. Rev. A, 22, 940.
3. Johnson, L. C. (1972). Astrophys. J., 174, 227.
4. Badnell, N. R. et al. (2021). MNRAS, 507, 2922.
5. Hoang Binh, D. & Walmsley, C. M. (1990). A&A, 227, 285.
6. Anderson, H. et al. (2000). J. Phys. B, 33, 1255.
7. Fujimoto, T. (2004). Plasma Spectroscopy. Oxford University Press.
8. Summers, H. P. et al. (2006). Plasma Phys. Control. Fusion, 48, 263.