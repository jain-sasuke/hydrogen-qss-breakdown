
# Non-Markovian Collisional-Radiative Kinetics

**Author:** Nikhil Jain
**Institution:** IIT Kanpur, Chemical Engineering 


## Overview

Master's thesis implementing non-Markovian collisional-radiative framework for hydrogen plasmas with physics-informed neural operators for spectroscopic diagnostics.

## Structure
```
src/          - Source code (CR model, memory kernel, neural operator)
data/         - Atomic data (Einstein A, oscillator strengths)
tests/        - Unit tests and validation
notebooks/    - Jupyter notebooks for analysis
figures/      - Publication-quality figures
docs/         - Documentation
results/      - Simulation results
```

## Status

- [x] Radiative transition matrix (1400+ transitions, n_max=15)
- [ ] Collision rates (ionization, excitation, recombination)
- [ ] Memory kernel implementation
- [ ] Neural operator training
- [ ] Validation suite
- [ ] Thesis writing

## Dependencies
```
numpy
scipy
matplotlib
pandas
torch (for neural operator)
```

## Installation
```bash
pip install numpy scipy matplotlib pandas torch
```

## Usage

Coming soon.

## References

- Vriens & Smeets (1980), Phys. Rev. A
- Johnson (1972), Astrophys. J.
- Hoang Binh et al. (atomic data)
