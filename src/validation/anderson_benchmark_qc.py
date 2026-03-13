"""
Anderson (2000) Benchmark QC — Check 5
=======================================
Compares K_CCC (Maxwell-averaged from Bray CCC cross sections)
against K_Anderson (converted from RMPS effective collision strengths).

Anderson Table 2 convention:
  Row: i_table  j_table  Aij  Upsilon(Te)
  i_table = UPPER state, j_table = LOWER state  (Aij confirms: upper->lower radiation)

Anderson Eq.(3) gives EXCITATION rate:
  q_exc = C / omega_lower * sqrt(IH/kTe) * exp(-dE/kTe) * Upsilon
  where omega_lower = (2S+1)(2L+1) = 2(2L+1) for hydrogen doublets
  and C = 2*sqrt(pi)*alpha*c*a0^2 = 2.1716e-8 cm^3/s

Thesis Te range: 1.0 -- 10.0 eV (4 Anderson Te columns)
Full range: 0.5 -- 25.0 eV (8 columns, for reference)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Constants ─────────────────────────────────────────────────────────────────
eV_to_J  = 1.60218e-19
me       = 9.10938e-31
a0_m     = 5.29177e-11       # Bohr radius [m]
a0_cm    = 5.29177e-9        # Bohr radius [cm]
alpha    = 7.29735e-3
c_cgs    = 2.99792e10        # [cm/s]
IH_eV    = 13.6058           # Rydberg [eV]

# Anderson Eq.(3) prefactor: 2*sqrt(pi)*alpha*c*a0^2
C_AND = 2.0 * np.sqrt(np.pi) * alpha * c_cgs * a0_cm**2
print(f"C_anderson = {C_AND:.6e} cm^3/s  (paper: 2.1716e-8)")

# Anderson Te grid [eV]
TE_AND = np.array([0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0])

# Thesis range indices (Te = 1, 3, 5, 10 eV)
THESIS_TE_IDX = [1, 2, 3, 4]   # indices into TE_AND
THESIS_TE     = TE_AND[THESIS_TE_IDX]

# ── Table 1: index → (n, l) ───────────────────────────────────────────────────
IDX_TO_NL = {
    1: (1,0),  2: (2,0),  3: (2,1),
    4: (3,0),  5: (3,1),  6: (3,2),
    7: (4,0),  8: (4,1),  9: (4,2), 10: (4,3),
   11: (5,0), 12: (5,1), 13: (5,2), 14: (5,3), 15: (5,4),
}
L_CHAR = ['S','P','D','F','G','H']

def nl_label(n, l):
    return f"{n}{L_CHAR[l]}"

def stat_weight(l):
    """omega = (2S+1)(2L+1) = 2(2l+1) for hydrogen doublets."""
    return 2 * (2*l + 1)

def threshold_eV(n_lower, n_upper):
    """ΔE = E(upper) - E(lower) > 0, from hydrogen Bohr energies."""
    return IH_eV * (1.0/n_lower**2 - 1.0/n_upper**2)

# ── Anderson Table 2 (all rows, read from paper) ──────────────────────────────
# Format: (i_table_upper, j_table_lower): [Υ at 0.5,1,3,5,10,15,20,25 eV]
# Notation: a±b = a×10^(±b)
ANDERSON_TABLE2 = {
    # ── j=1 (1s) rows (upper → 1s) ──────────────────────────────────────────
    ( 2, 1): [2.60e-1, 2.96e-1, 3.25e-1, 3.37e-1, 3.56e-1, 3.68e-1, 3.75e-1, 3.80e-1],
    ( 3, 1): [4.27e-1, 5.36e-1, 8.57e-1, 1.15e+0, 1.75e+0, 2.13e+0, 2.35e+0, 2.46e+0],
    ( 4, 1): [6.45e-2, 6.89e-2, 7.72e-2, 8.06e-2, 8.33e-2, 8.41e-2, 8.45e-2, 8.46e-2],
    ( 5, 1): [1.11e-1, 1.26e-1, 1.86e-1, 2.40e-1, 3.13e-1, 3.29e-1, 3.20e-1, 3.04e-1],
    ( 6, 1): [6.17e-2, 6.56e-2, 7.81e-2, 8.98e-2, 1.10e-1, 1.22e-1, 1.29e-1, 1.35e-1],
    ( 7, 1): [2.13e-2, 2.51e-2, 3.18e-2, 3.38e-2, 3.48e-2, 3.48e-2, 3.47e-2, 3.46e-2],
    ( 8, 1): [3.81e-2, 4.70e-2, 7.39e-2, 9.41e-2, 1.23e-1, 1.33e-1, 1.32e-1, 1.28e-1],
    ( 9, 1): [2.87e-2, 3.12e-2, 4.02e-2, 4.72e-2, 5.74e-2, 6.28e-2, 6.60e-2, 6.82e-2],
    (10, 1): [1.18e-2, 1.11e-2, 1.05e-2, 1.06e-2, 1.11e-2, 1.16e-2, 1.18e-2, 1.20e-2],
    (11, 1): [1.40e-2, 1.69e-2, 1.91e-2, 1.92e-2, 1.88e-2, 1.85e-2, 1.83e-2, 1.82e-2],
    (12, 1): [2.60e-2, 3.12e-2, 4.04e-2, 4.74e-2, 5.71e-2, 5.82e-2, 5.58e-2, 5.23e-2],
    (13, 1): [2.01e-2, 2.18e-2, 2.47e-2, 2.75e-2, 3.15e-2, 3.35e-2, 3.46e-2, 3.53e-2],
    (14, 1): [8.79e-3, 8.87e-3, 9.42e-3, 9.95e-3, 1.09e-2, 1.14e-2, 1.17e-2, 1.19e-2],
    (15, 1): [4.36e-3, 3.86e-3, 2.79e-3, 2.35e-3, 1.88e-3, 1.65e-3, 1.51e-3, 1.42e-3],
    # ── j=2 (2s) rows ────────────────────────────────────────────────────────
    ( 4, 2): [1.39e+0, 1.47e+0, 2.29e+0, 3.06e+0, 4.26e+0, 4.89e+0, 5.28e+0, 5.54e+0],
    ( 5, 2): [2.42e+0, 3.07e+0, 5.31e+0, 7.79e+0, 1.38e+1, 1.91e+1, 2.36e+1, 2.76e+1],
    ( 6, 2): [2.05e+0, 3.10e+0, 6.59e+0, 9.30e+0, 1.35e+1, 1.57e+1, 1.71e+1, 1.80e+1],
    ( 7, 2): [3.75e-1, 3.80e-1, 5.05e-1, 6.29e-1, 8.23e-1, 9.27e-1, 9.91e-1, 1.03e+0],
    ( 8, 2): [7.28e-1, 8.89e-1, 1.45e+0, 1.95e+0, 3.06e+0, 3.98e+0, 4.76e+0, 5.43e+0],
    ( 9, 2): [6.30e-1, 7.45e-1, 1.12e+0, 1.40e+0, 1.83e+0, 2.05e+0, 2.19e+0, 2.28e+0],
    (10, 2): [5.43e-1, 7.33e-1, 1.29e+0, 1.63e+0, 2.09e+0, 2.31e+0, 2.44e+0, 2.52e+0],
    (11, 2): [1.90e-1, 2.07e-1, 2.31e-1, 2.59e-1, 3.13e-1, 3.45e-1, 3.65e-1, 3.78e-1],
    (12, 2): [4.01e-1, 5.10e-1, 7.13e-1, 8.75e-1, 1.28e+0, 1.64e+0, 1.97e+0, 2.25e+0],
    (13, 2): [4.16e-1, 4.76e-1, 5.52e-1, 6.02e-1, 6.81e-1, 7.23e-1, 7.50e-1, 7.67e-1],
    (14, 2): [3.75e-1, 4.55e-1, 6.75e-1, 8.16e-1, 9.92e-1, 1.07e+0, 1.12e+0, 1.15e+0],
    (15, 2): [2.28e-1, 2.49e-1, 2.50e-1, 2.47e-1, 2.46e-1, 2.45e-1, 2.45e-1, 2.45e-1],
    # ── j=3 (2p) rows ────────────────────────────────────────────────────────
    ( 4, 3): [2.00e+0, 2.18e+0, 2.26e+0, 2.33e+0, 2.64e+0, 2.96e+0, 3.26e+0, 3.54e+0],
    ( 5, 3): [7.18e+0, 7.90e+0, 1.07e+1, 1.32e+1, 1.71e+1, 1.92e+1, 2.05e+1, 2.13e+1],
    ( 6, 3): [1.31e+1, 1.81e+1, 3.73e+1, 5.64e+1, 9.81e+1, 1.32e+2, 1.59e+2, 1.83e+2],
    ( 7, 3): [6.92e-1, 7.43e-1, 7.47e-1, 7.33e-1, 7.35e-1, 7.61e-1, 7.93e-1, 8.26e-1],
    ( 8, 3): [2.06e+0, 2.20e+0, 2.67e+0, 3.05e+0, 3.63e+0, 3.94e+0, 4.13e+0, 4.25e+0],
    ( 9, 3): [3.40e+0, 4.30e+0, 7.65e+0, 1.07e+1, 1.69e+1, 2.16e+1, 2.53e+1, 2.84e+1],
    (10, 3): [2.62e+0, 3.18e+0, 4.93e+0, 6.10e+0, 7.71e+0, 8.52e+0, 9.01e+0, 9.32e+0],
    (11, 3): [4.48e-1, 5.00e-1, 4.71e-1, 4.34e-1, 3.87e-1, 3.64e-1, 3.47e-1, 3.35e-1],
    (12, 3): [1.26e+0, 1.37e+0, 1.36e+0, 1.38e+0, 1.46e+0, 1.52e+0, 1.56e+0, 1.59e+0],
    (13, 3): [1.99e+0, 2.46e+0, 3.45e+0, 4.31e+0, 6.16e+0, 7.59e+0, 8.75e+0, 9.72e+0],
    (14, 3): [1.65e+0, 1.99e+0, 2.90e+0, 3.48e+0, 4.23e+0, 4.59e+0, 4.80e+0, 4.93e+0],
    (15, 3): [8.13e-1, 8.89e-1, 9.10e-1, 8.90e-1, 8.46e-1, 8.18e-1, 7.99e-1, 7.86e-1],
    # ── j=4 (3s) rows ────────────────────────────────────────────────────────
    ( 7, 4): [2.57e+0, 4.38e+0, 1.09e+1, 1.52e+1, 2.12e+1, 2.42e+1, 2.59e+1, 2.71e+1],
    ( 8, 4): [4.32e+0, 6.09e+0, 1.54e+1, 2.58e+1, 4.89e+1, 6.72e+1, 8.23e+1, 9.50e+1],
    ( 9, 4): [6.21e+0, 1.02e+1, 2.30e+1, 3.22e+1, 4.55e+1, 5.24e+1, 5.65e+1, 5.93e+1],
    (10, 4): [6.63e+0, 1.06e+1, 1.97e+1, 2.48e+1, 3.14e+1, 3.47e+1, 3.67e+1, 3.79e+1],
    (11, 4): [9.87e-1, 1.57e+0, 2.91e+0, 3.55e+0, 4.31e+0, 4.66e+0, 4.87e+0, 5.00e+0],
    (12, 4): [2.50e+0, 3.43e+0, 5.57e+0, 7.34e+0, 1.11e+1, 1.41e+1, 1.66e+1, 1.87e+1],
    (13, 4): [3.22e+0, 4.17e+0, 6.12e+0, 7.07e+0, 8.29e+0, 8.90e+0, 9.26e+0, 9.49e+0],
    (14, 4): [2.80e+0, 3.33e+0, 4.10e+0, 4.45e+0, 4.81e+0, 4.95e+0, 5.02e+0, 5.07e+0],
    (15, 4): [3.63e+0, 4.77e+0, 6.72e+0, 7.87e+0, 9.38e+0, 1.01e+1, 1.05e+1, 1.08e+1],
    # ── j=5 (3p) rows ────────────────────────────────────────────────────────
    ( 7, 5): [3.90e+0, 4.83e+0, 7.63e+0, 9.72e+0, 1.36e+1, 1.66e+1, 1.90e+1, 2.11e+1],
    ( 8, 5): [1.37e+1, 2.08e+1, 4.51e+1, 6.16e+1, 8.42e+1, 9.54e+1, 1.02e+2, 1.06e+2],
    ( 9, 5): [2.06e+1, 3.21e+1, 8.33e+1, 1.35e+2, 2.42e+2, 3.25e+2, 3.91e+2, 4.46e+2],
    (10, 5): [2.47e+1, 4.13e+1, 8.83e+1, 1.19e+2, 1.62e+2, 1.83e+2, 1.96e+2, 2.05e+2],
    (11, 5): [2.14e+0, 2.52e+0, 2.99e+0, 3.23e+0, 3.72e+0, 4.15e+0, 4.52e+0, 4.84e+0],
    (12, 5): [7.11e+0, 9.60e+0, 1.39e+1, 1.58e+1, 1.82e+1, 1.93e+1, 2.00e+1, 2.05e+1],
    (13, 5): [1.10e+1, 1.52e+1, 2.67e+1, 3.55e+1, 5.18e+1, 6.36e+1, 7.29e+1, 8.07e+1],
    (14, 5): [9.95e+0, 1.23e+1, 1.64e+1, 1.82e+1, 2.01e+1, 2.08e+1, 2.12e+1, 2.15e+1],
    (15, 5): [1.42e+1, 1.84e+1, 2.47e+1, 2.87e+1, 3.43e+1, 3.71e+1, 3.88e+1, 3.99e+1],
    # ── j=6 (3d) rows ────────────────────────────────────────────────────────
    ( 7, 6): [3.97e+0, 4.13e+0, 4.30e+0, 4.54e+0, 4.94e+0, 5.16e+0, 5.30e+0, 5.38e+0],
    ( 8, 6): [1.31e+1, 1.52e+1, 1.96e+1, 2.17e+1, 2.43e+1, 2.57e+1, 2.66e+1, 2.73e+1],
    ( 9, 6): [3.31e+1, 4.58e+1, 8.31e+1, 1.07e+2, 1.40e+2, 1.56e+2, 1.65e+2, 1.71e+2],
    (10, 6): [6.38e+1, 1.15e+2, 3.30e+2, 5.22e+2, 8.89e+2, 1.15e+3, 1.36e+3, 1.53e+3],
    (11, 6): [2.64e+0, 2.83e+0, 2.50e+0, 2.25e+0, 1.96e+0, 1.84e+0, 1.77e+0, 1.73e+0],
    (12, 6): [7.81e+0, 8.63e+0, 8.67e+0, 8.46e+0, 8.31e+0, 8.38e+0, 8.50e+0, 8.64e+0],
    (13, 6): [1.61e+1, 2.06e+1, 2.72e+1, 2.93e+1, 3.16e+1, 3.26e+1, 3.32e+1, 3.37e+1],
    (14, 6): [2.75e+1, 4.00e+1, 7.83e+1, 1.06e+2, 1.50e+2, 1.78e+2, 1.99e+2, 2.15e+2],
    (15, 6): [2.88e+1, 3.84e+1, 5.33e+1, 6.19e+1, 7.48e+1, 8.16e+1, 8.58e+1, 8.86e+1],
    # ── j=7 (4s) rows ────────────────────────────────────────────────────────
    (11, 7): [6.62e+0, 1.61e+1, 4.35e+1, 5.72e+1, 7.20e+1, 7.83e+1, 8.18e+1, 8.40e+1],
    (12, 7): [9.63e+0, 1.77e+1, 4.28e+1, 6.27e+1, 1.01e+2, 1.31e+2, 1.56e+2, 1.78e+2],
    (13, 7): [1.39e+1, 2.56e+1, 6.18e+1, 8.38e+1, 1.11e+2, 1.24e+2, 1.31e+2, 1.35e+2],
    (14, 7): [2.16e+1, 3.47e+1, 6.31e+1, 7.64e+1, 9.08e+1, 9.69e+1, 1.00e+2, 1.02e+2],
    (15, 7): [1.96e+1, 2.74e+1, 4.12e+1, 4.75e+1, 5.47e+1, 5.79e+1, 5.98e+1, 6.10e+1],
    # ── j=8 (4p) rows ────────────────────────────────────────────────────────
    (11, 8): [1.07e+1, 1.80e+1, 2.95e+1, 3.40e+1, 4.10e+1, 4.65e+1, 5.13e+1, 5.56e+1],
    (12, 8): [3.83e+1, 7.60e+1, 1.76e+2, 2.25e+2, 2.79e+2, 3.02e+2, 3.15e+2, 3.23e+2],
    (13, 8): [4.00e+1, 7.17e+1, 1.88e+2, 2.80e+2, 4.48e+2, 5.71e+2, 6.70e+2, 7.53e+2],
    (14, 8): [6.74e+1, 1.15e+2, 2.42e+2, 3.14e+2, 4.02e+2, 4.42e+2, 4.65e+2, 4.80e+2],
    (15, 8): [6.71e+1, 9.86e+1, 1.63e+2, 1.95e+2, 2.33e+2, 2.50e+2, 2.59e+2, 2.66e+2],
    # ── j=9 (4d) rows ────────────────────────────────────────────────────────
    (11, 9): [1.21e+1, 1.57e+1, 1.89e+1, 2.01e+1, 2.17e+1, 2.24e+1, 2.29e+1, 2.32e+1],
    (12, 9): [4.04e+1, 5.98e+1, 8.39e+1, 8.90e+1, 9.21e+1, 9.35e+1, 9.48e+1, 9.63e+1],
    (13, 9): [8.75e+1, 1.60e+2, 3.36e+2, 4.17e+2, 5.02e+2, 5.36e+2, 5.54e+2, 5.65e+2],
    (14, 9): [1.05e+2, 2.00e+2, 5.43e+2, 8.15e+2, 1.30e+3, 1.65e+3, 1.92e+3, 2.15e+3],
    (15, 9): [1.65e+2, 2.61e+2, 4.81e+2, 6.09e+2, 7.73e+2, 8.50e+2, 8.95e+2, 9.24e+2],
    # ── j=10 (4f) rows ───────────────────────────────────────────────────────
    (11,10): [9.63e+0, 1.05e+1, 9.45e+0, 8.68e+0, 7.81e+0, 7.43e+0, 7.22e+0, 7.09e+0],
    (12,10): [3.33e+1, 3.96e+1, 3.94e+1, 3.69e+1, 3.35e+1, 3.19e+1, 3.10e+1, 3.04e+1],
    (13,10): [7.24e+1, 1.02e+2, 1.31e+2, 1.30e+2, 1.16e+2, 1.04e+2, 9.46e+1, 8.76e+1],
    (14,10): [1.62e+2, 2.70e+2, 5.29e+2, 6.52e+2, 7.76e+2, 8.24e+2, 8.50e+2, 8.65e+2],
    (15,10): [3.81e+2, 7.45e+2, 1.77e+3, 2.47e+3, 3.62e+3, 4.42e+3, 5.03e+3, 5.53e+3],
}

print(f"Anderson Table 2: {len(ANDERSON_TABLE2)} transitions loaded")

# ── Rate coefficient functions ─────────────────────────────────────────────────

def K_from_anderson(upsilon, n_lower, l_lower, n_upper, Te_eV):
    """
    Convert Anderson Upsilon to excitation rate coefficient [cm^3/s].
    Uses Anderson Eq.(3) with correct omega = (2S+1)(2L+1) of LOWER state.
    """
    omega_lower = stat_weight(l_lower)
    dE = threshold_eV(n_lower, n_upper)
    return C_AND / omega_lower * np.sqrt(IH_eV / Te_eV) * np.exp(-dE / Te_eV) * upsilon

def K_maxwell_SI(sig_a0sq, E_eV, Te_eV):
    """
    Maxwellian-averaged rate coefficient in SI, returned in cm^3/s.
    K = sqrt(8/pi/me) * (kTe)^(-3/2) * integral[ sigma(E) * E * exp(-E/kTe) dE ]
    """
    sig_m2 = sig_a0sq * a0_m**2
    E_J    = E_eV * eV_to_J
    kTe_J  = Te_eV * eV_to_J
    prefac = np.sqrt(8.0 / np.pi / me) * kTe_J**(-1.5)
    igrd   = sig_m2 * E_J * np.exp(-E_eV / Te_eV)
    K_SI   = prefac * np.trapezoid(igrd, E_J)
    return K_SI * 1e6   # m^3/s → cm^3/s

# ── Load CCC cross sections ────────────────────────────────────────────────────
CCC_PATH = 'data/processed/collisions/ccc/ccc_crosssections.csv'
if not os.path.exists(CCC_PATH):
    # fallback: check uploads
    CCC_PATH = '/mnt/user-data/uploads/ccc_crosssections.csv'

print(f"\nLoading CCC data from: {CCC_PATH}")
df_ccc = pd.read_csv(CCC_PATH)
grp_ccc = df_ccc.groupby(['n_i','l_i','n_f','l_f'])
ccc_keys = set(grp_ccc.groups.keys())
print(f"CCC transitions available: {len(ccc_keys)}")

# ── Spot-check 3 anchor transitions (Check 1) ─────────────────────────────────
print("\n" + "="*70)
print("CHECK 1 — ANCHOR SPOT CHECKS (verify formula, 3 transitions)")
print("="*70)
anchors = [
    # (i_table_upper, j_table_lower, label, Te, expected_K_range)
    (3, 1, '1s→2p', 1.0, (6e-13, 9e-13)),
    (2, 1, '1s→2s', 1.0, (3e-13, 5e-13)),
    (5, 2, '2s→3p', 1.0, (1e-8,  3e-8)),
]
for i_t, j_t, label, Te, (lo, hi) in anchors:
    n_up, l_up = IDX_TO_NL[i_t]
    n_lo, l_lo = IDX_TO_NL[j_t]
    upsilon = np.interp(Te, TE_AND, ANDERSON_TABLE2[(i_t, j_t)])
    K_and = K_from_anderson(upsilon, n_lo, l_lo, n_up, Te)
    in_range = "✓" if lo <= K_and <= hi else "✗"
    print(f"  {label:8s} Te={Te:.1f}eV  Υ={upsilon:.4f}  "
          f"K_and={K_and:.3e} cm³/s  [{lo:.0e},{hi:.0e}] {in_range}")

# ── Main benchmark loop ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CHECK 2/3/4 — FULL BENCHMARK (all Anderson transitions, Te=1–10 eV)")
print("="*70)

results = []
not_in_ccc = []

for (i_t, j_t), upsilons in ANDERSON_TABLE2.items():
    n_up, l_up = IDX_TO_NL[i_t]
    n_lo, l_lo = IDX_TO_NL[j_t]
    label = f"{nl_label(n_lo,l_lo)}→{nl_label(n_up,l_up)}"
    dE    = threshold_eV(n_lo, n_up)

    # CCC data: excitation stored as (n_lower, l_lower, n_upper, l_upper)
    ccc_key = (n_lo, l_lo, n_up, l_up)
    if ccc_key not in ccc_keys:
        not_in_ccc.append(label)
        continue

    ccc_sub  = grp_ccc.get_group(ccc_key).sort_values('E_eV')
    E_ccc    = ccc_sub.E_eV.values
    sig_ccc  = ccc_sub.sigma_a0sq.values

    # Loop over thesis Te range only
    for k in THESIS_TE_IDX:
        Te = TE_AND[k]
        upsilon = upsilons[k]

        # Anderson K
        K_and = K_from_anderson(upsilon, n_lo, l_lo, n_up, Te)

        # CCC Maxwell-averaged K
        E_grid   = np.linspace(dE + 1e-4, E_ccc.max(), 5000)
        sig_grid = np.interp(E_grid, E_ccc, sig_ccc, left=0.0, right=0.0)
        K_ccc    = K_maxwell_SI(sig_grid, E_grid, Te)

        ratio   = K_ccc / K_and
        pct_err = (ratio - 1.0) * 100.0

        results.append({
            'i_table': i_t, 'j_table': j_t,
            'label': label,
            'n_lower': n_lo, 'l_lower': l_lo,
            'n_upper': n_up, 'l_upper': l_up,
            'Te': Te,
            'K_CCC': K_ccc, 'K_Anderson': K_and,
            'Upsilon': upsilon,
            'dE_eV': dE,
            'ratio': ratio, 'pct_err': pct_err,
            # transition class
            'class': 'ground_exc' if n_lo == 1 else 'excited_exc',
        })

df = pd.DataFrame(results)
print(f"  Matched transitions  : {df['label'].nunique()} / {len(ANDERSON_TABLE2)}")
print(f"  Not in CCC           : {len(not_in_ccc)}")
if not_in_ccc:
    print(f"    {not_in_ccc}")
print(f"  Total (Te,transition) points : {len(df)}")

# ── Summary statistics ─────────────────────────────────────────────────────────
def summary_stats(sub, label):
    n  = len(sub)
    w10 = (sub.pct_err.abs() < 10).mean() * 100
    w15 = (sub.pct_err.abs() < 15).mean() * 100
    w20 = (sub.pct_err.abs() < 20).mean() * 100
    mean_e = sub.pct_err.abs().mean()
    med_e  = sub.pct_err.abs().median()
    max_e  = sub.pct_err.abs().max()
    worst  = sub.loc[sub.pct_err.abs().idxmax(), 'label']
    pass_flag = "PASS" if w20 > 85 and mean_e < 15 else "PARTIAL PASS" if w20 > 70 else "FAIL"
    print(f"\n  {label}  (n={n})")
    print(f"    Within 10%  : {w10:.1f}%")
    print(f"    Within 15%  : {w15:.1f}%")
    print(f"    Within 20%  : {w20:.1f}%")
    print(f"    Mean |err|  : {mean_e:.2f}%")
    print(f"    Median |err|: {med_e:.2f}%")
    print(f"    Max |err|   : {max_e:.2f}%  (worst: {worst})")
    print(f"    VERDICT     : {pass_flag}")
    return {'label': label, 'n': n, 'w10': w10, 'w15': w15, 'w20': w20,
            'mean_err': mean_e, 'max_err': max_e, 'verdict': pass_flag}

print("\n" + "="*70)
print("SUMMARY STATISTICS  (Te = 1, 3, 5, 10 eV only)")
print("="*70)

stats_all    = summary_stats(df,                              "ALL transitions")
stats_gnd    = summary_stats(df[df['class']=='ground_exc'],   "Ground-state excitation (1s→nl)")
stats_exc    = summary_stats(df[df['class']=='excited_exc'],  "Excited-state excitation (n≥2→n'l')")

# Per-Te breakdown
print("\n  Per-Te breakdown:")
for Te in THESIS_TE:
    sub = df[df.Te == Te]
    w20 = (sub.pct_err.abs() < 20).mean() * 100
    mean_e = sub.pct_err.abs().mean()
    print(f"    Te={Te:5.1f} eV  n={len(sub):3d}  within20%={w20:.1f}%  mean|err|={mean_e:.2f}%")

# ── Check 5: detailed balance consistency ─────────────────────────────────────
print("\n" + "="*70)
print("CHECK 5 — DETAILED BALANCE CONSISTENCY")
print("  Verify: K_deexc_CCC / K_deexc_Anderson agrees same as excitation")
print("  K_deexc_Anderson = (omega_lower/omega_upper) * exp(+dE/kTe) * K_exc_Anderson")
print("  K_deexc_CCC      = loaded from K_CCC_deexc_table.npy")
print("="*70)

# Load saved deexcitation table if available
deexc_path = 'data/processed/collisions/ccc/K_CCC_deexc_table.npy'
meta_path  = 'data/processed/collisions/ccc/K_CCC_metadata.csv'
Te_path    = 'data/processed/collisions/ccc/Te_grid.npy'

if os.path.exists(deexc_path) and os.path.exists(meta_path):
    K_deexc_table = np.load(deexc_path)
    Te_grid_saved = np.load(Te_path)
    meta = pd.read_csv(meta_path)
    print(f"  Loaded K_CCC_deexc_table: {K_deexc_table.shape}")

    db_results = []
    # Check first 5 transitions where both exist
    checked = 0
    for (i_t, j_t) in list(ANDERSON_TABLE2.keys())[:20]:
        n_up, l_up = IDX_TO_NL[i_t]
        n_lo, l_lo = IDX_TO_NL[j_t]
        label = f"{nl_label(n_lo,l_lo)}→{nl_label(n_up,l_up)}"

        row = meta[(meta.n_i==n_lo)&(meta.l_i==l_lo)&(meta.n_f==n_up)&(meta.l_f==l_up)]
        if len(row) == 0:
            continue

        idx = row.iloc[0]['idx']
        dE  = threshold_eV(n_lo, n_up)
        omega_lo = stat_weight(l_lo)
        omega_up = stat_weight(l_up)

        for k in THESIS_TE_IDX:
            Te = TE_AND[k]
            upsilon = ANDERSON_TABLE2[(i_t, j_t)][k]

            K_exc_and  = K_from_anderson(upsilon, n_lo, l_lo, n_up, Te)
            K_deexc_and = (omega_lo / omega_up) * np.exp(+dE / Te) * K_exc_and

            # Find nearest Te in saved grid
            ti = np.argmin(np.abs(Te_grid_saved - Te))
            K_deexc_ccc = K_deexc_table[idx, ti]

            ratio   = K_deexc_ccc / K_deexc_and
            pct_err = (ratio - 1.0) * 100.0
            db_results.append({'label': label, 'Te': Te,
                                'K_deexc_CCC': K_deexc_ccc, 'K_deexc_and': K_deexc_and,
                                'ratio': ratio, 'pct_err': pct_err})
        checked += 1
        if checked >= 5:
            break

    if db_results:
        db_df = pd.DataFrame(db_results)
        print(f"\n  Spot-check de-excitation (5 transitions, Te=1–10 eV):")
        print(f"  {'Transition':12s}  {'Te':5s}  {'K_deexc_CCC':>13s}  {'K_deexc_and':>13s}  {'ratio':>7s}  {'%err':>7s}")
        print("  " + "-"*68)
        for _, r in db_df.iterrows():
            flag = " ***" if abs(r.pct_err) > 15 else ("  * " if abs(r.pct_err) > 10 else "")
            print(f"  {r.label:12s}  {r.Te:5.1f}  {r.K_deexc_CCC:13.4e}  {r.K_deexc_and:13.4e}  "
                  f"{r.ratio:7.4f}  {r.pct_err:+7.2f}%{flag}")
        w20_db = (db_df.pct_err.abs() < 20).mean() * 100
        print(f"\n  DB check within 20%: {w20_db:.1f}%  — "
              + ("CONSISTENT" if w20_db > 80 else "INCONSISTENCY DETECTED"))
else:
    print("  [SKIP] K_CCC_deexc_table.npy not found at expected path.")
    print("  Run compute_K_CCC.py first, or this check will be skipped.")

# ── Save CSV ───────────────────────────────────────────────────────────────────
out_csv = 'data/processed/collisions/anderson_benchmark_full.csv'
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}  ({len(df)} rows)")

# ── Figures ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Anderson (2000) RMPS Benchmark — Check 5\n"
             "Te = 1–10 eV (thesis range)", fontsize=13, fontweight='bold')

# Panel 1: K_CCC vs K_Anderson scatter (log-log), coloured by Te
ax = axes[0, 0]
colors = {'1.0': '#2166ac', '3.0': '#4dac26', '5.0': '#d6604d', '10.0': '#762a83'}
for Te, sub in df.groupby('Te'):
    ax.scatter(sub.K_Anderson, sub.K_CCC, s=20, alpha=0.7,
               label=f'Te={Te:.0f} eV', color=colors[str(Te)], zorder=3)
lims = [df[['K_Anderson','K_CCC']].min().min()*0.5,
        df[['K_Anderson','K_CCC']].max().max()*2]
ax.plot(lims, lims, 'k-', lw=1.5, label='1:1')
ax.plot(lims, [x*1.2 for x in lims], 'k--', lw=0.8, alpha=0.6)
ax.plot(lims, [x*0.8 for x in lims], 'k--', lw=0.8, alpha=0.6, label='±20%')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('K_Anderson (cm³/s)'); ax.set_ylabel('K_CCC (cm³/s)')
ax.set_title('K_CCC vs K_Anderson\n(all transitions, Te=1–10 eV)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2: % error histogram, split by class
ax = axes[0, 1]
ax.hist(df[df['class']=='ground_exc'].pct_err,   bins=25, alpha=0.6,
        color='steelblue', label='1s→nl', edgecolor='white')
ax.hist(df[df['class']=='excited_exc'].pct_err,  bins=25, alpha=0.6,
        color='darkorange', label='n≥2→n\'l\'', edgecolor='white')
ax.axvline(0,  color='k',  lw=1.5)
ax.axvline(20, color='r',  ls='--', lw=1, label='±20%')
ax.axvline(-20, color='r', ls='--', lw=1)
ax.set_xlabel('% error (K_CCC / K_Anderson − 1) × 100')
ax.set_ylabel('Count')
ax.set_title(f'Error distribution by class\n'
             f'All: mean={df.pct_err.mean():+.1f}%, σ={df.pct_err.std():.1f}%')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 3: 1s→2p K vs Te
ax = axes[0, 2]
trans = df[(df.n_lower==1)&(df.l_lower==0)&(df.n_upper==2)&(df.l_upper==1)]
if len(trans):
    ax.semilogy(trans.Te, trans.K_CCC,     'bo-', ms=7, lw=2, label='CCC (this work)')
    ax.semilogy(trans.Te, trans.K_Anderson, 'rs--', ms=7, lw=2, label='Anderson RMPS')
    ax.fill_between(trans.Te, trans.K_Anderson*0.8, trans.K_Anderson*1.2,
                    alpha=0.12, color='r', label='Anderson ±20%')
ax.set_xlabel('Te (eV)'); ax.set_ylabel('K (cm³/s)')
ax.set_title('1s → 2p  (Lyman-α driver)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 4: 2p→3d K vs Te
ax = axes[1, 0]
trans = df[(df.n_lower==2)&(df.l_lower==1)&(df.n_upper==3)&(df.l_upper==2)]
if len(trans):
    ax.semilogy(trans.Te, trans.K_CCC,     'bo-', ms=7, lw=2, label='CCC (this work)')
    ax.semilogy(trans.Te, trans.K_Anderson, 'rs--', ms=7, lw=2, label='Anderson RMPS')
    ax.fill_between(trans.Te, trans.K_Anderson*0.8, trans.K_Anderson*1.2,
                    alpha=0.12, color='r', label='Anderson ±20%')
ax.set_xlabel('Te (eV)'); ax.set_ylabel('K (cm³/s)')
ax.set_title('2p → 3d  (key stepwise)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 5: per-transition mean % error bar chart (top 30 by |error|)
ax = axes[1, 1]
per_t = df.groupby('label').pct_err.mean().sort_values(key=abs, ascending=False).head(30)
bar_colors = ['red' if abs(v) > 20 else ('orange' if abs(v) > 10 else 'green')
              for v in per_t.values]
ax.barh(range(len(per_t)), per_t.values, color=bar_colors)
ax.axvline(0,   color='k', lw=1)
ax.axvline(20,  color='r', ls='--', lw=1)
ax.axvline(-20, color='r', ls='--', lw=1)
ax.set_yticks(range(len(per_t)))
ax.set_yticklabels(per_t.index, fontsize=7)
ax.set_xlabel('Mean % error (K_CCC vs Anderson)')
ax.set_title('Top 30 transitions by |error|\n(red>20%, orange>10%, green<10%)')
ax.grid(True, alpha=0.3, axis='x')

# Panel 6: cumulative error distribution
ax = axes[1, 2]
abs_errs = np.sort(df.pct_err.abs().values)
cdf = np.arange(1, len(abs_errs)+1) / len(abs_errs) * 100
ax.plot(abs_errs, cdf, 'b-', lw=2)
for pct_thresh, color, ls in [(10,'g','--'), (15,'orange','--'), (20,'r','--')]:
    frac = (df.pct_err.abs() < pct_thresh).mean() * 100
    ax.axvline(pct_thresh, color=color, ls=ls, lw=1.5,
               label=f'{pct_thresh}% → {frac:.1f}% of pts')
    ax.axhline(frac, color=color, ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('|% error|')
ax.set_ylabel('Cumulative % of comparisons')
ax.set_title('Cumulative error distribution\n(Te=1–10 eV, all transitions)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 60)

plt.tight_layout()
fig.savefig('figures/anderson_benchmark_full.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/anderson_benchmark_full.png")

print("\nDone.")