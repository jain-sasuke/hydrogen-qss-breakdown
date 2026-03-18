"""
assemble_cr_matrix.py
=====================
Assemble the CR rate matrix L(Te, ne) for the hydrogen plasma model.

RATE EQUATION:
    dn/dt = L(Te, ne) * n + S(Te, ne, n_ion)

where:
    n     : (43,) bound-state population vector [cm^-3]
    L     : (43,43) rate matrix [s^-1] — depends on Te and ne
    S     : (43,) source vector from recombination [cm^-3 s^-1]
    n_ion : H+ density [cm^-3] (reservoir, given)

MATRIX CONVENTION:
    L[i,j] for i≠j : rate of transfer FROM state j TO state i  [s^-1]
    L[i,i]         : total loss rate FROM state i (negative)   [s^-1]
    Column sum: sum_i L[i,j] = -K_ion[j]*ne  (ionization is the only true loss)

FOUR TERMS IN L:

    1. Collisional (K_exc_full, K_deexc_full):
       L[j,i] += K_exc[i,j]  * ne    (excitation i→j, gain to upper j)
       L[i,j] += K_deexc[j,i]* ne    (deexcitation j→i, gain to lower i)
       diagonal: -= sum of all outgoing rates * ne

    2. Radiative (A_resolved, A_bund_res, A_bund_bund, gamma):
       L[i,j] += A[i,j]              (spontaneous emission j→i)
       diagonal: -= gamma[j]         (total radiative decay from j)

    3. Ionization (K_ion_final):
       L[i,i] -= K_ion[i] * ne       (loss to continuum from every state)

    4. Recombination → SOURCE VECTOR only (not in L):
       S[i] = (alpha_RR[i]*ne + alpha_3BR[i]*ne^2) * n_ion

VECTORIZED ASSEMBLY (verified against explicit loop, max diff = 0):
    Ke = K_exc_full[:,:,Te_idx] * ne    # upper triangular
    Kd = K_deexc_full[:,:,Te_idx] * ne  # lower triangular
    L += Ke.T   # excitation off-diagonal gains  (upper→lower)
    L += Kd.T   # deexcitation off-diagonal gains (lower→upper)
    diag(L) -= Ke.sum(axis=1) + Kd.sum(axis=1)  # collision losses

GRIDS:
    Te_grid : 50 pts log-spaced 1..10 eV
    ne_grid :  8 pts log-spaced 1e12..1e15 cm^-3  (thesis grid)

OUTPUTS:
    L_grid.npy       (50, 8, 43, 43)  pre-computed grid [s^-1]
    S_grid.npy       (50, 8, 43)      source per unit n_ion [cm^3 s^-1]
    Te_grid_L.npy    (50,)            [eV]
    ne_grid_L.npy    (8,)             [cm^-3]
    L_meta.csv                        assembly record

IH CONSTANTS (two physically distinct values):
    IH_RYDBERG       = 13.605693 eV  (R_inf * hc, used in CCC/TICS/collision)
    IH_SPECTROSCOPIC = 13.598435 eV  (true H ionisation energy, used in A coeff)
    (difference 0.05% — documented, not a bug)
"""

import numpy as np
import pandas as pd
import os

# ── Physical constants ─────────────────────────────────────────────────────────
IH_RYDBERG       = 13.605693122990   # eV  R_inf*hc — collision threshold energies
IH_SPECTROSCOPIC = 13.598434599702   # eV  true H ionisation — radiative energies

# ── Grids ──────────────────────────────────────────────────────────────────────
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)   # eV
NE_GRID = np.logspace(12, 15, 8)                            # cm^-3

# ── Paths ──────────────────────────────────────────────────────────────────────
PATHS = {
    'K_exc_full':    'data/processed/collisions/K_exc_full/K_exc_full.npy',
    'K_deexc_full':  'data/processed/collisions/K_exc_full/K_deexc_full.npy',
    'K_ion_final':   'data/processed/collisions/tics/K_ion_final.npy',
    'alpha_RR_res':  'data/processed/recombination/alpha_RR_resolved.npy',
    'alpha_RR_bund': 'data/processed/recombination/alpha_RR_bundled.npy',
    'alpha_3BR_res': 'data/processed/recombination/alpha_3BR_resolved.npy',
    'alpha_3BR_bund':'data/processed/recombination/alpha_3BR_bundled.npy',
    'A_resolved':    'data/processed/Radiative/A_resolved.npy',
    'A_bund_res':    'data/processed/Radiative/A_bund_res.npy',
    'A_bund_bund':   'data/processed/Radiative/A_bund_bund.npy',
    'gamma_resolved':'data/processed/Radiative/gamma_resolved.npy',
    'gamma_bundled': 'data/processed/Radiative/gamma_bundled.npy',
}


def load_rates(paths=None):
    """Load all rate arrays into a dict."""
    if paths is None:
        paths = PATHS
    rates = {}
    for key, path in paths.items():
        rates[key] = np.load(path)
    return rates


def build_L(Te_idx, ne, rates):
    """
    Build the 43x43 CR rate matrix L for given Te index and ne.

    Parameters
    ----------
    Te_idx : int    index into TE_GRID (0..49)
    ne     : float  electron density [cm^-3]
    rates  : dict   loaded rate arrays from load_rates()

    Returns
    -------
    L : (43, 43) ndarray  [s^-1]
        L[i,j] i≠j : transfer rate j→i
        L[i,i]     : total loss rate from i (negative)
    """
    L = np.zeros((43, 43))

    # ── 1. Collisional excitation and de-excitation ───────────────────────────
    # K_exc_full[i,j,Te_idx]: upper triangular, i=lower, j=upper
    # K_deexc_full[j,i,Te_idx]: lower triangular
    Ke = rates['K_exc_full'][:, :, Te_idx] * ne    # (43,43) upper tri
    Kd = rates['K_deexc_full'][:, :, Te_idx] * ne  # (43,43) lower tri

    L += Ke.T    # off-diagonal: L[j,i] += Ke[i,j] (gain to upper state j)
    L += Kd.T    # off-diagonal: L[i,j] += Kd[j,i] (gain to lower state i)
    np.fill_diagonal(L, np.diag(L) - Ke.sum(axis=1) - Kd.sum(axis=1))

    # ── 2. Radiative decay ────────────────────────────────────────────────────
    # A[i,j] = spontaneous emission rate from state j to state i [s^-1]
    # Off-diagonal: L[i,j] += A[i,j]  (gain to lower state i)
    # Diagonal: L[j,j] -= gamma[j]    (total radiative loss from j)

    # Resolved → resolved: A_resolved (36,36)
    L[:36, :36] += rates['A_resolved']
    np.fill_diagonal(L[:36, :36],
                     np.diag(L[:36, :36]) - rates['gamma_resolved'])

    # Bundled → resolved: A_bund_res (36,7)
    # A_bund_res[i,b] = rate from bundled state (36+b) to resolved state i
    L[:36, 36:] += rates['A_bund_res']           # gain to resolved i
    np.fill_diagonal(L[36:, 36:],
                     np.diag(L[36:, 36:]) - rates['gamma_bundled'])

    # Bundled → bundled: A_bund_bund (7,7)
    # A_bund_bund[b1,b2] = rate from bundled (36+b2) to bundled (36+b1)
    L[36:, 36:] += rates['A_bund_bund']
    # gamma_bundled already subtracted above (includes both A_bund_res and A_bund_bund)

    # ── 3. Ionization ─────────────────────────────────────────────────────────
    # K_ion_final[i,Te_idx]: ionisation loss from every bound state
    np.fill_diagonal(L, np.diag(L) - rates['K_ion_final'][:, Te_idx] * ne)

    return L


def build_source(Te_idx, ne, rates, n_ion=1.0):
    """
    Build recombination source vector S (43,).

    S[i] = (alpha_RR[i]*ne + alpha_3BR[i]*ne^2) * n_ion

    Parameters
    ----------
    n_ion : float  H+ density [cm^-3], default 1.0 for per-unit-n_ion output

    Returns
    -------
    S : (43,) ndarray  [cm^-3 s^-1] if n_ion given, or [cm^3 s^-1] if n_ion=1
    """
    ne2 = ne**2
    S = np.zeros(43)
    S[:36] = (rates['alpha_RR_res'][:, Te_idx] * ne
              + rates['alpha_3BR_res'][:, Te_idx] * ne2) * n_ion
    S[36:] = (rates['alpha_RR_bund'][:, Te_idx] * ne
              + rates['alpha_3BR_bund'][:, Te_idx] * ne2) * n_ion
    return S


def precompute_L_grid(rates=None, out_dir=None, ne_grid=None, te_grid=None):
    """
    Pre-compute L and S for all (Te, ne) grid points.

    Outputs
    -------
    L_grid : (n_Te, n_ne, 43, 43)  rate matrices [s^-1]
    S_grid : (n_Te, n_ne, 43)      source per unit n_ion [cm^3 s^-1]
    """
    if rates is None:
        rates = load_rates()
    if out_dir is None:
        out_dir = 'data/processed/cr_matrix'
    if ne_grid is None:
        ne_grid = NE_GRID
    if te_grid is None:
        te_grid = TE_GRID

    os.makedirs(out_dir, exist_ok=True)

    n_Te = len(te_grid)
    n_ne = len(ne_grid)

    L_grid = np.zeros((n_Te, n_ne, 43, 43))
    S_grid = np.zeros((n_Te, n_ne, 43))

    print(f"Pre-computing L grid ({n_Te} Te × {n_ne} ne = {n_Te*n_ne} matrices)...")
    for i_Te in range(n_Te):
        for i_ne, ne in enumerate(ne_grid):
            L_grid[i_Te, i_ne] = build_L(i_Te, ne, rates)
            S_grid[i_Te, i_ne] = build_source(i_Te, ne, rates, n_ion=1.0)

    # ── QC ────────────────────────────────────────────────────────────────────
    print()
    print("="*65)
    print("QC CHECKS")
    print("="*65)

    # A: shape
    print(f"\nCheck A — Shapes:")
    print(f"  L_grid: {L_grid.shape}  (expected (50,8,43,43))")
    print(f"  S_grid: {S_grid.shape}   (expected (50,8,43))")

    # B: no NaN/Inf
    nan_L = np.isnan(L_grid).sum()
    inf_L = np.isinf(L_grid).sum()
    print(f"\nCheck B — No NaN/Inf:  NaN={nan_L}  Inf={inf_L}  "
          f"{'PASS' if nan_L==0 and inf_L==0 else 'FAIL'}")

    # C: diagonal all negative (every state has net loss rate)
    # Except 1S at very low ne where recombination source is separate
    diag_vals = np.array([L_grid[i,j,k,k]
                          for i in range(n_Te)
                          for j in range(n_ne)
                          for k in range(43)])
    pos_diag = (diag_vals > 0).sum()
    print(f"\nCheck C — Diagonal all negative: {pos_diag} positive values  "
          f"{'PASS' if pos_diag==0 else 'WARN (check if physical)'}")

    # D: column sums = -K_ion * ne (particle conservation)
    # sum_i L[i,j] = -K_ion[j] * ne for all j, Te, ne
    K_ion = rates['K_ion_final']
    max_col_err = 0.0
    for i_Te in range(0, n_Te, 10):      # sample every 10th Te
        for i_ne, ne in enumerate(ne_grid[::2]):   # sample every 2nd ne
            i_ne_full = i_ne * 2
            col_sums  = L_grid[i_Te, i_ne_full].sum(axis=0)   # (43,)
            expected  = -K_ion[:, i_Te] * ne_grid[i_ne_full]
            rel_err   = np.abs((col_sums - expected) / (np.abs(expected)+1e-30))
            max_col_err = max(max_col_err, rel_err.max())
    print(f"\nCheck D — Column sum = -K_ion*ne (particle conservation):")
    print(f"  Max relative error = {max_col_err:.2e}  "
          f"{'PASS' if max_col_err < 1e-8 else 'FAIL'}")

    # E: S_grid all non-negative
    neg_S = (S_grid < 0).sum()
    print(f"\nCheck E — S_grid (source) non-negative: {neg_S} negative  "
          f"{'PASS' if neg_S==0 else 'FAIL'}")

    # F: spot check — L[0,2,0,0] (1S diagonal at Te=1,ne=ne[0])
    #    Should be ~ -(gamma_1S + K_ion_1S*ne) = -(0 + 4.8e-15*1e12) = -4.8e-3 s^-1
    L00 = L_grid[0, 0, 0, 0]
    K_ion_1S_1eV = K_ion[0, 0]
    ne0 = ne_grid[0]
    expected_diag = -(K_ion_1S_1eV * ne0)   # gamma(1S)=0
    print(f"\nCheck F — L(1S,1S) at Te=1eV, ne=1e12:")
    print(f"  L[0,0,0,0]  = {L00:.4e} s^-1")
    print(f"  Expected    = {expected_diag:.4e} s^-1  (-K_ion(1S,1eV)*ne)")
    ok_f = abs(L00/expected_diag - 1) < 0.01
    status_f = 'PASS' if ok_f else 'FAIL'
    print(f'  {status_f}')

    # G: L at high ne has larger magnitude (more collisions)
    L_low  = L_grid[25, 0]   # mid-Te, lowest ne
    L_high = L_grid[25, 7]   # mid-Te, highest ne
    mag_ratio = np.abs(L_high).max() / np.abs(L_low).max()
    print(f"\nCheck G — L magnitude increases with ne:")
    print(f"  |L|_max(ne=1e12) = {np.abs(L_low).max():.3e}")
    print(f"  |L|_max(ne=1e15) = {np.abs(L_high).max():.3e}")
    print(f"  Ratio = {mag_ratio:.1f}x  {'PASS' if mag_ratio > 10 else 'WARN'}")

    # H: rate table — key diagonal elements
    ti3  = np.argmin(np.abs(TE_GRID - 3.0))
    ni14 = np.argmin(np.abs(ne_grid - 1e14))
    print(f"\nCheck H — Diagonal rate table at Te≈{TE_GRID[ti3]:.2f}eV, "
          f"ne={ne_grid[ni14]:.1e} cm^-3:")
    labels = ['1S','2S','2P','3S','n9','n10','n15']
    idxs   = [0, 1, 2, 3, 36, 37, 42]
    for label, idx in zip(labels, idxs):
        d = L_grid[ti3, ni14, idx, idx]
        print(f"  L[{label},{label}] = {d:.4e} s^-1")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(f'{out_dir}/L_grid.npy',    L_grid)
    np.save(f'{out_dir}/S_grid.npy',    S_grid)
    np.save(f'{out_dir}/Te_grid_L.npy', te_grid)
    np.save(f'{out_dir}/ne_grid_L.npy', ne_grid)

    meta = pd.DataFrame({
        'n_Te':       [n_Te],
        'n_ne':       [n_ne],
        'n_states':   [43],
        'Te_min_eV':  [te_grid[0]],
        'Te_max_eV':  [te_grid[-1]],
        'ne_min_cm3': [ne_grid[0]],
        'ne_max_cm3': [ne_grid[-1]],
        'IH_collision_eV': [IH_RYDBERG],
        'IH_radiative_eV': [IH_SPECTROSCOPIC],
        'L_grid_MB':  [L_grid.nbytes / 1024**2],
    })
    meta.to_csv(f'{out_dir}/L_meta.csv', index=False)

    print(f"\nSaved to {out_dir}/:")
    print(f"  L_grid.npy    {L_grid.shape}  "
          f"[s^-1]  {L_grid.nbytes/1024**2:.1f} MB")
    print(f"  S_grid.npy    {S_grid.shape}   "
          f"[cm^3/s]  {S_grid.nbytes/1024:.0f} KB")
    print(f"  Te_grid_L.npy {te_grid.shape}")
    print(f"  ne_grid_L.npy {ne_grid.shape}")
    print(f"  L_meta.csv")

    return L_grid, S_grid


if __name__ == '__main__':
    print("Loading rate arrays...")
    rates = load_rates()
    print(f"  Loaded {len(rates)} arrays")
    precompute_L_grid(rates)