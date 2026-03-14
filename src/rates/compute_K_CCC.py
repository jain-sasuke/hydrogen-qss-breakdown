"""
compute_K_CCC.py
----------------
Maxwellian-average CCC electron-impact excitation cross sections → K_exc(Te).
Derive de-excitation rate coefficients K_deexc(Te) via detailed balance.

Location in repo
----------------
non_markovian_cr/src/rates/compute_K_CCC.py

Run from repo root
------------------
cd non_markovian_cr
python src/rates/compute_K_CCC.py

Physics: Excitation
-------------------
K_exc(Te) = sqrt(8/pi/me) * (kTe)^(-3/2)
            * integral[ sigma(E) * E * exp(-E/kTe) dE ]
lower limit : dE  (threshold from quantum numbers, not from data)
upper limit : E_max of CCC grid (~100-968 eV)

Physics: De-excitation via detailed balance
-------------------------------------------
K_deexc(Te) = K_exc(Te) * (omega_i / omega_f) * exp(+dE / kTe)

where:
  omega = (2S+1)(2L+1) = 2*(2*l+1)  for hydrogen doublets
  omega_i = statistical weight of LOWER state (initial for excitation)
  omega_f = statistical weight of UPPER state (final for excitation)
  dE > 0  = excitation energy [eV]

Note: K_deexc is NOT always > K_exc.
  The factor = (omega_i/omega_f) * exp(+dE/kTe) can be < 1 when
  omega_i << omega_f (upper state much more degenerate) AND dE small.
  Example: 1S->2P at Te=10 eV: factor = (2/6)*exp(10.2/10) = 0.925
  This is correct physics -- no energy barrier but less phase space.

Why detailed balance, not direct Maxwell averaging of sigma_deexc:
  1. Guarantees exact detailed balance in CR matrix -> populations
     reach Saha-Boltzmann at high ne (Gate C)
  2. Avoids near-threshold noise: sigma_deexc ~ 1/v as E->0
  3. CCC cross sections verified self-consistent to 0.05% (qc_ccc.py)

Units
-----
All internal computation in SI. Convert to cm^3/s only at final step.

Outputs (relative to repo root)
-------
data/processed/collisions/ccc/K_CCC_exc_table.npy    float64 (870, 12) [cm^3/s]
data/processed/collisions/ccc/K_CCC_deexc_table.npy  float64 (870, 12) [cm^3/s]
data/processed/collisions/ccc/K_CCC_metadata.csv     one row per transition pair
data/processed/collisions/ccc/Te_grid.npy            float64 (12,)     [eV]
figures/week2/K_CCC_diagnostic.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]   # src/rates/ -> src/ -> repo root

INPUT_CSV = REPO_ROOT / 'data/processed/collisions/ccc/ccc_crosssections.csv'
OUT_DIR   = REPO_ROOT / 'data/processed/collisions/ccc'
FIG_DIR   = REPO_ROOT / 'figures/week2'

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Physical constants (SI) ────────────────────────────────────────────────────
ME   = 9.10938e-31   # electron mass, kg
KB   = 1.60218e-19   # 1 eV in Joules
A0_M = 5.29177e-11   # Bohr radius, m
IH   = 13.6058       # hydrogen ionisation energy, eV

N_GRID = 5000        # interpolation grid points (validated: <2% error)

# ── Te grid (thesis parameter space, fixed for entire project) ─────────────────
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)   # eV

# ── Helper functions ───────────────────────────────────────────────────────────

def prefactor(Te_eV):
    """sqrt(8/pi/me) * (kTe)^(-3/2)  [SI]."""
    return np.sqrt(8.0 / np.pi / ME) * (Te_eV * KB)**(-1.5)

def threshold_eV(n_i, n_f):
    """Exact hydrogen threshold energy [eV]. Requires n_f > n_i."""
    return IH * (1.0 / n_i**2 - 1.0 / n_f**2)

def stat_weight(l):
    """Statistical weight for hydrogen doublet: omega = (2S+1)(2L+1) = 2(2l+1)."""
    return 2 * (2 * int(l) + 1)

def maxwell_average(E_raw, sig_raw, dE, Te_arr, n_grid=N_GRID):
    """
    Maxwellian-average a CCC excitation cross section over Te_arr.

    Parameters
    ----------
    E_raw   : 1D array   incident electron KE [eV], from CCC file
    sig_raw : 1D array   cross section [a0^2]
    dE      : float      threshold energy [eV]  (must be > 0)
    Te_arr  : 1D array   electron temperatures [eV]
    n_grid  : int        interpolation grid size

    Returns
    -------
    K_arr   : 1D array   K_exc [cm^3/s], same length as Te_arr
    """
    E_grid   = np.linspace(dE + 1e-4, E_raw.max(), n_grid)
    sig_grid = np.interp(E_grid, E_raw, sig_raw, left=0.0, right=0.0)

    K_arr = np.zeros(len(Te_arr))
    for k, Te in enumerate(Te_arr):
        integrand    = sig_grid * E_grid * np.exp(-E_grid / Te)   # a0^2 * eV
        integral_raw = np.trapezoid(integrand, E_grid)             # a0^2 * eV^2
        integral_SI  = integral_raw * A0_M**2 * KB**2             # m^2 * J^2
        K_arr[k]     = prefactor(Te) * integral_SI * 1e6          # cm^3/s

    return K_arr

def detailed_balance(K_exc_arr, l_i, l_f, dE_eV, Te_arr):
    """
    Derive K_deexc from K_exc via detailed balance.

        K_deexc = K_exc * (omega_i / omega_f) * exp(+dE / kTe)

    Parameters
    ----------
    K_exc_arr : 1D array   K_exc [cm^3/s] at each Te
    l_i       : int        angular momentum quantum number of LOWER state
    l_f       : int        angular momentum quantum number of UPPER state
    dE_eV     : float      excitation energy [eV]  (positive)
    Te_arr    : 1D array   electron temperatures [eV]

    Returns
    -------
    K_deexc   : 1D array   K_deexc [cm^3/s], same length as Te_arr
    """
    omega_i = stat_weight(l_i)    # lower state statistical weight
    omega_f = stat_weight(l_f)    # upper state statistical weight
    return K_exc_arr * (omega_i / omega_f) * np.exp(dE_eV / Te_arr)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading: {INPUT_CSV}")
df  = pd.read_csv(INPUT_CSV)
exc = df[df.n_f > df.n_i].copy()              # excitation only (n_f > n_i)
grp = exc.groupby(['n_i','l_i','n_f','l_f'], sort=True)
transitions = list(grp.groups.keys())
N_TRANS     = len(transitions)

print(f"Excitation transitions : {N_TRANS}")
print(f"Te grid (eV)           : {np.round(TE_GRID, 3)}")

# ── Main loop ──────────────────────────────────────────────────────────────────
K_exc_table   = np.zeros((N_TRANS, len(TE_GRID)))
K_deexc_table = np.zeros((N_TRANS, len(TE_GRID)))
metadata      = []

t0 = time.time()
for idx, (n_i, l_i, n_f, l_f) in enumerate(transitions):
    g       = grp.get_group((n_i, l_i, n_f, l_f)).sort_values('E_eV')
    E_raw   = g.E_eV.values
    sig_raw = g.sigma_a0sq.values
    dE      = threshold_eV(n_i, n_f)

    K_exc                = maxwell_average(E_raw, sig_raw, dE, TE_GRID)
    K_deexc              = detailed_balance(K_exc, l_i, l_f, dE, TE_GRID)
    K_exc_table[idx, :]  = K_exc
    K_deexc_table[idx,:] = K_deexc

    metadata.append({
        'idx'         : idx,
        'n_i'         : int(n_i),
        'l_i'         : int(l_i),
        'l_i_char'    : 'SPDFGHIJK'[int(l_i)],
        'n_f'         : int(n_f),
        'l_f'         : int(l_f),
        'l_f_char'    : 'SPDFGHIJK'[int(l_f)],
        'omega_i'     : stat_weight(l_i),
        'omega_f'     : stat_weight(l_f),
        'dE_eV'       : dE,
        'E_max_eV'    : round(E_raw.max(), 2),
        'n_raw_points': len(E_raw),
    })

    if (idx + 1) % 100 == 0 or idx == N_TRANS - 1:
        print(f"  {idx+1:4d}/{N_TRANS}  elapsed: {time.time()-t0:.1f}s")

meta_df = pd.DataFrame(metadata)
print(f"Loop complete in {time.time()-t0:.1f}s")

# ── Save ───────────────────────────────────────────────────────────────────────
np.save(OUT_DIR / 'K_CCC_exc_table.npy',   K_exc_table)
np.save(OUT_DIR / 'K_CCC_deexc_table.npy', K_deexc_table)
np.save(OUT_DIR / 'Te_grid.npy',           TE_GRID)
meta_df.to_csv(OUT_DIR / 'K_CCC_metadata.csv', index=False)

print(f"\nSaved K_CCC_exc_table.npy    shape={K_exc_table.shape}")
print(f"Saved K_CCC_deexc_table.npy  shape={K_deexc_table.shape}")
print(f"Saved K_CCC_metadata.csv     rows={len(meta_df)}")
print(f"Saved Te_grid.npy")

# ── Sanity checks ──────────────────────────────────────────────────────────────
print("\n── Sanity checks ──────────────────────────────────────────────────────")

# A: all K positive
n_neg_exc   = (K_exc_table   <= 0).sum()
n_neg_deexc = (K_deexc_table <= 0).sum()
print(f"Check A (all K > 0)")
print(f"  exc   : {'PASS' if n_neg_exc==0   else 'FAIL'}  ({n_neg_exc} non-positive)")
print(f"  deexc : {'PASS' if n_neg_deexc==0 else 'FAIL'}  ({n_neg_deexc} non-positive)")

# B: detailed balance ratio at machine precision
#    K_exc / K_deexc = (omega_f / omega_i) * exp(-dE/kTe)  exactly by construction
#    max error should be < 1e-10 % (floating point only)
print(f"Check B (detailed balance ratio -- 5 transitions, should be machine precision):")
check_pairs = [
    (1,0,2,1,'1S→2P'), (1,0,3,1,'1S→3P'), (2,0,3,1,'2S→3P'),
    (2,1,3,2,'2P→3D'), (3,1,4,2,'3P→4D'),
]
all_b_pass = True
for ni, li, nf, lf, label in check_pairs:
    row     = meta_df[(meta_df.n_i==ni)&(meta_df.l_i==li)&
                      (meta_df.n_f==nf)&(meta_df.l_f==lf)]
    idx_t   = row.idx.values[0]
    dE_t    = row.dE_eV.values[0]
    omega_i = row.omega_i.values[0]
    omega_f = row.omega_f.values[0]
    ratio_computed = K_exc_table[idx_t,:] / K_deexc_table[idx_t,:]
    ratio_expected = (omega_f / omega_i) * np.exp(-dE_t / TE_GRID)
    max_err = np.max(np.abs(ratio_computed / ratio_expected - 1)) * 100
    ok = max_err < 1e-10
    if not ok: all_b_pass = False
    print(f"  {label:>8}  max err = {max_err:.2e}%  {'PASS' if ok else 'FAIL'}")
print(f"  Overall : {'PASS' if all_b_pass else 'FAIL'}")

# C: magnitude and ratio check for 1S->2P
te_idx_3  = np.argmin(np.abs(TE_GRID - 3.0))
idx_1S2P  = meta_df[(meta_df.n_i==1)&(meta_df.l_i==0)&
                    (meta_df.n_f==2)&(meta_df.l_f==1)].idx.values[0]
dE_1S2P   = meta_df.loc[meta_df.idx==idx_1S2P, 'dE_eV'].values[0]
K_exc_3   = K_exc_table[idx_1S2P, te_idx_3]
K_deexc_3 = K_deexc_table[idx_1S2P, te_idx_3]
factor    = (2/6) * np.exp(dE_1S2P / TE_GRID[te_idx_3])
ok_c      = 1e-11 < K_exc_3 < 1e-8
print(f"\nCheck C (1S→2P at Te≈{TE_GRID[te_idx_3]:.2f} eV):")
print(f"  K_exc   = {K_exc_3:.4e} cm3/s  {'PASS' if ok_c else 'FAIL'}  (expect ~5e-10)")
print(f"  K_deexc = {K_deexc_3:.4e} cm3/s")
print(f"  factor  = (omega_i/omega_f)*exp(+dE/kTe) = (2/6)*exp({dE_1S2P:.2f}/{TE_GRID[te_idx_3]:.2f}) = {factor:.4f}")
print(f"  K_deexc/K_exc = {K_deexc_3/K_exc_3:.4f}  (matches factor: {'YES' if abs(K_deexc_3/K_exc_3 - factor) < 1e-6 else 'NO'})")

# D: no NaN or Inf
n_bad = (~np.isfinite(K_deexc_table)).sum() + (~np.isfinite(K_exc_table)).sum()
print(f"\nCheck D (no NaN/Inf) : {'PASS' if n_bad==0 else 'FAIL'}")

# ── Diagnostic figure ──────────────────────────────────────────────────────────
print("\nGenerating diagnostic figure...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

plot_pairs = [
    (1,0,2,1,'1S→2P'), (1,0,3,1,'1S→3P'), (2,0,3,1,'2S→3P'),
    (2,1,3,2,'2P→3D'), (3,1,4,2,'3P→4D'),
]

for i, (ni, li, nf, lf, label) in enumerate(plot_pairs):
    rows  = meta_df[(meta_df.n_i==ni)&(meta_df.l_i==li)&
                    (meta_df.n_f==nf)&(meta_df.l_f==lf)]
    ax    = axes[i]
    idx_t = rows.idx.values[0]
    dE_t  = rows.dE_eV.values[0]

    ax.semilogy(TE_GRID, K_exc_table[idx_t,:],   'bo-',  ms=5, lw=1.8, label='K_exc (Maxwell)')
    ax.semilogy(TE_GRID, K_deexc_table[idx_t,:], 'rs--', ms=5, lw=1.8, label='K_deexc (det.bal.)')
    ax.set_xlabel('Te (eV)')
    ax.set_ylabel('K (cm³/s)')
    ax.set_title(f'{label}   ΔE={dE_t:.2f} eV')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Panel 6: enhancement factor K_deexc/K_exc at Te~3 eV vs dE
ax = axes[5]
factor_at_3 = K_deexc_table[:, te_idx_3] / K_exc_table[:, te_idx_3]
dE_all      = meta_df.dE_eV.values
sc = ax.scatter(dE_all, factor_at_3, s=4, alpha=0.5,
                c=np.log10(factor_at_3), cmap='RdBu_r')
ax.axhline(1.0, color='k', ls='--', lw=1, label='factor=1')
ax.set_yscale('log')
ax.set_xlabel('ΔE (eV)')
ax.set_ylabel(f'K_deexc / K_exc  at Te≈{TE_GRID[te_idx_3]:.1f} eV')
ax.set_title('De-excitation enhancement factor\n'
             '(omega_i/omega_f)·exp(ΔE/kTe)\nBlue<1, Red>1')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='log₁₀(factor)')

plt.suptitle('K_CCC: Excitation (Maxwell avg) + De-excitation (Detailed Balance)\n'
             '870 transition pairs, Te = 1–10 eV',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'K_CCC_diagnostic.png', dpi=150, bbox_inches='tight')
print(f"Figure saved: {FIG_DIR / 'K_CCC_diagnostic.png'}")

# ── Summary table ──────────────────────────────────────────────────────────────
te_cols = [0, 3, 6, 9, 11]
print(f"\n── K(Te) summary (cm^3/s) ─────────────────────────────────────────────")
print(f"{'Transition':>10}  {'Type':>6}  " +
      "  ".join(f"Te={TE_GRID[j]:.1f}" for j in te_cols))
for ni, li, nf, lf, label in plot_pairs:
    rows  = meta_df[(meta_df.n_i==ni)&(meta_df.l_i==li)&
                    (meta_df.n_f==nf)&(meta_df.l_f==lf)]
    idx_t = rows.idx.values[0]
    for tag, table in [('exc', K_exc_table), ('deexc', K_deexc_table)]:
        vals = table[idx_t, te_cols]
        print(f"{label:>10}  {tag:>6}  " + "  ".join(f"{v:.3e}" for v in vals))
    print()

print("Done.")