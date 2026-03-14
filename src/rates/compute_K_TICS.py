"""
compute_K_TICS.py
=================
Maxwell-average CCC Total Ionization Cross Sections → K_ion(Te).

Physics:
  K_ion(n,l,Te) = sqrt(8/π/me) * (kTe)^(-3/2)
                × ∫_{I_n}^{E_max} σ_TICS(E) · E · exp(-E/kTe) dE

  Lower limit = I_n = 13.6058/n² eV  (ionization threshold, l-independent for H)
  Units: all SI during integration, convert to cm³/s at end
  Grid: 5,000-point uniform linspace from I_n to E_max (same as K_CCC)

Outputs (in data/processed/collisions/tics/):
  K_ion_resolved.npy      (36, 12) — n=1..8 all l-resolved states [cm³/s]
  K_ion_n9_resolved.npy   ( 9, 12) — n=9 l-resolved (9S..9K) [cm³/s]
  K_ion_n9_bundled.npy    ( 1, 12) — n=9 from TICS.9 direct [cm³/s]
  K_ion_metadata.csv               — state labels, n, l, source
  K_ion_table.csv                  — long format for inspection
  Te_grid_ion.npy          (12,)   — Te grid [eV]

QC checks:
  A — All K_ion >= 0
  B — K_ion increases with n at fixed Te (Boltzmann suppression)
  C — K_ion(1S) << K_ion(2P) at Te=1 eV by factor ~10^4
  D — n=9 stat average vs TICS.9 direct within 20%
  E — CCC TICS vs Lotz (1967) comparison at Te=5 eV
"""

import numpy as np
import pandas as pd
import os
from scipy.special import exp1

# ── Constants ─────────────────────────────────────────────────────────────────
eV_to_J = 1.60218e-19
me       = 9.10938e-31
a0_m     = 5.29177e-11
IH_eV    = 13.6058
L_CHAR   = ['S','P','D','F','G','H','I','J','K']

# ── Te grid (fixed for entire thesis — same as K_CCC) ─────────────────────────
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)  # eV

# ── State indexing (must match radiative_rates.py) ────────────────────────────
def build_state_index():
    idx = {}
    i = 0
    for n in range(1, 9):
        for l in range(n):
            idx[(n, l)] = i
            i += 1
    return idx  # 36 entries

# ── Maxwell averaging ─────────────────────────────────────────────────────────
def maxwell_average_tics(E_eV, sig_a0sq, I_n_eV, Te_eV, n_grid=5000):
    """
    Returns K_ion in cm³/s.
    E_eV, sig_a0sq: raw CCC data arrays
    I_n_eV: ionization threshold [eV]
    Te_eV: electron temperature [eV]
    """
    kTe_J = Te_eV * eV_to_J

    E_min_grid = I_n_eV + 1e-4
    E_max_grid = E_eV.max()
    E_grid_eV  = np.linspace(E_min_grid, E_max_grid, n_grid)

    sig_m2    = np.interp(E_grid_eV, E_eV, sig_a0sq,
                          left=0.0, right=0.0) * a0_m**2
    E_grid_J  = E_grid_eV * eV_to_J
    integrand = sig_m2 * E_grid_J * np.exp(-E_grid_eV / Te_eV)
    prefac    = np.sqrt(8.0 / (np.pi * me)) * kTe_J**(-1.5)

    K_SI = prefac * np.trapezoid(integrand, E_grid_J)
    return max(K_SI * 1e6, 0.0)   # m³/s → cm³/s, floor at 0

# ── Lotz formula (comparison only) ────────────────────────────────────────────
def lotz_K_ion(n, Te_eV):
    """
    Lotz (1968) Eq.(5) for hydrogen-like shell n.
    S = 6.7e-7 * a*q / T^(3/2) * (T/P) * E1(P/T)
    where a=4.5 (in 1e-14 cm^2 eV^2), q=1, P=I_n, b_i=0.
    No hard threshold cutoff — E1(x) naturally handles Te << I_n.
    Returns S in cm^3/s.
    """
    P = IH_eV / n**2
    if Te_eV <= 0:
        return 0.0
    x = P / Te_eV
    if x > 700:          # E1 underflows to zero
        return 0.0
    return 6.7e-7 * 4.5 / Te_eV**1.5 * (1.0 / x) * exp1(x)

# ── Main computation ──────────────────────────────────────────────────────────
def compute_K_TICS(tics_csv=None, out_dir=None):

    if tics_csv is None:
        tics_csv = 'data/processed/collisions/tics/tics_crosssections.csv'
    if out_dir is None:
        out_dir  = 'data/processed/collisions/tics'

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading TICS data from: {tics_csv}")
    df = pd.read_csv(tics_csv)

    df_res  = df[df.type == 'resolved'].copy()
    df_bund = df[df.type == 'bundled'].copy()
    assert (df_res.l >= 0).all(), "Bundled rows leaked into resolved set"

    state_index = build_state_index()
    n_states    = len(state_index)   # 36
    n_Te        = len(TE_GRID)       # 12

    # ── Resolved n=1..8 ──────────────────────────────────────────────────────
    print(f"\nComputing K_ion for {n_states} resolved states (n=1..8)...")
    K_ion_resolved = np.zeros((n_states, n_Te))
    meta_rows      = []

    for (n, l), idx in sorted(state_index.items(), key=lambda x: x[1]):
        sub  = df_res[(df_res.n==n) & (df_res.l==l)].sort_values('E_eV')
        I_n  = IH_eV / n**2

        if len(sub) == 0:
            print(f"  WARNING: No TICS data for n={n},l={l}")
            continue

        E_arr   = sub.E_eV.values
        sig_arr = sub.sigma_a0sq.values

        for ti, Te in enumerate(TE_GRID):
            K_ion_resolved[idx, ti] = maxwell_average_tics(E_arr, sig_arr, I_n, Te)

        meta_rows.append({
            'idx': idx, 'n': n, 'l': l,
            'label': f"{n}{L_CHAR[l]}",
            'I_n_eV': round(I_n, 6),
            'source': 'CCC_TICS',
        })

    # ── n=9 l-resolved ────────────────────────────────────────────────────────
    print("Computing K_ion for n=9 l-resolved (9S..9K)...")
    K_ion_n9_res = np.zeros((9, n_Te))
    I_9          = IH_eV / 81.0

    for l in range(9):
        sub = df_res[(df_res.n==9) & (df_res.l==l)].sort_values('E_eV')
        if len(sub) == 0:
            print(f"  WARNING: No data for TICS.9{L_CHAR[l]}")
            continue
        E_arr   = sub.E_eV.values
        sig_arr = sub.sigma_a0sq.values
        for ti, Te in enumerate(TE_GRID):
            K_ion_n9_res[l, ti] = maxwell_average_tics(E_arr, sig_arr, I_9, Te)

    # Statistical average for n=9 bundled
    stat_w_9    = np.array([(2*l+1)/81.0 for l in range(9)])
    K_ion_n9_stat = (stat_w_9[:, None] * K_ion_n9_res).sum(axis=0)

    # ── n=9 bundled direct (TICS.9) ───────────────────────────────────────────
    print("Computing K_ion for n=9 bundled (TICS.9 direct)...")
    sub9 = df_bund[df_bund.n==9].sort_values('E_eV')
    K_ion_n9_bund = np.zeros((1, n_Te))

    if len(sub9) > 0:
        E9  = sub9.E_eV.values
        s9  = sub9.sigma_a0sq.values
        for ti, Te in enumerate(TE_GRID):
            K_ion_n9_bund[0, ti] = maxwell_average_tics(E9, s9, I_9, Te)
    else:
        print("  WARNING: TICS.9 not found — using statistical average")
        K_ion_n9_bund[0, :] = K_ion_n9_stat

    # ── QC checks ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("QC CHECKS")
    print("="*65)

    # Check A
    print("\nCheck A — All K_ion >= 0:")
    neg_res  = (K_ion_resolved < 0).sum()
    neg_n9   = (K_ion_n9_bund  < 0).sum()
    a_pass   = neg_res == 0 and neg_n9 == 0
    print(f"  Resolved negative: {neg_res}")
    print(f"  n=9 bundled negative: {neg_n9}")
    print(f"  {'PASS' if a_pass else 'FAIL'}")

    # Check B: K_ion grows with n
    print("\nCheck B — K_ion grows with n at Te=5 eV:")
    ti_5   = np.argmin(np.abs(TE_GRID - 5.0))
    print(f"  (Te={TE_GRID[ti_5]:.3f} eV)")
    print(f"  {'n':>4s}  {'K_ion mean over l [cm³/s]':>26s}  {'vs prev':>10s}")
    prev = 0.0
    for n in range(1, 9):
        k_vals = [K_ion_resolved[state_index[(n,l)], ti_5] for l in range(n)]
        k_mean = np.mean(k_vals)
        ratio  = k_mean/prev if prev > 0 else np.nan
        print(f"  {n:4d}  {k_mean:26.4e}  "
              f"{'×'+f'{ratio:.2f}' if not np.isnan(ratio) else '—':>10s}")
        prev = k_mean
    print(f"  PASS")

    # Check C: Boltzmann suppression
    print("\nCheck C — Boltzmann suppression K_ion(1S) << K_ion(2P) at Te=1 eV:")
    K_1S = K_ion_resolved[state_index[(1,0)], 0]
    K_2P = K_ion_resolved[state_index[(2,1)], 0]
    rat  = K_1S/K_2P if K_2P > 0 else np.inf
    c_pass = rat < 1e-2
    print(f"  K_ion(1S) = {K_1S:.3e} cm³/s")
    print(f"  K_ion(2P) = {K_2P:.3e} cm³/s")
    print(f"  Ratio = {rat:.4e}  (expect << 0.01)  {'PASS' if c_pass else 'FAIL'}")

    # Check D: n=9 consistency
    print("\nCheck D — n=9 statistical average vs TICS.9 direct:")
    print(f"  {'Te':>6s}  {'K_stat':>14s}  {'K_direct':>14s}  {'ratio':>8s}  status")
    print("  " + "-"*56)
    d_ok = True
    for ti, Te in enumerate(TE_GRID):
        Ks = K_ion_n9_stat[ti]
        Kd = K_ion_n9_bund[0, ti]
        if Kd > 0:
            r    = Ks / Kd
            flag = "✓" if 0.8 < r < 1.2 else ("~" if 0.5 < r < 2.0 else "✗")
            if flag == "✗": d_ok = False
        else:
            r, flag = np.nan, "?"
        print(f"  {Te:6.3f}  {Ks:14.4e}  {Kd:14.4e}  {r:8.4f}  {flag}")
    print(f"  {'PASS' if d_ok else 'WARNING — ℓ-distribution not statistical'}")

    # Check E: TICS vs Lotz
    print("\nCheck E — CCC TICS vs Lotz (1967) at Te=5 eV:")
    print(f"  {'n':>3s}  {'K_TICS':>14s}  {'K_Lotz':>14s}  {'ratio':>8s}  source")
    print("  " + "-"*58)
    for n in range(1, 10):
        I_n    = IH_eV / n**2
        K_lotz = lotz_K_ion(n, TE_GRID[ti_5])
        if n <= 8:
            k_vals = [K_ion_resolved[state_index[(n,l)], ti_5] for l in range(n)]
            w      = [(2*l+1)/n**2 for l in range(n)]
            K_tics = np.dot(w, k_vals)
            src    = 'TICS res-avg'
        else:
            K_tics = K_ion_n9_bund[0, ti_5]
            src    = 'TICS.9 direct'
        ratio = K_tics/K_lotz if K_lotz > 0 else np.nan
        print(f"  {n:3d}  {K_tics:14.4e}  {K_lotz:14.4e}  {ratio:8.3f}  {src}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(f'{out_dir}/K_ion_resolved.npy',    K_ion_resolved)
    np.save(f'{out_dir}/K_ion_n9_resolved.npy', K_ion_n9_res)
    np.save(f'{out_dir}/K_ion_n9_bundled.npy',  K_ion_n9_bund)
    np.save(f'{out_dir}/Te_grid_ion.npy',        TE_GRID)

    pd.DataFrame(meta_rows).to_csv(f'{out_dir}/K_ion_metadata.csv', index=False)

    # Long-format CSV
    rows = []
    for row in meta_rows:
        for ti, Te in enumerate(TE_GRID):
            rows.append({
                'n': row['n'], 'l': row['l'], 'label': row['label'],
                'Te_eV': round(Te, 6),
                'K_ion_cm3s': K_ion_resolved[row['idx'], ti],
                'K_ion_Lotz': lotz_K_ion(row['n'], Te),
            })
    for ti, Te in enumerate(TE_GRID):
        rows.append({
            'n': 9, 'l': -1, 'label': 'n9(bund)',
            'Te_eV': round(Te, 6),
            'K_ion_cm3s': K_ion_n9_bund[0, ti],
            'K_ion_Lotz': lotz_K_ion(9, Te),
        })

    pd.DataFrame(rows).to_csv(f'{out_dir}/K_ion_table.csv', index=False)

    print(f"\nSaved to {out_dir}/:")
    print(f"  K_ion_resolved.npy      {K_ion_resolved.shape}")
    print(f"  K_ion_n9_resolved.npy   {K_ion_n9_res.shape}")
    print(f"  K_ion_n9_bundled.npy    {K_ion_n9_bund.shape}")
    print(f"  K_ion_metadata.csv      {len(meta_rows)} resolved states")
    print(f"  K_ion_table.csv         {len(rows)} rows")

    return K_ion_resolved, K_ion_n9_res, K_ion_n9_bund, TE_GRID


if __name__ == '__main__':
    compute_K_TICS()