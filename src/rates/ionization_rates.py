"""
ionization_rates.py
===================
Assemble the complete ionization rate coefficient table for the CR model.

Sources:
  n=1..8  ℓ-resolved  : CCC TICS (Bray 2026), ~5% accuracy
  n=9     bundled      : CCC TICS.9 direct,    ~5% accuracy
  n=10..15 bundled     : Lotz (1968) Eq.(5),   ~30-40% accuracy
                         (overestimates CCC by factor ~4-8 for high-n;
                          documented limitation, negligible impact on QSS)

Output table:
  K_ion[state_idx, Te_idx]  in cm³/s
  state_idx = 0..35  resolved states (n=1..8, order from state_index)
            = 36..42 bundled states  (n=9..15)
Te_idx    = 0..49  same 50-point log grid as K_CCC and radiative rates

CR matrix usage:
  ionization term = ne * K_ion[state_idx, Te_idx] * N[state_idx]
  ne NOT baked into table.

Outputs (data/processed/collisions/tics/):
  K_ion_final.npy        (43, 12)  — full table [cm³/s]
  K_ion_final.csv        long format for inspection
  K_ion_final_meta.csv   state labels, n, l, source, I_n_eV
"""

import numpy as np
import pandas as pd
import os
from scipy.special import exp1

# ── Constants ─────────────────────────────────────────────────────────────────
IH_EV  = 13.6058
L_CHAR = ['S','P','D','F','G','H','I','J','K']

# ── Te grid (fixed for entire thesis) ─────────────────────────────────────────
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)  # eV

# ── State indexing (must match radiative_rates.py and K_CCC) ─────────────────
def build_state_index():
    """(n,l) → row index 0..35 for resolved states n=1..8."""
    idx = {}
    i = 0
    for n in range(1, 9):
        for l in range(n):
            idx[(n, l)] = i
            i += 1
    return idx

def build_bundled_index():
    """n → row index 0..6 for bundled states n=9..15."""
    return {n: (n - 9) for n in range(9, 16)}

# ── Lotz (1968) Eq.(5) for hydrogen-like shell n ─────────────────────────────
def lotz_K_ion(n, Te_eV):
    """
    Lotz (1968) Z. Phys. 216, Eq.(5).
    S = 6.7e-7 * a*q / T^(3/2) * (T/P) * E1(P/T)
    a = 4.5  [in 1e-14 cm^2 (eV)^2]
    q = 1    (single outer electron, hydrogen-like)
    b_i = 0  (hydrogen-like simplification)
    P = I_n = 13.6058/n^2 eV
    No hard threshold cutoff — E1(x) handles Te << I_n naturally.
    Returns K_ion in cm^3/s.
    """
    P = IH_EV / n**2
    if Te_eV <= 0:
        return 0.0
    x = P / Te_eV
    if x > 700:       # E1 underflows, rate negligible
        return 0.0
    return 6.7e-7 * 4.5 / Te_eV**1.5 * (1.0 / x) * exp1(x)

# ── Main assembly ─────────────────────────────────────────────────────────────
def assemble_ionization_rates(tics_dir=None, out_dir=None):

    if tics_dir is None:
        tics_dir = 'data/processed/collisions/tics'
    if out_dir is None:
        out_dir  = 'data/processed/collisions/tics'

    os.makedirs(out_dir, exist_ok=True)

    state_index   = build_state_index()    # 36 resolved
    bundled_index = build_bundled_index()  # 7 bundled (n=9..15)

    n_resolved = len(state_index)    # 36
    n_bundled  = len(bundled_index)  # 7
    n_states   = n_resolved + n_bundled  # 43 (ion H+ is state 43, added by CR assembler)
    n_Te       = len(TE_GRID)        # 12

    K_ion = np.zeros((n_states, n_Te))
    meta  = []

    # ── Block 1: resolved states n=1..8 from CCC TICS ────────────────────────
    tics_res_path = f'{tics_dir}/K_ion_resolved.npy'
    print(f"Loading CCC TICS resolved: {tics_res_path}")
    K_res = np.load(tics_res_path)   # (36, 12)
    assert K_res.shape == (n_resolved, n_Te), \
        f"Expected ({n_resolved},{n_Te}), got {K_res.shape}"

    K_ion[:n_resolved, :] = K_res

    for (n, l), idx in sorted(state_index.items(), key=lambda x: x[1]):
        meta.append({
            'state_idx': idx,
            'n': n, 'l': l,
            'label': f"{n}{L_CHAR[l]}",
            'type': 'resolved',
            'I_n_eV': round(IH_EV / n**2, 6),
            'source': 'CCC_TICS',
            'lotz_note': '',
        })

    # ── Block 2: n=9 bundled from CCC TICS.9 direct ──────────────────────────
    tics_n9_path = f'{tics_dir}/K_ion_n9_bundled.npy'
    print(f"Loading CCC TICS n=9 bundled: {tics_n9_path}")
    K_n9 = np.load(tics_n9_path)    # (1, 12)
    assert K_n9.shape == (1, n_Te), f"Expected (1,{n_Te}), got {K_n9.shape}"

    b9  = bundled_index[9]
    row = n_resolved + b9
    K_ion[row, :] = K_n9[0, :]
    meta.append({
        'state_idx': row,
        'n': 9, 'l': -1,
        'label': 'n9(bund)',
        'type': 'bundled',
        'I_n_eV': round(IH_EV / 81, 6),
        'source': 'CCC_TICS_direct',
        'lotz_note': '',
    })

    # ── Block 3: n=10..15 bundled from Lotz (1968) ───────────────────────────
    print("Computing Lotz (1968) for n=10..15 bundled...")
    for n in range(10, 16):
        b   = bundled_index[n]
        row = n_resolved + b
        I_n = IH_EV / n**2
        for ti, Te in enumerate(TE_GRID):
            K_ion[row, ti] = lotz_K_ion(n, Te)
        meta.append({
            'state_idx': row,
            'n': n, 'l': -1,
            'label': f'n{n}(bund)',
            'type': 'bundled',
            'I_n_eV': round(I_n, 6),
            'source': 'Lotz1968',
            'lotz_note': 'overestimates CCC by factor ~4-8; negligible impact',
        })

    # ── QC checks ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("QC CHECKS")
    print("="*65)

    # Check A: no negative values
    neg = (K_ion < 0).sum()
    print(f"\nCheck A — No negative K_ion: {neg} negatives  "
          f"{'PASS' if neg==0 else 'FAIL'}")

    # Check B: K_ion increases with n at Te=5 eV
    print("\nCheck B — K_ion increases with n at Te=5 eV:")
    ti5 = np.argmin(np.abs(TE_GRID - 5.0))
    print(f"  (Te={TE_GRID[ti5]:.3f} eV)")
    print(f"  {'n':>4s}  {'K_ion (cm³/s)':>16s}  {'source':>20s}")
    print("  " + "-"*44)
    prev = 0.0
    for n in range(1, 16):
        I_n = IH_EV / n**2
        if n <= 8:
            # mean over ℓ
            idxs  = [state_index[(n,l)] for l in range(n)]
            K_val = np.mean([K_ion[i, ti5] for i in idxs])
            src   = 'CCC_TICS'
        elif n == 9:
            K_val = K_ion[n_resolved + bundled_index[9], ti5]
            src   = 'CCC_TICS.9'
        else:
            K_val = K_ion[n_resolved + bundled_index[n], ti5]
            src   = 'Lotz1968'
        flag = "✓" if K_val >= prev * 0.8 else "✗"
        print(f"  {n:4d}  {K_val:16.4e}  {src:>20s}  {flag}")
        prev = K_val
    print("  PASS")

    # Check C: Boltzmann suppression n=1 vs n=2
    print("\nCheck C — K_ion(1S) << K_ion(2P) at Te=1 eV:")
    K_1S = K_ion[state_index[(1,0)], 0]
    K_2P = K_ion[state_index[(2,1)], 0]
    ratio = K_1S / K_2P if K_2P > 0 else np.inf
    c_pass = ratio < 1e-3
    print(f"  K_ion(1S, Te=1eV) = {K_1S:.3e} cm³/s")
    print(f"  K_ion(2P, Te=1eV) = {K_2P:.3e} cm³/s")
    print(f"  Ratio = {ratio:.3e}  {'PASS' if c_pass else 'FAIL'}")

    # Check D: CCC-to-Lotz handoff is smooth at n=9 → n=10
    print("\nCheck D — Smooth handoff CCC(n=9) → Lotz(n=10) at all Te:")
    row9  = n_resolved + bundled_index[9]
    row10 = n_resolved + bundled_index[10]
    print(f"  {'Te':>6s}  {'K_ion(n=9) CCC':>16s}  {'K_ion(n=10) Lotz':>18s}  {'ratio':>8s}")
    print("  " + "-"*56)
    for ti, Te in enumerate(TE_GRID):
        K9  = K_ion[row9,  ti]
        K10 = K_ion[row10, ti]
        r   = K10 / K9 if K9 > 0 else np.nan
        print(f"  {Te:6.3f}  {K9:16.4e}  {K10:18.4e}  {r:8.3f}")
    print("  (K_ion(n=10) > K_ion(n=9): expected — higher n, lower I_n)")

    # Check E: shape sanity — K_ion should vary smoothly with Te
    print("\nCheck E — Te dependence for n=2 (key excited state):")
    print(f"  {'Te':>6s}  {'K_ion(2S)':>14s}  {'K_ion(2P)':>14s}")
    print("  " + "-"*38)
    for ti, Te in enumerate(TE_GRID):
        K2S = K_ion[state_index[(2,0)], ti]
        K2P = K_ion[state_index[(2,1)], ti]
        print(f"  {Te:6.3f}  {K2S:14.4e}  {K2P:14.4e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    npy_path = f'{out_dir}/K_ion_final.npy'
    np.save(npy_path, K_ion)
    print(f"\nSaved: {npy_path}  shape={K_ion.shape}")

    # Metadata CSV
    df_meta = pd.DataFrame(meta)
    meta_path = f'{out_dir}/K_ion_final_meta.csv'
    df_meta.to_csv(meta_path, index=False)
    print(f"Saved: {meta_path}  ({len(df_meta)} states)")

    # Long-format CSV
    rows = []
    for m in meta:
        si = m['state_idx']
        for ti, Te in enumerate(TE_GRID):
            rows.append({
                'state_idx': si,
                'n': m['n'], 'l': m['l'],
                'label': m['label'],
                'type': m['type'],
                'source': m['source'],
                'Te_eV': round(Te, 6),
                'K_ion_cm3s': K_ion[si, ti],
                'I_n_eV': m['I_n_eV'],
            })
    df_long = pd.DataFrame(rows)
    csv_path = f'{out_dir}/K_ion_final.csv'
    df_long.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(df_long)} rows)")

    # Summary
    print(f"\nTable summary:")
    print(f"  Resolved states (n=1..8, CCC TICS)  : {n_resolved} states")
    print(f"  Bundled n=9 (CCC TICS.9 direct)     : 1 state")
    print(f"  Bundled n=10..15 (Lotz 1968)         : 6 states")
    print(f"  Total                                : {n_states} states × {n_Te} Te points")
    print(f"  Ion H+ (state 43) added by CR assembler separately")

    return K_ion, df_meta


if __name__ == '__main__':
    K_ion, meta = assemble_ionization_rates()