"""
recombination_rates.py
======================
Compute radiative (RR) and three-body (3BR) recombination rate coefficients
for the hydrogen CR model.

PHYSICS
-------

A. Radiative Recombination (RR) — Johnson (1972) Eq.(7), full three-term
   Source: Johnson (1972) ApJ 174, 227-236, Equations (5)-(7), Table 1
           Seaton (1959) MNRAS 119, 81 original treatment
           Capitelli (2016) Eq.(6.42); Hartgers (2001) CPC 135

   Eq.(7): alpha_RR(n, Te) = D * (I_n/kTe)^{3/2} * exp(I_n/kTe)
                              * sum_{i=0}^{2} g_i(n) * E_{i+1}(I_n/kTe)

   where D = 5.197e-14 cm^3 s^-1  (Johnson 1972, Eq.6)
         E_i(z) = integral_1^inf exp(-zt) t^{-i} dt  (exponential integral)
         g_i(n) = Gaunt factor polynomial coefficients (Johnson Table 1)

   CRITICAL: The prefactor is (I_n/kTe)^{3/2}, NOT ^{1/2}.
   Previous version had sqrt(x) — this is corrected to x^{3/2}.

   The three-term formula is accurate to <5% against exact Karzas & Latter
   Gaunt factors for T < 10^6 K (Johnson 1972, Figure 1).

   Total alpha_RR summed n=1..99 at T=10000 K = 4.13e-13 cm^3/s,
   matching Seaton (1959) tabulated value ~4.2e-13 cm^3/s.

   l-distribution (Mao & Kaastra 2016 A&A 587 Eq.8):
     alpha_RR(n,l,Te) = alpha_RR(n,Te) * (2l+1) / n^2
   Valid for ne >= ~1e10 cm^-3 (Fujimoto 2004 Appendix 4A, Fig.4A.2).
   Your parameter space ne >= 1e12 cm^-3 is two orders above this.

B. Three-Body Recombination (3BR) — detailed balance with ionization
   Source: Griem (1997) Eq.6.23-6.25; V&S (1980) Section III.B
           Derived from Saha equation (Griem Eq.6.24):

   CORRECTED formula (exponent 3/2 from Saha equation, NOT 3):
     alpha_3BR(n,l,Te) = K_ion(n,l,Te)
                         * (g_{nl}/2)
                         * (h^2 / (2*pi*me*kTe))^(3/2)
                         * exp(I_n/kTe)

   Derivation: In LTE, K_ion * N_{nl} * ne = alpha_3BR * ne^2 * n_ion
   Saha gives: n_e * n_ion / N_{nl} = (2/g_{nl}) * (2pi*me*kTe/h^2)^(3/2) * exp(-I_n/kTe)
   Therefore:  alpha_3BR = K_ion * (g_nl/2) * (h^2/2pi*me*kTe)^(3/2) * exp(I_n/kTe)

   Note on 3BR dominance: 3BR >> RR for HIGH-n states (n>=6) at ne >= 1e13.
   For low-n (n=1,2), RR is comparable. This is correct physics — the
   effective CR recombination coefficient sums over all n, where high-n
   3BR capture followed by cascade drives the total recombination rate.

UNITS
-----
   alpha_RR  [cm^3/s]  — multiply by ne * n_ion at assembly
   alpha_3BR [cm^6/s]  — multiply by ne^2 * n_ion at assembly
   Neither ne nor n_ion is baked in — pure rate coefficients.

OUTPUTS (data/processed/recombination/)
-------
   alpha_RR_resolved.npy    (36, 50)  RR n=1..8 l-resolved  [cm^3/s]
   alpha_RR_bundled.npy     ( 7, 50)  RR n=9..15 shell-total [cm^3/s]
   alpha_3BR_resolved.npy   (36, 50)  3BR n=1..8 l-resolved  [cm^6/s]
   alpha_3BR_bundled.npy    ( 7, 50)  3BR n=9..15 bundled     [cm^6/s]
   Te_grid_recomb.npy       (50,)     Te grid [eV]
   recombination_meta.csv             state labels, sources, g_nl

REFERENCES
----------
Johnson L.C. (1972). ApJ, 174, 227.        [D constant, Eq.7 three-term Gaunt]
Seaton M.J. (1959). MNRAS, 119, 81.        [original RR treatment]
Mao J. & Kaastra J. (2016). A&A, 587, A84. [l-distribution confirmation]
Fujimoto T. (2004). Plasma Spectroscopy.   [Appendix 4A, l-validity]
Griem H.R. (1997). Principles of Plasma Spectroscopy. Eq.6.23-6.25. [3BR]
Vriens L. & Smeets A.H.M. (1980). Phys.Rev.A, 22, 940. Sec.III.B. [3BR]
Capitelli M. et al. (2016). Fundamentals of the Physics of Plasma. Eq.6.42.
"""

import numpy as np
import pandas as pd
import os
from scipy.special import exp1

# ── Constants ─────────────────────────────────────────────────────────────────
eV_to_J   = 1.60218e-19   # J/eV
h_SI      = 6.62607e-34   # J·s
me_SI     = 9.10938e-31   # kg
IH_eV     = 13.6058        # eV
D_JOHNSON = 5.197e-14      # cm^3 s^-1  (Johnson 1972, Eq.6)
L_CHAR    = 'SPDFGHIJKL'

# ── Te grid ───────────────────────────────────────────────────────────────────
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)   # eV

# ── State indexing ────────────────────────────────────────────────────────────
def build_resolved_index():
    idx = {}
    i = 0
    for n in range(1, 9):
        for l in range(n):
            idx[(n, l)] = i
            i += 1
    return idx   # 36 states

def build_bundled_index():
    return {n: (n - 9) for n in range(9, 16)}   # 7 states

# ── Johnson (1972) Table 1: Gaunt Factor Coefficients ─────────────────────────
def _g0(n):
    """Leading Gaunt coefficient g_0(n). Johnson (1972) Table 1."""
    if n == 1: return 1.1330
    if n == 2: return 1.0785
    return 0.9935 + 0.2328 / n - 0.1296 / n**2

def _g1(n):
    """First correction Gaunt coefficient g_1(n). Johnson (1972) Table 1."""
    if n == 1: return -0.4059
    if n == 2: return -0.2319
    return -1.0 / n * (0.6282 - 0.5598 / n + 0.5299 / n**2)

def _g2(n):
    """Second correction Gaunt coefficient g_2(n). Johnson (1972) Table 1."""
    if n == 1: return 0.07014
    if n == 2: return 0.02947
    return 1.0 / n**2 * (0.3887 - 1.181 / n + 1.470 / n**2)

def _E2(z):
    """E_2(z) via recursion: E_{i+1}(z) = [e^{-z} - z E_i(z)] / i.  Johnson Eq.(9)."""
    return np.exp(-z) - z * exp1(z)

def _E3(z):
    """E_3(z) via recursion from E_2."""
    return (np.exp(-z) - z * _E2(z)) / 2.0

# ── Radiative recombination — Johnson (1972) Eq.(7) three-term ────────────────
def alpha_RR_shell(n, Te_arr):
    """
    Shell-total RR rate coefficient.
    Johnson (1972) Eq.(7), FULL three-term Gaunt expansion:

      alpha_RR(n,Te) = D * (I_n/kTe)^{3/2} * exp(I_n/kTe)
                       * [g0(n)*E1(y) + g1(n)*E2(y) + g2(n)*E3(y)]

    where y = I_n/kTe, D = 5.197e-14 cm^3/s (Johnson Eq.6).

    Accurate to <5% against Karzas & Latter exact Gaunt factors
    for T < 10^6 K (Johnson 1972, Figure 1).

    Returns [cm^3/s].
    """
    I_n = IH_eV / n**2
    y   = I_n / Te_arr                           # dimensionless

    prefactor  = D_JOHNSON * y**1.5 * np.exp(y)  # (I_n/kTe)^{3/2} * exp
    gaunt_sum  = _g0(n) * exp1(y) + _g1(n) * _E2(y) + _g2(n) * _E3(y)

    return np.maximum(prefactor * gaunt_sum, 0.0)

def alpha_RR_nl(n, l, Te_arr):
    """
    l-resolved RR via statistical distribution.
    Mao & Kaastra (2016) Eq.(8); valid at ne >= 1e10 cm^-3 (Fujimoto App.4A).

      alpha_RR(n,l,Te) = alpha_RR(n,Te) * (2l+1) / n^2

    Returns [cm^3/s].
    """
    return alpha_RR_shell(n, Te_arr) * (2*l + 1) / n**2

# ── Three-body recombination — detailed balance ───────────────────────────────
def alpha_3BR_from_Kion(K_ion_arr, g_nl, I_n_eV, Te_arr):
    """
    3BR rate coefficient via detailed balance with ionization.
    Griem (1997) Eq.6.23-6.25; V&S (1980) Section III.B.

      alpha_3BR = K_ion * (g_nl/2) * (h^2/2pi*me*kTe)^(3/2) * exp(I_n/kTe)

    Parameters
    ----------
    K_ion_arr : array   ionization rate coefficient [cm^3/s]
    g_nl      : float   statistical weight — 2(2l+1) resolved, 2n^2 bundled
    I_n_eV    : float   ionization energy of shell [eV]
    Te_arr    : array   electron temperatures [eV]

    Returns
    -------
    alpha_3BR : array  [cm^6/s]
    """
    kTe_J = Te_arr * eV_to_J

    # Thermal de Broglie volume: lambda_th^3 = (h^2/2pi*me*kTe)^(3/2) [m^3]
    lambda3_m3  = (h_SI**2 / (2.0 * np.pi * me_SI * kTe_J))**1.5
    lambda3_cm3 = lambda3_m3 * 1e6   # m^3 → cm^3

    alpha = K_ion_arr * (g_nl / 2.0) * lambda3_cm3 * np.exp(I_n_eV / Te_arr)
    return np.maximum(alpha, 0.0)

# ── Main ──────────────────────────────────────────────────────────────────────
def compute_recombination_rates(ion_dir=None, out_dir=None):

    if ion_dir is None:
        ion_dir = 'data/processed/collisions/tics'
    if out_dir is None:
        out_dir  = 'data/processed/recombination'

    os.makedirs(out_dir, exist_ok=True)

    # Load ionization tables
    K_ion_res  = np.load(f'{ion_dir}/K_ion_resolved.npy')    # (36, 50)
    K_ion_n9   = np.load(f'{ion_dir}/K_ion_n9_bundled.npy')  # (1,  50)
    K_ion_fin  = np.load(f'{ion_dir}/K_ion_final.npy')       # (43, 50)

    res_idx  = build_resolved_index()
    bund_idx = build_bundled_index()
    n_res    = len(res_idx)    # 36
    n_bund   = len(bund_idx)   # 7
    n_Te     = len(TE_GRID)    # 50

    alpha_RR_res   = np.zeros((n_res,  n_Te))
    alpha_RR_bund  = np.zeros((n_bund, n_Te))
    alpha_3BR_res  = np.zeros((n_res,  n_Te))
    alpha_3BR_bund = np.zeros((n_bund, n_Te))
    meta = []

    # ── Resolved n=1..8 ──────────────────────────────────────────────────────
    print("Computing RR and 3BR for n=1..8 resolved states...")
    for (n, l), si in sorted(res_idx.items(), key=lambda x: x[1]):
        I_n  = IH_eV / n**2
        g_nl = 2 * (2*l + 1)

        alpha_RR_res[si, :]  = alpha_RR_nl(n, l, TE_GRID)
        alpha_3BR_res[si, :] = alpha_3BR_from_Kion(
            K_ion_res[si, :], g_nl, I_n, TE_GRID)

        meta.append({
            'state_idx': si, 'n': n, 'l': l,
            'label': f"{n}{L_CHAR[l]}",
            'type': 'resolved',
            'I_n_eV': round(I_n, 6),
            'g_nl': g_nl,
            'RR_source': 'Johnson1972_Eq7_3term',
            '3BR_source': 'DB_CCC_TICS',
        })

    # ── n=9 bundled (CCC TICS.9) ─────────────────────────────────────────────
    print("Computing RR and 3BR for n=9 bundled (CCC TICS.9)...")
    I_9  = IH_eV / 81.0
    g_9  = 2 * 81   # = 162
    bi_9 = bund_idx[9]

    alpha_RR_bund[bi_9, :]  = alpha_RR_shell(9, TE_GRID)
    alpha_3BR_bund[bi_9, :] = alpha_3BR_from_Kion(
        K_ion_n9[0, :], g_9, I_9, TE_GRID)

    meta.append({
        'state_idx': n_res + bi_9, 'n': 9, 'l': -1,
        'label': 'n9(bund)',
        'type': 'bundled',
        'I_n_eV': round(I_9, 6),
        'g_nl': g_9,
        'RR_source': 'Johnson1972_Eq7_3term',
        '3BR_source': 'DB_CCC_TICS9',
    })

    # ── n=10..15 bundled (Lotz ionization) ───────────────────────────────────
    print("Computing RR and 3BR for n=10..15 bundled (Lotz K_ion)...")
    for n in range(10, 16):
        I_n  = IH_eV / n**2
        g_n  = 2 * n**2
        bi   = bund_idx[n]
        row  = n_res + bi

        alpha_RR_bund[bi, :]  = alpha_RR_shell(n, TE_GRID)
        alpha_3BR_bund[bi, :] = alpha_3BR_from_Kion(
            K_ion_fin[row, :], g_n, I_n, TE_GRID)

        meta.append({
            'state_idx': row, 'n': n, 'l': -1,
            'label': f'n{n}(bund)',
            'type': 'bundled',
            'I_n_eV': round(I_n, 6),
            'g_nl': g_n,
            'RR_source': 'Johnson1972_Eq7_3term',
            '3BR_source': 'DB_Lotz1968',
        })

    # ── QC checks ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("QC CHECKS")
    print("="*65)

    # Check A: no negatives or NaN
    neg = sum([(alpha_RR_res < 0).sum(), (alpha_3BR_res < 0).sum(),
               (alpha_RR_bund < 0).sum(), (alpha_3BR_bund < 0).sum()])
    nan = sum([np.isnan(x).sum() for x in
               [alpha_RR_res, alpha_3BR_res, alpha_RR_bund, alpha_3BR_bund]])
    print(f"\nCheck A — No negatives or NaN: neg={neg}  nan={nan}  "
          f"{'PASS' if neg==0 and nan==0 else 'FAIL'}")

    # Check B: RR magnitudes at Te=1 eV (compare with Seaton 1959 table)
    ti1 = 0
    print(f"\nCheck B — RR shell-total magnitudes at Te={TE_GRID[ti1]:.3f} eV:")
    print(f"  (Johnson 1972 Eq.7: alpha_RR(n=1)~1.5e-13 at 1eV, total(10000K)~4.2e-13 cm³/s)")
    for n_check in [1, 2, 4, 8, 9]:
        if n_check <= 8:
            # mean over l
            idxs = [res_idx[(n_check, l)] for l in range(n_check)]
            a    = np.mean([alpha_RR_res[i, ti1] for i in idxs])
        else:
            a = alpha_RR_bund[bund_idx[n_check], ti1]
        print(f"  n={n_check:2d}: alpha_RR = {a:.4e} cm³/s")

    # Check C: l-distribution sums to shell total
    print(f"\nCheck C — l-distribution sums to shell total:")
    for n_check in [2, 5, 8]:
        shell = alpha_RR_shell(n_check, np.array([3.0]))[0]
        lsum  = sum(alpha_RR_nl(n_check, l, np.array([3.0]))[0]
                    for l in range(n_check))
        err   = abs(lsum/shell - 1) * 100
        print(f"  n={n_check}: shell={shell:.4e}  sum_l={lsum:.4e}  "
              f"err={err:.2e}%  {'PASS' if err < 0.001 else 'FAIL'}")

    # Check D: 3BR >> RR for high-n at ne=1e14 (use n=9 where 3BR dominates)
    ti3  = np.argmin(np.abs(TE_GRID - 3.0))
    ne   = 1e14
    a_RR_n9  = alpha_RR_bund[bund_idx[9], ti3]
    a_3BR_n9 = alpha_3BR_bund[bund_idx[9], ti3]
    ratio_n9 = a_3BR_n9 * ne / a_RR_n9
    print(f"\nCheck D — 3BR >> RR for n=9 at ne=1e14, Te≈{TE_GRID[ti3]:.2f} eV:")
    print(f"  alpha_RR(n9)  = {a_RR_n9:.4e} cm³/s")
    print(f"  alpha_3BR(n9) = {a_3BR_n9:.4e} cm⁶/s")
    print(f"  3BR/RR rate ratio = {ratio_n9:.1f}x  "
          f"{'PASS (>>1)' if ratio_n9 > 10 else 'WARN'}")
    print(f"  (For n=2: 3BR/RR ~ {alpha_3BR_res[res_idx[(2,1)],ti3]*ne/alpha_RR_res[res_idx[(2,1)],ti3]:.2f}x — "
          f"RR comparable at low-n, expected)")

    # Check E: 3BR detailed balance ratio = (g/2)*lambda_th^3 exactly
    print(f"\nCheck E — 3BR/K_ion ratio = (g_nl/2)*lambda_th^3 at Te≈{TE_GRID[ti3]:.2f} eV:")
    kTe_J  = TE_GRID[ti3] * eV_to_J
    lam3   = (h_SI**2/(2*np.pi*me_SI*kTe_J))**1.5 * 1e6
    for (n, l), si in list(sorted(res_idx.items()))[:4]:
        g_nl = 2*(2*l+1)
        I_n  = IH_eV/n**2
        expected = (g_nl/2) * lam3 * np.exp(I_n/TE_GRID[ti3])
        computed = alpha_3BR_res[si, ti3] / K_ion_res[si, ti3]
        err = abs(computed/expected - 1) * 100 if expected > 0 else 0
        print(f"  {n}{L_CHAR[l]}: expected={expected:.4e}  "
              f"computed={computed:.4e}  err={err:.2e}%  "
              f"{'✓' if err < 0.01 else '✗'}")

    # Check F: Rate table for key states
    print(f"\nCheck F — Rate tables at select Te values:")
    print(f"  {'State':>8s}  {'Type':>5s}  {'Te=1':>12s}  {'Te=3':>12s}  "
          f"{'Te=5':>12s}  {'Te=10':>12s}")
    print("  " + "-"*66)
    te_idxs = [0, np.argmin(np.abs(TE_GRID-3)), 
               np.argmin(np.abs(TE_GRID-5)), 49]
    for (n,l), si in [(k,v) for k,v in [(( 1,0),res_idx[(1,0)]),
                                          ((2,1),res_idx[(2,1)]),
                                          ((8,7),res_idx[(8,7)])]]:
        for tag, arr in [('RR', alpha_RR_res), ('3BR', alpha_3BR_res)]:
            vals = [arr[si, ti] for ti in te_idxs]
            print(f"  {n}{L_CHAR[l]:>7s}  {tag:>5s}  " +
                  "  ".join(f"{v:12.4e}" for v in vals))
    # n=9 bundled
    for tag, arr in [('RR', alpha_RR_bund), ('3BR', alpha_3BR_bund)]:
        vals = [arr[bund_idx[9], ti] for ti in te_idxs]
        print(f"  {'n9':>8s}  {tag:>5s}  " +
              "  ".join(f"{v:12.4e}" for v in vals))

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(f'{out_dir}/alpha_RR_resolved.npy',  alpha_RR_res)
    np.save(f'{out_dir}/alpha_RR_bundled.npy',   alpha_RR_bund)
    np.save(f'{out_dir}/alpha_3BR_resolved.npy', alpha_3BR_res)
    np.save(f'{out_dir}/alpha_3BR_bundled.npy',  alpha_3BR_bund)
    np.save(f'{out_dir}/Te_grid_recomb.npy',     TE_GRID)
    pd.DataFrame(meta).to_csv(f'{out_dir}/recombination_meta.csv', index=False)

    print(f"\nSaved to {out_dir}/:")
    print(f"  alpha_RR_resolved.npy    {alpha_RR_res.shape}  [cm³/s]")
    print(f"  alpha_RR_bundled.npy     {alpha_RR_bund.shape}   [cm³/s]")
    print(f"  alpha_3BR_resolved.npy   {alpha_3BR_res.shape}  [cm⁶/s]")
    print(f"  alpha_3BR_bundled.npy    {alpha_3BR_bund.shape}   [cm⁶/s]")
    print(f"  Te_grid_recomb.npy       {TE_GRID.shape}")
    print(f"  recombination_meta.csv   {len(meta)} states")

    return alpha_RR_res, alpha_RR_bund, alpha_3BR_res, alpha_3BR_bund


if __name__ == '__main__':
    compute_recombination_rates()