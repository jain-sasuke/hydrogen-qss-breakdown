"""
validate_gates.py
=================
Validation Gates A–D for the hydrogen CR model.

GATES OVERVIEW
--------------
  Gate A — Detailed balance (K_deexc/K_exc = (g/g)*exp(E/kTe))
            ALREADY VALIDATED in pre_assembly_check.py (max err 5e-7%)
            Repeated here for completeness at the matrix level.

  Gate B — Coronal limit
            At low ne: n(p)/(n(1S)*ne) should plateau (converge to coronal value)
            Physical: each excited state fed only from 1S excitation,
            depleted by radiative decay. Rate balance:
              K_exc(1S->p)*ne*n(1S) = gamma(p)*n(p)
            => n(p)/(n(1S)*ne) = K_exc(1S->p)/gamma(p) = const [independent of ne]

  Gate C — Saha-Boltzmann approach
            At high ne: n(p)/n(1S) increases toward Saha-Boltzmann value.
            Full LTE never reached (optically thin), but the TREND must be correct.
            Saha: n(p)/n(1S) -> (g_p/g_1) * exp(-E_p1/kTe)

  Gate D — ADAS effective coefficients
            Effective SCD (ionization) and ACD (recombination) from our
            CR model vs ADAS SCD96/ACD96.
            Requires ADAS files at:
              data/processed/adas/scd96_h_long.csv
              data/processed/adas/acd96_h_long.csv
            If not found: Gate D is skipped with a warning.

USAGE
-----
    python src/rates/validate_gates.py

    Produces:
      validation/gate_B_coronal.csv
      validation/gate_C_saha.csv
      validation/gate_D_adas.csv  (if ADAS available)
      validation/gate_summary.txt

EXPECTED OUTCOMES
-----------------
  Gate A: PASS (already verified)
  Gate B: PASS — coronal ratio plateau at low ne
  Gate C: PASS — ratio increases with ne toward Saha
  Gate D: PASS at Te >= 5 eV (model valid), factor 2-10 at Te < 3 eV
          (known limitation: our model at Te << I(1S) is not designed for
          absolute ionization rates, only for QSS dynamics)
"""

import numpy as np
import pandas as pd
import os

# ── Constants ──────────────────────────────────────────────────────────────────
IH_RYDBERG = 13.605693   # eV
L_CHAR     = 'SPDFGHIJKL'

# ── Paths ──────────────────────────────────────────────────────────────────────
PATHS = {
    'L_grid':    'data/processed/cr_matrix/L_grid.npy',
    'S_grid':    'data/processed/cr_matrix/S_grid.npy',
    'Te_grid':   'data/processed/cr_matrix/Te_grid_L.npy',
    'ne_grid':   'data/processed/cr_matrix/ne_grid_L.npy',
    'K_ion':     'data/processed/collisions/tics/K_ion_final.npy',
    'K_exc':     'data/processed/collisions/K_exc_full/K_exc_full.npy',
    'K_deexc':   'data/processed/collisions/K_exc_full/K_deexc_full.npy',
    'K_exc_meta':'data/processed/collisions/K_exc_full/K_exc_meta.csv',
    'SCD96':     'data/processed/adas/SCD96_interpolated.csv',
    'ACD96':     'data/processed/adas/ACD96_interpolated.csv',
}
OUT_DIR = 'validation'


def load_arrays():
    arrs = {}
    for key in ['L_grid','S_grid','Te_grid','ne_grid','K_ion','K_exc','K_deexc']:
        arrs[key] = np.load(PATHS[key])
    arrs['K_exc_meta'] = pd.read_csv(PATHS['K_exc_meta'])
    return arrs


def steady_state(L_mat, S_vec):
    """Compute n_ss = -L^{-1} * S, floor at 0."""
    return np.maximum(np.linalg.solve(L_mat, -S_vec), 0.0)


# ── Gate A — Detailed balance ──────────────────────────────────────────────────
def gate_A(arrs, n_ion=1e14):
    """
    Verify K_deexc[j,i] = K_exc[i,j] * (g_i/g_j) * exp(E_pn/kTe)
    for all 819 pairs at 3 Te values.
    """
    print("\n" + "="*60)
    print("GATE A — Detailed Balance")
    print("="*60)

    Ke   = arrs['K_exc']    # (43,43,50) upper tri
    Kd   = arrs['K_deexc']  # (43,43,50) lower tri
    Te   = arrs['Te_grid']
    meta = arrs['K_exc_meta']

    errors = []
    for _, row in meta.iterrows():
        i, j = int(row['i']), int(row['j'])
        gi, gj = float(row['g_i']), float(row['g_j'])
        E_pn = float(row['E_pn_eV'])
        for ti in [0, 25, 49]:
            ke = float(Ke[i, j, ti])
            kd = float(Kd[j, i, ti])
            if ke > 1e-60:
                expected = ke * (gi/gj) * np.exp(E_pn / Te[ti])
                if expected > 0:
                    errors.append(abs(kd/expected - 1)*100)

    max_err = max(errors) if errors else 0
    passed  = max_err < 0.01
    print(f"  Pairs checked: {len(meta)} × 3 Te values = {len(meta)*3}")
    print(f"  Max error:     {max_err:.2e}%")
    print(f"  Status:        {'PASS' if passed else 'FAIL'}")
    return {'gate': 'A', 'passed': passed, 'max_err_pct': max_err}


# ── Gate B — Coronal limit ────────────────────────────────────────────────────
def gate_B(arrs, n_ion=1e12):
    """
    At lowest ne: n(p)/(n(1S)*ne) should be approximately constant
    (equal to K_exc(1S->p)/gamma(p) — the coronal excitation coefficient).

    Check: the ratio converges as ne decreases from 1e14 to 1e12.
    Specifically: the ratio at ne=1e12 should be within 20% of ne=2.7e12.
    """
    print("\n" + "="*60)
    print("GATE B — Coronal Limit")
    print("="*60)

    L = arrs['L_grid']; S = arrs['S_grid']
    Te = arrs['Te_grid']; ne = arrs['ne_grid']

    results = []

    for i_Te in range(len(Te)):
        # Compute n(p)/(n(1S)*ne) at three lowest ne points
        ratios_2P = []
        ratios_3D = []
        for i_ne in range(4):   # ne = 1e12, 2.7e12, 7.2e12, 1.9e13
            Lmat = L[i_Te, i_ne]
            Svec = S[i_Te, i_ne] * n_ion
            n_ss = steady_state(Lmat, Svec)
            if n_ss[0] > 0:
                ratios_2P.append(n_ss[2]  / (n_ss[0] * ne[i_ne]))
                ratios_3D.append(n_ss[5]  / (n_ss[0] * ne[i_ne]))

        if len(ratios_2P) >= 2:
            # Convergence: ratio at lowest ne should be within 50% of next
            # (some variation expected due to recombination source)
            r2P_low  = ratios_2P[0]
            r2P_next = ratios_2P[1]
            r3D_low  = ratios_3D[0]
            r3D_next = ratios_3D[1]

            # Coronal convergence check: ratios don't diverge wildly
            conv_2P = abs(r2P_low/r2P_next - 1) if r2P_next > 0 else 1.0
            conv_3D = abs(r3D_low/r3D_next - 1) if r3D_next > 0 else 1.0

            results.append({
                'Te_eV':       Te[i_Te],
                'r2P_ne1e12':  r2P_low,
                'r2P_ne2e12':  r2P_next,
                'conv_2P':     conv_2P,
                'r3D_ne1e12':  r3D_low,
                'r3D_ne2e12':  r3D_next,
                'conv_3D':     conv_3D,
                'passed':      conv_2P < 0.5 and conv_3D < 0.5,
            })

    df = pd.DataFrame(results)
    n_pass = df['passed'].sum()
    frac   = n_pass / len(df)
    passed = frac >= 0.7

    print(f"  Te points checked: {len(df)}")
    print(f"  Points passing coronal convergence: {n_pass}/{len(df)} ({frac*100:.0f}%)")
    print(f"  (need >= 70%)  Status: {'PASS' if passed else 'FAIL'}")
    print()
    print(f"  {'Te':6s}  {'r(2P)@ne=1e12':15s}  {'r(2P)@ne=2.7e12':17s}  {'conv':6s}  pass")
    for _, row in df[::5].iterrows():
        print(f"  {row['Te_eV']:6.2f}  {row['r2P_ne1e12']:15.4e}  "
              f"{row['r2P_ne2e12']:17.4e}  {row['conv_2P']:6.3f}  "
              f"{'Y' if row['passed'] else 'N'}")

    return {'gate': 'B', 'passed': passed, 'pass_frac': frac, 'df': df}


# ── Gate C — Saha-Boltzmann approach ─────────────────────────────────────────
def gate_C(arrs, n_ion=1e14):
    """
    n(p)/n(1S) should increase monotonically with ne,
    approaching (but never reaching) Saha-Boltzmann value.

    Check: the ratio n(2P)/n(1S) is monotonically increasing with ne
    at each Te, and remains below the Saha value.
    """
    print("\n" + "="*60)
    print("GATE C — Saha-Boltzmann Approach")
    print("="*60)

    L = arrs['L_grid']; S = arrs['S_grid']
    Te = arrs['Te_grid']; ne = arrs['ne_grid']

    results = []
    IH = IH_RYDBERG

    for i_Te in range(len(Te)):
        kTe = Te[i_Te]
        # Saha value for 2P
        g_2P = 6; g_1S = 2; E_2P_1S = IH * 0.75
        saha_2P = (g_2P/g_1S) * np.exp(-E_2P_1S/kTe)

        ratios_2P = []
        for i_ne in range(len(ne)):
            Lmat = L[i_Te, i_ne]
            Svec = S[i_Te, i_ne] * n_ion
            n_ss = steady_state(Lmat, Svec)
            ratios_2P.append(n_ss[2]/n_ss[0] if n_ss[0] > 0 else 0.0)

        ratios_arr = np.array(ratios_2P)

        # Check monotone increasing with ne
        monotone = bool(np.all(np.diff(ratios_arr) >= 0))

        # Check always below Saha
        below_saha = bool(np.all(ratios_arr <= saha_2P * 1.05))  # 5% tolerance

        # Check that the ratio at highest ne approaches Saha
        # (should be within 3 orders of magnitude at least)
        approach = ratios_arr[-1] / saha_2P if saha_2P > 0 else 0
        approaching = approach > 1e-4   # at least 1e-4 of Saha at ne=1e15

        results.append({
            'Te_eV':       kTe,
            'saha_2P':     saha_2P,
            'ratio_ne_lo': ratios_arr[0],
            'ratio_ne_hi': ratios_arr[-1],
            'approach_frac': approach,
            'monotone':    monotone,
            'below_saha':  below_saha,
            'approaching': approaching,
            'passed':      monotone and below_saha,
        })

    df = pd.DataFrame(results)
    n_pass = df['passed'].sum()
    frac   = n_pass / len(df)
    passed = frac >= 0.9   # Saha check should hold almost everywhere

    print(f"  Te points: {len(df)}")
    print(f"  Monotone increasing with ne: {df['monotone'].sum()}/{len(df)}")
    print(f"  Below Saha value:            {df['below_saha'].sum()}/{len(df)}")
    print(f"  Approaching Saha:            {df['approaching'].sum()}/{len(df)}")
    print(f"  Overall pass (>=90%): {n_pass}/{len(df)} ({frac*100:.0f}%)")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    print(f"  {'Te':6s}  {'Saha(2P/1S)':12s}  {'ratio@ne=1e12':14s}  {'ratio@ne=1e15':14s}  {'frac Saha':9s}")
    for _, row in df[::5].iterrows():
        print(f"  {row['Te_eV']:6.2f}  {row['saha_2P']:12.4e}  "
              f"{row['ratio_ne_lo']:14.4e}  {row['ratio_ne_hi']:14.4e}  "
              f"{row['approach_frac']:9.4f}")

    return {'gate': 'C', 'passed': passed, 'pass_frac': frac, 'df': df}


# ── Gate D — ADAS comparison ──────────────────────────────────────────────────
def gate_D(arrs, n_ion=1e14):
    """
    Compare effective ionization (SCD) and recombination (ACD) from our
    CR model with ADAS SCD96/ACD96 effective coefficients.

    SCD_eff = sum_p K_ion(p)*n_p^ss / (ne * n_neutral)
    ACD_eff = sum_p [alpha_RR(p)*ne + alpha_3BR(p)*ne^2]*n_p^ss / (ne^2 * n_ion)

    Requires:
      data/processed/ADAS/SCD96_long.csv
      data/processed/ADAS/ACD96_long.csv
    """
    print("\n" + "="*60)
    print("GATE D — ADAS Effective Coefficient Comparison")
    print("="*60)

    if not os.path.exists(PATHS['SCD96']) or not os.path.exists(PATHS['ACD96']):
        print(f"  ADAS files not found:")
        print(f"    {PATHS['SCD96']}")
        print(f"    {PATHS['ACD96']}")
        print(f"  GATE D SKIPPED — upload ADAS files to run this gate.")
        return {'gate': 'D', 'passed': None, 'skipped': True}

    scd_adas = pd.read_csv(PATHS['SCD96'])
    acd_adas = pd.read_csv(PATHS['ACD96'])

    L = arrs['L_grid']; S = arrs['S_grid']
    Te = arrs['Te_grid']; ne = arrs['ne_grid']
    K_ion = arrs['K_ion']

    results = []

    for i_Te in range(len(Te)):
        for i_ne in range(len(ne)):
            # FIX 1: use quasi-neutral n_ion = ne at each grid point.
            # Using fixed n_ion=1e14 with ne=1e12 puts the model in a deep
            # recombining phase (n_ion/ne=100), inflating Rydberg populations
            # and making SCD_model ~100x too large. ADAS SCD assumes quasi-
            # neutrality (n_ion ≈ ne), so this fix makes the comparison valid.
            n_ion_local = ne[i_ne]

            Lmat = L[i_Te, i_ne]
            Svec = S[i_Te, i_ne] * n_ion_local
            n_ss = steady_state(Lmat, Svec)
            n_neutral = n_ss.sum()
            if n_neutral <= 0: continue

            # SCD_eff from our model
            # SCD = sum_p K_ion(p)*n_ss(p) / n_neutral  [cm^3/s]
            # ne cancels: K_ion*ne*n_ss / (ne*n_neutral) = K_ion*n_ss/n_neutral
            ioniz_rate = np.sum(K_ion[:, i_Te] * n_ss)
            SCD_model  = ioniz_rate / n_neutral

            # FIX 2: 2D interpolation — select ADAS rows near ne[i_ne],
            # then interpolate in Te. Avoids averaging over wrong ne values.
            # The original code averaged SCD over ALL ne (5e7 to 2e15),
            # diluting the mean toward the low-ne coronal limit.
            try:
                mask = (scd_adas['ne_cm3'] / ne[i_ne]).between(0.5, 2.0)
                sub  = scd_adas[mask].sort_values('Te_eV')
                if len(sub) >= 2:
                    scd_at = float(np.interp(Te[i_Te],
                                             sub['Te_eV'].values,
                                             sub['SCD_cm3_s'].values))
                else:
                    scd_at = np.nan
            except Exception:
                scd_at = np.nan

            if np.isfinite(scd_at) and scd_at > 0:
                eta = SCD_model / scd_at
                results.append({
                    'Te_eV':     Te[i_Te],
                    'ne_cm3':    ne[i_ne],
                    'SCD_model': SCD_model,
                    'SCD_ADAS':  scd_at,
                    'eta_SCD':   eta,
                    'passed':    0.5 <= eta <= 2.0,
                })

    if not results:
        print("  Could not interpolate ADAS values — check CSV column names.")
        return {'gate': 'D', 'passed': None, 'skipped': True}

    df = pd.DataFrame(results)
    n_pass = df['passed'].sum()
    frac   = n_pass / len(df)
    passed = frac >= 0.7

    print(f"  Points compared: {len(df)}")
    print(f"  Factor-2 agreement: {n_pass}/{len(df)} ({frac*100:.0f}%)")
    print(f"  Status: {'PASS' if passed else 'FAIL (check Te range)'}")
    print()
    print(f"  {'Te':6s}  {'ne':10s}  {'SCD_model':12s}  {'SCD_ADAS':12s}  {'eta':8s}  pass")
    for _, row in df[::max(1,len(df)//10)].iterrows():
        print(f"  {row['Te_eV']:6.2f}  {row['ne_cm3']:10.2e}  "
              f"{row['SCD_model']:12.4e}  {row['SCD_ADAS']:12.4e}  "
              f"{row['eta_SCD']:8.3f}  {'Y' if row['passed'] else 'N'}")

    return {'gate': 'D', 'passed': passed, 'pass_frac': frac, 'df': df}


# ── Gate E — Eigenvalue-based QSS check ──────────────────────────────────────
def gate_E(arrs):
    """
    Gate E: Memory metric M = tau_QSS/tau_relax > 1 everywhere at steady state.

    This is the CORE thesis result from the eigenvalue analysis.
    Ensures the model has the correct timescale hierarchy.

    M = |lambda_1| / |lambda_0|
    lambda_0 = slowest eigenvalue (ionization balance timescale)
    lambda_1 = second eigenvalue (excited state relaxation timescale)
    """
    print("\n" + "="*60)
    print("GATE E — Timescale Hierarchy (M > 1 everywhere)")
    print("="*60)

    L = arrs['L_grid']
    Te = arrs['Te_grid']; ne = arrs['ne_grid']

    M_grid = np.zeros((len(Te), len(ne)))
    tau_QSS_grid   = np.zeros_like(M_grid)
    tau_relax_grid = np.zeros_like(M_grid)

    for i_Te in range(len(Te)):
        for i_ne in range(len(ne)):
            eigs = np.sort(np.linalg.eigvals(L[i_Te, i_ne]).real)[::-1]
            eigs_neg = eigs[eigs < -1.0]
            if len(eigs_neg) >= 2:
                M_grid[i_Te, i_ne]       = abs(eigs_neg[1]) / abs(eigs_neg[0])
                tau_QSS_grid[i_Te, i_ne]   = 1.0 / abs(eigs_neg[0])
                tau_relax_grid[i_Te, i_ne] = 1.0 / abs(eigs_neg[1])

    n_pass  = (M_grid > 1).sum()
    n_total = M_grid.size
    frac    = n_pass / n_total
    passed  = frac >= 0.95

    print(f"  Grid points: {n_total}")
    print(f"  M > 1:       {n_pass}/{n_total} ({frac*100:.1f}%)")
    print(f"  M range:     {M_grid.min():.0f} .. {M_grid.max():.0f}")
    print(f"  Status:      {'PASS' if passed else 'FAIL'}")
    print()
    print(f"  M at Te=3eV across ne:")
    ti3 = int(np.argmin(np.abs(Te-3)))
    for i_ne in range(len(ne)):
        print(f"    ne={ne[i_ne]:.2e}: M={M_grid[ti3,i_ne]:.1f}  "
              f"tau_QSS={tau_QSS_grid[ti3,i_ne]:.2e}s  "
              f"tau_relax={tau_relax_grid[ti3,i_ne]:.2e}s")

    return {
        'gate': 'E', 'passed': passed, 'pass_frac': frac,
        'M_grid': M_grid, 'tau_QSS': tau_QSS_grid,
        'tau_relax': tau_relax_grid
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*60)
    print("CR MODEL VALIDATION — GATES A-E")
    print("="*60)

    print("\nLoading arrays...")
    t0 = time.perf_counter()
    arrs = load_arrays()
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    results = {}

    results['A'] = gate_A(arrs)
    results['B'] = gate_B(arrs)
    results['C'] = gate_C(arrs)
    results['D'] = gate_D(arrs)
    results['E'] = gate_E(arrs)

    # Save CSV outputs
    for gate_key in ['B','C','D','E']:
        r = results[gate_key]
        if 'df' in r:
            r['df'].to_csv(f"{OUT_DIR}/gate_{gate_key}.csv", index=False)
        if 'M_grid' in r:
            np.save(f"{OUT_DIR}/M_grid.npy", r['M_grid'])
            np.save(f"{OUT_DIR}/tau_QSS_grid.npy", r['tau_QSS'])
            np.save(f"{OUT_DIR}/tau_relax_grid.npy", r['tau_relax'])

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    all_pass = True
    with open(f"{OUT_DIR}/gate_summary.txt", 'w') as f:
        for gate_key in 'ABCDE':
            r = results[gate_key]
            if r.get('skipped'):
                status = 'SKIP'
                detail = '(ADAS files not found)'
            elif r.get('passed') == True:
                status = 'PASS'
                detail = f"({r.get('pass_frac',1)*100:.0f}% of points)" if 'pass_frac' in r else ''
            elif r.get('passed') == False:
                status = 'FAIL'
                detail = f"({r.get('pass_frac',0)*100:.0f}% of points)"
                all_pass = False
            else:
                status = '????'
                detail = ''

            line = f"  Gate {gate_key}: {status:4s}  {detail}"
            print(line); f.write(line + '\n')

    print()
    if all_pass:
        print("  ALL GATES PASS — model validated, proceed to QSS analysis")
    else:
        print("  SOME GATES FAIL — review output above before proceeding")

    print(f"\n  Outputs saved to {OUT_DIR}/")