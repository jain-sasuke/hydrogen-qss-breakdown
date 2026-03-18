"""
qss_analysis.py
===============
QSS breakdown analysis — Scenario C.

Computes:
  1. QSS error epsilon(t) after Te step — two-stage relaxation
  2. Breakdown map: epsilon at ITER timescales over (Te, ne) grid

PHYSICS — CORRECT QSS ERROR DEFINITION
---------------------------------------
The QSS approximation says excited states are slaved to ground state:
    n_p(t) / n_1S(t) = r_p(Te(t), ne(t))   [QSS ratio]

QSS error (ratio metric, Fujimoto & McWhirter 1990):
    epsilon_p(t) = |n_p(t)/n_1S(t)  -  r_p(Te,ne)| / r_p(Te,ne)

TWO-STAGE RELAXATION after a Te step:
  Stage 1 [fast, t ~ tau_relax ~ ns]:
    Excited states (2P, 3D, ...) relax to new QSS RATIO.
    epsilon drops from ~0.3 to ~0.18.

  Stage 2 [slow, t ~ tau_QSS ~ 10-100 us]:
    Ground state (1S) relaxes to new density.
    epsilon decays from ~0.18 to 0 as n_1S converges.

QSS IS VALID when tau_drive >> tau_QSS (both stages complete).
QSS BREAKS when tau_drive < tau_QSS (ground state hasn't caught up).
QSS IS MARGINAL when tau_drive ~ tau_QSS.

SCENARIO A: epsilon(t) traces after Te step of +0.6 eV at t=0.
  Shown for 3 (Te, ne) reference points.

SCENARIO B: Breakdown map.
  For each (Te, ne), compute:
    eps_max_step = initial epsilon after 0.6 eV step = |dr/r|
    eps_ELM_crash = eps_max_step * min(1, tau_ELM/tau_QSS)
    eps_slow = eps_max_step * min(1, tau_slow/tau_QSS)
  Breakdown if eps > 0.5 (50% QSS error).

OUTPUTS (validation/)
----------------------
  M_grid.npy              (50, 8)  memory metric
  tau_QSS_grid.npy        (50, 8)  [s]
  tau_relax_grid.npy      (50, 8)  [s]
  epsilon_traces.npz      epsilon(t) for 3 reference points
  breakdown_map.csv       eps at ITER timescales for all (Te, ne)
  qss_analysis_summary.txt
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import os, time

OUT_DIR = 'validation'
PATHS = {
    'L_grid':  'data/processed/cr_matrix/L_grid.npy',
    'S_grid':  'data/processed/cr_matrix/S_grid.npy',
    'Te_grid': 'data/processed/cr_matrix/Te_grid_L.npy',
    'ne_grid': 'data/processed/cr_matrix/ne_grid_L.npy',
}

# ITER timescales [s]
ITER_TIMESCALES = {
    'ELM_crash':         1e-4,   # 100 μs  — ELM thermal quench
    'fast_detachment':   1e-3,   # 1 ms    — fast detachment
    'slow_detachment':   1e-2,   # 10 ms   — slow detachment/ELMy H-mode
    'ELM_interELM':      1e-1,   # 100 ms  — inter-ELM period
}

# Te step size for scenarios (realistic ELM perturbation)
DTE_STEP = 0.6   # eV — step of +0.6 eV (20% at 3 eV ref point)

# Reference points for epsilon(t) traces
TRACE_POINTS = [
    (1.0,  1e12, 'Te=1eV,ne=1e12'),
    (3.0,  1e14, 'Te=3eV,ne=1e14 (ITER)'),
    (10.0, 1e15, 'Te=10eV,ne=1e15'),
]


def load_arrays():
    return {k: np.load(p) for k, p in PATHS.items()}


def build_interpolators(arrs):
    Te, ne = arrs['Te_grid'], arrs['ne_grid']
    Lf = arrs['L_grid'].reshape(len(Te), len(ne), 43*43)
    L_i = RegularGridInterpolator(
        (np.log(Te), np.log(ne)), Lf, method='linear',
        bounds_error=False, fill_value=None)
    S_i = RegularGridInterpolator(
        (np.log(Te), np.log(ne)), arrs['S_grid'], method='linear',
        bounds_error=False, fill_value=None)
    return L_i, S_i


def get_ss(L_i, S_i, Te_v, ne_v, n_ion):
    pt = np.array([[np.log(Te_v), np.log(ne_v)]])
    Lm = L_i(pt)[0].reshape(43, 43)
    Sv = S_i(pt)[0] * n_ion
    return np.maximum(np.linalg.solve(Lm, -Sv), 0.0)


def compute_timescale_map(arrs):
    L = arrs['L_grid']
    Te, ne = arrs['Te_grid'], arrs['ne_grid']
    n_Te, n_ne = len(Te), len(ne)
    tau_QSS = np.zeros((n_Te, n_ne))
    tau_relax = np.zeros((n_Te, n_ne))
    M = np.zeros((n_Te, n_ne))
    print("Computing timescale map...")
    t0 = time.perf_counter()
    for i in range(n_Te):
        for j in range(n_ne):
            eigs = np.sort(np.linalg.eigvals(L[i, j]).real)[::-1]
            neg  = eigs[eigs < -1.0]
            if len(neg) >= 2:
                tau_QSS[i, j]   = 1/abs(neg[0])
                tau_relax[i, j] = 1/abs(neg[1])
                M[i, j]         = abs(neg[1])/abs(neg[0])
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    return tau_QSS, tau_relax, M


def epsilon_after_step(L_i, S_i, Te_v, ne_v, dTe, n_ion, t_eval):
    """
    Compute QSS ratio-error epsilon(t) after a Te step of +dTe at t=0.

    Initial condition: n_SS(Te_v, ne_v)
    Dynamics: L(Te_v+dTe, ne_v) — constant after step
    Reference: r_QSS(Te_v+dTe, ne_v) = n_SS(Te_new)[1:] / n_SS(Te_new)[0]

    Returns
    -------
    eps_max : (n_t,) max ratio-error over all excited states
    n_t     : (43, n_t) populations
    """
    dTe_actual = dTe if (Te_v + dTe) <= Te[-1] else -dTe   # step down at grid edge
    Te_new = float(Te_v + dTe_actual)
    n0 = get_ss(L_i, S_i, Te_v, ne_v, n_ion)
    n_ss_new = get_ss(L_i, S_i, Te_new, ne_v, n_ion)
    r_qss = n_ss_new[1:] / max(n_ss_new[0], 1e-60)   # (42,) QSS ratios after step

    pt_new = np.array([[np.log(Te_new), np.log(ne_v)]])
    Lm = L_i(pt_new)[0].reshape(43, 43)
    Sv = S_i(pt_new)[0] * n_ion

    def rhs(t, n):
        return Lm @ n + Sv

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), n0,
                    method='Radau', t_eval=t_eval,
                    rtol=1e-8, atol=1e-12, dense_output=False)

    eps_arr = np.zeros((42, len(sol.t)))
    for k in range(len(sol.t)):
        n = sol.y[:, k]
        r_act = n[1:] / max(n[0], 1e-60)
        eps_arr[:, k] = np.abs(r_act - r_qss) / (r_qss + 1e-60)

    return eps_arr.max(axis=0), sol.y


# ── Scenario A: epsilon(t) traces ─────────────────────────────────────────────
def scenario_A(arrs, tau_QSS_map, tau_relax_map, L_i, S_i, n_ion=1e14):
    """
    Compute epsilon(t) after +0.6 eV Te step at 3 reference (Te, ne) points.
    Shows two-stage relaxation: fast (tau_relax) then slow (tau_QSS).
    """
    print("\nSCENARIO A — epsilon(t) after Te step (+0.6 eV)")
    Te, ne = arrs['Te_grid'], arrs['ne_grid']
    results = {}

    for Te_v, ne_v, label in TRACE_POINTS:
        ti = np.argmin(np.abs(Te-Te_v)); nj = np.argmin(np.abs(ne-ne_v))
        tQ = tau_QSS_map[ti, nj]; tr = tau_relax_map[ti, nj]

        # Log-spaced time from 1ns to 100*tau_QSS
        t_eval = np.logspace(np.log10(tr*0.01), np.log10(100*tQ), 200)

        t0c = time.perf_counter()
        eps, n_t = epsilon_after_step(L_i, S_i, Te_v, ne_v, DTE_STEP,
                                      n_ion, t_eval)
        dt = time.perf_counter() - t0c

        # Characterise two stages
        # Stage 1: t < 5*tau_relax — fast excited-state equilibration
        mask1 = t_eval < 5*tr
        eps1_final = eps[mask1][-1] if mask1.any() else eps[0]
        # Stage 2: t > 5*tau_relax — slow ground-state equilibration
        mask2 = t_eval > 5*tr
        eps2_final = eps[mask2][-1] if mask2.any() else eps[-1]

        print(f"\n  [{label}]  tau_relax={tr:.2e}s  tau_QSS={tQ:.2e}s  M={tQ/tr:.0f}")
        print(f"    Solved in {dt:.1f}s")
        print(f"    Stage 1 (excited relax): eps {eps[0]:.3f} -> {eps1_final:.3f}  "
              f"(t~{5*tr:.1e}s = 5*tau_relax)")
        print(f"    Stage 2 (ground relax):  eps {eps1_final:.3f} -> {eps2_final:.4f}  "
              f"(t~{100*tQ:.1e}s = 100*tau_QSS)")

        results[label] = {
            't': t_eval, 'eps_max': eps, 'n_t': n_t,
            'tau_QSS': tQ, 'tau_relax': tr,
            'Te_v': Te_v, 'ne_v': ne_v,
        }

    return results


# ── Scenario B: breakdown map ─────────────────────────────────────────────────
def scenario_B(arrs, tau_QSS_map, tau_relax_map, L_i, S_i, n_ion=1e14):
    """
    For each (Te, ne): compute eps_step = initial QSS error after +0.6 eV step,
    then scale by min(1, tau_drive/tau_QSS) to get effective epsilon at each
    ITER timescale.

    eps_effective(tau_drive) = eps_step * min(1, tau_drive/tau_QSS)
    Breakdown: eps_effective > 0.5
    """
    print("\nSCENARIO B — QSS breakdown map")
    Te, ne = arrs['Te_grid'], arrs['ne_grid']

    rows = []
    print("  Computing eps_step for all grid points...")
    for i, Te_v in enumerate(Te):
        for j, ne_v in enumerate(ne):
            tQ = tau_QSS_map[i, j]; tr = tau_relax_map[i, j]

            # Initial epsilon = |r_QSS_new - r_QSS_old| / r_QSS_new
            # Compute analytically (no time integration needed)
            dTe_b = DTE_STEP if (Te_v + DTE_STEP) <= Te[-1] else -DTE_STEP
            Te_new = float(Te_v + dTe_b)
            n_ss0 = get_ss(L_i, S_i, Te_v, ne_v, n_ion)
            n_ss1 = get_ss(L_i, S_i, Te_new, ne_v, n_ion)

            r0 = n_ss0[1:] / max(n_ss0[0], 1e-60)
            r1 = n_ss1[1:] / max(n_ss1[0], 1e-60)
            eps_step = (np.abs(r0 - r1) / (r1 + 1e-60)).max()

            row = {
                'Te_eV': Te_v, 'ne_cm3': ne_v,
                'tau_QSS_s': tQ, 'tau_relax_s': tr,
                'M': tQ/max(tr,1e-30),
                'eps_step': eps_step,
            }
            for name, td in ITER_TIMESCALES.items():
                # Effective epsilon scales with min(tau_drive, tau_QSS)/tau_QSS
                eff = eps_step * min(1.0, td/tQ)
                row[f'eps_{name}'] = min(eff, 5.0)
            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"  eps_step range: {df['eps_step'].min():.3f} .. {df['eps_step'].max():.3f}")
    print()
    print(f"  QSS breakdown at ITER timescales (eps_eff > 0.5):")
    print(f"  {'Timescale':20s}  {'tau_drive':10s}  "
          f"{'% breakdown':12s}  {'ITER regime %':14s}  Context")
    print("  " + "-"*72)

    mask_ITER = (df['Te_eV'] <= 5) & (df['ne_cm3'] >= 1e13)
    for name, td in ITER_TIMESCALES.items():
        col = f'eps_{name}'
        frac     = (df[col] > 0.5).mean()*100
        frac_it  = (df.loc[mask_ITER, col] > 0.5).mean()*100
        ctx = {'ELM_crash':'ELM crash',
               'fast_detachment':'fast detach',
               'slow_detachment':'slow detach',
               'ELM_interELM':'inter-ELM'}[name]
        print(f"  {name:20s}  {td:10.1e}  {frac:12.1f}%  {frac_it:14.1f}%  {ctx}")

    print()
    print("  tau_QSS summary (determines breakdown boundary):")
    print(f"  {'Te':6s}  " + "".join(f"  ne={ne[j]:.0e}" for j in range(len(ne))))
    for i in [0,12,25,37,49]:
        row_str = "".join(f"  {tau_QSS_map[i,j]:.1e}" for j in range(len(ne)))
        print(f"  {Te[i]:6.2f}  {row_str}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print("="*65); print("QSS BREAKDOWN ANALYSIS"); print("="*65)

    print("\nLoading arrays...")
    arrs = load_arrays()
    L_i, S_i = build_interpolators(arrs)
    Te, ne = arrs['Te_grid'], arrs['ne_grid']

    tau_QSS, tau_relax, M = compute_timescale_map(arrs)
    print(f"\ntau_QSS:   {tau_QSS.min():.2e} .. {tau_QSS.max():.2e} s")
    print(f"tau_relax: {tau_relax.min():.2e} .. {tau_relax.max():.2e} s")
    print(f"M:         {M.min():.0f} .. {M.max():.0f}")

    traces = scenario_A(arrs, tau_QSS, tau_relax, L_i, S_i)
    bd_df  = scenario_B(arrs, tau_QSS, tau_relax, L_i, S_i)

    # Save
    np.save(f'{OUT_DIR}/M_grid.npy',         M)
    np.save(f'{OUT_DIR}/tau_QSS_grid.npy',   tau_QSS)
    np.save(f'{OUT_DIR}/tau_relax_grid.npy', tau_relax)

    td = {}
    for lbl, r in traces.items():
        key = lbl.replace('=','').replace(',','_').replace(' ','_')
        td[f't_{key}']   = r['t']
        td[f'eps_{key}'] = r['eps_max']
        td[f'tQ_{key}']  = np.array([r['tau_QSS']])
        td[f'tr_{key}']  = np.array([r['tau_relax']])
    np.savez(f'{OUT_DIR}/epsilon_traces.npz', **td)
    bd_df.to_csv(f'{OUT_DIR}/breakdown_map.csv', index=False)

    # Summary
    ti_r=np.argmin(np.abs(Te-3.0)); nj_r=np.argmin(np.abs(ne-1e14))
    tQ_r=tau_QSS[ti_r,nj_r]
    with open(f'{OUT_DIR}/qss_analysis_summary.txt','w') as f:
        f.write("QSS ANALYSIS SUMMARY\n"+"="*50+"\n\n")
        f.write("PHYSICS:\n")
        f.write("  QSS metric: epsilon = max_p |r_actual(t) - r_QSS(Te,ne)| / r_QSS\n")
        f.write("  where r_p = n_p/n_1S  (excited-to-ground ratio)\n")
        f.write("  Two-stage relaxation after Te step:\n")
        f.write("    Stage 1 (t~tau_relax): excited states equilibrate -> eps drops fast\n")
        f.write("    Stage 2 (t~tau_QSS):  ground state equilibrates  -> eps -> 0\n\n")
        f.write(f"tau_QSS:   {tau_QSS.min():.2e}..{tau_QSS.max():.2e} s\n")
        f.write(f"tau_relax: {tau_relax.min():.2e}..{tau_relax.max():.2e} s\n")
        f.write(f"M:         {M.min():.0f}..{M.max():.0f}\n\n")
        f.write(f"ITER divertor (Te=3eV, ne=1e14):\n")
        f.write(f"  tau_QSS = {tQ_r:.3e}s\n")
        for name,td_v in ITER_TIMESCALES.items():
            eps = min(bd_df.loc[(bd_df.Te_eV.round(2)==3.0)&
                                (bd_df.ne_cm3==1e14),f'eps_{name}'].values[0],5.0) \
                  if len(bd_df.loc[(bd_df.Te_eV.round(2)==3.0)&
                                   (bd_df.ne_cm3==1e14)]) > 0 else 0
            st = 'BREAKDOWN' if eps>0.5 else ('marginal' if eps>0.1 else 'valid')
            f.write(f"  {name:20s} ({td_v:.0e}s): eps~{eps:.3f} [{st}]\n")
        f.write("\nScenario A epsilon(t) traces:\n")
        for lbl,r in traces.items():
            f.write(f"  {lbl}: tau_QSS={r['tau_QSS']:.2e}s  tau_relax={r['tau_relax']:.2e}s\n")

    print(f"\nSaved to {OUT_DIR}/")
    print("="*65); print("QSS ANALYSIS COMPLETE"); print("="*65)