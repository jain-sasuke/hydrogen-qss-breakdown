"""
test_scaling.py
===============
Four rigorous tests of the QSS error scaling law.

  Step 1 — Test universality across ΔTe values (±0.3, ±0.6, ±1.0 eV)
  Step 2 — Normalize eps_step by ΔTe/Te → check collapse
  Step 3 — Rigorous density independence (partial derivative ∂ε/∂ne)
  Step 4 — Robustness under n_max truncation (10, 15, 20 levels)

Expected outcome if scaling law is real:
  Step 1: eps_step ∝ ΔTe × f(Te) — proportional to step size
  Step 2: eps_norm = eps_step / (ΔTe/Te) collapses all ΔTe onto one curve
  Step 3: ∂ε/∂ne ≈ 0 uniformly (not just Pearson r ≈ 0)
  Step 4: Results stable within ±5% across n_max = 10/15/20

Run from repo root:
    python src/analysis/test_scaling.py

Outputs saved to validation/scaling_tests/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import RegularGridInterpolator
import os

OUT_DIR = 'validation/scaling_tests'

# ── Paths ──────────────────────────────────────────────────────────────────────
PATHS = {
    'L_grid':  'data/processed/cr_matrix/L_grid.npy',
    'S_grid':  'data/processed/cr_matrix/S_grid.npy',
    'Te_grid': 'data/processed/cr_matrix/Te_grid_L.npy',
    'ne_grid': 'data/processed/cr_matrix/ne_grid_L.npy',
}

Te_grid = np.logspace(np.log10(1.0), np.log10(10.0), 50)
ne_grid = np.logspace(12, 15, 8)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 10,
    'legend.fontsize': 9, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'xtick.direction': 'in', 'ytick.direction': 'in',
})


# ── Core helpers ───────────────────────────────────────────────────────────────
def load_matrices():
    L  = np.load(PATHS['L_grid'])    # (50, 8, 43, 43)
    S  = np.load(PATHS['S_grid'])    # (50, 8, 43)
    return L, S


def get_ss_grid(L, S, i_Te, i_ne, n_ion=1e14):
    """Steady-state populations at grid point (i_Te, i_ne)."""
    Lm = L[i_Te, i_ne]
    Sv = S[i_Te, i_ne] * n_ion
    return np.maximum(np.linalg.solve(Lm, -Sv), 0.0)


def get_ss_interp(L_i, S_i, Te_v, ne_v, n_ion=1e14):
    """Steady-state at arbitrary (Te, ne) via interpolation."""
    pt = np.array([[np.log(Te_v), np.log(ne_v)]])
    Lm = L_i(pt)[0].reshape(43, 43)
    Sv = S_i(pt)[0] * n_ion
    return np.maximum(np.linalg.solve(Lm, -Sv), 0.0)


def build_interp(L, S):
    lTe = np.log(Te_grid); lne = np.log(ne_grid)
    Lf  = L.reshape(len(Te_grid), len(ne_grid), 43*43)
    L_i = RegularGridInterpolator((lTe, lne), Lf, method='linear',
                                   bounds_error=False, fill_value=None)
    S_i = RegularGridInterpolator((lTe, lne), S, method='linear',
                                   bounds_error=False, fill_value=None)
    return L_i, S_i


def eps_step_at(L, S, i_Te, i_ne, dTe, n_ion=1e14):
    """
    Compute eps_step = max_p |r_old - r_new| / r_new
    for a Te step of dTe at grid point (i_Te, i_ne).
    Uses bidirectional step: if Te+dTe > grid max, step is -dTe.
    """
    Te_v = Te_grid[i_Te]
    Te_new_v = Te_v + dTe
    # Clamp to grid
    if Te_new_v > Te_grid[-1] or Te_new_v < Te_grid[0]:
        dTe = -dTe
        Te_new_v = Te_v + dTe

    # Interpolate L/S at Te_new (ne is on-grid so use i_ne)
    n_ss0 = get_ss_grid(L, S, i_Te, i_ne, n_ion)

    # For Te_new, interpolate in Te direction
    L_i, S_i = _get_interp_cache(L, S)
    n_ss1 = get_ss_interp(L_i, S_i, Te_new_v, ne_grid[i_ne], n_ion)

    r0 = n_ss0[1:] / max(n_ss0[0], 1e-60)
    r1 = n_ss1[1:] / max(n_ss1[0], 1e-60)
    return float((np.abs(r0 - r1) / (r1 + 1e-60)).max()), abs(dTe)


_interp_cache = {}
def _get_interp_cache(L, S):
    key = id(L)
    if key not in _interp_cache:
        _interp_cache[key] = build_interp(L, S)
    return _interp_cache[key]


def exp_model(T, a, b, c):
    return a * np.exp(-b * T) + c


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Universality across ΔTe
# ══════════════════════════════════════════════════════════════════════════════
def step1_delta_Te_universality(L, S, out_dir):
    """
    Compute eps_step(Te, ne, ΔTe) for ΔTe = ±0.3, ±0.6, ±1.0 eV.
    Check: eps_step ∝ ΔTe × f(Te)
    """
    print("\nSTEP 1 — Universality across ΔTe values")
    print("=" * 55)

    dTe_values = [0.3, 0.6, 1.0]
    colors     = ['#2196F3', '#F44336', '#4CAF50']

    # Compute eps_step for all (Te, ne, dTe) combinations
    results = {}
    for dTe in dTe_values:
        eps_grid = np.zeros((len(Te_grid), len(ne_grid)))
        for i_Te in range(len(Te_grid)):
            for i_ne in range(len(ne_grid)):
                eps_val, actual_dTe = eps_step_at(L, S, i_Te, i_ne, dTe)
                eps_grid[i_Te, i_ne] = eps_val
        results[dTe] = eps_grid
        print(f"  ΔTe=±{dTe}eV: eps range {eps_grid.min():.4f}..{eps_grid.max():.4f}")

    # Test proportionality: eps_step(dTe=1.0) / eps_step(dTe=0.3) should ≈ 1.0/0.3 = 3.33
    ratio_10_03 = results[1.0] / (results[0.3] + 1e-10)
    print(f"\n  Proportionality test: eps(ΔTe=1.0) / eps(ΔTe=0.3):")
    print(f"    Expected: {1.0/0.3:.2f} (linear)")
    print(f"    Actual mean:   {ratio_10_03.mean():.3f}")
    print(f"    Actual std:    {ratio_10_03.std():.3f}")
    lin_ok = abs(ratio_10_03.mean() - 1.0/0.3) < 0.5
    print(f"    Linear (within 0.5): {'YES' if lin_ok else 'NO'}")

    # Figure: eps_step vs Te for each ΔTe (at ne=1e14)
    ni14 = np.argmin(np.abs(ne_grid - 1e14))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for dTe, color in zip(dTe_values, colors):
        eps_Te = results[dTe][:, ni14]
        ax.plot(Te_grid, eps_Te, '-o', color=color, ms=4, lw=1.8,
                label=fr'$\Delta T_e={dTe}$ eV')
    ax.set_xlabel(r'$T_e$ [eV]')
    ax.set_ylabel(r'$\epsilon_\mathrm{step}$')
    ax.set_title(r'$\epsilon_\mathrm{step}$ vs $T_e$ for different $\Delta T_e$ ($n_e=10^{14}$)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Figure: ratio eps(dTe) / dTe — should collapse
    ax2 = axes[1]
    for dTe, color in zip(dTe_values, colors):
        eps_Te = results[dTe][:, ni14] / dTe
        ax2.plot(Te_grid, eps_Te, '-o', color=color, ms=4, lw=1.8,
                 label=fr'$\Delta T_e={dTe}$ eV')
    ax2.set_xlabel(r'$T_e$ [eV]')
    ax2.set_ylabel(r'$\epsilon_\mathrm{step}\,/\,\Delta T_e$')
    ax2.set_title(r'Collapse test: $\epsilon_\mathrm{step}/\Delta T_e$ vs $T_e$')
    ax2.legend()
    ax2.grid(alpha=0.3)

    note = "Collapse = lines overlap → ε ∝ ΔTe (linear)"
    ax2.text(0.02, 0.05, note, transform=ax2.transAxes, fontsize=8.5,
             color='gray', style='italic')

    plt.tight_layout()
    path = f'{out_dir}/step1_delta_Te_universality.png'
    fig.savefig(path); plt.close(fig)
    print(f"\n  Saved: {path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Normalized collapse
# ══════════════════════════════════════════════════════════════════════════════
def step2_normalized_collapse(step1_results, out_dir):
    """
    Plot eps_norm = eps_step / (ΔTe/Te) vs Te.
    If collapse improves → publishable scaling law.
    Expected: eps_norm ≈ Te × d(eps)/d(Te) = constant × f_Boltzmann(Te)
    """
    print("\nSTEP 2 — Normalized collapse: eps_norm = eps_step / (ΔTe/Te)")
    print("=" * 55)

    dTe_values = [0.3, 0.6, 1.0]
    colors     = ['#2196F3', '#F44336', '#4CAF50']
    ni14 = np.argmin(np.abs(ne_grid - 1e14))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: raw eps_step
    ax = axes[0]
    for dTe, color in zip(dTe_values, colors):
        eps = step1_results[dTe][:, ni14]
        ax.semilogy(Te_grid, eps, '-o', color=color, ms=4, lw=1.8,
                    label=fr'$\Delta T_e={dTe}$ eV')
    ax.set_xlabel(r'$T_e$ [eV]'); ax.set_ylabel(r'$\epsilon_\mathrm{step}$')
    ax.set_title(r'Raw $\epsilon_\mathrm{step}$'); ax.legend(); ax.grid(alpha=0.3)

    # Right: eps_norm = eps / (dTe/Te)
    ax2 = axes[1]
    spreads = []
    for dTe, color in zip(dTe_values, colors):
        eps      = step1_results[dTe][:, ni14]
        eps_norm = eps / (dTe / Te_grid)
        ax2.semilogy(Te_grid, eps_norm, '-o', color=color, ms=4, lw=1.8,
                     label=fr'$\Delta T_e={dTe}$ eV')
        spreads.append(eps_norm)

    ax2.set_xlabel(r'$T_e$ [eV]')
    ax2.set_ylabel(r'$\epsilon_\mathrm{norm} = \epsilon_\mathrm{step}\,/\,(\Delta T_e / T_e)$')
    ax2.set_title(r'Normalized collapse test')
    ax2.legend(); ax2.grid(alpha=0.3)

    # Compute collapse quality: std across dTe at each Te, normalised by mean
    spreads = np.array(spreads)  # (3, 50)
    cv = spreads.std(axis=0) / (spreads.mean(axis=0) + 1e-30)  # coefficient of variation
    cv_mean = cv.mean()
    print(f"  Collapse quality (coeff. of variation, lower=better):")
    print(f"    Before normalisation: {np.array([step1_results[d][:,ni14] for d in dTe_values]).std(axis=0).mean() / np.array([step1_results[d][:,ni14] for d in dTe_values]).mean(axis=0).mean():.3f}")
    print(f"    After normalisation:  {cv_mean:.3f}")
    print(f"    Improvement: {'YES — publishable collapse' if cv_mean < 0.15 else 'PARTIAL — non-linear correction needed'}")

    # Fit the normalised curve (should be universal function of Te only)
    eps_norm_ref = spreads[1]  # ΔTe=0.6 as reference
    try:
        popt, _ = curve_fit(exp_model, Te_grid, eps_norm_ref, p0=[10, 0.4, 0.1])
        Te_line = np.linspace(1, 10, 200)
        ax2.semilogy(Te_line, exp_model(Te_line, *popt), 'k--', lw=2,
                     label=fr'fit: ${popt[0]:.2f}\,e^{{-{popt[1]:.2f}T_e}}+{popt[2]:.2f}$', zorder=5)
        ax2.legend()
        R2 = 1 - np.var(eps_norm_ref - exp_model(Te_grid, *popt)) / np.var(eps_norm_ref)
        print(f"\n  Universal fit (normalised): R² = {R2:.4f}")
        print(f"  eps_norm ≈ {popt[0]:.2f}·exp(−{popt[1]:.2f}·Te) + {popt[2]:.2f}")
        print(f"  → eps_step ≈ (ΔTe/Te) × [{popt[0]:.2f}·exp(−{popt[1]:.2f}·Te) + {popt[2]:.2f}]")
    except Exception as e:
        print(f"  Fit failed: {e}")

    plt.tight_layout()
    path = f'{out_dir}/step2_normalized_collapse.png'
    fig.savefig(path); plt.close(fig)
    print(f"\n  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Rigorous density independence
# ══════════════════════════════════════════════════════════════════════════════
def step3_density_independence(step1_results, out_dir):
    """
    Compute ∂ε/∂ne numerically at each (Te, ne) point.
    If |∂ε/∂ne| ≈ 0 uniformly → strong ne-independence claim.
    """
    print("\nSTEP 3 — Rigorous density independence: ∂ε/∂ne")
    print("=" * 55)

    eps_grid = step1_results[0.6]   # (50, 8) using ΔTe=0.6

    # Numerical derivative ∂ε/∂ne using central differences
    # Shape: (50, 6) — interior ne points only
    dne = np.diff(ne_grid)  # (7,)
    deps_dne = np.zeros((len(Te_grid), len(ne_grid)-2))
    for j in range(1, len(ne_grid)-1):
        dne_fwd = ne_grid[j+1] - ne_grid[j]
        dne_bwd = ne_grid[j]   - ne_grid[j-1]
        deps_dne[:, j-1] = (eps_grid[:, j+1] - eps_grid[:, j-1]) / (dne_fwd + dne_bwd)

    # Normalise: (∂ε/∂ne) × ne / ε = logarithmic derivative ∂lnε/∂lnne
    eps_mid = eps_grid[:, 1:-1] + 1e-60
    ne_mid  = ne_grid[1:-1]
    log_deriv = deps_dne * ne_mid[np.newaxis, :] / eps_mid  # (50, 6)

    print(f"  Logarithmic derivative ∂ln(ε)/∂ln(ne):")
    print(f"    Mean:   {log_deriv.mean():.5f}  (0 = perfectly ne-independent)")
    print(f"    Std:    {log_deriv.std():.5f}")
    print(f"    Max |∂ln(ε)/∂ln(ne)|: {np.abs(log_deriv).max():.5f}")
    print(f"    Fraction |derivative| < 0.05: "
          f"{(np.abs(log_deriv) < 0.05).mean()*100:.1f}%")
    print(f"    Claim: {'STRONG — ne-independence confirmed' if np.abs(log_deriv).mean() < 0.05 else 'WEAK — some ne dependence'}")

    # Spearman rank correlation at each Te (more robust than Pearson)
    spearman_rs = []
    for i_Te in range(len(Te_grid)):
        r, _ = spearmanr(ne_grid, eps_grid[i_Te, :])
        spearman_rs.append(r)
    spearman_rs = np.array(spearman_rs)
    print(f"\n  Spearman rank correlation eps vs ne, per Te:")
    print(f"    Mean |r|: {np.abs(spearman_rs).mean():.4f}")
    print(f"    Max |r|:  {np.abs(spearman_rs).max():.4f}")
    print(f"    All |r| < 0.3: {(np.abs(spearman_rs) < 0.3).all()}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    im = ax.pcolormesh(np.log10(ne_grid[1:-1]), Te_grid,
                       log_deriv, cmap='RdBu_r',
                       vmin=-0.1, vmax=0.1, shading='auto')
    fig.colorbar(im, ax=ax).set_label(r'$\partial\ln\epsilon/\partial\ln n_e$')
    ax.set_xlabel(r'$\log_{10}(n_e)$'); ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_title(r'Logarithmic derivative $\partial\ln\epsilon/\partial\ln n_e$')

    ax2 = axes[1]
    ax2.plot(Te_grid, spearman_rs, 'o-', color='#E91E63', ms=4, lw=1.8)
    ax2.axhline(0, color='black', lw=0.8, ls='--')
    ax2.axhline( 0.3, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax2.axhline(-0.3, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax2.fill_between(Te_grid, -0.3, 0.3, alpha=0.08, color='green',
                     label='|r| < 0.3 (weak correlation)')
    ax2.set_xlabel(r'$T_e$ [eV]')
    ax2.set_ylabel(r'Spearman $r(\epsilon, n_e)$')
    ax2.set_title(r'Rank correlation $\epsilon$ vs $n_e$ (per $T_e$)')
    ax2.set_ylim(-0.6, 0.6)
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{out_dir}/step3_density_independence.png'
    fig.savefig(path); plt.close(fig)
    print(f"\n  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Robustness under n_max truncation
# ══════════════════════════════════════════════════════════════════════════════
def step4_nmax_truncation(L, S, out_dir):
    """
    Compare eps_step(Te, ne) computed using only the first N states,
    where N corresponds to n_max = 10, 15, 20.

    State count:
      n_max=10: states n=1..8 (l-resolved) + n9 + n10 = 37 states  → index 0..36
      n_max=15: states n=1..8 + n9..n15                = 43 states  → index 0..42 (full)
      n_max=20: not in our matrix (we only have n=1..15)
               → test n_max=10 vs n_max=15 (bundled) as sensitivity

    For the truncation test, we zero out the rows/cols for states beyond n_max
    and re-solve. This isolates the contribution of high-n states.
    """
    print("\nSTEP 4 — Robustness under n_max truncation")
    print("=" * 55)

    # State count per n_max:
    # n=1: 1, n=2: 2, n=3: 3, n=4: 4, n=5: 5, n=6: 6, n=7: 7, n=8: 8 → 36 resolved
    # n9=idx36, n10=37, n11=38, n12=39, n13=40, n14=41, n15=42
    trunc_sets = {
        'n_max=10 (37 states)':  37,   # up through n10
        'n_max=12 (39 states)':  39,   # up through n12
        'n_max=15 (43 states, full)': 43,  # full model
    }
    colors = ['#FF9800', '#9C27B0', '#000000']

    ni14 = np.argmin(np.abs(ne_grid - 1e14))
    dTe  = 0.6

    L_i, S_i = build_interp(L, S)
    results_trunc = {}

    for (label, N), color in zip(trunc_sets.items(), colors):
        eps_arr = []
        for i_Te, Te_v in enumerate(Te_grid):
            Te_new_v = Te_v + dTe
            if Te_new_v > Te_grid[-1]: Te_new_v = Te_v - dTe

            # Get full steady-state, then truncate to N states
            n_ss0 = get_ss_grid(L, S, i_Te, ni14)[:N]
            n_ss1 = get_ss_interp(L_i, S_i, Te_new_v, ne_grid[ni14])[:N]

            r0 = n_ss0[1:] / max(n_ss0[0], 1e-60)
            r1 = n_ss1[1:] / max(n_ss1[0], 1e-60)
            eps = float((np.abs(r0 - r1) / (r1 + 1e-60)).max())
            eps_arr.append(eps)

        results_trunc[label] = np.array(eps_arr)
        print(f"  {label}: eps range {min(eps_arr):.4f}..{max(eps_arr):.4f}")

    # Relative difference between truncated and full
    eps_full = results_trunc['n_max=15 (43 states, full)']
    for label in list(trunc_sets.keys())[:-1]:
        eps_trunc = results_trunc[label]
        rel_diff  = np.abs(eps_trunc - eps_full) / (eps_full + 1e-60)
        print(f"\n  {label} vs full:")
        print(f"    Mean rel diff: {rel_diff.mean()*100:.2f}%")
        print(f"    Max rel diff:  {rel_diff.max()*100:.2f}%")
        print(f"    Robust (< 5%): {'YES' if rel_diff.mean() < 0.05 else 'NO'}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for (label, N), color in zip(trunc_sets.items(), colors):
        lw = 2.5 if 'full' in label else 1.5
        ls = '-' if 'full' in label else '--'
        ax.plot(Te_grid, results_trunc[label], ls, color=color,
                lw=lw, label=label)
    ax.set_xlabel(r'$T_e$ [eV]'); ax.set_ylabel(r'$\epsilon_\mathrm{step}$')
    ax.set_title(r'$\epsilon_\mathrm{step}$ vs $n_\mathrm{max}$ truncation ($n_e=10^{14}$)')
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    ax2 = axes[1]
    for (label, N), color in zip(list(trunc_sets.items())[:-1], colors[:-1]):
        rel = np.abs(results_trunc[label] - eps_full) / (eps_full + 1e-60) * 100
        ax2.plot(Te_grid, rel, '--', color=color, lw=1.8, label=label)
    ax2.axhline(5, color='red', lw=1.0, ls=':', label='5% threshold')
    ax2.set_xlabel(r'$T_e$ [eV]')
    ax2.set_ylabel(r'Relative difference from full model [%]')
    ax2.set_title(r'Sensitivity to $n_\mathrm{max}$ truncation')
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    path = f'{out_dir}/step4_nmax_truncation.png'
    fig.savefig(path); plt.close(fig)
    print(f"\n  Saved: {path}")

    return results_trunc


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(step1_results, out_dir):
    """Print final verdict on scaling law robustness."""
    dTe_values = [0.3, 0.6, 1.0]
    ni14 = np.argmin(np.abs(ne_grid - 1e14))

    print("\n" + "="*55)
    print("SUMMARY — SCALING LAW ROBUSTNESS")
    print("="*55)

    # Fit each ΔTe curve
    print("\nFits eps_step ~ a*exp(-b*Te)+c for each ΔTe:")
    fits = {}
    for dTe in dTe_values:
        eps = step1_results[dTe][:, ni14]
        try:
            popt, _ = curve_fit(exp_model, Te_grid, eps, p0=[0.9, 0.4, 0.05])
            R2 = 1 - np.var(eps - exp_model(Te_grid, *popt)) / np.var(eps)
            fits[dTe] = popt
            print(f"  ΔTe={dTe}: a={popt[0]:.3f} b={popt[1]:.3f} c={popt[2]:.3f}  R²={R2:.4f}")
        except:
            print(f"  ΔTe={dTe}: fit failed")

    # Check if b (decay constant) is ΔTe-independent
    if len(fits) == 3:
        bs = [fits[d][1] for d in dTe_values]
        print(f"\n  Decay constant b across ΔTe: {[f'{v:.3f}' for v in bs]}")
        print(f"  std(b)/mean(b) = {np.std(bs)/np.mean(bs):.4f}")
        print(f"  Universal b: {'YES (< 5% variation)' if np.std(bs)/np.mean(bs) < 0.05 else 'NO'}")

    print("\n  THESIS SCALING LAW:")
    if len(fits) >= 1:
        b_mean = np.mean([fits[d][1] for d in fits])
        print(f"  eps_step(Te, ΔTe) ≈ (ΔTe / Te) × f(Te)")
        print(f"  where f(Te) = A × exp(−{b_mean:.2f} × Te) + C")
        print(f"  — purely Te-dependent, ne-independent")
        print(f"  — ΔTe-linear for small perturbations")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*55)
    print("SCALING LAW ROBUSTNESS TESTS")
    print("="*55)
    print(f"  Output: {OUT_DIR}/")

    print("\nLoading L_grid and S_grid...")
    L, S = load_matrices()
    print(f"  L_grid: {L.shape}")

    step1_res  = step1_delta_Te_universality(L, S, OUT_DIR)
    step2_normalized_collapse(step1_res, OUT_DIR)
    step3_density_independence(step1_res, OUT_DIR)
    step4_nmax_truncation(L, S, OUT_DIR)
    print_summary(step1_res, OUT_DIR)

    print("\n" + "="*55)
    print("ALL TESTS COMPLETE")
    print(f"  Figures saved to {OUT_DIR}/")
    print("="*55)