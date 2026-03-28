"""
test_scaling.py
===============
Thesis-safe diagnostics for the temperature-step QSS error metric

    eps_step(Te, ne; dTe) = max_p |r_old(p) - r_new(p)| / r_new(p),
    r_p = n_p / n_1S

This script does NOT claim a universal first-principles scaling law.
Instead, it evaluates four defensible diagnostics:

  Step 1 — Multi-ΔTe response:
      Compare eps_step(Te) for dTe = 0.3, 0.6, 1.0 eV.

  Step 2 — Small-step linear-response check:
      Test whether eps_step / dTe approximately collapses
      for small dTe, and quantify where nonlinearity appears.

  Step 3 — Density sensitivity map:
      Measure weak/strong dependence of eps_step on ne using
      logarithmic derivatives and rank correlations.

  Step 4 — Dominant-state robustness:
      Track which state dominates eps_step across Te, ne, dTe.
      This replaces the invalid "n_max truncation" test.

Outputs:
    validation/scaling_tests/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr

OUT_DIR = "validation/scaling_tests"

PATHS = {
    "L_grid":  "data/processed/cr_matrix/L_grid.npy",
    "S_grid":  "data/processed/cr_matrix/S_grid.npy",
    "Te_grid": "data/processed/cr_matrix/Te_grid_L.npy",
    "ne_grid": "data/processed/cr_matrix/ne_grid_L.npy",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# 43-state mapping
STATE_LABELS = [
    "1S",
    "2S", "2P",
    "3S", "3P", "3D",
    "4S", "4P", "4D", "4F",
    "5S", "5P", "5D", "5F", "5G",
    "6S", "6P", "6D", "6F", "6G", "6H",
    "7S", "7P", "7D", "7F", "7G", "7H", "7I",
    "8S", "8P", "8D", "8F", "8G", "8H", "8I", "8K",
    "n9", "n10", "n11", "n12", "n13", "n14", "n15"
]

N_STATES = 43


# -----------------------------------------------------------------------------
# I/O and interpolation
# -----------------------------------------------------------------------------
def load_matrices():
    L = np.load(PATHS["L_grid"])
    S = np.load(PATHS["S_grid"])
    Te_grid = np.load(PATHS["Te_grid"])
    ne_grid = np.load(PATHS["ne_grid"])
    return L, S, Te_grid, ne_grid


def build_interp(L, S, Te_grid, ne_grid):
    lTe = np.log(Te_grid)
    lne = np.log(ne_grid)

    L_flat = L.reshape(len(Te_grid), len(ne_grid), N_STATES * N_STATES)

    L_i = RegularGridInterpolator(
        (lTe, lne),
        L_flat,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    S_i = RegularGridInterpolator(
        (lTe, lne),
        S,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return L_i, S_i


_interp_cache = {}


def get_interp_cache(L, S, Te_grid, ne_grid):
    key = (id(L), id(S), id(Te_grid), id(ne_grid))
    if key not in _interp_cache:
        _interp_cache[key] = build_interp(L, S, Te_grid, ne_grid)
    return _interp_cache[key]


# -----------------------------------------------------------------------------
# Steady-state helpers
# -----------------------------------------------------------------------------
def solve_ss(Lm, Sv, clip_tol=1e-14):
    n = np.linalg.solve(Lm, -Sv)
    if np.min(n) < -clip_tol:
        print(f"  Warning: SS solve produced negative population min={np.min(n):.3e}; clipping to zero")
    return np.maximum(n, 0.0)


def get_ss_grid(L, S, i_Te, i_ne, n_ion):
    Lm = L[i_Te, i_ne]
    Sv = S[i_Te, i_ne] * n_ion
    return solve_ss(Lm, Sv)


def get_ss_interp(L_i, S_i, Te_v, ne_v, n_ion):
    pt = np.array([[np.log(Te_v), np.log(ne_v)]])
    Lm = L_i(pt)[0].reshape(N_STATES, N_STATES)
    Sv = S_i(pt)[0] * n_ion
    return solve_ss(Lm, Sv)


def choose_step(Te_v, dTe, Te_grid):
    """
    Bidirectional step:
      use +dTe if in range,
      otherwise use -dTe.
    Returns:
      Te_new, actual_dTe
    """
    Te_new = Te_v + dTe
    actual_dTe = dTe

    if Te_new > Te_grid[-1] or Te_new < Te_grid[0]:
        actual_dTe = -dTe
        Te_new = Te_v + actual_dTe

    return Te_new, actual_dTe


def eps_step_at(L, S, Te_grid, ne_grid, i_Te, i_ne, dTe, n_ion=None):
    """
    eps_step = max_p |r_old - r_new| / r_new
    using actual bidirectional step near grid edges.
    """
    if n_ion is None:
        # self-consistent hydrogenic default
        n_ion = ne_grid[i_ne]

    Te_v = Te_grid[i_Te]
    ne_v = ne_grid[i_ne]
    Te_new, actual_dTe = choose_step(Te_v, dTe, Te_grid)

    n0 = get_ss_grid(L, S, i_Te, i_ne, n_ion)

    L_i, S_i = get_interp_cache(L, S, Te_grid, ne_grid)
    n1 = get_ss_interp(L_i, S_i, Te_new, ne_v, n_ion)

    r0 = n0[1:] / max(n0[0], 1e-60)
    r1 = n1[1:] / max(n1[0], 1e-60)

    eps_all = np.abs(r0 - r1) / (r1 + 1e-60)
    eps = float(np.max(eps_all))
    dom_idx_42 = int(np.argmax(eps_all))
    dom_idx_43 = dom_idx_42 + 1  # because r excludes 1S
    dom_label = STATE_LABELS[dom_idx_43]

    return eps, abs(actual_dTe), dom_idx_43, dom_label


def exp_model(T, a, b, c):
    return a * np.exp(-b * T) + c


# -----------------------------------------------------------------------------
# Step 1 — Multi-dTe response
# -----------------------------------------------------------------------------
def step1_multi_dTe(L, S, Te_grid, ne_grid, out_dir):
    print("\nSTEP 1 — Multi-ΔTe response")
    print("=" * 60)

    dTe_values = [0.3, 0.6, 1.0]
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    eps_results = {}
    actual_dTe_results = {}

    for dTe in dTe_values:
        eps_grid = np.zeros((len(Te_grid), len(ne_grid)))
        act_grid = np.zeros((len(Te_grid), len(ne_grid)))

        for i_Te in range(len(Te_grid)):
            for i_ne in range(len(ne_grid)):
                eps, act, _, _ = eps_step_at(L, S, Te_grid, ne_grid, i_Te, i_ne, dTe)
                eps_grid[i_Te, i_ne] = eps
                act_grid[i_Te, i_ne] = act

        eps_results[dTe] = eps_grid
        actual_dTe_results[dTe] = act_grid

        print(f"  dTe={dTe:.1f} eV: eps range {eps_grid.min():.4f} .. {eps_grid.max():.4f}")

    ni14 = np.argmin(np.abs(ne_grid - 1e14))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for dTe, c in zip(dTe_values, colors):
        ax.plot(
            Te_grid,
            eps_results[dTe][:, ni14],
            "-o",
            color=c,
            lw=1.8,
            ms=4,
            label=fr"$\Delta T_e={dTe}$ eV"
        )
    ax.set_xlabel(r"$T_e$ [eV]")
    ax.set_ylabel(r"$\epsilon_{\rm step}$")
    ax.set_title(r"Raw $\epsilon_{\rm step}$ at $n_e=10^{14}$ cm$^{-3}$")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    for dTe, c in zip(dTe_values, colors):
        ax2.plot(
            Te_grid,
            eps_results[dTe][:, ni14] / dTe,
            "-o",
            color=c,
            lw=1.8,
            ms=4,
            label=fr"$\epsilon_{{\rm step}}/\Delta T_e$, $\Delta T_e={dTe}$ eV"
        )
    ax2.set_xlabel(r"$T_e$ [eV]")
    ax2.set_ylabel(r"$\epsilon_{\rm step}/\Delta T_e$ [eV$^{-1}$]")
    ax2.set_title(r"Linear-response diagnostic")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{out_dir}/step1_multi_dTe.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    return eps_results, actual_dTe_results


# -----------------------------------------------------------------------------
# Step 2 — Small-step linear-response collapse
# -----------------------------------------------------------------------------
def step2_small_step_check(eps_results, Te_grid, ne_grid, out_dir):
    print("\nSTEP 2 — Small-step linear-response check")
    print("=" * 60)

    ni14 = np.argmin(np.abs(ne_grid - 1e14))

    dTe_small = [0.3, 0.6]
    arr = np.array([eps_results[d][:, ni14] / d for d in dTe_small])  # (2, nTe)

    cv = arr.std(axis=0) / (arr.mean(axis=0) + 1e-60)

    # Conservative “small-step” regime criterion: dTe / Te <= 0.2 for dTe=0.6
    mask_small = (0.6 / Te_grid) <= 0.2
    cv_small = float(np.mean(cv[mask_small])) if np.any(mask_small) else np.nan
    cv_all = float(np.mean(cv))

    print(f"  Mean CV of eps/dTe over all Te:   {cv_all:.3f}")
    print(f"  Mean CV of eps/dTe for dTe/Te<=0.2: {cv_small:.3f}")
    print("  Interpretation:")
    print("    - low CV at high Te supports approximate linear response")
    print("    - poor collapse at low Te indicates nonlinearity / finite-step effects")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for dTe, color in zip([0.3, 0.6, 1.0], ["#2196F3", "#F44336", "#4CAF50"]):
        y = eps_results[dTe][:, ni14] / dTe
        ax.semilogy(Te_grid, y, "-o", color=color, lw=1.8, ms=4, label=fr"$\Delta T_e={dTe}$ eV")
    ax.axvline(3.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$T_e$ [eV]")
    ax.set_ylabel(r"$\epsilon_{\rm step}/\Delta T_e$ [eV$^{-1}$]")
    ax.set_title("Small-step collapse diagnostic")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(Te_grid, cv, "-o", color="#9C27B0", lw=1.8, ms=4)
    ax2.axhline(0.10, color="gray", lw=0.8, ls=":", label="CV = 0.10")
    ax2.axhline(0.15, color="gray", lw=0.8, ls="--", label="CV = 0.15")
    ax2.set_xlabel(r"$T_e$ [eV]")
    ax2.set_ylabel("Coefficient of variation")
    ax2.set_title(r"Collapse quality of $\epsilon_{\rm step}/\Delta T_e$")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{out_dir}/step2_small_step_check.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------------
# Step 3 — Density sensitivity
# -----------------------------------------------------------------------------
def step3_density_sensitivity(eps_results, Te_grid, ne_grid, out_dir):
    print("\nSTEP 3 — Density sensitivity")
    print("=" * 60)

    eps_grid = eps_results[0.6]  # shape (nTe, nne)

    # log-derivative d ln eps / d ln ne
    log_deriv = np.zeros((len(Te_grid), len(ne_grid) - 2))
    for j in range(1, len(ne_grid) - 1):
        dln_ne = np.log(ne_grid[j + 1]) - np.log(ne_grid[j - 1])
        log_deriv[:, j - 1] = (
            np.log(eps_grid[:, j + 1] + 1e-60) - np.log(eps_grid[:, j - 1] + 1e-60)
        ) / dln_ne

    abs_mean = float(np.mean(np.abs(log_deriv)))
    abs_max = float(np.max(np.abs(log_deriv)))

    spearman_rs = []
    for i_Te in range(len(Te_grid)):
        r, _ = spearmanr(np.log10(ne_grid), eps_grid[i_Te, :])
        spearman_rs.append(r)
    spearman_rs = np.array(spearman_rs)

    print(f"  Mean |d ln eps / d ln ne|: {abs_mean:.4f}")
    print(f"  Max  |d ln eps / d ln ne|: {abs_max:.4f}")
    print(f"  Mean |Spearman r| over Te: {np.mean(np.abs(spearman_rs)):.4f}")
    print("  Interpretation: this is a sensitivity map, not a claim of exact ne-independence.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    im = ax.pcolormesh(
        np.log10(ne_grid[1:-1]),
        Te_grid,
        log_deriv,
        cmap="RdBu_r",
        vmin=-0.2,
        vmax=0.2,
        shading="auto",
    )
    fig.colorbar(im, ax=ax).set_label(r"$\partial \ln \epsilon / \partial \ln n_e$")
    ax.set_xlabel(r"$\log_{10}(n_e)$")
    ax.set_ylabel(r"$T_e$ [eV]")
    ax.set_title(r"Density sensitivity of $\epsilon_{\rm step}$")

    ax2 = axes[1]
    ax2.plot(Te_grid, spearman_rs, "-o", color="#E91E63", lw=1.8, ms=4)
    ax2.axhline(0.0, color="black", lw=0.8, ls="--")
    ax2.axhline(0.3, color="gray", lw=0.8, ls=":")
    ax2.axhline(-0.3, color="gray", lw=0.8, ls=":")
    ax2.set_xlabel(r"$T_e$ [eV]")
    ax2.set_ylabel(r"Spearman $r(\epsilon, n_e)$")
    ax2.set_title(r"Rank correlation of $\epsilon_{\rm step}$ with $n_e$")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{out_dir}/step3_density_sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------------
# Step 4 — Dominant-state robustness
# -----------------------------------------------------------------------------
def step4_dominant_state(L, S, Te_grid, ne_grid, out_dir):
    print("\nSTEP 4 — Dominant-state robustness")
    print("=" * 60)

    dTe_values = [0.3, 0.6, 1.0]
    dominant_maps = {}

    for dTe in dTe_values:
        dom_idx = np.zeros((len(Te_grid), len(ne_grid)), dtype=int)

        for i_Te in range(len(Te_grid)):
            for i_ne in range(len(ne_grid)):
                _, _, idx43, _ = eps_step_at(L, S, Te_grid, ne_grid, i_Te, i_ne, dTe)
                dom_idx[i_Te, i_ne] = idx43

        dominant_maps[dTe] = dom_idx

        unique, counts = np.unique(dom_idx, return_counts=True)
        order = np.argsort(counts)[::-1]
        print(f"  dTe={dTe:.1f} eV dominant states:")
        for k in order[:5]:
            idx = unique[k]
            cnt = counts[k]
            frac = 100.0 * cnt / dom_idx.size
            print(f"    {STATE_LABELS[idx]:>4s}: {frac:5.1f}%")

    # Compare dominant-state stability between dTe=0.3 and dTe=0.6
    same_03_06 = np.mean(dominant_maps[0.3] == dominant_maps[0.6])
    same_06_10 = np.mean(dominant_maps[0.6] == dominant_maps[1.0])

    print(f"\n  Dominant-state agreement:")
    print(f"    0.3 vs 0.6 eV: {100*same_03_06:.1f}%")
    print(f"    0.6 vs 1.0 eV: {100*same_06_10:.1f}%")

    # Plot only dTe=0.6 map
    fig, ax = plt.subplots(figsize=(5.3, 4.2))

    im = ax.pcolormesh(
        np.log10(ne_grid),
        Te_grid,
        dominant_maps[0.6],
        shading="auto",
        cmap="tab20"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dominant state index")
    ax.set_xlabel(r"$\log_{10}(n_e)$")
    ax.set_ylabel(r"$T_e$ [eV]")
    ax.set_title(r"Dominant state controlling $\epsilon_{\rm step}$ ($\Delta T_e=0.6$ eV)")

    plt.tight_layout()
    path = f"{out_dir}/step4_dominant_state.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def print_summary(eps_results, Te_grid, ne_grid):
    print("\n" + "=" * 60)
    print("SUMMARY — THESIS-SAFE INTERPRETATION")
    print("=" * 60)

    ni14 = np.argmin(np.abs(ne_grid - 1e14))

    print("\n1. Multi-ΔTe response")
    try:
        for dTe in [0.3, 0.6, 1.0]:
            eps = eps_results[dTe][:, ni14]
            popt, _ = curve_fit(exp_model, Te_grid, eps, p0=[0.5, 0.4, 0.05], maxfev=10000)
            r2 = 1.0 - np.var(eps - exp_model(Te_grid, *popt)) / np.var(eps)
            print(f"   dTe={dTe:.1f} eV: empirical fit a*exp(-bTe)+c with b={popt[1]:.3f}, R^2={r2:.4f}")
    except Exception:
        print("   Empirical fits not robust enough to summarize globally.")

    print("\n2. Defensible claims")
    print("   - eps_step decreases strongly with Te.")
    print("   - eps_step depends only weakly on ne over much of the tested range, but not identically zero.")
    print("   - Approximate linear-response collapse is better at higher Te (small dTe/Te),")
    print("     and degrades at low Te where finite-step effects are strong.")
    print("   - The dominant state controlling eps_step can change with Te, ne, and dTe,")
    print("     so a single universal activation-energy interpretation is not justified.")

    print("\n3. Thesis-safe conclusion")
    print("   eps_step is a useful empirical QSS-jump metric, but the data support")
    print("   only an approximate small-step scaling at sufficiently high Te.")
    print("   The results do not justify a universal first-principles scaling law")
    print("   valid across all Te, ne, and dTe in the tested range.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("QSS ERROR SCALING DIAGNOSTICS")
    print("=" * 60)
    print(f"  Output directory: {OUT_DIR}/")

    print("\nLoading matrices and grids...")
    L, S, Te_grid, ne_grid = load_matrices()
    print(f"  L_grid shape:  {L.shape}")
    print(f"  S_grid shape:  {S.shape}")
    print(f"  Te_grid shape: {Te_grid.shape}")
    print(f"  ne_grid shape: {ne_grid.shape}")

    eps_results, _ = step1_multi_dTe(L, S, Te_grid, ne_grid, OUT_DIR)
    step2_small_step_check(eps_results, Te_grid, ne_grid, OUT_DIR)
    step3_density_sensitivity(eps_results, Te_grid, ne_grid, OUT_DIR)
    step4_dominant_state(L, S, Te_grid, ne_grid, OUT_DIR)
    print_summary(eps_results, Te_grid, ne_grid)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)