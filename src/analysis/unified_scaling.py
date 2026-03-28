# src/analysis/validate_unified_scaling.py
"""
Validate a reduced QSS-breakdown model against full transient CR truth.

What this script does
---------------------
1. Loads the precomputed 43-state CR matrices L(Te, ne) and source S(Te, ne)
2. Computes old/new steady states for a Te step
3. Runs the full time-dependent CR transient after the step
4. Computes H-alpha-specific QSS error:
      - instantaneous
      - end-of-event residual
      - event-averaged error
5. Builds a reduced model:
      eps_model_end = eps_res_Ha * exp(-tau_drive / tau_QSS)
      eps_model_avg = eps_res_Ha * (tau_QSS / tau_drive) * (1 - exp(-tau_drive / tau_QSS))
6. Compares reduced model vs transient truth over (Te, ne, tau_drive)

This is thesis-safe because it validates the reduced model against the
actual full CR transient instead of just asserting a heuristic scaling law.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
OUT_DIR = "validation/unified_scaling_validation"

PATHS = {
    "L_grid":  "data/processed/cr_matrix/L_grid.npy",
    "S_grid":  "data/processed/cr_matrix/S_grid.npy",
    "Te_grid": "data/processed/cr_matrix/Te_grid_L.npy",
    "ne_grid": "data/processed/cr_matrix/ne_grid_L.npy",
}

N_STATES = 43

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

# E1 Balmer-alpha channels only
A_3S_2P = 6.317e6
A_3P_2S = 2.245e7
A_3D_2P = 6.465e7

IDX_1S = 0
IDX_3S = 3
IDX_3P = 4
IDX_3D = 5

# ITER-relevant drive timescales [s]
TAU_DRIVES = {
    "ELM crash (100 us)": 100e-6,
    "Fast detachment (1 ms)": 1e-3,
    "Slow detachment (10 ms)": 10e-3,
    "Inter-ELM (100 ms)": 100e-3,
}

# For Te step
DTE_DEFAULT = 0.6  # eV

# Time grid controls
N_T_EVAL = 500
FAST_RES_FACTOR = 100.0  # eps_res measured at t = FAST_RES_FACTOR * tau_relax


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "xtick.direction": "in",
    "ytick.direction": "in",
})


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class LocalStepCase:
    Te_old: float
    Te_new: float
    ne: float
    n_ion: float
    L_new: np.ndarray
    S_new: np.ndarray
    n_ss_old: np.ndarray
    n_ss_new: np.ndarray
    delta_ss: np.ndarray
    tau_relax: float
    tau_qss: float


@dataclass
class ObservableErrors:
    eps_step_Ha: float
    eps_res_Ha: float
    eps_end_true: float
    eps_avg_true: float
    eps_end_model: float
    eps_avg_model: float
    peak_err_Ha: float
    t_peak: float
    t_10pct: float | None


# -----------------------------------------------------------------------------
# Loading / interpolation
# -----------------------------------------------------------------------------
def load_arrays():
    L = np.load(PATHS["L_grid"])
    S = np.load(PATHS["S_grid"])
    Te_grid = np.load(PATHS["Te_grid"])
    ne_grid = np.load(PATHS["ne_grid"])
    return L, S, Te_grid, ne_grid


def build_interp(L: np.ndarray, S: np.ndarray, Te_grid: np.ndarray, ne_grid: np.ndarray):
    lTe = np.log(Te_grid)
    lne = np.log(ne_grid)

    L_i = RegularGridInterpolator(
        (lTe, lne),
        L.reshape(len(Te_grid), len(ne_grid), N_STATES * N_STATES),
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


# -----------------------------------------------------------------------------
# CR helpers
# -----------------------------------------------------------------------------
def clip_nonnegative(x: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    if np.min(x) < -tol:
        print(f"Warning: negative population min={np.min(x):.3e}; clipping to zero.")
    return np.maximum(x, 0.0)


def solve_ss(Lm: np.ndarray, Sv: np.ndarray, n_ion: float) -> np.ndarray:
    # 0 = L n + S n_ion
    rhs = -Sv * n_ion
    n = np.linalg.solve(Lm, rhs)
    return clip_nonnegative(n)


def get_LS_interp(
    L_i,
    S_i,
    Te: float,
    ne: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pt = np.array([[np.log(Te), np.log(ne)]])
    Lm = L_i(pt)[0].reshape(N_STATES, N_STATES)
    Sv = S_i(pt)[0]
    return Lm, Sv


def choose_step(Te_old: float, dTe: float, Te_grid: np.ndarray) -> Tuple[float, float]:
    Te_new = Te_old + dTe
    actual_dTe = dTe
    if Te_new < Te_grid[0] or Te_new > Te_grid[-1]:
        actual_dTe = -dTe
        Te_new = Te_old + actual_dTe
    return Te_new, actual_dTe


def emissivity_Ha(n: np.ndarray) -> float:
    return (
        A_3S_2P * n[IDX_3S] +
        A_3P_2S * n[IDX_3P] +
        A_3D_2P * n[IDX_3D]
    )


def ratio_Ha_from_state(n: np.ndarray) -> float:
    n1 = max(n[IDX_1S], 1e-60)
    return emissivity_Ha(n) / n1


def slow_timescales(Lm: np.ndarray) -> Tuple[float, float]:
    """
    Returns:
        tau_relax ~ fast excited-state adjustment timescale
        tau_qss   ~ slowest non-zero decay timescale
    """
    eig = np.linalg.eigvals(Lm)
    re = np.real(eig)

    # stable modes only
    neg = re[re < -1e-16]
    if len(neg) == 0:
        raise RuntimeError("No decaying modes found in L matrix.")

    # Fast timescale: most negative mode magnitude
    lam_fast = np.min(neg)  # most negative real part
    tau_relax = 1.0 / abs(lam_fast)

    # Slow timescale: least negative non-zero mode
    lam_slow = np.max(neg)  # closest to zero, still negative
    tau_qss = 1.0 / abs(lam_slow)

    return tau_relax, tau_qss


def build_case(
    L_i,
    S_i,
    Te_grid: np.ndarray,
    Te_old: float,
    ne: float,
    dTe: float = DTE_DEFAULT,
    n_ion: float | None = None,
) -> LocalStepCase:
    if n_ion is None:
        n_ion = ne

    Te_new, _ = choose_step(Te_old, dTe, Te_grid)

    L_old, S_old = get_LS_interp(L_i, S_i, Te_old, ne)
    L_new, S_new = get_LS_interp(L_i, S_i, Te_new, ne)

    n_ss_old = solve_ss(L_old, S_old, n_ion)
    n_ss_new = solve_ss(L_new, S_new, n_ion)

    delta_ss = n_ss_old - n_ss_new
    tau_relax, tau_qss = slow_timescales(L_new)

    return LocalStepCase(
        Te_old=Te_old,
        Te_new=Te_new,
        ne=ne,
        n_ion=n_ion,
        L_new=L_new,
        S_new=S_new,
        n_ss_old=n_ss_old,
        n_ss_new=n_ss_new,
        delta_ss=delta_ss,
        tau_relax=tau_relax,
        tau_qss=tau_qss,
    )


# -----------------------------------------------------------------------------
# Transient truth
# -----------------------------------------------------------------------------
def rhs_linear(t, y, Lm, Sv, n_ion):
    return Lm @ y + Sv * n_ion


def solve_transient(case: LocalStepCase, t_end: float):
    # Start from old SS, evolve with new L,S
    y0 = case.n_ss_old.copy()

    # Dense enough to resolve both fast and slow parts
    t_min = max(case.tau_relax / 50.0, 1e-11)
    t_eval = np.geomspace(t_min, t_end, N_T_EVAL)
    t_eval = np.insert(t_eval, 0, 0.0)

    sol = solve_ivp(
        rhs_linear,
        t_span=(0.0, t_end),
        y0=y0,
        method="Radau",
        t_eval=t_eval,
        args=(case.L_new, case.S_new, case.n_ion),
        rtol=1e-8,
        atol=1e-12,
        jac=case.L_new,
    )
    if not sol.success:
        raise RuntimeError(f"Transient solve failed: {sol.message}")

    Y = clip_nonnegative(sol.y)
    return sol.t, Y


def qss_prediction_from_actual_n1(case: LocalStepCase, Y: np.ndarray) -> np.ndarray:
    """
    QSS prediction that uses actual CR n_1S(t) but freezes excited-state ratios
    at the new local steady-state values.
    This isolates excited-state QSS error only.
    """
    r_new = case.n_ss_new / max(case.n_ss_new[IDX_1S], 1e-60)
    Y_qss = np.zeros_like(Y)
    Y_qss[IDX_1S, :] = Y[IDX_1S, :]
    for k in range(Y.shape[1]):
        Y_qss[:, k] = r_new * Y[IDX_1S, k]
    return Y_qss


def analyze_Ha_errors(case: LocalStepCase, tau_drive: float) -> ObservableErrors:
    t_end = max(tau_drive, 10.0 * case.tau_qss)
    t, Y = solve_transient(case, t_end=t_end)
    Y_qss = qss_prediction_from_actual_n1(case, Y)

    I_true = np.array([emissivity_Ha(Y[:, k]) for k in range(Y.shape[1])])
    I_qss = np.array([emissivity_Ha(Y_qss[:, k]) for k in range(Y_qss.shape[1])])

    rel_err = (I_true - I_qss) / np.maximum(I_qss, 1e-60)
    abs_err = np.abs(rel_err)

    # Initial step error at t = 0+
    eps_step_Ha = float(abs_err[0])

    # Stage-1 residual after fast modes have died
    t_res = FAST_RES_FACTOR * case.tau_relax
    idx_res = int(np.argmin(np.abs(t - t_res)))
    eps_res_Ha = float(abs_err[idx_res])

    # End-of-event residual
    idx_drive = int(np.argmin(np.abs(t - tau_drive)))
    eps_end_true = float(abs_err[idx_drive])

    # Event-averaged error over [0, tau_drive]
    mask = t <= tau_drive
    if np.sum(mask) < 2:
        eps_avg_true = eps_end_true
    else:
        eps_avg_true = float(np.trapz(abs_err[mask], t[mask]) / tau_drive)

    # Reduced-model predictions
    eps_end_model = float(eps_res_Ha * np.exp(-tau_drive / case.tau_qss))
    eps_avg_model = float(
        eps_res_Ha * (case.tau_qss / tau_drive) * (1.0 - np.exp(-tau_drive / case.tau_qss))
    )

    peak_idx = int(np.argmax(abs_err))
    peak_err_Ha = float(abs_err[peak_idx])
    t_peak = float(t[peak_idx])

    # first time after which error stays below 10%
    t_10pct = None
    below = abs_err <= 0.10
    for i in range(len(t)):
        if below[i] and np.all(below[i:]):
            t_10pct = float(t[i])
            break

    return ObservableErrors(
        eps_step_Ha=eps_step_Ha,
        eps_res_Ha=eps_res_Ha,
        eps_end_true=eps_end_true,
        eps_avg_true=eps_avg_true,
        eps_end_model=eps_end_model,
        eps_avg_model=eps_avg_model,
        peak_err_Ha=peak_err_Ha,
        t_peak=t_peak,
        t_10pct=t_10pct,
    )


# -----------------------------------------------------------------------------
# Grid validation
# -----------------------------------------------------------------------------
def validate_reduced_model(
    L_i,
    S_i,
    Te_grid: np.ndarray,
    ne_grid: np.ndarray,
    dTe: float,
    tau_drives: Dict[str, float],
):
    results = {}

    for label, tau_drive in tau_drives.items():
        end_true = np.zeros((len(Te_grid), len(ne_grid)))
        end_model = np.zeros_like(end_true)
        avg_true = np.zeros_like(end_true)
        avg_model = np.zeros_like(end_true)
        tau_qss_map = np.zeros_like(end_true)
        tau_relax_map = np.zeros_like(end_true)
        eps_res_map = np.zeros_like(end_true)

        print(f"\nRunning validation for {label} ...")
        for i, Te in enumerate(Te_grid):
            for j, ne in enumerate(ne_grid):
                case = build_case(L_i, S_i, Te_grid, Te_old=Te, ne=ne, dTe=dTe, n_ion=ne)
                obs = analyze_Ha_errors(case, tau_drive=tau_drive)

                end_true[i, j] = obs.eps_end_true
                end_model[i, j] = obs.eps_end_model
                avg_true[i, j] = obs.eps_avg_true
                avg_model[i, j] = obs.eps_avg_model
                tau_qss_map[i, j] = case.tau_qss
                tau_relax_map[i, j] = case.tau_relax
                eps_res_map[i, j] = obs.eps_res_Ha

        results[label] = {
            "tau_drive": tau_drive,
            "end_true": end_true,
            "end_model": end_model,
            "avg_true": avg_true,
            "avg_model": avg_model,
            "tau_qss": tau_qss_map,
            "tau_relax": tau_relax_map,
            "eps_res": eps_res_map,
        }

    return results


# -----------------------------------------------------------------------------
# Metrics / plots
# -----------------------------------------------------------------------------
def error_stats(y_true: np.ndarray, y_pred: np.ndarray):
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / np.maximum(y_true, 1e-60)
    return {
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "mean_rel": float(np.mean(rel_err)),
        "max_rel": float(np.max(rel_err)),
    }


def plot_heatmap(Z, Te_grid, ne_grid, title, cbar_label, path, cmap="viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    im = ax.pcolormesh(np.log10(ne_grid), Te_grid, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    ax.set_xlabel(r"$\log_{10}(n_e\, [\mathrm{cm}^{-3}])$")
    ax.set_ylabel(r"$T_e$ [eV]")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_plots(results, Te_grid, ne_grid):
    os.makedirs(OUT_DIR, exist_ok=True)

    for label, d in results.items():
        safe = (
            label.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        # truth maps
        plot_heatmap(
            d["avg_true"], Te_grid, ne_grid,
            title=f"Hα average QSS bias (truth)\n{label}",
            cbar_label=r"$\bar{\epsilon}_{H\alpha,\mathrm{true}}$",
            path=f"{OUT_DIR}/{safe}_avg_true.png",
        )
        plot_heatmap(
            d["end_true"], Te_grid, ne_grid,
            title=f"Hα end-of-event QSS bias (truth)\n{label}",
            cbar_label=r"$\epsilon_{H\alpha,\mathrm{end,true}}$",
            path=f"{OUT_DIR}/{safe}_end_true.png",
        )

        # reduced model predictions
        plot_heatmap(
            d["avg_model"], Te_grid, ne_grid,
            title=f"Hα average QSS bias (reduced model)\n{label}",
            cbar_label=r"$\bar{\epsilon}_{H\alpha,\mathrm{model}}$",
            path=f"{OUT_DIR}/{safe}_avg_model.png",
        )
        plot_heatmap(
            d["end_model"], Te_grid, ne_grid,
            title=f"Hα end-of-event QSS bias (reduced model)\n{label}",
            cbar_label=r"$\epsilon_{H\alpha,\mathrm{end,model}}$",
            path=f"{OUT_DIR}/{safe}_end_model.png",
        )

        # model error maps
        plot_heatmap(
            np.abs(d["avg_model"] - d["avg_true"]), Te_grid, ne_grid,
            title=f"Reduced-model abs. error in Hα average bias\n{label}",
            cbar_label=r"$|\bar{\epsilon}_{\mathrm{model}}-\bar{\epsilon}_{\mathrm{true}}|$",
            path=f"{OUT_DIR}/{safe}_avg_model_abs_error.png",
            cmap="magma",
        )
        plot_heatmap(
            np.abs(d["end_model"] - d["end_true"]), Te_grid, ne_grid,
            title=f"Reduced-model abs. error in Hα end bias\n{label}",
            cbar_label=r"$|\epsilon_{\mathrm{end,model}}-\epsilon_{\mathrm{end,true}}|$",
            path=f"{OUT_DIR}/{safe}_end_model_abs_error.png",
            cmap="magma",
        )


def print_summary(results):
    print("\n" + "=" * 72)
    print("REDUCED-MODEL VALIDATION SUMMARY")
    print("=" * 72)

    for label, d in results.items():
        s_end = error_stats(d["end_true"], d["end_model"])
        s_avg = error_stats(d["avg_true"], d["avg_model"])

        print(f"\n{label}")
        print("-" * len(label))
        print("End-of-event residual model:")
        print(f"  MAE      = {s_end['mae']:.4e}")
        print(f"  Max abs  = {s_end['max_abs']:.4e}")
        print(f"  Mean rel = {100*s_end['mean_rel']:.2f}%")
        print(f"  Max rel  = {100*s_end['max_rel']:.2f}%")

        print("Event-averaged model:")
        print(f"  MAE      = {s_avg['mae']:.4e}")
        print(f"  Max abs  = {s_avg['max_abs']:.4e}")
        print(f"  Mean rel = {100*s_avg['mean_rel']:.2f}%")
        print(f"  Max rel  = {100*s_avg['max_rel']:.2f}%")

    print("\nInterpretation:")
    print("  - The reduced model is only credible where its error versus full CR truth is small.")
    print("  - Use the Hα average-bias map as the primary spectroscopic metric.")
    print("  - Use the end-of-event map as a secondary residual-relaxation metric.")


# -----------------------------------------------------------------------------
# Example point plot
# -----------------------------------------------------------------------------
def plot_example_transient(
    L_i,
    S_i,
    Te_grid,
    Te_old=3.0,
    ne=1e14,
    dTe=DTE_DEFAULT,
    tau_drive=100e-6,
):
    case = build_case(L_i, S_i, Te_grid, Te_old=Te_old, ne=ne, dTe=dTe, n_ion=ne)
    t_end = max(tau_drive, 10.0 * case.tau_qss)
    t, Y = solve_transient(case, t_end=t_end)
    Y_qss = qss_prediction_from_actual_n1(case, Y)

    I_true = np.array([emissivity_Ha(Y[:, k]) for k in range(Y.shape[1])])
    I_qss = np.array([emissivity_Ha(Y_qss[:, k]) for k in range(Y_qss.shape[1])])
    abs_err = np.abs((I_true - I_qss) / np.maximum(I_qss, 1e-60))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.loglog(t + 1e-30, I_true, lw=2, label="Full CR")
    ax.loglog(t + 1e-30, I_qss, "--", lw=2, label="QSS using actual n(1S)")
    ax.axvline(case.tau_relax, color="gray", ls=":", lw=1, label=r"$\tau_{\rm relax}$")
    ax.axvline(case.tau_qss, color="black", ls="--", lw=1, label=r"$\tau_{\rm QSS}$")
    ax.axvline(tau_drive, color="tab:red", ls="-.", lw=1, label=r"$\tau_{\rm drive}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$I_{H\alpha}$ [arb.]")
    ax.set_title(fr"Hα transient at $T_e: {case.Te_old:.1f}\to {case.Te_new:.1f}$ eV, $n_e={ne:.1e}$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogx(t + 1e-30, 100 * abs_err, lw=2)
    ax.axhline(10.0, color="gray", ls=":")
    ax.axvline(case.tau_relax, color="gray", ls=":", lw=1)
    ax.axvline(case.tau_qss, color="black", ls="--", lw=1)
    ax.axvline(tau_drive, color="tab:red", ls="-.", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\epsilon_{H\alpha}(t)$ [%]")
    ax.set_title("Hα excited-state QSS error")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/example_transient_Ha.png")
    plt.close(fig)

    obs = analyze_Ha_errors(case, tau_drive=tau_drive)
    print("\nExample point:")
    print(f"  Te step       : {case.Te_old:.2f} -> {case.Te_new:.2f} eV")
    print(f"  ne            : {case.ne:.3e} cm^-3")
    print(f"  tau_relax     : {case.tau_relax:.3e} s")
    print(f"  tau_QSS       : {case.tau_qss:.3e} s")
    print(f"  eps_step_Ha   : {100*obs.eps_step_Ha:.2f}%")
    print(f"  eps_res_Ha    : {100*obs.eps_res_Ha:.2f}%")
    print(f"  eps_end_true  : {100*obs.eps_end_true:.2f}%")
    print(f"  eps_end_model : {100*obs.eps_end_model:.2f}%")
    print(f"  eps_avg_true  : {100*obs.eps_avg_true:.2f}%")
    print(f"  eps_avg_model : {100*obs.eps_avg_model:.2f}%")
    print(f"  peak error    : {100*obs.peak_err_Ha:.2f}% at t={obs.t_peak:.3e} s")
    if obs.t_10pct is not None:
        print(f"  error < 10% after t={obs.t_10pct:.3e} s")
    else:
        print("  error does not stay below 10% in simulated window.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("UNIFIED SCALING VALIDATION AGAINST FULL TRANSIENT CR TRUTH")
    print("=" * 72)

    print("\nLoading CR matrices...")
    L, S, Te_grid, ne_grid = load_arrays()
    print(f"  L_grid shape:  {L.shape}")
    print(f"  S_grid shape:  {S.shape}")
    print(f"  Te_grid shape: {Te_grid.shape}")
    print(f"  ne_grid shape: {ne_grid.shape}")

    print("\nBuilding interpolators...")
    L_i, S_i = build_interp(L, S, Te_grid, ne_grid)

    plot_example_transient(
        L_i, S_i, Te_grid,
        Te_old=3.0,
        ne=1e14,
        dTe=DTE_DEFAULT,
        tau_drive=100e-6,
    )

    results = validate_reduced_model(
        L_i=L_i,
        S_i=S_i,
        Te_grid=Te_grid,
        ne_grid=ne_grid,
        dTe=DTE_DEFAULT,
        tau_drives=TAU_DRIVES,
    )

    make_plots(results, Te_grid, ne_grid)
    print_summary(results)

    print("\nDone.")