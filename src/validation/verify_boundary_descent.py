#!/usr/bin/env python3
"""
verify_boundary_descent.py — Does the relaxation eigenmode descend in n with density?
=====================================================================================

THE CLAIM UNDER TEST
--------------------
The relaxation eigenmode (lambda_1, the second-least-negative eigenvalue of L)
is localised on progressively LOWER principal quantum numbers as electron
density increases. That would be a direct observation of Griem's boundary
level: the slowest-relaxing level is the one where collisional and radiative
depopulation balance, and it descends as collisions strengthen.

Analytic estimate (van der Mullen 1990):  n_cr ~ 99 * ne^(-2/17)
Griem (1963) eq. 7.77:                    n_cr ~ 141 * ne^(-2/17)
                                          slope = -2/17 = -0.1176

Griem himself cautions this closed form "gives a substantial underestimate" for
hydrogen and "does not agree very well with CR model calculations", so a
discrepancy is expected. The point is to MEASURE it, not to match it.

NOTHING IS HARDCODED
--------------------
Temperature grid, density grid, state ordering, and principal quantum numbers
are all loaded from the files the pipeline writes (see cr_context.py). If the
model changes, this script follows it rather than silently mislabelling.

HOW <n> IS DEFINED  (a CHOICE — read it and decide whether you agree)
--------------------------------------------------------------------
For the lambda_1 eigenvector v:

    w_i = v_i^2 / sum_{j != ground} v_j^2        for i != ground
    <n> = sum_i w_i * n(i)

Ground state excluded: lambda_1 always carries large ground-state weight, since
it is the mode that exchanges population between ground and excited manifold.
Including n=1 would drag <n> toward 1 everywhere and mask the descent being
measured. The quantity of interest is where in the EXCITED manifold the slow
mode lives.

Weighting by v^2 rather than |v|: v^2 is the natural population weighting (mode
amplitudes enter observables quadratically) and is insensitive to the arbitrary
overall sign of v.

The script also reports <n> under |v| weighting and with the ground state
included. If the slope changes SIGN under those variations, the result is an
artifact of the definition and must not go in the thesis. If only the magnitude
moves, the descent is real and the exponent is definition-dependent — which is
what should then be reported.

USAGE
-----
    python verify_boundary_descent.py
    python verify_boundary_descent.py --root /path/to/repo
    python verify_boundary_descent.py --no-plot

OUTPUTS (under the repo root)
-----------------------------
    validation/boundary_descent.csv
    figures/fig_boundary_descent.png and .pdf
"""

from __future__ import annotations

import argparse

import numpy as np

from cr_context import CRContext

ANALYTIC_SLOPE = -2.0 / 17.0
VDM_PREFACTOR = 99.0            # van der Mullen 1990
GRIEM_PREFACTOR = 141.0         # Griem 1963 eq. 7.77


# ----------------------------------------------------------------------------

def relaxation_mode(L):
    """
    Return (lambda_1, v_1): the second-least-negative eigenvalue of L and its
    right eigenvector, normalised so max|component| = 1.

    Sorting is by real part, descending. If any eigenvalue carries a
    significant imaginary part that ordering is ambiguous, so we refuse rather
    than silently pick one.
    """
    ev, evec = np.linalg.eig(L)
    if np.abs(ev.imag).max() > 1e-6 * np.abs(ev.real).max():
        raise RuntimeError(
            "Complex eigenvalues present — sorting by real part is unsafe here.")
    order = np.argsort(ev.real)[::-1]
    k = order[1]
    v = evec[:, k].real
    return ev[k].real, v / np.abs(v).max()


def mean_n(v, n_values, ground_index, weighting="v2", include_ground=False):
    """Weighted-mean principal quantum number of an eigenvector."""
    w = v ** 2 if weighting == "v2" else np.abs(v)
    if include_ground:
        ww, nn = w, n_values
    else:
        mask = np.arange(len(v)) != ground_index
        ww, nn = w[mask], n_values[mask]
    s = ww.sum()
    if s <= 0:
        return np.nan
    return float((ww / s * nn).sum())


def fit_loglog(x, y):
    """
    Fit y = A x^s in log-log space. Returns (slope, stderr, intercept).
    The standard error distinguishes a meaningful slope from noise.
    """
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    dof = len(lx) - 2
    if dof <= 0:
        return slope, np.nan, intercept
    se = np.sqrt(np.sum(resid ** 2) / dof / np.sum((lx - lx.mean()) ** 2))
    return slope, se, intercept


def scan(ctx, weighting="v2", include_ground=False):
    """Compute <n>, tau_relax, tau_QSS across the full grid."""
    n_te, n_ne = ctx.L_grid.shape[0], ctx.L_grid.shape[1]
    nbar = np.zeros((n_te, n_ne))
    tau_rel = np.zeros((n_te, n_ne))
    tau_qss = np.zeros((n_te, n_ne))

    for ti in range(n_te):
        for ni in range(n_ne):
            L = ctx.L_grid[ti, ni]
            ev_sorted = np.sort(np.linalg.eigvals(L).real)[::-1]
            tau_qss[ti, ni] = 1.0 / abs(ev_sorted[0])
            lam1, v1 = relaxation_mode(L)
            tau_rel[ti, ni] = 1.0 / abs(lam1)
            nbar[ti, ni] = mean_n(v1, ctx.n_values, ctx.ground_index,
                                  weighting, include_ground)
    return nbar, tau_rel, tau_qss


def report_slopes(nbar, ctx, label):
    """Fit and print <n> vs ne slopes at several Te, with standard errors."""
    print(f"\n  slope of log<n> vs log(ne)   [{label}]")
    print(f"  analytic estimate: {ANALYTIC_SLOPE:+.4f}")
    print(f"  {'Te [eV]':>9s} {'slope':>10s} {'std err':>10s} {'vs analytic':>13s}")
    print("  " + "-" * 45)
    step = max(1, len(ctx.te_grid) // 7)
    slopes = []
    for ti in range(0, len(ctx.te_grid), step):
        s, se, _ = fit_loglog(ctx.ne_grid, nbar[ti, :])
        slopes.append(s)
        print(f"  {ctx.te_grid[ti]:9.2f} {s:+10.4f} {se:10.4f} "
              f"{s/ANALYTIC_SLOPE:12.2f}x")
    return np.array(slopes)


# ----------------------------------------------------------------------------

def make_figure(ctx, nbar, outstem):
    """
    Two panels:
      (a) |v_1| vs state index at three densities — the mode collapsing toward
          low n as density rises.
      (b) <n> vs ne on log-log for several Te, with the analytic slope shown
          for comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outstem.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # pick a mid-grid temperature and low / mid / high densities
    ti = len(ctx.te_grid) // 2
    ne_pick = [0, len(ctx.ne_grid) // 2, len(ctx.ne_grid) - 1]

    for ni, colour in zip(ne_pick, ["#1b4965", "#e07a5f", "#3d5a2c"]):
        _, v = relaxation_mode(ctx.L_grid[ti, ni])
        prof = np.abs(v)
        prof = prof / prof.max()
        ax1.plot(range(len(prof)), prof, "o-", ms=3.5, lw=1.2, color=colour,
                 label=f"$n_e$ = {ctx.ne_grid[ni]:.0e} cm$^{{-3}}$")
    ax1.set_xlabel("state index (ordering from state_index.csv)")
    ax1.set_ylabel(r"$|v_{\lambda_1}|$ (normalised)")
    ax1.set_title(f"(a) relaxation eigenvector, $T_e$ = {ctx.te_grid[ti]:.2f} eV")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.25, lw=0.5)

    # mark where the principal quantum number changes, from the loaded ordering
    changes = np.where(np.diff(ctx.n_values) != 0)[0]
    for c in changes:
        ax1.axvline(c + 0.5, color="grey", lw=0.4, alpha=0.5)

    te_pick = np.linspace(0, len(ctx.te_grid) - 1, 5).astype(int)
    for ti2, colour in zip(te_pick,
                           ["#1b4965", "#5fa8d3", "#e07a5f", "#9c5b3f", "#3d5a2c"]):
        ax2.loglog(ctx.ne_grid, nbar[ti2, :], "o-", ms=4, lw=1.2, color=colour,
                   label=f"$T_e$ = {ctx.te_grid[ti2]:.1f} eV")
    anchor = nbar[len(ctx.te_grid) // 2, 0]
    ax2.loglog(ctx.ne_grid,
               anchor * (ctx.ne_grid / ctx.ne_grid[0]) ** ANALYTIC_SLOPE,
               "k--", lw=1.4, label=r"analytic $n_e^{-2/17}$")
    ax2.set_xlabel(r"$n_e$  [cm$^{-3}$]")
    ax2.set_ylabel(r"$\langle n \rangle$ of relaxation mode")
    ax2.set_title("(b) boundary-level descent with density")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25, which="both", lw=0.5)

    fig.tight_layout()
    fig.savefig(f"{outstem}.png", dpi=200)
    fig.savefig(f"{outstem}.pdf")
    print(f"\n  figure written: {outstem}.png and {outstem}.pdf")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--lgrid", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ctx = CRContext.load(root=args.root, lgrid=args.lgrid)

    print()
    print("#" * 74)
    print("# BOUNDARY-LEVEL DESCENT VERIFICATION")
    print("#" * 74)
    print(ctx.describe())

    nbar, tau_rel, tau_qss = scan(ctx, "v2", include_ground=False)

    print("\n" + "=" * 74)
    print("PRIMARY RESULT — <n> of relaxation mode (v^2 weighted, ground excluded)")
    print("=" * 74)
    step = max(1, len(ctx.te_grid) // 7)
    print(f"  {'Te [eV]':>9s} |" + "".join(f"{ne:>10.1e}" for ne in ctx.ne_grid))
    print("  " + "-" * (11 + 10 * len(ctx.ne_grid)))
    for ti in range(0, len(ctx.te_grid), step):
        print(f"  {ctx.te_grid[ti]:9.2f} |"
              + "".join(f"{nbar[ti, ni]:10.2f}" for ni in range(len(ctx.ne_grid))))

    slopes = report_slopes(nbar, ctx, "v^2 weighted, ground excluded")
    print(f"\n  measured slope range : {slopes.min():+.4f} to {slopes.max():+.4f}")
    print(f"  analytic             : {ANALYTIC_SLOPE:+.4f}")

    print("\n" + "=" * 74)
    print("SENSITIVITY — does the conclusion survive a different definition of <n>?")
    print("=" * 74)
    print("  If the SIGN flips here, the result is an artifact of the weighting")
    print("  choice and must not be reported. If only the magnitude moves, the")
    print("  descent is real and the exponent is definition-dependent.")

    nbar_abs, _, _ = scan(ctx, "abs", include_ground=False)
    s_abs = report_slopes(nbar_abs, ctx, "|v| weighted, ground excluded")

    nbar_gs, _, _ = scan(ctx, "v2", include_ground=True)
    s_gs = report_slopes(nbar_gs, ctx, "v^2 weighted, ground INCLUDED")

    print("\n  slope ranges:")
    print(f"    v^2, ground excluded : {slopes.min():+.4f} .. {slopes.max():+.4f}")
    print(f"    |v|, ground excluded : {s_abs.min():+.4f} .. {s_abs.max():+.4f}")
    print(f"    v^2, ground included : {s_gs.min():+.4f} .. {s_gs.max():+.4f}")
    all_neg = all(a.max() < 0 for a in (slopes, s_abs, s_gs))
    print(f"\n  all definitions give a NEGATIVE slope (descent): {all_neg}")

    print("\n" + "=" * 74)
    print("ANALYTIC ESTIMATE (comparison only — Griem warns it is crude for H)")
    print("=" * 74)
    print(f"  {'ne [cm^-3]':>12s} {'n_cr (vdM 99)':>15s} {'n_cr (Griem 141)':>18s}")
    print("  " + "-" * 47)
    for ne in ctx.ne_grid:
        print(f"  {ne:12.2e} {VDM_PREFACTOR*ne**ANALYTIC_SLOPE:15.2f} "
              f"{GRIEM_PREFACTOR*ne**ANALYTIC_SLOPE:18.2f}")

    out_csv = ctx.root / "validation" / "boundary_descent.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [[ctx.te_grid[ti], ctx.ne_grid[ni], nbar[ti, ni], nbar_abs[ti, ni],
             nbar_gs[ti, ni], tau_rel[ti, ni], tau_qss[ti, ni],
             tau_qss[ti, ni] / tau_rel[ti, ni]]
            for ti in range(len(ctx.te_grid)) for ni in range(len(ctx.ne_grid))]
    np.savetxt(out_csv, np.array(rows), delimiter=",",
               header="Te_eV,ne_cm3,nbar_v2_noGS,nbar_abs_noGS,nbar_v2_withGS,"
                      "tau_relax_s,tau_QSS_s,M", comments="")
    print(f"\n  data written: {out_csv}")

    if not args.no_plot:
        make_figure(ctx, nbar, ctx.root / "figures" / "fig_boundary_descent")

    print("\n" + "#" * 74)
    print("# DONE — reproducible from the pipeline's own files.")
    print("#" * 74 + "\n")


if __name__ == "__main__":
    main()
