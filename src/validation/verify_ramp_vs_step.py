#!/usr/bin/env python3
"""
verify_ramp_vs_step.py — Does the step picture apply at divertor timescales?
============================================================================

THE QUESTION
------------
The step-response error

    delta_n(0+) = n_ss_old - n_ss_new

is the QSS error for an INSTANTANEOUS change in plasma conditions. Real drives
are not instantaneous: an ELM crash takes ~100 us, detachment ~1-10 ms.

If the excited-state manifold relaxes much faster than the drive, it TRACKS the
moving target and never falls far behind — the full step error is never
realised. So before claiming a step error is physically meaningful, one must
check whether the drive is fast enough to produce it.

THE CRITERION (derived, then tested here)
-----------------------------------------
Scalar model: target ramps linearly by J over duration tau_d; system chases at
rate 1/tau_r. The lag obeys

    d(delta)/dt = -delta/tau_r + J/tau_d

giving, at the end of the ramp,

    eps_peak / J = De * (1 - exp(-1/De)),      De = tau_r / tau_d

Limits:  De -> 0   =>  eps_peak/J -> De          (tracks; error suppressed)
         De -> inf =>  eps_peak/J -> 1           (acts as a true step)

Regime boundary at De ~ 1.

WHY THIS SCRIPT EXISTS RATHER THAN JUST THE FORMULA
---------------------------------------------------
The formula above is a SINGLE-MODE scalar result. The real system is
43-dimensional and strongly NON-NORMAL (mu(L) > 0; see derivation_04c).
Non-normal systems can transiently amplify in ways single-mode analysis misses,
so the suppression may be weaker than the scalar formula predicts.

Test 3 therefore integrates the full 43-state system under a real ramp and
compares against both the scalar prediction and the step error. If they agree,
the scalar criterion is safe to quote. If the full system shows more error, the
non-normality matters and the scalar formula understates the risk.

USAGE
-----
    python verify_ramp_vs_step.py
    python verify_ramp_vs_step.py --dte 0.6          # step size in eV
    python verify_ramp_vs_step.py --no-plot

OUTPUTS (under the repo root)
-----------------------------
    validation/ramp_vs_step.csv
    figures/fig_ramp_vs_step.png / .pdf
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.integrate import solve_ivp

from cr_context import CRContext

# Divertor drive timescales, from the ITER/JET literature.
DRIVES = [
    ("ELM crash",       1e-4),
    ("fast detachment", 1e-3),
    ("slow detachment", 1e-2),
    ("inter-ELM",       1e-1),
]


# ----------------------------------------------------------------------------

def load_source(ctx, path=None):
    """
    Load the real source vector S_grid.npy if available.

    b_p = alpha_p * n_ion * n_e  — recombination feed into level p. This is the
    inhomogeneous term of dn/dt = L n + b, and n_ss = -L^{-1} b depends on it.

    If S_grid.npy is missing we STOP rather than substitute a guess: n_ss, and
    therefore every error measure in this script, depends on b. A stand-in
    source produces plausible-looking numbers that mean nothing.
    """
    from pathlib import Path
    p = Path(path) if path else ctx.root / "data/processed/cr_matrix/S_grid.npy"
    if not p.is_file():
        raise FileNotFoundError(
            f"Source vector not found: {p}\n"
            "n_ss = -L^-1 b depends on the source, so this script cannot run\n"
            "without it. Pass --sgrid with the correct path."
        )
    S = np.load(p)
    print(f"  source vector : {p.relative_to(ctx.root)}  shape {S.shape}")
    return S


def nss(L, b):
    """Solve L n_ss = -b. Never form L^{-1}: L is stiff (see derivation_03)."""
    return np.linalg.solve(L, -b)


def tau_relax(L):
    """1/|lambda_1|, the second-least-negative eigenvalue (Framing A)."""
    ev = np.sort(np.linalg.eigvals(L).real)[::-1]
    return 1.0 / abs(ev[1])


def suppression(De):
    """Scalar prediction eps_peak/J = De (1 - exp(-1/De)), guarded at De -> 0."""
    De = np.asarray(De, dtype=float)
    return De * (1.0 - np.exp(-1.0 / np.maximum(De, 1e-300)))


# ----------------------------------------------------------------------------

def test_scalar_criterion():
    """
    TEST 1 — verify the analytic suppression formula against direct integration
    of the scalar model. This checks the FORMULA, not the plasma.
    """
    print("=" * 74)
    print("TEST 1 — Scalar criterion: analytic vs numerical")
    print("=" * 74)
    print("  Ramp of duration tau_d; single mode relaxing at tau_r; De = tau_r/tau_d")
    print()
    print(f"  {'De':>10s} {'numerical':>14s} {'analytic':>14s} {'rel. diff':>12s}")
    print("  " + "-" * 52)

    ok = True
    for De in [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]:
        tau_r, J = 1.0, 1.0
        tau_d = tau_r / De

        def target(t):
            return J * min(t / tau_d, 1.0)

        sol = solve_ivp(lambda t, x: [-(x[0] - target(t)) / tau_r],
                        [0, tau_d + 20 * tau_r], [0.0], dense_output=True,
                        rtol=1e-10, atol=1e-14,
                        max_step=min(tau_r, tau_d) / 50)
        ts = np.linspace(0, tau_d + 20 * tau_r, 20000)
        lag = np.array([target(t) for t in ts]) - sol.sol(ts)[0]
        num = np.abs(lag).max() / J
        ana = suppression(De)
        rel = abs(num - ana) / ana
        if rel > 1e-3:
            ok = False
        print(f"  {De:10.0e} {num:14.6f} {ana:14.6f} {rel:12.2e}")

    print()
    print(f"  VERDICT: {'formula confirmed' if ok else 'MISMATCH — investigate'}")
    print()
    return ok


def test_grid_suppression(ctx, out_csv=None):
    """
    TEST 2 — apply the scalar criterion across the real (Te, ne) grid.

    For each grid point and each divertor drive timescale, compute De and the
    suppression factor. This says how much of the idealised step error would
    actually be realised.
    """
    print("=" * 74)
    print("TEST 2 — Suppression across the grid (scalar criterion)")
    print("=" * 74)

    n_te, n_ne = ctx.L_grid.shape[0], ctx.L_grid.shape[1]
    tr = np.zeros((n_te, n_ne))
    for ti in range(n_te):
        for ni in range(n_ne):
            tr[ti, ni] = tau_relax(ctx.L_grid[ti, ni])

    print(f"  tau_relax range : {tr.min()*1e9:.3f} to {tr.max()*1e9:.3f} ns")
    ti_s, ni_s = np.unravel_index(tr.argmax(), tr.shape)
    print(f"  slowest at      : Te = {ctx.te_grid[ti_s]:.2f} eV, "
          f"ne = {ctx.ne_grid[ni_s]:.2e} cm^-3")
    print()

    rows = []
    for name, td in DRIVES:
        De = tr / td
        supp = suppression(De)
        frac10 = 100.0 * (supp > 0.1).mean()
        frac1 = 100.0 * (supp > 0.01).mean()
        print(f"  --- {name} (tau_drive = {td:.0e} s) ---")
        print(f"      De range          : {De.min():.2e} to {De.max():.2e}")
        print(f"      suppression       : {supp.min():.2e} to {supp.max():.2e}")
        print(f"      grid with supp>10%: {frac10:.1f} %")
        print(f"      grid with supp> 1%: {frac1:.1f} %")
        print()
        for ti in range(n_te):
            for ni in range(n_ne):
                rows.append([ctx.te_grid[ti], ctx.ne_grid[ni], name, td,
                             tr[ti, ni], De[ti, ni], supp[ti, ni]])

    tdrive_needed = tr.max()
    print(f"  For the step picture to apply anywhere (De ~ 1), the drive must be")
    print(f"  faster than the slowest relaxation: tau_drive <~ {tdrive_needed*1e9:.1f} ns.")
    print(f"  Fastest divertor process considered: {min(d[1] for d in DRIVES):.0e} s")
    print(f"  Ratio: {min(d[1] for d in DRIVES)/tdrive_needed:.0f}x too slow.")
    print()

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w") as fh:
            fh.write("Te_eV,ne_cm3,drive_name,tau_drive_s,tau_relax_s,De,suppression\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n")
        print(f"  written: {out_csv}")
        print()
    return tr


def test_full_system(ctx, S_grid, dte=0.6):
    """
    TEST 3 — THE ONE THAT MATTERS.

    Integrate the FULL 43-state system under a finite ramp and compare the peak
    error against (a) the idealised step error and (b) the scalar prediction.

    Why this is not redundant with Test 2: the scalar criterion assumes a single
    relaxation mode. The real L is 43-dimensional and strongly non-normal
    (mu(L) = +1.3e11 s^-1 vs spectral abscissa -4.4e4; derivation_04c), and
    non-normal systems can transiently amplify. If the full system shows more
    error than the scalar formula predicts, non-normality defeats part of the
    suppression and the scalar criterion is optimistic.

    Interpolation note: L and b are only known on the grid, so during the ramp
    we interpolate linearly in Te between the two bracketing grid points. This
    is an approximation; a finer Te grid would sharpen it.
    """
    print("=" * 74)
    print("TEST 3 — Full 43-state system under a finite ramp")
    print("=" * 74)

    ti0, ni = ctx.nearest_point(3.0, 1.4e14)
    te0 = ctx.te_grid[ti0]
    ti1 = int(np.argmin(np.abs(ctx.te_grid - (te0 + dte))))
    te1 = ctx.te_grid[ti1]

    L0, L1 = ctx.L_grid[ti0, ni], ctx.L_grid[ti1, ni]
    b0, b1 = S_grid[ti0, ni], S_grid[ti1, ni]

    n0, n1 = nss(L0, b0), nss(L1, b1)
    J = np.linalg.norm(n0 - n1) / np.linalg.norm(n1)
    tr = tau_relax(L0)

    print(f"  step        : Te {te0:.4f} -> {te1:.4f} eV at ne = {ctx.ne_grid[ni]:.3e}")
    print(f"  tau_relax   : {tr*1e9:.4f} ns")
    print(f"  J (step err): {J:.6f}   <- the idealised instantaneous-step error")
    print()
    print(f"  {'drive':>18s} {'tau_d [s]':>11s} {'De':>10s} "
          f"{'scalar pred':>13s} {'FULL system':>13s} {'ratio':>8s}")
    print("  " + "-" * 78)

    results = []
    # include some artificially fast drives to locate the crossover
    probes = DRIVES + [("(probe) 1 us", 1e-6), ("(probe) 100 ns", 1e-7),
                       ("(probe) 10 ns", 1e-8), ("(probe) 1 ns", 1e-9)]

    for name, td in probes:
        De = tr / td
        pred = suppression(De) * J

        def L_of(t):
            f = min(max(t / td, 0.0), 1.0)
            return (1 - f) * L0 + f * L1

        def b_of(t):
            f = min(max(t / td, 0.0), 1.0)
            return (1 - f) * b0 + f * b1

        def rhs(t, n):
            return L_of(t) @ n + b_of(t)

        T = td + 30 * tr
        sol = solve_ivp(rhs, [0, T], n0, dense_output=True,
                        method="LSODA", rtol=1e-9, atol=1e-6 * np.abs(n0).max(),
                        max_step=max(min(td, tr) / 20, 1e-15))
        ts = np.unique(np.concatenate([
            np.linspace(0, td, 4000),
            np.logspace(np.log10(max(td, 1e-15)), np.log10(T), 4000)]))
        ns = sol.sol(ts)

        # instantaneous QSS target along the ramp
        errs = []
        for k, t in enumerate(ts):
            tgt = nss(L_of(t), b_of(t))
            errs.append(np.linalg.norm(ns[:, k] - tgt) / np.linalg.norm(tgt))
        full = max(errs)

        results.append((name, td, De, pred, full))
        print(f"  {name:>18s} {td:11.0e} {De:10.2e} {pred:13.3e} "
              f"{full:13.3e} {full/pred if pred>0 else np.nan:8.2f}")

    print()
    print("  READING THIS:")
    print("    'FULL system' / 'scalar pred' ~ 1  -> scalar criterion is accurate")
    print("    ratio >> 1                         -> non-normality defeats part of")
    print("                                          the suppression; scalar formula")
    print("                                          is OPTIMISTIC and must not be")
    print("                                          quoted alone")
    print("    FULL system ~ J                    -> the step picture applies")
    print()
    return J, tr, results


# ----------------------------------------------------------------------------

def make_figure(ctx, tr, outstem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outstem.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    De = np.logspace(-6, 3, 400)
    ax1.loglog(De, suppression(De), lw=2, color="#1b4965")
    ax1.axvline(1.0, color="grey", ls="--", lw=1)
    ax1.text(1.3, 1e-4, "De = 1\nregime boundary", fontsize=8, color="grey")
    for name, td in DRIVES:
        d = tr.max() / td
        ax1.plot(d, suppression(d), "o", ms=7, label=f"{name} (max over grid)")
    ax1.set_xlabel(r"$De = \tau_{\rm relax}/\tau_{\rm drive}$")
    ax1.set_ylabel(r"$\varepsilon_{\rm peak}/J$  (suppression factor)")
    ax1.set_title("(a) how much of the step error is realised")
    ax1.legend(fontsize=7, frameon=False, loc="upper left")
    ax1.grid(alpha=0.25, which="both", lw=0.5)

    for name, td in DRIVES:
        s = suppression(tr / td)
        ax2.loglog(ctx.ne_grid, s.max(axis=0), "o-", ms=4, lw=1.2, label=name)
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.text(ctx.ne_grid[0], 1.3, "full step error", fontsize=8)
    ax2.set_xlabel(r"$n_e$ [cm$^{-3}$]")
    ax2.set_ylabel("max suppression over $T_e$")
    ax2.set_title("(b) suppression across the density grid")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25, which="both", lw=0.5)

    fig.tight_layout()
    fig.savefig(f"{outstem}.png", dpi=200)
    fig.savefig(f"{outstem}.pdf")
    print(f"  figure written: {outstem}.png and .pdf")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--lgrid", default=None)
    ap.add_argument("--sgrid", default=None, help="path to S_grid.npy")
    ap.add_argument("--dte", type=float, default=0.6, help="step size in eV")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--skip-full", action="store_true",
                    help="skip Test 3 (the slow one)")
    args = ap.parse_args()

    ctx = CRContext.load(root=args.root, lgrid=args.lgrid)
    print()
    print("#" * 74)
    print("# RAMP vs STEP — does the step picture apply at divertor timescales?")
    print("#" * 74)
    print(ctx.describe())

    test_scalar_criterion()
    tr = test_grid_suppression(ctx, out_csv=ctx.root / "validation" / "ramp_vs_step.csv")

    if not args.skip_full:
        S_grid = load_source(ctx, args.sgrid)
        test_full_system(ctx, S_grid, dte=args.dte)

    if not args.no_plot:
        make_figure(ctx, tr, ctx.root / "figures" / "fig_ramp_vs_step")

    print()
    print("#" * 74)
    print("# DONE — reproducible from the pipeline's own files.")
    print("#" * 74)
    print()


if __name__ == "__main__":
    main()
