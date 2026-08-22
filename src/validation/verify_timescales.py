#!/usr/bin/env python3
"""
verify_timescales.py — Independent verification of the two-timescale structure
==============================================================================

PURPOSE
-------
Tests from scratch whether the CR matrix L actually exhibits two separated
timescales, and identifies what the slow eigenmodes physically are. Written to
be READ and CHECKED, not trusted. Each test states what it computes, what
result supports the two-timescale claim, and what would refute it.

Nothing is taken from any note, report, or thesis text, and nothing about the
model is hardcoded: the temperature grid, density grid, and state ordering are
all loaded from the files the pipeline itself writes (see cr_context.py).

USAGE
-----
    python verify_timescales.py
    python verify_timescales.py --root /path/to/repo
    python verify_timescales.py --te 3.0 --ne 1.4e14      # choose test point
    python verify_timescales.py --skip-grid               # skip the slow scan

Outputs are written under validation/ in the repo root.

TESTS
-----
 1. Matrix sanity        structure of L (signs, conservation)
 2. Full spectrum        are there GAPS, or a continuum?
 3. Framing A vs B       two definitions of tau_relax; do they agree?
 4. Eigenvector identity what ARE the slow modes?
 5. ODE integration      INDEPENDENT check by matrix exponential; no
                         eigenvalues used at all
 6. Grid scan            how timescales and mode localisation vary with (Te,ne)
"""

from __future__ import annotations

import argparse

import numpy as np

from cr_context import CRContext


# ----------------------------------------------------------------------------

def test_matrix_sanity(L):
    """
    Structural checks. These must pass before any eigenvalue means anything.

    For a valid CR rate matrix with convention L[i,j] = rate (j -> i):
      diagonal negative      states lose population
      off-diagonal >= 0      gains are positive rates
      column sums <= 0       only ionisation removes population; every other
                             process conserves it
    """
    print("=" * 74)
    print("TEST 1 — Matrix sanity")
    print("=" * 74)

    diag = np.diag(L)
    off = L - np.diag(diag)
    colsum = L.sum(axis=0)
    scale = np.abs(diag).max()

    n_pos_diag = int((diag > 0).sum())
    n_neg_off = int((off < -1e-30).sum())
    n_pos_col = int((colsum > 1e-6 * scale).sum())

    print(f"  shape                     : {L.shape}")
    print(f"  NaN / Inf                 : {int(np.isnan(L).sum())} / {int(np.isinf(L).sum())}")
    print(f"  positive diagonals        : {n_pos_diag}   (expect 0)")
    print(f"  negative off-diagonals    : {n_neg_off}   (expect 0)")
    print(f"  positive column sums      : {n_pos_col}   (expect 0)")
    print(f"  column sums range         : [{colsum.min():.3e}, {colsum.max():.3e}]")
    print(f"  max|colsum| / max|diag|   : {np.abs(colsum).max()/scale:.3e}")
    print(f"  |diag| range              : [{np.abs(diag).min():.3e}, {scale:.3e}]")

    ok = (n_pos_diag == 0) and (n_neg_off == 0) and (n_pos_col == 0)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  >>> Structure is wrong. Do not interpret the eigenvalues below.")
    print()
    return ok


def test_spectrum(L, out_csv=None):
    """
    All eigenvalues, sorted slow to fast, with the ratio between consecutive
    timescales.

    The two-timescale claim predicts a LARGE gap between the slowest mode and
    the rest. A smooth continuum with no gap would refute it. Reporting every
    consecutive ratio lets you see how many genuine groups there are rather
    than assuming the answer.
    """
    print("=" * 74)
    print("TEST 2 — Full spectrum: gaps or continuum?")
    print("=" * 74)

    ev_c = np.linalg.eigvals(L)
    imag_rel = np.abs(ev_c.imag).max() / np.abs(ev_c.real).max()
    ev = np.sort(ev_c.real)[::-1]
    tau = 1.0 / np.abs(ev)

    print(f"  max|Im| / max|Re|         : {imag_rel:.3e}  (expect ~0)")
    print(f"  all eigenvalues negative  : {bool((ev < 0).all())}")
    print()
    print(f"  {'k':>3s} {'lambda_k [1/s]':>17s} {'tau_k [s]':>14s} {'tau_k/tau_k+1':>15s}")
    print("  " + "-" * 53)
    for k in range(len(ev)):
        if k < len(ev) - 1:
            r = tau[k] / tau[k + 1]
            flag = "   <== GAP" if r > 10 else ""
            print(f"  {k:3d} {ev[k]:17.6e} {tau[k]:14.6e} {r:15.2f}{flag}")
        else:
            print(f"  {k:3d} {ev[k]:17.6e} {tau[k]:14.6e} {'-':>15s}")

    ratios = tau[:-1] / tau[1:]
    big = np.where(ratios > 10)[0]
    print()
    print(f"  gaps with ratio > 10      : {len(big)} at k = {list(big)}")
    print(f"  largest gap               : {ratios.max():.1f}x between k={ratios.argmax()}"
          f" and k={ratios.argmax()+1}")
    print()
    print("  READING THIS:")
    print("    exactly 1 large gap -> two separated timescales")
    print("    2+ large gaps       -> three or more distinct groups")
    print("    0 large gaps        -> continuum; two-timescale picture NOT supported")
    print()

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_csv, np.column_stack([np.arange(len(ev)), ev, tau]),
                   delimiter=",", header="k,lambda_per_s,tau_s", comments="")
        print(f"  written: {out_csv}")
        print()
    return ev


def test_framings(L, ground_index):
    """
    Two definitions of tau_relax are in circulation:

      A (full matrix)   1/|lambda_1| with lambda_1 the second-least-negative
                        eigenvalue of the full L.
      B (excited block) delete the ground-state row and column, take the
                        least-negative eigenvalue of what remains. This matches
                        the textbook QSS framing: freeze the ground state as a
                        reservoir, ask how fast the excited manifold settles.

    They are conceptually different. If they agree numerically the choice is
    presentational; if not, the thesis must say which it uses and why.
    """
    print("=" * 74)
    print("TEST 3 — Framing A vs Framing B for tau_relax")
    print("=" * 74)

    evA = np.sort(np.linalg.eigvals(L).real)[::-1]
    tauA = 1.0 / abs(evA[1])

    keep = [i for i in range(L.shape[0]) if i != ground_index]
    L_exc = L[np.ix_(keep, keep)]
    evB = np.sort(np.linalg.eigvals(L_exc).real)[::-1]
    tauB = 1.0 / abs(evB[0])

    print(f"  A: full matrix, lambda_1           : {tauA:.6e} s")
    print(f"  B: excited block, slowest          : {tauB:.6e} s")
    print(f"  ratio B/A                          : {tauB/tauA:.8f}")
    print(f"  relative difference                : {abs(tauB/tauA - 1)*100:.4f} %")
    print()
    return tauA, tauB


def test_eigenvectors(L, ctx, n_modes=4, threshold=0.12):
    """
    Identify what each slow mode physically IS.

    An eigenvector is a pattern of population deviations that decays without
    changing shape. Its components say how much each state participates:

      one state dominant        single-level mode
      many states, mixed signs  collective mode (population moving BETWEEN
                                states rather than simply draining)

    DIAGNOSTICS REPORTED, and what each one actually means:

    ground-state weight fraction
        v_ground^2 / sum(v^2). Says how much of the mode lives on the ground
        state. Near 1 means the mode is essentially the ground state alone.

    participation ratio  PR = 1 / sum(p_i^2)  with p_i = v_i^2/sum(v^2)
        The effective number of states involved. PR = 1 means one state; PR = 5
        means roughly five states share the mode. This is a cleaner statement
        than eyeballing the largest few components, and it is the number to
        quote when arguing a mode is collective rather than a single-level
        lifetime.

    max |n| span
        The largest and smallest principal quantum number carrying appreciable
        weight. Says whether the mode is confined to one shell (intra-shell
        l-mixing) or spans several (inter-shell relaxation).

    WHAT IS DELIBERATELY NOT REPORTED, and why:

    sum of components, sum(v_k)
        An earlier version of this script printed this and labelled modes
        "conserving" or "loss" from it. That was WRONG. The reasoning would
        only hold if 1 (the all-ones vector) were a left eigenvector of L,
        which requires zero column sums. L's column sums are -K_ion*ne, not
        zero, so sum(v_k) has no reason to vanish for k != 0 and carries no
        conservation meaning.

    net population-change rate, (1^T L v_k)/(1^T v_k)
        Also dropped: by the eigenvalue equation this collapses identically to
        lambda_k, so it is a tautology rather than an independent diagnostic.
    """
    print("=" * 74)
    print("TEST 4 — Eigenvector identity: what ARE the slow modes?")
    print("=" * 74)
    print("  NOTE: sum-of-components is NOT reported, because L has non-zero")
    print("  column sums (ionisation), so it is not a conservation diagnostic.")
    print()

    ev, evec = np.linalg.eig(L)
    order = np.argsort(ev.real)[::-1]
    gi = ctx.ground_index

    for rank in range(min(n_modes, len(order))):
        k = order[rank]
        lam = ev[k].real
        v = evec[:, k].real
        v = v / np.abs(v).max()

        big = [(ctx.labels[i], v[i]) for i in range(len(v)) if abs(v[i]) > threshold]
        big.sort(key=lambda t: -abs(t[1]))

        gs_frac = v[gi] ** 2 / np.sum(v ** 2)
        p = v ** 2 / np.sum(v ** 2)
        pr = 1.0 / np.sum(p ** 2)

        # shells carrying appreciable weight -> intra-shell vs inter-shell
        active = np.abs(v) > threshold
        if active.any():
            n_active = ctx.n_values[active]
            span = f"n = {int(n_active.min())} to {int(n_active.max())}"
            kind = ("intra-shell (single n)" if n_active.min() == n_active.max()
                    else "inter-shell")
        else:
            span, kind = "none above threshold", "-"

        print(f"  --- mode {rank}: lambda = {lam:.6e} 1/s, tau = {1/abs(lam):.6e} s ---")
        print(f"      ground-state weight fraction : {gs_frac:.4f}")
        print(f"      participation ratio          : {pr:.2f} states involved")
        print(f"      shells with weight > {threshold}     : {span}  [{kind}]")
        print(f"      components > {threshold}           : "
              + ", ".join(f"{s}={w:+.3f}" for s, w in big[:8]))
        print()
    return ev, evec


def test_ode(L, seed=0, n_trials=3):
    """
    THE INDEPENDENT TEST — uses no eigenvalue decomposition at all.

    Integrate d(delta_n)/dt = L delta_n with the matrix exponential from random
    initial perturbations, and track ||delta_n(t)||/||delta_n(0)||.

      two timescales  -> fast drop, PLATEAU spanning decades, second drop
      continuum       -> smooth steady decline, no plateau

    Several random starts are used because a single one could accidentally have
    negligible overlap with a mode and miss it.

    Watch also for the ratio INCREASING. That is transient amplification. It
    cannot happen for a normal matrix and indicates L's eigenvectors are not
    orthogonal, meaning error can grow before it decays.
    """
    from scipy.linalg import expm

    print("=" * 74)
    print("TEST 5 — Direct ODE integration (no eigenvalues used)")
    print("=" * 74)

    times = np.logspace(-14, -3, 34)
    rng = np.random.default_rng(seed)
    n = L.shape[0]

    curves = []
    for _ in range(n_trials):
        d0 = rng.standard_normal(n)
        d0 /= np.linalg.norm(d0)
        curves.append([np.linalg.norm(expm(L * t) @ d0) for t in times])
    curves = np.array(curves)

    print(f"  {'t [s]':>12s}" + "".join(f"{'trial'+str(i):>12s}" for i in range(n_trials)))
    print("  " + "-" * (12 + 12 * n_trials))
    for i, t in enumerate(times):
        row = f"  {t:12.3e}"
        for tr in range(n_trials):
            row += f"{curves[tr, i]:12.6f}"
        print(row)

    grew = []
    for tr in range(n_trials):
        c = curves[tr]
        for i in range(len(c) - 1):
            if c[i + 1] > c[i] * 1.001:
                grew.append((tr, times[i], c[i], c[i + 1]))
                break
    print()
    if grew:
        print(f"  TRANSIENT GROWTH DETECTED in {len(grew)}/{n_trials} trials:")
        for tr, t, a, b in grew:
            print(f"    trial {tr}: at t={t:.2e} s the ratio rose {a:.4f} -> {b:.4f}")
        print("  This means L is non-normal: error can grow before it decays.")
    else:
        print("  No transient growth detected; decay is monotonic in all trials.")
    print()
    return times, curves


def test_grid_scan(ctx, out_csv=None):
    """
    Scan the full (Te, ne) grid, computing at each point:

      tau_QSS   = 1/|lambda_0|
      tau_relax = 1/|lambda_1|
      M         = tau_QSS / tau_relax
      <n>       = weight-squared-averaged principal quantum number of the
                  lambda_1 eigenvector, ground state EXCLUDED. This says which
                  shell the relaxation mode lives on.

    Griem's boundary-level picture predicts <n> decreases with density, with an
    analytic estimate n_cr ~ ne^(-2/17) = ne^(-0.118). We measure the actual
    exponent rather than assuming it.
    """
    print("=" * 74)
    print("TEST 6 — Grid scan")
    print("=" * 74)

    n_te, n_ne = ctx.L_grid.shape[0], ctx.L_grid.shape[1]
    gi = ctx.ground_index
    keep = np.array([i for i in range(ctx.n_states) if i != gi])
    n_exc = ctx.n_values[keep]

    tau_qss = np.zeros((n_te, n_ne))
    tau_rel = np.zeros((n_te, n_ne))
    nbar = np.zeros((n_te, n_ne))

    for ti in range(n_te):
        for ni in range(n_ne):
            ev, evec = np.linalg.eig(ctx.L_grid[ti, ni])
            order = np.argsort(ev.real)[::-1]
            tau_qss[ti, ni] = 1.0 / abs(ev[order[0]].real)
            tau_rel[ti, ni] = 1.0 / abs(ev[order[1]].real)
            v = evec[:, order[1]].real
            w = v[keep] ** 2
            s = w.sum()
            nbar[ti, ni] = float((w / s * n_exc).sum()) if s > 0 else np.nan

    M = tau_qss / tau_rel
    step = max(1, n_te // 7)

    def table(arr, title, fmt="{:11.3f}"):
        print(f"\n  {title}")
        print(f"  {'Te [eV]':>8s} |" + "".join(f"{ne:>11.1e}" for ne in ctx.ne_grid))
        print("  " + "-" * (10 + 11 * n_ne))
        for ti in range(0, n_te, step):
            print(f"  {ctx.te_grid[ti]:8.2f} |"
                  + "".join(fmt.format(arr[ti, ni]) for ni in range(n_ne)))

    table(M, "M = tau_QSS / tau_relax", "{:11.1f}")
    table(tau_rel * 1e9, "tau_relax [ns]")
    table(nbar, "<n> of relaxation mode (ground excluded)", "{:11.2f}")

    print("\n  log-log slope of <n> vs ne   (analytic estimate: -2/17 = -0.1176)")
    for ti in range(0, n_te, step):
        s = np.polyfit(np.log(ctx.ne_grid), np.log(nbar[ti, :]), 1)[0]
        print(f"    Te = {ctx.te_grid[ti]:6.2f} eV : {s:+.4f}")

    print("\n  log-log slope of tau_relax vs ne   (pure collisional would be -1.0)")
    for ti in range(0, n_te, step):
        s = np.polyfit(np.log(ctx.ne_grid), np.log(tau_rel[ti, :]), 1)[0]
        print(f"    Te = {ctx.te_grid[ti]:6.2f} eV : {s:+.4f}")

    # Flag extreme slow timescales. These arise physically at low Te (ionising
    # ground-state hydrogen from 13.6 eV is exponentially suppressed when
    # kTe ~ 1 eV), but they are also exactly where other code in the pipeline
    # may silently discard eigenvalues via a |lambda| threshold. If another
    # script reports a much smaller tau_QSS in this corner, suspect truncation
    # rather than a physics disagreement.
    long_mask = tau_qss > 1.0
    if long_mask.any():
        idx = np.argwhere(long_mask)
        print(f"\n  NOTE: {long_mask.sum()} grid point(s) have tau_QSS > 1 s "
              f"(max {tau_qss.max():.3e} s).")
        ti_w, ni_w = idx[np.argmax(tau_qss[long_mask])]
        print(f"        largest at Te = {ctx.te_grid[ti_w]:.2f} eV, "
              f"ne = {ctx.ne_grid[ni_w]:.2e} cm^-3")
        print("        These are physical (ionisation is exponentially slow at low")
        print("        Te) but lie outside any regime the model is meant for, AND")
        print("        are where eigenvalue-threshold filters elsewhere in the")
        print("        pipeline may silently truncate. Cross-check before quoting.")
    print()

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [[ctx.te_grid[ti], ctx.ne_grid[ni], tau_qss[ti, ni],
                 tau_rel[ti, ni], M[ti, ni], nbar[ti, ni]]
                for ti in range(n_te) for ni in range(n_ne)]
        np.savetxt(out_csv, np.array(rows), delimiter=",",
                   header="Te_eV,ne_cm3,tau_QSS_s,tau_relax_s,M,n_bar", comments="")
        print(f"  written: {out_csv}")
        print()
    return tau_qss, tau_rel, M, nbar


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="repo root (auto-discovered if omitted)")
    ap.add_argument("--lgrid", default=None,
                    help="explicit path to L_grid.npy (overrides default location)")
    ap.add_argument("--te", type=float, default=3.0,
                    help="Te [eV] for the single-point tests (default 3.0)")
    ap.add_argument("--ne", type=float, default=1.4e14,
                    help="ne [cm^-3] for the single-point tests (default 1.4e14)")
    ap.add_argument("--skip-grid", action="store_true",
                    help="skip test 6 (the slow full-grid scan)")
    args = ap.parse_args()

    ctx = CRContext.load(root=args.root, lgrid=args.lgrid)

    print()
    print("#" * 74)
    print("# TWO-TIMESCALE VERIFICATION")
    print("#" * 74)
    print(ctx.describe())

    ti, ni = ctx.nearest_point(args.te, args.ne)
    print(f"\n  requested test point : Te = {args.te:.4g} eV, ne = {args.ne:.4g} cm^-3")
    print(f"  nearest grid point   : index ({ti}, {ni}) -> "
          f"Te = {ctx.te_grid[ti]:.4f} eV, ne = {ctx.ne_grid[ni]:.4e} cm^-3")
    print()

    outdir = ctx.root / "validation"
    L = ctx.L_grid[ti, ni]

    if not test_matrix_sanity(L):
        print("Matrix sanity failed — stopping.")
        return

    test_spectrum(L, out_csv=outdir / "spectrum_testpoint.csv")
    test_framings(L, ctx.ground_index)
    test_eigenvectors(L, ctx)
    test_ode(L)

    if not args.skip_grid:
        test_grid_scan(ctx, out_csv=outdir / "timescale_verification.csv")

    print("#" * 74)
    print("# DONE — every number above is reproducible from the pipeline's own files.")
    print("#" * 74)
    print()


if __name__ == "__main__":
    main()
