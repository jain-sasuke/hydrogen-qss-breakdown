"""
verify_ch3_claims.py
====================
Report-only audit of every Chapter 3 claim that currently rests on a value
pasted into a chat window, a number quoted from another script, or an assumed
grid. Nothing here is repaired, substituted, or written to disk.

Each check states the EXPECTED value first and then the measured one, so a
disagreement is a falsification rather than something to rationalise. Checks
that cannot be settled from the matrix alone are printed as OPEN with the
artifact that would settle them named.

Run:
    python src/validation/verify_ch3_claims.py

Platform: macOS/BSD, zsh. Pure Python, no shell utilities, no GNU flags.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from cr_context import CRContext
except ImportError:  # invoked from elsewhere in the tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
    from cr_context import CRContext

FAILURES: list[str] = []
OPEN: list[str] = []


def check(label: str, expected, measured, rel_tol: float, unit: str = "") -> None:
    """Compare measured against the value the thesis currently asserts."""
    if expected is None:
        print(f"  [----] {label:<46s} measured {measured:.6g}{unit}")
        return
    rel = abs(measured - expected) / max(abs(expected), 1e-300)
    ok = rel <= rel_tol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label:<46s} thesis {expected:.6g}{unit}"
          f"   measured {measured:.6g}{unit}   rel {rel:.2e}")
    if not ok:
        FAILURES.append(f"{label}: thesis {expected:.6g}, measured {measured:.6g} "
                        f"(rel {rel:.2e})")


def main() -> None:
    ctx = CRContext.load()
    root = ctx.root
    L, Te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid

    # ---------------------------------------------------------------- 0. provenance
    print("=" * 78)
    print("0. PROVENANCE")
    print("=" * 78)
    print(f"  working dir      {Path.cwd()}")
    print(f"  repo root        {root}")
    for rel in ("data/processed/cr_matrix/L_grid.npy",
                "data/processed/cr_matrix/S_grid.npy"):
        f = root / rel
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"  {rel}")
        print(f"      sha256 {h}")
        print(f"      mtime  {f.stat().st_mtime}")
    print(f"  L_grid shape     {L.shape}")
    print(f"  grid             {len(Te)} Te x {len(ne)} ne = {len(Te)*len(ne)} points")

    # ---------------------------------------------------------------- 1. the grid
    print()
    print("=" * 78)
    print("1. IS THE GRID WHAT THE CHAPTER ASSUMES?")
    print("   Everything about the step-vs-grid-point argument depends on this.")
    print("=" * 78)
    r_te = Te[1:] / Te[:-1]
    r_ne = ne[1:] / ne[:-1]
    log_te = np.allclose(r_te, r_te[0], rtol=1e-9)
    log_ne = np.allclose(r_ne, r_ne[0], rtol=1e-9)
    print(f"  Te log-spaced?   {log_te}   ratio {r_te[0]:.8f} "
          f"(spread {r_te.max()/r_te.min() - 1:.2e})")
    print(f"  ne log-spaced?   {log_ne}   ratio {r_ne[0]:.8f}")
    check("Te[0]", 1.0, float(Te[0]), 1e-6, " eV")
    check("Te[-1]", 10.0, float(Te[-1]), 1e-6, " eV")
    check("ne[0]", 1.0e12, float(ne[0]), 1e-6, " cm^-3")
    check("ne[-1]", 1.0e15, float(ne[-1]), 1e-6, " cm^-3")
    if not log_te:
        FAILURES.append("Te grid is NOT log-spaced: the one-grid-step argument "
                        "in Sec 3.3 does not hold as written")

    ib = int(np.argmin(np.abs(Te - 2.947)))
    jb = int(np.argmin(np.abs(ne - 1.389e14)))
    print(f"  benchmark index  [{ib},{jb}]  (chapter asserts [23,5])")
    if (ib, jb) != (23, 5):
        FAILURES.append(f"benchmark index is [{ib},{jb}], chapter says [23,5]")
    check("Te at benchmark", 2.9471, float(Te[ib]), 1e-4, " eV")
    check("ne at benchmark", 1.3895e14, float(ne[jb]), 1e-4, " cm^-3")

    # ---------------------------------------------------------------- 2. spectra
    print()
    print("=" * 78)
    print("2. SPECTRUM, GRID-WIDE  (Eq. 3.14 and the isolation claim)")
    print("=" * 78)
    n_pts = len(Te) * len(ne)
    tq = np.zeros((len(Te), len(ne)))
    tr = np.zeros_like(tq)
    iso = np.zeros_like(tq)
    max_imag = 0.0
    max_real = -np.inf
    for i in range(len(Te)):
        for j in range(len(ne)):
            e = np.linalg.eigvals(L[i, j])
            max_real = max(max_real, float(e.real.max()))
            max_imag = max(max_imag, float(np.max(np.abs(e.imag)
                                                 / np.maximum(np.abs(e), 1e-300))))
            t = np.sort(1.0 / np.abs(e.real))[::-1]
            tq[i, j], tr[i, j] = t[0], t[1]
            r = t[:-1] / t[1:]
            iso[i, j] = r[0] / r[1:].max()
    M = tq / tr

    print(f"  max Re(lambda)   {max_real:.6e}   (must be < 0)")
    print(f"  max |Im|/|lam|   {max_imag:.6e}   PER-EIGENVALUE, not global/global")
    if max_real >= 0:
        FAILURES.append("a non-decaying eigenvalue exists on the grid")
    if max_imag > 1e-10:
        FAILURES.append(f"complex eigenvalues on the grid: {max_imag:.3e}")

    print("\n  -- Eq. (3.14), the six bounds the chapter states --")
    check("tau_relax min", 0.87e-9, float(tr.min()), 2e-2, " s")
    check("tau_relax max", 38.9e-9, float(tr.max()), 2e-2, " s")
    check("tau_QSS   min", 1.18e-6, float(tq.min()), 2e-2, " s")
    check("tau_QSS   max", 67.2, float(tq.max()), 2e-2, " s")
    check("M min", 86.8, float(M.min()), 2e-2)
    check("M max", 1.73e9, float(M.max()), 2e-2)

    a, b = np.unravel_index(int(np.argmin(M)), M.shape)
    c, d = np.unravel_index(int(np.argmax(M)), M.shape)
    e_, f_ = np.unravel_index(int(np.argmin(tq)), tq.shape)
    g_, h_ = np.unravel_index(int(np.argmin(iso)), iso.shape)
    print(f"\n  M min      at [{a},{b}]  Te={Te[a]:.4g} eV  ne={ne[b]:.4g}")
    print(f"  M max      at [{c},{d}]  Te={Te[c]:.4g} eV  ne={ne[d]:.4g}"
          f"     (findings_09 W2 records [1,0])")
    print(f"  tau_QSS min at [{e_},{f_}]  Te={Te[e_]:.4g} eV  ne={ne[f_]:.4g}")
    print(f"  isolation min at [{g_},{h_}]  Te={Te[g_]:.4g} eV  ne={ne[h_]:.4g}")
    same = (a, b) == (e_, f_) == (g_, h_)
    print(f"  chapter claims M-min, tau_QSS-min, tau_relax-min and isolation-min")
    print(f"  all coincide at one point: {same}")
    if not same:
        FAILURES.append("the four minima do NOT coincide; Sec 3.3.4 says they do")

    print(f"\n  isolation at benchmark   {iso[ib, jb]:.1f}x   (chapter: 3567)")
    print(f"  isolation grid minimum   {iso.min():.1f}x   (chapter: 24)")
    check("isolation, benchmark", 3567.0, float(iso[ib, jb]), 2e-2)
    check("isolation, grid min", 24.0, float(iso.min()), 3e-2)

    # scope of the tau_QSS floor: is the min point inside the analysed set?
    print("\n  -- scope of the tau_QSS floor --")
    print(f"  M at the tau_QSS-min point = {M[e_, f_]:.4g}")
    print(f"  window_ok requires M > 900 -> point is "
          f"{'INSIDE' if M[e_, f_] > 900 else 'OUTSIDE'} the analysed set")
    n_ok = int((M > 900).sum())
    print(f"  points with M > 900: {n_ok} of {n_pts}")
    if n_ok:
        print(f"  tau_QSS min over M>900 subset = {tq[M > 900].min():.6e} s"
              f"   <- compare with the chapter's 1.18e-6 s")
    OPEN.append("tau_QSS floor: if the M>900 subset reproduces 1.18e-6 s, then "
                "Eq. (3.14) mixes scopes (M over all 400, tau_QSS over the "
                "analysed subset) and needs a clause saying so.")

    # ---------------------------------------------------------------- 3. M convention
    print()
    print("=" * 78)
    print("3. THE M CONVENTION  (Sec 3.3, 'Which operator M belongs to')")
    print("=" * 78)
    print(f"  M^- = M[L(Te[{ib}],ne[{jb}])]  = {M[ib, jb]:.5g}   (chapter: 9982)")
    check("M at benchmark", 9982.0, float(M[ib, jb]), 2e-3)
    step_pct = 100.0 * (Te[ib + 1] / Te[ib] - 1.0)
    print(f"  one Te grid step = +{step_pct:.3f}%   -> Te[{ib+1}] = {Te[ib+1]:.4f} eV")
    print(f"  M^+ = M[L(Te[{ib+1}],ne[{jb}])] = {M[ib+1, jb]:.5g}   (chapter: 8243)")
    check("M one grid step up", 8243.0, float(M[ib + 1, jb]), 2e-3)
    slope = np.log(M[ib + 1, jb] / M[ib, jb]) / np.log(Te[ib + 1] / Te[ib])
    check("dlnM/dlnTe local", -4.0, float(slope), 1e-1)
    print(f"  => the post-step operator IS grid point [{ib+1},{jb}], already "
          f"inside Eq. (3.14).")
    print(f"     That is the argument against the post-step convention.")

    print("\n  -- the +0.6 eV step: is it a grid point at all? --")
    t_target = float(Te[ib]) + 0.6
    k = int(np.argmin(np.abs(Te - t_target)))
    off = 100.0 * (Te[k] / t_target - 1.0)
    print(f"  Te[{ib}] + 0.6 = {t_target:.4f} eV")
    print(f"  nearest grid point Te[{k}] = {Te[k]:.4f} eV, off by {off:+.3f}%")
    on_grid = abs(off) < 1e-6
    print(f"  exactly on the grid? {on_grid}")
    print(f"  M there = {M[k, jb]:.5g}   (notation doc records 4856 for '+0.6 eV')")
    if not on_grid:
        OPEN.append(f"The '+0.6 eV' operator is not a grid point (nearest Te[{k}] "
                    f"is {off:+.3f}% away). Find where that case is built: does it "
                    f"index L_grid, interpolate L, or rebuild rates? The M^-/M^+ "
                    f"notation presumes L^+ has known provenance.")

    # ---------------------------------------------------------------- 4. summary
    print()
    print("=" * 78)
    print("4. SUMMARY")
    print("=" * 78)
    if FAILURES:
        print(f"  {len(FAILURES)} DISAGREEMENT(S) with the chapter as written:")
        for x in FAILURES:
            print(f"    - {x}")
    else:
        print("  No disagreement with any value the chapter asserts.")
    if OPEN:
        print(f"\n  {len(OPEN)} question(s) this script cannot settle:")
        for x in OPEN:
            print(f"    - {x}")
    print("\n  Report only. Nothing was written.")


if __name__ == "__main__":
    main()