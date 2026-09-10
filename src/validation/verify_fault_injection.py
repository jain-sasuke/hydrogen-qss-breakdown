#!/usr/bin/env python
"""
verify_fault_injection.py
=========================
Reproduce the fault-injection table, and stamp the grid point it belongs to.

WHY THIS EXISTS
---------------
chapter4.tex Table tab:faultinject reports four deliberate faults injected into
the assembly of L and the conservation residual after each. Three faults that
move individual matrix elements by up to 2.7e11 s^-1 leave the residual
unchanged; only an A-versus-gamma inconsistency is caught. That table is the
evidence for the chapter's claim that the conservation gate is nearly
tautological, so it matters.

It carried an [UNVERIFIED] because CHANGE_REPORT.md section 4.2 records the
residuals without recording WHICH GRID POINT they were evaluated at. A residual
is not a property of the model; it is a property of the model at a point. Three
of the four max|dL| values scale with n_e, so without the point the table cannot
be reproduced or checked.

WHAT THIS DOES
--------------
1. Rebuilds L through the pipeline's own build_L, with the pipeline's own rate
   arrays, and checks it reproduces the stored L_grid at the same point. If it
   does not, the assembly has drifted and nothing below means anything.
2. Sweeps all 400 grid points looking for the one that reproduces the recorded
   max|dL| values, to RECOVER the point rather than replace it.
3. Reports the table at that point if found, and at the benchmark otherwise.

THE FOUR FAULTS, as chapter 4 names them
----------------------------------------
  a  K_exc(1s -> 2p) multiplied by ten
  b  all de-excitation deleted
  c  the excitation array transposed
  d  A_pq halved with gamma_p left alone

Faults a to c corrupt the physics while preserving the construction: the
diagonal is built as minus the column sum of the same array the check then sums,
so the identity is imposed rather than tested. Fault d breaks the construction
by making the off-diagonal A and the diagonal gamma inconsistent, and it is the
only one the check can see. That is the point of the table.

THE RESIDUAL
------------
    max_q | ( sum_p L[p,q] + Q_q n_e ) / ( Q_q n_e ) |

Everything leaving state q goes to another bound state or to the continuum, so
the column sum of L must equal minus the ionisation rate out of q. Chapter 3
states the same quantity over all 400 points.

Read-only. Writes only to validation/fault_injection/, and only with --write.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)
sys.path.insert(0, str(ROOT / "src" / "rates"))

# chapter4.tex Table tab:faultinject. Recomputed, not trusted.
REC = [
    ("none (baseline)",                      0.0,     2.238e-12),
    ("K_exc(1s->2p) x ten",                  6.9e5,   2.238e-12),
    ("all de-excitation deleted",            1.9e11,  2.171e-12),
    ("excitation array transposed",          2.7e11,  1.837e-12),
    ("A_pq halved, gamma_p left alone",      3.1e8,   3.63e1),
]


def residual(L: np.ndarray, K_ion_col: np.ndarray, ne: float) -> float:
    """max relative column-sum residual. See the docstring."""
    leak = K_ion_col * ne
    if np.any(leak <= 0):
        raise RuntimeError("a state has non-positive ionisation rate; the "
                           "relative residual is undefined there")
    return float(np.abs((L.sum(axis=0) + leak) / leak).max())


def faulted(rates: dict, which: str) -> dict:
    """Return a copy of the rate dict with one fault injected."""
    r = {k: (v.copy() if isinstance(v, np.ndarray) else v)
         for k, v in rates.items()}
    if which == "none":
        pass
    elif which == "exc10":
        # 1S is index 0 and 2P is index 2 in the pipeline's own ordering; both
        # are asserted against the state index by the caller.
        r["K_exc_full"][0, 2, :] *= 10.0
    elif which == "nodeexc":
        r["K_deexc_full"][...] = 0.0
    elif which == "transpose":
        r["K_exc_full"] = np.swapaxes(r["K_exc_full"], 0, 1).copy()
    elif which == "Ahalf":
        for k in ("A_resolved", "A_bund_res", "A_bund_bund"):
            r[k] = r[k] / 2.0
        # gamma_resolved and gamma_bundled deliberately left alone
    else:
        raise RuntimeError(f"unknown fault {which!r}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the 400-point search for the original point")
    a = ap.parse_args()

    from assemble_cr_matrix import build_L, load_rates, TE_GRID   # noqa: E402

    ctx = CRContext.load()
    ctx.validate()
    te, ne_grid, L_grid = ctx.te_grid, ctx.ne_grid, ctx.L_grid
    labels = ctx.labels
    if labels[0].upper() != "1S" or labels[2].upper() != "2P":
        raise RuntimeError(
            f"state ordering is not what the fault injection assumes: index 0 "
            f"is {labels[0]} and index 2 is {labels[2]}, expected 1S and 2P. "
            f"The K_exc(1s->2p) fault would corrupt the wrong transition.")
    rates = load_rates()
    if not np.allclose(np.asarray(TE_GRID), te, rtol=1e-12):
        raise RuntimeError("assemble_cr_matrix.TE_GRID and the stored "
                           "Te_grid_L.npy disagree; the Te index is unsafe")

    print("=" * 78)
    print("PROVENANCE AND WIRING")
    print("=" * 78)
    print(f"  rebuilding L through assemble_cr_matrix.build_L")
    print(f"  interpreter {sys.executable}   numpy {np.__version__}")
    ib, jb = 23, 5
    L_rebuilt = build_L(ib, float(ne_grid[jb]), rates)
    d = np.abs(L_rebuilt - L_grid[ib, jb]).max() / np.abs(L_grid[ib, jb]).max()
    print(f"  rebuilt L[{ib},{jb}] against the stored L_grid: relative "
          f"max|diff| {d:.3e}")
    if d > 1e-12:
        raise RuntimeError(
            f"the rebuild does not reproduce the stored matrix ({d:.3e}). The "
            f"assembly has changed since L_grid was written, and every residual "
            f"below would belong to a different model.")
    print("  The rebuild is the stored matrix. Faults below are injected into")
    print("  the same assembly that produced it.")
    print()

    def table_at(i: int, j: int, verbose: bool = True):
        base = build_L(i, float(ne_grid[j]), rates)
        Kion = rates["K_ion_final"][:, i]
        rows = []
        for (name, key) in (("none (baseline)", "none"),
                            ("K_exc(1s->2p) x ten", "exc10"),
                            ("all de-excitation deleted", "nodeexc"),
                            ("excitation array transposed", "transpose"),
                            ("A_pq halved, gamma_p left alone", "Ahalf")):
            Lf = build_L(i, float(ne_grid[j]), faulted(rates, key))
            dmax = float(np.abs(Lf - base).max())
            res = residual(Lf, Kion, float(ne_grid[j]))
            rows.append((name, dmax, res))
        return rows

    # ---- recover the original point --------------------------------------
    best = None
    if not a.no_sweep:
        print("=" * 78)
        print("SEARCHING ALL 400 POINTS FOR THE ONE THE TABLE BELONGS TO")
        print("=" * 78)
        print("  Matching on the three recorded max|dL| values, which scale")
        print("  with n_e and so identify the point if it is on the grid.")
        want = np.array([REC[1][1], REC[2][1], REC[3][1]])
        for i in range(len(te)):
            for j in range(len(ne_grid)):
                rows = table_at(i, j, verbose=False)
                got = np.array([rows[1][1], rows[2][1], rows[3][1]])
                err = float(np.abs(got / want - 1.0).max())
                if best is None or err < best[0]:
                    best = (err, i, j, rows)
        err, i0, j0, rows0 = best
        print(f"  closest point: [{i0},{j0}], Te = {te[i0]:.4g} eV, "
              f"ne = {ne_grid[j0]:.4g} cm^-3")
        print(f"  worst relative disagreement on the three max|dL|: {err*100:.2f}%")
        if err < 0.05:
            print("  RECOVERED. The table belongs to this point; the recorded")
            print("  values round to two significant figures, which is the")
            print("  precision the table was printed at.")
        else:
            print("  NOT RECOVERED at better than 5 percent. The table is")
            print("  reported below at the benchmark instead, per the todo.")
        print()

    use = (i0, j0) if (best and best[0] < 0.05) else (ib, jb)
    rows = table_at(*use)
    print("=" * 78)
    print(f"THE TABLE, AT GRID POINT [{use[0]},{use[1]}]: "
          f"Te = {te[use[0]]:.4g} eV, ne = {ne_grid[use[1]]:.4g} cm^-3")
    print("=" * 78)
    print(f"  {'injected fault':<34} {'max|dL| (1/s)':>14} {'residual':>12}"
          f"   {'recorded':>10}")
    ok = True
    for (name, dmax, res), rec in zip(rows, REC):
        rd = abs(res / rec[2] - 1.0)
        flag = "OK " if rd < 0.02 else "!! "
        ok &= rd < 0.02
        print(f"  {flag}{name:<31} {dmax:>14.3e} {res:>12.4e}   "
              f"{rec[2]:>10.3e}")
    print()
    caught = [r for r in rows if r[2] > 1e-6]
    print(f"  faults detected by the conservation check: {len(caught)} of 4")
    for c in caught:
        print(f"    {c[0]}  residual {c[2]:.4g}")
    print("  Faults that preserve the construction are invisible to it, which")
    print("  is the chapter's point and is now reproducible at a named point.")
    print()
    print("VERDICT:", "the table reproduces" if ok
          else "AT LEAST ONE RESIDUAL DISAGREES, see the !! rows")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "fault_injection"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "fault_injection.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["i", "j", "Te_eV", "ne_cm3", "fault",
                        "max_abs_dL", "residual"])
            for (name, dmax, res) in rows:
                w.writerow([use[0], use[1], f"{te[use[0]]:.6g}",
                            f"{ne_grid[use[1]]:.6g}", name,
                            f"{dmax:.8e}", f"{res:.8e}"])
        print(f"\n  wrote {out/'fault_injection.csv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
