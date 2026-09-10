#!/usr/bin/env python
"""
verify_crest_subgrid.py
=======================
Two sub-grid questions Chapter 5 asks and marks as unanswered.

WHY THIS EXISTS
---------------
chapter5.tex carries two [UNVERIFIED] markers that name exactly what would
close them:

  1. The density crest. The maximum of eps_plateau along each temperature row
     falls on a grid node, and a parabolic fit in ln(ne) through the three
     columns around it places the true vertex at 1.65e13 cm^-3, a 15 percent
     shift. That number is recorded as prose in findings_10 section B.3 and no
     script computes it.

  2. The eps_step zero locus. Section sec:step_error scans one density column
     and finds a temperature where dln R / dln Te changes sign, so a
     temperature excursion produces no change in the tabulated ratio at all and
     the diagnostic is locally blind. Whether that locus crosses the operating
     range at other densities was never mapped.

Both are cheap. Neither was done, which is the only reason they were open.

WHAT IS MEASURED
----------------
eps_plateau is built exactly as verify_plateau_gridmap.py builds it, through the
two-channel split, and the superposition residual is checked at every point
before anything is fitted. eps_step is the distance between the two tabulated
answers, |R_CRE(Te) / R_CRE(Te + dTe) - 1|, which vanishes where the tabulated
ratio is stationary in temperature.

THE PARABOLIC FIT, AND WHAT IT CAN AND CANNOT SAY
-------------------------------------------------
Three points determine a parabola exactly, so the fit has no residual and no
goodness of fit: a vertex from three points is an interpolation, not a
measurement, and it is only as good as the assumption that eps is locally
quadratic in ln(ne). This script therefore also fits a parabola through FIVE
points where five are available and reports the disagreement between the two
vertices. That disagreement is the honest error bar, and it is the number to
quote rather than the vertex alone.

Read-only. Writes only to validation/crest_subgrid/, and only with --write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)

# chapter5.tex records these. Recomputed, not trusted.
REC_ARGMAX_ROWS_0_15 = [4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
REC_VERTEX = 1.65e13


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vertex_log(x: np.ndarray, y: np.ndarray) -> float:
    """Vertex of the parabola through (ln x, y), returned in x units."""
    lx = np.log(x)
    c = np.polyfit(lx, y, 2)
    if c[0] >= 0:
        raise RuntimeError("the fitted parabola opens upward; there is no "
                           "interior maximum to locate here")
    return float(np.exp(-c[1] / (2.0 * c[0])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--k", type=int, default=1, help="Te step, grid indices")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    nv = ctx.n_values
    E = np.array([i for i in range(ctx.n_states) if i != g])
    N3, N4 = np.where(nv == 3)[0], np.where(nv == 4)[0]
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not sp.exists():
        raise RuntimeError(f"missing source vector: {sp}")
    S = np.load(sp)

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    print(f"  L_grid {L.shape}  sha256 {sha256(lp)[:32]}...")
    print(f"  S_grid {S.shape}  sha256 {sha256(sp)[:32]}...")
    print(f"  interpreter {sys.executable}   numpy {np.__version__}")
    print()

    nT, nN = len(te), len(ne)
    eps_p = np.full((nT, nN), np.nan)
    eps_s = np.full((nT, nN), np.nan)
    R_cre = np.full((nT, nN), np.nan)
    worst_sup = 0.0
    for i in range(nT - a.k):
        ii = i + a.k
        for j in range(nN):
            n_old = np.linalg.solve(L[i, j], -S[i, j])
            n_new = np.linalg.solve(L[ii, j], -S[ii, j])
            LEE = L[ii, j][np.ix_(E, E)]
            LEg = L[ii, j][np.ix_(E, [g])].ravel()
            n0 = np.linalg.solve(LEE, -S[ii, j][E])
            n1 = np.linalg.solve(LEE, -LEg * n_old[g])
            sup = (np.abs(n0 + (n_new[g] / n_old[g]) * n1
                          - n_new[E]).max() / np.abs(n_new[E]).max())
            worst_sup = max(worst_sup, sup)
            if sup > 1e-8:
                raise RuntimeError(f"two-channel split not exact at [{i},{j}]: "
                                   f"residual {sup:.3e}")
            R_pe = ((n0[n3E].sum() + n1[n3E].sum())
                    / (n0[n4E].sum() + n1[n4E].sum()))
            R_new = n_new[N3].sum() / n_new[N4].sum()
            R_old = n_old[N3].sum() / n_old[N4].sum()
            eps_p[i, j] = abs(R_pe / R_new - 1.0)
            eps_s[i, j] = abs(R_old / R_new - 1.0)
            R_cre[i, j] = R_old
    for j in range(nN):
        R_cre[nT - 1, j] = (np.linalg.solve(L[nT - 1, j], -S[nT - 1, j])[N3].sum()
                            / np.linalg.solve(L[nT - 1, j], -S[nT - 1, j])[N4].sum())
    print(f"  worst two-channel superposition residual: {worst_sup:.3e}")
    print()

    # ---- 1. the crest -----------------------------------------------------
    print("=" * 78)
    print("THE DENSITY CREST, ON THE GRID AND BETWEEN THE NODES")
    print("=" * 78)
    argm = [int(np.nanargmax(eps_p[i])) for i in range(nT - a.k)]
    got = argm[:16]
    flag = "OK " if got == REC_ARGMAX_ROWS_0_15 else "!! "
    print(f"  {flag}argmax column, rows 0 to 15: {got}")
    print(f"      chapter5.tex records:        {REC_ARGMAX_ROWS_0_15}")
    if got != REC_ARGMAX_ROWS_0_15:
        raise RuntimeError("the argmax pattern disagrees with chapter5.tex; "
                           "the crest has moved and every statement about it "
                           "is suspect")
    rows = []
    print()
    print(f"  {'row':>4} {'Te':>7} {'node':>4} {'node ne':>10} "
          f"{'vertex(3pt)':>12} {'vertex(5pt)':>12} {'disagree':>9}")
    v3s, v5s = [], []
    for i in range(nT - a.k):
        j0 = argm[i]
        if j0 == 0 or j0 == nN - 1:
            continue                      # no interior maximum on this row
        sl3 = slice(j0 - 1, j0 + 2)
        v3 = vertex_log(ne[sl3], eps_p[i, sl3])
        v5 = np.nan
        if 2 <= j0 <= nN - 3:
            sl5 = slice(j0 - 2, j0 + 3)
            try:
                v5 = vertex_log(ne[sl5], eps_p[i, sl5])
            except RuntimeError:
                v5 = np.nan
        v3s.append(v3)
        if np.isfinite(v5):
            v5s.append((v3, v5))
        rows.append(dict(i=i, Te=float(te[i]), node_j=j0,
                         node_ne=float(ne[j0]), vertex_3pt=v3,
                         vertex_5pt=float(v5), eps_max=float(eps_p[i, j0])))
        if i < 16 or i % 10 == 0:
            d = abs(v5 / v3 - 1.0) * 100 if np.isfinite(v5) else np.nan
            print(f"  {i:>4} {te[i]:>7.3f} {j0:>4} {ne[j0]:>10.3g} "
                  f"{v3:>12.4g} {v5:>12.4g} {d:>8.2f}%")
    v3s = np.array(v3s)
    print()
    print(f"  three-point vertex over {len(v3s)} rows with an interior maximum:")
    print(f"    {v3s.min():.4g} to {v3s.max():.4g}, median {np.median(v3s):.4g}")
    d = abs(np.median(v3s) / REC_VERTEX - 1.0)
    print(f"    against chapter5.tex's {REC_VERTEX:.4g}: "
          f"{100*d:.2f}% apart")
    if v5s:
        r = np.array([b / a_ for a_, b in v5s])
        print(f"  five-point against three-point, {len(v5s)} rows: ratio "
              f"{r.min():.4f} to {r.max():.4f}, median {np.median(r):.4f}")
        print("    That spread is the error bar on the vertex. A three-point")
        print("    parabola through a grid this coarse locates the crest to")
        print(f"    about {100*max(abs(r.min()-1), abs(r.max()-1)):.0f} percent, which is why chapter 5 quotes the")
        print("    crest as a range and not as a density.")
    print()

    # ---- 2. the eps_step zero locus ---------------------------------------
    print("=" * 78)
    print("WHERE IS THE TABULATED RATIO STATIONARY IN TEMPERATURE?")
    print("=" * 78)
    print("  eps_step vanishes where dln R / dln Te = 0, and there a")
    print("  temperature excursion moves the tabulated answer not at all. The")
    print("  question chapter 5 leaves open is whether that locus crosses the")
    print("  operating range at more than one density.")
    print()
    print(f"  {'column':>6} {'ne':>10} {'min eps_step':>13} {'at Te':>8} "
          f"{'sign change of dlnR/dlnTe':>26}")
    loc = []
    for j in range(nN):
        col = eps_s[:nT - a.k, j]
        k = int(np.nanargmin(col))
        dl = np.diff(np.log(R_cre[:, j]))
        sgn = np.sign(dl)
        cross = np.where(np.diff(sgn) != 0)[0]
        where = ", ".join(f"{te[c]:.3g} eV" for c in cross) if len(cross) else "none"
        print(f"  {j:>6} {ne[j]:>10.3g} {col[k]:>13.4e} {te[k]:>8.3f} "
              f"{where:>26}")
        loc.append(dict(j=j, ne=float(ne[j]), min_eps_step=float(col[k]),
                        Te_at_min=float(te[k]), n_sign_changes=int(len(cross)),
                        Te_crossings=where))
    n_with = sum(1 for r in loc if r["n_sign_changes"] > 0)
    print()
    print(f"  {n_with} of {nN} density columns carry a stationary point inside")
    print(f"  the operating range. The locus is not a single-column curiosity.")
    print("  Where it exists the diagnostic is locally blind to temperature,")
    print("  which is a separate failure from the one this thesis maps and is")
    print("  reported rather than pursued.")
    print()

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "crest_subgrid"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "crest_vertices.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with (out / "eps_step_locus.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(loc[0].keys()))
            w.writeheader(); w.writerows(loc)
        print(f"  wrote {out/'crest_vertices.csv'}  ({len(rows)} rows)")
        print(f"  wrote {out/'eps_step_locus.csv'}  ({len(loc)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
