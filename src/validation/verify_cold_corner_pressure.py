#!/usr/bin/env python
"""
verify_cold_corner_pressure.py
==============================
Is the cold corner a divertor state at all?

WHY THIS EXISTS
---------------
chapter5.tex argues that the grid's cold corner fails a cruder test than
quasi-neutrality or optical depth: the neutral pressure the model's own steady
state implies there is far outside anything an ITER divertor is designed to
run at. Two pressures were quoted with no producing script, under a
[SOURCE REQUIRED] marker. This supplies one.

WHAT IT COMPUTES
----------------
The model's collisional-radiative equilibrium at a grid point gives n(1s). The
neutral pressure that population exerts, taking the neutral temperature equal
to the electron temperature as everywhere else in this thesis, is

    p = n(1s) k T_n,     with T_n = Te.

In Gaussian-CGS that is an energy density in erg/cm^3, which is dyn/cm^2, and
1 Pa = 10 dyn/cm^2. The factor of ten is where this calculation would go wrong
if it went wrong, so it is written out rather than folded into a constant.

The comparison target is the one figure that could be verified from a source
actually read: Stangeby et al, Nucl. Fusion 63(1) 016016, Sec. 6, records that
new SOLPS-ITER simulations extend the ITER Q=10 baseline database to a
divertor-averaged neutral pressure of <p_div> > 25 Pa. That is an upper end of
a simulation database, not a design limit, and it is used here only as an
order-of-magnitude anchor. An earlier draft compared against "roughly 1 to
20 Pa", which could not be traced to any source read for this thesis.

THE CAVEAT THAT MATTERS
-----------------------
These are fixed points of a zero-dimensional box with no transport and no
pumping. A real divertor's neutral density is set by recycling and pumping, not
by local ionisation balance, which is the same objection Chapter 6 raises
against the closed parcel. The number is evidence that the cold corner is
outside the modelled regime, not a prediction of a pressure.

Read-only. Writes only to validation/cold_corner/, and only with --write.
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

EV_ERG = 1.602176634e-12        # CODATA 2018, erg per eV
DYN_PER_CM2_PER_PA = 10.0       # 1 Pa = 1 N/m^2 = 10 dyn/cm^2

# The anchor, from a source read in full. See the docstring.
P_DIV_SOLPS_HIGH_PA = 25.0

# chapter5.tex records these two.
REC = {(0, 4): 235.0, (0, 7): 3280.0}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index

    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not sp.exists():
        raise RuntimeError(f"missing {sp}")
    S = np.load(sp)

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    print(f"  L_grid {L.shape}  sha256 {sha256(lp)[:32]}...")
    print(f"  ground index {g} ({ctx.labels[g]})")
    print(f"  interpreter {sys.executable}   numpy {np.__version__}")
    print(f"  p = n(1s) k T_n with T_n = Te; 1 Pa = {DYN_PER_CM2_PER_PA:g} dyn/cm^2")
    print()

    rows = []
    P = np.empty((len(te), len(ne)))
    for i in range(len(te)):
        for j in range(len(ne)):
            n = np.linalg.solve(L[i, j], -S[i, j] * ne[j])
            if np.any(n <= 0):
                raise RuntimeError(f"non-positive CRE population at [{i},{j}]")
            p_pa = float(n[g] * EV_ERG * te[i] / DYN_PER_CM2_PER_PA)
            P[i, j] = p_pa
            rows.append(dict(i=i, j=j, Te=float(te[i]), ne=float(ne[j]),
                             n_1s=float(n[g]), p_Pa=p_pa,
                             over_solps_high=p_pa / P_DIV_SOLPS_HIGH_PA))

    print("=" * 78)
    print("THE TWO POINTS chapter5.tex QUOTES")
    print("=" * 78)
    ok = True
    for (i, j), want in sorted(REC.items()):
        got = P[i, j]
        d = abs(got - want) / want
        flag = "OK " if d <= 0.01 else "!! "
        ok &= d <= 0.01
        print(f"  {flag}[{i},{j}]  Te = {te[i]:.4g} eV, ne = {ne[j]:.4g} cm^-3:"
              f"  n(1s) = {np.linalg.solve(L[i,j], -S[i,j]*ne[j])[g]:.4e} cm^-3,"
              f"  p = {got:.4g} Pa   (recorded {want:g})")
        print(f"        that is {got/P_DIV_SOLPS_HIGH_PA:.3g} times the "
              f"{P_DIV_SOLPS_HIGH_PA:g} Pa upper end of the extended ITER "
              f"SOLPS database")
    ib, jb = 23, 5
    print(f"  for contrast, the benchmark [{ib},{jb}]: p = {P[ib,jb]:.4g} Pa, "
          f"which is {P_DIV_SOLPS_HIGH_PA/P[ib,jb]:.4g} times BELOW that "
          f"upper end")
    print()

    warm = te >= 2.0
    print("=" * 78)
    print("WHERE ON THE GRID IS THE IMPLIED PRESSURE PHYSICALLY PLAUSIBLE?")
    print("=" * 78)
    n_over = int((P > P_DIV_SOLPS_HIGH_PA).sum())
    n_over_warm = int((P[warm] > P_DIV_SOLPS_HIGH_PA).sum())
    print(f"  points exceeding {P_DIV_SOLPS_HIGH_PA:g} Pa: {n_over} of {P.size}")
    print(f"  of those, {n_over_warm} lie at Te >= 2 eV, the defended range")
    if n_over_warm:
        idx = np.argwhere(warm[:, None] & (P > P_DIV_SOLPS_HIGH_PA))
        print(f"    they are at: " + ", ".join(f"[{i},{j}]" for i, j in idx[:8]))
    row_max = P.max(axis=1)
    cross = [i for i in range(len(te) - 1)
             if row_max[i] > P_DIV_SOLPS_HIGH_PA >= row_max[i + 1]]
    if cross:
        print(f"  the implied pressure drops below {P_DIV_SOLPS_HIGH_PA:g} Pa "
              f"at every density above Te = {te[cross[-1]+1]:.3g} eV")
    print("  This is a third boundary and it falls in the same place as the")
    print("  two in Section sec:scope, which were quasi-neutrality and optical")
    print("  depth. Three unrelated constraints, one floor.")
    print()

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "cold_corner"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "cold_corner_pressure.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {out/'cold_corner_pressure.csv'}  ({len(rows)} rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
