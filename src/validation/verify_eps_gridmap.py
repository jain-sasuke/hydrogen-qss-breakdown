"""
verify_eps_gridmap.py
=====================
Settles one question before Chapter 5 is drafted: WHERE does the CRE error
live, and does the headline number survive the optical-thickness restriction
that Section 3.5.3 now imposes?

Section 3.5.3 restricts the quantitative results to Te >~ 2 eV, because the
Lyman-alpha escape factor falls to 3e-5 along the Te = 1 eV edge and the bare
Einstein coefficients in L are wrong there.  If the "at least 38.7%" headline
sits at Te = 1 eV, it is inside the excluded region and Chapter 5 cannot lead
with it as written.

DEFINITIONS, taken from notation_and_definitions.md Sec. 7:

    eps_step     = | R^CRE_old / R^CRE_new - 1 |
    eps_plateau  = | R^PE      / R^CRE_new - 1 |

with, at each grid point and for a temperature step of k grid intervals,

    L_old = L[i, j],       L_new = L[i+k, j]
    n^CRE = solve(L, -S)                        (CR equilibrium of that L)
    R^CRE = n_3 / n_4      from n^CRE
    n_F^PE = a_new * n_1s^old + c_new           (excited manifold equilibrated
                                                 under the NEW operator to the
                                                 OLD, unmoved ground state)
    a_new = -L_FF_new^-1 L_FS_new,   c_new = -L_FF_new^-1 S_F_new

This is the analytic partial-equilibrium construction of Chapter 3, not a time
integration: it gives the plateau value directly, which is what the headline
number is.

*** CROSS-CHECK BEFORE USING ***
If the repository already contains a plateau grid map, compare against it
before quoting anything from here.  Two files describing the same quantity
with nothing to distinguish them is how three retractions happened.  This
script computes eps_plateau from the definition above; if the existing map
used a different R^PE, the numbers will differ and the DIFFERENCE is the
finding, not either number.

Report only.  Writes nothing.

    python src/validation/verify_eps_gridmap.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "validation", HERE.parent / "analysis"):
    sys.path.insert(0, str(p))
from cr_context import CRContext  # noqa: E402

TE_FLOOR = 2.0          # Sec 3.5.3 optical-thickness restriction, eV
NE_FLOOR = 1.0e14       # lowest citable ITER divertor density, cm^-3
                        # (Guillemaut 2011: 1e20-1e21 m^-3, detached)
STEPS = (1, 4)          # grid intervals: +4.81% and +20.68%


def main() -> None:
    ctx = CRContext.load()
    root = ctx.root
    L, Te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    S = np.load(root / "data/processed/cr_matrix/S_grid.npy")
    nv = np.asarray(ctx.n_values)
    g = int(ctx.ground_index)
    E = np.array([k for k in range(ctx.n_states) if k != g])
    pos = {s: k for k, s in enumerate(E)}
    idx = {m: [pos[s] for s in np.where(nv == m)[0]] for m in (3, 4)}
    idx_full = {m: np.where(nv == m)[0] for m in (3, 4)}
    nT, nN = len(Te), len(ne)

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    for rel in ("data/processed/cr_matrix/L_grid.npy",
                "data/processed/cr_matrix/S_grid.npy"):
        f = root / rel
        print(f"  {rel}\n      sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")
    print(f"  grid {nT} x {nN};  Te floor for the restricted maximum "
          f"= {TE_FLOOR} eV (Sec 3.5.3)")
    print(f"  shells: n=3 has {len(idx[3])} sublevels, n=4 has {len(idx[4])}")

    print()
    print("  EXPECTED, stated first:")
    print("    - eps_plateau > eps_step at essentially every point (Ch. 5 claims")
    print("      680/680 for the small step); refuted if the count is not near 100%")
    print("    - the unrestricted maximum sits at or near the Te = 1 eV edge")
    print("    - the question is whether the RESTRICTED maximum is still tens of")
    print("      per cent. If it collapses to single digits, the thesis headline")
    print("      number changes and Chapter 5's opening changes with it.")

    for k in STEPS:
        dpct = 100.0 * (Te[k] / Te[0] - 1.0)
        print()
        print("=" * 78)
        print(f"STEP = {k} grid interval(s)  (+{dpct:.2f}% in Te)")
        print("=" * 78)

        eps_p = np.full((nT, nN), np.nan)
        eps_s = np.full((nT, nN), np.nan)

        for i in range(nT - k):
            for j in range(nN):
                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(L[i + k, j], -S[i + k, j])

                Ln = L[i + k, j]
                LEE = Ln[np.ix_(E, E)]
                a = np.linalg.solve(LEE, -Ln[np.ix_(E, [g])].ravel())
                c = np.linalg.solve(LEE, -S[i + k, j][E])
                nF_pe = a * n_old[g] + c

                R_pe = nF_pe[idx[3]].sum() / nF_pe[idx[4]].sum()
                R_old = n_old[idx_full[3]].sum() / n_old[idx_full[4]].sum()
                R_new = n_new[idx_full[3]].sum() / n_new[idx_full[4]].sum()

                eps_p[i, j] = abs(R_pe / R_new - 1.0)
                eps_s[i, j] = abs(R_old / R_new - 1.0)

        ok = ~np.isnan(eps_p)
        hot = ok & (Te[:, None] >= TE_FLOOR)

        amp = int((eps_p[ok] > eps_s[ok]).sum())
        print(f"  eps_plateau > eps_step at {amp} of {int(ok.sum())} points"
              f"  ({100*amp/ok.sum():.1f}%)")

        # The citable ITER divertor range (Guillemaut et al. 2011, Sec. 4):
        # Te = 1-10 eV, ne = 1e20-1e21 m^-3 = 1e14-1e15 cm^-3. The ridge at
        # ne ~ 2e13 lies a factor 7 BELOW that, so the unrestricted maximum
        # cannot be quoted as an ITER divertor number.
        iter_box = hot & (ne[None, :] >= NE_FLOOR)
        for label, mask in (("ALL points", ok),
                            (f"Te >= {TE_FLOOR} eV", hot),
                            (f"+ ne >= {NE_FLOOR:.0e}", iter_box)):
            e = np.where(mask, eps_p, -np.inf)
            a_, b_ = np.unravel_index(int(np.argmax(e)), e.shape)
            vals = eps_p[mask]
            print(f"  {label:>16s}: n={mask.sum():3d}  "
                  f"max {100*vals.max():7.3f}%  at [{a_},{b_}] "
                  f"(Te={Te[a_]:.4g} eV -> {Te[a_+k]:.4g}, ne={ne[b_]:.4g})   "
                  f"median {100*np.median(vals):6.3f}%")

        # the ridge: where in ne does the maximum sit, temperature by temperature
        j_ridge = int(np.argmax(np.nanmax(eps_p, axis=0)))
        print(f"  ridge column: ne = {ne[j_ridge]:.4g} cm^-3 "
              f"(j={j_ridge});  lowest citable ITER divertor density "
              f"{NE_FLOOR:.0e} is j={int(np.argmax(ne >= NE_FLOOR))}")
        print("  ridge, restricted to the defensible range:")
        print(f"    {'Te (eV)':>9s}  {'argmax ne':>11s}  {'eps_plateau':>11s}")
        rows = [i for i in range(nT - k) if Te[i] >= TE_FLOOR]
        for i in rows[::max(1, len(rows) // 8)]:
            j = int(np.argmax(eps_p[i]))
            print(f"    {Te[i]:9.3f}  {ne[j]:11.3e}  {100*eps_p[i, j]:10.3f}%")

    print()
    print("=" * 78)
    print("READING")
    print("=" * 78)
    print("  If the restricted maximum is still tens of per cent, Chapter 5")
    print("  keeps its headline and gains a caveat naming the excluded corner.")
    print("  If it is not, the headline number is the one quoted for the")
    print("  restricted range, and the cold-edge value is reported separately")
    print("  as asymptotic behaviour rather than as a prediction.")
    print()
    print("  Either way: compare the ALL-points maximum against the existing")
    print("  plateau grid map before quoting either. A disagreement is a")
    print("  finding about the definitions, not a number to average.")
    print()
    print("  Report only. Nothing was written.")


if __name__ == "__main__":
    main()