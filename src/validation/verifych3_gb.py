"""
verify_ch3_groupB.py
====================
Settles the three Chapter 3 claims that no amount of re-reading can settle,
raised independently by two external reviews.

  B1  Is M >> 1 a sufficient condition for QSS?
      The quasi-static deviation obeys  delta ~= L_FF^-1 * d(n_F^QSS)/dt,
      so  ||delta|| <= ||L_FF^-1|| ||dn/dt||.  The operative timescale is
      ||L_FF^-1||_2 = 1/sigma_min(L_FF), NOT 1/|lambda_1|.  For every matrix
      sigma_min <= min|lambda|, so the rigorous timescale is at least the
      spectral one and the true separation is at most M.  Question: by how
      much, and does the conclusion survive?

  B2  Section 3.3.5 reports that lambda_1(L) and lambda_min(L_FF) agree to
      0.0098% at the benchmark and 0.35% grid-wide, and cites this as
      independent evidence that the subsystems separate.  But 1/M = 0.01002%
      at the benchmark -- a 2% match.  If the agreement IS the O(1/M)
      block-decoupling correction then it is a measurement of the gap, not
      independent evidence for it.  Question: is (disagreement x M) O(1)
      across the grid?

  B3  The optically-thin assumption is undeclared where the A-coefficients
      enter, and the chapter now explains the b_1^CRE range by photons
      escaping.  Question: what is the Lyman-alpha escape factor Theta_P at
      the grid corners, and is "optically thin" defensible?

Report only.  Loads L_grid, S_grid and the state index; writes nothing;
repairs nothing.  Expected values are printed BEFORE measured ones so a
disagreement falsifies rather than invites rationalisation.

Platform: macOS/BSD, zsh.  Pure Python.

    python src/validation/verify_ch3_groupB.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "validation"))
sys.path.insert(0, str(HERE.parent / "analysis"))
from cr_context import CRContext  # noqa: E402

try:
    from escape_factor import escape_factor_slab, lyman_alpha_sigma0
    HAVE_ESCAPE = True
except ImportError as exc:  # do not substitute a stand-in
    HAVE_ESCAPE = False
    ESCAPE_ERR = str(exc)

CHI_H = 13.605693122994
H_PLANCK = 6.62607015e-34
M_E = 9.1093837015e-31
EV_TO_J = 1.602176634e-19


def saha_Z(p: int, te_ev: float) -> float:
    """Saha-Boltzmann coefficient of level p, cm^3, Te in eV."""
    return (1e6 * p ** 2
            * (H_PLANCK ** 2 / (2 * np.pi * M_E * EV_TO_J * te_ev)) ** 1.5
            * np.exp((CHI_H / p ** 2) / te_ev))


def main() -> None:
    ctx = CRContext.load()
    root = ctx.root
    L, Te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    S = np.load(root / "data/processed/cr_matrix/S_grid.npy")
    nT, nN = len(Te), len(ne)

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    for rel in ("data/processed/cr_matrix/L_grid.npy",
                "data/processed/cr_matrix/S_grid.npy"):
        f = root / rel
        print(f"  {rel}\n      sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")
    print(f"  grid {nT} x {nN} = {nT * nN} points")

    g = 0                                   # ground state index
    E = np.arange(1, L.shape[-1])           # excited block

    ib = int(np.argmin(np.abs(Te - 2.947)))
    jb = int(np.argmin(np.abs(ne - 1.389e14)))
    print(f"  benchmark [{ib},{jb}]  Te={Te[ib]:.4f} eV  ne={ne[jb]:.4e} cm^-3")

    # =================================================================== B1/B2
    print()
    print("=" * 78)
    print("B1  Is the spectral timescale a bound?   ||L_FF^-1|| vs 1/|lambda_1|")
    print("B2  Is the 0.0098% agreement independent evidence, or just 1/M?")
    print("=" * 78)
    print("  EXPECTED, stated first:")
    print("    B1  ratio = ||L_FF^-1|| |lambda_min(L_FF)| >= 1 at every point,")
    print("        by sigma_min <= min|lambda|.  If it is O(1-10) the QSS")
    print("        conclusion survives and only the wording changes.  If it is")
    print("        O(1e3) anywhere, M >> 1 does not establish QSS validity there.")
    print("    B2  disagreement x M ~ O(1) if the agreement is the O(1/M)")
    print("        block-decoupling correction.  Refuted if it is not O(1).")
    print()

    tau_spec = np.zeros((nT, nN))   # 1/|lambda_min(L_FF)|
    tau_bound = np.zeros((nT, nN))  # ||L_FF^-1||_2
    M_gr = np.zeros((nT, nN))
    disag = np.zeros((nT, nN))      # |lambda_1(L) - lambda_min(L_FF)| / |lambda_1(L)|

    for i in range(nT):
        for j in range(nN):
            e_full = np.sort(np.linalg.eigvals(L[i, j]).real)[::-1]
            lam0, lam1 = e_full[0], e_full[1]
            M_gr[i, j] = abs(lam1) / abs(lam0)

            LFF = L[i, j][np.ix_(E, E)]
            e_ff = np.sort(np.linalg.eigvals(LFF).real)[::-1]
            lam_ff = e_ff[0]                       # least negative of L_FF
            tau_spec[i, j] = 1.0 / abs(lam_ff)
            tau_bound[i, j] = 1.0 / np.linalg.svd(LFF, compute_uv=False).min()
            disag[i, j] = abs(lam1 - lam_ff) / abs(lam1)

    ratio = tau_bound / tau_spec
    a, b = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    print("  --- B1 ---")
    print(f"  ||L_FF^-1|| / (1/|lambda_min(L_FF)|):"
          f"  min {ratio.min():.3f}   median {np.median(ratio):.3f}"
          f"   max {ratio.max():.3f}")
    print(f"  worst at [{a},{b}]  Te={Te[a]:.4g} eV  ne={ne[b]:.4g} cm^-3")
    if ratio.min() < 0.999:
        print(f"  !! ratio below 1 somewhere ({ratio.min():.6f}) -- that would"
              f" contradict sigma_min <= min|lambda|; check the block indexing")
    M_eff = M_gr * tau_spec / tau_bound
    c, d = np.unravel_index(int(np.argmin(M_eff)), M_eff.shape)
    print(f"  effective separation M_eff = tau_QSS / ||L_FF^-1||:")
    print(f"      benchmark {M_eff[ib, jb]:.4g}   (spectral M = {M_gr[ib, jb]:.4g})")
    print(f"      grid min  {M_eff.min():.4g} at [{c},{d}] "
          f"(Te={Te[c]:.4g}, ne={ne[d]:.4g})   (spectral M min = {M_gr.min():.4g})")
    print(f"  READING: if M_eff grid-min >> 1 the QSS conclusion survives and")
    print(f"           Sec 3.4 needs 'separation indicator', not 'sufficient'.")

    print()
    print("  --- B2 ---")
    prod = disag * M_gr
    print(f"  disagreement at benchmark {100*disag[ib, jb]:.4f}%"
          f"   (chapter quotes 0.0098%)")
    print(f"  1/M at benchmark          {100/M_gr[ib, jb]:.4f}%")
    print(f"  grid max disagreement     {100*disag.max():.4f}%"
          f"   (chapter quotes 0.35%)")
    print(f"  (disagreement x M):  min {prod.min():.4g}"
          f"   median {np.median(prod):.4g}   max {prod.max():.4g}")
    print(f"  READING: O(1) across the grid means the agreement measures the")
    print(f"           gap rather than evidencing separability independently.")

    # ====================================================================== B3
    print()
    print("=" * 78)
    print("B3  Lyman-alpha escape factor at the grid corners")
    print("=" * 78)
    if not HAVE_ESCAPE:
        print(f"  HALT: could not import escape_factor ({ESCAPE_ERR}).")
        print("  No stand-in substituted. Fix the path and re-run.")
        return
    print("  EXPECTED: Theta_P ~ 1 would retire the objection. Theta_P << 1 at")
    print("  any analysed point means the optically-thin A-coefficients in")
    print("  Eqs. (3.1),(3.3),(3.4) are wrong there, and b_1^CRE with them.")
    print("  n_1s is taken PHYSICAL: n_1s = n[1s] * n_i with n_i = n_e.")
    print()
    corners = [(0, 0), (ib, jb), (nT - 1, nN - 1), (0, nN - 1), (nT - 1, 0)]
    print(f"  {'point':>9s} {'Te':>7s} {'ne':>10s} {'n_1s':>11s} {'sigma0':>10s}"
          f" {'tau_c(10cm)':>12s} {'Th(1cm)':>9s} {'Th(10cm)':>9s} {'Th(100cm)':>10s}")
    for (i, j) in corners:
        n_sol = np.linalg.solve(L[i, j], -S[i, j])
        n1s = float(n_sol[g]) * ne[j]          # n_ion = n_e by quasineutrality
        sig0 = lyman_alpha_sigma0(Te[i])       # T_atom = T_e in this model
        row = []
        for D in (1.0, 10.0, 100.0):
            row.append(escape_factor_slab(n1s, sig0, D))
        print(f"  [{i:2d},{j:2d}] {Te[i]:7.3f} {ne[j]:10.3e} {n1s:11.3e}"
              f" {sig0:10.3e} {row[1]['tau_c']:12.3e}"
              f" {row[0]['theta_P']:9.3e} {row[1]['theta_P']:9.3e}"
              f" {row[2]['theta_P']:10.3e}")
    print()
    print("  Note: D is the slab thickness and is an INPUT, not a model output.")
    print("  The model has no geometry; quote the D you are prepared to defend")
    print("  for an ITER divertor and say so in Chapter 6.")

    print()
    print("=" * 78)
    print("Report only. Nothing was written.")


if __name__ == "__main__":
    main()