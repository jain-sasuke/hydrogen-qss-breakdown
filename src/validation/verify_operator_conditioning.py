#!/usr/bin/env python
"""
verify_operator_conditioning.py
===============================
Give the operator-conditioning and non-normality numbers a producing script.

WHY THIS EXISTS
---------------
A Gate 5 claim-to-evidence audit found 57 numbers in this thesis whose only
artifact is a markdown file. Four of them are conditioning and non-normality
quantities that appear in written LaTeX and in backlog entries marked Verified:

  kappa(L_FF)     chapter3.tex:719-721 quotes 1.48e3 to 1.74e5 across the grid.
                  Backlog H11 claimed this "now has a script". It did not.
                  The number was computed once in an ad-hoc shell session.
  mu(L)           backlog C1, closed as Verified on a markdown-only number.
  PR(v_0), PR(v_1) backlog C2, same defect. PR(v_1) = 2.64 also refutes a
                  sentence in chapter3.tex:444-446.
  spectral abscissa  CLAUDE.md reference table, -4.40e4 s^-1.

CLAUDE.md rule 4 requires provenance on every number. A correct number with no
producing script is a provenance defect even when the value is right, because
nobody can reproduce it and nobody can tell when it goes stale.

WHAT IT COMPUTES, over every grid point
---------------------------------------
  kappa(L_FF)          2-norm condition number of the excited block. Chapter 3
                       uses it to argue the inversion in the two-channel split
                       is numerically safe.
  column dominance     min |column sum| of L_FF. This is the PREMISE of the
                       Levy-Desplanques argument chapter3.tex:714-719 uses to
                       prove L_FF invertible. An argument whose premise is not
                       measured is an assertion.
  mu(L)                numerical abscissa, max eigenvalue of (L + L^T)/2. It
                       bounds transient growth: mu > 0 means the operator can
                       amplify before it decays, which is what non-normality
                       means operationally.
  spectral abscissa    max Re(lambda). Negative everywhere for a stable system.
  PR(v_k)              participation ratio of the right eigenvectors of the two
                       slowest modes, in BOTH the raw and the population-scaled
                       measure, because chapter3.tex uses different norms in
                       adjacent sentences without naming either.

THE NORM PROBLEM, STATED
------------------------
PR and eigenvector weight are norm-dependent. In the raw measure v_1's largest
component is the ground state; in the population-scaled measure v_p/n_p^CRE it
is 1.9e-4. Both are true and they describe different things. This script
reports both and labels them, so a reader cannot pick one up without its norm.

Read-only. Nothing outside validation/operator_conditioning/ is written, and
only with --write.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)


def participation_ratio(v: np.ndarray) -> float:
    """PR = (sum |v_i|^2)^2 / sum |v_i|^4. Equals 1 for a single-site vector and
    N for one spread evenly over N components."""
    a = np.abs(v) ** 2
    s = a.sum()
    if s <= 0:
        raise RuntimeError("zero eigenvector; cannot form a participation ratio")
    return float(s ** 2 / (a ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    E = np.array([i for i in range(ctx.n_states) if i != g])

    s_path = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not s_path.exists():
        raise RuntimeError(f"missing source vector: {s_path}")
    S = np.load(s_path)

    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    print(f"  L_grid {L.shape}  sha256 {hashlib.sha256(lp.read_bytes()).hexdigest()[:32]}...")
    print(f"  states {ctx.n_states}  ground index {g} ({ctx.labels[g]})")
    print(f"  interpreter {sys.executable}   numpy {np.__version__}")
    print()

    nT, nN = len(te), len(ne)
    kap = np.empty((nT, nN))
    dom = np.empty((nT, nN))
    mu = np.empty((nT, nN))
    alpha = np.empty((nT, nN))
    pr0_raw = np.empty((nT, nN)); pr1_raw = np.empty((nT, nN))
    pr0_pop = np.empty((nT, nN)); pr1_pop = np.empty((nT, nN))
    g_wt_raw = np.empty((nT, nN)); g_wt_pop = np.empty((nT, nN))

    for i in range(nT):
        for j in range(nN):
            A = L[i, j]
            Aff = A[np.ix_(E, E)]

            kap[i, j] = np.linalg.cond(Aff)
            cs = Aff.sum(axis=0)
            if np.any(cs >= 0):
                raise RuntimeError(
                    f"column sum of L_FF is not strictly negative at [{i},{j}]: "
                    f"max {cs.max():.3e}. The Levy-Desplanques argument in "
                    f"chapter3 requires strict column diagonal dominance.")
            dom[i, j] = np.abs(cs).min()

            sym = 0.5 * (A + A.T)
            mu[i, j] = np.linalg.eigvalsh(sym).max()

            w, V = np.linalg.eig(A)
            if np.abs(w.imag).max() > 1e-10 * np.abs(w.real).max():
                raise RuntimeError(
                    f"complex spectrum at [{i},{j}]: max |Im|/|Re| = "
                    f"{np.abs(w.imag).max()/np.abs(w.real).max():.3e}")
            wr = w.real
            alpha[i, j] = wr.max()
            order = np.argsort(wr)[::-1]          # least negative first
            v0 = V[:, order[0]].real
            v1 = V[:, order[1]].real

            n_cre = np.linalg.solve(A, -S[i, j])
            if np.any(n_cre <= 0):
                raise RuntimeError(f"non-positive CRE population at [{i},{j}]")

            pr0_raw[i, j] = participation_ratio(v0)
            pr1_raw[i, j] = participation_ratio(v1)
            pr0_pop[i, j] = participation_ratio(v0 / n_cre)
            pr1_pop[i, j] = participation_ratio(v1 / n_cre)
            g_wt_raw[i, j] = abs(v1[g]) / np.abs(v1).max()
            vp = v1 / n_cre
            g_wt_pop[i, j] = abs(vp[g]) / np.abs(vp).max()

    ib = int(np.argmin(np.abs(te - 2.947)))
    jb = int(np.argmin(np.abs(np.log10(ne) - np.log10(1.389e14))))
    if (ib, jb) != (23, 5):
        raise RuntimeError(f"benchmark index resolved to ({ib},{jb}), not (23,5); "
                           f"the grid has changed and every quoted number is suspect")

    def rep(name, arr, fmt="{:.4e}", recorded=None, rtol=0.02):
        lo, hi = arr.min(), arr.max()
        ilo = np.unravel_index(arr.argmin(), arr.shape)
        ihi = np.unravel_index(arr.argmax(), arr.shape)
        line = (f"  {name:<26} min {fmt.format(lo)} at {list(ilo)}"
                f"   max {fmt.format(hi)} at {list(ihi)}"
                f"   median {fmt.format(np.median(arr))}"
                f"   benchmark {fmt.format(arr[ib, jb])}")
        print(line)
        if recorded is not None:
            got, want, what = recorded
            d = abs(got - want) / abs(want)
            flag = "OK " if d <= rtol else "!! "
            print(f"      {flag}against {what}: {fmt.format(want)}, "
                  f"{100*d:.2f}% apart")

    print("=" * 78)
    print("CONDITIONING AND NON-NORMALITY, over all 400 points")
    print("=" * 78)
    rep("kappa(L_FF)", kap, recorded=(kap.min(), 1.48e3, "chapter3.tex:719 lower"))
    print(f"      upper: measured {kap.max():.4e} against chapter3.tex:721's "
          f"1.74e5, {100*abs(kap.max()-1.74e5)/1.74e5:.2f}% apart")
    rep("min |col sum| L_FF", dom)
    print("      premise of the Levy-Desplanques argument at chapter3.tex:714-719")
    rep("mu(L) numerical abscissa", mu,
        recorded=(mu[ib, jb], 1.28e11, "CLAUDE.md benchmark"))
    rep("spectral abscissa", alpha,
        recorded=(alpha[ib, jb], -4.40e4, "CLAUDE.md benchmark"))
    print()
    print("=" * 78)
    print("PARTICIPATION RATIO, BOTH NORMS. Neither is 'the' answer.")
    print("=" * 78)
    rep("PR(v0) raw", pr0_raw, "{:.4f}")
    rep("PR(v1) raw", pr1_raw, "{:.4f}")
    rep("PR(v0) population-scaled", pr0_pop, "{:.4f}")
    rep("PR(v1) population-scaled", pr1_pop, "{:.4f}")
    print()
    rep("v1 ground weight, raw", g_wt_raw, "{:.4e}")
    rep("v1 ground weight, pop-scaled", g_wt_pop, "{:.4e}")
    print()
    print("  chapter3.tex:444-446 calls v1 'distributed across the excited")
    print(f"  manifold'. PR(v1) raw at the benchmark is {pr1_raw[ib,jb]:.2f}, which")
    print("  describes a vector on about two or three components, not a")
    print("  distributed one. Reported, not repaired.")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "operator_conditioning"
        out.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with (out / "operator_conditioning.csv").open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["i", "j", "Te", "ne", "kappa_LFF", "min_abs_colsum_LFF",
                        "mu_L", "spectral_abscissa", "PR_v0_raw", "PR_v1_raw",
                        "PR_v0_pop", "PR_v1_pop", "v1_ground_wt_raw",
                        "v1_ground_wt_pop"])
            for i in range(nT):
                for j in range(nN):
                    w.writerow([i, j, f"{te[i]:.6g}", f"{ne[j]:.6g}",
                                f"{kap[i,j]:.8e}", f"{dom[i,j]:.8e}",
                                f"{mu[i,j]:.8e}", f"{alpha[i,j]:.8e}",
                                f"{pr0_raw[i,j]:.6f}", f"{pr1_raw[i,j]:.6f}",
                                f"{pr0_pop[i,j]:.6f}", f"{pr1_pop[i,j]:.6f}",
                                f"{g_wt_raw[i,j]:.6e}", f"{g_wt_pop[i,j]:.6e}"])
        print(f"\n  wrote {out/'operator_conditioning.csv'}  ({nT*nN} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
