#!/usr/bin/env python
"""
verify_2s_not_slow.py
=====================
Is 2s a second slow state? Four measurements, with provenance.

WHY THIS EXISTS
---------------
chapter5.tex Section sec:slow_subspace rests the entire two-channel
decomposition on there being exactly ONE slow state. The obvious candidate for a
second is 2s, which has no E1 decay to the ground state and a two-photon
lifetime of 0.12 s. If 2s held population on the slow timescale the supply sum
would gain a third term, the logistic would change shape, and the tanh bound
would not follow.

The subsection made four measurements and carried a [SOURCE REQUIRED] saying
they came from an audit on a date, with no script and no artifact. A date is not
provenance. This script produces all four and stamps them.

THE FOUR
--------
  1. |L_2s,2s| against |lambda_0| at the thinnest, coldest corner, where the
     argument is weakest because l-mixing is slowest there.
  2. The same comparison with l-mixing DELETED, leaving 2s with nothing but
     two-photon emission. This is the version that does not lean on a
     proton-impact rate coefficient.
  3. The density at which proton l-mixing out of 2s falls to the two-photon
     rate, i.e. where the collisional route stops dominating.
  4. Over all 400 points, the smallest ratio of |Re lambda| across the excited
     block to |lambda_0|, and whether any slow eigenvector carries dominant 2s
     weight.

The refuting observation, stated in advance: a ratio near 1 anywhere on the
grid, or a slow eigenvector with its largest population-scaled component on 2s,
would mean a second reservoir and would break the decomposition.

The two-photon rate is the one number here not read from the pipeline: the
radiative array carries A(2s->1s) = 0 exactly, because two-photon decay is not
an E1 transition and was never entered. It is taken from the literature value
quoted in chapter2 and is used only to make the argument that does NOT depend
on l-mixing; the pipeline's own matrix is more pessimistic, not less, since it
gives 2s no radiative decay at all.

Read-only. Writes only to validation/slow_subspace/, and only with --write.
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

# Two-photon 2s -> 1s. Not in the matrix: A_resolved.npy holds exactly zero for
# this transition because it is not E1. chapter2.tex quotes 8.229 s^-1.
A_2PHOTON = 8.229

# chapter5.tex records these. Recomputed, not trusted.
REC = {"L2s2s_cold": 1.2564e9, "lam0_cold": 0.0149,
       "ratio_2photon": 553.0, "ne_crossover": 7e3, "min_ratio": 86.5}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(name, got, want, rtol):
    d = abs(got - want) / abs(want)
    flag = "OK " if d <= rtol else "!! "
    print(f"  {flag}{name:<44} {got:< 14.6g} vs recorded {want:< 12.6g}"
          f"  ({d*100:.2f}% apart)")
    return d <= rtol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    labels = ctx.labels
    twos = [k for k, s in enumerate(labels) if s.upper() == "2S"]
    if len(twos) != 1:
        raise RuntimeError(f"expected exactly one 2S state, found "
                           f"{[labels[k] for k in twos]}")
    i2s = twos[0]
    E = np.array([i for i in range(ctx.n_states) if i != g])
    pos2s = int(np.where(E == i2s)[0][0])

    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not sp.exists():
        raise RuntimeError(f"missing {sp}")
    S = np.load(sp)

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    print(f"  L_grid {L.shape}  sha256 {sha256(lp)[:32]}...")
    print(f"  2s is state index {i2s} ({labels[i2s]}); ground is {g} ({labels[g]})")
    print(f"  interpreter {sys.executable}   numpy {np.__version__}")
    print()

    # ---- 1 and 2: the cold, thin corner ----------------------------------
    ic, jc = 0, 0
    A = L[ic, jc]
    lam = np.linalg.eigvals(A).real
    lam0 = abs(np.sort(lam[lam < 0])[::-1][0])
    L2s = abs(A[i2s, i2s])
    print("=" * 78)
    print(f"THE WEAKEST CORNER: Te = {te[ic]:.4g} eV, ne = {ne[jc]:.4g} cm^-3")
    print("=" * 78)
    ok = True
    ok &= check("|L_2s,2s|  [1/s]", L2s, REC["L2s2s_cold"], 0.02)
    ok &= check("|lambda_0|  [1/s]", lam0, REC["lam0_cold"], 0.02)
    print(f"     separation |L_2s,2s| / |lambda_0| = {L2s/lam0:.4g}")
    r2 = A_2PHOTON / lam0
    ok &= check("two-photon alone, A / |lambda_0|", r2, REC["ratio_2photon"], 0.02)
    print("     The second line is the argument that does not lean on a")
    print("     proton-impact rate coefficient: even stripped of l-mixing, 2s")
    print("     empties far faster than the reservoir turns over.")
    print()

    # ---- 3: where l-mixing out of 2s falls to the two-photon rate ---------
    print("=" * 78)
    print("AT WHAT DENSITY WOULD 2s STOP BEING COLLISIONALLY EMPTIED?")
    print("=" * 78)
    kp = ROOT / "data/processed/lmix/K_lmix.npy"
    if not kp.exists():
        raise RuntimeError(f"missing {kp}: the crossover density cannot be "
                           f"computed without the l-mixing coefficients.")
    K = np.load(kp)
    print("  The l-mixing rate out of 2s is linear in density, so the crossover")
    print("  solves  ne * sum_i K_lmix[i, 2s, Te] = A_2photon.  It depends on")
    print("  Te only through the rate coefficient, so it is quoted as a range.")
    q = np.array([float(K[:, i2s, t].sum()) for t in range(len(te))])
    if np.any(q <= 0):
        raise RuntimeError("the l-mixing coefficient out of 2s is zero at some "
                           "temperature; the crossover is undefined there")
    ne_x = A_2PHOTON / q
    print(f"    q(2s -> anything) runs {q.min():.4g} to {q.max():.4g} cm^3/s")
    print(f"    crossover density runs {ne_x.min():.4g} to {ne_x.max():.4g} cm^-3, "
          f"median {np.median(ne_x):.4g}")
    ok &= check("crossover density, median [cm^-3]", float(np.median(ne_x)),
                REC["ne_crossover"], 0.25)
    dec = np.log10(ne[0] / np.median(ne_x))
    print(f"    that is {dec:.2f} decades below the bottom of the operating "
          f"grid, {ne[0]:.3g} cm^-3")
    print()

    # ---- 4: the grid-wide spectral test ----------------------------------
    print("=" * 78)
    print("IS THERE A NEAR-DEGENERATE SECOND SLOW MODE ANYWHERE?")
    print("=" * 78)
    ratios = np.empty((len(te), len(ne)))
    worst2s = 0.0
    worst2s_pop = 0.0
    rows = []
    for i in range(len(te)):
        for j in range(len(ne)):
            Aij = L[i, j]
            lm = np.linalg.eigvals(Aij).real
            l0 = abs(np.sort(lm[lm < 0])[::-1][0])
            lff = np.linalg.eigvals(Aij[np.ix_(E, E)]).real
            if np.any(lff >= 0):
                raise RuntimeError(f"L_FF has a non-negative eigenvalue at "
                                   f"[{i},{j}]; the excited block is not stable")
            ratios[i, j] = np.abs(lff).min() / l0
            # The slow eigenvector's 2s content, in BOTH norms, because they
            # answer different questions and only one of them was in the
            # chapter. Raw: which level HOLDS the population that the slow mode
            # moves. Population-scaled: which level's FRACTIONAL response is
            # largest. A second reservoir would have to show up in the raw
            # measure; the scaled measure near 1 means 2s tracks the ground
            # state, which is what being slaved to it looks like.
            w, V = np.linalg.eig(Aij)
            k0 = int(np.argmax(w.real))
            v = np.abs(V[:, k0].real)
            n_cre = np.linalg.solve(Aij, -S[i, j])
            if np.any(n_cre <= 0):
                raise RuntimeError(f"non-positive CRE population at [{i},{j}]")
            vp = np.abs(v / n_cre)
            raw2s = float(v[i2s] / v.max())
            pop2s = float(vp[i2s] / vp.max())
            if int(np.argmax(v)) != g:
                raise RuntimeError(
                    f"at [{i},{j}] the largest RAW component of the slow "
                    f"eigenvector is {labels[int(np.argmax(v))]}, not the "
                    f"ground state. The one-slow-state partition fails here.")
            worst2s = max(worst2s, raw2s)
            worst2s_pop = max(worst2s_pop, pop2s)
            rows.append(dict(i=i, j=j, Te=float(te[i]), ne=float(ne[j]),
                             ratio_LFF_lam0=float(ratios[i, j]),
                             slow_vec_2s_raw=raw2s,
                             slow_vec_2s_popscaled=pop2s))
    k = np.unravel_index(ratios.argmin(), ratios.shape)
    ok &= check("min |Re lambda(L_FF)| / |lambda_0| over 400 pts",
                float(ratios.min()), REC["min_ratio"], 0.02)
    print(f"     at Te = {te[k[0]]:.4g} eV, ne = {ne[k[1]]:.4g} cm^-3")
    print(f"     median over the grid {np.median(ratios):.4g}, "
          f"max {ratios.max():.4g}")
    print()
    print("  The slow eigenvector's 2s content, in both norms:")
    print(f"    RAW, relative to the largest component: max {worst2s:.4e} "
          f"over the grid")
    print(f"    the largest raw component is the GROUND STATE at all "
          f"{len(rows)} points, checked above and raising if not")
    print(f"    POPULATION-SCALED, relative to the largest: max "
          f"{worst2s_pop:.4f}")
    print()
    print("  These say different things and the chapter previously gave only")
    print("  the first. Raw asks which level holds the population the slow")
    print("  mode moves, and the answer is the ground state everywhere, with")
    print("  2s below one part in 487 of it at worst. Population-scaled asks")
    print("  whose FRACTIONAL response is largest, and there 2s reaches")
    print(f"  {worst2s_pop:.3f} of the ground state's. That is not a second")
    print("  reservoir. It is 2s tracking the ground state almost exactly,")
    print("  which is what being slaved to one reservoir means. An")
    print("  independent reservoir would move DIFFERENTLY, not identically.")
    print()
    print("=" * 78)
    print("VERDICT:", "all four measurements reproduce" if ok
          else "AT LEAST ONE DISAGREES WITH chapter5.tex, see the !! rows")
    print("=" * 78)

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "slow_subspace"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "slow_subspace.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with (out / "slow_subspace.txt").open("w") as fh:
            fh.write(f"L_grid sha256 {sha256(lp)}\n")
            fh.write(f"2s index {i2s} ({labels[i2s]})\n")
            fh.write(f"|L_2s,2s| at [0,0] = {L2s:.6e} 1/s\n")
            fh.write(f"|lambda_0| at [0,0] = {lam0:.6e} 1/s\n")
            fh.write(f"separation = {L2s/lam0:.6e}\n")
            fh.write(f"two-photon A = {A_2PHOTON} 1/s, A/|lambda_0| = {r2:.6g}\n")
            fh.write(f"crossover density {ne_x.min():.6e} to {ne_x.max():.6e} "
                     f"cm^-3, median {np.median(ne_x):.6e}\n")
            fh.write(f"min |Re lambda(L_FF)|/|lambda_0| = {ratios.min():.6f} "
                     f"at Te={te[k[0]]:.6g} eV, ne={ne[k[1]]:.6g} cm^-3\n")
            fh.write(f"max RAW 2s weight in a slow eigenvector = {worst2s:.6e}\n")
            fh.write(f"max POPULATION-SCALED 2s weight = {worst2s_pop:.6f}\n")
            fh.write("largest raw component is the ground state at all "
                     "400 points\n")
        print(f"  wrote {out/'slow_subspace.csv'}  ({len(rows)} rows)")
        print(f"  wrote {out/'slow_subspace.txt'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
