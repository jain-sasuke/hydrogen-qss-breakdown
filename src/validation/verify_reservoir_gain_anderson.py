#!/usr/bin/env python
"""
verify_reservoir_gain_anderson.py
=================================
Carry the CCC -> Anderson(2002) rate substitution through to the structural
coefficients of Chapter 5: Delta, the cap tanh(|Delta|/4), Sbar, G and eps.

WHY THIS EXISTS
---------------
Section "How well do we know these rates?" measures an 11% atomic-data
systematic on the eigenvalues. Chapter 5 states that the sharp bound's value is
set by the atomic data ("change the rate set and the cap moves") without saying
by how much, and Chapter 7 lists carrying the substitution through to the
observable as the cheapest open calculation. This script performs it.

WHAT IT DOES
------------
L_grid is linear in ne, so it splits exactly into a radiative and a collisional
part. For every transition the Anderson 2002 corrigendum covers (n <= 5,
Delta n != 0, 85 transitions), the CCC rate coefficient in the collisional part
is replaced by the RMPS one, in BOTH directions by the same factor so detailed
balance survives, with the diagonal compensated so column sums are unchanged.

S_grid is NOT touched. It carries recombination into the bound states; the
substitution concerns bound-bound electron-impact excitation only. Leaving S
fixed is the whole reason the two channels can be compared at all.

The two-channel algebra is copied from verify_reservoir_gain.py:137-165 without
modification, so baseline and substituted differ only by the operator.

THE GATE
--------
Before any substituted number is reported, the baseline pass must reproduce
validation/reservoir_gain/reservoir_gain.csv row by row. If it does not, the
script raises and reports nothing: a substitution measured against an
unreproduced baseline measures nothing.

Two algebraic identities are checked at every row for BOTH operators, and are
the reason to believe the perturbed matrix is still a valid CR operator:
  - the two-channel superposition residual must stay below 1e-8
  - eps must equal |expm1(Sbar*G*dlnTe)| to a part in 1e10

Run:  python src/validation/verify_reservoir_gain_anderson.py [--write]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)

_spec = importlib.util.spec_from_file_location(
    "and02bm", _HERE.parent / "anderson2002_benchmark.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def build_substituted_L(L, te, ne):
    """Return L with Anderson 2002 rate coefficients in place of CCC's."""
    nT, nN, nS, _ = L.shape
    A = np.vstack([np.ones_like(ne), ne]).T
    coef = np.linalg.lstsq(A, L.transpose(1, 0, 2, 3).reshape(nN, -1), rcond=None)[0]
    R = coef[0].reshape(nT, nS, nS)
    C = coef[1].reshape(nT, nS, nS)
    rel = np.abs((R[:, None] + ne[None, :, None, None] * C[:, None]) - L).max() \
        / np.abs(L).max()
    if rel > 1e-10:
        raise RuntimeError(f"L is not linear in ne (residual {rel:.2e}); "
                           "the radiative/collisional split is invalid.")
    print(f"  L = L_rad + ne*L_coll exact to {rel:.2e} (relative)")

    AND = bm.parse_anderson2002(bm.PDF_PATH)
    K_st = np.load(ROOT / "data/processed/collisions/ccc/K_CCC_exc_table.npy")
    meta = pd.read_csv(ROOT / "data/processed/collisions/ccc/K_CCC_metadata.csv")
    Te_k = np.load(ROOT / "data/processed/collisions/ccc/Te_grid.npy")
    if not np.allclose(Te_k, te):
        raise RuntimeError("CCC Te grid differs from Te_grid_L.")
    idx_of = {(int(r.n_i), int(r.l_i), int(r.n_f), int(r.l_f)): int(r.idx)
              for _, r in meta.iterrows()}

    si = pd.read_csv(ROOT / "data/processed/collisions/K_exc_full/state_index.csv")
    st = {(int(r.n), int(r.l)): int(r.idx) for _, r in si.iterrows() if not r.bundled}

    Cp = C.copy()
    n_sub = 0
    for _, r in AND.iterrows():
        n_up, l_up = bm.IDX_TO_NL[int(r.i_upper)]
        n_lo, l_lo = bm.IDX_TO_NL[int(r.j_lower)]
        key = (n_lo, l_lo, n_up, l_up)
        i, j = st.get((n_lo, l_lo)), st.get((n_up, l_up))
        if key not in idx_of or i is None or j is None:
            continue
        ups = np.exp(np.interp(np.log(te), np.log(bm.TE_AND),
                               np.log([r[f"ups_{t:g}eV"] for t in bm.TE_AND])))
        f = bm.K_from_upsilon(ups, n_lo, l_lo, n_up, te) / K_st[idx_of[key], :]
        for (aa, bb) in ((j, i), (i, j)):
            d = C[:, aa, bb] * (f - 1.0)
            Cp[:, aa, bb] += d
            Cp[:, bb, bb] -= d
        n_sub += 1
    drift = np.abs((Cp - C).sum(axis=1)).max()
    print(f"  substituted {n_sub} transitions; column-sum drift {drift:.2e}")
    if drift > 1e-12 * np.abs(C).max():
        raise RuntimeError("substitution changed the column sums.")
    return R[:, None] + ne[None, :, None, None] * Cp[:, None]


def two_channel(Lop, Sop, Lop_i, Sop_i, E, g, n3E, n4E, N3, N4, tag):
    """The algebra of verify_reservoir_gain.py, unmodified."""
    n_old = np.linalg.solve(Lop_i, -Sop_i)
    n_new = np.linalg.solve(Lop, -Sop)
    LEE = Lop[np.ix_(E, E)]
    LEg = Lop[np.ix_(E, [g])].ravel()
    n0 = np.linalg.solve(LEE, -Sop[E])
    n1 = np.linalg.solve(LEE, -LEg * n_old[g])
    sup = (np.abs(n0 + (n_new[g] / n_old[g]) * n1 - n_new[E]).max()
           / np.abs(n_new[E]).max())
    if sup > 1e-8:
        raise RuntimeError(f"[{tag}] superposition residual {sup:.3e}: "
                           "the two-channel split is not exact")
    lnx = np.log(n_new[g] / n_old[g])
    a3, a4 = n1[n3E].sum(), n1[n4E].sum()
    c3, c4 = n0[n3E].sum(), n0[n4E].sum()
    R_pe = (c3 + a3) / (c4 + a4)
    R_q = n_new[N3].sum() / n_new[N4].sum()
    eps = abs(R_pe / R_q - 1.0)
    Sbar = np.log(R_pe / R_q) / lnx
    Delta = np.log((a3 / a4) / (c3 / c4))
    return dict(lnx=lnx, Sbar=Sbar, eps=eps, Delta=Delta,
                cap=np.tanh(abs(Delta) / 4.0), a3=a3, a4=a4, c3=c3, c4=c4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--win-lo", type=float, default=30.0)
    ap.add_argument("--win-hi", type=float, default=30.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g, nv = ctx.ground_index, ctx.n_values
    S = np.load(ROOT / "data/processed/cr_matrix/S_grid.npy")

    E = np.array([i for i in range(ctx.n_states) if i != g])
    N3 = np.where(nv == 3)[0]; N4 = np.where(nv == 4)[0]
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3]); n4E = np.array([pos[s] for s in N4])

    print("=" * 78)
    print("BUILDING THE SUBSTITUTED OPERATOR")
    print("=" * 78)
    Lsub = build_substituted_L(L, te, ne)
    print("  S_grid is NOT modified (recombination feed; substitution is "
          "bound-bound excitation only)")

    rows = []
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for k in a.steps:
            for i in range(len(te)):
                j_te = i + sgn * k
                if not (0 <= j_te < len(te)):
                    continue
                dlnTe = np.log(te[j_te] / te[i])
                for j in range(len(ne)):
                    lam = np.sort(np.linalg.eigvals(L[j_te, j]).real)[::-1]
                    neg = lam[lam < 0]
                    tQ, tR = 1.0 / abs(neg[0]), 1.0 / abs(neg[1])
                    window_ok = (a.win_lo * tR) < (tQ / a.win_hi)

                    b = two_channel(L[j_te, j], S[j_te, j], L[i, j], S[i, j],
                                    E, g, n3E, n4E, N3, N4, f"base {i},{j}")
                    s_ = two_channel(Lsub[j_te, j], S[j_te, j], Lsub[i, j],
                                     S[i, j], E, g, n3E, n4E, N3, N4,
                                     f"sub {i},{j}")
                    for d, r in ((b, "base"), (s_, "sub")):
                        Gv = d["lnx"] / dlnTe
                        ep = abs(np.expm1(d["Sbar"] * Gv * dlnTe))
                        if d["eps"] > 0 and abs(ep - d["eps"]) / d["eps"] > 1e-10:
                            raise RuntimeError(f"identity broken ({r}) at [{i},{j}]")
                        d["G"] = Gv

                    rows.append(dict(
                        direction=dlab, k=k, i=i, j=j, Te=float(te[i]),
                        ne=float(ne[j]), dlnTe=dlnTe, window_ok=bool(window_ok),
                        M=tQ / tR,
                        G=b["G"], Sbar=b["Sbar"], eps=b["eps"],
                        Delta=b["Delta"], cap=b["cap"],
                        G_sub=s_["G"], Sbar_sub=s_["Sbar"], eps_sub=s_["eps"],
                        Delta_sub=s_["Delta"], cap_sub=s_["cap"]))
    df = pd.DataFrame(rows)

    # ---------------- GATE: reproduce the recorded baseline ----------------
    print("\n" + "=" * 78)
    print("GATE  --  baseline must reproduce validation/reservoir_gain/reservoir_gain.csv")
    print("=" * 78)
    ref = pd.read_csv(ROOT / "validation/reservoir_gain/reservoir_gain.csv")
    m = df.merge(ref, on=["direction", "k", "i", "j"], suffixes=("", "_ref"))
    if len(m) != len(ref):
        raise RuntimeError(f"row mismatch: {len(m)} joined vs {len(ref)} recorded")
    worst = {}
    for q in ("G", "Sbar", "eps"):
        rel = np.abs(m[q] - m[f"{q}_ref"]) / np.maximum(np.abs(m[f"{q}_ref"]), 1e-300)
        worst[q] = rel.max()
        print(f"  {q:6s} max relative difference vs recorded: {rel.max():.3e}")
    if max(worst.values()) > 1e-10:
        raise RuntimeError("baseline does not reproduce the recorded file; "
                           "nothing below is trustworthy.")
    print(f"  {len(m)} rows reproduced. GATE PASSED.")

    # ---------------- results ----------------
    def rep(sub, name):
        print(f"\n  {name}  (n={len(sub)})")
        for q in ("Delta", "cap", "Sbar", "G", "eps"):
            b_, s_ = sub[q].abs(), sub[f"{q}_sub"].abs()
            d = (s_ / b_ - 1.0) * 100.0
            print(f"    {q:6s} base median {b_.median():10.5f}   "
                  f"sub median {s_.median():10.5f}   "
                  f"change: median {d.median():+7.2f}%  mean|.| {d.abs().mean():6.2f}%  "
                  f"max|.| {d.abs().max():6.2f}%")

    print("\n" + "=" * 78)
    print("EFFECT OF THE SUBSTITUTION ON THE STRUCTURAL COEFFICIENTS")
    print("=" * 78)
    rep(df, "all rows")
    rep(df[df.window_ok & (df.k == 1) & (df.direction == "heat")],
        "k=1 heating, window_ok")
    rep(df[df.window_ok & (df.Te >= 2.0)], "window_ok, Te >= 2 eV (defended range)")

    ok = df[df.window_ok & (df.k == 1) & (df.direction == "heat")]
    print("\n" + "=" * 78)
    print("THE CAP, AND WHETHER THE HEADROOM STATEMENT SURVIVES")
    print("=" * 78)
    print(f"  cap = tanh(|Delta|/4), k=1 heating, window_ok")
    print(f"    baseline    min {ok.cap.min():.4f}  median {ok.cap.median():.4f}  "
          f"max {ok.cap.max():.4f}")
    print(f"    substituted min {ok.cap_sub.min():.4f}  median {ok.cap_sub.median():.4f}  "
          f"max {ok.cap_sub.max():.4f}")
    frac_b = (ok.Sbar.abs() / ok.cap) * 100
    frac_s = (ok.Sbar_sub.abs() / ok.cap_sub) * 100
    print(f"  |Sbar| as a percentage of its own cap:")
    print(f"    baseline    median {frac_b.median():.1f}%   max {frac_b.max():.1f}%")
    print(f"    substituted median {frac_s.median():.1f}%   max {frac_s.max():.1f}%")
    print(f"  bound |Sbar| <= cap violated: baseline {(ok.Sbar.abs()>ok.cap).sum()}, "
          f"substituted {(ok.Sbar_sub.abs()>ok.cap_sub).sum()} of {len(ok)}")

    print("\n" + "=" * 78)
    print("eps AT THE POINTS CHAPTER 5 QUOTES")
    print("=" * 78)
    for (i, j, lab) in [(23, 5, "benchmark [23,5]"), (15, 3, "ridge [15,3]"),
                        (0, 4, "worst [0,4]")]:
        r = df[(df.direction == "heat") & (df.k == 1) & (df.i == i) & (df.j == j)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        print(f"  {lab:>18}  eps {r.eps*100:7.3f}% -> {r.eps_sub*100:7.3f}%  "
              f"({(r.eps_sub/r.eps-1)*100:+6.2f}%)   "
              f"|Sbar*G| {abs(r.Sbar*r.G):6.3f} -> {abs(r.Sbar_sub*r.G_sub):6.3f}")

    if a.write:
        out = ROOT / "validation/reservoir_gain_anderson"
        out.mkdir(exist_ok=True)
        df.to_csv(out / "reservoir_gain_anderson.csv", index=False)
        print(f"\n  wrote {out.relative_to(ROOT)}/reservoir_gain_anderson.csv")
    else:
        print("\n  (--write not given; nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
