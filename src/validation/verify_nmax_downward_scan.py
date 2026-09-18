#!/usr/bin/env python
"""
verify_nmax_downward_scan.py
============================
Convergence of f_3 - f_4 under downward truncation of the level ladder, as a
stamped run (backlog item N8).

WHY THIS EXISTS
---------------
Section sec:convergence of Chapter 4, the validation-ladder row "Truncation
convergence", and Appendix C all quote the same result: rebuilding L with
successively lower n_max and recomputing f_3 - f_4 gives increments that decay
geometrically with ratio about 0.75, and the extrapolated truncation error is
+0.9 % at the benchmark [23,5], -1.4 % at the cold corner [0,0] and -0.55 % at
the ridge [15,3]. Those numbers rest on findings_10 ADDENDUM D.8, a working
note with no producing script. The rule is that every number in the thesis
comes from a run; this is that run.

METHOD
------
From L_grid and S_grid (cr_context), at every grid point and for
n_max = 15, 14, ..., 9. The bundled shells n9..n15 are single states, so each
step removes exactly one state; the l-resolved shells n <= 8 are never touched.
  keep  = states with n <= n_max,  drop = states with n > n_max
  L'    = L[keep, keep]  with  L'_ii += sum_{k in drop} L_ki  for every kept i.
          The transition i -> k no longer exists, so its loss leaves the
          diagonal of i. Column sums over kept states are then unchanged (G2).
  S'    = S[keep]  (recombination into removed shells is absent, exactly as
          recombination into n > 15 is absent from the production matrix)
  a = -L'_EE^{-1} L'_Eg,  c = -L'_EE^{-1} S'_E,  a_p, c_p = shell sums, p = 3, 4
  u_CRE(n_max) = [-L'^{-1} S']_ground
  S(n_max) = f_3 - f_4 at u = u_CRE(n_max),   f_p(u) = a_p u / (a_p u + c_p)
Increments d(n) = S(n) - S(n-1), n = 10..15; ratios r(n) = d(n)/d(n-1).
Geometric tail beyond n = 15 with r = d(15)/d(14):  T = d(15) r / (1 - r);
extrapolated truncation error  e = T / S(15)  (positive: truncation makes S
too small).
Definitional sensitivity (method 4): S is also evaluated at the fixed
u_CRE(15), giving S_fixed and e_fixed, so that the reader can see whether the
answer depends on letting the operating point move with the truncation.
Secondary, reported without a prediction: the same increments and
extrapolation for u_CRE, Delta = ln[(a3/a4)/(c3/c4)] and tau_slow = 1/|lambda_0|
(the eigenvalue of L' of smallest magnitude), because Chapter 4 currently says
their convergence under truncation "is not established".
The trap (D.8's methodological warning): dropping states WITHOUT the diagonal
correction is run at the three named points, for n_max = 14 and n_max = 10, to
reproduce the spurious jump the note records; that is the severity check that
the correction matters.

GATES (the run stops if either fails)
-------------------------------------
G1  n_max = 15 reproduces validation/molecular_channel/molecular_channel.csv
    (a3, a4, c3, c4, u_CRE, Delta) at all 400 points to 1e-8.
G2  column sums of L' over kept states equal the column sums of L over all 43
    states for every kept column, every n_max, every point (relative 1e-10).

PREDICTIONS (written before the run, from Chapter 4 sec:convergence and D.8)
---------------------------------------------------------------------------
P1  d(n) has one sign at each named point for n = 11..15, and r(15) lies
    between 0.6 and 0.9 (the note says "about 0.75").
P2  e = +0.9 % at [23,5], -1.4 % at [0,0], -0.55 % at [15,3], to the quoted
    digits (D.8 rounding allowed: the test is agreement to 0.1 percentage
    point).
P3  Sign: e > 0 at the benchmark, e < 0 at the cold corner.
P4  The trap moves S by 22 to 37 % at at least one of the three points for
    at least one of n_max = 14, 10.
P5  u_CRE, Delta, tau_slow convergence: no prediction exists; reported.

REFUTING OBSERVATION (for the Chapter 4 sentence)
-------------------------------------------------
r(15) outside (0, 1) at any named point, or d(n) changing sign between
n = 12 and 15 there. The geometric extrapolation is then invalid and the
sentence must be withdrawn rather than requoted.

OUTPUTS (with --write)
----------------------
validation/nmax_downward_scan/nmax_downward_scan.csv    per point, per n_max
validation/nmax_downward_scan/nmax_downward_scan_extrap.csv  per point: d, r, e
validation/nmax_downward_scan/nmax_downward_scan.txt    this run's log
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr_context import CRContext  # noqa: E402

BENCH_TE, BENCH_NE = 2.947, 1.389e14
NAMED = {"benchmark": (23, 5), "cold corner": (0, 0), "ridge": (15, 3)}
NMAX_LIST = [15, 14, 13, 12, 11, 10, 9]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate(Lij: np.ndarray, Sij: np.ndarray, keep: list[int], drop: list[int], fix_diag: bool):
    Lp = Lij[np.ix_(keep, keep)].copy()
    if fix_diag and drop:
        Lp[np.arange(len(keep)), np.arange(len(keep))] += Lij[np.ix_(drop, keep)].sum(axis=0)
    return Lp, Sij[keep]


def quantities(Lp: np.ndarray, Sp: np.ndarray, gpos: int, idx3p: list[int], idx4p: list[int]):
    n = Lp.shape[0]
    E = [k for k in range(n) if k != gpos]
    LEE = Lp[np.ix_(E, E)]; LEg = Lp[E, gpos]; SE = Sp[E]
    a = -np.linalg.solve(LEE, LEg); c = -np.linalg.solve(LEE, SE)
    pos = {k: q for q, k in enumerate(E)}
    a3 = sum(a[pos[k]] for k in idx3p); a4 = sum(a[pos[k]] for k in idx4p)
    c3 = sum(c[pos[k]] for k in idx3p); c4 = sum(c[pos[k]] for k in idx4p)
    u = (-np.linalg.solve(Lp, Sp))[gpos]
    ev = np.linalg.eigvals(Lp)
    lam0 = ev[np.argmin(np.abs(ev))]
    return a3, a4, c3, c4, u, 1.0 / abs(lam0.real)


def fdiff(a3, a4, c3, c4, u):
    f = lambda a_, c_, u_: a_ * u_ / (a_ * u_ + c_)
    return f(a3, c3, u) - f(a4, c4, u)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ctx = CRContext.load(); root = ctx.root
    L, TeL, neL = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    nT, nN, nS, _ = L.shape
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    mol_path = root / "validation/molecular_channel/molecular_channel.csv"
    for p in (S_path, mol_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    S = np.load(S_path)
    if S.shape != (nT, nN, nS):
        raise ValueError(f"S_grid shape {S.shape} does not match L_grid {L.shape}")

    si = pd.read_csv(ctx.state_index_path)
    g = ctx.ground_index
    idx3 = si.index[(si.n == 3) & (~si.bundled)].tolist()
    idx4 = si.index[(si.n == 4) & (~si.bundled)].tolist()
    if len(idx3) != 3 or len(idx4) != 4:
        raise ValueError(f"expected 3 and 4 l-resolved sublevels, got {len(idx3)}, {len(idx4)}")
    nmax_present = int(si.n.max())
    if nmax_present != 15:
        raise ValueError(f"state index tops out at n = {nmax_present}, expected 15")
    ti, ni = ctx.nearest_point(BENCH_TE, BENCH_NE)
    if (ti, ni) != NAMED["benchmark"]:
        raise ValueError(f"benchmark resolves to [{ti},{ni}], the docstring names [23,5]")

    log: list[str] = []
    def say(s: str = "") -> None:
        print(s); log.append(s)
    say("=" * 78)
    say("DOWNWARD n_max SCAN -- convergence of f_3 - f_4 under truncation of the ladder")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"interpreter   {sys.executable}")
    say(ctx.describe())
    say(f"  named points   : " + ", ".join(f"{k} [{i},{j}] Te={TeL[i]:.4g} eV ne={neL[j]:.3g}" for k, (i, j) in NAMED.items()))
    say(f"  n_max scanned  : {NMAX_LIST}  (each step removes one bundled state; n <= 8 untouched)")
    say("=" * 78)

    # --- the scan --------------------------------------------------------------
    keepmap = {}
    for nm in NMAX_LIST:
        drop = si.index[si.n > nm].tolist(); keep = [k for k in range(nS) if k not in drop]
        keepmap[nm] = (keep, drop)
    Q = {nm: {q: np.empty((nT, nN)) for q in ("a3", "a4", "c3", "c4", "u", "tau")} for nm in NMAX_LIST}
    g2_worst = 0.0
    for i in range(nT):
        for j in range(nN):
            Lij, Sij = L[i, j], S[i, j]
            cs_full = Lij.sum(axis=0)
            for nm in NMAX_LIST:
                keep, drop = keepmap[nm]
                Lp, Sp = truncate(Lij, Sij, keep, drop, fix_diag=True)
                cs = Lp.sum(axis=0)
                rel = np.abs(cs - cs_full[keep]) / np.maximum(np.abs(cs_full[keep]), 1e-300)
                g2_worst = max(g2_worst, float(rel.max()))
                gpos = keep.index(g)
                i3 = [keep.index(k) for k in idx3]; i4 = [keep.index(k) for k in idx4]
                a3, a4, c3, c4, u, tau = quantities(Lp, Sp, gpos, i3, i4)
                for q, v in zip(("a3", "a4", "c3", "c4", "u", "tau"), (a3, a4, c3, c4, u, tau)):
                    Q[nm][q][i, j] = v
    say(f"\nG2  column sums over kept states vs full matrix, worst relative difference over all n_max and points: {g2_worst:.2e}")
    if g2_worst > 1e-10:
        raise AssertionError("G2 FAILED: the diagonal correction does not conserve the column sums; do not read on")
    say("    G2 PASSED")

    # --- G1 gate ---------------------------------------------------------------
    m = pd.read_csv(mol_path, comment="#").sort_values(["i", "j"])
    if len(m) != nT * nN:
        raise ValueError(f"molecular_channel.csv has {len(m)} rows, expected {nT*nN}")
    q15 = Q[15]
    D15 = np.log((q15["a3"] / q15["a4"]) / (q15["c3"] / q15["c4"]))
    gates = {"a3": np.abs(q15["a3"].ravel() / m.a3.values - 1).max(),
             "a4": np.abs(q15["a4"].ravel() / m.a4.values - 1).max(),
             "c3": np.abs(q15["c3"].ravel() / m.c3.values - 1).max(),
             "c4": np.abs(q15["c4"].ravel() / m.c4.values - 1).max(),
             "u_CRE": np.abs(q15["u"].ravel() / m.u_CRE.values - 1).max(),
             "Delta": np.abs(D15.ravel() - m.Delta_atomic.values).max()}
    say("\nG1  n_max = 15 reproduces molecular_channel.csv at all 400 points (worst relative / absolute):")
    for k, v in gates.items():
        say(f"    {k:6s} {v:.2e}")
    if max(gates.values()) > 1e-8:
        raise AssertionError("G1 FAILED: the untruncated matrix does not reproduce the stamped artifact; do not read on")
    say("    G1 REPRODUCED")

    # --- S(n_max), increments, extrapolation -----------------------------------
    Sown = {nm: fdiff(Q[nm]["a3"], Q[nm]["a4"], Q[nm]["c3"], Q[nm]["c4"], Q[nm]["u"]) for nm in NMAX_LIST}
    Sfix = {nm: fdiff(Q[nm]["a3"], Q[nm]["a4"], Q[nm]["c3"], Q[nm]["c4"], Q[15]["u"]) for nm in NMAX_LIST}
    Dl = {nm: np.log((Q[nm]["a3"] / Q[nm]["a4"]) / (Q[nm]["c3"] / Q[nm]["c4"])) for nm in NMAX_LIST}

    def extrap(series: dict[int, np.ndarray]):
        d = {n: series[n] - series[n - 1] for n in range(10, 16)}
        r15 = d[15] / d[14]
        T = d[15] * r15 / (1.0 - r15)
        return d, r15, T

    dS, rS, TS = extrap(Sown); eS = TS / Sown[15]
    dSf, rSf, TSf = extrap(Sfix); eSf = TSf / Sfix[15]
    dU, rU, TU = extrap(Q_u := {nm: Q[nm]["u"] for nm in NMAX_LIST}); eU = TU / Q[15]["u"]
    dD, rD, TD = extrap(Dl); eD = TD / Dl[15]
    dT, rT, TT = extrap({nm: Q[nm]["tau"] for nm in NMAX_LIST}); eT = TT / Q[15]["tau"]

    say("\n" + "-" * 78)
    say("RESULT 1: f_3 - f_4 at its own u_CRE(n_max), the three named points")
    say("-" * 78)
    for name, (i, j) in NAMED.items():
        say(f"\n  {name} [{i},{j}]  Te = {TeL[i]:.4f} eV  ne = {neL[j]:.3e} cm^-3")
        say(f"    n_max   S(own u)      d(n)=S(n)-S(n-1)   r(n)=d(n)/d(n-1)   S(fixed u15)   u_CRE         Delta      tau_slow [s]")
        for nm in NMAX_LIST:
            dn = f"{dS[nm][i, j]:+.3e}" if nm >= 10 else "   --      "
            rn = f"{dS[nm][i, j] / dS[nm - 1][i, j]:+.4f}" if nm >= 11 else "  --   "
            say(f"    {nm:5d}   {Sown[nm][i, j]:+.6f}    {dn}        {rn}            {Sfix[nm][i, j]:+.6f}      {Q[nm]['u'][i, j]:.4e}   {Dl[nm][i, j]:+.5f}   {Q[nm]['tau'][i, j]:.5e}")
        say(f"    r(15) = {rS[i, j]:+.4f}   geometric tail T = {TS[i, j]:+.3e}   extrapolated error e = T/S(15) = {100*eS[i, j]:+.3f} %"
            f"   [fixed-u: r = {rSf[i, j]:+.4f}, e = {100*eSf[i, j]:+.3f} %]")
    say("\n  P1  one sign of d(n) for n = 11..15 and r(15) in (0.6, 0.9):")
    p1 = True
    for name, (i, j) in NAMED.items():
        signs = {np.sign(dS[n][i, j]) for n in range(11, 16)}
        ok = (len(signs) == 1) and (0.6 < rS[i, j] < 0.9)
        p1 &= ok
        say(f"      {name:12s} signs {sorted(signs)}  r(15) {rS[i, j]:+.4f}  -> {'as predicted' if ok else 'NOT as predicted'}")
    say("\n  P2  extrapolated error against the quoted +0.9 / -1.4 / -0.55 %:")
    quoted = {"benchmark": 0.9, "cold corner": -1.4, "ridge": -0.55}
    p2 = True
    for name, (i, j) in NAMED.items():
        e = 100 * eS[i, j]; ok = abs(e - quoted[name]) <= 0.1
        p2 &= ok
        say(f"      {name:12s} e = {e:+.3f} %   quoted {quoted[name]:+.2f} %   -> {'reproduced to 0.1 pp' if ok else 'NOT reproduced'}"
            f"   (fixed-u e = {100*eSf[i, j]:+.3f} %)")
    say(f"\n  P3  sign: benchmark e {'>' if eS[23,5] > 0 else '<='} 0, cold corner e {'<' if eS[0,0] < 0 else '>='} 0  -> "
        f"{'as predicted' if (eS[23,5] > 0 and eS[0,0] < 0) else 'NOT as predicted'}")
    refuted = False
    for name, (i, j) in NAMED.items():
        if not (0 < rS[i, j] < 1) or len({np.sign(dS[n][i, j]) for n in range(12, 16)}) > 1:
            refuted = True
            say(f"  REFUTER FIRED at {name}: r(15) = {rS[i, j]:+.4f}, signs of d(12..15) = {[np.sign(dS[n][i,j]) for n in range(12,16)]}")
    if not refuted:
        say("  REFUTER (r(15) outside (0,1) or sign change in d(12..15) at a named point): did not appear")

    # --- RESULT 2: the trap ------------------------------------------------------
    say("\n" + "-" * 78)
    say("RESULT 2: the trap -- dropping states without returning their loss to the diagonal")
    say("-" * 78)
    p4 = False
    for name, (i, j) in NAMED.items():
        Lij, Sij = L[i, j], S[i, j]
        out = []
        for nm in (14, 10, 9):
            keep, drop = keepmap[nm]
            Lp, Sp = truncate(Lij, Sij, keep, drop, fix_diag=False)
            gpos = keep.index(g); i3 = [keep.index(k) for k in idx3]; i4 = [keep.index(k) for k in idx4]
            a3, a4, c3, c4, u, _ = quantities(Lp, Sp, gpos, i3, i4)
            St = fdiff(a3, a4, c3, c4, u)
            jump = 100 * (St / Sown[15][i, j] - 1)
            good = 100 * (Sown[nm][i, j] / Sown[15][i, j] - 1)
            p4 |= (22 <= abs(jump) <= 37)
            out.append(f"n_max={nm}: S jumps {jump:+.1f} % (correct truncation: {good:+.2f} %)")
        say(f"  {name:12s} " + ";  ".join(out))
    say(f"  P4  a 22-37 % jump at at least one named point: {'reproduced' if p4 else 'NOT reproduced'}")

    # --- RESULT 3: the whole grid -------------------------------------------------
    say("\n" + "-" * 78)
    say("RESULT 3: extrapolated truncation error of f_3 - f_4 over the grid (own u_CRE)")
    say("-" * 78)
    defended = (TeL >= 2.0)[:, None] & np.ones((1, nN), bool)
    geometric = (rS > 0) & (rS < 1)
    onesign = np.ones((nT, nN), bool)
    for n in range(12, 16):
        onesign &= np.sign(dS[n]) == np.sign(dS[15])
    valid = geometric & onesign
    for nm_, mask in (("all 400 points", np.ones_like(defended)), ("Te >= 2 eV (280 points)", defended)):
        e = 100 * eS[mask & valid]
        say(f"  [{nm_}]  geometric and one-signed at {valid[mask].sum()} of {mask.sum()} points;"
            f"  there e: median {np.median(e):+.3f} %  range {e.min():+.3f} .. {e.max():+.3f} %;"
            f"  r(15): median {np.median(rS[mask & valid]):.3f}  range {rS[mask & valid].min():.3f} .. {rS[mask & valid].max():.3f}")
        bad = mask & ~valid
        if bad.any():
            ii, jj = np.where(bad)
            say(f"      not geometric/one-signed at: " + ", ".join(f"[{a},{b}]" for a, b in zip(ii, jj)))
    say(f"  |e| > 2 % at {int((np.abs(100*eS) > 2)[defended & valid].sum())} of the defended geometric points;"
        f"  largest |e| on the defended set: {np.abs(100*eS[defended & valid]).max():.3f} %")
    bad = ~valid
    if bad.any():
        cols = sorted(set(np.where(bad)[1].tolist()))
        say(f"\n  RESULT 3b: the non-geometric points (density column(s) {cols}, ne = "
            + ", ".join(f"{neL[c]:.3e}" for c in cols) + " cm^-3): the last two measured increments there")
        for lab, mask in (("all", bad), ("Te >= 2 eV", bad & defended)):
            if mask.any():
                l15 = 100 * np.abs(dS[15][mask] / Sown[15][mask]); l14 = 100 * np.abs(dS[14][mask] / Sown[15][mask])
                say(f"    [{lab}] |S(15)-S(14)|/S(15): max {l15.max():.4f} %  median {np.median(l15):.4f} %;"
                    f"   |S(14)-S(13)|/S(15): max {l14.max():.4f} %;   the sum of all six measured increments |S(15)-S(9)|/S(15): max {100*np.abs((Sown[15]-Sown[9])[mask]/Sown[15][mask]).max():.3f} %")
        ex_i = int(np.where(bad & defended)[0][0]) if (bad & defended).any() else int(np.where(bad)[0][0])
        ex_j = int(np.where(bad[ex_i])[0][0])
        say(f"    example [{ex_i},{ex_j}]: d(10..15) = " + ", ".join(f"{dS[n][ex_i, ex_j]:+.2e}" for n in range(10, 16))
            + f";  S(15) = {Sown[15][ex_i, ex_j]:+.6f}")
    # Delta ln u across the heating step (i -> i+1 at fixed j), the two-point reservoir quantity
    dlnu = {nm: np.log(Q[nm]["u"][1:, :] / Q[nm]["u"][:-1, :]) for nm in NMAX_LIST}
    ddl = {n: dlnu[n] - dlnu[n - 1] for n in range(10, 16)}
    rdl = ddl[15] / ddl[14]; Tdl = ddl[15] * rdl / (1 - rdl)
    stepdef = (TeL[:-1] >= 2.0)[:, None] & np.ones((1, nN), bool)
    geo = (rdl > 0) & (rdl < 1)
    say(f"\n  Delta ln u across the {int(stepdef.sum())} defended heating steps (i -> i+1): |Dlnu(15) - Dlnu(14)| max {np.abs(ddl[15][stepdef]).max():.2e},"
        f" median {np.median(np.abs(ddl[15][stepdef])):.2e};  geometric at {int((geo & stepdef).sum())} of {int(stepdef.sum())}, tail |T| max there {np.abs(Tdl[geo & stepdef]).max():.2e}"
        f"  (against |Dlnu(15)| median {np.median(np.abs(dlnu[15][stepdef])):.3f}, min {np.abs(dlnu[15][stepdef]).min():.2e})")

    # --- RESULT 4: reservoir quantities -----------------------------------------
    say("\n" + "-" * 78)
    say("RESULT 4 (P5, no prediction): u_CRE, Delta and tau_slow under the same truncation, named points")
    say("-" * 78)
    for name, (i, j) in NAMED.items():
        say(f"  {name:12s} u_CRE: S(15)-S(14) rel {100*dU[15][i,j]/Q[15]['u'][i,j]:+.3f} %, r(15) {rU[i,j]:+.3f}, e {100*eU[i,j]:+.3f} %"
            f"   |  Delta: d(15) {dD[15][i,j]:+.2e}, r(15) {rD[i,j]:+.3f}, e {100*eD[i,j]:+.3f} %"
            f"   |  tau_slow: d(15) rel {100*dT[15][i,j]/Q[15]['tau'][i,j]:+.3f} %, r(15) {rT[i,j]:+.3f}, e {100*eT[i,j]:+.3f} %")
    for lab, rr, ee in (("u_CRE", rU, eU), ("Delta", rD, eD), ("tau_slow", rT, eT)):
        vv = (rr > 0) & (rr < 1) & defended
        say(f"  [Te >= 2 eV] {lab:8s}: geometric at {vv.sum()} of 280; e there median {100*np.median(ee[vv]):+.3f} %  "
            f"range {100*ee[vv].min():+.3f} .. {100*ee[vv].max():+.3f} %")

    say("\nREADING: f_3 - f_4 converges geometrically under downward truncation at the three named points\n"
        "and the extrapolated errors are as printed above; the Chapter 4 sentence is to be requoted from\n"
        "this run, not from findings_10 D.8. The trap result is the severity check for the diagonal\n"
        "correction. Assumptions: removing a shell removes recombination into it (as n > 15 is absent in\n"
        "production); the geometric tail uses the last measured ratio r(15) = d(15)/d(14).")

    if args.write:
        out = Path(args.out) if args.out else root / "validation/nmax_downward_scan"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}",
               f"# interpreter {sys.executable}  numpy {np.__version__}",
               f"# L_grid.npy      sha256 {sha256_file(root / 'data/processed/cr_matrix/L_grid.npy')}",
               f"# S_grid.npy      sha256 {sha256_file(S_path)}",
               f"# state_index.csv sha256 {sha256_file(ctx.state_index_path)}",
               f"# molecular_channel.csv sha256 {sha256_file(mol_path)}  (G1 gate)",
               f"# benchmark point [{ti},{ni}]  Te={TeL[ti]:.4f} eV  ne={neL[ni]:.4e} cm^-3"]
        rows = []
        for nm in NMAX_LIST:
            for i in range(nT):
                for j in range(nN):
                    rows.append(dict(n_max=nm, i=i, j=j, Te=TeL[i], ne=neL[j], a3=Q[nm]["a3"][i, j], a4=Q[nm]["a4"][i, j],
                                     c3=Q[nm]["c3"][i, j], c4=Q[nm]["c4"][i, j], u_CRE=Q[nm]["u"][i, j], Delta=Dl[nm][i, j],
                                     cap=np.tanh(abs(Dl[nm][i, j]) / 4), S_own_u=Sown[nm][i, j], S_fixed_u15=Sfix[nm][i, j],
                                     tau_slow=Q[nm]["tau"][i, j]))
        ex = []
        for i in range(nT):
            for j in range(nN):
                ex.append(dict(i=i, j=j, Te=TeL[i], ne=neL[j], S15=Sown[15][i, j],
                               **{f"d{n}": dS[n][i, j] for n in range(10, 16)}, r15=rS[i, j], tail=TS[i, j], e_S=eS[i, j],
                               r15_fixed=rSf[i, j], e_S_fixed=eSf[i, j], r15_u=rU[i, j], e_u=eU[i, j],
                               r15_Delta=rD[i, j], e_Delta=eD[i, j], r15_tau=rT[i, j], e_tau=eT[i, j],
                               geometric_one_signed=bool(valid[i, j])))
        with open(out / "nmax_downward_scan.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "nmax_downward_scan_extrap.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(ex).to_csv(fh, index=False)
        with open(out / "nmax_downward_scan.txt", "w") as fh:
            fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/nmax_downward_scan{{.csv,_extrap.csv,.txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
