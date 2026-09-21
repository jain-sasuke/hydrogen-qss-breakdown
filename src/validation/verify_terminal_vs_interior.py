#!/usr/bin/env python
"""
verify_terminal_vs_interior.py
==============================
Over-population of a shell when it is the terminal shell of the truncated
ladder, measured against the same shell with levels above it, as a stamped
run (backlog item N9).

WHY THIS EXISTS
---------------
Section sec:convergence of Chapter 4 (thesis_tex/chapter4.tex ~718-726) says:
"Comparing each shell's population when it is terminal against its population
when interior levels sit above it gives excesses of 9.4 at p=8, 10.7 at p=9
and 6.3 at p=10: of order ten, with no monotone trend over those three shells.
The externally measured excess at the terminal shell, a factor 4.9 to 6.2
against published values, is of the same order ... (it does not sit inside
the 6.3 to 10.7 interval, as findings_10 ADDENDUM D.8 wrongly states ...)".
The 9.4 / 10.7 / 6.3 (with 4.4 at p=11 and 3.1 at p=12) come from
outputs/findings_10_four_agent_review.md ADDENDUM D.8, a working note with no
producing script, which names neither the population compared (total,
ground-fed or recombination-fed) nor the grid point. The rule is that every
number in the thesis comes from a run; this is that run. The 4.9 to 6.2 is
the stamped r_1(15) ratio model/Fujimoto in validation/fujimoto_table41/
(PRODUCTION variant, 4.886 at [49,0] and 6.166 at [49,7]); it is read from
that file here, not retyped. Because Z(p)/Z(1) cancels in a ratio at fixed
Te, the ground-fed coefficient a_p is the internal quantity that external
ratio measures.

METHOD
------
From L_grid and S_grid (cr_context), at every grid point and for every shell
p = 8..14, two operators:
  TERMINAL  n_max = p:  keep = states with n <= p, drop = states with n > p,
            L' = L[keep, keep] with L'_ii += sum_{k in drop} L_ki for every
            kept i (the transition i -> k no longer exists, so its loss leaves
            the diagonal of i; column sums over kept states are unchanged,
            gate G2), S' = S[keep]. The bundled shells n9..n15 are single
            states (indices 36..42 in state_index.csv), n = 8 is eight
            l-resolved states (28..35); both are read from the file, not
            assumed.
  INTERIOR  the full n_max = 15 operator, where p has 15 - p shells above it.
For each operator, at CRE per unit n_ion, summed over the sublevels of shell p:
  (i)   n_p  = sum_p [-L^{-1} S]            total CRE population
  (ii)  a_p  = sum_p [-L_EE^{-1} L_Eg]      ground-fed coefficient (r_1 up to
                                            the Saha factor Z(p)/Z(1))
  (iii) c_p  = sum_p [-L_EE^{-1} S_E]       recombination-fed coefficient
and the excess X_q(p) = q(terminal)/q(interior), q in {n, a, c}.
Reported at the benchmark [23,5], the cold corner [0,0], the ridge [15,3] and
the two Fujimoto points [49,0] (10 eV, 1e12) and [49,7] (10 eV, 1e15); the
grid-wide median and range of X_n, X_a, X_c for p = 8, 9, 10 over the 280
defended points (Te >= 2 eV).
Definitional sensitivity (method 4), because D.8 does not say what it did:
  ALT-REF   the interior reference taken as the n_max = p + 1 operator (one
            shell above) instead of the full one; X^{(+1)}_q(p).
  TRAP      the truncation WITHOUT the diagonal correction, the artifact D.8
            itself warns about in the same section; X^{trap}_q(p). Reported
            as a diagnostic only; it is not a physical truncation.
The search for D.8's numbers runs over every (variant, reference, quantity,
point) combination, 60 in all; the primary definition is (correct
truncation, full reference).
Caveat carried into the reading: for p close to 15 the interior reference is
itself a truncated ladder with only 15 - p shells above p, so X(p) measures
the effect of adding those shells, not the excess against an infinite ladder;
a decline of X with p partly reflects the reference getting worse. X(15) has
no interior reference in this operator and is not defined.

GATES (the run stops if either fails)
-------------------------------------
G1  the untruncated operator reproduces validation/molecular_channel/
    molecular_channel.csv (a3, a4, c3, c4, u_CRE, Delta) at all 400 points to
    1e-8 (relative; Delta absolute).
G2  column sums of every corrected truncated matrix over kept states equal
    the column sums of the full matrix, every p, every point, 1e-10 relative.

PREDICTIONS (written before the run)
------------------------------------
P1  at least one (quantity, point) combination reproduces D.8's 9.4 / 10.7 /
    6.3 at p = 8 / 9 / 10 to within 5 %, and the same combination gives about
    4.4 and 3.1 (5 %) at p = 11 and 12.
P1a (this script's own expectation, added to P1) the reproducing quantity is
    a_p, and X_n and X_c lie within 10 % of unity at every p and named point:
    the top shell sits within 6e-5 of Saha-Boltzmann
    (validation/terminal_shell_budget/), so the exchange that truncation
    removes is nearly balanced for the total population and only the
    ground-fed channel, whose net flux through the top is upward, can be
    biased by a large factor. The external comparison saw the same: r_0(15)
    agrees, r_1(15) is 4.9 to 6.2 high.
P2  whichever combination that is, the excess is "of order ten" (between 3
    and 30) at p = 8, 9, 10 at the benchmark.
P3  (D.8's own claim, under test) the excess decreases monotonically with p
    from p = 9 upward, in the reproducing combination.
P4  (the chapter's other claim) the external 4.9 to 6.2 does not lie inside
    the p = 8..10 band of the reproducing combination but is of the same
    order, every band value within a factor of 3 of every external value.

REFUTING OBSERVATION (for the Chapter 4 sentence)
-------------------------------------------------
No combination within 25 % of all three quoted numbers (max relative
deviation over p = 8, 9, 10 above 0.25). The sentence's numbers are then
unsupported and must be requoted from this run with the definition and grid
point stated; the closest combination is reported either way.

OUTPUTS (with --write)
----------------------
validation/terminal_vs_interior/terminal_vs_interior.csv   per point, per p:
    terminal, interior (full and +1) and trap values of n_p, a_p, c_p, and
    every ratio
validation/terminal_vs_interior/terminal_vs_interior_named.csv   the named
    points, the P1 search table
validation/terminal_vs_interior/terminal_vs_interior.txt   this run's log
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
FUJI_TE, FUJI_NE_LO, FUJI_NE_HI = 10.0, 1e12, 1e15
NAMED = {"benchmark": (23, 5), "cold corner": (0, 0), "ridge": (15, 3),
         "Fujimoto 1e12": (49, 0), "Fujimoto 1e15": (49, 7)}
P_LIST = [8, 9, 10, 11, 12, 13, 14]
NMAX_FULL = 15
D8 = {8: 9.4, 9: 10.7, 10: 6.3, 11: 4.4, 12: 3.1}       # findings_10 ADDENDUM D.8, no script
QUANT = ("n", "a", "c")
QNAME = {"n": "n_p total CRE population", "a": "a_p ground-fed coefficient", "c": "c_p recombination-fed coefficient"}


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


def channels(Lp: np.ndarray, Sp: np.ndarray, gpos: int):
    """a = -L_EE^{-1} L_Eg, c = -L_EE^{-1} S_E (zero at the ground slot), n = -L^{-1} S, u = n_ground."""
    m = Lp.shape[0]
    E = [k for k in range(m) if k != gpos]
    LEE = Lp[np.ix_(E, E)]; LEg = Lp[E, gpos]; SE = Sp[E]
    a = np.zeros(m); c = np.zeros(m)
    a[E] = -np.linalg.solve(LEE, LEg); c[E] = -np.linalg.solve(LEE, SE)
    nvec = -np.linalg.solve(Lp, Sp)
    return a, c, nvec, nvec[gpos]


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
    fuji_path = root / "validation/fujimoto_table41/fujimoto_table41.csv"
    for p in (S_path, mol_path, fuji_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    S = np.load(S_path)
    if S.shape != (nT, nN, nS):
        raise ValueError(f"S_grid shape {S.shape} does not match L_grid {L.shape}")

    si = pd.read_csv(ctx.state_index_path)
    g = ctx.ground_index
    if int(si.n.max()) != NMAX_FULL:
        raise ValueError(f"state index tops out at n = {int(si.n.max())}, expected {NMAX_FULL}")
    for p in range(9, NMAX_FULL + 1):
        rows = si.index[si.n == p].tolist()
        if len(rows) != 1 or not bool(si.bundled[rows[0]]):
            raise ValueError(f"shell n = {p} is not a single bundled state: rows {rows}")
    if len(si.index[(si.n == 8) & (~si.bundled)]) != 8:
        raise ValueError("shell n = 8 is not eight l-resolved states")
    shell_idx = {p: si.index[si.n == p].tolist() for p in range(1, NMAX_FULL + 1)}
    idx3, idx4 = shell_idx[3], shell_idx[4]
    if len(idx3) != 3 or len(idx4) != 4:
        raise ValueError(f"expected 3 and 4 l-resolved sublevels, got {len(idx3)}, {len(idx4)}")
    for name, (te, ne, want) in {"benchmark": (BENCH_TE, BENCH_NE, (23, 5)),
                                 "Fujimoto 1e12": (FUJI_TE, FUJI_NE_LO, (49, 0)),
                                 "Fujimoto 1e15": (FUJI_TE, FUJI_NE_HI, (49, 7))}.items():
        got = ctx.nearest_point(te, ne)
        if got != want:
            raise ValueError(f"{name} resolves to {got}, the docstring names {want}")

    # the external excess, read from the stamped artifact rather than retyped
    fj = pd.read_csv(fuji_path, comment="#")
    fj15 = fj[(fj.p == 15) & (fj.variant == "PRODUCTION")].sort_values("ne_cm3")
    if len(fj15) != 2:
        raise ValueError(f"fujimoto_table41.csv: expected 2 PRODUCTION rows at p = 15, got {len(fj15)}")
    ext_lo, ext_hi = float(fj15.r1_ratio.iloc[0]), float(fj15.r1_ratio.iloc[1])
    if not (abs(ext_lo - 4.9) < 0.05 and abs(ext_hi - 6.2) < 0.05):
        raise ValueError(f"fujimoto_table41.csv r1_ratio(15) = {ext_lo:.3f}, {ext_hi:.3f}; the chapter quotes 4.9 to 6.2")

    log: list[str] = []
    def say(s: str = "") -> None:
        print(s); log.append(s)
    say("=" * 78)
    say("TERMINAL vs INTERIOR -- over-population of shell p when it is the top of the ladder")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"interpreter   {sys.executable}")
    say(ctx.describe())
    say("  named points   : " + ", ".join(f"{k} [{i},{j}] Te={TeL[i]:.4g} eV ne={neL[j]:.3g}" for k, (i, j) in NAMED.items()))
    say(f"  shells p       : {P_LIST}  terminal = n_max = p (corrected truncation);  interior = full n_max = {NMAX_FULL}")
    say(f"  external excess: r_1(15) model/Fujimoto, PRODUCTION, from {fuji_path.relative_to(root)}: "
        f"{ext_lo:.3f} at ne={float(fj15.ne_cm3.iloc[0]):.0e}, {ext_hi:.3f} at ne={float(fj15.ne_cm3.iloc[1]):.0e}")
    say("=" * 78)

    # --- the scan: every n_max in 8..15, corrected and trap ------------------------
    NM_LIST = P_LIST + [NMAX_FULL]
    keepmap = {}
    for nm in NM_LIST:
        drop = si.index[si.n > nm].tolist(); keep = [k for k in range(nS) if k not in drop]
        keepmap[nm] = (keep, drop)
    # shell sums, indexed [variant][n_max][q] -> (nT, nN, 16), NaN where p > n_max
    SH = {v: {nm: {q: np.full((nT, nN, NMAX_FULL + 1), np.nan) for q in QUANT} for nm in NM_LIST} for v in ("fix", "trap")}
    U = {v: {nm: np.empty((nT, nN)) for nm in NM_LIST} for v in ("fix", "trap")}
    g2_worst = 0.0
    for i in range(nT):
        for j in range(nN):
            Lij, Sij = L[i, j], S[i, j]
            cs_full = Lij.sum(axis=0)
            for nm in NM_LIST:
                keep, drop = keepmap[nm]
                gpos = keep.index(g)
                for v, fix in (("fix", True), ("trap", False)):
                    if nm == NMAX_FULL and v == "trap":
                        continue
                    Lp, Sp = truncate(Lij, Sij, keep, drop, fix_diag=fix)
                    if fix:
                        cs = Lp.sum(axis=0)
                        rel = np.abs(cs - cs_full[keep]) / np.maximum(np.abs(cs_full[keep]), 1e-300)
                        g2_worst = max(g2_worst, float(rel.max()))
                    a, c, nvec, u = channels(Lp, Sp, gpos)
                    U[v][nm][i, j] = u
                    for p in range(1, nm + 1):
                        pos = [keep.index(k) for k in shell_idx[p]]
                        SH[v][nm]["n"][i, j, p] = nvec[pos].sum()
                        SH[v][nm]["a"][i, j, p] = a[pos].sum()
                        SH[v][nm]["c"][i, j, p] = c[pos].sum()
    for nm in P_LIST:
        for q in QUANT:
            SH["trap"][NMAX_FULL][q] = SH["fix"][NMAX_FULL][q]   # the untruncated operator is the same in both
    U["trap"][NMAX_FULL] = U["fix"][NMAX_FULL]
    say(f"\nG2  column sums over kept states vs full matrix, worst relative difference over all p and points: {g2_worst:.2e}")
    if g2_worst > 1e-10:
        raise AssertionError("G2 FAILED: the diagonal correction does not conserve the column sums; do not read on")
    say("    G2 PASSED")

    # --- G1 gate -------------------------------------------------------------------
    m = pd.read_csv(mol_path, comment="#").sort_values(["i", "j"])
    if len(m) != nT * nN:
        raise ValueError(f"molecular_channel.csv has {len(m)} rows, expected {nT*nN}")
    F = SH["fix"][NMAX_FULL]
    a3, a4, c3, c4 = F["a"][:, :, 3], F["a"][:, :, 4], F["c"][:, :, 3], F["c"][:, :, 4]
    D15 = np.log((a3 / a4) / (c3 / c4))
    gates = {"a3": np.abs(a3.ravel() / m.a3.values - 1).max(), "a4": np.abs(a4.ravel() / m.a4.values - 1).max(),
             "c3": np.abs(c3.ravel() / m.c3.values - 1).max(), "c4": np.abs(c4.ravel() / m.c4.values - 1).max(),
             "u_CRE": np.abs(U["fix"][NMAX_FULL].ravel() / m.u_CRE.values - 1).max(),
             "Delta": np.abs(D15.ravel() - m.Delta_atomic.values).max()}
    say("\nG1  n_max = 15 reproduces molecular_channel.csv at all 400 points (worst relative / absolute):")
    for k, v in gates.items():
        say(f"    {k:6s} {v:.2e}")
    if max(gates.values()) > 1e-8:
        raise AssertionError("G1 FAILED: the untruncated matrix does not reproduce the stamped artifact; do not read on")
    say("    G1 REPRODUCED")

    # --- the excesses ----------------------------------------------------------------
    # X[variant][ref][q][p] -> (nT, nN)
    X = {v: {r: {q: {} for q in QUANT} for r in ("full", "next")} for v in ("fix", "trap")}
    for v in ("fix", "trap"):
        for q in QUANT:
            for p in P_LIST:
                term = SH[v][p][q][:, :, p]
                X[v]["full"][q][p] = term / SH[v][NMAX_FULL][q][:, :, p]
                X[v]["next"][q][p] = term / SH[v][p + 1][q][:, :, p]
    defended = (TeL >= 2.0)[:, None] & np.ones((1, nN), bool)

    say("\n" + "-" * 78)
    say("RESULT 1: primary definition -- corrected truncation, interior = full n_max = 15 operator")
    say("          X_q(p) = q(terminal, n_max = p) / q(interior), q = n_p, a_p, c_p; the +1 columns use n_max = p+1 as reference")
    say("-" * 78)
    for name, (i, j) in NAMED.items():
        say(f"\n  {name} [{i},{j}]  Te = {TeL[i]:.4f} eV  ne = {neL[j]:.3e} cm^-3   u_CRE(15) = {U['fix'][NMAX_FULL][i, j]:.4e}")
        say(f"    p    X_n(full)   X_a(full)   X_c(full)  |  X_n(+1)   X_a(+1)   X_c(+1)  |  a_p(term)    a_p(int)     n_p(term)    n_p(int)     D.8")
        for p in P_LIST:
            d8 = f"{D8[p]:.1f}" if p in D8 else "  --"
            say(f"    {p:2d}   {X['fix']['full']['n'][p][i,j]:9.4f}   {X['fix']['full']['a'][p][i,j]:9.4f}   {X['fix']['full']['c'][p][i,j]:9.4f}  |"
                f"  {X['fix']['next']['n'][p][i,j]:7.4f}   {X['fix']['next']['a'][p][i,j]:7.4f}   {X['fix']['next']['c'][p][i,j]:7.4f}  |"
                f"  {SH['fix'][p]['a'][i,j,p]:.4e}   {SH['fix'][NMAX_FULL]['a'][i,j,p]:.4e}   {SH['fix'][p]['n'][i,j,p]:.4e}   {SH['fix'][NMAX_FULL]['n'][i,j,p]:.4e}   {d8}")

    # --- RESULT 2: the search for D.8's numbers --------------------------------------
    say("\n" + "-" * 78)
    say("RESULT 2: P1 -- which (variant, reference, quantity, point) reproduces D.8's 9.4 / 10.7 / 6.3 at p = 8 / 9 / 10")
    say("          dev3 = max_{p=8,9,10} |X/D8 - 1|;  dev5 adds p = 11, 12 (4.4, 3.1)")
    say("-" * 78)
    combos = []
    for v in ("fix", "trap"):
        for r in ("full", "next"):
            for q in QUANT:
                for name, (i, j) in NAMED.items():
                    vals = {p: float(X[v][r][q][p][i, j]) for p in P_LIST}
                    dev3 = max(abs(vals[p] / D8[p] - 1) for p in (8, 9, 10))
                    dev5 = max(abs(vals[p] / D8[p] - 1) for p in (8, 9, 10, 11, 12))
                    combos.append(dict(variant=v, reference=r, quantity=q, point=name, i=i, j=j, dev3=dev3, dev5=dev5,
                                       **{f"X{p}": vals[p] for p in P_LIST}))
    cdf = pd.DataFrame(combos).sort_values("dev3").reset_index(drop=True)
    say("  closest ten combinations:")
    say("    variant  ref   q  point            dev3     dev5     X8      X9      X10     X11     X12     X13     X14")
    for _, r_ in cdf.head(10).iterrows():
        say(f"    {r_.variant:7s}  {r_.reference:4s}  {r_.quantity}  {r_.point:14s}  {r_.dev3:6.3f}   {r_.dev5:6.3f}   "
            + "  ".join(f"{r_[f'X{p}']:6.2f}" for p in P_LIST))
    hits = cdf[cdf.dev3 <= 0.05]
    hits_primary = hits[(hits.variant == "fix") & (hits.reference == "full")]
    within25 = cdf[cdf.dev3 <= 0.25]
    best = cdf.iloc[0]
    if len(hits):
        say(f"\n  P1  {len(hits)} combination(s) within 5 % at p = 8, 9, 10 ({len(hits_primary)} under the primary definition):")
        for _, r_ in hits.iterrows():
            ok5 = r_.dev5 <= 0.05
            say(f"      {r_.variant}/{r_.reference}/{r_.quantity} at {r_.point} [{r_.i},{r_.j}]: dev3 {r_.dev3:.3f}; p = 11, 12 give "
                f"{r_.X11:.2f}, {r_.X12:.2f} against 4.4, 3.1 (dev5 {r_.dev5:.3f}) -> {'as predicted' if ok5 else 'p = 11/12 NOT within 5 %'}")
        p1 = bool((hits.dev5 <= 0.05).any())
    else:
        p1 = False
        say("\n  P1  NOT as predicted: no combination within 5 % of 9.4 / 10.7 / 6.3")
    say(f"  P1  -> {'held' if p1 else 'NOT held'}")
    if len(within25) == 0:
        say(f"  REFUTER FIRED: no combination within 25 % of the three quoted numbers. Closest: "
            f"{best.variant}/{best.reference}/{best.quantity} at {best.point}, dev3 {best.dev3:.3f}, "
            f"X(8,9,10) = {best.X8:.2f}, {best.X9:.2f}, {best.X10:.2f}")
    else:
        say(f"  REFUTER (no combination within 25 %): did not appear; {len(within25)} combination(s) within 25 %")

    # the combination that carries the rest of the tests: the best primary-definition hit if any,
    # else the best hit of any definition, else the primary a_p at the benchmark
    if len(hits_primary):
        carry = hits_primary.iloc[0]; carry_why = "best hit under the primary definition"
    elif len(hits):
        carry = hits.iloc[0]; carry_why = "best hit (NOT the primary definition)"
    else:
        carry = cdf[(cdf.variant == "fix") & (cdf.reference == "full") & (cdf.quantity == "a") & (cdf.point == "benchmark")].iloc[0]
        carry_why = "no hit; falling back to the primary a_p at the benchmark"
    cv, cr, cq, cpt = carry.variant, carry.reference, carry.quantity, carry.point
    ci, cj = NAMED[cpt]
    say(f"\n  carried forward: {cv}/{cr}/{cq} at {cpt} [{ci},{cj}]  ({carry_why})")

    # P1a: quantity is a_p; X_n, X_c within 10 % of unity everywhere named
    say("\n  P1a (this script's expectation): the reproducing quantity is a_p and X_n, X_c are within 10 % of 1 at every p and named point")
    worst_nc = 0.0; worst_where = ""
    for name, (i, j) in NAMED.items():
        for q in ("n", "c"):
            for p in P_LIST:
                d = abs(float(X["fix"]["full"][q][p][i, j]) - 1)
                if d > worst_nc:
                    worst_nc, worst_where = d, f"X_{q}({p}) at {name}"
    p1a = (cq == "a") and (len(hits) > 0) and (worst_nc <= 0.10)
    say(f"      quantity carried: {cq};  worst |X_n - 1|, |X_c - 1| over named points and p = 8..14: {worst_nc:.4f} ({worst_where})"
        f"  -> {'as predicted' if p1a else 'NOT as predicted'}")

    # P2: order ten at the benchmark, p = 8..10
    bi, bj = NAMED["benchmark"]
    vb = [float(X[cv][cr][cq][p][bi, bj]) for p in (8, 9, 10)]
    p2 = all(3.0 <= x <= 30.0 for x in vb)
    say(f"\n  P2  {cv}/{cr}/{cq} at the benchmark, p = 8, 9, 10: " + ", ".join(f"{x:.2f}" for x in vb)
        + f"  -> {'of order ten (3..30), as predicted' if p2 else 'NOT within 3..30'}")

    # P3: monotone decrease from p = 9 upward, carried combination
    vc = [float(X[cv][cr][cq][p][ci, cj]) for p in P_LIST]
    dec = all(vc[k + 1] < vc[k] for k in range(1, len(vc) - 1))
    say(f"\n  P3  (D.8's claim) monotone decrease from p = 9 upward, {cv}/{cr}/{cq} at {cpt}: "
        + ", ".join(f"X({p}) = {x:.2f}" for p, x in zip(P_LIST, vc)) + f"  -> {'as claimed' if dec else 'NOT monotone'}")
    say(f"      X(9) {'>' if vc[1] > vc[0] else '<='} X(8): the chapter's 'no monotone trend over p = 8..10' is "
        f"{'consistent' if (vc[1] > vc[0] and vc[2] < vc[1]) else 'NOT what this run shows'}")
    say("      all named points, primary definition, X_a: " + "; ".join(
        f"{name} " + "/".join(f"{X['fix']['full']['a'][p][i, j]:.2f}" for p in P_LIST) for name, (i, j) in NAMED.items()))

    # P4: the external band
    band_lo, band_hi = min(vc[:3]), max(vc[:3])
    inside = not (ext_hi < band_lo or ext_lo > band_hi)
    same_order = (band_hi / ext_lo < 3.0) and (ext_hi / band_lo < 3.0)
    in_range_p = [p for p, x in zip(P_LIST, vc) if ext_lo <= x <= ext_hi]
    say(f"\n  P4  external r_1(15) excess {ext_lo:.3f} to {ext_hi:.3f} against the p = 8..10 band {band_lo:.2f} to {band_hi:.2f} "
        f"({cv}/{cr}/{cq} at {cpt}): {'INTERSECTS the band' if inside else 'does not sit inside the band'}; "
        f"same order (factor 3): {'yes' if same_order else 'NO'}  -> {'as predicted' if (not inside and same_order) else 'NOT as predicted'}")
    say(f"      shells whose X lies within {ext_lo:.2f}..{ext_hi:.2f}: {in_range_p if in_range_p else 'none'};  X(14) = {vc[-1]:.2f} "
        f"(caveat: the reference for p = 14 has one shell above it; X(15) is undefined in this operator)")

    # --- RESULT 3: the whole grid --------------------------------------------------
    say("\n" + "-" * 78)
    say("RESULT 3: grid-wide excess, primary definition, Te >= 2 eV (280 points)")
    say("-" * 78)
    for q in QUANT:
        for p in (8, 9, 10):
            x = X["fix"]["full"][q][p][defended]
            k = np.unravel_index(np.argmax(np.where(defended, X["fix"]["full"][q][p], -np.inf)), (nT, nN))
            say(f"  X_{q}({p:2d}): median {np.median(x):8.4f}   range {x.min():8.4f} .. {x.max():8.4f}   (max at [{k[0]},{k[1]}] Te={TeL[k[0]]:.3g} ne={neL[k[1]]:.2g})")
    xa = X["fix"]["full"]["a"]
    mono = np.ones((nT, nN), bool)
    for p in range(9, 14):
        mono &= xa[p + 1] < xa[p]
    say(f"  X_a decreasing monotonically from p = 9 to 14 at {int(mono[defended].sum())} of {int(defended.sum())} defended points"
        f" ({int(mono.sum())} of {nT*nN} over the whole grid);  X_a(9) > X_a(8) at {int((xa[9] > xa[8])[defended].sum())} of 280")
    xall = np.stack([xa[p] for p in (8, 9, 10)])
    say(f"  X_a(8..10) between 3 and 30 at {int(((xall >= 3) & (xall <= 30)).all(axis=0)[defended].sum())} of 280 defended points;"
        f"  over all 400: {int(((xall >= 3) & (xall <= 30)).all(axis=0).sum())}")

    # --- RESULT 4: the trap ---------------------------------------------------------
    say("\n" + "-" * 78)
    say("RESULT 4: diagnostic -- the trap (no diagonal correction), X_q(p) against the full operator, named points")
    say("-" * 78)
    for name, (i, j) in NAMED.items():
        say(f"  {name:14s} X_a: " + " ".join(f"{X['trap']['full']['a'][p][i, j]:6.3f}" for p in P_LIST)
            + "   X_n: " + " ".join(f"{X['trap']['full']['n'][p][i, j]:6.3f}" for p in P_LIST))

    say("\nREADING: the excess of the terminal shell over the same shell inside the ladder is as printed in\n"
        "RESULT 1; RESULT 2 states whether any definition reproduces D.8's 9.4 / 10.7 / 6.3 and which. The\n"
        "Chapter 4 sentence is to be requoted from this run with the quantity, the reference and the grid point\n"
        "named. Assumptions: removing a shell removes recombination into it; the interior reference is the\n"
        "n_max = 15 operator, itself truncated, so X(p) for p near 15 understates the excess against an infinite\n"
        "ladder and X(15) is not defined here.")

    if args.write:
        out = Path(args.out) if args.out else root / "validation/terminal_vs_interior"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}",
               f"# interpreter {sys.executable}  numpy {np.__version__}",
               f"# L_grid.npy      sha256 {sha256_file(root / 'data/processed/cr_matrix/L_grid.npy')}",
               f"# S_grid.npy      sha256 {sha256_file(S_path)}",
               f"# state_index.csv sha256 {sha256_file(ctx.state_index_path)}",
               f"# molecular_channel.csv sha256 {sha256_file(mol_path)}  (G1 gate)",
               f"# fujimoto_table41.csv sha256 {sha256_file(fuji_path)}  (external r_1(15) excess {ext_lo:.4f}, {ext_hi:.4f})",
               f"# benchmark point [{bi},{bj}]  Te={TeL[bi]:.4f} eV  ne={neL[bj]:.4e} cm^-3",
               "# X_q_full = q(n_max = p) / q(n_max = 15); X_q_next = q(n_max = p) / q(n_max = p + 1); trap = no diagonal correction"]
        rows = []
        for p in P_LIST:
            for i in range(nT):
                for j in range(nN):
                    row = dict(p=p, i=i, j=j, Te=TeL[i], ne=neL[j], defended=bool(defended[i, j]),
                               u_CRE_term=U["fix"][p][i, j], u_CRE_full=U["fix"][NMAX_FULL][i, j])
                    for q in QUANT:
                        row[f"{q}_term"] = SH["fix"][p][q][i, j, p]
                        row[f"{q}_int_full"] = SH["fix"][NMAX_FULL][q][i, j, p]
                        row[f"{q}_int_next"] = SH["fix"][p + 1][q][i, j, p]
                        row[f"X_{q}_full"] = X["fix"]["full"][q][p][i, j]
                        row[f"X_{q}_next"] = X["fix"]["next"][q][p][i, j]
                        row[f"X_{q}_trap_full"] = X["trap"]["full"][q][p][i, j]
                    rows.append(row)
        with open(out / "terminal_vs_interior.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "terminal_vs_interior_named.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n" + "# D8 = " + ", ".join(f"p{p}:{v}" for p, v in D8.items()) + "\n")
            cdf.to_csv(fh, index=False)
        with open(out / "terminal_vs_interior.txt", "w") as fh:
            fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/terminal_vs_interior{{.csv,_named.csv,.txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
