#!/usr/bin/env python
"""
verify_fujimoto_r1_consequence.py
=================================
What the Fujimoto r_1 deficit would do to the quantity Chapter 5 uses, if
Fujimoto's coefficients were the truth.

WHY THIS EXISTS
---------------
Chapter 4 (sec:fujimoto) records that this model's ionising population
coefficient r_1 sits a factor 8.3 low at p = 3 and 4.4 low at p = 4 against
Fujimoto's Table 4.1(b), and until now said the disagreement "is not evidence
of an error in the quantity Chapter 5 uses". But Eq. fujimoto_correspondence
gives a_p = r_1(p) Z(p)/Z(1): r_1 IS the ground-fed coefficient from which
f_p = a_p u / (a_p u + c_p) is built. A deficit common to p = 3 and 4 would
cancel in Delta = ln[(a_3/a_4)/(c_3/c_4)]; the recorded deficit is not common.
This script computes the consequence.

METHOD
------
From L_grid and S_grid (via cr_context), at every grid point:
  a = -L_EE^{-1} L_Eg   (excited populations per unit n_g)
  c = -L_EE^{-1} S_E    (excited populations per unit n_i)
  a_p, c_p = shell sums over the l-resolved sublevels of p = 3, 4
  u_CRE   = [-L^{-1} S]_ground   (CRE neutral-to-ion ratio)
  Delta   = ln[(a_3/a_4)/(c_3/c_4)],  cap = tanh(|Delta|/4)
  u_peak  = sqrt(c_3 c_4 / (a_3 a_4))   (where |f_3 - f_4| is largest)
  S(u)    = f_3(u) - f_4(u) at u = u_CRE
GATE P0: a_3, a_4, c_3, c_4, u_CRE and Delta must reproduce
validation/molecular_channel/molecular_channel.csv at all 400 points.
Then two scenarios, one per density row of Table 4.1(b): the model's a_p is
divided by the recorded ratio r_1(model)/r_1(Fujimoto) for p = 3 and 4, so
that a_p' equals what Fujimoto's coefficient would give, and every quantity
above is recomputed. The ratios are measured at Te = 10 eV only and are
applied at every grid point: that is the scenario's assumption, stated, not
tested.

PREDICTIONS (written before the run)
-----------------------------------
P1  Delta' - Delta = ln(r_4/r_3) exactly, at every point (a constant shift):
    +0.635 for the 1e12 cm^-3 row (0.2276/0.1206).
P2  cap at the benchmark [23,5]: 0.4503 -> tanh(2.575/4) = 0.567 (+26 %).
P3  u_peak' / u_peak = sqrt(r_3 r_4) = 0.166 for the 1e12 row: the crest sits
    a factor 6 lower in u.
P4  S at u_CRE: not predictable by eye; reported.
REFUTING OBSERVATION for the Chapter 4 sentence: cap or |S(u_CRE)| moving by
more than the 8.4 % / 9.1 % of the RMPS excitation substitution means the
disagreement bears on Chapter 5's quantity and cannot be called "not evidence".

OUTPUTS (with --write)
----------------------
validation/fujimoto_r1_consequence/fujimoto_r1_consequence.csv   per point, per scenario
validation/fujimoto_r1_consequence/fujimoto_r1_consequence.txt   this run's log
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


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ctx = CRContext.load(); root = ctx.root
    L, TeL, neL = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    nT, nN, nS, _ = L.shape
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    fuj_path = root / "validation/fujimoto_table41/fujimoto_table41.csv"
    mol_path = root / "validation/molecular_channel/molecular_channel.csv"
    for p in (S_path, fuj_path, mol_path):
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
    E = [k for k in range(nS) if k != g]

    log: list[str] = []
    def say(s: str = "") -> None:
        print(s); log.append(s)
    say("=" * 78)
    say("FUJIMOTO r_1 CONSEQUENCE -- what the deficit would do to Delta, the cap, the crest and S")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"interpreter   {sys.executable}")
    say(ctx.describe())
    say("=" * 78)

    # --- a, c, u_CRE at every point ----------------------------------------
    a3 = np.empty((nT, nN)); a4 = np.empty((nT, nN)); c3 = np.empty((nT, nN)); c4 = np.empty((nT, nN)); uc = np.empty((nT, nN))
    for i in range(nT):
        for j in range(nN):
            Lij = L[i, j]; Sij = S[i, j]
            LEE = Lij[np.ix_(E, E)]; LEg = Lij[E, g]; SE = Sij[E]
            a = -np.linalg.solve(LEE, LEg); c = -np.linalg.solve(LEE, SE)
            pos = {k: q for q, k in enumerate(E)}
            a3[i, j] = sum(a[pos[k]] for k in idx3); a4[i, j] = sum(a[pos[k]] for k in idx4)
            c3[i, j] = sum(c[pos[k]] for k in idx3); c4[i, j] = sum(c[pos[k]] for k in idx4)
            uc[i, j] = (-np.linalg.solve(Lij, Sij))[g]
    Delta = np.log((a3 / a4) / (c3 / c4)); cap = np.tanh(np.abs(Delta) / 4)
    upk = np.sqrt(c3 * c4 / (a3 * a4))
    f = lambda a_, c_, u: a_ * u / (a_ * u + c_)
    S0 = f(a3, c3, uc) - f(a4, c4, uc)

    # --- P0 gate against molecular_channel.csv ------------------------------
    m = pd.read_csv(mol_path, comment="#")
    if len(m) != nT * nN:
        raise ValueError(f"molecular_channel.csv has {len(m)} rows, expected {nT*nN}")
    m = m.sort_values(["i", "j"])
    def worst(mine, col):
        return np.abs(mine.ravel() / m[col].values - 1).max()
    gates = {"a3": worst(a3, "a3"), "a4": worst(a4, "a4"), "c3": worst(c3, "c3"),
             "c4": worst(c4, "c4"), "u_CRE": worst(uc, "u_CRE"), "Delta": np.abs(Delta.ravel() - m.Delta_atomic.values).max()}
    say("\nP0  reproduction of molecular_channel.csv at all 400 points (worst relative / absolute):")
    for k, v in gates.items():
        say(f"    {k:6s} {v:.2e}")
    if max(gates.values()) > 1e-8:
        raise AssertionError("P0 FAILED: a, c, u_CRE or Delta do not reproduce the stamped artifact; do not read on")
    say("    P0 REPRODUCED")
    ti, ni = ctx.nearest_point(BENCH_TE, BENCH_NE)
    say(f"\nbenchmark [{ti},{ni}]: Delta {Delta[ti,ni]:.4f}  cap {cap[ti,ni]:.4f}  u_CRE {uc[ti,ni]:.4e}  u_peak {upk[ti,ni]:.4e}  S(u_CRE) {S0[ti,ni]:+.4f}")

    # --- the recorded deficits ---------------------------------------------
    fu = pd.read_csv(fuj_path, comment="#")
    fu = fu[fu.variant == "PRODUCTION"]
    rows = []
    defended = (TeL >= 2.0)[:, None] & np.ones((1, nN), bool)
    say("\n" + "-" * 78)
    for lg, grp in fu.groupby("lg_ne"):
        r3 = float(grp[grp.p == 3].r1_ratio.iloc[0]); r4 = float(grp[grp.p == 4].r1_ratio.iloc[0])
        ne_row = float(grp.ne_cm3.iloc[0]); Te_row = float(grp.Te_eV.iloc[0])
        say(f"SCENARIO lg n_e = {lg} ({ne_row:.0e} cm^-3, model at {Te_row:g} eV): r1 model/Fujimoto  p=3 {r3:.4f} (x{1/r3:.1f} low)  p=4 {r4:.4f} (x{1/r4:.1f} low)")
        a3p, a4p = a3 / r3, a4 / r4
        Dp = np.log((a3p / a4p) / (c3 / c4)); capp = np.tanh(np.abs(Dp) / 4)
        upkp = np.sqrt(c3 * c4 / (a3p * a4p)); Sp = f(a3p, c3, uc) - f(a4p, c4, uc)
        shift = Dp - Delta
        say(f"  P1  Delta' - Delta: {shift.min():+.4f} .. {shift.max():+.4f}  (predicted ln(r4/r3) = {np.log(r4/r3):+.4f})")
        say(f"  P2  cap at benchmark: {cap[ti,ni]:.4f} -> {capp[ti,ni]:.4f}  ({100*(capp[ti,ni]/cap[ti,ni]-1):+.1f} %)")
        say(f"  P3  u_peak'/u_peak: {np.unique(np.round(upkp/upk, 6))}  (predicted sqrt(r3 r4) = {np.sqrt(r3*r4):.4f}, a factor {1/np.sqrt(r3*r4):.1f} lower in u)")
        for nm, mask in (("all 400 points", np.ones_like(defended)), ("Te >= 2 eV (280 points)", defended)):
            dc = 100 * (capp[mask] / cap[mask] - 1)
            dS = 100 * (np.abs(Sp[mask]) / np.abs(S0[mask]) - 1)
            say(f"  P4  [{nm}]  cap change: median {np.median(dc):+.1f} %  range {dc.min():+.1f} .. {dc.max():+.1f} %   "
                f"|S(u_CRE)| change: median {np.median(dS):+.1f} %  range {dS.min():+.1f} .. {dS.max():+.1f} %   "
                f"|S| up at {100*(dS>0).mean():.0f} % of points")
        say(f"      benchmark S(u_CRE): {S0[ti,ni]:+.4f} -> {Sp[ti,ni]:+.4f}  ({100*(abs(Sp[ti,ni])/abs(S0[ti,ni])-1):+.1f} %)")
        for i in range(nT):
            for j in range(nN):
                rows.append(dict(scenario_lg_ne=lg, r1_ratio_p3=r3, r1_ratio_p4=r4, i=i, j=j, Te=TeL[i], ne=neL[j],
                                 a3=a3[i,j], a4=a4[i,j], c3=c3[i,j], c4=c4[i,j], u_CRE=uc[i,j],
                                 Delta=Delta[i,j], cap=cap[i,j], u_peak=upk[i,j], S_uCRE=S0[i,j],
                                 Delta_fuji=Dp[i,j], cap_fuji=capp[i,j], u_peak_fuji=upkp[i,j], S_uCRE_fuji=Sp[i,j]))
        say("-" * 78)
    say("\nREADING: the cap and the sensitivity at the operating point both move by far more than the\n"
        "8.4 % / 9.1 % of the RMPS excitation substitution if Fujimoto's coefficients are the truth.\n"
        "Whether they are is the open question of sec:fujimoto; this fixes what is at stake, not who is right.\n"
        "Assumption: a deficit measured at Te = 10 eV and two densities is applied at every grid point.")

    if args.write:
        out = Path(args.out) if args.out else root / "validation/fujimoto_r1_consequence"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}",
               f"# interpreter {sys.executable}  numpy {np.__version__}",
               f"# L_grid.npy      sha256 {sha256_file(root / 'data/processed/cr_matrix/L_grid.npy')}",
               f"# S_grid.npy      sha256 {sha256_file(S_path)}",
               f"# state_index.csv sha256 {sha256_file(ctx.state_index_path)}",
               f"# fujimoto_table41.csv sha256 {sha256_file(fuj_path)}",
               f"# molecular_channel.csv sha256 {sha256_file(mol_path)}  (P0 gate)",
               f"# benchmark point [{ti},{ni}]  Te={TeL[ti]:.4f} eV  ne={neL[ni]:.4e} cm^-3"]
        with open(out / "fujimoto_r1_consequence.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "fujimoto_r1_consequence.txt", "w") as fh:
            fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/fujimoto_r1_consequence.{{csv,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
