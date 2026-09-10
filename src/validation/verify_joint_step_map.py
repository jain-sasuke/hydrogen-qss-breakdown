#!/usr/bin/env python
"""
verify_joint_step_map.py
========================
Step BOTH plasma parameters, not just the temperature.

WHY THIS EXISTS
---------------
Every map in Chapter 5 steps Te at fixed ne. A real edge-localised mode raises
the electron density as well, and chapter5.tex and chapter6.tex both record the
joint map as spot-checked but not run, with a todo saying an examiner familiar
with ELM physics will ask for it. This runs it.

WHAT IS AT STAKE, STATED BEFORE THE RESULT
------------------------------------------
The framework's claim is that the plateau error factorises as

    eps = | exp( Sbar * dln u ) - 1 |

where Sbar belongs to the OBSERVABLE, being a property of the post-step
operator at fixed conditions, and dln u belongs to the PLASMA, being however
far the reservoir was displaced. If that is right, the identity must hold
unchanged when the displacement is produced by a density step instead of a
temperature step, or by both at once. Nothing in the derivation refers to what
moved the reservoir.

The refuting observation, stated in advance: if the identity fails, or if Sbar
measured on a joint step differs from Sbar measured on a Te-only step at the
same post-step operator, then Sbar is not a property of the operator and the
decomposition is a curve fit.

The second question is quantitative and runs the other way. A density rise
shortens tau_slow, and the ELM-averaged bound falls with tau_slow at fixed
event duration. So a joint step may well produce a LARGER instantaneous error
and a SMALLER time-averaged one. Chapter 5 records a spot check at the
benchmark showing exactly that, a factor 4 fall. This script asks whether the
census survives it.

CONVENTIONS, AND ONE HONEST LIMITATION
--------------------------------------
L is tabulated only on the grid, so every step is an integer number of grid
indices in each axis. The two axes are not comparable in size: one temperature
index is +4.81 percent, one density index is +168 percent, because 8 density
points span three decades while 50 temperature points span one. A step of one
index in each is therefore NOT a balanced ELM, and no combination available
here is. The script prints both step sizes with every result so that no number
can be read without them.

Read-only. Writes only to validation/joint_step/, and only with --write.
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

WIN_LO = WIN_HI = 30.0          # same plateau-window test as the Te-only map
TAU_DRIVE = 100e-6              # s, the ELM crash duration used in Chapter 5


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--dT", type=int, nargs="+", default=[0, 1, 2, -1, -2],
                    help="temperature steps, in grid indices")
    ap.add_argument("--dn", type=int, nargs="+", default=[-1, 0, 1],
                    help="density steps, in grid indices")
    ap.add_argument("--tau-drive", type=float, default=TAU_DRIVE)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--te-floor", type=float, default=2.0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    nv = ctx.n_values
    E = np.array([i for i in range(ctx.n_states) if i != g])
    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    if len(N3) == 0 or len(N4) == 0:
        raise RuntimeError("no n=3 or n=4 states in the state index")
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
    print(f"  tau_drive {a.tau_drive*1e6:.4g} us   plateau window "
          f"M > {WIN_LO*WIN_HI:.0f}")
    print()
    dlnTe1 = float(np.log(te[1] / te[0]))
    dlnne1 = float(np.log(ne[1] / ne[0]))
    print(f"  one temperature index = {100*np.expm1(dlnTe1):+.4g}%  "
          f"(dlnTe = {dlnTe1:.6f})")
    print(f"  one density index     = {100*np.expm1(dlnne1):+.4g}%  "
          f"(dln ne = {dlnne1:.6f})")
    print("  The two axes differ by a factor 35 in logarithmic step. Every")
    print("  number below is quoted with the step that produced it.")
    print()

    rows = []
    worst_sup = 0.0
    worst_id = 0.0
    for kT in a.dT:
        for kn in a.dn:
            for i in range(len(te)):
                ii = i + kT
                if not (0 <= ii < len(te)):
                    continue
                for j in range(len(ne)):
                    jj = j + kn
                    if not (0 <= jj < len(ne)):
                        continue
                    if kT == 0 and kn == 0:
                        continue          # the null step has nothing to say
                    A = L[ii, jj]
                    lam = np.sort(np.linalg.eigvals(A).real)[::-1]
                    neg = lam[lam < 0]
                    if len(neg) < 2:
                        raise RuntimeError(
                            f"fewer than two negative eigenvalues at "
                            f"[{ii},{jj}]; the operator is not stable")
                    t_slow, t_rel = 1.0 / abs(neg[0]), 1.0 / abs(neg[1])
                    M = t_slow / t_rel
                    window_ok = (WIN_LO * t_rel) < (t_slow / WIN_HI)

                    n_old = np.linalg.solve(L[i, j], -S[i, j])
                    n_new = np.linalg.solve(A, -S[ii, jj])
                    LEE = A[np.ix_(E, E)]
                    LEg = A[np.ix_(E, [g])].ravel()
                    n0 = np.linalg.solve(LEE, -S[ii, jj][E])
                    n1 = np.linalg.solve(LEE, -LEg * n_old[g])

                    sup = (np.abs(n0 + (n_new[g] / n_old[g]) * n1
                                  - n_new[E]).max() / np.abs(n_new[E]).max())
                    worst_sup = max(worst_sup, sup)
                    if sup > 1e-8:
                        raise RuntimeError(
                            f"two-channel split not exact at [{i},{j}] "
                            f"dT={kT} dn={kn}: residual {sup:.3e}")

                    lnx = float(np.log(n_new[g] / n_old[g]))
                    a3, a4 = n1[n3E].sum(), n1[n4E].sum()
                    c3, c4 = n0[n3E].sum(), n0[n4E].sum()
                    R_pe = (c3 + a3) / (c4 + a4)
                    R_new = n_new[N3].sum() / n_new[N4].sum()
                    eps = abs(R_pe / R_new - 1.0)
                    Sbar = float(np.log(R_pe / R_new) / lnx) if lnx != 0 else 0.0

                    pred = abs(np.expm1(Sbar * lnx))
                    if eps > 0:
                        worst_id = max(worst_id, abs(pred - eps) / eps)

                    x = a.tau_drive / t_slow
                    lo = eps * (1.0 / x) * (1.0 - np.exp(-x)) if x > 0 else eps

                    rows.append(dict(
                        dT=kT, dn=kn, i=i, j=j, Te=float(te[i]),
                        ne=float(ne[j]), Te_new=float(te[ii]),
                        ne_new=float(ne[jj]),
                        dlnTe=float(np.log(te[ii] / te[i])),
                        dlnne=float(np.log(ne[jj] / ne[j])),
                        dlnu=lnx, Sbar=Sbar, eps=eps, tau_slow=t_slow, M=M,
                        window_ok=bool(window_ok), lo_ELM=float(lo)))

    print("=" * 78)
    print("THE TWO EXACTNESS CHECKS (wiring, not physics)")
    print("=" * 78)
    print(f"  worst two-channel superposition residual over "
          f"{len(rows)} cases: {worst_sup:.3e}")
    print(f"  worst departure from eps = |exp(Sbar dln u) - 1|: {worst_id:.3e}")
    print("  The identity holds when the reservoir is displaced by a density")
    print("  step, by a temperature step, or by both. It never referred to")
    print("  which one, and the numbers confirm the algebra rather than")
    print("  discovering anything.")
    print()

    ok = [r for r in rows if r["window_ok"]]
    print("=" * 78)
    print("IS Sbar A PROPERTY OF THE OPERATOR, AS CLAIMED?")
    print("=" * 78)
    print("  Sbar is defined on the POST-step operator. If it belongs to the")
    print("  operator, then two different steps landing on the same (Te, ne)")
    print("  must give the same Sbar. Comparing every pair of steps that share")
    print("  a destination:")
    by_dest = {}
    for r in ok:
        by_dest.setdefault((r["i"] + r["dT"], r["j"] + r["dn"]), []).append(r)
    spreads = []
    for dest, rs in by_dest.items():
        if len(rs) < 2:
            continue
        s = np.array([abs(x["Sbar"]) for x in rs])
        spreads.append(s.max() / s.min())
    spreads = np.array(spreads)
    print(f"    {len(spreads)} destinations reached by two or more steps")
    print(f"    spread of |Sbar| at a shared destination: median "
          f"{np.median(spreads):.4f}, 90th pct {np.percentile(spreads,90):.4f}, "
          f"max {spreads.max():.4f}")
    print("    Sbar is a MEAN sensitivity over the excursion, so it is not")
    print("    expected to be identical across steps of different size; it")
    print("    tends to f3 - f4 of the post-step operator only as the step")
    print("    shrinks. A spread near 1 is the claim; a large spread would")
    print("    mean the coefficient is a fit.")
    print()

    print("=" * 78)
    print("WHAT A DENSITY RISE DOES, AT FIXED TEMPERATURE STEP")
    print("=" * 78)
    print(f"  {'dT':>3} {'dn':>3}  {'n':>5}  {'median eps':>11}  "
          f"{'median tau_slow':>15}  {'median ELM bound':>17}  "
          f"{'bound > 10%':>11}")
    for kT in a.dT:
        for kn in a.dn:
            sub = [r for r in ok if r["dT"] == kT and r["dn"] == kn]
            if not sub:
                continue
            e = np.array([r["eps"] for r in sub])
            ts = np.array([r["tau_slow"] for r in sub])
            lo = np.array([r["lo_ELM"] for r in sub])
            print(f"  {kT:>3} {kn:>3}  {len(sub):>5}  {np.median(e):>11.4g}  "
                  f"{np.median(ts):>15.4g}  {np.median(lo):>17.4g}  "
                  f"{int((lo > a.threshold).sum()):>5}/{len(sub):<5}")
    print()
    print("  A density rise raises the instantaneous error and shortens")
    print("  tau_slow. The ELM-averaged bound carries both, and the table says")
    print("  which wins.")
    print()

    print("=" * 78)
    print(f"THE CENSUS UNDER JOINT STEPS, Te >= {a.te_floor} eV, "
          f"tau_drive = {a.tau_drive*1e6:.4g} us")
    print("=" * 78)
    for kn in a.dn:
        sub = [r for r in ok if r["dn"] == kn and r["Te"] >= a.te_floor
               and r["dT"] == 1]
        if not sub:
            continue
        lo = np.array([r["lo_ELM"] for r in sub])
        k = int(np.argmax(lo))
        print(f"  dT = +1, dn = {kn:+d}:  {int((lo > a.threshold).sum())} of "
              f"{len(sub)} exceed {100*a.threshold:.0f}%, worst {lo[k]:.4f} at "
              f"Te = {sub[k]['Te']:.3g} eV, ne = {sub[k]['ne']:.3g} cm^-3")
    print()
    print("  Combined over both temperature directions, which is how chapter 5")
    print("  counts (point-and-direction pairs):")
    for kn in a.dn:
        sub = [r for r in ok if r["dn"] == kn and r["Te"] >= a.te_floor
               and abs(r["dT"]) == 1]
        if not sub:
            continue
        lo = np.array([r["lo_ELM"] for r in sub])
        k = int(np.argmax(lo))
        print(f"    dn = {kn:+d}:  {int((lo > a.threshold).sum())} of "
              f"{len(sub)} exceed {100*a.threshold:.0f}%, worst {lo[k]:.4f} "
              f"at Te = {sub[k]['Te']:.3g} eV, ne = {sub[k]['ne']:.3g}, "
              f"dT = {sub[k]['dT']:+d}")
    print()
    print("  The benchmark point [23,5], which chapter 5 spot-checked:")
    for kT in (1,):
        for kn in a.dn:
            m = [r for r in rows if r["i"] == 23 and r["j"] == 5
                 and r["dT"] == kT and r["dn"] == kn]
            if not m:
                continue
            r = m[0]
            print(f"    dT = {kT:+d}, dn = {kn:+d}:  eps = {r['eps']:.4f}, "
                  f"tau_slow = {r['tau_slow']*1e6:.4g} us, ELM bound = "
                  f"{r['lo_ELM']:.4f}   (Te {r['Te']:.3g} -> {r['Te_new']:.3g} eV, "
                  f"ne {r['ne']:.3g} -> {r['ne_new']:.3g})")
    print()
    print("=" * 78)
    print("DOES THE GAIN SPLIT INTO TWO COMPONENTS?")
    print("=" * 78)
    print("  The Te-only chapters factorise dln u = G dlnTe with a single")
    print("  gain. A joint step needs a gradient. Measuring the two partial")
    print("  gains from the single-axis steps at each starting point,")
    print("    G_T = dln u / dlnTe   at dn = 0,")
    print("    G_n = dln u / dln ne  at dT = 0,")
    print("  and predicting the joint displacement as G_T dlnTe + G_n dln ne:")
    idx = {(r["dT"], r["dn"], r["i"], r["j"]): r for r in rows}
    errs, gts, gns = [], [], []
    for r in rows:
        if r["dT"] == 0 or r["dn"] == 0:
            continue
        rt = idx.get((r["dT"], 0, r["i"], r["j"]))
        rn = idx.get((0, r["dn"], r["i"], r["j"]))
        if rt is None or rn is None:
            continue
        G_T = rt["dlnu"] / rt["dlnTe"]
        G_n = rn["dlnu"] / rn["dlnne"]
        pred = G_T * r["dlnTe"] + G_n * r["dlnne"]
        if r["dlnu"] != 0:
            errs.append(abs(pred / r["dlnu"] - 1.0))
        gts.append(G_T); gns.append(G_n)
    errs = np.array(errs); gts = np.array(gts); gns = np.array(gns)
    print(f"    {len(errs)} joint steps with both single-axis partners")
    print(f"    |G_T| runs {np.abs(gts).min():.4g} to {np.abs(gts).max():.4g}, "
          f"median {np.median(np.abs(gts)):.4g}")
    print(f"    |G_n| runs {np.abs(gns).min():.4g} to {np.abs(gns).max():.4g}, "
          f"median {np.median(np.abs(gns)):.4g}")
    print(f"    relative error of the additive prediction: median "
          f"{np.median(errs)*100:.3g}%, 90th pct "
          f"{np.percentile(errs,90)*100:.3g}%, max {errs.max()*100:.3g}%")
    big = np.array([e for e in errs if e > 0.05])
    print(f"    {len(big)} of {len(errs)} exceed 5%. Those sit where the two")
    print("    contributions nearly cancel: heating empties the reservoir and")
    print("    a density rise refills it, so dln u passes through zero and a")
    print("    relative error on a vanishing quantity blows up. The same")
    print("    caution applies here as to the amplification ratio in chapter 5.")
    dl = np.array([abs(r["dlnu"]) for r in rows
                   if r["dT"] != 0 and r["dn"] != 0
                   and (r["dT"], 0, r["i"], r["j"]) in idx
                   and (0, r["dn"], r["i"], r["j"]) in idx and r["dlnu"] != 0])
    m = errs > 0.05
    if m.any():
        print(f"    |dln u| at those points: median {np.median(dl[m]):.4g}, "
              f"against {np.median(dl[~m]):.4g} elsewhere")
    print("    Additivity is a first-order statement and these are finite")
    print("    steps, one of which is +168%, so a departure is expected. The")
    print("    size of it is what decides whether a two-component gain is")
    print("    worth tabulating or whether the joint case must be solved.")
    print()

    print("  The dn = 0 row must reproduce chapter 5's Table 6.4 middle row,")
    print("  45 of 448, because it is the same calculation. If it does not,")
    print("  this script and the Te-only map disagree and one is wrong.")
    print()

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "joint_step"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "joint_step_map.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {out/'joint_step_map.csv'}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
