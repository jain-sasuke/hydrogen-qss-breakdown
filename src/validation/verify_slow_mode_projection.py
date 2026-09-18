#!/usr/bin/env python
"""
verify_slow_mode_projection.py
==============================
The one hypothesis behind "the shell ratio cannot decay faster than tau_slow",
measured at every grid point: S(u_CRE) = f3 - f4 != 0, and the slow eigenvector
of L projected onto ln R reproduces S/u*.

WHY THIS EXISTS
---------------
Chapter 4, section 4.8.2 (thesis_tex/chapter4.tex, "no intermediate mode",
~line 975) argues that an observable depending on the excited populations only
through their ratio cannot decay faster than tau_slow, because the spectrum of
L has one slow eigenvalue and then a band. Given QSS and a one-dimensional slow
subspace that is a theorem with exactly ONE hypothesis: the shell ratio must
actually see the slow coordinate u = n_g/n_ion, i.e.

    ln R - ln R_CRE = (S/u*) delta u + O(delta u^2),   S = f3 - f4,

with f_p = a_p u/(a_p u + c_p) the ground-fed fraction of shell p. If S were
zero anywhere, the ratio would be blind to the slow mode there and its decay
would be governed by the band alone, so the sentence would be false at that
point. No script measured S over the grid or checked that the slow eigenvector
of L, projected onto ln R, has the slope S/u* that the QSS argument assumes.

METHOD
------
  Split L into ground g and excited block E; S_grid is the recombination source
  per unit n_ion.  At each of the 400 grid points:
    a = -L_EE^{-1} L_Eg,   c = -L_EE^{-1} S_E          (n_E/n_ion = a u + c)
    a_p, c_p = shell sums over the l-resolved n=p sublevels, p = 3, 4
    u_CRE = [-L^{-1} S]_g,  f_p = a_p u/(a_p u + c_p),  S = f3 - f4
    Delta = ln[(a3/a4)/(c3/c4)]  (the two-channel contrast of the earlier scripts)
  Slow mode: eigenvalue of L with the smallest |Re| (numpy eig), eigenvector v.
  Projection coefficient  P = (v3/n3* - v4/n4*) / (v_g/n_ion), with n* the CRE
  populations and v3, v4 shell sums; QSS predicts P = S/u*.
  Two eigenvectors are compared:
    raw      numpy's eigenvector as returned;
    refined  the eigenvalue polished by Newton on the Schur scalar
             h(lam) = L_gg - lam - L_gE (L_EE - lam I)^{-1} L_Eg = 0
             and v_E = -(L_EE - lam I)^{-1} L_Eg (v_g = 1), which is the exact
             eigenvector for that eigenvalue and avoids the loss of relative
             accuracy in the tiny excited components of numpy's vector.
  The refined vector differs from the QSS vector a only through lam in the
  resolvent, so its deviation from S/u* is the genuine O(lam_slow/lam_2)
  correction; the raw vector adds numpy's floor on top of it.

GATES (before any result is reported)
------------------------------------
  A  a3, a4, c3, c4, u_CRE and Delta reproduce validation/molecular_channel/
     molecular_channel.csv at all 400 points to 1e-8 relative
  B  every CRE population is positive at every point
  C  the slow eigenvalue is real (|Im| < 1e-6 |Re|) and simple (lam_2 != lam_1)
     at every point; refined eigenpair residual ||L v - lam v|| / (||L|| ||v||)
     < 1e-12 at every point

PREDICTIONS (written before the run; values from the read-only reconstruction
adjudicate.py of 17 Sep 2026, which used the raw numpy eigenvector)
-----------------------------------------------------------------------------
  P1  S > 0 at 400/400 points
  P2  min S = 0.046 at [44,7]; max S = 0.493
  P3  raw projection coefficient equals S/u* to 1.2e-4 at [23,5] and to
      1.5e-8 at [0,4]; the raw deviation tracks lam_slow/lam_2 where that
      ratio is above numpy's floor (~1e-8) and sits on the floor below it
  P4  refined deviation is within a factor of 10 of |lam_slow/lam_2| at every
      grid point (the correction is first order in the eigenvalue)

REFUTING OBSERVATION
--------------------
  S <= 0 at any grid point, or the projection coefficient (either vector)
  deviating from S/u* by more than 1 % anywhere on the grid.  Either would
  mean the shell ratio does not ride the slow coordinate as section 4.8.2
  assumes and the "no intermediate mode" argument is unsupported there.

OUTPUTS (with --write): validation/slow_mode_projection/slow_mode_projection.{csv,txt}
Read-only on the pipeline.
"""
from __future__ import annotations
import argparse, hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; g, nv = ctx.ground_index, np.asarray(ctx.n_values); nT, nN, nS, _ = L.shape
    P = lambda *p: ROOT.joinpath(*p)
    S_path = P("data/processed/cr_matrix/S_grid.npy"); mol_path = P("validation/molecular_channel/molecular_channel.csv")
    for p in (S_path, mol_path):
        if not p.is_file(): raise FileNotFoundError(f"missing input: {p}")
    S = np.load(S_path)
    if S.shape != L.shape[:3]: raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("SLOW-MODE PROJECTION -- does the shell ratio ride the slow coordinate?")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}  numpy {np.__version__}")
    say(ctx.describe()); say(f"  L_grid sha256 {sha(P('data/processed/cr_matrix/L_grid.npy'))}"); say(f"  S_grid sha256 {sha(S_path)}")
    say(f"  state_index sha256 {sha(ctx.state_index_path)}"); say(f"  molecular_channel.csv sha256 {sha(mol_path)}"); say("=" * 78)

    E = np.array([i for i in range(nS) if i != g]); pos = {s: k for k, s in enumerate(E)}
    N3 = np.where(nv == 3)[0]; N4 = np.where(nv == 4)[0]
    if len(N3) != 3 or len(N4) != 4: raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")
    n3E = np.array([pos[s] for s in N3]); n4E = np.array([pos[s] for s in N4])
    say(f"\nn=3 sublevels {[ctx.labels[i] for i in N3]}   n=4 sublevels {[ctx.labels[i] for i in N4]}   ground {ctx.labels[g]}")

    def schur_refine(A, lam0, iters=20):
        """Newton on h(lam) = A_gg - lam - A_gE (A_EE - lam)^-1 A_Eg; returns (lam, v) with v_g = 1."""
        AEE, AEg, AgE = A[np.ix_(E, E)], A[E, g], A[g, E]; lam = float(lam0); I = np.eye(len(E))
        for _ in range(iters):
            w = np.linalg.solve(AEE - lam * I, AEg); h = A[g, g] - lam - AgE @ w
            dh = -1.0 - AgE @ np.linalg.solve(AEE - lam * I, w); step = -h / dh; lam += step
            if abs(step) < 1e-15 * max(abs(lam), 1e-300): break
        v = np.empty(nS); v[g] = 1.0; v[E] = -np.linalg.solve(AEE - lam * I, AEg)
        return lam, v

    rows = []; worstA = 0.0; worstA_where = None
    mol = pd.read_csv(mol_path, comment="#").sort_values(["i", "j"]).reset_index(drop=True)
    if len(mol) != nT * nN: raise RuntimeError(f"molecular_channel.csv has {len(mol)} rows, expected {nT * nN}")
    for i in range(nT):
        for j in range(nN):
            A, s = L[i, j], S[i, j]; LEE = A[np.ix_(E, E)]
            sol = np.linalg.solve(LEE, -np.column_stack([A[E, g], s[E]])); av, cv = sol[:, 0], sol[:, 1]
            nstar = np.linalg.solve(A, -s)                        # CRE populations per unit n_ion
            if nstar.min() <= 0: raise RuntimeError(f"Gate B: non-positive CRE population at [{i},{j}]")
            u = float(nstar[g]); a3, a4 = av[n3E].sum(), av[n4E].sum(); c3, c4 = cv[n3E].sum(), cv[n4E].sum()
            f3, f4 = a3 * u / (a3 * u + c3), a4 * u / (a4 * u + c4); Sv = f3 - f4; Delta = np.log((a3 / a4) / (c3 / c4))
            r = mol.iloc[i * nN + j]
            if int(r.i) != i or int(r.j) != j: raise RuntimeError(f"molecular_channel.csv row order broken at [{i},{j}]")
            for mine, ref in ((a3, r.a3), (a4, r.a4), (c3, r.c3), (c4, r.c4), (u, r.u_CRE), (Delta, r.Delta_atomic)):
                d = abs(mine / ref - 1)
                if d > worstA: worstA, worstA_where = d, (i, j)
            w, V = np.linalg.eig(A); order = np.argsort(-w.real); w = w[order]; V = V[:, order]
            lam1, lam2, lam3 = w[0], w[1], w[2]
            if abs(lam1.imag) > 1e-6 * abs(lam1.real) or lam1.real == lam2.real: raise RuntimeError(f"Gate C: slow eigenvalue not real/simple at [{i},{j}]: {w[:3]}")
            vraw = V[:, 0].real; vraw = vraw / vraw[g]
            proj = lambda v: (v[N3].sum() / nstar[N3].sum() - v[N4].sum() / nstar[N4].sum()) / (v[g] / 1.0)
            lam_ref, vref = schur_refine(A, lam1.real)
            res_raw = np.linalg.norm(A @ vraw - lam1.real * vraw) / (np.linalg.norm(A) * np.linalg.norm(vraw))
            res_ref = np.linalg.norm(A @ vref - lam_ref * vref) / (np.linalg.norm(A) * np.linalg.norm(vref))
            if res_ref > 1e-12: raise RuntimeError(f"Gate C: refined eigenpair residual {res_ref:.2e} at [{i},{j}]")
            P_raw, P_ref, Su = proj(vraw), proj(vref), Sv / u
            rows.append(dict(i=i, j=j, Te=float(te[i]), ne=float(ne[j]), u_CRE=u, a3=a3, a4=a4, c3=c3, c4=c4, f3=f3, f4=f4, S=Sv, Delta=Delta,
                             lam_slow=lam1.real, lam_2=lam2.real, lam_3=lam3.real, lam_2_imag=lam2.imag, gap_ratio=lam1.real / lam2.real,
                             lam_slow_refined=lam_ref, S_over_u=Su, proj_raw=P_raw, proj_refined=P_ref,
                             dev_raw=P_raw / Su - 1, dev_refined=P_ref / Su - 1, eig_residual_raw=res_raw, eig_residual_refined=res_ref))
    df = pd.DataFrame(rows)
    say(f"\n  A  a3,a4,c3,c4,u_CRE,Delta reproduce molecular_channel.csv at {len(df)} points: max rel diff {worstA:.3e} at {worstA_where}")
    if worstA > 1e-8: raise RuntimeError("Gate A")
    say("  B  all CRE populations positive at every point: OK")
    say(f"  C  slow eigenvalue real and simple everywhere; refined residual max {df.eig_residual_refined.max():.2e}, raw residual max {df.eig_residual_raw.max():.2e}: OK")
    say(f"     refined eigenvalue vs numpy: max |lam_ref/lam_raw - 1| = {np.abs(df.lam_slow_refined / df.lam_slow - 1).max():.2e}")
    say("\n  ALL GATES PASSED.")

    say("\n" + "=" * 78); say("RESULT 1  S = f3 - f4 at u_CRE over the grid"); say("=" * 78)
    kmin, kmax = df.S.idxmin(), df.S.idxmax()
    say(f"  S > 0 at {int((df.S > 0).sum())}/{len(df)} points;  min S = {df.S[kmin]:.4f} at [{df.i[kmin]},{df.j[kmin]}] (Te {df.Te[kmin]:.3g} eV, ne {df['ne'][kmin]:.2e});"
        f"  max S = {df.S[kmax]:.4f} at [{df.i[kmax]},{df.j[kmax]}] (Te {df.Te[kmax]:.3g} eV, ne {df['ne'][kmax]:.2e})")
    say(f"  S by density column (min..max over Te): " + "  ".join(f"{ne[j]:.0e}:{df.S[df.j == j].min():.3f}..{df.S[df.j == j].max():.3f}" for j in range(nN)))
    say("\n" + "=" * 78); say("RESULT 2  slow-eigenvector projection P = (v3/n3* - v4/n4*)/(v_g/n_ion) against S/u*"); say("=" * 78)
    say(f"  {'point':>16} {'S':>8} {'u*':>11} {'S/u*':>11} {'P_raw':>11} {'P_refined':>11} {'dev_raw':>10} {'dev_ref':>10} {'lam_slow':>11} {'lam_2':>11} {'lam_s/lam_2':>11} {'dev_ref/(lam_s/lam_2)':>21}")
    for (i, j, lab) in [(23, 5, "benchmark"), (0, 4, "cold corner"), (44, 7, "min S"), (15, 3, "ridge"), (49, 7, "hot dense")]:
        r = df[(df.i == i) & (df.j == j)].iloc[0]
        say(f"  {lab + f' [{i},{j}]':>16} {r.S:8.4f} {r.u_CRE:11.4e} {r.S_over_u:11.4e} {r.proj_raw:11.4e} {r.proj_refined:11.4e} {r.dev_raw:+10.2e} {r.dev_refined:+10.2e} {r.lam_slow:11.4e} {r.lam_2:11.4e} {r.gap_ratio:11.3e} {r.dev_refined / r.gap_ratio:21.3f}")
    kr, kf = df.dev_raw.abs().idxmax(), df.dev_refined.abs().idxmax()
    say(f"\n  max |dev_raw| over the grid     = {abs(df.dev_raw[kr]):.3e} at [{df.i[kr]},{df.j[kr]}]  (lam_slow/lam_2 there {df.gap_ratio[kr]:.3e})")
    say(f"  max |dev_refined| over the grid = {abs(df.dev_refined[kf]):.3e} at [{df.i[kf]},{df.j[kf]}]  (lam_slow/lam_2 there {df.gap_ratio[kf]:.3e})")
    ratio = (df.dev_refined / df.gap_ratio); say(f"  dev_refined / (lam_slow/lam_2): min {ratio.min():.3f}, median {ratio.median():.3f}, max {ratio.max():.3f} over 400 points")
    say(f"  |lam_slow/lam_2| over the grid: min {df.gap_ratio.abs().min():.3e}, max {df.gap_ratio.abs().max():.3e} at [{df.i[df.gap_ratio.abs().idxmax()]},{df.j[df.gap_ratio.abs().idxmax()]}]")
    say("\n  three slowest eigenvalues (s^-1):")
    for (i, j) in [(1, 4), (0, 0), (23, 5), (0, 4)]:
        r = df[(df.i == i) & (df.j == j)].iloc[0]
        say(f"    [{i},{j}] Te {te[i]:.4g} eV ne {ne[j]:.3e}:  {r.lam_slow:.4e}  {r.lam_2:.4e}{'' if abs(r.lam_2_imag) < 1e-9 * abs(r.lam_2) else f' (Im {r.lam_2_imag:.2e})'}  {r.lam_3:.4e}   gap lam_2/lam_slow = {r.lam_2 / r.lam_slow:.3e}   tau_slow = {-1 / r.lam_slow:.4g} s")

    ok1 = bool((df.S > 0).all()); ok2 = (abs(df.S.min() - 0.046) < 0.0015) and (df.i[kmin], df.j[kmin]) == (44, 7) and abs(df.S.max() - 0.493) < 0.0015
    rb, rc = df[(df.i == 23) & (df.j == 5)].iloc[0], df[(df.i == 0) & (df.j == 4)].iloc[0]
    ok3 = abs(rb.dev_raw) < 1.5e-4 and abs(rc.dev_raw) < 3e-8
    ok4 = bool(((ratio.abs() < 10) & (ratio.abs() > 0.1)).all())
    say("\nPREDICTIONS: " + ", ".join(f"{k} {'reproduced' if v else 'NOT reproduced'}" for k, v in (("P1", ok1), ("P2", ok2), ("P3", ok3), ("P4", ok4))))
    refuted = (not ok1) or df.dev_raw.abs().max() > 0.01 or df.dev_refined.abs().max() > 0.01
    say("REFUTER (S <= 0 anywhere, or projection off S/u* by > 1 % anywhere): " + ("APPEARED" if refuted else "did not appear"))
    say("\nThe projection test is exact given a one-dimensional slow subspace: the refined eigenvector differs from the QSS vector\n"
        "a = -L_EE^{-1} L_Eg only through lam_slow in the resolvent, so dev_refined is the first-order correction lam_slow/lam_2 itself.\n"
        "dev_raw adds numpy eig's relative floor on the small excited components of a ground-dominated eigenvector.")
    if a.write:
        out = Path(a.out) if a.out else P("validation/slow_mode_projection"); out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}",
               f"# L_grid.npy sha256 {sha(P('data/processed/cr_matrix/L_grid.npy'))}", f"# S_grid.npy sha256 {sha(S_path)}",
               f"# state_index.csv sha256 {sha(ctx.state_index_path)}", f"# molecular_channel.csv sha256 {sha(mol_path)}",
               "# P = (v3/n3* - v4/n4*)/(v_g/n_ion) along the slow eigenvector; dev = P/(S/u*) - 1; refined = Schur-polished eigenpair"]
        with open(out / "slow_mode_projection.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "slow_mode_projection.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/slow_mode_projection.{{csv,txt}}")
    return 0

if __name__ == "__main__": sys.exit(main())
