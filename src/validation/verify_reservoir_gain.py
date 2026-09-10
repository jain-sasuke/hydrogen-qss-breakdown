#!/usr/bin/env python
"""
verify_reservoir_gain.py
========================
Produce the two structural coefficients Chapter 5 is built on, with provenance.

WHY THIS EXISTS
---------------
`outputs/pivot_decision.md` reframed the headline from a step-dependent
percentage to the exact response

    eps = | exp( Sbar * G * dlnTe ) - 1 |,     G = dln u / dln Te

and quoted nine values of G. Those nine numbers were computed in an ad-hoc
shell session and never written to an artifact. Chapter 5 then headlined them,
which left the chapter's central quantity with no producing script. CLAUDE.md
rule 4 requires provenance on every number. This script supplies it.

WHAT IT COMPUTES
----------------
For every grid point and every step size k (in grid indices):

    u        = n_g / n_ion, the reservoir variable, from the CRE solve
    Sbar     = ln(R_PE / R_QSSnew) / ln x, the mean sensitivity over the
               excursion; equals f_3 - f_4 in the small-step limit
    G        = ln x / dlnTe, the reservoir gain
    eps      = |R_PE / R_QSSnew - 1|, the plateau error
    eps_pred = |exp(Sbar * G * dlnTe) - 1|, which is an identity and must
               reproduce eps to machine precision. That is a wiring check,
               not a physics check, and is labelled as such.

THE CLAIM BEING TESTED
----------------------
G is stable against step size while eps is not. The script measures the spread
of G across k = 1, 2, 4 at every point and reports the distribution, so the
word "stable" carries a number rather than an impression.

The refuting observation, stated in advance: if G varied across k by as much as
eps does (a factor of about 5 over this range), there would be no step-
independent coefficient and the reframing would fail. The script prints the
comparison directly.

CONVENTIONS
-----------
- Heating and cooling are reported separately. eps and G differ between them.
- The two-channel construction is the same algebra as
  verify_plateau_gridmap.py:200-209. It is not re-derived differently here.
- window_ok is the M > win_lo*win_hi gate used by the gridmap, carried through
  so the two files can be compared row by row.
- Nothing is written outside validation/reservoir_gain/, and only with --write.

Provenance is printed at the top of every run: input SHA-256, grid shapes,
interpreter, numpy version.
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


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4],
                    help="step sizes in grid indices")
    ap.add_argument("--win-lo", type=float, default=30.0)
    ap.add_argument("--win-hi", type=float, default=30.0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    nv = ctx.n_values

    s_path = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not s_path.exists():
        raise RuntimeError(f"missing source vector: {s_path}")
    S = np.load(s_path)
    if S.shape[:2] != L.shape[:2] or S.shape[2] != L.shape[2]:
        raise RuntimeError(f"S_grid shape {S.shape} incompatible with L_grid {L.shape}")

    E = np.array([i for i in range(ctx.n_states) if i != g])
    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    if len(N3) == 0 or len(N4) == 0:
        raise RuntimeError("no n=3 or n=4 states found in the state index")
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    print(f"  L_grid  {L.shape}  sha256 {sha256_file(lp)[:32]}...")
    print(f"  S_grid  {S.shape}  sha256 {sha256_file(s_path)[:32]}...")
    print(f"  states  {ctx.n_states}   ground index {g} ({ctx.labels[g]})")
    print(f"  n=3 states {list(N3)}   n=4 states {list(N4)}")
    print(f"  interpreter {sys.executable}")
    print(f"  numpy {np.__version__}")
    print()

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
                    if len(neg) < 2:
                        raise RuntimeError(
                            f"fewer than two negative eigenvalues at "
                            f"[{j_te},{j}]; the operator is not stable")
                    tQ, tR = 1.0 / abs(neg[0]), 1.0 / abs(neg[1])
                    window_ok = (a.win_lo * tR) < (tQ / a.win_hi)

                    n_old = np.linalg.solve(L[i, j], -S[i, j])
                    n_new = np.linalg.solve(L[j_te, j], -S[j_te, j])
                    LEE = L[j_te, j][np.ix_(E, E)]
                    LEg = L[j_te, j][np.ix_(E, [g])].ravel()
                    n0 = np.linalg.solve(LEE, -S[j_te, j][E])
                    n1 = np.linalg.solve(LEE, -LEg * n_old[g])

                    sup = (np.abs(n0 + (n_new[g] / n_old[g]) * n1
                                  - n_new[E]).max() / np.abs(n_new[E]).max())
                    if sup > 1e-8:
                        raise RuntimeError(
                            f"superposition residual {sup:.3e} at [{i},{j}] "
                            f"{dlab} k={k}: the two-channel split is not exact")

                    lnx = np.log(n_new[g] / n_old[g])
                    a3, a4 = n1[n3E].sum(), n1[n4E].sum()
                    c3, c4 = n0[n3E].sum(), n0[n4E].sum()
                    R_pe = (c3 + a3) / (c4 + a4)
                    R_q = n_new[N3].sum() / n_new[N4].sum()

                    eps = abs(R_pe / R_q - 1.0)
                    Sbar = np.log(R_pe / R_q) / lnx
                    G = lnx / dlnTe
                    eps_pred = abs(np.expm1(Sbar * G * dlnTe))
                    # identity check: wiring, not physics
                    if eps > 0 and abs(eps_pred - eps) / eps > 1e-10:
                        raise RuntimeError(
                            f"identity broken at [{i},{j}] {dlab} k={k}: "
                            f"{eps_pred:.6e} vs {eps:.6e}")

                    rows.append(dict(direction=dlab, k=k, i=i, j=j,
                                     Te=float(te[i]), ne=float(ne[j]),
                                     dlnTe=dlnTe, lnx=lnx, G=G, Sbar=Sbar,
                                     eps=eps, tau_QSS=tQ, M=tQ / tR,
                                     window_ok=bool(window_ok)))

    print("=" * 78)
    print("IS G STABLE AGAINST STEP SIZE, WHERE eps IS NOT?")
    print("=" * 78)
    for dlab in ("heat", "cool"):
        sub = [r for r in rows if r["direction"] == dlab]
        pts = sorted({(r["i"], r["j"]) for r in sub})
        gs, es = [], []
        for (i, j) in pts:
            rr = {r["k"]: r for r in sub if r["i"] == i and r["j"] == j}
            if set(a.steps) - set(rr):
                continue
            gv = np.array([abs(rr[k]["G"]) for k in a.steps])
            ev = np.array([rr[k]["eps"] for k in a.steps])
            gs.append(gv.max() / gv.min())
            es.append(ev.max() / ev.min() if ev.min() > 0 else np.nan)
        gs, es = np.array(gs), np.array(es)
        print(f"  {dlab}: {len(gs)} points with all of k={a.steps}")
        print(f"    spread of |G| across k   median {np.median(gs):.4f}   "
              f"90th pct {np.nanpercentile(gs,90):.4f}   max {gs.max():.4f}")
        print(f"    spread of eps across k   median {np.nanmedian(es):.4f}   "
              f"90th pct {np.nanpercentile(es,90):.4f}   max {np.nanmax(es):.4f}")
    print()
    print("  A step-independent coefficient requires the first spread to be")
    print("  small where the second is large. If they were comparable, the")
    print("  reframing would fail.")
    print()

    print("=" * 78)
    print("THE TABLE QUOTED IN pivot_decision.md AND CHAPTER 5 (heating)")
    print("=" * 78)
    print(f"  {'point':>22} " + " ".join(f"{'k='+str(k):>10}" for k in a.steps)
          + f" {'spread':>8}")
    for (i, j, lab) in [(23, 5, "benchmark [23,5]"), (15, 3, "ridge [15,3]"),
                        (0, 4, "worst [0,4]")]:
        vals = []
        for k in a.steps:
            m = [r for r in rows if r["direction"] == "heat" and r["k"] == k
                 and r["i"] == i and r["j"] == j]
            vals.append(m[0]["G"] if m else np.nan)
        vals = np.array(vals)
        sp = np.nanmax(np.abs(vals)) / np.nanmin(np.abs(vals))
        print(f"  {lab:>22} " + " ".join(f"{v:10.3f}" for v in vals)
              + f" {100*(sp-1):7.1f}%")
    print()

    ok = [r for r in rows if r["window_ok"] and r["k"] == 1
          and r["direction"] == "heat"]
    Sb = np.array([abs(r["Sbar"]) for r in ok])
    Gv = np.array([abs(r["G"]) for r in ok])
    print("=" * 78)
    print("THE TWO STRUCTURAL MAPS, k=1 heating, window_ok only")
    print("=" * 78)
    print(f"  |Sbar|  min {Sb.min():.5f}  median {np.median(Sb):.5f}  max {Sb.max():.5f}")
    print(f"  |G|     min {Gv.min():.4f}  median {np.median(Gv):.4f}  max {Gv.max():.4f}")
    print(f"  d(eps)/d(lnTe) = |Sbar*G|:  min {(Sb*Gv).min():.4f}  "
          f"median {np.median(Sb*Gv):.4f}  max {(Sb*Gv).max():.4f}")
    print()

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "reservoir_gain"
        out.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with (out / "reservoir_gain.csv").open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {out/'reservoir_gain.csv'}  ({len(rows)} rows)")

        # The aggregations chapter 7 quotes. They were previously computed by
        # reading the csv by hand, which is why chapter7.tex carried an
        # [UNVERIFIED] saying no script performed them. It does now, and every
        # aggregate is written with the filter that produced it, because the
        # same quantity over a different filter is a different number and this
        # file is where the two get confused.
        def agg(name, sel, key):
            v = np.array([abs(r[key]) for r in rows if sel(r)])
            if v.size == 0:
                raise RuntimeError(f"filter '{name}' selected no rows")
            return dict(filter=name, quantity=key, n=int(v.size),
                        min=float(v.min()), median=float(np.median(v)),
                        max=float(v.max()))

        summ = []
        for dlab in ("heat", "cool"):
            sub = [r for r in rows if r["direction"] == dlab]
            pts = sorted({(r["i"], r["j"]) for r in sub})
            gs, es = [], []
            for (i, j) in pts:
                rr = {r["k"]: r for r in sub if r["i"] == i and r["j"] == j}
                if set(a.steps) - set(rr):
                    continue
                gv = np.array([abs(rr[k]["G"]) for k in a.steps])
                ev = np.array([rr[k]["eps"] for k in a.steps])
                gs.append(gv.max() / gv.min())
                if ev.min() > 0:
                    es.append(ev.max() / ev.min())
            gs, es = np.array(gs), np.array(es)
            summ.append(dict(filter=f"{dlab}, all of k={a.steps}",
                             quantity="spread of |G| across k", n=int(gs.size),
                             min=float(gs.min()), median=float(np.median(gs)),
                             max=float(gs.max())))
            summ.append(dict(filter=f"{dlab}, all of k={a.steps}",
                             quantity="spread of eps across k", n=int(es.size),
                             min=float(es.min()), median=float(np.median(es)),
                             max=float(es.max())))
        summ.append(agg("all rows", lambda r: True, "G"))
        summ.append(agg("all rows", lambda r: True, "Sbar"))
        summ.append(agg("k=1 heating, window_ok",
                        lambda r: r["k"] == 1 and r["direction"] == "heat"
                        and r["window_ok"], "Sbar"))
        summ.append(agg("k=1 heating, window_ok",
                        lambda r: r["k"] == 1 and r["direction"] == "heat"
                        and r["window_ok"], "G"))
        summ.append(agg("heating, window_ok, any k",
                        lambda r: r["direction"] == "heat" and r["window_ok"],
                        "Sbar"))
        summ.append(agg("heating, window_ok, any k, Te >= 2 eV",
                        lambda r: r["direction"] == "heat" and r["window_ok"]
                        and r["Te"] >= 2.0, "Sbar"))
        summ.append(agg("heating, window_ok, any k, Te >= 2 eV",
                        lambda r: r["direction"] == "heat" and r["window_ok"]
                        and r["Te"] >= 2.0, "G"))
        with (out / "reservoir_gain_summary.csv").open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
            w.writeheader()
            w.writerows(summ)
        print(f"  wrote {out/'reservoir_gain_summary.csv'}  ({len(summ)} rows)")
        for s in summ:
            print(f"    {s['quantity']:<26} [{s['filter']:<38}] n={s['n']:>5}  "
                  f"{s['min']:.4g} to {s['max']:.4g}, median {s['median']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
