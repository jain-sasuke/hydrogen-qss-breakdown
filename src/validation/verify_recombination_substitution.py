#!/usr/bin/env python
"""
verify_recombination_substitution.py
====================================
Complete the atomic-data bound on Sbar, G and eps by substituting the
recombination channel, which verify_reservoir_gain_anderson.py left untouched.

WHAT AN INDEPENDENT RECOMBINATION DATASET MEANS HERE
----------------------------------------------------
The source vector is  S = ne*alpha_RR + ne^2*alpha_3BR, per unit n_ion.
The two terms have different standing:

  alpha_3BR is NOT independent data. It is built from the ionisation
  coefficient by detailed balance (Saha), so substituting the ionisation
  dataset substitutes the three-body recombination with it. The model uses
  CCC TICS for n <= 9 and Lotz (1968) for n = 10-15. Lotz is therefore an
  independent dataset already in the repository, and replacing CCC TICS with
  Lotz everywhere makes the ionisation table one consistent source instead of
  a hybrid. That substitution moves BOTH alpha_3BR (through S) and the
  ionisation loss on the diagonal of L. It is a genuine two-dataset
  comparison, Lotz's 1968 semi-empirical fit against Bray's close-coupling
  calculation.

  alpha_RR has no second dataset in this repository. It is Johnson (1972)
  Eq.(7), whose quoted accuracy is <5% against exact Karzas & Latter Gaunt
  factors below 10^6 K. What is done here is therefore a SENSITIVITY TEST at
  that stated accuracy, not a substitution, and it is labelled as such
  everywhere it is reported. A +-5% scaling is not a measurement of a rival
  dataset and must not be quoted as one.

GATES  (all must pass before any substituted number is reported)
---------------------------------------------------------------
  A  this script's Lotz reproduces the stored K_ion for the Lotz-sourced
     shells (n = 10-15) exactly -- proves it is the pipeline's own Lotz
  B  S_grid reconstructs from the stored alpha_RR / alpha_3BR arrays
  C  the stored alpha_3BR / K_ion ratio equals the analytic Saha factor, which
     is what licenses rebuilding alpha_3BR from a new K_ion
  D  the baseline two-channel pass reproduces
     validation/reservoir_gain/reservoir_gain.csv row by row
  E  after substitution the column sums of L equal -K_ion_sub*ne, the
     ionisation identity of assemble_cr_matrix.py

Note that unlike the excitation substitution, the column sums MUST change
here: ionisation is the loss to the continuum, so changing the ionisation
dataset changes it. Gate E checks the new sums are right, not unchanged.

Run:  python src/validation/verify_recombination_substitution.py [--write]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import exp1

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)
IH_EV = 13.6058

_s = importlib.util.spec_from_file_location(
    "rga", _HERE.parent / "verify_reservoir_gain_anderson.py")
rga = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rga)
two_channel = rga.two_channel


def lotz_K_ion(n, Te):
    """Lotz (1968) Z. Phys. 216 Eq.(5), hydrogen-like. Copied from
    src/rates/ionization_rates.py:57-74 unmodified."""
    x = (IH_EV / n**2) / Te
    return 6.7e-7 * 4.5 / Te**1.5 * (1.0 / x) * exp1(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--win-lo", type=float, default=30.0)
    ap.add_argument("--win-hi", type=float, default=30.0)
    ap.add_argument("--rr-frac", type=float, default=0.05,
                    help="alpha_RR sensitivity scaling (Johnson's stated accuracy)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g, nv = ctx.ground_index, ctx.n_values
    nT, nN, nS, _ = L.shape
    S = np.load(ROOT / "data/processed/cr_matrix/S_grid.npy")

    K_ion = np.load(ROOT / "data/processed/collisions/tics/K_ion_final.npy")
    kmeta = pd.read_csv(ROOT / "data/processed/collisions/tics/K_ion_final_meta.csv")
    aRR = np.concatenate([
        np.load(ROOT / "data/processed/recombination/alpha_RR_resolved.npy"),
        np.load(ROOT / "data/processed/recombination/alpha_RR_bundled.npy")])
    a3BR = np.concatenate([
        np.load(ROOT / "data/processed/recombination/alpha_3BR_resolved.npy"),
        np.load(ROOT / "data/processed/recombination/alpha_3BR_bundled.npy")])
    si = pd.read_csv(ROOT / "data/processed/collisions/K_exc_full/state_index.csv")

    print("=" * 78)
    print("GATES")
    print("=" * 78)

    # --- Gate A ---
    lotz_rows = kmeta.index[kmeta.source.astype(str).str.startswith("Lotz")].tolist()
    worstA = 0.0
    for i in lotz_rows:
        n = int(kmeta.iloc[i].n)
        worstA = max(worstA, np.abs(lotz_K_ion(n, te) / K_ion[i] - 1).max())
    print(f"  A  Lotz reproduces stored K_ion on the {len(lotz_rows)} Lotz shells: "
          f"max rel diff {worstA:.3e}")
    if worstA > 1e-12:
        raise RuntimeError("this script's Lotz is not the pipeline's Lotz")

    # --- Gate B ---
    S_rec = ne[None, :, None] * (aRR.T[:, None, :] + ne[None, :, None] * a3BR.T[:, None, :])
    relB = np.abs(S_rec - S).max() / np.abs(S).max()
    print(f"  B  S_grid = ne*alpha_RR + ne^2*alpha_3BR: max rel diff {relB:.3e}")
    if relB > 1e-12:
        raise RuntimeError("S_grid does not reconstruct from the alpha arrays")

    # --- Gate C: the Saha factor ---
    h, me_kg, kB = 6.62607015e-34, 9.1093837015e-31, 1.380649e-23
    gst = si.g.values.astype(float)[:, None]
    I_eV = si.I_eV.values.astype(float)[:, None]
    kT_J = te[None, :] * 1.60218e-19
    saha = (gst / 2.0) * (h**2 / (2 * np.pi * me_kg * kT_J))**1.5 * 1e6 \
        * np.exp(I_eV / te[None, :])
    fac = np.where(K_ion > 0, a3BR / np.maximum(K_ion, 1e-300), np.nan)
    relC = np.nanmax(np.abs(fac / saha - 1))
    print(f"  C  stored alpha_3BR/K_ion vs analytic Saha factor: max rel diff {relC:.3e}")
    if relC > 5e-3:
        raise RuntimeError("the 3BR detailed-balance factor is not what it claims; "
                           "rebuilding alpha_3BR from a new K_ion is not licensed")

    # --- build the substituted ionisation table (Lotz everywhere) ---
    K_lotz = np.vstack([lotz_K_ion(int(r.n), te) for _, r in si.iterrows()])
    ratio = K_lotz / K_ion
    print(f"\n  Lotz / model K_ion, over the CCC-sourced states: "
          f"min {ratio[:37].min():.2f}  median {np.median(ratio[:37]):.2f}  "
          f"max {ratio[:37].max():.2f}")

    a3BR_sub = K_lotz * fac                       # detailed balance, same factor
    S_ion = ne[None, :, None] * (aRR.T[:, None, :]
                                 + ne[None, :, None] * a3BR_sub.T[:, None, :])
    dK = (K_lotz - K_ion)                          # (43, nT)
    L_ion = L.copy()
    idx = np.arange(nS)
    L_ion[:, :, idx, idx] -= dK.T[:, None, :] * ne[None, :, None]

    # --- Gate E: ionisation identity on the column sums ---
    colsum = L_ion.sum(axis=2)
    target = -(K_lotz.T[:, None, :] * ne[None, :, None])
    relE = np.abs(colsum - target).max() / np.abs(target).max()
    print(f"  E  column sums of substituted L equal -K_ion_sub*ne: "
          f"max rel diff {relE:.3e}")
    if relE > 1e-10:
        raise RuntimeError("the substituted L violates the ionisation identity")

    # --- alpha_RR sensitivity (NOT a substitution) ---
    S_rr_hi = ne[None, :, None] * (aRR.T[:, None, :] * (1 + a.rr_frac)
                                   + ne[None, :, None] * a3BR.T[:, None, :])
    S_rr_lo = ne[None, :, None] * (aRR.T[:, None, :] * (1 - a.rr_frac)
                                   + ne[None, :, None] * a3BR.T[:, None, :])

    # --- excitation substitution, for the combined case ---
    print()
    L_exc = rga.build_substituted_L(L, te, ne)
    L_both = L_exc.copy()
    L_both[:, :, idx, idx] -= dK.T[:, None, :] * ne[None, :, None]

    cases = [("base",   L,      S),
             ("ion",    L_ion,  S_ion),
             ("exc",    L_exc,  S),
             ("both",   L_both, S_ion),
             ("rr_hi",  L,      S_rr_hi),
             ("rr_lo",  L,      S_rr_lo)]

    E = np.array([i for i in range(nS) if i != g])
    N3 = np.where(nv == 3)[0]; N4 = np.where(nv == 4)[0]
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3]); n4E = np.array([pos[s] for s in N4])

    rows = []
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for k in a.steps:
            for i in range(nT):
                j_te = i + sgn * k
                if not (0 <= j_te < nT):
                    continue
                dlnTe = np.log(te[j_te] / te[i])
                for j in range(nN):
                    lam = np.sort(np.linalg.eigvals(L[j_te, j]).real)[::-1]
                    neg = lam[lam < 0]
                    tQ, tR = 1.0 / abs(neg[0]), 1.0 / abs(neg[1])
                    rec = dict(direction=dlab, k=k, i=i, j=j, Te=float(te[i]),
                               ne=float(ne[j]), dlnTe=dlnTe, M=tQ / tR,
                               window_ok=bool((a.win_lo * tR) < (tQ / a.win_hi)))
                    for nm, Lc, Sc in cases:
                        d = two_channel(Lc[j_te, j], Sc[j_te, j], Lc[i, j],
                                        Sc[i, j], E, g, n3E, n4E, N3, N4,
                                        f"{nm} {i},{j}")
                        Gv = d["lnx"] / dlnTe
                        ep = abs(np.expm1(d["Sbar"] * Gv * dlnTe))
                        if d["eps"] > 0 and abs(ep - d["eps"]) / d["eps"] > 1e-10:
                            raise RuntimeError(f"identity broken ({nm}) at [{i},{j}]")
                        sfx = "" if nm == "base" else f"_{nm}"
                        rec[f"G{sfx}"] = Gv
                        rec[f"Sbar{sfx}"] = d["Sbar"]
                        rec[f"eps{sfx}"] = d["eps"]
                        rec[f"Delta{sfx}"] = d["Delta"]
                        rec[f"cap{sfx}"] = d["cap"]
                    rows.append(rec)
    df = pd.DataFrame(rows)

    # --- Gate D ---
    ref = pd.read_csv(ROOT / "validation/reservoir_gain/reservoir_gain.csv")
    m = df.merge(ref, on=["direction", "k", "i", "j"], suffixes=("", "_ref"))
    if len(m) != len(ref):
        raise RuntimeError(f"row mismatch {len(m)} vs {len(ref)}")
    wd = max(np.abs((m[q] - m[f"{q}_ref"]) / m[f"{q}_ref"]).max()
             for q in ("G", "Sbar", "eps"))
    print(f"  D  baseline reproduces reservoir_gain.csv over {len(m)} rows: "
          f"max rel diff {wd:.3e}")
    if wd > 1e-10:
        raise RuntimeError("baseline not reproduced; nothing below is trustworthy")
    print("\n  ALL GATES PASSED.")

    LAB = {"ion":  "ionisation: Lotz 1968 for CCC TICS  [SUBSTITUTION]",
           "exc":  "excitation: Anderson 2002 RMPS for CCC  [SUBSTITUTION]",
           "both": "both substitutions together  [SUBSTITUTION]",
           "rr_hi": f"alpha_RR x{1+a.rr_frac:.2f}  [SENSITIVITY, not a dataset]",
           "rr_lo": f"alpha_RR x{1-a.rr_frac:.2f}  [SENSITIVITY, not a dataset]"}

    for scope, sel in (("all rows", np.ones(len(df), bool)),
                       ("window_ok, Te >= 2 eV (defended range)",
                        (df.window_ok & (df.Te >= 2.0)).values)):
        sub = df[sel]
        print("\n" + "=" * 78)
        print(f"EFFECT ON THE STRUCTURAL COEFFICIENTS  --  {scope}  (n={len(sub)})")
        print("=" * 78)
        for nm in ("ion", "exc", "both", "rr_hi", "rr_lo"):
            print(f"\n  {LAB[nm]}")
            for q in ("Delta", "cap", "Sbar", "G", "eps"):
                b_, s_ = sub[q].abs(), sub[f"{q}_{nm}"].abs()
                d = (s_ / b_ - 1.0) * 100.0
                print(f"    {q:6s} median {d.median():+8.2f}%   "
                      f"mean|.| {d.abs().mean():7.2f}%   max|.| {d.abs().max():7.2f}%")

    ok = df[df.window_ok & (df.k == 1) & (df.direction == "heat")]
    print("\n" + "=" * 78)
    print("THE CAP AND ITS HEADROOM, k=1 heating, window_ok")
    print("=" * 78)
    print(f"  {'case':<10} {'cap min':>9} {'cap med':>9} {'cap max':>9} "
          f"{'|Sbar|/cap med':>15} {'max':>7} {'bound viol':>11}")
    for nm in ("", "_ion", "_exc", "_both"):
        c, sb = ok[f"cap{nm}"], ok[f"Sbar{nm}"].abs()
        print(f"  {(nm[1:] or 'base'):<10} {c.min():9.4f} {c.median():9.4f} "
              f"{c.max():9.4f} {(sb/c).median()*100:14.1f}% "
              f"{(sb/c).max()*100:6.1f}% {int((sb>c).sum()):11d}")

    print("\n" + "=" * 78)
    print("eps AT THE POINTS CHAPTER 5 QUOTES  (k=1 heating)")
    print("=" * 78)
    print(f"  {'point':>18} {'base':>9} {'ion':>9} {'exc':>9} {'both':>9}")
    for (i, j, lab) in [(23, 5, "benchmark [23,5]"), (15, 3, "ridge [15,3]"),
                        (0, 4, "worst [0,4]")]:
        r = df[(df.direction == "heat") & (df.k == 1) & (df.i == i) & (df.j == j)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        print(f"  {lab:>18} {r.eps*100:8.3f}% {r.eps_ion*100:8.3f}% "
              f"{r.eps_exc*100:8.3f}% {r.eps_both*100:8.3f}%")

    if a.write:
        out = ROOT / "validation/recombination_substitution"
        out.mkdir(exist_ok=True)
        df.to_csv(out / "recombination_substitution.csv", index=False)
        pd.DataFrame({"state": si.label, "n": si.n,
                      **{f"lotz_over_model_Te{te[t]:.2f}": ratio[:, t]
                         for t in (0, 23, 49)}}).to_csv(
            out / "lotz_vs_ccc_ionisation_ratio.csv", index=False)
        print(f"\n  wrote {out.relative_to(ROOT)}/")
    else:
        print("\n  (--write not given; nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
