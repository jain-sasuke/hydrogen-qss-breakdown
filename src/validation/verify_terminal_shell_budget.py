#!/usr/bin/env python
"""
verify_terminal_shell_budget.py
===============================
The loss budget of the terminal shell n = 15, and an estimate of the loss
channel that truncating the ladder at n_max = 15 removed from it.

WHY THIS EXISTS
---------------
Chapter 4 (thesis_tex/chapter4.tex, ~line 710) explains the over-population of
the terminal shell with "The top of a truncated ladder has nothing above it to
cascade out to, so it accumulates." Cascade is downward: a level above n = 15
would cascade INTO n = 15, not receive cascade from it. What truncation
actually removes from the n = 15 balance is (i) collisional excitation OUT of
n = 15 to n = 16, 17, ... (a loss channel, dominant at high n where Delta n = 1
excitation is fast), and (ii) de-excitation and radiative cascade IN from those
levels (gain channels). Whether the removed loss is large compared with the
retained losses is a measurable property of the operator; if it is, truncation
biases the terminal shell UP through the missing upward channel and the
sentence's mechanism is wrong even though its conclusion (over-population)
stands. Nothing in the thesis rests on the top shell (the observable is built
from n = 3 and n = 4); this fixes the explanation, not a result.

METHOD
------
  L is affine in ne: L = R + ne C (checked exact to round-off across all 8
  columns; the same decomposition verify_molecular_channel.py uses).  For the
  n = 15 bundled state k (the unique state with n = n_max, confirmed from
  state_index.csv) at every grid point:
    total loss          -L[k,k]
    radiative loss      sum_{i != k} R[i,k]           (the ne -> 0 intercept)
    coll. de-exc -> 14  ne C[k14,k]                   (the state one below)
    coll. de-exc -> <=13  ne sum_{i not in {k,k14}} C[i,k]
    ionisation          -sum_i L[i,k]                 (column-sum deficit)
  reported as fractions of the total (they must sum to 1).
  REMOVED 15 -> 16 EXCITATION, by geometric continuation.  ASSUMPTION, stated
  here and in the output: the ratio of successive Delta n = +1 excitation
  coefficients, rho(n) = q(n -> n+1)/q(n-1 -> n), continues unchanged one step
  past the ladder, q(15->16) = q(14->15) * rho(14) with rho(14) =
  C[k15,k14]/C[k14,k13].  rho(13) = C[k14,k13]/C[k13,k12] is printed beside it
  so the reader can judge whether the ratio is settled; a classical n^4
  continuation, q(15->16) = q(14->15) (15/14)^4, is reported as a sensitivity
  only.  The removed loss ne q(15->16) is quoted as a multiple of the retained
  total loss -L[k,k].
  DEPARTURE FROM SAHA-BOLTZMANN.  b_n = n_n*/(n_n^Saha) with n* = -L^{-1} S the
  CRE populations per unit n_ion, and n^Saha/n_ion = ne (g_n/2)
  (h^2 / 2 pi m_e k T)^{3/2} exp(I_n/kT) from state_index.csv's own g and I_eV
  (g_ion = 1 for the proton).  b_n - 1 reported for n = 13, 14, 15.
  NET IONISING FLUX through n = 15 at CRE: ionisation flux (-sum_i L[i,k]) n_k*
  minus recombination flux S[k] (both per unit n_ion), as a fraction of the
  gross exchange (their sum) and of the ionisation flux alone.

GATES (before any result is reported)
------------------------------------
  A  L = R + ne C exact: max residual over all 8 densities < 1e-12 relative
  B  state k is the unique n = 15 state, bundled, the last index; k14, k13,
     k12 unique
  C  ionisation identity: -sum_i L[i,s] = ne K_ion[s] for every state s at
     every grid point (data/processed/collisions/tics/K_ion_final.npy), 1e-10
  D  R[i,k] >= 0 and C[i,k] >= 0 for i != k (no negative rates in the column);
     the four budget fractions sum to 1 within 1e-12
  E  the analytic Saha factor used for b_n agrees with the model's own
     alpha_3BR/K_ion on every row with K_ion > 0 to 5e-3 (the tolerance
     verify_recombination_substitution.py established for this comparison)
  F  all CRE populations positive

PREDICTIONS (written before the run; from the read-only reconstruction
adjudicate.py / followup.py of 17 Sep 2026)
-----------------------------------------------------------------------------
  at [23,5] (Te 2.947 eV, ne 1.389e14):
    total loss 2.04e11 s^-1; radiative 8.2e4 s^-1 (< 0.01 %); de-exc to n=14
    about 80.7 %; to n <= 13 about 12.8 %; ionisation about 6.6 %;
    removed 15->16 (geometric) about 1.30 x the retained total;
    b_13 - 1, b_14 - 1, b_15 - 1 about +6.6e-5, +6.1e-5, +5.9e-5
  at [0,4] (Te 1 eV, ne 5.18e13):
    total 5.64e10 s^-1; 74.3 %, 15.2 %, 10.5 %; removed 1.20 x
  b_n - 1 > 0 and decreasing with n at both points (net upward flux at the top)

REFUTING OBSERVATION (of the sentence's mechanism being the right one)
----------------------------------------------------------------------
  The removed upward loss being negligible, < 10 % of the retained total
  loss, at the benchmark and cold-corner points: truncation would then not
  bias the top shell up through this channel and the missing-cascade wording
  would be harmless.  Also refuting the numbers above: any budget fraction
  off its prediction by more than 2 percentage points.

OUTPUTS (with --write): validation/terminal_shell_budget/terminal_shell_budget.{csv,txt}
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
H_JS, ME_KG, EV_J = 6.62607015e-34, 9.1093837015e-31, 1.60218e-19   # CODATA 2018 h, m_e; eV->J as in verify_recombination_substitution.py

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; g, nv = ctx.ground_index, np.asarray(ctx.n_values); nT, nN, nS, _ = L.shape
    P = lambda *p: ROOT.joinpath(*p)
    paths = dict(L=P("data/processed/cr_matrix/L_grid.npy"), S=P("data/processed/cr_matrix/S_grid.npy"), si=ctx.state_index_path,
                 K=P("data/processed/collisions/tics/K_ion_final.npy"), KTe=P("data/processed/collisions/tics/Te_grid_ion.npy"),
                 a3r=P("data/processed/recombination/alpha_3BR_resolved.npy"), a3b=P("data/processed/recombination/alpha_3BR_bundled.npy"))
    for p in paths.values():
        if not p.is_file(): raise FileNotFoundError(f"missing input: {p}")
    S = np.load(paths["S"]); K_ion = np.load(paths["K"]); KTe = np.load(paths["KTe"])
    a3BR = np.concatenate([np.load(paths["a3r"]), np.load(paths["a3b"])]); si = pd.read_csv(paths["si"])
    if S.shape != L.shape[:3]: raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    if K_ion.shape != (nS, nT) or not np.array_equal(KTe, te): raise ValueError(f"K_ion_final {K_ion.shape} / its Te grid do not match L_grid")
    if a3BR.shape != (nS, nT) or len(si) != nS: raise ValueError("alpha_3BR or state_index.csv size does not match L_grid")
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("TERMINAL SHELL BUDGET -- what n = 15 loses to, and what truncation removed")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}  numpy {np.__version__}"); say(ctx.describe())
    for nm, p in paths.items(): say(f"  {nm:4s} sha256 {sha(p)}  ({p.relative_to(ROOT)})")
    say("=" * 78)

    # ---- Gate B: identify the terminal shell and the three below it from state_index.csv
    nmax = int(nv.max()); ks = np.where(nv == nmax)[0]
    if len(ks) != 1 or ks[0] != nS - 1: raise RuntimeError(f"Gate B: n = {nmax} states {ks}, expected exactly the last index {nS - 1}")
    k = int(ks[0]); row = si.iloc[k]
    if int(row.n) != 15 or int(row.idx) != k or not bool(row.bundled) or str(row.label) != "n15": raise RuntimeError(f"Gate B: state {k} is {row.to_dict()}, not the bundled n15")
    def unique_state(n):
        w = np.where(nv == n)[0]
        if len(w) != 1: raise RuntimeError(f"Gate B: n = {n} has {len(w)} states, expected one bundled state")
        return int(w[0])
    k14, k13, k12 = unique_state(14), unique_state(13), unique_state(12)
    say(f"\n  B  terminal state k={k} '{row.label}' n={int(row.n)} g={int(row.g)} I={float(row.I_eV):.6f} eV bundled={bool(row.bundled)};  k14={k14} k13={k13} k12={k12}: OK")
    # ---- Gate A: affine decomposition
    C = (L[:, 1] - L[:, 0]) / (ne[1] - ne[0]); R = L[:, 0] - ne[0] * C
    resA = max(np.abs(L[:, j] - (R + ne[j] * C)).max() / np.abs(L[:, j]).max() for j in range(nN))
    say(f"  A  L = R + ne C exact across all {nN} densities: max rel residual {resA:.2e}")
    if resA > 1e-12: raise RuntimeError("Gate A")
    # ---- Gate C: ionisation identity for every state
    colsum = -L.sum(axis=2)                                   # (nT, nN, nS): -sum_i L[i,s]
    target = ne[None, :, None] * K_ion.T[:, None, :]
    resC = np.abs(colsum - target).max() / np.abs(target).max()
    say(f"  C  -sum_i L[i,s] = ne K_ion[s] for all states, all points: max rel diff {resC:.2e}")
    if resC > 1e-10: raise RuntimeError("Gate C")
    # ---- Gate D: no negative rates in the k column
    off = np.array([i for i in range(nS) if i != k])
    if R[:, off, k].min() < 0 or C[:, off, k].min() < 0: raise RuntimeError(f"Gate D: negative entry in column {k}: R min {R[:, off, k].min():.2e}, C min {C[:, off, k].min():.2e}")
    say(f"  D  R[i,k], C[i,k] >= 0 for i != k at every Te (mins {R[:, off, k].min():.2e}, {C[:, off, k].min():.2e}): OK")
    # ---- Gate E: Saha factor convention against the model's own alpha_3BR/K_ion
    gst = si.g.values.astype(float)[:, None]; I_eV = si.I_eV.values.astype(float)[:, None]
    saha_fac = (gst / 2.0) * (H_JS**2 / (2 * np.pi * ME_KG * te[None, :] * EV_J))**1.5 * 1e6 * np.exp(I_eV / te[None, :])   # cm^3: n_s^Saha/(ne n_ion)
    fac = np.where(K_ion > 0, a3BR / np.where(K_ion > 0, K_ion, 1.0), np.nan)
    resE = np.nanmax(np.abs(fac / saha_fac - 1))
    say(f"  E  analytic Saha factor vs the model's alpha_3BR/K_ion, all rows with K_ion > 0: max rel diff {resE:.2e}")
    if resE > 5e-3: raise RuntimeError("Gate E")

    rows = []
    for i in range(nT):
        Ri, Ci = R[i], C[i]
        rad = Ri[off, k].sum(); rad_def = -Ri[:, k].sum()
        q1213, q1314, q1415 = Ci[k13, k12], Ci[k14, k13], Ci[k, k14]
        rho13, rho14 = q1314 / q1213, q1415 / q1314
        q1516_geo = q1415 * rho14; q1516_n4 = q1415 * (15.0 / 14.0)**4
        for j in range(nN):
            A = L[i, j]; tot = -A[k, k]; ion = -A[:, k].sum()
            d14 = ne[j] * Ci[k14, k]; dle13 = ne[j] * Ci[[s for s in off if s != k14], k].sum()
            fsum = (rad + d14 + dle13 + ion) / tot
            if abs(fsum - 1) > 1e-12: raise RuntimeError(f"Gate D: budget fractions sum to {fsum:.15f} at [{i},{j}]")
            nstar = np.linalg.solve(A, -S[i, j])
            if nstar.min() <= 0: raise RuntimeError(f"Gate F: non-positive CRE population at [{i},{j}]")
            b = nstar / (ne[j] * saha_fac[:, i])
            ion_flux = ion * nstar[k]; rec_flux = S[i, j][k]; net = ion_flux - rec_flux
            feed_below = A[k, off] @ nstar[off]
            rows.append(dict(i=i, j=j, Te=float(te[i]), ne=float(ne[j]), total_loss=tot, rad_loss=rad, rad_frac=rad / tot, rad_col_deficit=rad_def,
                             deexc_14=d14, deexc_14_frac=d14 / tot, deexc_le13=dle13, deexc_le13_frac=dle13 / tot, ion_loss=ion, ion_frac=ion / tot, frac_sum=fsum,
                             q_12_13=q1213, q_13_14=q1314, q_14_15=q1415, rho_13=rho13, rho_14=rho14, q_15_16_geo=q1516_geo, removed_geo=ne[j] * q1516_geo,
                             removed_geo_over_total=ne[j] * q1516_geo / tot, q_15_16_n4=q1516_n4, removed_n4_over_total=ne[j] * q1516_n4 / tot,
                             exc_14_15_over_loss14=ne[j] * q1415 / (-A[k14, k14]),
                             b13m1=b[k13] - 1, b14m1=b[k14] - 1, b15m1=b[k] - 1, n15_star=nstar[k], u_CRE=nstar[g],
                             ion_flux=ion_flux, rec_flux=rec_flux, feed_from_below=feed_below, net_flux=net, net_over_gross=net / (ion_flux + rec_flux), net_over_ion=net / ion_flux))
    df = pd.DataFrame(rows)
    say(f"  D  budget fractions sum to 1 at all {len(df)} points (max |sum-1| {np.abs(df.frac_sum - 1).max():.1e}): OK"); say("  F  all CRE populations positive: OK")
    say(f"     radiative column deficit -sum_i R[i,k] relative to radiative loss: max {np.abs(df.rad_col_deficit / df.rad_loss).max():.1e} (no ne-independent ionisation)")
    say("\n  ALL GATES PASSED.")

    say("\n" + "=" * 78); say(f"RESULT 1  loss budget of the terminal shell n = 15 (state {k})"); say("=" * 78)
    pts = [(23, 5, "benchmark"), (0, 4, "cold corner"), (0, 0, "[0,0]"), (49, 7, "hot dense"), (49, 0, "hot thin")]
    for (i, j, lab) in pts:
        r = df[(df.i == i) & (df.j == j)].iloc[0]
        say(f"\n  {lab} [{i},{j}]  Te {r.Te:.4g} eV  ne {r['ne']:.3e} cm^-3")
        say(f"    total loss -L[k,k]         {r.total_loss:.4e} s^-1")
        say(f"    radiative (ne->0)          {r.rad_loss:.4e} s^-1  ({100 * r.rad_frac:8.4f} %)")
        say(f"    coll. de-exc -> n=14       {r.deexc_14:.4e} s^-1  ({100 * r.deexc_14_frac:8.3f} %)")
        say(f"    coll. de-exc -> n<=13      {r.deexc_le13:.4e} s^-1  ({100 * r.deexc_le13_frac:8.3f} %)")
        say(f"    ionisation (col. deficit)  {r.ion_loss:.4e} s^-1  ({100 * r.ion_frac:8.3f} %)")
        say(f"    upward coefficients (cm^3 s^-1): 12->13 {r.q_12_13:.4e}  13->14 {r.q_13_14:.4e}  14->15 {r.q_14_15:.4e};  rho(13) {r.rho_13:.4f}  rho(14) {r.rho_14:.4f}")
        say(f"    REMOVED 15->16, geometric  q {r.q_15_16_geo:.4e} cm^3 s^-1 -> ne q = {r.removed_geo:.4e} s^-1 = {r.removed_geo_over_total:.3f} x retained total"
            f"   [sensitivity, n^4: {r.removed_n4_over_total:.3f} x]")
        say(f"    (the retained 14->15 excitation is {r.exc_14_15_over_loss14:.3f} of n=14's own total loss)")
        say(f"    b_n - 1 at CRE:  n=13 {r.b13m1:+.3e}   n=14 {r.b14m1:+.3e}   n=15 {r.b15m1:+.3e}")
        say(f"    n=15 balance per unit n_ion: feed from bound states {r.feed_from_below:.4e}, recombination {r.rec_flux:.4e}, ionisation {r.ion_flux:.4e} s^-1;"
            f"  net ionising flux {r.net_flux:+.4e} = {r.net_over_gross:+.4f} of gross exchange (ion+rec), {r.net_over_ion:+.4f} of the ionisation flux")
    say("\n" + "=" * 78); say("RESULT 2  over the 400 grid points"); say("=" * 78)
    for col, lab in (("rad_frac", "radiative fraction"), ("deexc_14_frac", "de-exc -> 14 fraction"), ("deexc_le13_frac", "de-exc -> <=13 fraction"), ("ion_frac", "ionisation fraction"),
                     ("removed_geo_over_total", "removed/retained (geometric)"), ("removed_n4_over_total", "removed/retained (n^4)"), ("rho_14", "rho(14)"), ("rho_13", "rho(13)")):
        kmn, kmx = df[col].idxmin(), df[col].idxmax()
        say(f"  {lab:30s} min {df[col][kmn]:.4f} at [{df.i[kmn]},{df.j[kmn]}]   median {df[col].median():.4f}   max {df[col][kmx]:.4f} at [{df.i[kmx]},{df.j[kmx]}]")
    say(f"  removed/retained (geometric) >= 0.10 at {int((df.removed_geo_over_total >= 0.10).sum())}/{len(df)} points;  >= 1 at {int((df.removed_geo_over_total >= 1).sum())}/{len(df)}")
    say(f"  b_15 - 1: min {df.b15m1.min():+.3e} at [{df.i[df.b15m1.idxmin()]},{df.j[df.b15m1.idxmin()]}]   max {df.b15m1.max():+.3e} at [{df.i[df.b15m1.idxmax()]},{df.j[df.b15m1.idxmax()]}];"
        f"  b_13 > b_14 > b_15 at {int(((df.b13m1 > df.b14m1) & (df.b14m1 > df.b15m1)).sum())}/{len(df)} points;  b_15 > 1 at {int((df.b15m1 > 0).sum())}/{len(df)}")
    say(f"  net ionising flux / gross exchange through n=15: min {df.net_over_gross.min():+.4f}  median {df.net_over_gross.median():+.4f}  max {df.net_over_gross.max():+.4f};  positive (net upward) at {int((df.net_flux > 0).sum())}/{len(df)}")

    rb, rc = df[(df.i == 23) & (df.j == 5)].iloc[0], df[(df.i == 0) & (df.j == 4)].iloc[0]
    def within(x, y, tol): return abs(x - y) <= tol
    pb = [within(rb.total_loss / 2.04e11, 1, 0.01), rb.rad_frac < 1e-4, within(rb.rad_loss / 8.2e4, 1, 0.02), within(100 * rb.deexc_14_frac, 80.7, 2), within(100 * rb.deexc_le13_frac, 12.8, 2),
          within(100 * rb.ion_frac, 6.6, 2), within(rb.removed_geo_over_total, 1.30, 0.02), within(rb.b13m1 / 6.6e-5, 1, 0.03), within(rb.b14m1 / 6.1e-5, 1, 0.03), within(rb.b15m1 / 5.9e-5, 1, 0.03)]
    pc = [within(rc.total_loss / 5.64e10, 1, 0.01), within(100 * rc.deexc_14_frac, 74.3, 2), within(100 * rc.deexc_le13_frac, 15.2, 2), within(100 * rc.ion_frac, 10.5, 2), within(rc.removed_geo_over_total, 1.20, 0.02)]
    pmono = all(r.b13m1 > r.b14m1 > r.b15m1 > 0 for r in (rb, rc))
    say(f"\nPREDICTIONS: benchmark {sum(pb)}/{len(pb)} reproduced, cold corner {sum(pc)}/{len(pc)} reproduced, b_n - 1 > 0 and decreasing at both: {'yes' if pmono else 'NO'}")
    if not all(pb) or not all(pc): say("  not reproduced: " + ", ".join([f"benchmark item {n + 1}" for n, ok in enumerate(pb) if not ok] + [f"cold-corner item {n + 1}" for n, ok in enumerate(pc) if not ok]))
    negligible = rb.removed_geo_over_total < 0.10 or rc.removed_geo_over_total < 0.10
    say("REFUTER (removed upward loss < 10 % of retained at benchmark or cold corner): " + ("APPEARED -- truncation does not bias n=15 through this channel" if negligible else "did not appear"))
    say("\nASSUMPTION behind the removed-loss estimate: rho(n) = q(n->n+1)/q(n-1->n) continues one step past the ladder. This is an\n"
        "estimate of a channel the model does not contain, not a model output; rho(13) and the n^4 alternative bound how much the\n"
        "continuation choice matters. The budget fractions and b_n are exact properties of the stored operator.")
    if a.write:
        out = Path(a.out) if a.out else P("validation/terminal_shell_budget"); out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}"] + \
              [f"# {p.name} sha256 {sha(p)}" for p in paths.values()] + \
              [f"# terminal state k={k} ({row.label}); k14={k14} k13={k13} k12={k12}; L = R + ne C; removed 15->16 by geometric continuation rho(14) (n^4 as sensitivity);"
               " b_n from analytic Saha with state_index g, I_eV; fluxes per unit n_ion"]
        with open(out / "terminal_shell_budget.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "terminal_shell_budget.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/terminal_shell_budget.{{csv,txt}}")
    return 0

if __name__ == "__main__": sys.exit(main())
