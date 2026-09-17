"""
CCC vs Anderson 2002: full Te sweep, full thesis Te grid, and (Te, ne) impact
============================================================================

Three checks, in increasing order of what they actually test.

  [A] FULL ANDERSON Te RANGE, 0.5 - 25 eV (8 points).
      The rate-coefficient benchmark itself. K(Te) is a Maxwell average of
      sigma(E); it has NO ne dependence, so there is no "ne grid" for this
      check. Extending past 10 eV is where the 2002 corrigendum bites.

  [B] FULL THESIS Te GRID, 50 points, 1 - 10 eV.
      Anderson Upsilon log-log interpolated onto Te_grid_L (interpolation
      only -- 1-10 eV sits inside Anderson's 0.5-25 eV range), compared
      against the stored K table on its own native grid.

  [C] (Te, ne) IMPACT ON THE CR MODEL -- where ne finally enters.
      L(Te, ne) is linear in ne to machine precision, so it decomposes
      exactly as
                  L(Te, ne) = R(Te) + ne * C(Te)
      with R radiative and C collisional. We substitute Anderson's rate
      coefficients for CCC's in C for every transition Anderson covers
      (n <= 5, l-resolved, 85 transitions), rebuild L, and re-solve the
      timescales over all 50 x 8 grid points.

      This asks the question that matters: if the CCC data were replaced by
      the independent RMPS calculation wherever that calculation exists, would
      any thesis conclusion move? A discrepancy in a rate coefficient only
      matters insofar as it moves an observable.

      Detailed balance is preserved: excitation and de-excitation elements of
      a scaled transition are multiplied by the SAME factor. Column sums
      (net loss to the continuum) are preserved by compensating the diagonal.

Nothing is regenerated on disk: L_grid.npy is read, perturbed in memory only.

Run:  python src/validation/ccc_anderson_grid_impact.py
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

spec = importlib.util.spec_from_file_location(
    "and02bm", os.path.join(HERE, "anderson2002_benchmark.py"))
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

OUT = os.path.join(ROOT, "data", "processed", "collisions")
FIG = os.path.join(ROOT, "figures")

AND = bm.parse_anderson2002(bm.PDF_PATH)
CCC = pd.read_csv(bm.CCC_PATH)
GRP = CCC.groupby(["n_i", "l_i", "n_f", "l_f"])
KEYS = set(GRP.groups.keys())


def ccc_K_curve(n_lo, l_lo, n_up, l_up, Te_array):
    """Maxwell-average the CCC cross section at every Te in Te_array."""
    s  = GRP.get_group((n_lo, l_lo, n_up, l_up)).sort_values("E_eV")
    dE = bm.threshold_eV(n_lo, n_up)
    E  = np.linspace(dE + 1e-4, s.E_eV.max(), 5000)
    sg = np.interp(E, s.E_eV.values, s.sigma_a0sq.values, left=0.0, right=0.0)
    return np.array([bm.K_maxwell(sg, E, Te) for Te in Te_array])


def and_K_curve(row, n_lo, l_lo, n_up, Te_array):
    """Log-log interpolate Upsilon in Te, then apply Anderson Eq.(3)."""
    ups_tab = np.array([row[f"ups_{t:g}eV"] for t in bm.TE_AND], float)
    if Te_array.min() < bm.TE_AND.min() - 1e-9 or Te_array.max() > bm.TE_AND.max() + 1e-9:
        raise ValueError("Te outside Anderson's tabulated range -- would extrapolate.")
    ups = np.exp(np.interp(np.log(Te_array), np.log(bm.TE_AND), np.log(ups_tab)))
    return bm.K_from_upsilon(ups, n_lo, l_lo, n_up, Te_array), ups


# Every Anderson transition, as (label, n_lo, l_lo, n_up, l_up, row)
TRANS = []
for _, r in AND.iterrows():
    n_up, l_up = bm.IDX_TO_NL[int(r.i_upper)]
    n_lo, l_lo = bm.IDX_TO_NL[int(r.j_lower)]
    if (n_lo, l_lo, n_up, l_up) in KEYS:
        TRANS.append((f"{bm.nl_label(n_lo,l_lo)}->{bm.nl_label(n_up,l_up)}",
                      n_lo, l_lo, n_up, l_up, r))
print(f"{len(TRANS)} Anderson transitions matched in CCC\n")


# ══════════════════════════════════════════════════════════════════════════════
# [A] Full Anderson Te range, 0.5 - 25 eV
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("[A] FULL ANDERSON Te RANGE  0.5 - 25 eV   (rate coefficients; ne-independent)")
print("=" * 78)

rows = []
for lab, n_lo, l_lo, n_up, l_up, r in TRANS:
    Kc = ccc_K_curve(n_lo, l_lo, n_up, l_up, bm.TE_AND)
    for k, Te in enumerate(bm.TE_AND):
        Ka = bm.K_from_upsilon(r[f"ups_{Te:g}eV"], n_lo, l_lo, n_up, Te)
        rows.append({"label": lab, "Te_eV": Te, "n_lower": n_lo, "n_upper": n_up,
                     "l_upper": l_up, "dn": n_up - n_lo,
                     "dipole": abs(l_up - l_lo) == 1,
                     "K_CCC": Kc[k], "K_And2002": Ka,
                     "pct_err": (Kc[k] / Ka - 1.0) * 100.0})
dfA = pd.DataFrame(rows)
dfA.to_csv(os.path.join(OUT, "ccc_vs_anderson2002_full_Te_range.csv"), index=False)

print(f"  {'Te [eV]':>8} {'n':>5} {'within20%':>10} {'mean|err|':>10} "
      f"{'n<=4 within20%':>15} {'n<=4 mean|err|':>15}")
for Te in bm.TE_AND:
    s = dfA[dfA.Te_eV == Te]
    s4 = s[s.n_upper <= 4]
    print(f"  {Te:8.1f} {len(s):>5} {(s.pct_err.abs()<20).mean()*100:>9.1f}% "
          f"{s.pct_err.abs().mean():>9.2f}% {(s4.pct_err.abs()<20).mean()*100:>14.1f}% "
          f"{s4.pct_err.abs().mean():>14.2f}%")

print("\n  Dipole transitions only (the ones the 2002 corrigendum recomputed):")
print(f"  {'Te [eV]':>8} {'within20%':>10} {'mean|err|':>10} {'median signed':>14}")
for Te in bm.TE_AND:
    s = dfA[(dfA.Te_eV == Te) & dfA.dipole]
    print(f"  {Te:8.1f} {(s.pct_err.abs()<20).mean()*100:>9.1f}% "
          f"{s.pct_err.abs().mean():>9.2f}% {s.pct_err.median():>13.2f}%")

print("\n  Thesis anchors across the full range (% error):")
for n_lo, l_lo, n_up, l_up, nm in [(1,0,2,1,"1s->2p"), (1,0,3,1,"1s->3p"),
                                   (2,1,3,2,"2p->3d"), (2,0,3,1,"2s->3p")]:
    lab = f"{bm.nl_label(n_lo,l_lo)}->{bm.nl_label(n_up,l_up)}"
    s = dfA[dfA.label == lab].sort_values("Te_eV")
    print(f"    {nm:8s} " + "  ".join(f"{t:g}eV:{e:+6.1f}" for t, e in
                                      zip(s.Te_eV, s.pct_err)))

# ══════════════════════════════════════════════════════════════════════════════
# [B] Full 50-point thesis Te grid
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("[B] FULL THESIS Te GRID  (50 points, 1 - 10 eV, Te_grid_L)")
print("=" * 78)

TeL = np.load(os.path.join(ROOT, "data", "processed", "cr_matrix", "Te_grid_L.npy"))
neL = np.load(os.path.join(ROOT, "data", "processed", "cr_matrix", "ne_grid_L.npy"))
Te_ccc = np.load(os.path.join(ROOT, "data", "processed", "collisions", "ccc",
                              "Te_grid.npy"))
if not np.allclose(TeL, Te_ccc):
    raise ValueError("Te_grid_L and the CCC Te grid differ -- the stored K table "
                     "cannot be used against the CR grid without interpolation.")
print(f"  Te_grid_L ({len(TeL)}) == CCC Te grid: confirmed identical")
print(f"  ne_grid_L ({len(neL)}): {neL.min():.2e} - {neL.max():.2e} cm^-3")

K_st = np.load(os.path.join(ROOT, "data", "processed", "collisions", "ccc",
                            "K_CCC_exc_table.npy"))
meta = pd.read_csv(os.path.join(ROOT, "data", "processed", "collisions", "ccc",
                                "K_CCC_metadata.csv"))
idx_of = {(int(r.n_i), int(r.l_i), int(r.n_f), int(r.l_f)): int(r.idx)
          for _, r in meta.iterrows()}

ratio = np.zeros((len(TRANS), len(TeL)))     # K_Anderson / K_CCC, per transition
labels, nup_arr = [], []
rowsB = []
for m, (lab, n_lo, l_lo, n_up, l_up, r) in enumerate(TRANS):
    Ka, ups = and_K_curve(r, n_lo, l_lo, n_up, TeL)
    Kc = K_st[idx_of[(n_lo, l_lo, n_up, l_up)], :]
    ratio[m] = Ka / Kc
    labels.append(lab); nup_arr.append(n_up)
    for k, Te in enumerate(TeL):
        rowsB.append({"label": lab, "Te_eV": Te, "n_lower": n_lo, "n_upper": n_up,
                      "Upsilon_interp": ups[k], "K_CCC_stored": Kc[k],
                      "K_And2002": Ka[k], "pct_err": (Kc[k] / Ka[k] - 1.0) * 100.0})
dfB = pd.DataFrame(rowsB)
dfB.to_csv(os.path.join(OUT, "ccc_vs_anderson2002_thesis_Te_grid.csv"), index=False)
nup_arr = np.array(nup_arr)

print(f"  {len(TRANS)} transitions x {len(TeL)} Te = {len(dfB)} comparisons")
print(f"\n  {'subset':<22} {'n':>6} {'within20%':>10} {'mean|err|':>10} "
      f"{'median signed':>14}")
for sel, nm in [(np.ones(len(dfB), bool), "all"),
                (dfB.n_upper.values <= 4, "n_upper <= 4"),
                (dfB.n_upper.values <= 3, "n_upper <= 3"),
                (dfB.n_upper.values == 5, "n_upper == 5")]:
    s = dfB[sel]
    print(f"  {nm:<22} {len(s):>6} {(s.pct_err.abs()<20).mean()*100:>9.1f}% "
          f"{s.pct_err.abs().mean():>9.2f}% {s.pct_err.median():>13.2f}%")

print("\n  Variation across the 50-point grid (is the error Te-stable?):")
g = dfB[dfB.n_upper <= 4].groupby("Te_eV").pct_err.apply(lambda x: x.abs().mean())
print(f"    n_upper<=4 mean|err|:  min {g.min():.2f}% @ Te={g.idxmin():.2f} eV,  "
      f"max {g.max():.2f}% @ Te={g.idxmax():.2f} eV")
g5 = dfB[dfB.n_upper == 5].groupby("Te_eV").pct_err.apply(lambda x: x.abs().mean())
print(f"    n_upper==5 mean|err|:  min {g5.min():.2f}% @ Te={g5.idxmin():.2f} eV,  "
      f"max {g5.max():.2f}% @ Te={g5.idxmax():.2f} eV")

# ══════════════════════════════════════════════════════════════════════════════
# [C] (Te, ne) impact on CR timescales
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("[C] (Te, ne) GRID IMPACT: substitute Anderson rates into L, re-solve")
print("=" * 78)

L = np.load(os.path.join(ROOT, "data", "processed", "cr_matrix", "L_grid.npy"))
si = pd.read_csv(os.path.join(ROOT, "data", "processed", "collisions",
                              "K_exc_full", "state_index.csv"))
nl_to_state = {(int(r.n), int(r.l)): int(r.idx)
               for _, r in si.iterrows() if not r.bundled}
nT, nN, nS, _ = L.shape
print(f"  L_grid {L.shape};  {len(nl_to_state)} l-resolved states, "
      f"{int(si.bundled.sum())} bundled")

# --- decompose L = R + ne*C, and verify the decomposition is exact ---
Amat = np.vstack([np.ones_like(neL), neL]).T
coef = np.linalg.lstsq(Amat, L.transpose(1, 0, 2, 3).reshape(nN, -1), rcond=None)[0]
R = coef[0].reshape(nT, nS, nS)
C = coef[1].reshape(nT, nS, nS)
recon = R[:, None] + neL[None, :, None, None] * C[:, None]
rel = np.abs(recon - L).max() / np.abs(L).max()
print(f"  L = R + ne*C  exact to {rel:.2e} (relative) -- decomposition verified")
if rel > 1e-10:
    raise ValueError("L is not linear in ne; the substitution below is invalid.")

def timescales(Lg):
    """tau_QSS = 1/|lam0|, tau_relax = 1/|lam1| from the two slowest modes."""
    ev = np.linalg.eigvals(Lg.reshape(-1, nS, nS)).real
    ev = np.sort(np.abs(ev), axis=1)          # ascending |Re lambda|
    l0, l1 = ev[:, 0], ev[:, 1]
    return (1.0 / l0).reshape(nT, nN), (1.0 / l1).reshape(nT, nN)

tQ0, tR0 = timescales(L)
print(f"\n  Baseline at CLAUDE.md benchmark point (Te idx 23 = {TeL[23]:.3f} eV, "
      f"ne idx 5 = {neL[5]:.3e} cm^-3):")
print(f"    tau_QSS   = {tQ0[23,5]*1e6:.3f} us   (recorded: 22.73 us)")
print(f"    tau_relax = {tR0[23,5]*1e9:.4f} ns   (recorded: 2.277 ns)")
print(f"    M         = {tQ0[23,5]/tR0[23,5]:.1f}      (recorded: 9982)")
ok = (abs(tQ0[23,5]*1e6 - 22.73) < 0.05) and (abs(tR0[23,5]*1e9 - 2.277) < 0.005)
print(f"    reproduces recorded values: {'YES' if ok else 'NO -- investigate'}")

# --- substitute Anderson rates into the collisional block ---
Cp = C.copy()
n_sub = 0
for m, (lab, n_lo, l_lo, n_up, l_up, r) in enumerate(TRANS):
    i, j = nl_to_state.get((n_lo, l_lo)), nl_to_state.get((n_up, l_up))
    if i is None or j is None:
        continue
    f = ratio[m]                                  # (nT,) Anderson/CCC
    for (a, b) in ((j, i), (i, j)):               # excitation and de-excitation
        d = C[:, a, b] * (f - 1.0)                # change in the off-diagonal
        Cp[:, a, b] += d
        Cp[:, b, b] -= d                          # keep column sum (loss) intact
    n_sub += 1
print(f"\n  Substituted Anderson rates for {n_sub} transitions "
      f"(both directions, detailed balance preserved)")
csum = np.abs((Cp - C).sum(axis=1)).max()
print(f"  Column-sum drift after substitution: {csum:.2e} "
      f"(must be ~0; continuum sink unchanged)")

Lp = R[:, None] + neL[None, :, None, None] * Cp[:, None]
tQ1, tR1 = timescales(Lp)

dQ = (tQ1 / tQ0 - 1.0) * 100.0
dR = (tR1 / tR0 - 1.0) * 100.0
dM = ((tQ1 / tR1) / (tQ0 / tR0) - 1.0) * 100.0

print(f"\n  Change over the WHOLE {nT} x {nN} grid (CCC -> Anderson, where available):")
for nm, d in [("tau_QSS", dQ), ("tau_relax", dR), ("M = tQSS/trelax", dM)]:
    k = np.unravel_index(np.abs(d).argmax(), d.shape)
    print(f"    {nm:16s} mean|d| {np.abs(d).mean():6.3f}%   max|d| {np.abs(d).max():6.3f}%"
          f"   at Te={TeL[k[0]]:.2f} eV, ne={neL[k[1]]:.2e}")

print(f"\n  ne dependence of the impact (mean |d| over the 50 Te points):")
print(f"    {'ne [cm^-3]':>12} {'d tau_QSS':>12} {'d tau_relax':>12} {'d M':>10}")
for b in range(nN):
    print(f"    {neL[b]:12.2e} {np.abs(dQ[:,b]).mean():11.3f}% "
          f"{np.abs(dR[:,b]).mean():11.3f}% {np.abs(dM[:,b]).mean():9.3f}%")

print(f"\n  Te dependence of the impact (mean |d| over the 8 ne points, every 7th Te):")
print(f"    {'Te [eV]':>9} {'d tau_QSS':>12} {'d tau_relax':>12} {'d M':>10}")
for a in range(0, nT, 7):
    print(f"    {TeL[a]:9.3f} {np.abs(dQ[a]).mean():11.3f}% "
          f"{np.abs(dR[a]).mean():11.3f}% {np.abs(dM[a]).mean():9.3f}%")

print(f"\n  At the CLAUDE.md benchmark point (23, 5):")
print(f"    tau_QSS   {tQ0[23,5]*1e6:8.3f} -> {tQ1[23,5]*1e6:8.3f} us  ({dQ[23,5]:+.3f}%)")
print(f"    tau_relax {tR0[23,5]*1e9:8.4f} -> {tR1[23,5]*1e9:8.4f} ns  ({dR[23,5]:+.3f}%)")
print(f"    M         {tQ0[23,5]/tR0[23,5]:8.1f} -> {tQ1[23,5]/tR1[23,5]:8.1f}      "
      f"({dM[23,5]:+.3f}%)")

pd.DataFrame({"Te_eV": np.repeat(TeL, nN), "ne_cm3": np.tile(neL, nT),
              "tau_QSS_s": tQ0.ravel(), "tau_QSS_And_s": tQ1.ravel(),
              "tau_relax_s": tR0.ravel(), "tau_relax_And_s": tR1.ravel(),
              "pct_d_tau_QSS": dQ.ravel(), "pct_d_tau_relax": dR.ravel(),
              "pct_d_M": dM.ravel()}).to_csv(
    os.path.join(OUT, "ccc_anderson_grid_impact.csv"), index=False)
print(f"\n  wrote data/processed/collisions/ccc_anderson_grid_impact.csv")

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
fig.suptitle("CCC vs Anderson 2002 — full Te range, thesis grid, and (Te, ne) impact",
             fontsize=12, fontweight="bold")

a = ax[0]
for sel, nm, c in [(dfA.n_upper <= 4, "n_upper ≤ 4", "tab:green"),
                   (dfA.n_upper == 5, "n_upper = 5", "tab:orange")]:
    g = dfA[sel].groupby("Te_eV").pct_err.apply(lambda x: x.abs().mean())
    a.plot(g.index, g.values, "o-", color=c, lw=2, label=nm)
g = dfA[dfA.dipole].groupby("Te_eV").pct_err.apply(lambda x: x.abs().mean())
a.plot(g.index, g.values, "s--", color="tab:blue", lw=1.6, label="dipole (all n)")
a.axhline(20, color="r", ls=":", lw=1)
a.axvspan(1, 10, color="k", alpha=.06)
a.set(xscale="log", xlabel="Te [eV]", ylabel="mean |% error|",
      title="[A] full Anderson range\n(shaded = thesis range)")
a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1]
for sel, nm, c in [(dfB.n_upper.values <= 3, "n_upper ≤ 3", "tab:blue"),
                   (dfB.n_upper.values == 4, "n_upper = 4", "tab:green"),
                   (dfB.n_upper.values == 5, "n_upper = 5", "tab:orange")]:
    g = dfB[sel].groupby("Te_eV").pct_err.apply(lambda x: x.abs().mean())
    a.plot(g.index, g.values, "-", color=c, lw=2, label=nm)
a.axhline(20, color="r", ls=":", lw=1)
a.set(xlabel="Te [eV]", ylabel="mean |% error|",
      title="[B] thesis Te grid (50 pts)")
a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[2]
im = a.pcolormesh(TeL, neL, np.abs(dM).T, shading="nearest", cmap="magma")
a.set(yscale="log", xlabel="Te [eV]", ylabel="ne [cm$^{-3}$]",
      title="[C] |Δ M| when CCC→Anderson  [%]")
a.plot(TeL[23], neL[5], "c*", ms=14, mec="k", label="benchmark pt")
a.legend(fontsize=8, loc="lower right")
fig.colorbar(im, ax=a, label="|Δ M| [%]")

plt.tight_layout()
fp = os.path.join(FIG, "ccc_anderson_grid_impact.png")
fig.savefig(fp, dpi=150, bbox_inches="tight")
print(f"  wrote figures/ccc_anderson_grid_impact.png")
print("\nDone.")
