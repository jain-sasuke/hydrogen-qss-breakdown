#!/usr/bin/env python
"""
verify_high_n_ionisation_scaling.py
===================================
What the high-n Lotz ionisation block, and the three-body recombination
built from it, do to the reservoir quantities: u_CRE, tau_slow, Delta ln u,
eps_plateau and the defended census.

WHY THIS EXISTS
---------------
The model takes ionisation from CCC TICS for n <= 9 and from Lotz (1968) for
n = 10-15, and its own metadata records that Lotz "overestimates CCC by a
factor ~4-8" there. Because alpha_3BR is built from the same coefficient by
detailed balance (Saha), the same block feeds the high-n recombination
ladder. A Round 2 reviewer called this a load-bearing unresolved input for
the reservoir balance, distinct from the Balmer-shell sensitivity already
tested. No alternative dataset exists above n = 9, so this is a scan, not a
substitution: the six Lotz rows and their 3BR are scaled together by
s = 1/2, 1/4, 1/8 (the direction the metadata implies) and everything the
reservoir depends on is recomputed.

METHOD (the machinery of verify_recombination_substitution.py)
--------------------------------------------------------------
  S = ne alpha_RR + ne^2 alpha_3BR per unit n_ion;  alpha_3BR = K_ion * Saha factor
  scaled:  K' = K on all rows except the Lotz rows, where K' = s K;
           alpha_3BR' = K' * (same Saha factor);  L' diagonal -= (K' - K) ne
  at every (direction, k = 1, i, j): tau_slow, tau_relax of the post-step
  operator; two_channel -> Delta ln u (lnx), Sbar, eps_plateau, Delta, cap;
  ELM estimate = eps (tau_slow/tau_d)(1 - e^{-tau_d/tau_slow}), tau_d = 1e-4 s,
  the census statistic of verify_divertor_map.py;  u_CRE = [-L^{-1} S]_g at
  every grid point.

GATES (all must pass before any scaled number is reported)
---------------------------------------------------------
  A  this script's Lotz reproduces the stored K_ion on the six Lotz rows
  B  S_grid reconstructs from the alpha arrays
  C  alpha_3BR / K_ion equals the analytic Saha factor
  D  the baseline two-channel pass reproduces validation/reservoir_gain/reservoir_gain.csv
  D' the baseline ELM estimate reproduces divertor_map.csv lo_ELM_crash, and
     baseline u_CRE reproduces molecular_channel.csv

PREDICTIONS (written before the run)
-----------------------------------
Scaling ionisation out of n >= 10 and 3BR into n >= 10 together preserves
their Saha ratio with the continuum and only slows the exchange, so
P1  u_CRE moves by < 5 % at every grid point under s = 1/8, and < 2 % at ne >= 1e14
P2  tau_slow (post-step) moves by < 5 % at every defended pair under s = 1/8
P3  eps_plateau moves by < 5 % at the median over the 448 defended pairs under s = 1/8
P4  the defended census (ELM estimate > 0.10) stays within +-2 of 45 under every s
REFUTER (the reviewer's "load-bearing" confirmed): eps_plateau or tau_slow
moving by more than 10 % at any defended pair.

OUTPUTS (with --write): validation/high_n_ionisation_scaling/high_n_ionisation_scaling.{csv,txt}
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import exp1

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE); IH_EV = 13.6058; TAU_D = 1e-4
_s = importlib.util.spec_from_file_location("rga", _HERE.parent / "verify_reservoir_gain_anderson.py")
rga = importlib.util.module_from_spec(_s); _s.loader.exec_module(rga); two_channel = rga.two_channel

def lotz_K_ion(n, Te):
    """Lotz (1968) Eq.(5), as in verify_recombination_substitution.py / ionization_rates.py."""
    x = (IH_EV / n**2) / Te
    return 6.7e-7 * 4.5 / Te**1.5 * (1.0 / x) * exp1(x)

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    ap.add_argument("--win-lo", type=float, default=30.0); ap.add_argument("--win-hi", type=float, default=30.0)
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; g, nv = ctx.ground_index, ctx.n_values; nT, nN, nS, _ = L.shape
    P = lambda *p: ROOT.joinpath(*p)
    S = np.load(P("data/processed/cr_matrix/S_grid.npy"))
    K_ion = np.load(P("data/processed/collisions/tics/K_ion_final.npy")); kmeta = pd.read_csv(P("data/processed/collisions/tics/K_ion_final_meta.csv"))
    aRR = np.concatenate([np.load(P("data/processed/recombination/alpha_RR_resolved.npy")), np.load(P("data/processed/recombination/alpha_RR_bundled.npy"))])
    a3BR = np.concatenate([np.load(P("data/processed/recombination/alpha_3BR_resolved.npy")), np.load(P("data/processed/recombination/alpha_3BR_bundled.npy"))])
    si = pd.read_csv(ctx.state_index_path)
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("HIGH-n IONISATION / 3BR SCALING -- what the Lotz block does to the reservoir")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}"); say(ctx.describe()); say("=" * 78)

    lotz_rows = kmeta.index[kmeta.source.astype(str).str.startswith("Lotz")].tolist()
    say(f"\nLotz-sourced rows: {len(lotz_rows)}, n = {sorted(int(kmeta.iloc[i].n) for i in lotz_rows)}")
    worstA = max(np.abs(lotz_K_ion(int(kmeta.iloc[i].n), te) / K_ion[i] - 1).max() for i in lotz_rows)
    say(f"  A  Lotz reproduces the stored K_ion on those rows: max rel diff {worstA:.3e}")
    if worstA > 1e-12: raise RuntimeError("Gate A")
    S_rec = ne[None, :, None] * (aRR.T[:, None, :] + ne[None, :, None] * a3BR.T[:, None, :])
    relB = np.abs(S_rec - S).max() / np.abs(S).max(); say(f"  B  S_grid reconstructs from the alpha arrays: max rel diff {relB:.3e}")
    if relB > 1e-12: raise RuntimeError("Gate B")
    h, me_kg, kB = 6.62607015e-34, 9.1093837015e-31, 1.380649e-23
    saha = (si.g.values.astype(float)[:, None] / 2.0) * (h**2 / (2 * np.pi * me_kg * te[None, :] * 1.60218e-19))**1.5 * 1e6 * np.exp(si.I_eV.values.astype(float)[:, None] / te[None, :])
    fac = np.where(K_ion > 0, a3BR / np.maximum(K_ion, 1e-300), np.nan)
    relC = np.nanmax(np.abs(fac / saha - 1)); say(f"  C  alpha_3BR / K_ion against the analytic Saha factor: max rel diff {relC:.3e}")
    if relC > 5e-3: raise RuntimeError("Gate C")

    idx = np.arange(nS)
    def build(s):
        K = K_ion.copy(); K[lotz_rows] *= s
        Ss = ne[None, :, None] * (aRR.T[:, None, :] + ne[None, :, None] * (K * fac).T[:, None, :])
        Ls = L.copy(); Ls[:, :, idx, idx] -= (K - K_ion).T[:, None, :] * ne[None, :, None]
        return Ls, Ss
    cases = [("base", L, S)] + [(f"s{s:g}", *build(s)) for s in a.scales]
    for nm, Lc, Sc in cases[1:]:
        colsum = Lc.sum(axis=2); Kc = K_ion.copy(); Kc[lotz_rows] *= float(nm[1:])
        relE = np.abs(colsum + Kc.T[:, None, :] * ne[None, :, None]).max() / np.abs(Kc.T[:, None, :] * ne[None, :, None]).max()
        if relE > 1e-10: raise RuntimeError(f"ionisation identity broken for {nm}: {relE:.2e}")
    say("  E  column sums of every scaled L equal -K'_ion ne (ionisation identity): OK")

    # u_CRE at every grid point, every case
    u = {nm: np.array([[np.linalg.solve(Lc[i, j], -Sc[i, j])[g] for j in range(nN)] for i in range(nT)]) for nm, Lc, Sc in cases}
    mol = pd.read_csv(P("validation/molecular_channel/molecular_channel.csv"), comment="#").sort_values(["i", "j"])
    relU = np.abs(u["base"].ravel() / mol.u_CRE.values - 1).max(); say(f"  D' baseline u_CRE reproduces molecular_channel.csv: max rel diff {relU:.3e}")
    if relU > 1e-8: raise RuntimeError("Gate D' (u_CRE)")

    E = np.array([i for i in range(nS) if i != g]); N3 = np.where(nv == 3)[0]; N4 = np.where(nv == 4)[0]
    pos = {s_: k for k, s_ in enumerate(E)}; n3E = np.array([pos[s_] for s_ in N3]); n4E = np.array([pos[s_] for s_ in N4])
    rows = []
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(nT):
            jt = i + sgn
            if not (0 <= jt < nT): continue
            dlnTe = np.log(te[jt] / te[i])
            for j in range(nN):
                rec = dict(direction=dlab, k=1, i=i, j=j, Te=float(te[i]), ne=float(ne[j]), dlnTe=dlnTe)
                for nm, Lc, Sc in cases:
                    lam = np.sort(np.linalg.eigvals(Lc[jt, j]).real)[::-1]; neg = lam[lam < 0]
                    tQ, tR = 1.0 / abs(neg[0]), 1.0 / abs(neg[1])
                    d = two_channel(Lc[jt, j], Sc[jt, j], Lc[i, j], Sc[i, j], E, g, n3E, n4E, N3, N4, f"{nm} {i},{j}")
                    sfx = "" if nm == "base" else f"_{nm}"
                    if nm == "base":
                        rec["M"] = tQ / tR; rec["window_ok"] = bool((a.win_lo * tR) < (tQ / a.win_hi))
                    rec[f"tau_slow{sfx}"] = tQ; rec[f"tau_relax{sfx}"] = tR; rec[f"lnu{sfx}"] = d["lnx"]; rec[f"G{sfx}"] = d["lnx"] / dlnTe
                    rec[f"Sbar{sfx}"] = d["Sbar"]; rec[f"eps{sfx}"] = d["eps"]; rec[f"Delta{sfx}"] = d["Delta"]; rec[f"cap{sfx}"] = d["cap"]
                    rec[f"elm{sfx}"] = d["eps"] * (tQ / TAU_D) * (1 - np.exp(-TAU_D / tQ)); rec[f"uCRE_pre{sfx}"] = u[nm][i, j]
                rows.append(rec)
    df = pd.DataFrame(rows)
    ref = pd.read_csv(P("validation/reservoir_gain/reservoir_gain.csv")); ref = ref[ref.k == 1]
    m = df.merge(ref, on=["direction", "k", "i", "j"], suffixes=("", "_ref"))
    if len(m) != len(ref): raise RuntimeError(f"Gate D row mismatch {len(m)} vs {len(ref)}")
    wd = max(np.abs((m[q] - m[f"{q}_ref"]) / m[f"{q}_ref"]).max() for q in ("G", "Sbar", "eps"))
    say(f"  D  baseline reproduces reservoir_gain.csv (k=1, {len(m)} rows): max rel diff {wd:.3e}")
    if wd > 1e-10: raise RuntimeError("Gate D")
    dm = pd.read_csv(P("validation/divertor_map/divertor_map.csv"), comment="#")
    m2 = df.merge(dm[["direction", "i", "j", "lo_ELM_crash", "window_ok"]], on=["direction", "i", "j"], suffixes=("", "_dm"))
    if len(m2) != len(dm): raise RuntimeError(f"Gate D' row mismatch {len(m2)} vs {len(dm)}")
    wdm = np.abs(m2.elm - m2.lo_ELM_crash).max() / m2.lo_ELM_crash.max()
    say(f"  D' baseline ELM estimate reproduces divertor_map.csv lo_ELM_crash ({len(m2)} rows): max rel diff {wdm:.3e};  window_ok agrees at {int((m2.window_ok == m2.window_ok_dm).sum())}/{len(m2)}")
    if wdm > 1e-8 or not (m2.window_ok == m2.window_ok_dm).all(): raise RuntimeError("Gate D' (ELM estimate)")
    say("\n  ALL GATES PASSED.")

    dfd = df[df.window_ok & (df.Te >= 2.0)]; n_def = len(dfd)
    base_census = int((dfd.elm > 0.10).sum())
    say("\n" + "=" * 78); say(f"RESULT over the {n_def} defended pairs (window_ok, Te >= 2 eV, k = 1);  baseline census {base_census}")
    say("=" * 78)
    hi = ne >= 1e14; verdict = {}
    for nm, _, _ in cases[1:]:
        say(f"\n  scale {nm[1:]} on n = 10-15 ionisation and its 3BR")
        for q, lab in (("uCRE_pre", "u_CRE (pre-step point)"), ("tau_slow", "tau_slow (post-step)"), ("lnu", "Delta ln u"), ("eps", "eps_plateau"), ("elm", "ELM estimate"), ("Sbar", "Sbar"), ("G", "G"), ("cap", "cap")):
            d = (dfd[f"{q}_{nm}"] / dfd[q] - 1) * 100
            say(f"    {lab:24s} median {d.median():+7.2f}%   mean|.| {d.abs().mean():6.2f}%   max|.| {d.abs().max():6.2f}%  (at [{int(dfd.loc[d.abs().idxmax(),'i'])},{int(dfd.loc[d.abs().idxmax(),'j'])}] {dfd.loc[d.abs().idxmax(),'direction']})")
        du = (u[nm] / u["base"] - 1) * 100
        say(f"    u_CRE over all 400 grid points: max|.| {np.abs(du).max():.2f}%;  over ne >= 1e14: max|.| {np.abs(du[:, hi]).max():.2f}%")
        say("    u_CRE max|.| by density column: " + "  ".join(f"{ne[j]:.0e}:{np.abs(du[:, j]).max():.2f}%" for j in range(nN)))
        cen = int((dfd[f"elm_{nm}"] > 0.10).sum()); say(f"    defended census (ELM estimate > 0.10): {cen} of {n_def}  (baseline {base_census})")
        de = (dfd[f"eps_{nm}"] / dfd.eps - 1).abs() * 100; dt = (dfd[f"tau_slow_{nm}"] / dfd.tau_slow - 1).abs() * 100
        verdict[nm] = dict(u_all=np.abs(du).max(), u_hi=np.abs(du[:, hi]).max(), tau_max=dt.max(), eps_med=de.median(), eps_max=de.max(), census=cen)
        for (i, j, lab) in [(23, 5, "benchmark [23,5]"), (15, 3, "ridge [15,3]"), (0, 4, "cold corner [0,4]")]:
            r = df[(df.direction == "heat") & (df.i == i) & (df.j == j)].iloc[0]
            say(f"    {lab:>18}: eps {r.eps*100:7.3f}% -> {r[f'eps_{nm}']*100:7.3f}%   tau_slow {r.tau_slow*1e6:9.3f} -> {r[f'tau_slow_{nm}']*1e6:9.3f} us   u_CRE {r.uCRE_pre:.4e} -> {r[f'uCRE_pre_{nm}']:.4e}")
    s8 = [nm for nm, _, _ in cases[1:] if abs(float(nm[1:]) - 0.125) < 1e-9]
    missed = []; refuted = False
    if s8:
        v = verdict[s8[0]]
        if not (v["u_all"] < 5 and v["u_hi"] < 2): missed.append("P1")
        if not v["tau_max"] < 5: missed.append("P2")
        if not v["eps_med"] < 5: missed.append("P3")
    if any(abs(v["census"] - 45) > 2 for v in verdict.values()): missed.append("P4")
    refuted = any(v["eps_max"] > 10 or v["tau_max"] > 10 for v in verdict.values())
    say("\nPREDICTIONS: " + ("P1-P4 all reproduced" if not missed else "not reproduced as written: " + ", ".join(missed)))
    say("REFUTER (eps_plateau or tau_slow moving > 10 % at any defended pair): " + ("APPEARED -- the high-n block reaches the reservoir quantities" if refuted else "did not appear"))
    say("\nThis is a scan, not a substitution: no second dataset exists above n = 9. The scale is applied uniformly in\n"
        "Te and to all six Lotz rows; the 3BR is rebuilt with the same Saha factor, so detailed balance is preserved.")
    if a.write:
        out = Path(a.out) if a.out else P("validation/high_n_ionisation_scaling"); out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}",
               f"# L_grid.npy sha256 {sha(P('data/processed/cr_matrix/L_grid.npy'))}", f"# S_grid.npy sha256 {sha(P('data/processed/cr_matrix/S_grid.npy'))}",
               f"# K_ion_final.npy sha256 {sha(P('data/processed/collisions/tics/K_ion_final.npy'))}", f"# state_index.csv sha256 {sha(ctx.state_index_path)}",
               f"# scales {a.scales} on Lotz rows n = 10-15; tau_d = {TAU_D} s; window {a.win_lo}/{a.win_hi}"]
        with open(out / "high_n_ionisation_scaling.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "high_n_ionisation_scaling.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/high_n_ionisation_scaling.{{csv,txt}}")
    return 0

if __name__ == "__main__": sys.exit(main())
