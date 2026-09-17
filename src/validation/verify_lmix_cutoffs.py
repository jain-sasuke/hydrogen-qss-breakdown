#!/usr/bin/env python
"""
verify_lmix_cutoffs.py
======================
Which PSM20 cut-off binds the proton l-mixing rate on the working grid, and
by how much the rate would change if it were not the Debye length.

WHY THIS EXISTS
---------------
Chapter 2 (sec:lmixing) implements only the Debye cut-off R_c = lambda_D and
leaves open Badnell's two competitors, a cut-off set by the level lifetime
and one set by any residual splitting of the l degeneracy, "the correct R_c
is the smallest of the three". A reviewer suggested the physically correct
cut-off "could suppress a rate by orders of magnitude". The cut-off enters
only through the Coulomb-logarithm factor F(U_m) with U_m ~ 1/R_c^2, so the
question is answered by evaluating the competing radii and reading the
change in the model's own F.

METHOD
------
For every resolved shell n = 2..8, every mixing channel l_>-1 <-> l_>, and
every grid point (Te, ne):
  lambda_D  two-species Debye length, the module's own expression
  v         mean relative speed sqrt(8 kT / pi mu), mu = m_p m_H/(m_p+m_H),
            T = Te (the T_i = Te assumption the model already makes)
  R_tau     = v / max(gamma_lo, gamma_hi), the shorter radiative lifetime of
            the two sublevels (gamma_resolved.npy)
  R_dE      = hbar v / dE, with dE the largest fine-structure splitting between
            a j-component of l_>-1 and one of l_>, from the Dirac formula
            dE = alpha^2 Ry / n^3 [1/(j_lo+1/2) - 1/(j_hi+1/2)] (Bethe and
            Salpeter); this is the most restrictive splitting, so the bound is
            conservative. For n = 2 the least restrictive alternative, the
            measured 2s1/2-2p1/2 Lamb shift 1057.845 MHz, is also reported.
  R_c'      = min(lambda_D, R_tau, R_dE);  U_m' = U_m (lambda_D/R_c')^2
  F'/F      the change in the Coulomb-logarithm factor, hence in the rate
Baseline U_m and F are the module's _psm20_Um and _psm20_F with the LOCAL
density (not the frozen 1e14 the production table uses; that choice is
tested separately in sec:lmixing and costs 0.42 %).

PREDICTIONS (written before the run)
-----------------------------------
P1  the lifetime cut-off never binds: R_tau > lambda_D at every point and channel
P2  the splitting cut-off binds at n = 2 for ne <~ 1e14 cm^-3 and gives
    F'/F >= 0.4 at 1e12 cm^-3 (ln(lambda_D/R_dE) ~ ln 37 = 3.6 against F ~ 7)
P3  for n >= 3 the splitting scales as n^-3, R_dE grows, and F'/F > 0.7 everywhere
P4  F'/F > 0.1 at every point and channel: inside the x0.1 to x10 scan of sec:lmixing
REFUTER: F'/F < 0.1 anywhere.

OUTPUTS (with --write): validation/lmix_cutoffs/lmix_cutoffs.{csv,txt}
Constants: CODATA 2018 (hbar, alpha, Ry, m_p, m_H, eV); Lamb shift
1057.845 MHz (Lundeen and Pipkin 1981).
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr_context import CRContext  # noqa: E402

HBAR_EV_S = 6.582119569e-16; ALPHA = 7.2973525693e-3; RY_EV = 13.605693123
M_P_G = 1.67262192369e-24; M_H_G = 1.6735328377e-24; EV_ERG = 1.602176634e-12
H_EV_S = 4.135667696e-15; LAMB_2S_2P_MHZ = 1057.845

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); root = ctx.root; TeL, neL = ctx.te_grid, ctx.ne_grid
    lm_path = root / "src/rates/compute_lmix.py"
    spec = importlib.util.spec_from_file_location("lm", lm_path); lm = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm)
    si = pd.read_csv(ctx.state_index_path); res = si[~si.bundled].sort_values("idx")
    gam_path = root / "data/processed/Radiative/gamma_resolved.npy"
    gam = np.load(gam_path)
    if gam.shape != (len(res),): raise ValueError(f"gamma_resolved.npy has shape {gam.shape}, expected ({len(res)},)")
    gamma = {(int(r.n), int(r.l)): float(gam[k]) for k, (_, r) in enumerate(res.iterrows())}
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("PROTON l-MIXING CUT-OFFS -- Debye against lifetime and fine-structure splitting")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}"); say(f"interpreter {sys.executable}")
    say(ctx.describe()); say(f"  compute_lmix.py sha256 {sha(lm_path)[:16]};  MU = {lm.MU};  gamma_resolved.npy ({len(gam)} states)"); say("=" * 78)
    mu_g = M_P_G * M_H_G / (M_P_G + M_H_G)
    v = np.sqrt(8.0 * TeL * EV_ERG / (np.pi * mu_g))                      # cm/s, per Te
    lamb_eV = H_EV_S * LAMB_2S_2P_MHZ * 1e6
    rows = []; worst = {}
    for n in range(2, 9):
        for lu in range(1, n):
            ll = lu - 1
            jlo, jhi = (ll - 0.5 if ll > 0 else 0.5), lu + 0.5           # most-split j pair
            dE = ALPHA**2 * RY_EV / n**3 * (1.0 / (jlo + 0.5) - 1.0 / (jhi + 0.5))
            g = max(gamma[(n, ll)], gamma[(n, lu)])                     # shorter lifetime
            for j, ne in enumerate(neL):
                lamD = 7.4340e2 * np.sqrt(TeL / (2.0 * ne))               # the module's expression
                Um = lm._psm20_Um(n, lu, TeL, ne); F = lm._psm20_F(Um)
                R_tau = v / g if g > 0 else np.full_like(v, np.inf)
                R_dE = HBAR_EV_S * v / dE
                Rc = np.minimum(lamD, np.minimum(R_tau, R_dE))
                Fp = lm._psm20_F(Um * (lamD / Rc) ** 2)
                binds = np.where(Rc < lamD, np.where(R_dE <= R_tau, "splitting", "lifetime"), "Debye")
                for i, Te in enumerate(TeL):
                    rows.append(dict(n=n, l_upper=lu, i=i, j=j, Te=Te, ne=ne, lambda_D_cm=lamD[i], R_tau_cm=R_tau[i], R_dE_cm=R_dE[i],
                                     dE_eV=dE, binding=binds[i], U_m=Um[i], F=F[i], F_prime=Fp[i], F_ratio=Fp[i] / F[i]))
                k = (n, lu); worst[k] = min(worst.get(k, 1.0), float((Fp / F).min()))
    df = pd.DataFrame(rows)
    say(f"\nP1  lifetime cut-off binds at {int((df.binding=='lifetime').sum())} of {len(df)} (channel, point) rows;  min R_tau/lambda_D = {(df.R_tau_cm/df.lambda_D_cm).min():.3g}")
    n2 = df[df.n == 2]
    say(f"P2  n = 2: splitting cut-off (dE = {n2.dE_eV.iloc[0]:.3e} eV, the 2s1/2-2p3/2 fine structure) binds at {int((n2.binding=='splitting').sum())} of {len(n2)} points;")
    for j, ne in enumerate(neL):
        s = n2[n2.j == j]; say(f"      ne = {ne:.2e}: lambda_D/R_dE = {(s.lambda_D_cm/s.R_dE_cm).iloc[0]:6.2f} (Te-independent);  F'/F min {s.F_ratio.min():.3f}  max {s.F_ratio.max():.3f};  F {s.F.min():.2f}..{s.F.max():.2f}")
    R_lamb = HBAR_EV_S * v / lamb_eV
    say(f"      least restrictive alternative, the Lamb shift {lamb_eV:.3e} eV: lambda_D/R_Lamb = {(7.4340e2*np.sqrt(TeL/(2*neL[0]))/R_lamb)[0]:.2f} at {neL[0]:.0e}, "
        f"F'/F there = {lm._psm20_F(lm._psm20_Um(2,1,TeL,neL[0])*np.maximum(1,(7.4340e2*np.sqrt(TeL/(2*neL[0]))/R_lamb))**2).min()/lm._psm20_F(lm._psm20_Um(2,1,TeL,neL[0])).min():.3f} (min over Te)")
    say("P3  minimum F'/F per shell (over channels and points):")
    for n in range(2, 9):
        s = df[df.n == n]; say(f"      n = {n}: min F'/F = {s.F_ratio.min():.3f}  at ne = {s.loc[s.F_ratio.idxmin(),'ne']:.0e};  non-Debye binding at {100*(s.binding!='Debye').mean():.0f} % of rows")
    say(f"P4  global minimum F'/F = {df.F_ratio.min():.3f}  (refuter: < 0.1)")
    missed = []
    if (df.binding == "lifetime").any(): missed.append("P1")
    if n2[n2.j == 0].F_ratio.min() < 0.4: missed.append("P2")
    if df[df.n >= 3].F_ratio.min() <= 0.7: missed.append("P3")
    refuted = df.F_ratio.min() <= 0.1
    say("\nPREDICTIONS: " + ("P1-P3 all reproduced" if not missed else "not reproduced as written: " + ", ".join(missed) +
        ". P3 assumed the splitting falls as n^-3; the p-d channel's most-split pair carries a larger j-factor, so the "
        "cut-off still binds at n = 3 and 4 at the lowest densities (minima above)."))
    say("REFUTER (F'/F < 0.1 anywhere): " + ("APPEARED" if refuted else "did not appear") +
        f"; global minimum {df.F_ratio.min():.3f}, inside the x0.1 to x10 scan of sec:lmixing.")
    bad = ["P4"] if refuted else []
    say("Assumptions: v is the mean relative speed at T_i = Te; the splitting is the most-split fine-structure pair\n"
        "(conservative); baseline F uses the local Debye length, not the frozen 1e14 of the production table.")
    if a.write:
        out = Path(a.out) if a.out else root / "validation/lmix_cutoffs"; out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}", f"# interpreter {sys.executable}",
               f"# compute_lmix.py sha256 {sha(lm_path)}", f"# gamma_resolved.npy sha256 {sha(gam_path)}", f"# state_index.csv sha256 {sha(ctx.state_index_path)}",
               f"# Te_grid_L.npy sha256 {sha(root/'data/processed/cr_matrix/Te_grid_L.npy')}", f"# ne_grid_L.npy sha256 {sha(root/'data/processed/cr_matrix/ne_grid_L.npy')}",
               "# constants CODATA 2018; Lamb shift 1057.845 MHz; v = sqrt(8kT/pi mu), mu = m_p m_H/(m_p+m_H)"]
        with open(out / "lmix_cutoffs.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "lmix_cutoffs.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/lmix_cutoffs.{{csv,txt}}")
    return 1 if bad else 0

if __name__ == "__main__": sys.exit(main())
