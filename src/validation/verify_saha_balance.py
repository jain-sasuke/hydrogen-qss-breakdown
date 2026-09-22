#!/usr/bin/env python
"""
verify_saha_balance.py
======================
Ionisation against three-body recombination through the Saha relation, at every
level and every temperature, with independent constants.

WHY THIS EXISTS
---------------
Chapter 4 (sec:one_ladder, and Table 4.4 row "One energy ladder on both sides")
says: "Testing ionisation against three-body recombination through the Saha
relation at every level gives a ratio of 0.999997 for every one of the 43
states, at every temperature, to the same eight digits", and cites a working
note (findings_10 ADDENDUM D.6) with no producing script. The point of the
sentence is not the closeness to unity, which detailed balance imposes by
construction (recombination_rates.alpha_3BR_from_Kion builds alpha_3BR from
K_ion), but the IDENTITY of the residual across states and temperatures: a
second energy table, or one indexed differently from the other, would leave a
state- or temperature-dependent residual of order exp(dI/kT). This script
stamps the number.

METHOD
------
  ratio[k, t] = alpha_3BR[k, t] / ( K_ion[k, t] * (g_k/2) * Lambda(T_t)^3 * exp(I_k/kT_t) )
  Lambda^3 = (h^2 / (2 pi m_e k T))^(3/2) in cm^3, evaluated with CODATA 2018
  constants (h = 6.62607015e-34 J s exact, m_e = 9.1093837015e-31 kg,
  eV = 1.602176634e-19 J exact), NOT the module's own rounded constants
  (h = 6.62607e-34, m_e = 9.10938e-31, eV = 1.60218e-19), so the residual
  measures the module's constant set against the accepted values.
  g_k and I_k from state_index.csv; alpha_3BR from the stored resolved and
  bundled arrays; K_ion_final.npy; the Te grid from the L grid.
  The energy ladder: the collisional detailed balance in the CCC tables uses
  dE = IH (1/n_i^2 - 1/n_f^2); the Saha side uses I_k = IH/n^2 from
  state_index. Both are checked against one another to 1e-9 eV.

GATES
-----
G0  shapes agree (43 x 50); state_index has g and I_eV for all 43 states.
G1  recomputing the module's own factor with ITS constants reproduces
    alpha_3BR/K_ion to 1e-12 (names what the stored arrays are).
G2  the CCC thresholds dE_eV in K_CCC_metadata.csv equal I_eV[n_i] - I_eV[n_f]
    from state_index.csv to 1e-6 eV for every row (one energy ladder).

PREDICTIONS (written before the run)
-----------------------------------
P1  ratio = 0.999997 +- 0.000002 (the working note's value) at every one of
    the 43 x 50 entries.
P2  the spread max - min over all 2150 entries is below 1e-8 (identical to
    eight digits), i.e. the residual is the constant set and nothing else.
P3  the residual is explained by the constants: (h_mod/h)^3 (m_e/m_e_mod)^(3/2)
    (eV/eV_mod)^(3/2) reproduces the ratio to 1e-9.
REFUTER of chapter 4's sentence: any entry differing from the median ratio by
more than 1e-6, or a ratio outside 0.999997 +- 0.000002.

OUTPUT (with --write): validation/saha_balance/saha_balance.{csv,txt}
"""
from __future__ import annotations
import argparse, hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)
# CODATA 2018
H_SI, ME_SI, EV_J = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19
# recombination_rates.py lines 84-88 (the module's constants), copied for P3 only
H_MOD, ME_MOD, EV_MOD = 6.62607e-34, 9.10938e-31, 1.60218e-19


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); ctx.validate(); te = ctx.te_grid; nS = ctx.n_states
    P = lambda *p: ROOT.joinpath(*p)
    paths = dict(state_index=ctx.state_index_path, K_ion_final=P("data/processed/collisions/tics/K_ion_final.npy"),
                 alpha_3BR_res=P("data/processed/recombination/alpha_3BR_resolved.npy"), alpha_3BR_bund=P("data/processed/recombination/alpha_3BR_bundled.npy"),
                 K_CCC_meta=P("data/processed/collisions/ccc/K_CCC_metadata.csv"), Te_grid_L=P("data/processed/cr_matrix/Te_grid_L.npy"),
                 recombination_rates=P("src/rates/recombination_rates.py"))
    for q in paths.values():
        if not q.is_file(): raise FileNotFoundError(f"required input missing: {q}")
    si = pd.read_csv(paths["state_index"]); K_ion = np.load(paths["K_ion_final"])
    a3 = np.vstack([np.load(paths["alpha_3BR_res"]), np.load(paths["alpha_3BR_bund"])]); meta = pd.read_csv(paths["K_CCC_meta"])
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 90); say("SAHA BALANCE: alpha_3BR against K_ion at every level and temperature, CODATA constants"); say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}"); say(ctx.describe()); say("=" * 90)
    g = si.g.values.astype(float); I_eV = si.I_eV.values.astype(float); nv = si.n.values.astype(int); lv = si.l.values.astype(int)
    if K_ion.shape != (nS, len(te)) or a3.shape != K_ion.shape or len(si) != nS: raise RuntimeError(f"Gate G0: shapes K_ion {K_ion.shape}, alpha_3BR {a3.shape}, states {len(si)}")
    if not np.array_equal(np.where(lv >= 0)[0], np.arange(int((lv >= 0).sum()))): raise RuntimeError("Gate G0: resolved states are not the leading block; the vstack order would mislabel")
    say(f"\n  G0  shapes ({nS} x {len(te)}) agree; g from 2 to {g.max():.0f}, I_eV from {I_eV.min():.5f} to {I_eV.max():.4f} eV")
    # state_index.csv stores I_eV rounded to 8 decimals (assemble_K_exc.build_state_index); the module uses IH/n^2 unrounded.
    # The Saha factor is built from IH/n^2 with IH = the 1s entry, as the module does; the rounding effect is reported, not absorbed.
    IH = float(I_eV[nv == 1][0]); I_n = IH / nv.astype(float) ** 2
    if np.abs(I_n - I_eV).max() > 1e-8: raise RuntimeError("state_index I_eV is not IH/n^2 to its own printed precision")
    lam3_mod = (H_MOD**2 / (2 * np.pi * ME_MOD * te * EV_MOD)) ** 1.5 * 1e6                      # cm^3, the module's constants
    fac_mod = (g[:, None] / 2.0) * lam3_mod[None, :] * np.exp(I_n[:, None] / te[None, :])
    d1 = np.abs(a3 / (K_ion * fac_mod) - 1).max()
    d1r = np.abs(np.exp((I_eV - I_n)[:, None] / te[None, :]) - 1).max()
    say(f"  G1  alpha_3BR / (K_ion x module-constant Saha factor) = 1 to {d1:.1e} at all {a3.size} entries (recombination_rates.alpha_3BR_from_Kion reproduced; the 8-decimal rounding of I_eV in state_index.csv would contribute {d1r:.1e} through exp(I/kT))")
    if d1 > 1e-12: raise RuntimeError("Gate G1")
    byn = {(int(n), int(l)): k for k, (n, l) in enumerate(zip(nv, lv))}
    In = {int(n): float(I_eV[nv == n][0]) for n in np.unique(nv)}
    d2 = max(abs(float(r.dE_eV) - (In[int(r.n_i)] - In[int(r.n_f)])) for r in meta.itertuples(index=False))
    say(f"  G2  CCC thresholds dE_eV vs I_eV[n_i] - I_eV[n_f] from state_index over {len(meta)} rows: max |diff| {d2:.1e} eV (one energy ladder)")
    if d2 > 1e-6: raise RuntimeError("Gate G2")
    say("\n  ALL GATES PASSED.")

    lam3 = (H_SI**2 / (2 * np.pi * ME_SI * te * EV_J)) ** 1.5 * 1e6
    fac = (g[:, None] / 2.0) * lam3[None, :] * np.exp(I_n[:, None] / te[None, :])
    ratio = a3 / (K_ion * fac)
    med = float(np.median(ratio)); spread = float(ratio.max() - ratio.min()); dev = np.abs(ratio - med).max()
    expl = (H_MOD / H_SI) ** 3 * (ME_SI / ME_MOD) ** 1.5 * (EV_J / EV_MOD) ** 1.5
    say("\n" + "=" * 90); say("RESULT"); say("=" * 90)
    say(f"  ratio alpha_3BR / (K_ion x Saha factor, CODATA 2018): median {med:.9f}, min {ratio.min():.9f}, max {ratio.max():.9f}, spread {spread:.2e}, max |entry - median| {dev:.2e}")
    say(f"  by state (n = 1..15, all Te): " + " ".join(f"{np.median(ratio[nv == n]):.7f}" for n in np.unique(nv)))
    say(f"  by temperature (Te index 0, 23, 49, all states): " + ", ".join(f"{np.median(ratio[:, t]):.9f}" for t in (0, 23, len(te) - 1)))
    say(f"  constant-set explanation (h_mod/h)^3 (m_e/m_e,mod)^1.5 (eV/eV_mod)^1.5 = {expl:.9f}; ratio/explanation - 1 = {med / expl - 1:.1e}")
    say(f"  the module's constants: h {H_MOD:.6e} (CODATA {H_SI:.8e}), m_e {ME_MOD:.6e} ({ME_SI:.10e}), eV {EV_MOD:.6e} ({EV_J:.9e})")
    say("\n" + "=" * 90); say("PREDICTIONS (written before the run) against what came out"); say("=" * 90)
    missed = []
    p1 = np.all(np.abs(ratio - 0.999997) <= 2e-6); say(f"  P1 every entry within 0.999997 +- 0.000002: {'reproduced' if p1 else 'NOT reproduced'} (range {ratio.min():.7f} to {ratio.max():.7f})"); missed += [] if p1 else ["P1"]
    p2 = spread < 1e-8; say(f"  P2 spread over {ratio.size} entries below 1e-8: {'reproduced' if p2 else 'NOT reproduced'} ({spread:.2e})"); missed += [] if p2 else ["P2"]
    p3 = abs(med / expl - 1) < 1e-9; say(f"  P3 residual explained by the constant set to 1e-9: {'reproduced' if p3 else 'NOT reproduced'} ({med / expl - 1:.1e})"); missed += [] if p3 else ["P3"]
    say("\nPREDICTIONS: " + ("P1-P3 all reproduced" if not missed else "not reproduced as written: " + ", ".join(missed)))
    ref = dev > 1e-6 or not p1
    say(f"REFUTER (an entry off the median by > 1e-6, or a ratio outside 0.999997 +- 2e-6): {'APPEARED' if ref else 'did not appear'}")
    say("\nREADING. The closeness to unity is detailed balance by construction (G1). What the sentence in chapter 4 rests on is the\n"
        "  identity of the residual across the 43 states and 50 temperatures: a second energy table, or one indexed differently, would\n"
        "  leave a state- or temperature-dependent residual of order exp(dI/kT); the measured spread bounds any such mismatch. The residual\n"
        "  itself is the module's rounded constants against CODATA 2018 (P3), 3e-6, and is immaterial; it is reported, not repaired.")

    if a.write:
        out = Path(a.out) if a.out else P("validation/saha_balance"); out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}"]
        hdr += [f"# {q.relative_to(ROOT) if q.is_relative_to(ROOT) else q} sha256 {sha(q)}" for q in paths.values()]
        hdr += [f"# CODATA 2018: h {H_SI!r}, m_e {ME_SI!r}, eV {EV_J!r}; module constants h {H_MOD!r}, m_e {ME_MOD!r}, eV {EV_MOD!r}"]
        K, T = np.meshgrid(np.arange(nS), np.arange(len(te)), indexing="ij")
        df = pd.DataFrame(dict(state=K.ravel(), label=np.array(ctx.labels)[K.ravel()], n=nv[K.ravel()], t=T.ravel(), Te_eV=te[T.ravel()], ratio=ratio.ravel(), ratio_module_constants=(a3 / (K_ion * fac_mod)).ravel()))
        with open(out / "saha_balance.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "saha_balance.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/saha_balance.{{csv,txt}}")
    return 0


if __name__ == "__main__": sys.exit(main())
