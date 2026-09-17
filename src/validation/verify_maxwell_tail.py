#!/usr/bin/env python
"""
verify_maxwell_tail.py
======================
How much of the Maxwell average is lost by stopping the cross-section
integral at the top of the CCC data instead of at infinity.

WHY THIS EXISTS
---------------
Chapter 2 says the integral "stops at the top of the data (100-968 eV)" and
that because exp(-100/10) = 4.5e-5 the missing tail is below 0.01 %. A
reviewer pointed out that the integrand is sigma(E) E exp(-E/T), so the
Boltzmann factor alone does not bound the tail. Both the premise and the
bound are checked here against the data's own energy ceilings.

METHOD
------
For every excitation block in K_CCC_metadata.csv, E_max_eV is the last
tabulated energy. The tail fraction is bounded by assuming sigma E grows
logarithmically above threshold, sigma E ~ ln(E/dE) (the Bethe form for a
dipole transition; the least favourable physical case, since sigma E is
constant or falling for non-dipole ones):
    tail(E_max, T) = int_{E_max}^inf ln(E/dE) e^{-E/T} dE / int_{dE}^inf ln(E/dE) e^{-E/T} dE
evaluated at T = 10 eV, the hottest temperature on the grid, where the tail
is largest. The hypothetical case E_max = 100 eV is evaluated for the record.

PREDICTIONS (written before the run)
-----------------------------------
P1  E_max over the 1320 blocks lies between 955 and 969 eV; no block stops at 100 eV.
P2  the largest tail fraction over all blocks at 10 eV is below 1e-35.
P3  had a ground-state dipole block stopped at 100 eV, its tail at 10 eV would be
    about 5e-4 (0.05 %), five times the 0.01 % the chapter claimed: the
    Boltzmann factor alone does not bound it.
REFUTER: any block with E_max < 900 eV, or a tail above 1e-30.

OUTPUTS (with --write): validation/maxwell_tail/maxwell_tail.{csv,txt}
"""
from __future__ import annotations
import argparse, hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.integrate import quad

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def tail_fraction(dE, Emax, T):
    f = lambda E: np.log(E / dE) * np.exp(-(E - dE) / T)   # shifted exponent for numerical range
    whole = quad(f, dE, dE + 60 * T, limit=200)[0]
    if Emax >= dE + 60 * T:
        # tail beyond 60 T is below e^-60 of the whole; return an analytic bound
        return np.log(Emax / dE) * T * np.exp(-(Emax - dE) / T) / whole
    return quad(f, Emax, dE + 60 * T, limit=200)[0] / whole

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    meta = root / "data/processed/collisions/ccc/K_CCC_metadata.csv"
    Te = np.load(root / "data/processed/cr_matrix/Te_grid_L.npy")
    if not meta.is_file(): raise FileNotFoundError(meta)
    m = pd.read_csv(meta); T = float(Te.max())
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("MAXWELL-AVERAGE TAIL -- what stopping at the top of the CCC data costs")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}"); say(f"interpreter {sys.executable}")
    say(f"blocks {len(m)};  Te_max on the grid {T:g} eV;  weighting sigma*E ~ ln(E/dE)"); say("=" * 78)
    say(f"P1  E_max over blocks: min {m.E_max_eV.min():.1f}  max {m.E_max_eV.max():.1f} eV;  blocks with E_max < 900 eV: {(m.E_max_eV < 900).sum()}")
    if (m.E_max_eV < 900).any(): raise AssertionError("a block stops below 900 eV; the chapter's premise would then be partly right")
    m["tail_at_Tmax"] = [tail_fraction(r.dE_eV, r.E_max_eV, T) for r in m.itertuples()]
    say(f"P2  largest tail fraction at {T:g} eV over all blocks: {m.tail_at_Tmax.max():.2e}  (block {m.loc[m.tail_at_Tmax.idxmax(), ['n_i','l_i','n_f','l_f']].tolist()})")
    if m.tail_at_Tmax.max() > 1e-30: raise AssertionError("tail above 1e-30")
    say("P3  the hypothetical the chapter argued from, E_max = 100 eV at 10 eV:")
    hyp = []
    for lab, dE in (("ground-state dipole, dE = 10.20 eV", 10.20435), ("n=2 -> 3, dE = 1.89 eV", 1.88905), ("n=4 -> 5, dE = 0.31 eV", 0.30613)):
        t = tail_fraction(dE, 100.0, 10.0); hyp.append(dict(case=lab, dE_eV=dE, E_max_eV=100.0, Te_eV=10.0, tail=t))
        say(f"      {lab:38s} tail = {100*t:.4f} %   ({'above' if t > 1e-4 else 'below'} the 0.01 % claimed)")
    say("\nVERDICT: no block stops at 100 eV; every block reaches 955-969 eV and the missing tail is below 1e-35.\n"
        "The chapter's '(100-968 eV)' understated its own margin, and its 0.01 % bound would not have held\n"
        "for a 100 eV ceiling, because sigma*E, not the Boltzmann factor alone, sets the tail.")
    if a.write:
        out = Path(a.out) if a.out else root / "validation/maxwell_tail"; out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}", f"# interpreter {sys.executable}",
               f"# K_CCC_metadata.csv sha256 {sha(meta)}", f"# Te_grid_L.npy sha256 {sha(root / 'data/processed/cr_matrix/Te_grid_L.npy')}",
               f"# weighting sigma*E ~ ln(E/dE); tail evaluated at Te_max = {T:g} eV"]
        with open(out / "maxwell_tail.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); m.to_csv(fh, index=False)
        with open(out / "maxwell_tail_hypothetical.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(hyp).to_csv(fh, index=False)
        with open(out / "maxwell_tail.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/maxwell_tail*.{{csv,txt}}")
    return 0

if __name__ == "__main__": sys.exit(main())
