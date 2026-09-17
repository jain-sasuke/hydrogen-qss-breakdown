#!/usr/bin/env python
"""
verify_rr_limits.py
===================
The temperature limits of the radiative-recombination coefficient the model
uses, taken from the model's own function rather than asserted.

WHY THIS EXISTS
---------------
Chapter 2 said Eq. alpha_RR "gives alpha ~ Te^{-1/2} at high energy and
drives alpha -> 0 at low energy, so the maximum lies below 1 eV". Both halves
are backwards, and the sentence contradicts the preceding "decreases
monotonically ... with no interior maximum". From the equation,
    alpha = D y^{3/2} e^y [g0 E1(y) + g1 E2(y) + g2 E3(y)],  y = I_n / kT,
e^y E_k(y) -> 1/y as y -> inf, so alpha -> D (g0+g1+g2) y^{1/2}  ~ T^{-1/2}
at LOW temperature; and e^y E1(y) -> -gamma - ln y as y -> 0, so
alpha ~ y^{3/2} ln(1/y) ~ T^{-3/2} ln(T/I_n) -> 0 at HIGH temperature.

METHOD
------
Import src/rates/recombination_rates.py (module-level code defines
functions and a grid only) and evaluate alpha_RR_shell(n, T) for n = 2..8 at
fixed y = I_n/kT from 340 to 3.4e-4, so every shell is tested at the same
distance into each limit. Measure the local logarithmic slope d ln alpha / d ln T
at the two ends, check monotonicity, and read g0+g1+g2 from the module.

PREDICTIONS (written before the run)
-----------------------------------
P1  g0+g1+g2 > 0 for n = 2..8 (0.876, 0.908, 0.925 at n = 2, 3, 4 by hand)
P2  low-T slope -> -0.500 (within 0.01 at y = I_n/kT = 340) for every n
P3  high-T slope -> -1.5 with a logarithmic correction: between -1.5 and -1.3 at y = 3.4e-4
P4  alpha decreases monotonically on the whole range for every n; no maximum
REFUTER: a positive slope anywhere, or a low-T slope not equal to -1/2.

OUTPUTS (with --write): validation/rr_limits/rr_limits.{csv,txt}
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]; src = root / "src/rates/recombination_rates.py"
    spec = importlib.util.spec_from_file_location("rr", src); rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("RADIATIVE-RECOMBINATION LIMITS -- from the model's own alpha_RR_shell")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}"); say(f"interpreter {sys.executable}"); say(f"module {src.relative_to(root)}  sha256 {sha(src)[:16]}"); say("=" * 78)
    # The module evaluates y^{3/2} e^y [...] literally and overflows for y = I_n/kT >~ 700,
    # i.e. below T ~ I_n/700 (5e-3 eV at n=2); the limit test therefore starts at 1e-2 eV,
    # where y(n=2) = 340 and the 1/y corrections to the -1/2 slope are ~0.3 %.
    # The limits are limits in y = I_n/kT, not in T, so the two ends are placed at the
    # same y for every shell: y = 340 (T = I_n/340; the 1/y corrections to the -1/2
    # slope are then ~0.3 %) and y = 3.4e-4. The module evaluates y^{3/2} e^y [...]
    # literally and overflows for y >~ 700, which caps the low-T end.
    Y = np.logspace(np.log10(340.0), np.log10(3.4e-4), 601); rows = []; bad = []
    say(f"  {'n':>2} {'g0+g1+g2':>9} {'slope@y=340':>12} {'slope@y=3e-4':>13} {'monotone':>9} {'alpha(1eV)':>12} {'alpha(10eV)':>12}")
    for n in range(2, 9):
        gs = rr._g0(n) + rr._g1(n) + rr._g2(n)
        # the module's own hydrogen ionisation energy, found by value so the name cannot drift
        ry = {k: v for k, v in vars(rr).items() if isinstance(v, float) and 13.5 < v < 13.7}
        if len(ry) != 1: raise KeyError(f"expected one Rydberg-energy constant in the module, found {ry}")
        In = next(iter(ry.values())) / n**2
        T = In / Y
        al = np.asarray(rr.alpha_RR_shell(n, T), float)
        if not np.all(np.isfinite(al)) or np.any(al <= 0): raise ValueError(f"alpha_RR_shell({n}) not positive-finite on the range")
        s = np.gradient(np.log(al), np.log(T)); mono = bool(np.all(np.diff(al) < 0))
        Tg = np.logspace(-1, 2, 301); ag = np.asarray(rr.alpha_RR_shell(n, Tg), float)
        rows.append(dict(n=n, I_n_eV=In, g_sum=gs, T_low_eV=T[0], slope_lowT=s[0], T_high_eV=T[-1], slope_highT=s[-1], monotone=mono,
                         alpha_1eV=float(np.interp(1.0, Tg, ag)), alpha_10eV=float(np.interp(10.0, Tg, ag))))
        say(f"  {n:>2} {gs:9.4f} {s[0]:12.4f} {s[-1]:13.4f} {str(mono):>9} {rows[-1]['alpha_1eV']:12.4e} {rows[-1]['alpha_10eV']:12.4e}")
        if gs <= 0: bad.append(f"P1 n={n} g-sum {gs}")
        if abs(s[0] + 0.5) > 0.01: bad.append(f"P2 n={n} low-T slope {s[0]}")
        if not (-1.5 <= s[-1] <= -1.3): bad.append(f"P3 n={n} high-T slope {s[-1]}")
        if not mono: bad.append(f"P4 n={n} not monotone")
    say("\nVERDICT: " + ("all predictions reproduced: alpha ~ T^-1/2 at LOW T with a positive coefficient, ~ T^-3/2 ln T at HIGH T, monotone, no maximum" if not bad else "REFUTED: " + "; ".join(bad)))
    if bad: raise AssertionError(bad)
    if a.write:
        out = Path(a.out) if a.out else root / "validation/rr_limits"; out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}", f"# interpreter {sys.executable}", f"# recombination_rates.py sha256 {sha(src)}", "# y = I_n/kT from 340 to 3.4e-4, 601 log points per shell; slopes by np.gradient of ln alpha vs ln T"]
        with open(out / "rr_limits.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "rr_limits.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/rr_limits.{{csv,txt}}")
    return 0

if __name__ == "__main__": sys.exit(main())
