"""
How far up in n is the Anderson (2002) RMPS benchmark actually usable?
======================================================================

Two separate questions:

  (1) COVERAGE  -- what does the table contain?  Trivially answered by reading
      it: levels 1-15, i.e. n <= 5, l-resolved. Nothing above n=5 exists, so
      CCC data for n >= 6 is simply unvalidated by this benchmark.

  (2) RELIABILITY -- inside that coverage, up to which n do CCC and RMPS
      actually agree, and when they disagree, which one looks anomalous?

For (2) this script uses a discriminator Anderson cannot provide for itself:
CCC runs to n=8, so the CCC dipole series can be continued past n=5 and tested
for smoothness against the exact hydrogenic asymptote.

For a Rydberg series of dipole transitions n_lo -> n_up, the oscillator
strength obeys f(n_lo -> n_up) ~ C / n_up^3 as n_up -> infinity (exact for
hydrogen). At fixed Te well above threshold the rate coefficient is dominated
by the Bethe term, K ~ f, so the reduced quantity

        R(n_up) = K(n_lo -> n_up) * n_up^3

should approach a constant. A dataset whose R(n_up) is smooth through the
series is internally consistent; one that kinks at a single n is the anomalous
one at that n. This does not prove which dataset is right in absolute terms --
it localizes where one of them stops behaving like hydrogen.

Run:  python src/validation/anderson_validity_range.py
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys_path_note = "shares constants/helpers with anderson2002_benchmark.py"

import importlib.util
spec = importlib.util.spec_from_file_location(
    "and02bm", os.path.join(HERE, "anderson2002_benchmark.py"))
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)      # module-level code only defines things

CCC = pd.read_csv(bm.CCC_PATH)
GRP = CCC.groupby(["n_i", "l_i", "n_f", "l_f"])
KEYS = set(GRP.groups.keys())
AND = bm.parse_anderson2002(bm.PDF_PATH)

NL_TO_IDX = {v: k for k, v in bm.IDX_TO_NL.items()}
TES = [1.0, 3.0, 5.0, 10.0]


def K_ccc(n_lo, l_lo, n_up, l_up, Te):
    key = (n_lo, l_lo, n_up, l_up)
    if key not in KEYS:
        return np.nan
    s = GRP.get_group(key).sort_values("E_eV")
    dE = bm.threshold_eV(n_lo, n_up)
    E = np.linspace(dE + 1e-4, s.E_eV.max(), 5000)
    return bm.K_maxwell(np.interp(E, s.E_eV.values, s.sigma_a0sq.values,
                                  left=0.0, right=0.0), E, Te)


def K_and(n_lo, l_lo, n_up, l_up, Te):
    key = (NL_TO_IDX.get((n_up, l_up)), NL_TO_IDX.get((n_lo, l_lo)))
    if None in key:
        return np.nan
    row = AND[(AND.i_upper == key[0]) & (AND.j_lower == key[1])]
    if len(row) == 0:
        return np.nan
    return bm.K_from_upsilon(row[f"ups_{Te:g}eV"].iloc[0], n_lo, l_lo, n_up, Te)


print("=" * 78)
print("ANDERSON (2002) VALIDITY RANGE")
print("=" * 78)

# ── (1) Coverage ──────────────────────────────────────────────────────────────
ns = sorted({bm.IDX_TO_NL[i][0] for i in bm.IDX_TO_NL})
print(f"\n(1) COVERAGE of the corrigendum table")
print(f"    levels          : {len(bm.IDX_TO_NL)}  (l-resolved)")
print(f"    principal qnum n: {min(ns)} .. {max(ns)}")
print(f"    transitions     : {len(AND)}  (all n != n' pairs among those levels)")
print(f"    highest l       : {bm.L_CHAR[max(bm.IDX_TO_NL[i][1] for i in bm.IDX_TO_NL)]}"
      f"  (5g)")
ccc_nmax = int(max(CCC.n_i.max(), CCC.n_f.max()))
print(f"    CCC reaches n={ccc_nmax}; transitions with n_upper >= 6 have NO Anderson")
print(f"    counterpart and are therefore unvalidated by this benchmark.")

# ── (2) Dipole-series smoothness ─────────────────────────────────────────────
print(f"\n(2) DIPOLE-SERIES TEST:  R(n_up) = K * n_up^3  should flatten with n_up")

series = [("1s->np", 1, 0, 1), ("2p->nd", 2, 1, 2), ("2s->np", 2, 0, 1),
          ("3d->nf", 3, 2, 3), ("4s->np", 4, 0, 1)]

out = []
for name, n_lo, l_lo, l_up in series:
    print(f"\n  --- {name} ---")
    n_ups = [n for n in range(max(n_lo + 1, l_up + 1), ccc_nmax + 1)]
    for Te in (1.0, 10.0):
        rc, ra = [], []
        for n_up in n_ups:
            kc = K_ccc(n_lo, l_lo, n_up, l_up, Te)
            ka = K_and(n_lo, l_lo, n_up, l_up, Te)
            rc.append(kc * n_up**3 if np.isfinite(kc) else np.nan)
            ra.append(ka * n_up**3 if np.isfinite(ka) else np.nan)
            out.append({"series": name, "Te_eV": Te, "n_upper": n_up,
                        "K_CCC": kc, "K_And": ka,
                        "R_CCC": rc[-1], "R_And": ra[-1]})
        hdr = "  ".join(f"n={n:<11d}" for n in n_ups)
        print(f"    Te={Te:4.1f} eV   {hdr}")
        print("      R_CCC      " +
              "  ".join(f"{v:<13.4g}" if np.isfinite(v) else f"{'--':<13}" for v in rc))
        print("      R_Anderson " +
              "  ".join(f"{v:<13.4g}" if np.isfinite(v) else f"{'--':<13}" for v in ra))
        # step-to-step change of R, as a smoothness measure
        def steps(r):
            r = np.asarray(r, float)
            s = []
            for a, b in zip(r[:-1], r[1:]):
                s.append((b / a - 1) * 100 if np.isfinite(a) and np.isfinite(b)
                         and a > 0 else np.nan)
            return s
        sc, sa = steps(rc), steps(ra)
        lbl = ["  ".join(f"{n_ups[i]}->{n_ups[i+1]}" for i in range(len(n_ups) - 1))]
        print("      ΔR_CCC %   " +
              "  ".join(f"{v:<13.1f}" if np.isfinite(v) else f"{'--':<13}" for v in sc))
        print("      ΔR_And %   " +
              "  ".join(f"{v:<13.1f}" if np.isfinite(v) else f"{'--':<13}" for v in sa))

df = pd.DataFrame(out)
p = os.path.join(ROOT, "data", "processed", "collisions",
                 "anderson_validity_dipole_series.csv")
df.to_csv(p, index=False)
print(f"\n  wrote {os.path.relpath(p, ROOT)}")

# ── (3) Where does agreement break, as a function of n_upper? ────────────────
print(f"\n(3) AGREEMENT vs n_upper  (all 85 matched transitions, Te=1,3,5,10 eV)")
b = pd.read_csv(os.path.join(ROOT, "data", "processed", "collisions",
                             "ccc_vs_anderson2002_benchmark.csv"))
print(f"    {'n_up':>5} {'n_pts':>6} {'within20%':>10} {'mean|err|':>10} "
      f"{'median signed':>14}")
for n_up in sorted(b.n_upper.unique()):
    s = b[b.n_upper == n_up]
    print(f"    {n_up:>5} {len(s):>6} {(s.pct_err.abs()<20).mean()*100:>9.1f}% "
          f"{s.pct_err.abs().mean():>9.2f}% {s.pct_err.median():>13.2f}%")
print(f"\n    Same, split by whether the UPPER state is the diffuse 5g/5f:")
for sel, nm in [((b.n_upper == 5) & (b.l_upper >= 3), "n=5, l>=3 (5f,5g)"),
                ((b.n_upper == 5) & (b.l_upper < 3), "n=5, l<=2 (5s,5p,5d)")]:
    s = b[sel]
    print(f"      {nm:22s} n={len(s):3d}  mean|err|={s.pct_err.abs().mean():6.2f}%"
          f"  median signed={s.pct_err.median():+7.2f}%")
print("\nDone.")
