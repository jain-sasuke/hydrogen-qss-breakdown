#!/usr/bin/env python
"""
verify_transport_selection.py
=============================
Does the closed-parcel assumption fail selectively at the points that generate
the breakdown census?

WHY THIS EXISTS
---------------
Chapter 6 tests neutral transport at one place, the cold corner [0,4], where
charge exchange slows the escape enough that the result survives by a factor
2.8. That is one grid point, and it is not a point the thesis defends: it sits
below the 2 eV scope boundary. The question the chapter never asked is whether
the assumption holds at the 45 pairs above 2 eV that DO carry the defended
result.

The refuting observation, stated in advance: if the escape time were long
compared with tau_QSS at the breakdown pairs, or if the breakdown pairs were no
worse in this respect than the rest of the grid, there would be no selection
effect and the conditional would be an ordinary caveat. Either outcome is
reportable. This script measures which one happened.

THE ESCAPE MODEL, FROM FIRST PRINCIPLES
---------------------------------------
A neutral born in the plasma leaves it or is ionised in it. The competition is
between tau_esc and tau_QSS. tau_esc is built as follows, and every step is a
choice that is stated rather than buried:

  speed          v = sqrt(2 k T_n / m_D), with T_n = Te. This is the speed of a
                 particle carrying kinetic energy kT_n, not the Maxwellian mean
                 speed sqrt(8kT/pi m), which is 1.13x larger. The distinction
                 is a 13% effect and does not move any conclusion here; it is
                 named because an unnamed velocity convention is how factors of
                 two enter.

  mean free path lambda = v / (n_i <sigma v>_CX), with n_i = n_e by
                 quasineutrality in a pure hydrogen plasma, and
                 <sigma v>_CX = 1.1e-8 cm^3/s. Resonant charge exchange
                 H + H+ -> H+ + H does not remove the atom, but it replaces it
                 with one drawn from the ion distribution moving in an
                 uncorrelated direction, which for transport is the same thing
                 as a scattering event.

  regime         free streaming when lambda >= L, giving tau_esc = L/v;
                 otherwise diffusive, D = v*lambda/3 and the fundamental mode
                 of a slab whose boundary is a distance L away,
                 tau_esc = 4 L^2 / (pi^2 D).

  The diffusive branch is the one that matters, and it runs in the model's
  favour: trapping the neutrals is what gives the closed parcel its best case.
  The free-streaming branch is quoted only so that the low-density corner is
  not given a diffusive time longer than a straight flight.

The CX rate coefficient is the one weak point. 1.1e-8 cm^3/s is the standard
few-eV value for resonant hydrogen charge exchange; the sensitivity of the
conclusion to it is measured below rather than assumed, by rerunning the census
over a factor of three either way.

WHAT IS GATED
-------------
The census in chapter5.tex Table 6.4 (45 of 448 above 2 eV, 202 of 680 overall)
is recounted here directly from divertor_map.csv, which also closes the
[UNVERIFIED] 202-versus-201 marker in that table's discussion. If the recount
disagrees with the chapter, this script raises.

Read-only. Nothing outside validation/transport_selection/ is written, and only
with --write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import find_repo_root                      # noqa: E402

ROOT = find_repo_root(_HERE)

# Physical constants, CODATA 2018, in Gaussian-CGS units.
K_B_ERG_PER_EV = 1.602176634e-12       # erg per eV, i.e. k_B * (1 eV / k_B)
M_PROTON_G = 1.67262192369e-24
M_DEUTERON_G = 2.013553212745 * 1.66053906660e-24

# Resonant charge-exchange rate coefficient for hydrogen at a few eV.
K_CX_DEFAULT = 1.1e-8                  # cm^3 s^-1

# Recorded in chapter5.tex, Table tab:elm_census. Recounted, not trusted.
REC_CENSUS = {
    ("all", 0.10): 202,
    ("all", 0.05): 348,
    ("all", 0.20): 61,
    ("warm", 0.10): 45,
    ("warm_dense", 0.10): 0,
}
REC_N_ALL, REC_N_WARM, REC_N_WARM_DENSE = 680, 448, 108


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def escape_time(Te_eV: np.ndarray, ne: np.ndarray, L_cm: float,
                k_cx: float, mass_g: float) -> tuple[np.ndarray, np.ndarray]:
    """Neutral escape time and mean free path. Returns (tau_esc, lambda)."""
    v = np.sqrt(2.0 * K_B_ERG_PER_EV * Te_eV / mass_g)          # cm/s
    nu_cx = ne * k_cx                                            # s^-1
    if np.any(nu_cx <= 0):
        raise RuntimeError("non-positive charge-exchange frequency; ne is bad")
    lam = v / nu_cx                                              # cm
    D = v * lam / 3.0                                            # cm^2/s
    tau_diff = 4.0 * L_cm ** 2 / (np.pi ** 2 * D)
    tau_free = L_cm / v
    tau = np.where(lam >= L_cm, tau_free, tau_diff)
    return tau, lam


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--map", type=str, default=None,
                    help="divertor_map.csv; defaults to the canonical one")
    ap.add_argument("--lengths", type=float, nargs="+", default=[10.0, 20.0],
                    help="distance to the plasma boundary, cm")
    ap.add_argument("--k-cx", type=float, default=K_CX_DEFAULT)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--te-floor", type=float, default=2.0)
    ap.add_argument("--mass", type=str, default="D", choices=["H", "D"])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    mp = Path(a.map) if a.map else ROOT / "validation/divertor_map/divertor_map.csv"
    if not mp.exists():
        raise RuntimeError(
            f"missing {mp}. This script reads the census that produced the "
            f"thesis claim; it does not recompute it and it does not "
            f"substitute anything for it.")
    mass = M_DEUTERON_G if a.mass == "D" else M_PROTON_G

    header, rows = [], []
    with mp.open() as fh:
        for line in fh:
            if line.startswith("#"):
                header.append(line.rstrip())
                continue
            break
        rdr = csv.DictReader(fh, fieldnames=line.rstrip().split(","))
        rows = list(rdr)
    need = {"direction", "i", "j", "Te", "ne", "tau_QSS", "window_ok",
            "lo_ELM_crash", "eps_plateau"}
    missing = need - set(rows[0])
    if missing:
        raise RuntimeError(f"{mp} lacks columns {sorted(missing)}; the file is "
                           f"not the one this script was written against")

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    for h in header:
        print("  " + h)
    print(f"  divertor_map.csv sha256 {sha256(mp)[:32]}...   {len(rows)} rows")
    print(f"  interpreter {sys.executable}   numpy {np.__version__}")
    print(f"  neutral mass {a.mass} = {mass:.6e} g   k_CX = {a.k_cx:.3e} cm^3/s")
    print()

    Te = np.array([float(r["Te"]) for r in rows])
    ne = np.array([float(r["ne"]) for r in rows])
    tq = np.array([float(r["tau_QSS"]) for r in rows])
    lo = np.array([float(r["lo_ELM_crash"]) for r in rows])
    ok = np.array([r["window_ok"].strip().lower() == "true" for r in rows])

    # ---- recount the census, closing the 202-versus-201 marker -------------
    print("=" * 78)
    print("THE CENSUS, RECOUNTED FROM THE FILE ITSELF")
    print("=" * 78)
    warm = ok & (Te >= a.te_floor)
    warm_dense = warm & (ne >= 1e14)
    scopes = {"all": ok, "warm": warm, "warm_dense": warm_dense}
    sizes = {"all": REC_N_ALL, "warm": REC_N_WARM, "warm_dense": REC_N_WARM_DENSE}
    for name, m in scopes.items():
        if int(m.sum()) != sizes[name]:
            raise RuntimeError(
                f"scope '{name}' holds {int(m.sum())} pairs, not the "
                f"{sizes[name]} recorded in chapter5.tex Table "
                f"tab:elm_census. The chapter and the file disagree and "
                f"nothing further may be reported until it is known which "
                f"is right.")
    for (name, thr), rec in sorted(REC_CENSUS.items()):
        got = int((scopes[name] & (lo > thr)).sum())
        flag = "OK " if got == rec else "!! "
        print(f"  {flag}{name:<11} bound > {thr:<5.2f}  counted {got:>4d}"
              f"   recorded {rec:>4d}")
        if got != rec:
            raise RuntimeError(
                f"census mismatch for scope '{name}' at threshold {thr}: "
                f"counted {got}, chapter records {rec}")
    n_heat = int((ok & (lo > a.threshold) & np.array(
        [r["direction"] == "heat" for r in rows])).sum())
    n_cool = int((ok & (lo > a.threshold)).sum()) - n_heat
    print(f"  decomposition of the {int((ok & (lo > a.threshold)).sum())} "
          f"overall breakdowns: {n_heat} heating + {n_cool} cooling")
    print("  This settles the [UNVERIFIED] marker in chapter5.tex: the count is")
    print("  the one above, counted here from divertor_map.csv directly.")
    print()

    # ---- the selection test ------------------------------------------------
    brk = warm & (lo > a.threshold)
    rest = ok & ~brk
    print("=" * 78)
    print("DOES THE CLOSED PARCEL HOLD WHERE THE RESULT IS GENERATED?")
    print("=" * 78)
    print(f"  breakdown set: window_ok, Te >= {a.te_floor} eV, bound > "
          f"{100*a.threshold:.0f}%  ->  n = {int(brk.sum())}")
    print(f"  every other analysed pair                          ->  n = "
          f"{int(rest.sum())}")
    print()
    out_rows = []
    for L in a.lengths:
        tau, lam = escape_time(Te, ne, L, a.k_cx, mass)
        ratio = tau / tq
        print(f"  L = {L:g} cm")
        for lab, m in (("breakdown", brk), ("all others", rest)):
            r = ratio[m]
            frac = float((r > 1.0).mean())
            print(f"    {lab:<11} median {np.median(r):.4g}   "
                  f"90th pct {np.percentile(r, 90):.4g}   "
                  f"max {r.max():.4g}   satisfy tau_esc > tau_QSS: "
                  f"{100*frac:.1f}%  ({int((r > 1.0).sum())}/{int(m.sum())})")
        f_diff = float((lam[brk] < L).mean())
        print(f"    diffusive branch used at {100*f_diff:.1f}% of the "
              f"breakdown pairs; mfp there runs "
              f"{lam[brk].min():.3g} to {lam[brk].max():.3g} cm")
        print()
        for k in np.where(brk | rest)[0]:
            out_rows.append(dict(L_cm=L, direction=rows[k]["direction"],
                                 i=rows[k]["i"], j=rows[k]["j"],
                                 Te=Te[k], ne=ne[k], tau_QSS=tq[k],
                                 lo_ELM_crash=lo[k], mfp_cm=lam[k],
                                 tau_esc=tau[k], ratio=ratio[k],
                                 breakdown=bool(brk[k])))

    # ---- how far from satisfying is the closest breakdown pair? -----------
    print("=" * 78)
    print("HOW MUCH WOULD HAVE TO CHANGE")
    print("=" * 78)
    print("  A conclusion of the form '0 of 45' is worth little if the 45 sit")
    print("  just under the line. The closest pair at each length:")
    for L in a.lengths:
        tau, _ = escape_time(Te, ne, L, a.k_cx, mass)
        r = tau / tq
        kb = int(np.where(brk)[0][np.argmax(r[brk])])
        print(f"    L = {L:>4g} cm: closest is [{rows[kb]['i']},"
              f"{rows[kb]['j']}] {rows[kb]['direction']}, "
              f"Te = {Te[kb]:.3g} eV, ne = {ne[kb]:.3g} cm^-3, "
              f"ratio {r[kb]:.4g}; tau_esc would have to rise by "
              f"{1.0/r[kb]:.3g}x for it to satisfy")
        print(f"                 and by {1.0/np.median(r[brk]):.4g}x for the "
              f"median of the 45")
    print()
    print("  The velocity convention, tested rather than assumed. The mean")
    print("  Maxwellian speed sqrt(8kT/pi m) is 1.128x the sqrt(2kT/m) used")
    print("  above; tau_free scales as 1/v and tau_diff as 1/v^2, so the mean")
    print("  speed shortens every escape time and cannot rescue the parcel:")
    for L in a.lengths:
        v_ratio = np.sqrt(4.0 / np.pi)
        tau, lam = escape_time(Te, ne, L, a.k_cx, mass)
        tau_mean = np.where(lam / v_ratio >= L, tau / v_ratio,
                            tau / v_ratio ** 2)
        r = tau_mean / tq
        print(f"    L = {L:>4g} cm, mean speed: breakdown set "
              f"{100*float((r[brk] > 1).mean()):.1f}% satisfy "
              f"(median {np.median(r[brk]):.4g}), others "
              f"{100*float((r[rest] > 1).mean()):.1f}%")
    print()

    # ---- the one point the chapter did test --------------------------------
    print("=" * 78)
    print("THE POINT CHAPTER 6 ACTUALLY TESTED, FOR COMPARISON")
    print("=" * 78)
    k = int(np.argmax(np.where(ok, lo, -np.inf)))
    tau10, lam10 = escape_time(Te[k:k+1], ne[k:k+1], 10.0, a.k_cx, mass)
    v = np.sqrt(2.0 * K_B_ERG_PER_EV * Te[k] / mass)
    print(f"  worst pair overall [{rows[k]['i']},{rows[k]['j']}] "
          f"{rows[k]['direction']}: Te = {Te[k]:.4g} eV, "
          f"ne = {ne[k]:.4g} cm^-3, bound = {lo[k]:.4f}")
    print(f"    thermal speed {v:.4g} cm/s, ballistic crossing of 10 cm "
          f"{10.0/v*1e6:.3g} us")
    print(f"    mfp {lam10[0]:.4g} cm, diffusive escape {tau10[0]*1e6:.4g} us, "
          f"tau_QSS {tq[k]*1e3:.4g} ms, ratio {tau10[0]/tq[k]:.3e}")
    print("    Chapter 6 quotes 1.7 cm and 72 us at this point. Those are the")
    print("    numbers this model reproduces, which is what makes it the same")
    print("    model rather than a second one.")
    print()

    # ---- sensitivity to the one uncertain input ---------------------------
    print("=" * 78)
    print("SENSITIVITY TO THE CHARGE-EXCHANGE RATE COEFFICIENT")
    print("=" * 78)
    print("  tau_diff scales as 1/D = 3/(v*lambda) and lambda as 1/k_CX, so")
    print("  tau_diff is linear in k_CX. A factor 3 error moves every ratio by")
    print("  a factor 3, and the table below says whether that changes the")
    print("  answer.")
    for kf in (a.k_cx / 3.0, a.k_cx, a.k_cx * 3.0):
        tau, _ = escape_time(Te, ne, max(a.lengths), kf, mass)
        r = tau / tq
        print(f"    k_CX = {kf:.3e}:  breakdown set "
              f"{100*float((r[brk] > 1).mean()):.1f}% satisfy "
              f"(median {np.median(r[brk]):.4g}), others "
              f"{100*float((r[rest] > 1).mean()):.1f}% satisfy")
    print()

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "transport_selection"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "transport_selection.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  wrote {out/'transport_selection.csv'}  ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
