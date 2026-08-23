"""
verify_plateau_gridmap.py
=========================
Maps the analytic plateau error and its two-channel origin across the full
(Te, ne) grid. Companion to verify_plateau_slowmode.py, which does the same
physics at two points with full time integration.

NO TIME INTEGRATION IS NEEDED. verify_plateau_slowmode.py established that the
plateau state is exactly the partial-equilibrium solve, to 6 digits at the cold
corner and to 0.005% at the ITER reference once window sampling is accounted
for. So each grid point costs three linear solves, not a stiff ODE.

THE ONE-PARAMETER FAMILY
------------------------
Under the NEW operator, with the ground density scaled by x relative to its
old value, the excited block is exactly

    n_E(x) = n0 + x * n1
      n0 = -L_EE^{-1} S_E             recombination-fed channel
      n1 = -L_EE^{-1} L_Eg n_g^old    ground-fed channel

    x = 1                    -> partial equilibrium  (the plateau)
    x = n_g^new / n_g^old    -> the new QSS target

Superposition verified to 1.5e-15 at both test points. This is Fujimoto
eq. (4.20)'s r_0 / r_1 split realised in the model's own matrix.

With f_p(x) = ground-fed fraction of shell p,

    dlnR/dlnx = f3(x) - f4(x),      R = N3/N4

so the observable's sensitivity to the ground reservoir is the DIFFERENCE in
ground-fed fraction between the two shells. It vanishes in both Fujimoto
limits -- fully ionizing (f3 = f4 = 1) and fully recombining (f3 = f4 = 0) --
and is largest where the two supply channels compete.

PREDICTION UNDER TEST (student's, from the Fujimoto algebra, before this run)
----------------------------------------------------------------------------
The plateau error and the amplification eps_plateau/eps_step are ~zero in both
asymptotic limits and maximal in between, tracing a RIDGE through the
(Te, ne) plane along the ionizing-recombining crossover -- not a single
interior maximum.

Supporting evidence so far (2 points):
    ITER ref   f3=0.474 f4=0.116  sens=+0.358  amplification 10.2x
    cold corner f3=0.9996 f4=0.9987 sens=+0.00093 amplification 0.99x

WHAT WOULD REFUTE IT
--------------------
  - amplification uncorrelated with |f3 - f4|
  - the maximum sitting at a grid CORNER rather than in the interior
    (would indicate the range is truncated, not that a ridge was found)
  - a strong dependence on step direction that |f3-f4| does not predict
  - the achieved fractional step varying systematically with Te (index
    snapping on a log grid) -- reported per point so this is visible

CONTROLLED STEP
---------------
The earlier runs used dTe = +0.6 eV ABSOLUTE, which is a 60% step at Te = 1 eV
and 6% at Te = 10 eV. Every contour of those maps is confounded with that.
This script uses a fixed FRACTIONAL step and reports the ACHIEVED fraction at
every point, since the Te grid is logarithmic (ratio 10^(1/49) = 1.0481 per
index) and the requested step snaps to a grid index.

SCOPE
-----
  - The ion density is a fixed reservoir in S and is never evolved.
  - Observable is the n=3 / n=4 SHELL POPULATION RATIO, not yet the Balmer
    line ratio (needs the C8 l-distribution check).
  - Plateau values are the analytic partial-equilibrium result. They describe
    the error for tau_relax << t << tau_QSS; where no such window exists
    (M below WIN_LO*WIN_HI) the point is flagged, not silently included.

Report only. Writes to the output directory; modifies nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05,
                   help="fractional Te step (default 0.05)")
    p.add_argument("--win-lo", type=float, default=30.0)
    p.add_argument("--win-hi", type=float, default=30.0)
    p.add_argument("--require-window", action="store_true", default=True,
                   help="report statistics ONLY over points that have a "
                        "timescale-separated plateau window (default on)")
    p.add_argument("--include-all", dest="require_window",
                   action="store_false",
                   help="include points with no plateau window in the "
                        "statistics (they are always written to the csv)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a = parse_args()
    ctx = CRContext.load()
    root = ctx.root
    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    if not S_path.exists():
        raise FileNotFoundError(f"missing source vector: {S_path}")

    L, S = ctx.L_grid, np.load(S_path)
    Te, ne = ctx.te_grid, ctx.ne_grid
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")

    g = int(ctx.ground_index)
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)
    N3 = np.where(np.asarray(ctx.n_values) == 3)[0]
    N4 = np.where(np.asarray(ctx.n_values) == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    out = a.out or (root / "validation" / "plateau_gridmap")
    out.mkdir(parents=True, exist_ok=True)

    lines, rows = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("PLATEAU GRID MAP -- analytic partial equilibrium, no integration")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"repo root      {root}")
    say(f"L_grid sha256  {sha256(L_path)}")
    say(f"S_grid sha256  {sha256(S_path)}")
    say(f"state idx sha  {sha256(ctx.state_index_path)}")
    say(f"n=3 {[int(i) for i in N3]} {[ctx.labels[i] for i in N3]}   "
        f"n=4 {[int(i) for i in N4]} {[ctx.labels[i] for i in N4]}")
    say(f"grid           {len(Te)} Te x {len(ne)} ne = {len(Te)*len(ne)} points")
    say(f"Te grid ratio  {Te[1]/Te[0]:.6f} per index  "
        f"(a {a.frac:.0%} step is {np.log1p(a.frac)/np.log(Te[1]/Te[0]):.3f} indices)")
    say(f"requested step +/- {a.frac:.1%} fractional; achieved fraction "
        f"reported per point")
    say("=" * 78)

    max_sup = 0.0
    n_skip_window = n_skip_move = 0

    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(len(Te)):
            k = int(np.argmin(np.abs(Te - Te[i] * (1 + sgn * a.frac))))
            if k == i:
                n_skip_move += 1
                continue
            achieved = Te[k] / Te[i] - 1.0
            for j in range(len(ne)):
                lam = np.linalg.eigvals(L[k, j])
                lam = lam[np.argsort(lam.real)[::-1]]
                if lam[0].real >= 0 or lam[1].real >= 0:
                    raise RuntimeError(
                        f"unstable operator at Te={Te[k]:g} ne={ne[j]:g}")
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                window_ok = (a.win_lo * tR) < (tQ / a.win_hi)
                if not window_ok:
                    n_skip_window += 1

                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(L[k, j], -S[k, j])

                LEE = L[k, j][np.ix_(E, E)]
                LEg = L[k, j][np.ix_(E, [g])].ravel()
                n0 = np.linalg.solve(LEE, -S[k, j][E])
                n1 = np.linalg.solve(LEE, -LEg * n_old[g])
                x_new = n_new[g] / n_old[g]

                sup = (np.abs(n0 + x_new * n1 - n_new[E]).max()
                       / np.abs(n_new[E]).max())
                max_sup = max(max_sup, sup)

                a3_0, a4_0 = n0[n3E].sum(), n0[n4E].sum()
                a3_1, a4_1 = n1[n3E].sum(), n1[n4E].sum()

                def sens(x):
                    f3 = x * a3_1 / (a3_0 + x * a3_1)
                    f4 = x * a4_1 / (a4_0 + x * a4_1)
                    return f3, f4, f3 - f4

                f3o, f4o, so = sens(1.0)
                f3n, f4n, sn = sens(x_new)

                Rq = n_new[N3].sum() / n_new[N4].sum()
                R_pe = (a3_0 + a3_1) / (a4_0 + a4_1)
                R_old = n_old[N3].sum() / n_old[N4].sum()
                d_step = R_old / Rq - 1.0
                d_pe = R_pe / Rq - 1.0
                amp = abs(d_pe) / abs(d_step) if d_step != 0 else np.nan

                lin_pred = abs(so * np.log(x_new))

                rows.append(dict(
                    direction=dlab, i=i, j=j, Te=Te[i], Te_new=Te[k],
                    ne=ne[j], frac_achieved=achieved,
                    tau_QSS=tQ, tau_relax=tR, M=tQ / tR, window_ok=window_ok,
                    x_new=x_new, superposition_err=sup,
                    eps_step=abs(d_step), eps_plateau=abs(d_pe),
                    signed_step=d_step, signed_plateau=d_pe,
                    amplification=amp,
                    f3_old=f3o, f4_old=f4o, sens_old=so,
                    f3_new=f3n, f4_new=f4n, sens_new=sn,
                    abs_sens_old=abs(so),
                    lin_pred=lin_pred, abs_ln_x=abs(np.log(x_new)),
                ))

    if not rows:
        raise RuntimeError("no grid points evaluated")

    keys = list(rows[0])
    A = {k: np.array([r[k] for r in rows]) for k in keys}

    say(f"\nevaluated {len(rows)} (point, direction) pairs")
    say(f"skipped {n_skip_move} Te rows where the step did not move an index")
    say(f"points with NO timescale-separated plateau window: {n_skip_window}")
    say(f"max superposition error over the whole grid: {max_sup:.3e}   "
        f"{'OK' if max_sup < 1e-8 else '*** TWO-CHANNEL SPLIT FAILS ***'}")
    say(f"achieved fractional step: {A['frac_achieved'].min():+.4f} .. "
        f"{A['frac_achieved'].max():+.4f}  (requested +/-{a.frac:.4f})")
    if a.require_window:
        say("STATISTICS BELOW ARE OVER window_ok POINTS ONLY. All points, "
            "flagged, are in the csv.")
    else:
        say("*** statistics include points with NO plateau window (--include-all) ***")

    say("")
    say("WHY eps_plateau AND NOT amplification IS THE PRIMARY QUANTITY")
    say("  amplification = eps_plateau/eps_step is unstable wherever eps_step")
    say("  passes through zero. In the first run the grid maximum was 1271x at")
    say("  a point whose eps_step (3.5e-5) was the grid MINIMUM while its")
    say("  eps_plateau (0.044) was BELOW the median. That ratio measures a")
    say("  vanishing denominator, not a large error. Amplification is still")
    say("  reported, but as a distribution, never as a headline maximum.")
    say("  Note also log(amp) = log(eps_plateau) - log(eps_step) identically,")
    say("  so corr(log eps_step, log amp) is partly tautological.")

    for dlab in ("heat", "cool"):
        sel = A["direction"] == dlab
        if a.require_window:
            sel = sel & A["window_ok"].astype(bool)
        if not sel.any():
            continue
        say("\n" + "-" * 78)
        say(f"{dlab.upper()}   {sel.sum()} points"
            f"{' (window_ok only)' if a.require_window else ''}")
        for name in ("eps_step", "eps_plateau", "amplification",
                     "abs_sens_old", "abs_ln_x", "f3_old", "f4_old"):
            v = A[name][sel]
            v = v[np.isfinite(v)]
            say(f"  {name:16s} min {v.min():.6g}  median {np.median(v):.6g}  "
                f"max {v.max():.6g}")

        ep, es = A["eps_plateau"][sel], A["eps_step"][sel]
        say(f"  eps_plateau > eps_step at {int((ep > es).sum())}/{int(sel.sum())} "
            f"points ({(ep > es).mean():.1%})")

        q = int(np.nanargmax(ep))
        idx = np.where(sel)[0][q]
        say(f"  max eps_plateau {ep[q]:.6f} at Te={A['Te'][idx]:.4f} eV, "
            f"ne={A['ne'][idx]:.4e} cm^-3 (grid [{int(A['i'][idx])},"
            f"{int(A['j'][idx])}])")
        say(f"    f3={A['f3_old'][idx]:.6f} f4={A['f4_old'][idx]:.6f} "
            f"|f3-f4|={A['abs_sens_old'][idx]:.6f} "
            f"|ln x_new|={A['abs_ln_x'][idx]:.4f} "
            f"eps_step={A['eps_step'][idx]:.6f}")
        edge = (A["i"][idx] in (0, len(Te) - 1)) or \
               (A["j"][idx] in (0, len(ne) - 1))
        say(f"    sits on a grid EDGE: {bool(edge)}   "
            f"{'-> range may be truncated, not a true ridge' if edge else '-> interior maximum'}")

        # --- MECHANISM TEST: the linearised prediction from the algebra -----
        lp = A["lin_pred"][sel]
        good = np.isfinite(lp) & np.isfinite(ep) & (lp > 0) & (ep > 0)
        if good.sum() > 10:
            r = np.corrcoef(np.log10(lp[good]), np.log10(ep[good]))[0, 1]
            ratio = ep[good] / lp[good]
            say(f"  MECHANISM corr( log |f3-f4|*|ln x_new| , log eps_plateau ) "
                f"= {r:+.4f}  (n={int(good.sum())})")
            say(f"    eps_plateau / linearised prediction: "
                f"min {ratio.min():.4f}  median {np.median(ratio):.4f}  "
                f"max {ratio.max():.4f}")
            say("    (a ratio near 1 means the linearised two-channel formula "
                "predicts the plateau error; large excursions |ln x_new| >> 1 "
                "are expected to break it)")
        sens_arr = A["abs_sens_old"][sel]
        g2 = np.isfinite(sens_arr) & (sens_arr > 0) & (ep > 0)
        if g2.sum() > 10:
            r2 = np.corrcoef(np.log10(sens_arr[g2]), np.log10(ep[g2]))[0, 1]
            say(f"  corr( log |f3-f4| alone , log eps_plateau ) = {r2:+.4f}")

    # --- heating vs cooling, matched for step size -------------------------
    say("\n" + "-" * 78)
    say("HEAT vs COOL -- the achieved steps differ (+4.81% vs -4.59% on a log")
    say("grid), so raw medians are not directly comparable. Per-point ratio,")
    say("matched at the same (i,j), normalised by the achieved step:")
    mh = (A["direction"] == "heat")
    mc = (A["direction"] == "cool")
    if a.require_window:
        mh = mh & A["window_ok"].astype(bool)
        mc = mc & A["window_ok"].astype(bool)
    hk = {(int(A["i"][t]), int(A["j"][t])): t for t in np.where(mh)[0]}
    ck = {(int(A["i"][t]), int(A["j"][t])): t for t in np.where(mc)[0]}
    both = sorted(set(hk) & set(ck))
    if both:
        rr = np.array([
            (A["eps_plateau"][hk[key]] / abs(A["frac_achieved"][hk[key]])) /
            (A["eps_plateau"][ck[key]] / abs(A["frac_achieved"][ck[key]]))
            for key in both])
        say(f"  matched pairs: {len(both)}")
        say(f"  step-normalised eps_plateau(heat)/eps_plateau(cool): "
            f"min {rr.min():.4f}  median {np.median(rr):.4f}  "
            f"max {rr.max():.4f}")
        say("  (a median near 1 means the heat/cool difference in the raw "
            "medians was a step-size effect, not an asymmetry)")

    say("\nsensitivity |f3-f4| vs Te at ne = "
        f"{ne[len(ne)//2]:.3e} (heating, window_ok only):")
    m = ((A["direction"] == "heat") & (A["j"] == len(ne) // 2)
         & A["window_ok"].astype(bool))
    o = np.argsort(A["Te"][m])
    for t_, s_, f3_, f4_, ep_, es_ in zip(
            A["Te"][m][o], A["abs_sens_old"][m][o], A["f3_old"][m][o],
            A["f4_old"][m][o], A["eps_plateau"][m][o], A["eps_step"][m][o]):
        say(f"   Te={t_:6.3f}  f3={f3_:.4f}  f4={f4_:.4f}  "
            f"|f3-f4|={s_:.4e}  eps_plateau={ep_:.6f}  eps_step={es_:.6f}")

    say("\nsensitivity |f3-f4| vs ne at Te = "
        f"{Te[len(Te)//2]:.3f} eV (heating, window_ok only) -- the direction")
    say("in which the recombining limit should be approached:")
    m = ((A["direction"] == "heat") & (A["i"] == len(Te) // 2)
         & A["window_ok"].astype(bool))
    o = np.argsort(A["ne"][m])
    for n_, s_, f3_, f4_, ep_ in zip(A["ne"][m][o], A["abs_sens_old"][m][o],
                                     A["f3_old"][m][o], A["f4_old"][m][o],
                                     A["eps_plateau"][m][o]):
        say(f"   ne={n_:.4e}  f3={f3_:.4f}  f4={f4_:.4f}  "
            f"|f3-f4|={s_:.4e}  eps_plateau={ep_:.6f}")

    txt, csv = out / "plateau_gridmap.txt", out / "plateau_gridmap.csv"
    txt.write_text("\n".join(lines) + "\n")
    with csv.open("w") as f:
        f.write(f"# generated {datetime.now():%Y-%m-%d %H:%M} by "
                f"{Path(__file__).name}\n")
        f.write(f"# L_grid sha256 {sha256(L_path)}\n")
        f.write(f"# S_grid sha256 {sha256(S_path)}\n")
        f.write(f"# state_index sha256 {sha256(ctx.state_index_path)}\n")
        f.write(f"# requested fractional step {a.frac}\n")
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    say(f"\nwrote {txt}")
    say(f"wrote {csv}")


if __name__ == "__main__":
    main()