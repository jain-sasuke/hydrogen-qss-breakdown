"""
verify_divertor_map.py
======================
The run that decides the thesis claim.

WHAT THIS ANSWERS
-----------------
Everything so far measured the QSS error in the n=3/n=4 shell ratio for an
INSTANTANEOUS step: eps_step at t=0, eps_plateau for tau_relax << t << tau_QSS.
But a divertor event has a finite duration. An ELM crash lasts ~100 us against
tau_QSS ~ 22.7 us at the ITER reference, so the ground state partly catches up
DURING the event. The question the thesis actually has to answer is:

    over a divertor event of duration tau_drive, how wrong is a QSS-based
    Balmer diagnostic, in the observable an experimentalist measures?

TWO BOUNDS, NOT ONE ESTIMATE
----------------------------
verify_plateau_slowmode.py established that at the ITER reference the residual
decays as a single exponential at lambda_0 (R^2 = 0.9999999, tau_fit = 1.079 x
tau_QSS) -- but at the cold corner it does NOT (tau_fit = 207 x tau_QSS; the
observable is pinned at its asymptote because both shells are ~100% ground-fed
and dlnR/dlnx ~ 9e-4). So a single decay law cannot be applied grid-wide
without validating it grid-wide, which would cost 400 stiff integrations.

Instead this script brackets the answer with two bounds that need no such
validation:

  LOWER (fast-recovery bound): the error decays on tau_QSS from the moment of
  the step, so the time-average over the event is

      eps_bar_lo = eps_plateau * (tau_QSS/tau_d) * (1 - exp(-tau_d/tau_QSS))

  UPPER (pinned bound): the error does not decay at all during the event,

      eps_bar_hi = eps_plateau

The truth lies between. Where the observable is pinned (cold corner), the
upper bound is the right one -- so the LOWER bound is CONSERVATIVE for the
breakdown claim: it understates the error wherever the decay is slower than
tau_QSS. Any breakdown reported at the lower bound is therefore robust.

Note the lower-bound law has the same functional form as the one already in
chapter4.tex line 953, but built on eps_plateau (a ground-free observable)
rather than eps_res (94% ground-state contamination -- see derivation_07).

EXTERNAL VALIDATION -- Fujimoto Appendix 4B
-------------------------------------------
Fujimoto, Plasma Spectroscopy (2004), Appendix 4B computes the temporal
development of excited populations for Te = 10 eV, ne = 1e18 m^-3 = 1e12 cm^-3
-- a point that lies exactly on this grid. He reports an excited-state
response time ~1e-7 s and a ground-state depletion time ~1e-4 s. This script
prints the model's values at that point for direct comparison.

This is the only comparison in the project against published timescales from
an independent code with independent atomic data. Gates A-E are all internal
consistency checks; this is not.

SCOPE / CAVEATS
---------------
  - Observable is the n=3/n=4 SHELL POPULATION RATIO. Calling it Halpha/Hbeta
    additionally needs the C8 l-distribution check.
  - Fixed FRACTIONAL Te step; the achieved fraction is reported. An absolute
    dTe on a log grid is a 60% step at 1 eV and 6% at 10 eV, and every contour
    of such a map is confounded with that.
  - Points with M < win_lo*win_hi have no separated plateau stage; flagged.
  - The ion density is a fixed reservoir in S and is never evolved.
  - eps_step here is the relative change of ONE scalar observable between two
    QSS states. It is NOT the eps_step in qss_analysis.py, which is a
    max-over-42-states ratio error normalised by the ground state. Same name,
    different quantity -- must be resolved in the Ch. 3 notation.

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

DRIVES = [
    ("ELM_crash", 1e-4),
    ("fast_detachment", 1e-3),
    ("slow_detachment", 1e-2),
    ("inter_ELM", 1e-1),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05,
                   help="fractional Te step (default 0.05)")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="breakdown threshold on the time-averaged error")
    p.add_argument("--win-lo", type=float, default=30.0)
    p.add_argument("--win-hi", type=float, default=30.0)
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

    out = a.out or (root / "validation" / "divertor_map")
    out.mkdir(parents=True, exist_ok=True)
    lines, rows = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("DIVERTOR DRIVE MAP -- time-averaged shell-ratio error, two bounds")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"repo root      {root}")
    say(f"L_grid sha256  {sha256(L_path)}")
    say(f"S_grid sha256  {sha256(S_path)}")
    say(f"state idx sha  {sha256(ctx.state_index_path)}")
    say(f"n=3 {[int(i) for i in N3]} {[ctx.labels[i] for i in N3]}   "
        f"n=4 {[int(i) for i in N4]} {[ctx.labels[i] for i in N4]}")
    say(f"fractional step +/-{a.frac:.1%}   breakdown threshold "
        f"{a.threshold:.0%}")
    say("LOWER bound: eps_plateau*(tQ/td)*(1-exp(-td/tQ))  -- decays on tau_QSS")
    say("UPPER bound: eps_plateau                          -- no decay (pinned)")
    say("The lower bound is CONSERVATIVE: it understates the error wherever")
    say("the observable decays more slowly than tau_QSS (as at the cold corner,")
    say("where tau_fit = 207 x tau_QSS).")
    say("OBSERVABLE: n=3/n=4 shell population ratio (NOT yet the Balmer line")
    say("ratio -- needs the C8 l-distribution check)")
    say("=" * 78)

    n_nowin = 0
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(len(Te)):
            k = int(np.argmin(np.abs(Te - Te[i] * (1 + sgn * a.frac))))
            if k == i:
                continue
            for j in range(len(ne)):
                lam = np.linalg.eigvals(L[k, j])
                lam = lam[np.argsort(lam.real)[::-1]]
                if lam[0].real >= 0 or lam[1].real >= 0:
                    raise RuntimeError(f"unstable operator at [{k},{j}]")
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                win = (a.win_lo * tR) < (tQ / a.win_hi)
                if not win:
                    n_nowin += 1

                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(L[k, j], -S[k, j])
                LEE = L[k, j][np.ix_(E, E)]
                LEg = L[k, j][np.ix_(E, [g])].ravel()
                n0 = np.linalg.solve(LEE, -S[k, j][E])
                n1 = np.linalg.solve(LEE, -LEg * n_old[g])
                x_new = n_new[g] / n_old[g]
                sup = (np.abs(n0 + x_new * n1 - n_new[E]).max()
                       / np.abs(n_new[E]).max())

                a3_0, a4_0 = n0[n3E].sum(), n0[n4E].sum()
                a3_1, a4_1 = n1[n3E].sum(), n1[n4E].sum()
                f3 = a3_1 / (a3_0 + a3_1)
                f4 = a4_1 / (a4_0 + a4_1)

                Rq = n_new[N3].sum() / n_new[N4].sum()
                ep = abs((a3_0 + a3_1) / (a4_0 + a4_1) / Rq - 1.0)
                es = abs(n_old[N3].sum() / n_old[N4].sum() / Rq - 1.0)

                r = dict(direction=dlab, i=i, j=j, Te=Te[i], ne=ne[j],
                         frac_achieved=Te[k] / Te[i] - 1.0,
                         tau_QSS=tQ, tau_relax=tR, M=tQ / tR, window_ok=win,
                         superposition_err=sup, eps_step=es, eps_plateau=ep,
                         f3=f3, f4=f4, sens=f3 - f4,
                         abs_ln_x=abs(np.log(x_new)))
                for name, td in DRIVES:
                    r[f"lo_{name}"] = ep * (tQ / td) * (1 - np.exp(-td / tQ))
                    r[f"hi_{name}"] = ep
                rows.append(r)

    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    ok = A["window_ok"].astype(bool)
    say(f"\nevaluated {len(rows)} (point, direction) pairs; "
        f"{int((~ok).sum())} have no separated plateau stage")
    say(f"max superposition error: {A['superposition_err'].max():.3e}   "
        f"{'OK' if A['superposition_err'].max() < 1e-8 else '*** FAILS ***'}")
    say(f"achieved fractional step {A['frac_achieved'].min():+.4f} .. "
        f"{A['frac_achieved'].max():+.4f}")
    say(f"tau_QSS over the grid: {A['tau_QSS'][ok].min():.3e} .. "
        f"{A['tau_QSS'][ok].max():.3e} s")

    say("\n" + "=" * 78)
    say("BREAKDOWN COUNTS (window_ok points only)")
    say(f"{'drive':<20} {'tau_d':>8}  {'LOWER bound':>28}  {'UPPER bound':>28}")
    say(f"{'':<20} {'':>8}  {'median':>10} {'max':>8} {'>thr':>8}  "
        f"{'median':>10} {'max':>8} {'>thr':>8}")
    for name, td in DRIVES:
        lo, hi = A[f"lo_{name}"][ok], A[f"hi_{name}"][ok]
        say(f"{name:<20} {td:>8.0e}  "
            f"{np.median(lo):>10.4f} {lo.max():>8.4f} "
            f"{(lo > a.threshold).sum():>4d}/{ok.sum():<3d}  "
            f"{np.median(hi):>10.4f} {hi.max():>8.4f} "
            f"{(hi > a.threshold).sum():>4d}/{ok.sum():<3d}")

    say("\n" + "=" * 78)
    say("EXTERNAL VALIDATION -- Fujimoto, Plasma Spectroscopy (2004) App. 4B")
    say("His example: Te = 10 eV, ne = 1e18 m^-3 = 1e12 cm^-3")
    say("  excited-state response time  ~1e-7 s")
    say("  ground-state depletion time  ~1e-4 s")
    it = int(np.argmin(np.abs(Te - 10.0)))
    jn = int(np.argmin(np.abs(np.log(ne) - np.log(1e12))))
    lam = np.linalg.eigvals(L[it, jn])
    lam = lam[np.argsort(lam.real)[::-1]]
    tQ_f, tR_f = -1.0 / lam[0].real, -1.0 / lam[1].real
    say(f"This model at grid [{it},{jn}] (Te={Te[it]:.4f} eV, "
        f"ne={ne[jn]:.3e} cm^-3):")
    say(f"  tau_relax = {tR_f:.4e} s   ratio to Fujimoto ~1e-7: "
        f"{tR_f/1e-7:.3f}")
    say(f"  tau_QSS   = {tQ_f:.4e} s   ratio to Fujimoto ~1e-4: "
        f"{tQ_f/1e-4:.3f}")
    say(f"  M = {tQ_f/tR_f:.4g}")
    say("  NOTE Fujimoto's response time is max_p t_tr(p), the 63% rise time")
    say("  of the slowest-responding level (= t_rl at Griem's boundary level),")
    say("  NOT 1/|lambda_1|. Agreement to a factor of a few is the meaningful")
    say("  comparison; exact agreement would be suspicious.")

    say("\n" + "=" * 78)
    say("ITER REFERENCE, all drives")
    ir = [r for r in rows if r["direction"] == "heat"
          and abs(r["Te"] - 2.947052) < 1e-4 and abs(r["ne"] - 1.389495e14)
          / 1.389495e14 < 1e-3]
    if ir:
        r = ir[0]
        say(f"  Te={r['Te']:.4f} eV  ne={r['ne']:.4e} cm^-3  "
            f"step {r['frac_achieved']:+.4f}")
        say(f"  tau_QSS={r['tau_QSS']:.4e} s  tau_relax={r['tau_relax']:.4e} s "
            f" M={r['M']:.4g}")
        say(f"  f3={r['f3']:.4f}  f4={r['f4']:.4f}  |f3-f4|={abs(r['sens']):.4f}")
        say(f"  eps_step={r['eps_step']:.6f}   eps_plateau={r['eps_plateau']:.6f}")
        for name, td in DRIVES:
            say(f"    {name:<18} td={td:.0e}  lower {r['lo_'+name]:.6f}   "
                f"upper {r['hi_'+name]:.6f}")

    say("\n" + "=" * 78)
    say("WHERE THE LOWER BOUND EXCEEDS THE THRESHOLD AT ELM TIMESCALES")
    m = ok & (A["lo_ELM_crash"] > a.threshold) & (A["direction"] == "heat")
    if m.any():
        say(f"  {int(m.sum())} heating points. Te range "
            f"{A['Te'][m].min():.3f}-{A['Te'][m].max():.3f} eV, ne range "
            f"{A['ne'][m].min():.3e}-{A['ne'][m].max():.3e} cm^-3")
        q = np.where(m)[0][int(np.argmax(A["lo_ELM_crash"][m]))]
        say(f"  worst: Te={A['Te'][q]:.4f} ne={A['ne'][q]:.4e}  "
            f"lower={A['lo_ELM_crash'][q]:.4f}  "
            f"eps_plateau={A['eps_plateau'][q]:.4f}  "
            f"tau_QSS={A['tau_QSS'][q]:.3e}")
    else:
        say("  none -- the lower bound stays under the threshold everywhere")

    txt, csv = out / "divertor_map.txt", out / "divertor_map.csv"
    txt.write_text("\n".join(lines) + "\n")
    keys = list(rows[0])
    with csv.open("w") as f:
        f.write(f"# generated {datetime.now():%Y-%m-%d %H:%M} by "
                f"{Path(__file__).name}\n")
        f.write(f"# L_grid sha256 {sha256(L_path)}\n")
        f.write(f"# S_grid sha256 {sha256(S_path)}\n")
        f.write(f"# state_index sha256 {sha256(ctx.state_index_path)}\n")
        f.write(f"# fractional step {a.frac}, threshold {a.threshold}\n")
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    say(f"\nwrote {txt}")
    say(f"wrote {csv}")


if __name__ == "__main__":
    main()
