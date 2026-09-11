#!/usr/bin/env python
"""
verify_trajectory_census.py
===========================
The full post-step trajectory at every (point, direction) pair of the divertor
map, and three things the thesis currently asserts from two points or from
none:

  A. the dynamic bridge, grid-wide (Round 2 item 2): over the plateau window
     W = [k tau_relax, tau_slow/k], k = 30, the maximum departure from the
     instantaneous QSS manifold, eps_track(t) = |R(t)/R_QSS+(u(t)) - 1|, the
     maximum departure from the analytic plateau, eps_PE(t) = |R(t)/R_PE - 1|,
     and the sustained-tracking onset t_track (first time after which
     eps_track < 2e-3 for all later samples), in units of tau_relax.
  B. the single-slow-mode estimate against the true time average, grid-wide
     (Round 2 item 4): <eps_CRE>_{tau_d} = (1/tau_d) int_0^{tau_d} |R(t)/R_new - 1| dt
     against eps_plateau (tau_slow/tau_d)(1 - exp(-tau_d/tau_slow)).
  C. the finite-exposure observable (Round 2 item 5): a detector with
     exposure tau_d records I_alpha = int j_alpha dt and I_beta = int j_beta dt
     separately, so the measured ratio is R_exp = I_alpha/I_beta, a
     brightness-weighted average of the instantaneous ratio, not the time
     average of the instantaneous error. eps_exp = |R_exp/R_CRE_new - 1|,
     with j = w . n, w the Einstein-A weights (Halpha 3s,3p,3d -> 2; Hbeta
     4s,4p,4d -> 2; 4f dark), and the same with shell indicator weights.

The census (window_ok, Te >= 2 eV, value > 0.10) is then counted on each of:
  the single-slow-mode estimate on the shell ratio (the thesis: 45 of 448),
  the true time average, shell and line,
  the exposure-integrated ratio, shell and line,
at tau_d = 100 us and 506 us.

THE SOLUTION
------------
Post-step conditions are constant, so with d0 = n_old - n_new,
    n(t) = n_new + exp(L+ t) d0                                  (exact)
    int_0^tau n dt = n_new tau + [exp(M tau)]_{0:N, N} ,  M = [[L+, d0],[0, 0]]
The second line is the standard augmented-matrix identity for
int_0^tau exp(L s) d0 ds, evaluated with scipy.linalg.expm: exact to round-off,
no quadrature. For the dense time sampling needed by A and B the propagator
is evaluated through the eigendecomposition L+ = V diag(lam) V^-1, which is
fast but can be ill-conditioned for a non-normal operator (the bridge script
refuses it for that reason). Here it is not trusted: every row is checked
against scipy.linalg.expm at NCHK log-spaced times, and the script raises if
the two disagree by more than the tolerance
    tol(t) = max(EIG_TOL, EPS_MACH * ||L||_2 * t)
relative to max|n|. The second term is the intrinsic float64 limit: the
exponent rates carry absolute error of order EPS_MACH ||L||, so the propagated
state carries relative error of order EPS_MACH ||L|| t, whichever method is
used. Evidence (11 Sep 2026, cold corner heat [0,0], cond(V) = 11,
||L||_2 = 3.2e10, tau_slow = 36 s): against a 30-digit mpmath exponential the
eigen path errs by 1.6e-10, 9.1e-9, 8.1e-8 at t = 1e-3, 5.6e-2, 0.5 s and
scipy expm by 1.5e-10, 4.4e-8, 1.8e-7, so at late times the eigen path is the
more accurate of the two and a fixed 1e-8 tolerance would reject the better
answer. The gate normalised to max|n| is weak for the observable, because
max|n| is the ground state and the n = 3, 4 populations are ~1e4 smaller, so
a second gate is applied ON THE OBSERVABLE: at the same check times the
CRE error from the two propagators must agree to max(1e-8, eps ||L|| t)
absolute (at t = 1 s in the cold corner both propagators carry the intrinsic
7e-6; at every census time the bound is below 4e-6). Measured (skeptic pass, 11 Sep 2026): eps ||L||_2 t at
t = 100 us reaches 3.7e-6 on the grid, the state-level deviation reaches
3.7e-7 of max|n|, and the observable-level disagreement is at most 1.5e-10
on eps_track over the window and 5e-16 absolute on eps_CRE at 100 us. The trapezoid
integral of the sampled eps_CRE is also checked against the exact augmented
integral of n(t) at the benchmark, and its convergence is checked by
re-integrating at four times the sample density at the five tab:lowerbound
points.

PREDICTIONS, WRITTEN BEFORE THE FIRST RUN
-----------------------------------------
P1  t_track at heat [23,5] and heat [15,3] reproduces the bridge script's
    1.94 and 3.72 tau_relax (TOL 2e-3), and max_W eps_PE reproduces 3.33% and
    1.28% on one of the two windows the bridge script uses (k = 30, or the
    historical 20 tau_relax .. 0.02 tau_slow).
    CORRECTION (11 Sep 2026, skeptic pass): the second half compared the wrong
    quantity. Chapter 5's 3.33% and 1.28% are max |eps_CRE(t)/eps_plateau - 1|
    over the plateau, the flatness of the CRE error, not max_W eps_PE. That
    quantity is now computed as `flat_w30_*` and reproduces 0.0331 and 0.0130.
    t_track reproduces to 2% at 100 samples/decade and to 0.4% at 400.
P2  max_W eps_track < 1e-3 at every window_ok pair: the QSS closure holds
    dynamically everywhere, not only at two points. Refuting observation:
    any window_ok pair with max_W eps_track > 2e-3.
P3  The true 100 us average reproduces tab:lowerbound (chapter 4) at its
    five points: 0.386785, 0.175272, 0.072175, 0.011790, 0.083065, ratios
    0.9999, 1.0026, 1.0063, 1.0066, 1.0007. The [0,4] ratio below 1 is
    physical, not a trapezoid artifact: during the rise on tau_relax the
    error is eps_step < eps_plateau, and the estimate assumes the plateau
    from t = 0. Its size should be of order (eps_plateau - eps_step)
    tau_relax / (eps_plateau tau_d). Refuting observation: a ratio below 1
    that does not survive a fourfold denser time grid.
    CORRECTION (11 Sep 2026, skeptic pass): the arithmetic first quoted for
    this formula, 2.2e-5, dropped the division by eps_plateau; the formula
    gives 5.8e-5. The converged deficit at [0,4] is 2.19e-5 (stable to 4e-9
    under a fourfold denser grid), and its exact decomposition is a rise
    deficit of -5.21e-5 plus +3.0e-5 from the curvature term described under
    MECHANISM below, residual 1.5e-9. The formula models the rise well; the
    number first quoted matched the net only through that cancellation.

MECHANISM OF THE ESTIMATE'S FAILURE (established by the skeptic pass, now
computed per row as delta_slow_*)
-----------------------------------------------------------------------
Over the 448 warm pairs the ratio true/estimate runs 0.979 to 1.022 at 100 us.
The slow mode projects strongly onto the observable everywhere, no second mode
contributes (residual 1e-9), and the departure is not a transient: it is that
eps_CRE = |R(t)/R_new - 1| is a RATIO whose denominator relaxes on the same
tau_slow. With delta = (w_b . slow-mode component of d0)/(w_b . n_new),
    eps_CRE(t) ~ |alpha B - beta A| e^{lambda_0 t} / (A (B + beta e^{lambda_0 t}))
which is not an exponential; for tau_d >> tau_slow the ratio tends to
(1 + delta) ln(1 + delta)/delta ~ 1 + delta/2. Heating gives delta > 0 (the
estimate is low), cooling delta < 0 (the estimate is high). For the 4.8% step
|delta| <= 0.09, which bounds the estimate's error at about 4%.
P4  The estimate is within 1% of the true average at every warm pair, so the
    census on the true average equals 45 of 448 to within a few
    threshold-adjacent pairs. Refuting observation: a warm pair where the
    estimate and the true average differ by more than 5%.
P5  The exposure-integrated line ratio gives a census within a few pairs of
    the time-averaged one, because R(t) moves by at most eps_plateau (< 40%)
    while j_beta(t) is dominated by the CRE level, so the brightness
    weighting is mild. Refuting observation: a census that changes by more
    than 10 pairs, or a worst case that moves by more than 20%.

Read-only. Writes under validation/trajectory_census/ only with --write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import expm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)
sys.path.insert(0, str(ROOT / "src" / "rates"))

FRAC = 0.05
WIN_K = 30.0
V1_LO, V1_HI = 20.0, 0.02          # the bridge script's historical window
TOL_TRACK = 2.0e-3
THRESHOLD = 0.10
TE_FLOOR = 2.0
DRIVES_S = (100e-6, 506e-6)
PTS_PER_DECADE = 400          # 100 carries a +8e-5 relative trapezoid bias
OBS_TOL = 1e-8                # absolute floor on eps_CRE at the check times; the same
                              # eps_mach ||L|| t term applies (both propagators share it)
NCHK = 12
EIG_TOL = 1e-8
EPS_MACH = np.finfo(float).eps
BENCH = (23, 5)
TAB_LOWERBOUND = [("heat", 0, 4), ("heat", 15, 3), ("heat", 15, 5),
                  ("heat", 23, 5), ("heat", 0, 0)]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv_with_comments(p: Path) -> list[dict]:
    with p.open() as fh:
        return list(csv.DictReader([ln for ln in fh if not ln.startswith("#")]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--only-warm", action="store_true",
                    help="propagate only window_ok rows with Te >= floor")
    ap.add_argument("--pts-per-decade", type=int, default=PTS_PER_DECADE,
                    help="time samples per decade (convergence check: run at 4x)")
    a = ap.parse_args()
    ppd = a.pts_per_decade

    import Balmer_transient_ratio as btr                  # noqa: E402

    t_start = time.time()
    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = int(ctx.ground_index); nv = np.asarray(ctx.n_values); labels = ctx.labels
    N = ctx.n_states
    E = np.array([i for i in range(N) if i != g], dtype=int)
    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not sp.exists():
        raise FileNotFoundError(f"missing {sp}")
    S = np.load(sp)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    N3 = np.where(nv == 3)[0]; N4 = np.where(nv == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")
    if ctx.nearest_point(2.947, 1.389e14) != BENCH:
        raise RuntimeError("benchmark moved; grids are not the recorded ones")

    dmap_p = ROOT / "validation/divertor_map/divertor_map.csv"
    wc_p = ROOT / "validation/weighted_census/weighted_census.csv"
    for p in (dmap_p, wc_p):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    dmap = {(r["direction"], int(r["i"]), int(r["j"])): r for r in read_csv_with_comments(dmap_p)}
    wcen = {(r["direction"], int(r["i"]), int(r["j"])): r for r in read_csv_with_comments(wc_p)}
    with dmap_p.open() as fh:
        hdr = [ln for ln in fh if ln.startswith("#")]
    if sha256(lp) not in hdr[1] or sha256(sp) not in hdr[2]:
        raise RuntimeError("divertor_map.csv was built on different L_grid/S_grid")

    # weights: photon emissivity, same construction as verify_weighted_census.py
    lw = btr.load_radiative_weights(use_photon_energy=False)
    W = {"shell": {"a": np.zeros(N), "b": np.zeros(N)},
         "line": {"a": np.zeros(N), "b": np.zeros(N)}}
    W["shell"]["a"][N3] = 1.0; W["shell"]["b"][N4] = 1.0
    for line, key in (("Halpha", "a"), ("Hbeta", "b")):
        for (idx_u, _il, _nu, _lu, _nl, _ll, lab) in btr.LINE_CHANNELS[line]:
            if labels[idx_u].upper() != lab.split("_")[0]:
                raise RuntimeError(f"state ordering mismatch at {idx_u}: {labels[idx_u]} vs {lab}")
            W["line"][key][idx_u] = lw.weights[line][lab]
    for key in ("a", "b"):
        if W["line"][key].max() <= 0 or W["line"][key].min() < 0:
            raise RuntimeError("bad line weights")

    out_lines: list[str] = []
    def say(s: str = "") -> None:
        print(s); out_lines.append(s)

    say("=" * 78)
    say("TRAJECTORY CENSUS: dynamic bridge, true time average, finite exposure")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}")
    say(f"interpreter {sys.executable}   numpy {np.__version__}")
    say(f"L_grid sha256      {sha256(lp)}")
    say(f"S_grid sha256      {sha256(sp)}")
    say(f"state_index sha256 {sha256(ctx.state_index_path)}")
    say(f"radiative sha256   {sha256(btr.DATA_RAD)}")
    say(f"window k = {WIN_K:g}; TOL_TRACK {TOL_TRACK:g}; threshold {THRESHOLD}; "
        f"Te floor {TE_FLOOR}; drives {[f'{d*1e6:g} us' for d in DRIVES_S]}; "
        f"{ppd} samples/decade; eigen-vs-expm check at {NCHK} times, tol {EIG_TOL:g}")
    say("=" * 78)

    def ratio(w, n):
        return float(w["a"] @ n) / float(w["b"] @ n)

    rows = []
    worst_eig = 0.0
    worst_imag = 0.0
    n_prop = 0
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(len(te)):
            k = int(np.argmin(np.abs(te - te[i] * (1 + sgn * FRAC))))
            if k == i:
                continue
            for j in range(len(ne)):
                key = (dlab, i, j)
                d = dmap[key]
                win = d["window_ok"] == "True"
                warm = win and te[i] >= TE_FLOOR
                if a.only_warm and not warm:
                    continue
                Lp, Sp = L[k, j], S[k, j]
                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(Lp, -Sp)
                d0 = n_old - n_new
                lam, V = np.linalg.eig(Lp)
                order = np.argsort(lam.real)[::-1]
                lam, V = lam[order], V[:, order]
                if lam[0].real >= 0 or lam[1].real >= 0:
                    raise RuntimeError(f"unstable post-step operator at [{k},{j}]")
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                if abs(tQ / float(d["tau_QSS"]) - 1) > 1e-9:
                    raise RuntimeError(f"tau_slow differs from divertor_map at {key}")
                coef = np.linalg.solve(V, d0)

                # post-step QSS channels, R_PE, R_new, eps_plateau
                LEE = Lp[np.ix_(E, E)]; LEg = Lp[np.ix_(E, [g])].ravel()
                avec = np.linalg.solve(LEE, -LEg)
                cvec = np.linalg.solve(LEE, -Sp[E])
                nPE = np.zeros(N); nPE[E] = avec * n_old[g] + cvec; nPE[g] = n_old[g]
                obs = {}
                for wname, w in W.items():
                    wa, wb = w["a"][E], w["b"][E]
                    A3, C3, A4, C4 = wa @ avec, wa @ cvec, wb @ avec, wb @ cvec
                    R_new = ratio(w, n_new); R_PE = ratio(w, nPE); R_old = ratio(w, n_old)
                    ep = abs(R_PE / R_new - 1); es = abs(R_old / R_new - 1)
                    ref = float(d["eps_plateau"]) if wname == "shell" else float(wcen[key]["eps_plateau_line"])
                    if abs(ep - ref) > 1e-9 * max(ref, 1e-300):
                        raise RuntimeError(f"eps_plateau ({wname}) {ep:.9e} != recorded {ref:.9e} at {key}")
                    obs[wname] = dict(A3=A3, C3=C3, A4=A4, C4=C4, R_new=R_new, R_PE=R_PE,
                                      R_old=R_old, ep=ep, es=es)

                # time grid
                t_lo = 1e-4 * tR
                t_hi = max(10.0 * tQ, 2.0 * max(DRIVES_S))
                ts = np.logspace(np.log10(t_lo), np.log10(t_hi),
                                 int(np.ceil(np.log10(t_hi / t_lo) * ppd)) + 1)
                extras = [WIN_K * tR, tQ / WIN_K, V1_LO * tR, V1_HI * tQ, *DRIVES_S]
                ts = np.unique(np.concatenate([ts, [x for x in extras if t_lo <= x <= t_hi]]))
                # eigen-propagation, real part; imaginary part is a check
                ex = np.exp(np.outer(ts, lam))                       # (T, N) complex
                ntc = n_new[None, :] + (ex * coef[None, :]) @ V.T    # (T, N)
                worst_imag = max(worst_imag, float(np.abs(ntc.imag).max() / np.abs(ntc.real).max()))
                nt = ntc.real
                # check against expm at NCHK times
                chk_idx = np.unique(np.linspace(0, len(ts) - 1, NCHK).astype(int))
                Lnorm = np.linalg.norm(Lp, 2)
                for q in chk_idx:
                    n_ex = n_new + expm(Lp * ts[q]) @ d0
                    dev = float(np.abs(n_ex - nt[q]).max() / np.abs(n_ex).max())
                    tol_q = max(EIG_TOL, EPS_MACH * Lnorm * ts[q])
                    worst_eig = max(worst_eig, dev / tol_q)
                    # observable-level gate: shell ratio from both propagators
                    ws_a, ws_b = W["shell"]["a"], W["shell"]["b"]
                    R_e, R_x = (ws_a @ nt[q]) / (ws_b @ nt[q]), (ws_a @ n_ex) / (ws_b @ n_ex)
                    Rn = (ws_a @ n_new) / (ws_b @ n_new)
                    if abs(abs(R_e / Rn - 1) - abs(R_x / Rn - 1)) > max(OBS_TOL, EPS_MACH * Lnorm * ts[q]):
                        raise RuntimeError(
                            f"propagators disagree on eps_CRE at {key}, t={ts[q]:.3e}: "
                            f"{abs(R_e / Rn - 1):.3e} vs {abs(R_x / Rn - 1):.3e}")
                    if dev > tol_q:
                        raise RuntimeError(
                            f"eigen-propagation disagrees with expm at {key}, t={ts[q]:.3e}: "
                            f"{dev:.3e} against tol {tol_q:.3e}. The operator is too non-normal for this path; "
                            f"switch this row to expm before trusting anything.")
                n_prop += 1
                if nt.min() < -1e-8 * nt.max():
                    raise RuntimeError(f"negative population at {key}: {nt.min():.3e}")

                r = dict(direction=dlab, i=i, j=j, Te=float(te[i]), ne=float(ne[j]),
                         tau_slow=tQ, tau_relax=tR, M=tQ / tR, window_ok=win, warm=warm)
                ug = nt[:, g]
                # growth factor of the excited-block deviation from the frozen-reservoir
                # state: max_t ||n_E(t) - n_E^PE|| / ||n_E(0) - n_E^PE||, over t <= 10 tau_relax
                # (beyond that the reservoir drift pulls the state away from n^PE, which
                # is the slow relaxation and not transient growth). L2 norm.
                dPE0 = n_old[E] - nPE[E]
                m_early = ts <= 10.0 * tR
                dPE = nt[m_early][:, E] - nPE[E][None, :]
                r["growth_L2_10tr"] = float(np.linalg.norm(dPE, axis=1).max() / np.linalg.norm(dPE0))
                dPE_all = nt[:, E] - nPE[E][None, :]
                r["growth_L2_all"] = float(np.linalg.norm(dPE_all, axis=1).max() / np.linalg.norm(dPE0))
                for wname, w in W.items():
                    o = obs[wname]
                    Rt = (nt @ w["a"]) / (nt @ w["b"])
                    Rq = (o["A3"] * ug + o["C3"]) / (o["A4"] * ug + o["C4"])
                    e_track = np.abs(Rt / Rq - 1.0)
                    e_pe = np.abs(Rt / o["R_PE"] - 1.0)
                    e_cre = np.abs(Rt / o["R_new"] - 1.0)
                    # A. bridge metrics on the k window and the historical window
                    for wl, lo, hi in (("w30", WIN_K * tR, tQ / WIN_K), ("v1", V1_LO * tR, V1_HI * tQ)):
                        m = (ts >= lo * (1 - 1e-12)) & (ts <= hi * (1 + 1e-12))
                        if m.any() and lo < hi:
                            r[f"max_track_{wl}_{wname}"] = float(e_track[m].max())
                            r[f"max_pe_{wl}_{wname}"] = float(e_pe[m].max())
                            r[f"cre_start_{wl}_{wname}"] = float(e_cre[m][0])
                            r[f"cre_end_{wl}_{wname}"] = float(e_cre[m][-1])
                            # flatness of the CRE error over the window: chapter 5's 3.33%
                            r[f"flat_{wl}_{wname}"] = float(np.abs(e_cre[m] / o["ep"] - 1.0).max())
                        else:
                            for nm in ("max_track", "max_pe", "cre_start", "cre_end", "flat"):
                                r[f"{nm}_{wl}_{wname}"] = np.nan
                    # slow-mode amplitude in the denominator channel relative to CRE
                    slow_comp = (coef[0] * V[:, 0]).real
                    delta = float(w["b"] @ slow_comp) / float(w["b"] @ n_new)
                    r[f"delta_slow_{wname}"] = delta
                    r[f"asymptote_{wname}"] = float((1 + delta) * np.log1p(delta) / delta) if abs(delta) > 1e-14 else 1.0
                    # sustained tracking onset (up to 10 tau_slow)
                    m10 = ts <= 10.0 * tQ
                    below = e_track[m10] < TOL_TRACK
                    if below[-1]:
                        # last index where it is NOT below, then the next sample
                        bad = np.where(~below)[0]
                        onset = ts[m10][bad[-1] + 1] if len(bad) else ts[m10][0]
                        r[f"t_track_{wname}"] = float(onset / tR)
                    else:
                        r[f"t_track_{wname}"] = np.nan
                    r[f"eps_plateau_{wname}"] = o["ep"]; r[f"eps_step_{wname}"] = o["es"]
                    # does the SIGNED error change sign during the rise? (then the
                    # exposure integral can cancel in principle)
                    r[f"sign_change_{wname}"] = bool(np.sign(o["R_old"] / o["R_new"] - 1.0)
                                                     != np.sign(o["R_PE"] / o["R_new"] - 1.0))
                    # B. true time average of eps_CRE over [0, tau_d], trapezoid incl. t=0
                    for td in DRIVES_S:
                        m = ts <= td
                        tt = np.concatenate([[0.0], ts[m]])
                        ee = np.concatenate([[o["es"]], e_cre[m]])
                        if tt[-1] < td:               # td above grid (never, but guard)
                            raise RuntimeError(f"time grid does not reach tau_d at {key}")
                        avg = float(np.trapezoid(ee, tt) / td)
                        est = o["ep"] * (tQ / td) * (1.0 - np.exp(-td / tQ))
                        tag = f"{td*1e6:g}us"
                        r[f"true_avg_{tag}_{wname}"] = avg
                        # effect of the whole early transient (t < 50 tau_relax) on the
                        # average: replace eps there by the plateau value and compare
                        m50 = tt <= 50.0 * tR
                        if m50.sum() > 1 and 50.0 * tR < td:
                            early = float(np.trapezoid(ee[m50] - o["ep"], tt[m50]))
                            r[f"early_effect_{tag}_{wname}"] = early / (td * avg)
                        else:
                            r[f"early_effect_{tag}_{wname}"] = np.nan
                        r[f"estimate_{tag}_{wname}"] = est
                        r[f"ratio_true_est_{tag}_{wname}"] = avg / est if est > 0 else np.nan
                        # C. exposure-integrated ratio, exact augmented expm
                        Maug = np.zeros((N + 1, N + 1)); Maug[:N, :N] = Lp; Maug[:N, N] = d0
                        In = n_new * td + expm(Maug * td)[:N, N]
                        if wname == "shell" and td == DRIVES_S[0] and key == ("heat", *BENCH):
                            m2 = ts <= td
                            tt2 = np.concatenate([[0.0], ts[m2]])
                            nn2 = np.vstack([n_old[None, :], nt[m2]])
                            In_trap = np.trapezoid(nn2, tt2, axis=0)
                            r["exposure_integral_check"] = float(np.abs(In_trap - In).max() / np.abs(In).max())
                        R_exp = float(w["a"] @ In) / float(w["b"] @ In)
                        r[f"eps_exp_{tag}_{wname}"] = abs(R_exp / o["R_new"] - 1.0)
                rows.append(r)

    say(f"\npropagated {n_prop} (point, direction) pairs in {time.time()-t_start:.0f} s")
    say(f"eigen-propagation against expm: worst deviation/tolerance {worst_eig:.3e} "
        f"(tolerance max({EIG_TOL:g}, eps ||L|| t)); worst relative imaginary residue {worst_imag:.3e}")
    rb = [r for r in rows if (r["direction"], r["i"], r["j"]) == ("heat", *BENCH)][0]
    say(f"exposure integral, augmented expm against trapezoid of the sampled trajectory, "
        f"benchmark 100 us: {rb['exposure_integral_check']:.3e}")

    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    ok = A["window_ok"]; warm = A["warm"]
    def at(key):
        return [r for r in rows if (r["direction"], r["i"], r["j"]) == key][0]

    say("\n" + "=" * 78)
    say("A. DYNAMIC BRIDGE, GRID-WIDE (shell observable; k = 30 window unless stated)")
    say("=" * 78)
    for key, want_t, want_pe in ((("heat", 23, 5), 1.94, 0.0333), (("heat", 15, 3), 3.72, 0.0128)):
        r = at(key)
        say(f"  {key}: t_track {r['t_track_shell']:.3f} tau_relax (bridge script: {want_t}); "
            f"flatness max_W |eps_CRE/eps_plateau - 1| k=30 {r['flat_w30_shell']:.4f} "
            f"(chapter 5: {want_pe}); max_W eps_PE k=30 {r['max_pe_w30_shell']:.5f}, "
            f"historical window {r['max_pe_v1_shell']:.5f}; max_W eps_track k=30 {r['max_track_w30_shell']:.3e}")
    for wname in ("shell", "line"):
        mt = A[f"max_track_w30_{wname}"][ok]; mp = A[f"max_pe_w30_{wname}"][ok]; tt = A[f"t_track_{wname}"][ok]
        say(f"  {wname:5s} over {ok.sum()} window_ok pairs: max_W eps_track max {np.nanmax(mt):.3e}, "
            f"median {np.nanmedian(mt):.3e}; pairs above TOL {int(np.nansum(mt > TOL_TRACK))}")
        say(f"        max_W eps_PE: median {np.nanmedian(mp):.4f}, max {np.nanmax(mp):.4f} at "
            f"{[ (r['direction'], r['i'], r['j']) for r in rows if r['window_ok'] and r[f'max_pe_w30_{wname}'] == np.nanmax(mp)][0]}")
        say(f"        t_track: median {np.nanmedian(tt):.2f} tau_relax, max {np.nanmax(tt):.2f}, "
            f"never settled {int(np.isnan(tt).sum())}")
    mt = A["max_track_w30_shell"][warm]
    say(f"  warm (Te >= 2, {warm.sum()} pairs): max_W eps_track max {np.nanmax(mt):.3e}")
    prod = A["max_track_w30_shell"][ok] * A["M"][ok]
    cc = np.corrcoef(np.log(A["max_track_w30_shell"][ok]), np.log(1.0 / A["M"][ok]))[0, 1]
    say(f"  scaling: max_W eps_track * M over 680 window pairs: median {np.median(prod):.3f}, "
        f"range {prod.min():.3f} to {prod.max():.3f}; log-log correlation of max_W eps_track with 1/M {cc:+.3f}")
    gr = A["growth_L2_10tr"]; gra = A["growth_L2_all"]
    say(f"  growth of the excited-block deviation from partial equilibrium (L2, t <= 10 tau_relax): "
        f"max over 784 pairs {gr.max():.4f} (never above 1: {bool((gr <= 1 + 1e-12).all())}); "
        f"over all t the max is {gra.max():.3f}, from the slow reservoir drift, not transient growth")
    ee_b = rb["early_effect_100us_shell"]
    ee_all = A["early_effect_100us_shell"][warm]
    say(f"  early transient (t < 50 tau_relax) effect on the 100 us average: benchmark {ee_b:+.2e} relative; "
        f"warm pairs {np.nanmin(ee_all):+.2e} to {np.nanmax(ee_all):+.2e}")

    say("\n" + "=" * 78)
    say("B. TRUE 100 us AVERAGE AGAINST THE SINGLE-SLOW-MODE ESTIMATE (shell)")
    say("=" * 78)
    # convergence: recompute the five tab:lowerbound averages at 4x the sampling density
    say(f"  convergence check at {4*ppd} samples/decade (four times this run), 100 us, shell:")
    for key in TAB_LOWERBOUND:
        dlab, i, j = key
        k = int(np.argmin(np.abs(te - te[i] * (1 + (1 if dlab == "heat" else -1) * FRAC))))
        Lp, Sp = L[k, j], S[k, j]
        n_old = np.linalg.solve(L[i, j], -S[i, j]); n_new = np.linalg.solve(Lp, -Sp); d0 = n_old - n_new
        lam, V = np.linalg.eig(Lp); order = np.argsort(lam.real)[::-1]; lam, V = lam[order], V[:, order]
        coef = np.linalg.solve(V, d0); tR = -1.0 / lam[1].real
        td = DRIVES_S[0]
        ts4 = np.logspace(np.log10(1e-4 * tR), np.log10(td), int(np.ceil(np.log10(td / (1e-4 * tR)) * 4 * ppd)) + 1)
        nt4 = (n_new[None, :] + (np.exp(np.outer(ts4, lam)) * coef[None, :]) @ V.T).real
        ws_a, ws_b = W["shell"]["a"], W["shell"]["b"]
        Rn = (ws_a @ n_new) / (ws_b @ n_new); Ro = (ws_a @ n_old) / (ws_b @ n_old)
        ee4 = np.concatenate([[abs(Ro / Rn - 1)], np.abs((nt4 @ ws_a) / (nt4 @ ws_b) / Rn - 1)])
        avg4 = float(np.trapezoid(ee4, np.concatenate([[0.0], ts4])) / td)
        r5 = at(key)
        r5["true_avg_100us_shell_4x"] = avg4
        say(f"    {key}: {r5['true_avg_100us_shell']:.9f} -> {avg4:.9f}  relative change {avg4 / r5['true_avg_100us_shell'] - 1:+.2e}")
    say("  tab:lowerbound (chapter 4) reproduction:")
    for key, want_avg, want_ratio in zip(TAB_LOWERBOUND,
                                         (0.386785, 0.175272, 0.072175, 0.011790, 0.083065),
                                         (0.9999, 1.0026, 1.0063, 1.0066, 1.0007)):
        r = at(key)
        say(f"    {key}: eps_plateau {r['eps_plateau_shell']:.6f}  true avg {r['true_avg_100us_shell']:.6f} "
            f"(table {want_avg})  estimate {r['estimate_100us_shell']:.6f}  ratio {r['ratio_true_est_100us_shell']:.4f} "
            f"(table {want_ratio})  eps_step {r['eps_step_shell']:.6f}  tau_relax/tau_d {r['tau_relax']/DRIVES_S[0]:.2e}  "
            f"early-transient effect {r['early_effect_100us_shell']:+.2e}")
    for tag in [f"{td*1e6:g}us" for td in DRIVES_S]:
        for wname in ("shell", "line"):
            q = A[f"ratio_true_est_{tag}_{wname}"][warm]
            say(f"  {tag} {wname:5s}, warm: true/estimate {np.nanmin(q):.4f} to {np.nanmax(q):.4f}, "
                f"median {np.nanmedian(q):.4f}; below 1 at {int((q < 1).sum())} of {warm.sum()}; "
                f"|ratio-1| > 5% at {int((np.abs(q-1) > 0.05).sum())}")
        q = A[f"ratio_true_est_{tag}_shell"][ok]
        say(f"  {tag} shell, all window_ok: {np.nanmin(q):.4f} to {np.nanmax(q):.4f}")
    dl = A["delta_slow_line"][warm]; asy = A["asymptote_line"][warm]
    say(f"  mechanism: slow-mode amplitude in the Hbeta channel relative to CRE, delta, over warm pairs: "
        f"{dl.min():+.4f} to {dl.max():+.4f}; heating all positive {bool((dl[A['direction'][warm]=='heat']>0).all())}, "
        f"cooling all negative {bool((dl[A['direction'][warm]=='cool']<0).all())}")
    say(f"  asymptote (1+delta)ln(1+delta)/delta for tau_d >> tau_slow: {asy.min():.4f} to {asy.max():.4f}; "
        f"the 506 us true/estimate range above should sit inside it where tau_d/tau_slow is large")

    say("\n" + "=" * 78)
    say("C. THE CENSUS ON FIVE DEFINITIONS OF THE 100 us / 506 us QUANTITY")
    say("=" * 78)
    say(f"  scope: window_ok and Te >= {TE_FLOOR} eV, {warm.sum()} pairs; threshold {THRESHOLD}")
    census = []
    for td in DRIVES_S:
        tag = f"{td*1e6:g}us"
        defs = [("estimate, shell (thesis)", f"estimate_{tag}_shell"),
                ("estimate, line", f"estimate_{tag}_line"),
                ("true average, shell", f"true_avg_{tag}_shell"),
                ("true average, line", f"true_avg_{tag}_line"),
                ("exposure ratio, shell", f"eps_exp_{tag}_shell"),
                ("exposure ratio, line", f"eps_exp_{tag}_line")]
        base_by = {"shell": warm & (A[f"estimate_{tag}_shell"] > THRESHOLD),
                   "line": warm & (A[f"estimate_{tag}_line"] > THRESHOLD)}
        say(f"\n  tau_d = {tag}")
        say(f"  {'definition':28s} {'count':>9s} {'worst':>8s} {'at':>16s} {'enter':>5s} {'leave':>5s}   (enter/leave relative to the estimate on the SAME observable)")
        for name, col in defs:
            base = base_by["line"] if name.endswith("line") else base_by["shell"]
            est_col = f"estimate_{tag}_line" if name.endswith("line") else f"estimate_{tag}_shell"
            v = A[col]; c = warm & (v > THRESHOLD)
            kw = np.where(warm)[0][np.argmax(v[warm])]
            loc = f"{A['direction'][kw]} [{A['i'][kw]},{A['j'][kw]}]"
            ent = np.where(c & ~base)[0]; lea = np.where(base & ~c)[0]
            say(f"  {name:28s} {c.sum():>4d}/{warm.sum():<4d} {v[warm].max():>8.4f} {loc:>16s} "
                f"{len(ent):>5d} {len(lea):>5d}")
            for n_ in ent:
                say(f"      enters: {A['direction'][n_]} [{A['i'][n_]},{A['j'][n_]}] {v[n_]:.6f} (estimate, same observable {A[est_col][n_]:.6f})")
            for n_ in lea:
                say(f"      leaves: {A['direction'][n_]} [{A['i'][n_]},{A['j'][n_]}] {v[n_]:.6f} (estimate, same observable {A[est_col][n_]:.6f})")
            census.append(dict(tau_d_us=round(td * 1e6, 3), definition=name, n_scope=int(warm.sum()),
                               count=int(c.sum()), worst=float(v[warm].max()), worst_at=loc,
                               n_enter=len(ent), n_leave=len(lea)))
        # how different are the exposure ratio and the time average, pointwise
        q = A[f"eps_exp_{tag}_line"][warm] / A[f"true_avg_{tag}_line"][warm]
        say(f"  exposure-ratio / true-average (line), warm: {q.min():.4f} to {q.max():.4f}, median {np.median(q):.4f}")
    say("\n  thesis: 45 of 448, worst 0.1748 at heat [15,3] (100 us); 24 of 448, worst 0.1533 (506 us)")
    sc = A["sign_change_line"]
    memb = warm & (A["true_avg_100us_line"] > THRESHOLD)
    say(f"  signed error changes sign during the rise (sign(eps_step) != sign(eps_plateau), line): "
        f"{int(sc.sum())} of 784 pairs, {int((sc & warm).sum())} warm, {int((sc & memb).sum())} census members; "
        f"largest eps_plateau among them {A['eps_plateau_line'][sc].max():.4f}")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "trajectory_census"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
               f"# L_grid sha256 {sha256(lp)}", f"# S_grid sha256 {sha256(sp)}",
               f"# state_index sha256 {sha256(ctx.state_index_path)}",
               f"# radiative_rates sha256 {sha256(btr.DATA_RAD)}",
               f"# frac {FRAC}, window k {WIN_K}, TOL_TRACK {TOL_TRACK}, threshold {THRESHOLD}, "
               f"Te floor {TE_FLOOR}, samples/decade {ppd}"]
        cols = sorted(set().union(*[r.keys() for r in rows]), key=lambda c: list(rows[0].keys()).index(c) if c in rows[0] else 999)
        with (out / "trajectory_census.csv").open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        with (out / "trajectory_census_summary.csv").open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(census[0].keys())); w.writeheader(); w.writerows(census)
        (out / "trajectory_census.txt").write_text("\n".join(out_lines) + "\n")
        say(f"\nwrote {out}/ ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
