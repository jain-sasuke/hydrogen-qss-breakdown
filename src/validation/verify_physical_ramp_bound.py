#!/usr/bin/env python
"""
verify_physical_ramp_bound.py
=============================
How far does an interpolated operator sit from the TRUE operator inside a grid
interval, and what does that do to the ramp-plateau result?

WHY THIS EXISTS
---------------
Chapter 5 sec 5.8.2 now says: the ramp test of verify_ramp_plateau.py is an
operator-ramp sensitivity test, "because no rate matrix exists at intermediate
temperatures inside one interval; the operator along a true temperature ramp,
L[Te(t)], was not computed". Rate coefficients are stored only on the 50-point
Te grid (K_exc_full.npy is (43,43,50); assemble_cr_matrix.build_L takes a Te
INDEX). The stored plateau fractions at [15,3] are 0.9995, 0.9954, 0.9555,
0.6469 (linear operator ramp) and 0.6576 (log-linear at t_ramp = tau_slow).
This script measures, from stamped data only, the distance between an
interpolated operator and the true operator, and what that distance does to
the ramp result.

METHOD (part B, mandatory)
--------------------------
The canonical L_grid holds TRUE operators at every node. A two-interval
heating span i -> i+2 at fixed j has the true mid-node operator L[i+1,j] as
its physical intermediate, so an interpolant between L[i,j] and L[i+2,j] can
be tested against a true operator inside the span. The grid is logarithmic:
Delta ln Te = ln(10)/49 = 0.04699 per interval, so Te[i+1] sits at f = 1/2 in
ln Te and at f_lin = (Te[i+1]-Te[i])/(Te[i+2]-Te[i]) = 1/(r+1) = 0.48825
(r = 10^(1/49)) in Te.
  B1  operator level. For linear (weight f_lin) and log-linear (weight 1/2)
      interpolation between L[i,j], L[i+2,j] (S likewise), built with
      verify_ramp_plateau.make_interpolants: entrywise relative error
      X_interp / X_true - 1 over the nonzero entries of L[i+1,j], reported as
      median, 90th percentile and max of |e| (and the signed median), for the
      classes: all nonzero entries; the diagonal; the 1S column off the
      diagonal (rates out of the ground state); the ionisation loss per state
      (minus the column sum, which the assembly makes equal to K_ion ne);
      and the recombination source S. Then the error in tau_slow (= -1/lambda_0
      of the operator), u_CRE (= [-L^-1 S]_ground) and eps_plateau (of the
      one-interval heating step i -> i+1 whose post-step operator is either
      the true L[i+1,j] or the interpolant; formulae as verify_ramp_plateau.py)
      of the interpolated operator against the true one. At [15,3] (the ramp
      test point), [23,5] (benchmark), [0,0] (cold corner) and over all
      48 x 8 = 384 two-interval spans.
  B2  trajectory level, at [15,3]: the two-interval ramp Te(t) linear in time
      from Te[15] to Te[17] over t_ramp/tau_slow = 1e-2, 1e-1, 1 (tau_slow,
      tau_relax of the post-step operator L[17,3]), then held on L[17,3],
      integrated four ways with the same Te(t):
        (a)  piecewise log-linear through the true mid node L[16,3]
             (the closest available approximation to L[Te(t)]),
        (b)  plain log-linear between the ends, weight
             f_log(t) = ln(Te(t)/Te_i) / ln(Te_k/Te_i),
        (c)  plain linear between the ends, weight f = t/t_ramp
             (= (Te(t)-Te_i)/(Te_k-Te_i) for a Te-linear ramp),
        (a') piecewise linear through the true mid node (secondary: the
             interpolation-family sensitivity once the mid node is known).
      Plateau fraction = eps_CRE(t_ramp + 30 tau_relax) / eps_CRE^step(30
      tau_relax), the "plateau" column of verify_ramp_plateau.py, shell
      observable n3/n4; the "max" column is carried along. Differences
      (a)-(b), (a)-(c), (b)-(c), (a)-(a').
  B3  one-interval bound. Both interpolants are second-order (error ~ h^2 in
      the span), so the one-interval error is ESTIMATED as one quarter of
      the measured two-interval error. This is an estimate, not a measurement,
      and is labelled so wherever printed.

INTEGRATION, OBSERVABLE, WINDOW
-------------------------------
make_interpolants, spectrum and check_generator are imported from
verify_ramp_plateau.py. Its integrate() hard-codes f = t/t_ramp between two
operators, so integrate_path() below is a copy of it that takes the operator
path L(t), S(t) as callables and allows the ramp segment to be split at the
mid-node time (Radau, rtol 1e-10, atol 1e-12 max|n_old|, max_step
t_ramp/50, hold in segments ending exactly at t_ws, t_we, t_end, readouts
from step endpoints, expm cross-check after the ramp). Gate G1b proves the
copy reproduces the original on the original problem. Observable: shell
n3/n4 only (the line observable of verify_ramp_plateau.py differs from it in
the fifth digit and is not what chapter 5 quotes).

PART A (optional) -- NOT FEASIBLE WITHOUT EDITING MODULES; SKIPPED
-------------------------------------------------------------------
compute_K_CCC.py has no function that returns the table: the Maxwell-average
loop over all CCC transitions and the np.save calls to
data/processed/collisions/ccc/ execute at module import (lines ~150-206), so
importing it regenerates pipeline data. compute_K_TICS.compute_K_TICS,
ionization_rates.assemble_ionization_rates,
recombination_rates.compute_recombination_rates and
assemble_K_exc.assemble_K_exc bind the module-level 50-point TE_GRID, take no
Te argument, read (.,50)-shaped tables and write files. Only
compute_lmix.compute_K_lmix(te_grid=...) accepts an arbitrary Te. Building the
true L at interior temperatures therefore needs either module edits or a
re-implementation of the whole rate assembly from the primitive integrators,
which this brief forbids. Part A is not run.

GATES (the run stops if either fails)
-------------------------------------
G1   With verify_ramp_plateau.integrate itself, the one-interval ramp at
     [15,3] reproduces validation/ramp_plateau/ramp_plateau.csv (shell rows,
     point "defended maximum"): tau_slow, tau_relax, eps_plateau_step and
     plateau_ratio for step, 1e-3, 1e-2, 1e-1, 1 tau_slow (linear) and
     1 tau_slow (log-linear), to 1e-8 relative; the CSV's L_grid / S_grid
     sha256 match the files on disk.
G1b  integrate_path with the f = t/t_ramp linear path and no interior break
     reproduces integrate() on the 1 tau_slow one-interval ramp to 1e-10 in
     the window-start state.

PREDICTIONS (written 21 Sep 2026 before the first run)
------------------------------------------------------
Hand model: Arrhenius entry K ~ exp(-chi/Te). Linear-in-Te interpolation
error at the mid node ~ +(chi/Te^2)(chi/Te^2 - 2/Te)|_mid Te_i^2 r (r-1)^2/2
(over-estimate, convex); log-linear error ~ -(chi/Te_mid) h^2/2 with
h = 0.04699 (under-estimate, ln K concave in ln Te). So the true entry lies
BETWEEN the two interpolants, and the two errors have opposite signs.
  [15,3] Te 2.024 -> mid 2.121: 1S->2P (chi 10.20 eV) linear +1.5 %,
         log-linear -0.53 %; 1S ionisation (13.6 eV) linear +3.1 %,
         log-linear -0.71 %.
  [23,5] Te 2.947 -> mid 3.089: 1S->2P +0.47 % / -0.36 %;
         ionisation +1.2 % / -0.49 %.
  [0,0]  Te 1.000 -> mid 1.048: 1S->2P +8 % / -1.1 %;
         ionisation +16 % / -1.4 %.
  Radiative parts of L are Te-independent and are reproduced exactly;
  de-excitation entries vary slowly; so the all-entry median is well below
  1 % for both interpolants and the max is set by the 1S column / ionisation.
P1  (task) log-linear beats linear at the mid node in every entry class
    (median, p90 and max of |e|), and its all-entry median is below 2 % for
    the two-interval span, at the three named points and grid-wide (median
    of the per-span medians). Hand: grid-wide max |e| ~ 16 % linear (cold
    corner 1S ionisation), ~ 1.5 % log-linear.
    tau_slow and u_CRE: set by the 1S loss, so their errors are minus the
    1S-column errors: linear -1 to -3 % at [15,3], about -1 % at [23,5],
    -8 to -15 % at [0,0]; log-linear +0.4 to +1.4 %. eps_plateau: a ratio of
    ratios, expected below 1 % at [15,3]; no strong prediction.
P2  (task) |(a)-(b)| < |(b)-(c)| at all three t_ramp/tau_slow. Hand: (a)
    lies between (b) and (c), closer to (b): |a-b|/|b-c| ~ 0.2 to 0.4. At
    De = 1 the two-interval (b)-(c) ~ 4 x 0.0107 ~ +0.04 (log-linear above
    linear, as stored), (a)-(b) ~ -0.01, (a)-(c) ~ +0.03.
P3  (task) the one-interval scaled estimate at De = 1 is below 0.5 % of the
    plateau fraction. Read for each interpolant: P3b |a-b|/4 / plateau(a)
    < 0.005 (hand: ~0.4 %, holds, borderline); P3c |a-c|/4 / plateau(a)
    < 0.005 (hand: ~1.2 %, expected to FAIL). Both are reported.
    Secondary: (a)-(a') at De = 1 (two consecutive one-interval spans) is of
    the order of the stored one-interval 0.6576 - 0.6469 = +0.0107.

REFUTING OBSERVATION
--------------------
(a) falls outside the interval spanned by (b) and (c) by more than that
interval's width at any of the three ratios. The two operator ramps would
then not bracket the physical ramp and chapter 5's "insensitive to the two
operator ramps tested" could not be extended.

POST-HOC ADDITIONS (21 Sep 2026, after the first run; no prediction or
threshold above was changed)
-----------------------------------------------------------------------
a)  The "ion" class is the column-sum residual -sum_a L_ab = K_ion,b ne. A
    linear interpolant preserves column sums exactly, so its ion error is the
    linear interpolation error of K_ion itself; an entrywise geometric
    interpolant does not, and for excited states, where ionisation is a small
    residual of large collisional entries, a 0.05 % entry error becomes a
    10-50 % error of the residual. The first run showed exactly that (P1
    fails on this class everywhere). Added: the population-weighted net
    ionisation rate sum_b K_ion,b ne n_b / sum_b n_b at the true CRE
    populations of the mid node, per interpolant, so the reader can see
    whether the residual error matters in the net. Reported, not a test.
b)  B3 scaling check. The (b)-(c) gap is the one quantity measured at BOTH
    spans: two intervals (this run) and one interval (ramp_plateau.csv,
    reproduced by G1). Its ratio is printed; the requested quarter-scaled
    estimate is kept and an empirically scaled estimate
    (|a-b|, |a-c| times the measured ratio) is printed beside it. Both are
    estimates. P3 keeps its original quarter definition.

CONVERGENCE (required)
----------------------
Every B2 trajectory is recomputed at rtol 1e-8 and the relative change of the
plateau fraction, the max column and the differences is printed and stored
(conv_*). Variant (c) is also run without the interior break at the mid-node
time, to show the break itself changes nothing.

OUTPUTS (with --write)
----------------------
validation/physical_ramp_bound/physical_ramp_bound.txt              this log
validation/physical_ramp_bound/physical_ramp_bound_operator.csv     B1, 384 spans x 2 interpolants
validation/physical_ramp_bound/physical_ramp_bound_trajectory.csv   B2, per ratio / variant / rtol
Read-only otherwise.
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
from scipy.integrate import solve_ivp
from scipy.linalg import expm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext                                   # noqa: E402
import verify_ramp_plateau as vrp                                  # noqa: E402

ROOT = vrp.ROOT
RAMP_POINT = ("defended maximum", 2.024, 1.931e13, (15, 3))
NAMED = {"ramp point [15,3]": (15, 3), "benchmark [23,5]": (23, 5), "cold corner [0,0]": (0, 0)}
RATIOS = (1e-2, 1e-1, 1.0)
G1_TOL = 1e-8
G1B_TOL = 1e-10
VARIANTS = (("a", "piecewise log-linear via true mid node"),
            ("b", "log-linear between ends, f_log(t)"),
            ("c", "linear between ends, f = t/t_ramp"),
            ("a_lin", "piecewise linear via true mid node (secondary)"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# integrator: copy of verify_ramp_plateau.integrate with a general operator
# path L_of(t), b_of(t) on [0, td] and optional interior ramp breaks
# --------------------------------------------------------------------------
def integrate_path(L_of, b_of, L1, b1, n_init, td, ramp_breaks, breakpoints, rtol, atol):
    nfev = 0
    ramp_segs = []
    if td > 0:
        edges = [0.0] + sorted(set(float(x) for x in ramp_breaks if 0.0 < x < td)) + [td]
        t0, n0 = 0.0, n_init
        for t1 in edges[1:]:
            sol = solve_ivp(lambda t, n: L_of(t) @ n + b_of(t), [t0, t1], n0,
                            method="Radau", jac=lambda t, n: L_of(t),
                            rtol=rtol, atol=atol, dense_output=True,
                            max_step=td / 50.0)
            if not sol.success:
                raise RuntimeError(f"ramp segment [{t0:.3e}, {t1:.3e}] failed: {sol.message}")
            if sol.t[-1] != t1:
                raise RuntimeError(f"ramp segment did not end at {t1:.6e}: {sol.t[-1]:.6e}")
            ramp_segs.append((t0, t1, sol))
            nfev += sol.nfev
            t0, n0 = t1, sol.y[:, -1]
        n_td = n0.copy()
    else:
        n_td = n_init.copy()

    bps = sorted(set(float(b) for b in breakpoints if b > td))
    segs = []
    at = {}
    t0, n0 = td, n_td
    for t1 in bps:
        sol = solve_ivp(lambda t, n: L1 @ n + b1, [t0, t1], n0,
                        method="Radau", jac=lambda t, n: L1,
                        rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"hold segment [{t0:.3e}, {t1:.3e}] failed: {sol.message}")
        if sol.t[-1] != t1:
            raise RuntimeError(f"hold segment did not end at {t1:.6e}: {sol.t[-1]:.6e}")
        segs.append((t0, t1, sol))
        at[t1] = sol.y[:, -1].copy()
        nfev += sol.nfev
        t0, n0 = t1, sol.y[:, -1]

    def n_of(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty((len(n_init), len(t)))
        rest = np.ones(len(t), dtype=bool)
        for (a_, b_, sol) in ramp_segs:
            sel = rest & (t >= a_) & (t < b_)
            if sel.any():
                out[:, sel] = sol.sol(t[sel])
                rest &= ~sel
        for (a_, b_, sol) in segs:
            sel = rest & (t >= a_) & (t <= b_)
            if sel.any():
                out[:, sel] = sol.sol(t[sel])
                rest &= ~sel
        if rest.any():
            raise RuntimeError(f"times outside integrated range: {t[rest]}")
        return out

    return n_of, n_td, at, nfev


def weight(Te, Ta, Tb, kind):
    if kind == "linear":
        return (Te - Ta) / (Tb - Ta)
    return np.log(Te / Ta) / np.log(Tb / Ta)


def make_path(variant, Li, Lm, Lk, bi, bm, bk, Te_i, Te_m, Te_k, td):
    """Operator path L(t), S(t) on [0, td] for Te(t) = Te_i + (Te_k - Te_i) t/td."""
    def s_of(t):
        return min(max(t / td, 0.0), 1.0) if td > 0 else 1.0

    def Te_of(t):
        return Te_i + (Te_k - Te_i) * s_of(t)

    if variant == "c":
        La, ba = vrp.make_interpolants(Li, Lk, bi, bk, "linear")
        return (lambda t: La(s_of(t))), (lambda t: ba(s_of(t)))
    if variant == "b":
        La, ba = vrp.make_interpolants(Li, Lk, bi, bk, "loglinear")
        return (lambda t: La(weight(Te_of(t), Te_i, Te_k, "loglinear"))), \
               (lambda t: ba(weight(Te_of(t), Te_i, Te_k, "loglinear")))
    kind = {"a": "loglinear", "a_lin": "linear"}[variant]
    L1a, b1a = vrp.make_interpolants(Li, Lm, bi, bm, kind)
    L2a, b2a = vrp.make_interpolants(Lm, Lk, bm, bk, kind)

    def L_of(t):
        Te = Te_of(t)
        if Te <= Te_m:
            return L1a(weight(Te, Te_i, Te_m, kind))
        return L2a(weight(Te, Te_m, Te_k, kind))

    def b_of(t):
        Te = Te_of(t)
        if Te <= Te_m:
            return b1a(weight(Te, Te_i, Te_m, kind))
        return b2a(weight(Te, Te_m, Te_k, kind))

    return L_of, b_of


# --------------------------------------------------------------------------
# B1 helpers
# --------------------------------------------------------------------------
def err_stats(Xt, Xi, mask):
    if mask.sum() == 0:
        return dict(n=0, med=np.nan, p90=np.nan, max=np.nan, med_signed=np.nan, n_over=0, n_under=0)
    e = Xi[mask] / Xt[mask] - 1.0
    ae = np.abs(e)
    return dict(n=int(mask.sum()), med=float(np.median(ae)), p90=float(np.percentile(ae, 90)),
                max=float(ae.max()), med_signed=float(np.median(e)),
                n_over=int((e > 0).sum()), n_under=int((e < 0).sum()))


def derived(L0, b0, Lp, bp, g, E, N3, N4):
    tau_slow, tau_relax = vrp.spectrum(Lp)
    n_old = np.linalg.solve(L0, -b0)
    n_new = np.linalg.solve(Lp, -bp)
    LEE = Lp[np.ix_(E, E)]
    LEg = Lp[np.ix_(E, [g])].ravel()
    nPE = np.zeros(len(b0))
    nPE[E] = np.linalg.solve(LEE, -(bp[E] + LEg * n_old[g]))
    nPE[g] = n_old[g]
    R_new = n_new[N3].sum() / n_new[N4].sum()
    R_pe = nPE[N3].sum() / nPE[N4].sum()
    return tau_slow, tau_relax, float(n_new[g]), abs(R_pe / R_new - 1.0)


CLASSES = ("all", "diag", "col1s", "ion", "rec")


def span_errors(L, S, i, j, te, g, E, N3, N4):
    """B1 at one two-interval span i -> i+2 at fixed j. Returns dict per interpolant."""
    Li, Lm, Lk = L[i, j], L[i + 1, j], L[i + 2, j]
    bi, bm, bk = S[i, j], S[i + 1, j], S[i + 2, j]
    f_lin = (te[i + 1] - te[i]) / (te[i + 2] - te[i])
    f_log = np.log(te[i + 1] / te[i]) / np.log(te[i + 2] / te[i])
    n = L.shape[-1]
    eye = np.eye(n, dtype=bool)
    nz = Lm != 0
    masks = {"all": nz, "diag": nz & eye, "col1s": nz & ~eye & (np.arange(n)[None, :] == g)}
    ion_t = -Lm.sum(axis=0)
    ts, tr, ut, et = derived(Li, bi, Lm, bm, g, E, N3, N4)
    out = {"truth": dict(tau_slow=ts, tau_relax=tr, u=ut, eps=et, f_lin=f_lin, f_log=f_log,
                         n_nonzero=int(nz.sum()),
                         pattern_mid_vs_lo=int((nz != (Li != 0)).sum()),
                         pattern_mid_vs_hi=int((nz != (Lk != 0)).sum()))}
    for kind, f in (("linear", f_lin), ("loglinear", f_log)):
        try:
            La, ba = vrp.make_interpolants(Li, Lk, bi, bk, kind)
        except RuntimeError as ex:
            out[kind] = dict(defined=False, reason=str(ex))
            continue
        Lp, bp = La(f), ba(f)
        r = dict(defined=True, f=f)
        for c in ("all", "diag", "col1s"):
            r[c] = err_stats(Lm, Lp, masks[c])
        r["ion"] = err_stats(ion_t, -Lp.sum(axis=0), ion_t != 0)
        r["rec"] = err_stats(bm, bp, bm != 0)
        # post-hoc (a): population-weighted net ionisation rate at the TRUE mid-node CRE populations
        n_cre = np.linalg.solve(Lm, -bm)
        ion_w_true = float(ion_t @ n_cre / n_cre.sum())
        ion_w_interp = float((-Lp.sum(axis=0)) @ n_cre / n_cre.sum())
        r["ion_w_true"] = ion_w_true
        r["e_ion_w"] = ion_w_interp / ion_w_true - 1.0
        r["e_ion_1s"] = float(-Lp.sum(axis=0)[g] / ion_t[g] - 1.0)
        r["n_interp_nz_truth_zero"] = int(((Lp != 0) & ~nz).sum())
        tsi, tri, ui, ei = derived(Li, bi, Lp, bp, g, E, N3, N4)
        r.update(tau_slow=tsi, tau_relax=tri, u=ui, eps=ei,
                 e_tau_slow=tsi / ts - 1.0, e_tau_relax=tri / tr - 1.0,
                 e_u=ui / ut - 1.0, e_eps=ei / et - 1.0)
        out[kind] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    t_wall0 = time.time()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = int(ctx.ground_index)
    nv = np.asarray(ctx.n_values)
    nT, nN, nS, _ = L.shape
    E = np.array([q for q in range(nS) if q != g], dtype=int)
    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")

    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not sp.exists():
        raise FileNotFoundError(f"missing source vector {sp}")
    S = np.load(sp)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    rp_p = ROOT / "validation/ramp_plateau/ramp_plateau.csv"
    if not rp_p.exists():
        raise FileNotFoundError(f"missing {rp_p}; G1 cannot run")
    rp_rows = vrp.read_csv_with_comments(rp_p)
    with rp_p.open() as fh:
        rp_hdr = [ln.rstrip("\n") for ln in fh if ln.startswith("#")]
    for want, line in ((sha256(lp), rp_hdr[2]), (sha256(sp), rp_hdr[3])):
        if want not in line:
            raise RuntimeError(f"ramp_plateau.csv was built on different data:\n  {line}\n"
                               f"  current sha256 {want}")

    # grid facts, from the loaded grid, not assumed
    dln = np.diff(np.log(te))
    if not np.allclose(dln, dln[0], rtol=1e-10, atol=0):
        raise RuntimeError("Te grid is not logarithmic; the f = 1/2 mid-node statement does not hold")
    h = float(dln[0])
    r_grid = float(np.exp(h))

    out_lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
           f"# interpreter {sys.executable}  numpy {np.__version__}  scipy {__import__('scipy').__version__}",
           f"# L_grid sha256 {sha256(lp)}", f"# S_grid sha256 {sha256(sp)}",
           f"# state_index sha256 {sha256(ctx.state_index_path)}",
           f"# ramp_plateau.csv sha256 {sha256(rp_p)}  (G1 gate)  {rp_hdr[0]}",
           f"# integrator: copy of verify_ramp_plateau.integrate (Radau, rtol {vrp.RTOL_MAIN}, conv rerun "
           f"{vrp.RTOL_CONV}, atol {vrp.ATOL_SCALE}*max|n_old|, max_step t_ramp/50, window t_ramp+{vrp.WIN_LO:g} tau_relax)",
           f"# grid: {nT} Te x {nN} ne, Delta ln Te = {h:.6f} per interval, r = {r_grid:.6f}, "
           f"f_lin(mid) = {1/(r_grid+1):.6f}, f_log(mid) = 0.5"]
    say("=" * 78)
    say("PHYSICAL RAMP BOUND: interpolated operator vs the true mid-node operator")
    for s_ in hdr:
        say(s_)
    say(ctx.describe())
    say("=" * 78)

    # ------------------------------------------------------------------ G1
    say()
    say("G1  reproduce validation/ramp_plateau/ramp_plateau.csv at [15,3] with verify_ramp_plateau.integrate")
    pname, te_req, ne_req, expect_ij = RAMP_POINT
    i, j = ctx.nearest_point(te_req, ne_req)
    if (i, j) != expect_ij:
        raise RuntimeError(f"nearest_point({te_req}, {ne_req}) = [{i},{j}], expected {expect_ij}")
    k1 = i + 1
    L0, L1, b0, b1 = L[i, j], L[k1, j], S[i, j], S[k1, j]
    tau_slow1, tau_relax1 = vrp.spectrum(L1)
    n_old = np.linalg.solve(L0, -b0)
    n_new1 = np.linalg.solve(L1, -b1)
    R_new1 = n_new1[N3].sum() / n_new1[N4].sum()
    LEE = L1[np.ix_(E, E)]
    LEg = L1[np.ix_(E, [g])].ravel()
    nPE = np.zeros(nS)
    nPE[E] = np.linalg.solve(LEE, -(b1[E] + LEg * n_old[g]))
    nPE[g] = n_old[g]
    eps_plateau1 = abs((nPE[N3].sum() / nPE[N4].sum()) / R_new1 - 1.0)
    atol1 = vrp.ATOL_SCALE * np.abs(n_old).max()

    def eps_of(n, R_new):
        return float(abs((n[N3].sum() / n[N4].sum()) / R_new - 1.0))

    def readout_original(td, interp):
        t_end = td + 10.0 * tau_slow1
        t_ws = td + vrp.WIN_LO * tau_relax1
        t_we = tau_slow1 / vrp.WIN_HI
        n_of, n_td, at, nfev = vrp.integrate(L0, L1, b0, b1, n_old, td, [t_ws, t_we, t_end],
                                             vrp.RTOL_MAIN, atol1, interp)
        return eps_of(at[t_ws], R_new1), at[t_ws], nfev

    csv_rows = [q for q in rp_rows if q["point"] == pname and q["observable"] == "shell"]
    if not csv_rows:
        raise RuntimeError("ramp_plateau.csv has no shell rows for the ramp point")
    c0 = csv_rows[0]
    g1_worst = 0.0
    for mine, key in ((tau_slow1, "tau_slow"), (tau_relax1, "tau_relax"), (eps_plateau1, "eps_plateau_step")):
        rel = abs(mine / float(c0[key]) - 1.0)
        g1_worst = max(g1_worst, rel)
        say(f"    {key:17s} mine {mine:.9e}  csv {float(c0[key]):.9e}  rel {rel:.2e}")
    eps_ws_step, _, nfev_s = readout_original(0.0, "linear")
    G1_CASES = [("step", 0.0, "none"), ("0.001 tau_slow", 1e-3, "linear"), ("0.01 tau_slow", 1e-2, "linear"),
                ("0.1 tau_slow", 1e-1, "linear"), ("1 tau_slow", 1.0, "linear"), ("1 tau_slow", 1.0, "loglinear")]
    g1_plateau = {}
    for cname, ratio_, interp in G1_CASES:
        row = [q for q in csv_rows if q["ramp"] == cname and q["interp"] == interp]
        if len(row) != 1:
            raise RuntimeError(f"ramp_plateau.csv: expected one shell row ({cname}, {interp}), found {len(row)}")
        row = row[0]
        if ratio_ == 0.0:
            eps_ws, plateau = eps_ws_step, 1.0
        else:
            eps_ws, _, _ = readout_original(ratio_ * tau_slow1, interp)
            plateau = eps_ws / eps_ws_step
        g1_plateau[(ratio_, interp)] = plateau
        rel_e = abs(eps_ws / float(row["eps_at_window_start"]) - 1.0)
        rel_p = abs(plateau / float(row["plateau_ratio"]) - 1.0)
        g1_worst = max(g1_worst, rel_e, rel_p)
        say(f"    {cname:>14s} {interp:>9s}  eps(ws) {eps_ws:.9e} (csv {float(row['eps_at_window_start']):.9e}, "
            f"rel {rel_e:.2e})   plateau {plateau:.6f} (csv {float(row['plateau_ratio']):.6f}, rel {rel_p:.2e})")
    say(f"    worst relative difference against the CSV: {g1_worst:.2e}  (tolerance {G1_TOL:g})")
    if g1_worst > G1_TOL:
        raise AssertionError("G1 FAILED: the stored ramp_plateau values do not reproduce; do not read on")
    say("    G1 REPRODUCED: 0.9995 / 0.9954 / 0.9555 / 0.6469 linear, 0.6576 log-linear at De = 1  ->  "
        + ", ".join(f"{g1_plateau[(r_, it)]:.4f}" for (_, r_, it) in G1_CASES[1:]))

    # ------------------------------------------------------------------ G1b
    td1 = 1.0 * tau_slow1
    t_ws1 = td1 + vrp.WIN_LO * tau_relax1
    _, n_ws_orig, _ = readout_original(td1, "linear")
    La1, ba1 = vrp.make_interpolants(L0, L1, b0, b1, "linear")
    L_of1 = lambda t: La1(min(max(t / td1, 0.0), 1.0))            # noqa: E731
    b_of1 = lambda t: ba1(min(max(t / td1, 0.0), 1.0))            # noqa: E731
    _, _, at1, _ = integrate_path(L_of1, b_of1, L1, b1, n_old, td1, [],
                                  [t_ws1, tau_slow1 / vrp.WIN_HI, td1 + 10 * tau_slow1], vrp.RTOL_MAIN, atol1)
    g1b = float(np.abs(at1[t_ws1] / n_ws_orig - 1.0).max())
    say(f"G1b integrate_path (no interior break) vs verify_ramp_plateau.integrate, 1 tau_slow linear, "
        f"max entrywise rel diff of n(t_ws): {g1b:.2e}  (tolerance {G1B_TOL:g})")
    if g1b > G1B_TOL:
        raise AssertionError("G1b FAILED: the integrator copy does not reproduce the original")
    say("    G1b PASSED")

    # ------------------------------------------------------------------ B1
    say()
    say("=" * 78)
    say("B1  OPERATOR-LEVEL ERROR at the true mid node of a two-interval span i -> i+2")
    say("    e = X_interp / X_true - 1 over the nonzero entries of L[i+1,j]; classes: all, diagonal,")
    say("    1S column (off-diagonal), ionisation (= -column sum), recombination (S).")
    say("    linear evaluated at f_lin = (Te[i+1]-Te[i])/(Te[i+2]-Te[i]); log-linear at f = 1/2 in ln Te.")
    say("=" * 78)
    op_rows: list[dict] = []
    grid = {}
    for ii in range(nT - 2):
        for jj in range(nN):
            grid[(ii, jj)] = span_errors(L, S, ii, jj, te, g, E, N3, N4)
    undefined = [(key, kind) for key, res in grid.items() for kind in ("linear", "loglinear")
                 if not res[kind]["defined"]]
    pat = [(key, res["truth"]["pattern_mid_vs_lo"], res["truth"]["pattern_mid_vs_hi"])
           for key, res in grid.items() if res["truth"]["pattern_mid_vs_lo"] or res["truth"]["pattern_mid_vs_hi"]]
    say(f"  spans: {len(grid)} (i = 0..{nT-3}, j = 0..{nN-1});  interpolant undefined (sparsity/sign mismatch "
        f"between the ends): {len(undefined)}" + (f"  {undefined}" if undefined else ""))
    say(f"  spans where the mid-node sparsity pattern differs from an end: {len(pat)}"
        + (f"  {pat[:10]}{' ...' if len(pat) > 10 else ''}" if pat else ""))

    def fmt_stats(st):
        return (f"n {st['n']:4d}  median {100*st['med']:7.3f} %  p90 {100*st['p90']:7.3f} %  "
                f"max {100*st['max']:7.3f} %  signed median {100*st['med_signed']:+8.3f} %  "
                f"(over {st['n_over']}, under {st['n_under']})")

    for name, (ii, jj) in NAMED.items():
        res = grid[(ii, jj)]
        tr = res["truth"]
        say()
        say(f"  {name}: span [{ii},{jj}] -> [{ii+2},{jj}], Te {te[ii]:.4f} -> {te[ii+1]:.4f} (true mid) -> "
            f"{te[ii+2]:.4f} eV, ne {ne[jj]:.3e};  f_lin {tr['f_lin']:.5f}, f_log {tr['f_log']:.5f}; "
            f"nonzero entries of L[mid] {tr['n_nonzero']}")
        say(f"    true mid node: tau_slow {tr['tau_slow']:.6e} s  tau_relax {tr['tau_relax']:.6e} s  "
            f"u_CRE {tr['u']:.6e}  eps_plateau(step i->i+1) {tr['eps']:.6f}")
        for kind in ("linear", "loglinear"):
            r = res[kind]
            if not r["defined"]:
                say(f"    {kind:9s}: UNDEFINED -- {r['reason']}")
                continue
            say(f"    {kind:9s} (f = {r['f']:.5f}); interpolant nonzero where truth is zero: {r['n_interp_nz_truth_zero']}")
            for c in CLASSES:
                say(f"      {c:5s} {fmt_stats(r[c])}")
            say(f"      post-hoc (a): 1S ionisation rate error {100*r['e_ion_1s']:+.3f} %;  population-weighted net "
                f"ionisation rate at true CRE populations {r['ion_w_true']:.4e} s^-1, error {100*r['e_ion_w']:+.3f} %")
            say(f"      derived: tau_slow {r['tau_slow']:.6e} ({100*r['e_tau_slow']:+.3f} %)  "
                f"tau_relax ({100*r['e_tau_relax']:+.3f} %)  u_CRE {r['u']:.6e} ({100*r['e_u']:+.3f} %)  "
                f"eps_plateau {r['eps']:.6f} ({100*r['e_eps']:+.3f} %)")

    # grid-wide distribution
    say()
    say("  GRID-WIDE (all defined spans): distribution over spans of per-span statistics")
    say(f"  {'interp':9s} {'class':5s} {'stat':6s} {'median':>10s} {'p90':>10s} {'max':>10s}  worst span")
    dist = {}
    for kind in ("linear", "loglinear"):
        keys = [key for key, res in grid.items() if res[kind]["defined"]]
        for c in CLASSES:
            for stat in ("med", "p90", "max"):
                v = np.array([grid[key][kind][c][stat] for key in keys])
                kw = keys[int(np.nanargmax(v))]
                dist[(kind, c, stat)] = (float(np.nanmedian(v)), float(np.nanpercentile(v, 90)), float(np.nanmax(v)), kw)
                say(f"  {kind:9s} {c:5s} {stat:6s} {100*np.nanmedian(v):9.3f}% {100*np.nanpercentile(v, 90):9.3f}% "
                    f"{100*np.nanmax(v):9.3f}%  [{kw[0]},{kw[1]}] Te {te[kw[0]]:.3f}->{te[kw[0]+2]:.3f}")
        for q in ("e_tau_slow", "e_u", "e_eps"):
            v = np.array([grid[key][kind][q] for key in keys])
            kw = keys[int(np.argmax(np.abs(v)))]
            dist[(kind, q)] = (float(np.median(v)), float(np.median(np.abs(v))), float(np.abs(v).max()), kw,
                               float(v.min()), float(v.max()))
            say(f"  {kind:9s} {q:10s} signed median {100*np.median(v):+8.3f} %  median |.| {100*np.median(np.abs(v)):7.3f} %  "
                f"max |.| {100*np.abs(v).max():7.3f} %  range {100*v.min():+.3f} .. {100*v.max():+.3f} %  "
                f"worst [{kw[0]},{kw[1]}]")
    for kind in ("linear", "loglinear"):
        keys = [key for key, res in grid.items() if res[kind]["defined"]]
        v = np.array([grid[key][kind]["e_ion_w"] for key in keys])
        v1 = np.array([grid[key][kind]["e_ion_1s"] for key in keys])
        say(f"  {kind:9s} post-hoc (a) net ionisation rate (population-weighted): signed median {100*np.median(v):+7.3f} %  "
            f"max |.| {100*np.abs(v).max():6.3f} %;   1S ionisation rate: signed median {100*np.median(v1):+7.3f} %  "
            f"max |.| {100*np.abs(v1).max():6.3f} %")
    say("  reading note (post-hoc a): the ion class is the column-sum residual; entrywise geometric interpolation does not")
    say("  preserve column sums, so for excited states (ionisation a small residual of large entries) its relative error is")
    say("  large although the entries themselves are accurate; the population-weighted line above is the net effect.")
    eps_true_min = min(res["truth"]["eps"] for res in grid.values())
    say(f"  smallest true eps_plateau over the spans: {eps_true_min:.3e} (relative errors of eps are well defined)")
    # spans where log-linear does NOT beat linear, per class and stat
    say()
    say("  P1 grid-wide: spans where log-linear |e| is NOT below linear |e| (count of 384), per class / stat")
    beats = {}
    for c in CLASSES:
        for stat in ("med", "p90", "max"):
            keys = [key for key, res in grid.items() if res["linear"]["defined"] and res["loglinear"]["defined"]]
            nb = sum(1 for key in keys if not (grid[key]["loglinear"][c][stat] < grid[key]["linear"][c][stat]))
            beats[(c, stat)] = nb
        say(f"    {c:5s} med {beats[(c,'med')]:3d}  p90 {beats[(c,'p90')]:3d}  max {beats[(c,'max')]:3d}")

    for (ii, jj), res in grid.items():
        for kind in ("linear", "loglinear"):
            r = res[kind]
            row = dict(i=ii, j=jj, Te_lo=float(te[ii]), Te_mid=float(te[ii + 1]), Te_hi=float(te[ii + 2]),
                       ne=float(ne[jj]), interp=kind, defined=r["defined"],
                       f=r.get("f", np.nan), n_nonzero_true=res["truth"]["n_nonzero"],
                       pattern_mid_vs_lo=res["truth"]["pattern_mid_vs_lo"],
                       pattern_mid_vs_hi=res["truth"]["pattern_mid_vs_hi"],
                       n_interp_nz_truth_zero=r.get("n_interp_nz_truth_zero", np.nan))
            for c in CLASSES:
                st = r.get(c, {})
                for stat in ("n", "med", "p90", "max", "med_signed"):
                    row[f"{c}_{stat}"] = st.get(stat, np.nan)
            row.update(tau_slow_true=res["truth"]["tau_slow"], tau_relax_true=res["truth"]["tau_relax"],
                       u_true=res["truth"]["u"], eps_plateau_true=res["truth"]["eps"],
                       tau_slow_interp=r.get("tau_slow", np.nan), u_interp=r.get("u", np.nan),
                       eps_plateau_interp=r.get("eps", np.nan),
                       e_tau_slow=r.get("e_tau_slow", np.nan), e_tau_relax=r.get("e_tau_relax", np.nan),
                       e_u=r.get("e_u", np.nan), e_eps=r.get("e_eps", np.nan),
                       e_ion_1s=r.get("e_ion_1s", np.nan), e_ion_net_weighted=r.get("e_ion_w", np.nan))
            op_rows.append(row)

    # ------------------------------------------------------------------ B2
    say()
    say("=" * 78)
    say("B2  TRAJECTORY LEVEL at [15,3]: two-interval ramp Te linear in t from Te[15] to Te[17], held on L[17,3]")
    say("=" * 78)
    im, ik = i + 1, i + 2
    Li, Lm, Lk = L[i, j], L[im, j], L[ik, j]
    bi, bm, bk = S[i, j], S[im, j], S[ik, j]
    Te_i, Te_m, Te_k = float(te[i]), float(te[im]), float(te[ik])
    tau_slow2, tau_relax2 = vrp.spectrum(Lk)
    n_new2 = np.linalg.solve(Lk, -bk)
    R_new2 = n_new2[N3].sum() / n_new2[N4].sum()
    LEE = Lk[np.ix_(E, E)]
    LEg = Lk[np.ix_(E, [g])].ravel()
    nPE2 = np.zeros(nS)
    nPE2[E] = np.linalg.solve(LEE, -(bk[E] + LEg * n_old[g]))
    nPE2[g] = n_old[g]
    eps_plateau2 = abs((nPE2[N3].sum() / nPE2[N4].sum()) / R_new2 - 1.0)
    eps_step2 = abs((n_old[N3].sum() / n_old[N4].sum()) / R_new2 - 1.0)
    s_mid = (Te_m - Te_i) / (Te_k - Te_i)
    say(f"  span [{i},{j}] -> [{ik},{j}] via true mid node [{im},{j}]: Te {Te_i:.4f} -> {Te_m:.4f} -> {Te_k:.4f} eV, "
        f"ne {ne[j]:.3e}; mid node at t/t_ramp = {s_mid:.5f}")
    say(f"  post-step operator L[{ik},{j}]: tau_slow {tau_slow2:.6e} s  tau_relax {tau_relax2:.6e} s  "
        f"M {tau_slow2/tau_relax2:.1f}   (one-interval, L[{k1},{j}]: tau_slow {tau_slow1:.6e}, tau_relax {tau_relax1:.6e})")
    say(f"  two-interval step: eps_step {eps_step2:.6f}  eps_plateau {eps_plateau2:.6f}   "
        f"(one-interval eps_plateau {eps_plateau1:.6f})")

    for v, desc in VARIANTS:
        L_of, b_of = make_path(v, Li, Lm, Lk, bi, bm, bk, Te_i, Te_m, Te_k, 1.0)
        wo, wc, wf = vrp.check_generator(lambda f: L_of(f), lambda f: b_of(f),
                                         fs=(0.0, 0.25, s_mid, 0.5, 0.75, 1.0))
        say(f"  generator check ({v:5s} {desc}): min off-diagonal {wo:.3e} (>= 0 asserted), "
            f"worst column sum {wc:.4e} s^-1 at t/t_ramp = {wf:.4f}")

    def run_case(variant, td, rtol, breaks=None):
        t_end = td + 10.0 * tau_slow2
        t_ws = td + vrp.WIN_LO * tau_relax2
        t_we = tau_slow2 / vrp.WIN_HI
        if td > 0:
            L_of, b_of = make_path(variant, Li, Lm, Lk, bi, bm, bk, Te_i, Te_m, Te_k, td)
            if breaks is None:
                breaks = [s_mid * td]
        else:
            L_of, b_of, breaks = (lambda t: Lk), (lambda t: bk), []
        n_of, n_td, at, nfev = integrate_path(L_of, b_of, Lk, bk, n_old, td, breaks,
                                              [t_ws, t_we, t_end], rtol, atol1)
        t_chk = td + np.logspace(np.log10(tau_relax2 * 1e-2), np.log10(10.0 * tau_slow2),
                                 vrp.N_EXPM_CHECK if td == 0 else vrp.N_EXPM_RAMP)
        worst = 0.0
        for t in t_chk:
            n_ex = n_new2 + expm(Lk * (t - td)) @ (n_td - n_new2)
            n_od = n_of(t)[:, 0]
            worst = max(worst, np.linalg.norm(n_od - n_ex) / np.linalg.norm(n_ex))
        if worst >= vrp.EXPM_TOL:
            raise RuntimeError(f"ODE disagrees with expm after the ramp ({variant}, td={td:.3e}): {worst:.3e}")
        t_post = td + np.logspace(np.log10(tau_relax2 * 1e-3), np.log10(10.0 * tau_slow2), 6000)
        ts = [[0.0], t_post, [t_ws, t_end]]
        if td > 0:
            ts.append(np.linspace(0.0, td, 2000))
        if t_we > td:
            ts.append([t_we])
        ts = np.unique(np.concatenate([np.asarray(x, dtype=float) for x in ts]))
        ns = n_of(ts)
        eps = np.abs((ns[N3].sum(axis=0) / ns[N4].sum(axis=0)) / R_new2 - 1.0)
        post = ts > td
        kmax = np.where(post)[0][np.argmax(eps[post])]
        return dict(eps_max_post=float(eps[kmax]), t_max=float(ts[kmax]),
                    eps_ws=eps_of(at[t_ws], R_new2), eps_we=eps_of(at[t_we], R_new2) if t_we > td else np.nan,
                    eps_end=eps_of(at[t_end], R_new2), eps_ramp_end=eps_of(n_td, R_new2),
                    ng_drift=abs(n_td[g] / n_old[g] - 1.0), expm_resid=worst, nfev=nfev, t_ws=t_ws)

    step2 = run_case("c", 0.0, vrp.RTOL_MAIN)
    step2c = run_case("c", 0.0, vrp.RTOL_CONV)
    say(f"  pure two-interval step: eps(ws) {step2['eps_ws']:.6e}  max/eps_plateau {step2['eps_max_post']/eps_plateau2:.4f}  "
        f"expm resid {step2['expm_resid']:.1e}  nfev {step2['nfev']}  (conv rerun rel change of eps(ws) "
        f"{abs(step2c['eps_ws']/step2['eps_ws']-1):.1e})")
    ng_total2 = abs(n_new2[g] / n_old[g] - 1.0)

    traj_rows: list[dict] = []
    P = {}
    say()
    say(f"  {'td/tau_slow':>11s} {'variant':>7s} {'plateau':>9s} {'conv':>8s} {'max/eps_pl':>10s} {'conv':>8s} "
        f"{'eps(ws)':>12s} {'eps(we)':>12s} {'eps(end)':>9s} {'ng drift/tot':>12s} {'expm':>8s} {'nfev':>6s}")
    for ratio_ in RATIOS:
        td = ratio_ * tau_slow2
        for v, desc in VARIANTS:
            rm = run_case(v, td, vrp.RTOL_MAIN)
            rc = run_case(v, td, vrp.RTOL_CONV)
            plateau = rm["eps_ws"] / step2["eps_ws"]
            plateau_c = rc["eps_ws"] / step2c["eps_ws"]
            mx = rm["eps_max_post"] / eps_plateau2
            mx_c = rc["eps_max_post"] / eps_plateau2
            P[(ratio_, v)] = dict(plateau=plateau, plateau_conv=plateau_c, max=mx, max_conv=mx_c, res=rm)
            say(f"  {ratio_:11.0e} {v:>7s} {plateau:9.6f} {abs(plateau_c/plateau-1):8.1e} {mx:10.6f} "
                f"{abs(mx_c/mx-1):8.1e} {rm['eps_ws']:12.6e} {rm['eps_we']:12.6e} {rm['eps_end']:9.2e} "
                f"{rm['ng_drift']/ng_total2:12.6f} {rm['expm_resid']:8.1e} {rm['nfev']:6d}")
            traj_rows.append(dict(
                point=pname, i=i, j=j, i_mid=im, k=ik, Te_i=Te_i, Te_mid=Te_m, Te_k=Te_k, ne=float(ne[j]),
                tau_slow=tau_slow2, tau_relax=tau_relax2, tau_ramp_over_tau_slow=ratio_, tau_ramp_s=td,
                variant=v, description=desc, observable="shell",
                eps_step=eps_step2, eps_plateau_step=eps_plateau2,
                eps_at_window_start=rm["eps_ws"], eps_at_window_start_step=step2["eps_ws"],
                plateau_ratio=plateau, ratio_max_to_plateau=mx, t_of_max=rm["t_max"],
                eps_at_ramp_end=rm["eps_ramp_end"], eps_at_window_end=rm["eps_we"], eps_at_t_end=rm["eps_end"],
                ng_drift_ramp=rm["ng_drift"], ng_drift_total=ng_total2, ng_drift_fraction=rm["ng_drift"] / ng_total2,
                expm_resid=rm["expm_resid"], nfev=rm["nfev"], nfev_conv=rc["nfev"],
                conv_plateau_ratio=abs(plateau_c / plateau - 1.0), conv_ratio_max=abs(mx_c / mx - 1.0),
                conv_eps_at_window_start=abs(rc["eps_ws"] / rm["eps_ws"] - 1.0),
                conv_ng_drift_ramp=abs(rc["ng_drift"] / rm["ng_drift"] - 1.0)))
        # numerics severity: (c) without the interior break
        rnb = run_case("c", td, vrp.RTOL_MAIN, breaks=[])
        say(f"  {'':11s} {'c/nobrk':>7s} {rnb['eps_ws']/step2['eps_ws']:9.6f}   [variant c without the mid-node break; "
            f"rel diff vs c {abs(rnb['eps_ws']/P[(ratio_, 'c')]['res']['eps_ws']-1):.1e}]")

    say()
    say("  DIFFERENCES of the plateau fraction (and of the max column), with the rtol 1e-8 rerun alongside")
    say(f"  {'td/tau_slow':>11s} {'(a)-(b)':>10s} {'conv':>10s} {'(a)-(c)':>10s} {'conv':>10s} {'(b)-(c)':>10s} "
        f"{'(a)-(a_lin)':>11s} {'|a-b|/|b-c|':>11s} {'bracket':>8s} {'max:(a)-(b)':>11s} {'max:(a)-(c)':>11s}")
    verdicts: list[tuple[str, object]] = []
    refuted = False
    for ratio_ in RATIOS:
        pa, pb, pc, pal = (P[(ratio_, v)]["plateau"] for v in ("a", "b", "c", "a_lin"))
        ca, cb, cc = (P[(ratio_, v)]["plateau_conv"] for v in ("a", "b", "c"))
        ma, mb, mc = (P[(ratio_, v)]["max"] for v in ("a", "b", "c"))
        lo, hi = min(pb, pc), max(pb, pc)
        width = hi - lo
        inside = lo <= pa <= hi
        excess = max(lo - pa, pa - hi, 0.0)
        fire = excess > width
        refuted |= fire
        say(f"  {ratio_:11.0e} {pa-pb:+10.6f} {ca-cb:+10.6f} {pa-pc:+10.6f} {ca-cc:+10.6f} {pb-pc:+10.6f} "
            f"{pa-pal:+11.6f} {abs(pa-pb)/abs(pb-pc) if pb != pc else np.nan:11.4f} "
            f"{'inside' if inside else f'out {excess:.1e}':>8s} {ma-mb:+11.6f} {ma-mc:+11.6f}")
        verdicts.append((f"P2 td/tau_slow = {ratio_:g}: |(a)-(b)| = {abs(pa-pb):.6f} < |(b)-(c)| = {abs(pb-pc):.6f}",
                         abs(pa - pb) < abs(pb - pc)))
        verdicts.append((f"REFUTER td/tau_slow = {ratio_:g}: (a) = {pa:.6f} vs [(b),(c)] = [{lo:.6f}, {hi:.6f}], "
                         f"excess outside {excess:.2e} <= width {width:.2e}", not fire))

    # ------------------------------------------------------------------ B3
    say()
    say("  B3  ONE-INTERVAL BOUND -- AN ESTIMATE, NOT A MEASUREMENT: second-order interpolants, error ~ span^2,")
    say("      so the one-interval distance is taken as one quarter of the measured two-interval distance.")
    say(f"  {'td/tau_slow':>11s} {'plateau(a)':>10s} {'|a-b|/4':>10s} {'% of (a)':>9s} {'|a-c|/4':>10s} {'% of (a)':>9s} "
        f"{'measured (a)-(a_lin)':>20s} {'stored 1-interval log-lin':>25s}")
    for ratio_ in RATIOS:
        pa, pb, pc, pal = (P[(ratio_, v)]["plateau"] for v in ("a", "b", "c", "a_lin"))
        eb, ec = abs(pa - pb) / 4.0, abs(pa - pc) / 4.0
        stored = ""
        if ratio_ == 1.0:
            stored = f"{g1_plateau[(1.0, 'loglinear')] - g1_plateau[(1.0, 'linear')]:+.6f}"
        elif ratio_ == 1e-2:
            row_l = [q for q in csv_rows if q["ramp"] == "0.01 tau_slow" and q["interp"] == "loglinear"][0]
            stored = f"{float(row_l['plateau_ratio']) - g1_plateau[(1e-2, 'linear')]:+.6f}"
        say(f"  {ratio_:11.0e} {pa:10.6f} {eb:10.6f} {100*eb/pa:8.3f}% {ec:10.6f} {100*ec/pa:8.3f}% "
            f"{pa-pal:+20.6f} {stored:>25s}   [estimate]")
        if ratio_ == 1.0:
            verdicts.append((f"P3b De = 1: estimate |a-b|/4 / plateau(a) = {100*eb/pa:.3f} % < 0.5 %  [estimate]",
                             eb / pa < 0.005))
            verdicts.append((f"P3c De = 1: estimate |a-c|/4 / plateau(a) = {100*ec/pa:.3f} % < 0.5 %  [estimate]",
                             ec / pa < 0.005))

    # post-hoc (b): scaling check of the (b)-(c) gap, measured at one interval (stored) and two intervals (here)
    say()
    say("  B3 POST-HOC (b): does the quarter rule hold for the PLATEAU FRACTION? The (b)-(c) gap is measured at both spans.")
    say(f"  {'td/tau_slow':>11s} {'(b)-(c) 2-int':>13s} {'(b)-(c) 1-int stored':>20s} {'ratio 1/2':>9s} "
        f"{'|a-b| x ratio':>13s} {'% of (a)':>9s} {'|a-c| x ratio':>13s} {'% of (a)':>9s}")
    stored_gap = {1.0: g1_plateau[(1.0, "loglinear")] - g1_plateau[(1.0, "linear")],
                  1e-2: float([q for q in csv_rows if q["ramp"] == "0.01 tau_slow" and q["interp"] == "loglinear"][0]
                              ["plateau_ratio"]) - g1_plateau[(1e-2, "linear")],
                  1e-1: float([q for q in csv_rows if q["ramp"] == "0.1 tau_slow" and q["interp"] == "loglinear"][0]
                              ["plateau_ratio"]) - float([q for q in csv_rows if q["ramp"] == "0.1 tau_slow"
                                                          and q["interp"] == "linear"][0]["plateau_ratio"])}
    for ratio_ in RATIOS:
        pa, pb, pc = (P[(ratio_, v)]["plateau"] for v in ("a", "b", "c"))
        gap2 = pb - pc
        gap1 = stored_gap[ratio_]
        rr = gap1 / gap2
        say(f"  {ratio_:11.0e} {gap2:+13.6f} {gap1:+20.6f} {rr:9.3f} {abs(pa-pb)*rr:13.6f} {100*abs(pa-pb)*rr/pa:8.3f}% "
            f"{abs(pa-pc)*rr:13.6f} {100*abs(pa-pc)*rr/pa:8.3f}%   [estimate; the quarter rule would give 0.250]")
    say("  reading (post-hoc b): the operator error scales as span^2 but the reservoir excursion that normalises the plateau")
    say("  fraction scales as span, so the plateau-fraction sensitivity scales about as span; the quarter rule is for")
    say("  operator-level quantities (B1). P3 keeps its quarter definition; the empirically scaled line is the better estimate.")

    # ------------------------------------------------------------------ P1 verdicts
    for name, (ii, jj) in NAMED.items():
        res = grid[(ii, jj)]
        if not (res["linear"]["defined"] and res["loglinear"]["defined"]):
            verdicts.append((f"P1 {name}: an interpolant is undefined at this span", False))
            continue
        ok_all = True
        worst = []
        for c in CLASSES:
            for stat in ("med", "p90", "max"):
                lin, log = res["linear"][c][stat], res["loglinear"][c][stat]
                if not (log < lin):
                    ok_all = False
                    worst.append(f"{c}/{stat} log {100*log:.3f}% >= lin {100*lin:.3f}%")
        verdicts.append((f"P1 {name}: log-linear below linear in every class and statistic"
                         + ("" if ok_all else "; exceptions: " + "; ".join(worst)), ok_all))
        med = res["loglinear"]["all"]["med"]
        verdicts.append((f"P1 {name}: log-linear all-entry median |e| = {100*med:.3f} % < 2 %", med < 0.02))
    n_bad = sum(beats.values())
    verdicts.append((f"P1 grid-wide: log-linear below linear in every class/statistic at every defined span "
                     f"(violations {n_bad} of {len(beats)*len(grid)} class/stat/span combinations)", n_bad == 0))
    gmed = dist[("loglinear", "all", "med")][0]
    verdicts.append((f"P1 grid-wide: median over spans of the log-linear all-entry median |e| = {100*gmed:.3f} % < 2 %",
                     gmed < 0.02))

    say()
    say("=" * 78)
    say("PREDICTIONS")
    say("=" * 78)
    n_fail = 0
    for text, ok in verdicts:
        say(f"  [{'HELD' if ok else 'FAILED'}] {text}")
        n_fail += int(not ok)
    say(f"  {len(verdicts) - n_fail} of {len(verdicts)} held")
    say(f"  REFUTER ((a) outside [(b),(c)] by more than the interval width): {'FIRED' if refuted else 'did not appear'}")
    say("  Part A (true L at interior temperatures): NOT RUN -- needs module edits or a re-implementation of the")
    say("  rate assembly (see docstring); everything above is from stamped grid operators only.")
    say(f"  runtime {time.time() - t_wall0:.1f} s")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "physical_ramp_bound"
        out.mkdir(parents=True, exist_ok=True)
        for fname, rows_ in (("physical_ramp_bound_operator.csv", op_rows),
                             ("physical_ramp_bound_trajectory.csv", traj_rows)):
            with (out / fname).open("w", newline="") as fh:
                fh.write("\n".join(hdr) + "\n")
                w = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()))
                w.writeheader()
                w.writerows(rows_)
        (out / "physical_ramp_bound.txt").write_text("\n".join(out_lines) + "\n")
        say(f"\nwrote {out.relative_to(ROOT)}/physical_ramp_bound_operator.csv ({len(op_rows)} rows), "
            f"physical_ramp_bound_trajectory.csv ({len(traj_rows)} rows), physical_ramp_bound.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
