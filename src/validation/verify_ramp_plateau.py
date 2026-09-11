#!/usr/bin/env python
"""
verify_ramp_plateau.py
======================
Does a finite-duration temperature ramp reach the same partial-equilibrium
plateau as an instantaneous step?

WHY THIS EXISTS
---------------
chapter5.tex (sec. on the ELM objections, lines ~1590-1598) asserts, without a
script behind it:

    "Any ramp with tau_relax << t_ramp << tau_slow reaches the same plateau,
     because the fast manifold tracks the ramp adiabatically while the slow
     reservoir cannot follow it at all. ... The step idealisation is safe."

The reviewer's objection: during a ramp of duration tau_ramp the ground-state
reservoir moves by roughly tau_ramp/tau_slow of its total excursion, so the
plateau reached is only approximately eps_plateau. This script integrates the
full 43-state system dn/dt = L(t) n + S(t) through a linear ramp between two
grid-point operators and measures the plateau actually reached.

PHYSICS
-------
Post-step operator L_new = L[k, j], S_new = S[k, j] with k = i + 1 (heating by
one grid interval, which at the two points below is also the divertor map's
5 percent step: argmin |Te - 1.05 Te[i]| = i + 1, asserted). Ion reservoir
normalised to 1 in S_grid.npy.

    n_old        = -L_old^-1 S_old                      pre-step CR equilibrium
    n_new        = -L_new^-1 S_new                      post-step CR equilibrium
    n_E^PE       = -L_EE^-1 (S_E + L_Eg n_g_old)        frozen-reservoir plateau
    R            = n_3 / n_4  (shell sums via ctx.n_values)
    eps_plateau  = |R_PE / R_CRE_new - 1|
    eps_CRE(t)   = |R(t) / R_CRE_new - 1|

Second observable: A-weighted Halpha/Hbeta, weights from
Balmer_transient_ratio.load_radiative_weights (photon counting, the same
convention as verify_weighted_census.py), contracted over LINE_CHANNELS.

Ramp: L(t) = (1-f) L_old + f L_new, S likewise, f = min(t/tau_ramp, 1), exactly
the interpolation of verify_ramp_vs_step.py lines 273-285, then held constant.

--interp linear (default) | loglinear
    loglinear: entrywise geometric interpolation of the magnitudes,
    L_ab(f) = sign(L_ab) |L_ab,old|^(1-f) |L_ab,new|^f, S likewise; zero entries
    stay zero. Requires identical sparsity and sign patterns at the two grid
    points (raised otherwise). The interpolated operator is checked to be a
    CR generator: off-diagonals >= 0 asserted, worst (largest) column sum
    reported, at f = 0, 0.25, 0.5, 0.75, 1. The main block uses --interp;
    the LOG-LINEAR (or, under --interp loglinear, LINEAR) sensitivity block
    re-runs tau_ramp/tau_slow = 1e-2, 1e-1, 1 with the other interpolation,
    so a --write run always stores both, distinguished by the CSV column
    `interp`.

TWO COLUMNS, KEPT APART
-----------------------
    "max"      = max_{t > tau_ramp} eps_CRE(t) / eps_plateau
    "plateau"  = eps_CRE(tau_ramp + 30 tau_relax) / eps_CRE^step(30 tau_relax)
The "max" column contains, at the benchmark, a transient peak of the RATIO
n3/n4 during the fast relaxation (see OVERSHOOT below); the "plateau" column
is the value at the window start and is the one compared with the De formula.

PREDICTIONS, WRITTEN BEFORE THE FIRST RUN (11 Sep 2026)
--------------------------------------------------------
P1  Pure step: the ODE solution agrees with n_new + expm(L_new t)(n_old - n_new)
    to better than 1e-6 relative at 20 log-spaced times, and the shell
    eps_plateau reproduces divertor_map.csv (heat, [23,5] and [15,3]) to 1e-9.
    tau_slow, tau_relax of the post-step operator reproduce the CSV's tau_QSS,
    tau_relax columns to 1e-9. These are guards; failure stops the script.
P2  tau_ramp / tau_slow = 1e-3: the maximum of eps_CRE(t) over t > tau_ramp
    matches eps_plateau to better than 0.5 percent.
P3  tau_ramp / tau_slow = 1e-2: matches to a few percent (order 1 percent,
    the reservoir having moved ~1 percent of its excursion during the ramp).
P4  tau_ramp / tau_slow = 1e-1: the maximum falls noticeably below eps_plateau
    (of order 5 to 10 percent below).
P5  tau_ramp / tau_slow = 1: the transient error is a fraction of eps_plateau
    (of order half or less), because the reservoir tracks most of the way.
P6  The fast probes tau_ramp = 1 and 10 tau_relax behave as the step to
    better than 0.1 percent.
    [Skeptic pass, 11 Sep 2026: originally stated on the post-ramp maximum,
    which at the benchmark carries the ratio's transient peak; now stated on
    the plateau (window-start) column, as the claim under test requires.]
P7  The step's own overshoot (max over t > 0 of eps_CRE relative to
    max(eps_step, eps_plateau)) is below 1 percent.
    [Wording corrected after the skeptic pass; see OVERSHOOT below.]
P8  Ground-state drift during the ramp, |n_g(tau_ramp)/n_g_old - 1|, divided
    by (tau_ramp/tau_slow) times the total drift |n_g_new/n_g_old - 1|, equals
    the first-order-lag result 1/2 - 1/(6 De), De = tau_slow/tau_ramp, to
    within 0.02.
    [Skeptic pass, 11 Sep 2026: the original band was 0.5 to 2.0, which the
    exact value 1/2 sits on. Replaced by the lag law. The benchmark's
    De = 1000 point is outside the law's domain (tau_ramp = 8.2 tau_relax,
    the fast manifold is not adiabatic) and is reported but not counted.]

REFUTING OBSERVATION
--------------------
A ramp at 1e-3 tau_slow whose post-ramp maximum eps_CRE differs from
eps_plateau by more than 1 percent; or a ramp whose maximum EXCEEDS
eps_plateau by more than the step's own overshoot (P7) plus 0.1 percent,
which would mean the ramp amplifies the error beyond the step.

OVERSHOOT (found on the first run; wording fixed after the skeptic pass)
------------------------------------------------------------------------
At the benchmark [23,5] the pure step's eps_CRE(t) peaks at 1.066 eps_plateau
near t = 1.2 tau_relax before settling. This is a property of the RATIO
n3/n4, not of the state: the growth factor max_t ||d(t)|| / ||d(0)|| of the
deviation from partial equilibrium over the excited block is never above 1 at
any of the 784 (point, direction) pairs (worst 0.998 in L2, exactly 1 at
t = 0 in L1 and Linf; skeptic pass, 11 Sep 2026). After the heating step
both shells start above their frozen-reservoir values, n = 4 relaxes faster
than n = 3, and their relative deviations cross at 0.72 tau_relax and again
at 4.66 tau_relax; between the crossings the ratio exceeds R_PE. At [15,3]
the shells start on opposite sides and the approach is monotone. The
interior-maximum guard below prints "no interior maximum" wherever the
maximum sits at an endpoint of the sampled interval.

POST-HOC ADDITIONS (11 Sep 2026, after the first run; no threshold above was
changed except P6 and P8 as recorded)
-----------------------------------------------------------------------------
a)  The plateau column tracks De (1 - exp(-1/De)) with De = tau_slow/tau_ramp,
    the scalar suppression formula of verify_ramp_vs_step.py with tau_relax
    replaced by tau_slow. Printed as a reading aid, not a test.
b)  Observable sensitivity of the step overshoot at the benchmark (skeptic):
    n = 3 population 1.020, n3/n4 and Halpha/Hbeta 1.066, n3/n5 1.132,
    n4/n5 2.92 (all as max_t eps / eps_plateau of that observable).
c)  Interpolation sensitivity (skeptic): with log-linear instead of linear
    interpolation of L and S along the ramp, the plateau ratio at
    tau_ramp = tau_slow at [15,3] moves from 0.6469 to 0.6576; at
    tau_ramp = 0.01 tau_slow the change is 1.6e-4. Now computed by this
    script (--interp, LOG-LINEAR block) rather than quoted.
e)  Lag law: the exact single-pole result for a linear ramp of the target is
    drift/total = 1 - De (1 - exp(-1/De)), i.e. drift/(td/tau_slow) =
    De [1 - De (1 - exp(-1/De))], whose expansion is 1/2 - 1/(6 De). Both
    are printed; P8 tests the truncated form as recorded above.
d)  The overshoot changes the 100 us average of eps_CRE by -4e-5 relative at
    the benchmark (skeptic); it is invisible to the census.

CONVERGENCE (required)
----------------------
Everything is recomputed at rtol 1e-8 and the relative change of every
reported number is printed and stored (columns conv_*). Window readouts are
taken from integrator step endpoints (the hold is integrated in segments
ending exactly at t_ws, t_we, t_end), never from dense-output interpolation;
every post-ramp trajectory is cross-checked against expm on the constant
post-step operator (column expm_resid, must be below 1e-6).

Read-only. Writes only under validation/ramp_plateau/ with --write.
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
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)
sys.path.insert(0, str(ROOT / "src" / "rates"))

# The two points, as the task specifies them; the indices are DERIVED from the
# grid through ctx.nearest_point and then asserted, never used directly.
POINTS = [
    ("benchmark", 2.947, 1.389e14, (23, 5)),
    ("defended maximum", 2.024, 1.931e13, (15, 3)),
]
FRAC = 0.05                       # divertor map's step definition, to assert k == i+1
RAMP_OVER_SLOW = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
RAMP_OVER_RELAX = [1.0, 10.0]
WIN_LO, WIN_HI = 30.0, 30.0       # window: tau_ramp + 30 tau_relax  ..  tau_slow / 30
RTOL_MAIN, RTOL_CONV = 1e-10, 1e-8
ATOL_SCALE = 1e-12
N_EXPM_CHECK = 20
N_EXPM_RAMP = 10
EXPM_TOL = 1e-6
CSV_REPRO_TOL = 1e-9
P8_DOMAIN_MIN_RELAX = 30.0        # the lag law needs an adiabatic fast manifold
SENS_RATIOS = (1e-2, 1e-1, 1.0)   # tau_ramp/tau_slow re-run with the other interpolation


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv_with_comments(path: Path) -> list[dict]:
    with path.open() as fh:
        body = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(body))


def spectrum(L: np.ndarray) -> tuple[float, float]:
    lam = np.linalg.eigvals(L)
    lam = lam[np.argsort(lam.real)[::-1]]
    if lam[0].real >= 0 or lam[1].real >= 0:
        raise RuntimeError("operator is not strictly stable")
    return -1.0 / lam[0].real, -1.0 / lam[1].real


def make_interpolants(L0, L1, b0, b1, interp):
    """
    Return (L_at(f), b_at(f)) for f in [0, 1].
      linear    : (1-f) X0 + f X1                       (verify_ramp_vs_step.py)
      loglinear : sign(X) |X0|^(1-f) |X1|^f entrywise; zero stays zero.
                  Requires identical sparsity and sign patterns, else raises.
    """
    if interp == "linear":
        return (lambda f: (1 - f) * L0 + f * L1), (lambda f: (1 - f) * b0 + f * b1)
    if interp != "loglinear":
        raise ValueError(f"unknown interpolation {interp!r}")
    for name, X0, X1 in (("L", L0, L1), ("S", b0, b1)):
        if not np.array_equal(X0 == 0, X1 == 0):
            raise RuntimeError(f"loglinear: {name} sparsity pattern differs between the two "
                               f"grid points ({int((X0 == 0).sum())} vs {int((X1 == 0).sum())} zeros)")
        if not np.array_equal(np.sign(X0), np.sign(X1)):
            raise RuntimeError(f"loglinear: {name} has entries that change sign between the "
                               f"two grid points; geometric interpolation is undefined there")
    nzL = L0 != 0
    nzb = b0 != 0
    sgnL, sgnb = np.sign(L0), np.sign(b0)
    lnL0 = np.where(nzL, np.log(np.abs(np.where(nzL, L0, 1.0))), 0.0)
    lnL1 = np.where(nzL, np.log(np.abs(np.where(nzL, L1, 1.0))), 0.0)
    lnb0 = np.where(nzb, np.log(np.abs(np.where(nzb, b0, 1.0))), 0.0)
    lnb1 = np.where(nzb, np.log(np.abs(np.where(nzb, b1, 1.0))), 0.0)

    def L_at(f):
        return np.where(nzL, sgnL * np.exp((1 - f) * lnL0 + f * lnL1), 0.0)

    def b_at(f):
        return np.where(nzb, sgnb * np.exp((1 - f) * lnb0 + f * lnb1), 0.0)

    return L_at, b_at


def check_generator(L_at, b_at, fs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """
    The interpolated operator must still be a CR generator: off-diagonals
    (rates into a state) non-negative, source non-negative. Asserted. The
    column sums (minus the ionisation loss of each column) are reported: the
    worst is the largest, and a positive one would mean the interpolation
    creates population.
    """
    n = L_at(0.0).shape[0]
    off = ~np.eye(n, dtype=bool)
    worst_offdiag, worst_colsum, worst_f = np.inf, -np.inf, None
    for f in fs:
        Lf, bf = L_at(f), b_at(f)
        worst_offdiag = min(worst_offdiag, float(Lf[off].min()))
        if Lf[off].min() < 0:
            raise RuntimeError(f"interpolated operator at f={f} has a negative off-diagonal "
                               f"{Lf[off].min():.3e}: not a CR generator")
        if bf.min() < 0:
            raise RuntimeError(f"interpolated source at f={f} has a negative entry")
        cs = float(Lf.sum(axis=0).max())
        if cs > worst_colsum:
            worst_colsum, worst_f = cs, f
    return worst_offdiag, worst_colsum, worst_f


def integrate(L0, L1, b0, b1, n_init, td, breakpoints, rtol, atol, interp="linear"):
    """
    Radau on the ramp [0, td] (L, S interpolated in t per `interp`, max_step
    td/50 so the ramp is resolved), then the autonomous hold on the post-step
    operator integrated in consecutive segments ending exactly at each
    breakpoint (window start, window end, t_end). Returns a callable n(t) on
    [0, t_end], the state at the end of the ramp, a dict {breakpoint: state}
    of integrator step endpoints (no interpolation), and the total rhs count.
    """
    L_at, b_at = make_interpolants(L0, L1, b0, b1, interp)

    def L_of(t):
        f = min(max(t / td, 0.0), 1.0) if td > 0 else 1.0
        return L_at(f)

    def b_of(t):
        f = min(max(t / td, 0.0), 1.0) if td > 0 else 1.0
        return b_at(f)

    nfev = 0
    sol_ramp = None
    if td > 0:
        sol_ramp = solve_ivp(lambda t, n: L_of(t) @ n + b_of(t), [0.0, td], n_init,
                             method="Radau", jac=lambda t, n: L_of(t),
                             rtol=rtol, atol=atol, dense_output=True,
                             max_step=td / 50.0)
        if not sol_ramp.success:
            raise RuntimeError(f"ramp segment failed: {sol_ramp.message}")
        n_td = sol_ramp.y[:, -1]
        nfev += sol_ramp.nfev
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
        m = t < td
        if m.any():
            out[:, m] = sol_ramp.sol(t[m])
        rest = ~m
        for (a, b, sol) in segs:
            sel = rest & (t >= a) & (t <= b)
            if sel.any():
                out[:, sel] = sol.sol(t[sel])
                rest = rest & ~sel
        if rest.any():
            raise RuntimeError(f"times outside integrated range: {t[rest]}")
        return out

    return n_of, n_td, at, nfev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--interp", choices=("linear", "loglinear"), default="linear",
                    help="interpolation of L and S along the ramp (see docstring)")
    a = ap.parse_args()
    t_wall0 = time.time()

    import Balmer_transient_ratio as btr                  # noqa: E402

    ctx = CRContext.load()
    ctx.validate()
    if Path(btr.REPO).resolve() != ctx.root.resolve():
        raise RuntimeError(f"Balmer_transient_ratio.REPO {btr.REPO} differs from "
                           f"cr_context root {ctx.root}")
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = int(ctx.ground_index)
    nv = np.asarray(ctx.n_values)
    labels = ctx.labels
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)

    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not sp.exists():
        raise FileNotFoundError(f"missing source vector {sp}")
    S = np.load(sp)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")

    dmap_p = ROOT / "validation/divertor_map/divertor_map.csv"
    if not dmap_p.exists():
        raise FileNotFoundError(f"missing {dmap_p}; the step eps_plateau must be "
                                f"reproduced from it before anything else is trusted")
    dmap = read_csv_with_comments(dmap_p)
    with dmap_p.open() as fh:
        dmap_header = [ln.rstrip("\n") for ln in fh if ln.startswith("#")]
    for want, line in ((sha256(lp), dmap_header[1]), (sha256(sp), dmap_header[2]),
                       (sha256(ctx.state_index_path), dmap_header[3])):
        if want not in line:
            raise RuntimeError(
                f"divertor_map.csv was built on different data:\n  {line}\n"
                f"  current sha256 {want}\nRegenerate it before comparing.")

    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")

    # ---- weight vectors, copied from verify_weighted_census.weight_vectors ----
    def weight_vectors(use_photon_energy: bool) -> dict[str, np.ndarray]:
        lw = btr.load_radiative_weights(use_photon_energy=use_photon_energy)
        w = {}
        for line in ("Halpha", "Hbeta"):
            v = np.zeros(ctx.n_states)
            for (idx_u, _il, _nu, _lu, _nl, _ll, lab) in btr.LINE_CHANNELS[line]:
                if labels[idx_u].upper() != lab.split("_")[0]:
                    raise RuntimeError(
                        f"state ordering mismatch: index {idx_u} is "
                        f"{labels[idx_u]} but the channel table calls it {lab}")
                v[idx_u] = lw.weights[line][lab]
            if v.min() < 0 or v.max() <= 0:
                raise RuntimeError(f"bad weight vector for {line}")
            w[line] = v
        for line, m in (("Halpha", 3), ("Hbeta", 4)):
            bad = [labels[i] for i in np.where(w[line] > 0)[0] if nv[i] != m]
            if bad:
                raise RuntimeError(f"{line} weights touch states outside "
                                   f"n={m}: {bad}")
        return w

    wp = weight_vectors(False)
    w3s = np.zeros(ctx.n_states); w3s[N3] = 1.0
    w4s = np.zeros(ctx.n_states); w4s[N4] = 1.0
    OBS = {"shell": (w3s, w4s), "line": (wp["Halpha"], wp["Hbeta"])}

    out_lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
           f"# interpreter {sys.executable}  numpy {np.__version__}  "
           f"scipy {__import__('scipy').__version__}",
           f"# L_grid sha256 {sha256(lp)}", f"# S_grid sha256 {sha256(sp)}",
           f"# state_index sha256 {sha256(ctx.state_index_path)}",
           f"# radiative_rates sha256 {sha256(btr.DATA_RAD)}",
           f"# divertor_map.csv {dmap_header[0]}",
           f"# rtol {RTOL_MAIN} (convergence rerun {RTOL_CONV}), atol {ATOL_SCALE}*max|n_old|, "
           f"Radau, t_end = tau_ramp + 10 tau_slow; window readouts from step endpoints",
           f"# main block interpolation: {a.interp}; sensitivity block: the other one at "
           f"tau_ramp/tau_slow in {SENS_RATIOS}"]

    say("=" * 78)
    say("RAMP PLATEAU: does a finite ramp reach the step's partial-equilibrium plateau?")
    for h in hdr:
        say(h)
    say(f"Halpha weights: " + ", ".join(f"{labels[i]}={wp['Halpha'][i]:.4e}"
                                        for i in np.where(wp["Halpha"] > 0)[0]))
    say(f"Hbeta  weights: " + ", ".join(f"{labels[i]}={wp['Hbeta'][i]:.4e}"
                                        for i in np.where(wp["Hbeta"] > 0)[0]))
    say("observables: shell = n3/n4 (shell sums); line = A-weighted Halpha/Hbeta")
    say("=" * 78)

    rows: list[dict] = []
    verdicts: list[tuple[str, object]] = []      # (text, True/False/None); None = not counted

    for pname, te_req, ne_req, expect_ij in POINTS:
        i, j = ctx.nearest_point(te_req, ne_req)
        if (i, j) != expect_ij:
            raise RuntimeError(f"{pname}: nearest_point({te_req}, {ne_req}) = [{i},{j}], "
                               f"expected {expect_ij}; grids are not those the "
                               f"reference values were measured on")
        k = i + 1
        k_frac = int(np.argmin(np.abs(te - te[i] * (1 + FRAC))))
        if k_frac != k:
            raise RuntimeError(f"{pname}: divertor map's {FRAC:.0%} step lands on "
                               f"[{k_frac}], not i+1 = {k}; its eps_plateau is not "
                               f"the reference for a one-interval step")
        L0, L1, b0, b1 = L[i, j], L[k, j], S[i, j], S[k, j]
        tau_slow, tau_relax = spectrum(L1)
        n_old = np.linalg.solve(L0, -b0)
        n_new = np.linalg.solve(L1, -b1)
        LEE = L1[np.ix_(E, E)]
        LEg = L1[np.ix_(E, [g])].ravel()
        nPE = np.zeros(ctx.n_states)
        nPE[E] = np.linalg.solve(LEE, -(b1[E] + LEg * n_old[g]))
        nPE[g] = n_old[g]

        d = [r for r in dmap if (r["direction"], int(r["i"]), int(r["j"])) == ("heat", i, j)]
        if len(d) != 1:
            raise RuntimeError(f"divertor_map.csv: expected one row heat [{i},{j}], found {len(d)}")
        d = d[0]

        say()
        say("-" * 78)
        say(f"{pname.upper()}  [{i},{j}] -> [{k},{j}]   Te {te[i]:.4f} -> {te[k]:.4f} eV   "
            f"ne {ne[j]:.4e} cm^-3")
        say("-" * 78)
        say(f"  post-step operator [{k},{j}]: tau_slow {tau_slow:.6e} s   tau_relax {tau_relax:.6e} s   "
            f"M {tau_slow/tau_relax:.1f}")
        for mine, theirs, name in ((tau_slow, float(d["tau_QSS"]), "tau_QSS"),
                                   (tau_relax, float(d["tau_relax"]), "tau_relax")):
            rel = abs(mine - theirs) / abs(theirs)
            say(f"  divertor_map.csv {name:9s} {theirs:.6e}   rel diff {rel:.2e}")
            if rel > CSV_REPRO_TOL:
                raise RuntimeError(f"{name} does not reproduce divertor_map.csv: {rel:.3e}")
        ng_total = abs(n_new[g] / n_old[g] - 1.0)
        say(f"  n_g_old {n_old[g]:.6e}   n_g_new {n_new[g]:.6e}   total drift "
            f"|n_g_new/n_g_old - 1| = {ng_total:.6e}")

        eps_plateau, eps_step, R_new = {}, {}, {}
        for oname, (wa, wb) in OBS.items():
            R_new[oname] = (wa @ n_new) / (wb @ n_new)
            R_pe = (wa @ nPE) / (wb @ nPE)
            R_old = (wa @ n_old) / (wb @ n_old)
            eps_plateau[oname] = abs(R_pe / R_new[oname] - 1.0)
            eps_step[oname] = abs(R_old / R_new[oname] - 1.0)
            say(f"  {oname:5s} R_old {R_old:.6f}  R_PE {R_pe:.6f}  R_CRE_new {R_new[oname]:.6f}   "
                f"eps_step {eps_step[oname]:.6f}   eps_plateau {eps_plateau[oname]:.6f}")
        rel = abs(eps_plateau["shell"] - float(d["eps_plateau"])) / float(d["eps_plateau"])
        say(f"  divertor_map.csv eps_plateau (shell) {float(d['eps_plateau']):.6f}   rel diff {rel:.2e}")
        if rel > CSV_REPRO_TOL:
            raise RuntimeError(f"shell eps_plateau does not reproduce divertor_map.csv: {rel:.3e}")

        atol = ATOL_SCALE * np.abs(n_old).max()

        cases = [("step", 0.0)]
        cases += [(f"{r:g} tau_slow", r * tau_slow) for r in RAMP_OVER_SLOW if r > 0]
        cases += [(f"{r:g} tau_relax", r * tau_relax) for r in RAMP_OVER_RELAX]

        def run_case(td: float, rtol: float, interp: str = a.interp) -> dict:
            t_end = td + 10.0 * tau_slow
            t_ws = td + WIN_LO * tau_relax
            t_we = tau_slow / WIN_HI
            n_of, n_td, at, nfev = integrate(L0, L1, b0, b1, n_old, td,
                                             [t_ws, t_we, t_end], rtol, atol, interp)
            # expm cross-check on the constant post-step operator (item 3):
            # n(t) = n_new + expm(L1 (t - td)) (n(td) - n_new) for t > td
            t_chk = td + np.logspace(np.log10(tau_relax * 1e-2), np.log10(10.0 * tau_slow),
                                     N_EXPM_CHECK if td == 0 else N_EXPM_RAMP)
            worst_norm, worst_R = 0.0, 0.0
            for t in t_chk:
                n_ex = n_new + expm(L1 * (t - td)) @ (n_td - n_new)
                n_od = n_of(t)[:, 0]
                worst_norm = max(worst_norm, np.linalg.norm(n_od - n_ex) / np.linalg.norm(n_ex))
                worst_R = max(worst_R, abs((n_od[N3].sum() / n_od[N4].sum())
                                           / (n_ex[N3].sum() / n_ex[N4].sum()) - 1.0))
            if worst_norm >= EXPM_TOL:
                raise RuntimeError(f"ODE disagrees with expm after the ramp (td={td:.3e}, "
                                   f"rtol={rtol}): {worst_norm:.3e} >= {EXPM_TOL}")
            # sampling: t = 0, uniform on the ramp, log-spaced in (t - td) afterwards
            t_post = td + np.logspace(np.log10(tau_relax * 1e-3), np.log10(10.0 * tau_slow), 6000)
            ts = [[0.0], t_post, [t_ws, t_end]]
            if td > 0:
                ts.append(np.linspace(0.0, td, 2000))
            if t_we > td:
                ts.append([t_we])
            ts = np.unique(np.concatenate([np.asarray(x, dtype=float) for x in ts]))
            ns = n_of(ts)
            res = dict(nfev=nfev, t_end=t_end, expm_resid=worst_norm, expm_resid_R=worst_R,
                       ng_drift_ramp=abs(n_td[g] / n_old[g] - 1.0))
            post = ts > td
            for oname, (wa, wb) in OBS.items():
                def eps_of(n):
                    return float(abs((wa @ n) / (wb @ n) / R_new[oname] - 1.0))
                eps = np.abs((wa @ ns) / (wb @ ns) / R_new[oname] - 1.0)
                kmax = np.where(post)[0][np.argmax(eps[post])]
                kall = int(np.argmax(eps))
                interior = 0 < kall < len(ts) - 1
                res[oname] = dict(
                    eps_max_post=float(eps[kmax]), t_max=float(ts[kmax]),
                    eps_max_all=float(eps.max()), t_max_all=float(ts[kall]),
                    interior_max=interior,
                    eps_at_ramp_end=eps_of(n_td),
                    eps_ws=eps_of(at[t_ws]),
                    eps_we=eps_of(at[t_we]) if t_we > td else np.nan,
                    t_ws=t_ws, t_we=t_we if t_we > td else np.nan,
                    eps_end=eps_of(at[t_end]))
            return res

        for interp in ("linear", "loglinear"):
            wo, wc, wf = check_generator(*make_interpolants(L0, L1, b0, b1, interp))
            say(f"  generator check, {interp:9s}: min off-diagonal {wo:.3e} (>= 0 asserted), "
                f"worst column sum {wc:.4e} s^-1 at f = {wf:g}")

        res_step = run_case(0.0, RTOL_MAIN)
        say(f"  P1 ODE vs expm, pure step, {N_EXPM_CHECK} log-spaced times: worst ||dn||/||n|| "
            f"{res_step['expm_resid']:.2e}, worst shell-ratio rel err {res_step['expm_resid_R']:.2e}"
            f"   (nfev {res_step['nfev']})")

        say()
        say(f"  {'ramp':>14s} {'tau_ramp [s]':>12s} {'td/t_slow':>9s} {'td/t_rel':>8s} "
            f"{'obs':>5s} {'max eps t>td':>12s} {'MAX':>8s} {'eps(ws)':>10s} {'PLATEAU':>8s} "
            f"{'eps(we)':>10s} {'eps(t_end)':>10s} {'ng drift':>9s} {'/total':>7s} {'expm':>8s}")
        say(f"  {'':>14s} {'':>12s} {'':>9s} {'':>8s} {'':>5s} {'':>12s} {'/eps_pl':>8s} "
            f"{'':>10s} {'/step ws':>8s}")
        say("  " + "-" * 150)
        def emit(cname, td, interp):
            res = res_step if td == 0 else run_case(td, RTOL_MAIN, interp)
            res_c = run_case(td, RTOL_CONV, interp)
            for oname in OBS:
                r, rc = res[oname], res_c[oname]
                ratio = r["eps_max_post"] / eps_plateau[oname]
                plateau = r["eps_ws"] / res_step[oname]["eps_ws"]
                say(f"  {cname:>14s} {td:12.4e} {td/tau_slow:9.1e} {td/tau_relax:8.1e} "
                    f"{oname:>5s} {r['eps_max_post']:12.6f} {ratio:8.4f} {r['eps_ws']:10.6f} "
                    f"{plateau:8.4f} {r['eps_we']:10.6f} {r['eps_end']:10.2e} "
                    f"{res['ng_drift_ramp']:9.2e} {res['ng_drift_ramp']/ng_total:7.4f} "
                    f"{res['expm_resid']:8.1e}")

                def rel(x, y):
                    if np.isnan(x) and np.isnan(y):
                        return 0.0
                    return abs(x - y) / max(abs(y), 1e-300)

                rows.append(dict(
                    point=pname, interp=interp if td > 0 else "none",
                    i=i, j=j, k=k, Te_old=float(te[i]), Te_new=float(te[k]),
                    ne=float(ne[j]), tau_slow=tau_slow, tau_relax=tau_relax,
                    ramp=cname, tau_ramp_s=td, tau_ramp_over_tau_slow=td / tau_slow,
                    tau_ramp_over_tau_relax=td / tau_relax, observable=oname,
                    R_CRE_new=R_new[oname], eps_step=eps_step[oname],
                    eps_plateau_step=eps_plateau[oname],
                    eps_max_post_ramp=r["eps_max_post"], t_of_max=r["t_max"],
                    ratio_max_to_plateau=ratio,
                    eps_max_all_t=r["eps_max_all"], t_of_max_all=r["t_max_all"],
                    interior_max=bool(r["interior_max"]),
                    overshoot=r["eps_max_all"] / max(eps_step[oname], eps_plateau[oname]),
                    eps_at_ramp_end=r["eps_at_ramp_end"],
                    t_window_start=r["t_ws"], eps_at_window_start=r["eps_ws"],
                    plateau_ratio=plateau,
                    t_window_end=r["t_we"], eps_at_window_end=r["eps_we"],
                    eps_at_t_end=r["eps_end"],
                    ng_drift_ramp=res["ng_drift_ramp"], ng_drift_total=ng_total,
                    ng_drift_fraction=res["ng_drift_ramp"] / ng_total,
                    expm_resid=res["expm_resid"], expm_resid_shell_ratio=res["expm_resid_R"],
                    nfev=res["nfev"], nfev_conv=res_c["nfev"],
                    conv_eps_max_post_ramp=rel(rc["eps_max_post"], r["eps_max_post"]),
                    conv_eps_at_window_start=rel(rc["eps_ws"], r["eps_ws"]),
                    conv_plateau_ratio=rel(rc["eps_ws"] / res_step[oname]["eps_ws"], plateau),
                    conv_eps_at_window_end=rel(rc["eps_we"], r["eps_we"]),
                    conv_eps_at_ramp_end=rel(rc["eps_at_ramp_end"], r["eps_at_ramp_end"]),
                    conv_eps_at_t_end=rel(rc["eps_end"], r["eps_end"]),
                    conv_ng_drift_ramp=rel(res_c["ng_drift_ramp"], res["ng_drift_ramp"]),
                    conv_t_of_max=rel(rc["t_max"], r["t_max"]),
                ))

        for cname, td in cases:
            emit(cname, td, a.interp)

        # ---- sensitivity block: the other interpolation at SENS_RATIOS ------
        other = "loglinear" if a.interp == "linear" else "linear"
        say()
        say(f"  {other.upper()} INTERPOLATION SENSITIVITY (same columns; compare with the "
            f"{a.interp} rows above)")
        say("  " + "-" * 150)
        for ratio_ in SENS_RATIOS:
            emit(f"{ratio_:g} tau_slow", ratio_ * tau_slow, other)
        say()
        say(f"  {'ramp':>14s} {'obs':>5s} {a.interp+' MAX':>16s} {other+' MAX':>16s} {'diff':>10s} "
            f"{a.interp+' PLATEAU':>20s} {other+' PLATEAU':>20s} {'diff':>10s}")
        for ratio_ in SENS_RATIOS:
            cname = f"{ratio_:g} tau_slow"
            for oname in OBS:
                rm = [q for q in rows if q["point"] == pname and q["ramp"] == cname
                      and q["observable"] == oname and q["interp"] == a.interp][0]
                ro = [q for q in rows if q["point"] == pname and q["ramp"] == cname
                      and q["observable"] == oname and q["interp"] == other][0]
                say(f"  {cname:>14s} {oname:>5s} {rm['ratio_max_to_plateau']:16.4f} "
                    f"{ro['ratio_max_to_plateau']:16.4f} {ro['ratio_max_to_plateau']-rm['ratio_max_to_plateau']:+10.2e} "
                    f"{rm['plateau_ratio']:20.4f} {ro['plateau_ratio']:20.4f} "
                    f"{ro['plateau_ratio']-rm['plateau_ratio']:+10.2e}")

        # ---- lag law, exact and truncated, all tau_slow ramps (main interp) --
        say()
        say("  LAG LAW for the reservoir, linear ramp of the target, single pole tau_slow:")
        say("    exact      drift/total = 1 - De (1 - exp(-1/De));  drift/(td/tau_slow) = De [1 - De (1 - exp(-1/De))]")
        say("    truncated  drift/(td/tau_slow) = 1/2 - 1/(6 De)")
        say(f"  {'ramp':>14s} {'interp':>9s} {'De':>8s} {'drift/total':>12s} {'exact':>12s} "
            f"{'drift/(td/ts)':>14s} {'exact':>10s} {'truncated':>10s}")
        for r in [q for q in rows if q["point"] == pname and q["observable"] == "shell"
                  and q["tau_ramp_s"] > 0 and "tau_slow" in q["ramp"]]:
            De = 1.0 / r["tau_ramp_over_tau_slow"]
            y_exact = 1.0 - De * (1.0 - np.exp(-1.0 / De))
            say(f"  {r['ramp']:>14s} {r['interp']:>9s} {De:8.0f} {r['ng_drift_fraction']:12.6f} "
                f"{y_exact:12.6f} {r['ng_drift_fraction']*De:14.6f} {y_exact*De:10.6f} "
                f"{0.5 - 1.0/(6.0*De):10.6f}")

        # ---- overshoot metric with interior-maximum guard (item 4) ----------
        say()
        say("  OVERSHOOT of the ratio, max_t eps_CRE / max(eps_step, eps_plateau), per observable "
            f"({a.interp} rows):")
        for r in [q for q in rows if q["point"] == pname and q["interp"] in ("none", a.interp)]:
            if r["interior_max"]:
                after = ("" if r["tau_ramp_s"] == 0 else
                         f" = td + {(r['t_of_max_all'] - r['tau_ramp_s']) / tau_relax:.2f} tau_relax")
                at_end = "  [at the ramp end]" if (r["tau_ramp_s"] > 0 and r["t_of_max_all"] == r["tau_ramp_s"]) else ""
                say(f"    {r['ramp']:>14s} {r['observable']:>5s}  {r['overshoot']:.4f}  "
                    f"at t = {r['t_of_max_all']:.3e} s = {r['t_of_max_all']/tau_relax:.2f} tau_relax"
                    + after + at_end)
            else:
                say(f"    {r['ramp']:>14s} {r['observable']:>5s}  {r['overshoot']:.4f}  "
                    f"no interior maximum (max at t = {r['t_of_max_all']:.3e} s)")

        # ---- predictions for this point ----------------------------------
        def row(cname, oname):
            return [r for r in rows if r["point"] == pname and r["ramp"] == cname
                    and r["observable"] == oname and r["interp"] in ("none", a.interp)][0]

        for oname in OBS:
            ep = eps_plateau[oname]
            rs = row("step", oname)
            over_step = rs["eps_max_all_t"] / ep - 1.0
            verdicts.append((f"{pname} {oname} P7 step overshoot of the ratio "
                             f"{rs['overshoot']:.4f} (max/eps_plateau - 1 = {over_step:+.2e}), "
                             f"{'interior' if rs['interior_max'] else 'no interior maximum'}: < 1e-2",
                             abs(over_step) < 1e-2))
            r3 = row("0.001 tau_slow", oname)["ratio_max_to_plateau"]
            verdicts.append((f"{pname} {oname} P2 1e-3 (max column): |ratio-1| = {abs(r3-1):.2e} < 5e-3",
                             abs(r3 - 1) < 5e-3))
            verdicts.append((f"{pname} {oname} REFUTER 1e-3 (max column): |ratio-1| = {abs(r3-1):.2e} < 1e-2",
                             abs(r3 - 1) < 1e-2))
            r2 = row("0.01 tau_slow", oname)["ratio_max_to_plateau"]
            verdicts.append((f"{pname} {oname} P3 1e-2 (max column): |ratio-1| = {abs(r2-1):.2e} in [3e-3, 5e-2]",
                             3e-3 <= abs(r2 - 1) <= 5e-2))
            r1 = row("0.1 tau_slow", oname)["ratio_max_to_plateau"]
            verdicts.append((f"{pname} {oname} P4 1e-1 (max column): ratio = {r1:.4f} in [0.85, 0.97]",
                             0.85 <= r1 <= 0.97))
            r0 = row("1 tau_slow", oname)["ratio_max_to_plateau"]
            verdicts.append((f"{pname} {oname} P5 1 (max column): ratio = {r0:.4f} <= 0.6",
                             r0 <= 0.6))
            for cname in ("1 tau_relax", "10 tau_relax"):
                pf = row(cname, oname)["plateau_ratio"]
                verdicts.append((f"{pname} {oname} P6 {cname} (plateau column): |plateau-1| = "
                                 f"{abs(pf-1):.2e} < 1e-3", abs(pf - 1) < 1e-3))
            worst_exceed = max(r["ratio_max_to_plateau"] - 1.0 for r in rows
                               if r["point"] == pname and r["observable"] == oname
                               and r["ramp"] != "step" and r["interp"] == a.interp)
            verdicts.append((f"{pname} {oname} REFUTER overshoot: worst ramp excess "
                             f"{worst_exceed:+.2e} <= step overshoot {over_step:+.2e} + 1e-3",
                             worst_exceed <= over_step + 1e-3))
            for cname in ("0.001 tau_slow", "0.01 tau_slow", "0.1 tau_slow"):
                rr = row(cname, oname)
                De = 1.0 / rr["tau_ramp_over_tau_slow"]
                law = 0.5 - 1.0 / (6.0 * De)
                exact = De * (1.0 - De * (1.0 - np.exp(-1.0 / De)))
                obs = rr["ng_drift_fraction"] / rr["tau_ramp_over_tau_slow"]
                in_domain = rr["tau_ramp_over_tau_relax"] >= P8_DOMAIN_MIN_RELAX
                text = (f"{pname} {oname} P8 {cname}: drift/(td/tau_slow) = {obs:.4f}, "
                        f"truncated 1/2 - 1/(6 De) = {law:.4f} (exact {exact:.4f}), "
                        f"|diff| = {abs(obs-law):.4f} < 0.02")
                if not in_domain:
                    text += (f"   [OUTSIDE LAW DOMAIN: td = {rr['tau_ramp_over_tau_relax']:.1f} "
                             f"tau_relax < {P8_DOMAIN_MIN_RELAX:g}; not counted]")
                verdicts.append((text, (abs(obs - law) < 0.02) if in_domain else None))

    say()
    say("=" * 78)
    say("POST-HOC (a): PLATEAU column against the scalar formula De (1 - exp(-1/De)), De = tau_slow/tau_ramp")
    say("  plateau = eps(window start, ramp) / eps(window start, step)   [the column compared]")
    say("  max     = eps_max_post / eps_plateau                           [carries the ratio's transient peak]")
    say("=" * 78)
    say(f"  {'point':>16s} {'ramp':>14s} {'interp':>9s} {'obs':>5s} {'De':>9s} {'formula':>9s} {'plateau':>9s} "
        f"{'max':>9s} {'plateau/formula':>15s}")
    for r in rows:
        if r["tau_ramp_s"] == 0:
            continue
        De = 1.0 / r["tau_ramp_over_tau_slow"]
        formula = De * (1.0 - np.exp(-1.0 / De))
        say(f"  {r['point']:>16s} {r['ramp']:>14s} {r['interp']:>9s} {r['observable']:>5s} {De:9.2e} {formula:9.5f} "
            f"{r['plateau_ratio']:9.5f} {r['ratio_max_to_plateau']:9.5f} "
            f"{r['plateau_ratio']/formula:15.5f}")

    say()
    say("=" * 78)
    say("EXPM CROSS-CHECK: worst ||n_ode - n_expm|| / ||n_expm|| after the ramp (must be < 1e-6)")
    say("=" * 78)
    for r in rows:
        if r["observable"] == "shell":
            say(f"  {r['point']:>16s} {r['ramp']:>14s} {r['interp']:>9s}  state {r['expm_resid']:.2e}   "
                f"shell ratio {r['expm_resid_shell_ratio']:.2e}")

    say()
    say("=" * 78)
    say("CONVERGENCE: relative change of every reported number at rtol 1e-8 vs 1e-10")
    say("=" * 78)
    conv_keys = [kk for kk in rows[0] if kk.startswith("conv_")]
    for ck in conv_keys:
        vals = np.array([r[ck] for r in rows])
        kmax = int(np.argmax(vals))
        say(f"  {ck:26s} worst {vals.max():.2e}  ({rows[kmax]['point']}, {rows[kmax]['ramp']}, "
            f"{rows[kmax]['observable']})   median {np.median(vals):.2e}")

    say()
    say("=" * 78)
    say("PREDICTIONS")
    say("=" * 78)
    n_fail = n_na = 0
    for text, ok in verdicts:
        tag = "N/A" if ok is None else ("HELD" if bool(ok) else "FAILED")
        say(f"  [{tag}] {text}")
        n_fail += int(ok is not None and not bool(ok))
        n_na += int(ok is None)
    n_counted = len(verdicts) - n_na
    say(f"  {n_counted - n_fail} of {n_counted} held ({n_na} not counted)")
    say(f"  runtime {time.time() - t_wall0:.1f} s")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "ramp_plateau"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "ramp_plateau.csv").open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (out / "ramp_plateau.txt").write_text("\n".join(hdr) + "\n" + "\n".join(out_lines) + "\n")
        say(f"\nwrote {out.relative_to(ROOT)}/ramp_plateau.csv ({len(rows)} rows), ramp_plateau.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
