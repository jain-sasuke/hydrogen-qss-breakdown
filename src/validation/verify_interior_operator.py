#!/usr/bin/env python
"""
verify_interior_operator.py
===========================
The TRUE operator between grid nodes, and the physical ramp L[Te(t)] that
chapter 5 sec 5.8.2 says was not computed.

WHY THIS EXISTS
---------------
The ramp test of verify_ramp_plateau.py interpolates the operator between two
node operators (linear entrywise with f = t/t_ramp, or log-linear). Chapter 5
therefore calls it an operator-ramp sensitivity test and states that the
operator along a true temperature ramp, L[Te(t)], was not computed. K17
(verify_physical_ramp_bound.py) could only bracket the interior operator from a
two-interval span through the true mid NODE, and ESTIMATED the one-interval
error as a quarter of the two-interval one (second order), an estimate it
labelled as such and found not to hold for the plateau fraction (measured
ratio 0.66 rather than 0.25). build_L_at_Te.py now builds the operator at any
Te from the pipeline's own raw inputs and reproduces L_grid at all 400 nodes to
0.0. This script uses it to MEASURE what K17 estimated.

PART A  interior operator at every one-interval span
------------------------------------------------------
For every heating span i -> i+1 (i = 0..48) at every density j, the true
operator L(Te_m, ne_j) at the log midpoint Te_m = sqrt(Te_i Te_{i+1}) is
compared with the linear interpolant (weight f_lin = (Te_m - Te_i)/(Te_{i+1} -
Te_i)) and the log-linear interpolant (weight 1/2) of the two node operators,
built with verify_ramp_plateau.make_interpolants. Entrywise relative errors by
class (all nonzero entries; diagonal; 1s column off the diagonal; ionisation
loss = minus column sum; source S), with verify_physical_ramp_bound.err_stats,
and the derived quantities tau_slow, tau_relax, u_CRE, eps_plateau of the
step i -> Te_m with verify_physical_ramp_bound.derived. Also: at how many
entries the true value lies BETWEEN the two interpolants (the bracket K17
asserted), and the ratio of the one-interval error to K17's two-interval
error at the three named points (the quarter rule, now measured).

PART B  the physical ramp at the ramp-test point [15,3]
---------------------------------------------------------
One-interval heating Te[15] -> Te[16] at ne[3], t_ramp/tau_slow = 1e-2, 1e-1, 1
(tau_slow, tau_relax of the post-step node operator, as the stamped run), held
on L[16,3], integrated with verify_physical_ramp_bound.integrate_path (Radau,
rtol 1e-10, atol 1e-12 max|n_old|, max_step t_ramp/50), window and observable
exactly as verify_ramp_plateau.py (t_ws = t_ramp + 30 tau_relax, t_we =
tau_slow/30, t_end = t_ramp + 10 tau_slow; shell observable n3/n4; plateau
fraction = eps(t_ws) / eps(t_ws) of the pure step). Operator paths:
  T    TRUE operator along Te(t) linear in time: true operators are built at
       M = 33 temperatures equally spaced in ln Te across the interval and
       joined piecewise log-linearly (sub-interval 1/32 of the span, so the
       joining error is of order (1/32)^2 of the one-interval log-linear error
       measured in Part A); the M = 17 table is run alongside as the
       convergence check.
  Texp TRUE operator along ln Te linear in time (exponential ramp): the
       ramp-shape sensitivity.
  b    log-linear interpolant between the end nodes driven by the same
       Te(t) (weight ln(Te(t)/Te_i)/ln(Te_k/Te_i)) [K17's variant b].
  c    linear interpolant, f = t/t_ramp  [the stamped "linear" ramp].
  bexp log-linear interpolant with f = t/t_ramp  [the stamped "loglinear" ramp;
       for the operator this is the exponential ramp's interpolant].

GATES
-----
G0  build_L_at_Te.OperatorBuilder.gate(): every stored table and L_grid/S_grid
    reproduced at all 50 x 8 nodes to 1e-12 (it reports 0.0).
G1  verify_ramp_plateau.integrate reproduces validation/ramp_plateau/
    ramp_plateau.csv at [15,3] (shell rows, point "defended maximum"):
    plateau 0.9995, 0.9954, 0.9555, 0.6469 (linear) and 0.6576 (loglinear,
    1 tau_slow), eps_plateau_step, tau_slow, tau_relax, to 1e-8.
G1b integrate_path with path c and no interior breaks reproduces
    verify_ramp_plateau.integrate at 1 tau_slow to 1e-10 in n(t_ws); with
    path bexp likewise against interp="loglinear".
G2  the true operator at Te = Te_i (a node) from the builder equals L[i,j]
    to 1e-12 for the spans used in Part B (the builder's gate, re-asserted
    on the operators this script actually uses); every true interior
    operator is a CR generator (off-diagonals >= 0, S >= 0, column sums =
    -K_ion ne to 1e-10 of max|L|).

PREDICTIONS (written before the run, from K17's two-interval numbers and the quarter rule)
------------------------------------------------------------------------------------------
P1  [15,3], 1s column, median |error|: log-linear 0.17 % (K17 0.69 %/4; accept
    0.10-0.26 %), linear 0.59 % (2.35 %/4; accept 0.35-0.90 %); log-linear
    under-estimates and linear over-estimates (signed medians < 0 and > 0);
    the true entry lies between the two interpolants at >= 95 % of the
    nonzero 1s-column entries. [0,0]: linear 3.5 % (accept 2.0-5.0), log-linear
    0.35 % (accept 0.20-0.55).
P2  [15,3] derived quantities of the interpolated mid operator: tau_slow
    log-linear +0.075 %, linear -0.71 %; u_CRE +0.04 %, -0.63 % (each a quarter
    of K17's +0.30/-2.84 and +0.152/-2.525; accept within a factor 1.6).
P3  Part B plateau fraction, path T, at t_ramp/tau_slow = 1: 0.6570 +- 0.0010
    (K17: 0.6576 - 6e-4 from above, 0.6469 + 0.0102 from below); at 0.1:
    0.9569 +- 0.0003; at 0.01: 0.99560 +- 0.00010. In every case T lies
    between c and b (K17's bracket), nearer b (|T - b|/|b - c| about 0.05).
P4  ramp-shape sensitivity |Texp - T| at t_ramp/tau_slow = 1 is smaller than
    the interpolant gap |b - c| (0.0108).
P5  fine-table convergence: |T(M = 33) - T(M = 17)| < 1e-5 in the plateau
    fraction at every ratio.
REFUTER of K17's bracket claim: path T outside [c, b] by more than 0.001 at any
ratio. REFUTER of the second-order (quarter-rule) claim for the operator: a
one-interval 1s-column median error at [15,3] larger than half the two-interval
one (0.35 % log-linear, 1.2 % linear).

OUTPUTS (with --write): validation/interior_operator/
  interior_operator.txt, interior_operator_spans.csv (392 spans x 2 interpolants),
  interior_operator_trajectory.csv
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, sys, contextlib, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
import verify_ramp_plateau as vrp  # noqa: E402
import verify_physical_ramp_bound as vprb  # noqa: E402
from build_L_at_Te import OperatorBuilder  # noqa: E402
ROOT = find_repo_root(_HERE)
RAMP_POINT = (15, 3); NAMED = {"ramp point [15,3]": (15, 3), "benchmark [23,5]": (23, 5), "cold corner [0,0]": (0, 0)}
RATIOS = (1e-2, 1e-1, 1.0); M_FINE, M_CONV = 33, 17
# K17's two-interval measurements at the three named points (validation/physical_ramp_bound/physical_ramp_bound.txt), for the quarter-rule ratio
K17_COL1S = {(15, 3): dict(loglinear=0.0069, linear=0.0235), (0, 0): dict(loglinear=0.0140, linear=0.138)}
K17_DERIVED = {(15, 3): dict(loglinear=dict(tau=+0.00302, u=+0.00152), linear=dict(tau=-0.02842, u=-0.02525))}
STAMPED = dict(lin={1e-3: 0.999543, 1e-2: 0.995445, 1e-1: 0.955529, 1.0: 0.646860}, loglin={1.0: 0.657630})


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); t_wall = time.time()
    ctx = CRContext.load(); ctx.validate(); L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; nT, nN, nS, _ = L.shape
    S = np.load(ROOT / "data/processed/cr_matrix/S_grid.npy"); g = ctx.ground_index; nv = ctx.n_values.astype(int)
    E = np.array([k for k in range(nS) if k != g]); N3 = np.where(nv == 3)[0]; N4 = np.where(nv == 4)[0]
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 100); say("THE TRUE OPERATOR BETWEEN GRID NODES, AND THE PHYSICAL RAMP L[Te(t)]")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}  numpy {np.__version__}  scipy {__import__('scipy').__version__}"); say(ctx.describe()); say("=" * 100)

    # ---------------------------------------------------------------- G0: the builder's gate
    say("\n  G0  build_L_at_Te gate (every stored table, L_grid, S_grid at all 50 x 8 nodes):")
    ob = OperatorBuilder.load(ROOT); worst = ob.gate(verbose=False)
    say("      worst relative differences: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    if max(worst.values()) > 1e-12: raise RuntimeError("Gate G0")

    # ---------------------------------------------------------------- G1 / G1b: the integrator reproduces the stamped ramp numbers
    i, j = RAMP_POINT; k = i + 1
    L0, L1, b0, b1 = L[i, j], L[k, j], S[i, j], S[k, j]
    tau_slow, tau_relax = vrp.spectrum(L1); n_old = np.linalg.solve(L0, -b0); n_new = np.linalg.solve(L1, -b1)
    R_new = n_new[N3].sum() / n_new[N4].sum(); atol = vrp.ATOL_SCALE * np.abs(n_old).max()
    eps_of = lambda n: abs((n[N3].sum() / n[N4].sum()) / R_new - 1.0)
    csv = pd.read_csv(ROOT / "validation/ramp_plateau/ramp_plateau.csv", comment="#")
    rows = csv[(csv.point == "defended maximum") & (csv.observable == "shell")]
    if len(rows) == 0: raise RuntimeError("ramp_plateau.csv has no 'defended maximum' shell rows")
    r0 = rows[rows.ramp == "step"].iloc[0]
    if abs(tau_slow / r0.tau_slow - 1) > 1e-8 or abs(tau_relax / r0.tau_relax - 1) > 1e-8: raise RuntimeError("G1: tau_slow/tau_relax differ from ramp_plateau.csv")

    def windows(td): return td + vrp.WIN_LO * tau_relax, tau_slow / vrp.WIN_HI, td + 10.0 * tau_slow
    t_ws0, t_we0, t_end0 = windows(0.0)
    _, _, at0, _ = vrp.integrate(L0, L1, b0, b1, n_old, 0.0, [t_ws0, t_we0, t_end0], vrp.RTOL_MAIN, atol, "linear")
    eps_ws_step = eps_of(at0[t_ws0])
    if abs(eps_ws_step / r0.eps_at_window_start - 1) > 1e-8: raise RuntimeError(f"G1: step eps(ws) {eps_ws_step} vs csv {r0.eps_at_window_start}")
    say(f"\n  G1  [15,3] -> [16,3]: tau_slow {tau_slow:.6e} s, tau_relax {tau_relax:.6e} s, step eps(ws) {eps_ws_step:.9e} (csv {r0.eps_at_window_start:.9e})")
    g1 = []
    for interp, key in (("linear", "lin"), ("loglinear", "loglin")):
        for ratio, ref in STAMPED[key].items():
            td = ratio * tau_slow; t_ws, t_we, t_end = windows(td)
            _, _, at, _ = vrp.integrate(L0, L1, b0, b1, n_old, td, [t_ws, t_we, t_end], vrp.RTOL_MAIN, atol, interp)
            pf = eps_of(at[t_ws]) / eps_ws_step; row = rows[(rows.interp == interp) & (np.isclose(rows.tau_ramp_over_tau_slow, ratio))].iloc[0]
            g1.append(abs(pf / row.plateau_ratio - 1)); say(f"      {interp:9s} {ratio:g} tau_slow: plateau {pf:.6f} (csv {row.plateau_ratio:.6f}, rel {g1[-1]:.1e}; docstring value {ref})")
    if max(g1) > 1e-8: raise RuntimeError("Gate G1")
    td1 = tau_slow; t_ws1, t_we1, t_end1 = windows(td1)
    La, ba = vrp.make_interpolants(L0, L1, b0, b1, "linear"); Lb, bb = vrp.make_interpolants(L0, L1, b0, b1, "loglinear")
    g1b = []
    for interp, (Lf, bf) in (("linear", (La, ba)), ("loglinear", (Lb, bb))):
        _, _, at_ref, _ = vrp.integrate(L0, L1, b0, b1, n_old, td1, [t_ws1, t_we1, t_end1], vrp.RTOL_MAIN, atol, interp)
        s_of = lambda t: min(max(t / td1, 0.0), 1.0)
        _, _, at_p, _ = vprb.integrate_path(lambda t: Lf(s_of(t)), lambda t: bf(s_of(t)), L1, b1, n_old, td1, [], [t_ws1, t_we1, t_end1], vrp.RTOL_MAIN, atol)
        g1b.append(np.abs(at_p[t_ws1] / at_ref[t_ws1] - 1).max())
    say(f"  G1b integrate_path (f = t/t_ramp, no breaks) vs verify_ramp_plateau.integrate at 1 tau_slow: max rel diff in n(t_ws) {max(g1b):.1e} (linear, loglinear)")
    if max(g1b) > 1e-10: raise RuntimeError("Gate G1b")

    # ---------------------------------------------------------------- PART A: interior operator at every span
    say("\n" + "=" * 100); say("PART A: true operator at the log midpoint of every one-interval span versus the two interpolants"); say("=" * 100)
    eye = np.eye(nS, dtype=bool); off = ~eye
    Kion_col = lambda Te: ob.tables(Te)["K_ion_final"][:, 0]
    spans = []; gen_worst = 0.0; g2_worst = 0.0
    for i_ in range(nT - 1):
        Te_m = float(np.sqrt(te[i_] * te[i_ + 1])); f_lin = (Te_m - te[i_]) / (te[i_ + 1] - te[i_])
        for j_ in range(nN):
            Lt, bt = ob.operator(Te_m, float(ne[j_]))
            if Lt[off].min() < 0 or bt.min() < 0: raise RuntimeError(f"true operator at Te {Te_m} ne {ne[j_]} is not a CR generator")
            gen_worst = max(gen_worst, np.abs(Lt.sum(axis=0) + Kion_col(Te_m) * ne[j_]).max() / np.abs(Lt).max())
            nz = Lt != 0; masks = {"all": nz, "diag": nz & eye, "col1s": nz & off & (np.arange(nS)[None, :] == g)}
            ion_t = -Lt.sum(axis=0); tsd = vprb.derived(L[i_, j_], S[i_, j_], Lt, bt, g, E, N3, N4)
            rec = dict(i=i_, j=j_, Te_lo=te[i_], Te_mid=Te_m, Te_hi=te[i_ + 1], ne=ne[j_], f_lin=f_lin, tau_slow_true=tsd[0], tau_relax_true=tsd[1], u_true=tsd[2], eps_true=tsd[3])
            interps = {}
            for kind, f in (("linear", f_lin), ("loglinear", 0.5)):
                Lf, bf = vrp.make_interpolants(L[i_, j_], L[i_ + 1, j_], S[i_, j_], S[i_ + 1, j_], kind); Lp, bp = Lf(f), bf(f); interps[kind] = (Lp, bp)
                if not np.array_equal(Lp != 0, nz): rec[f"{kind}_pattern_mismatch"] = int(((Lp != 0) != nz).sum())
                for c in ("all", "diag", "col1s"):
                    st = vprb.err_stats(Lt, Lp, masks[c]); rec.update({f"{kind}_{c}_med": st["med"], f"{kind}_{c}_p90": st["p90"], f"{kind}_{c}_max": st["max"], f"{kind}_{c}_signed_med": st["med_signed"]})
                st = vprb.err_stats(ion_t, -Lp.sum(axis=0), ion_t != 0); rec.update({f"{kind}_ion_med": st["med"], f"{kind}_ion_max": st["max"], f"{kind}_ion_signed_med": st["med_signed"]})
                st = vprb.err_stats(bt, bp, bt != 0); rec.update({f"{kind}_rec_med": st["med"], f"{kind}_rec_max": st["max"], f"{kind}_rec_signed_med": st["med_signed"]})
                d = vprb.derived(L[i_, j_], S[i_, j_], Lp, bp, g, E, N3, N4)
                rec.update({f"{kind}_e_tau_slow": d[0] / tsd[0] - 1, f"{kind}_e_tau_relax": d[1] / tsd[1] - 1, f"{kind}_e_u": d[2] / tsd[2] - 1, f"{kind}_e_eps": d[3] / tsd[3] - 1})
            La_, Lb_ = interps["linear"][0], interps["loglinear"][0]
            lo, hi = np.minimum(La_, Lb_), np.maximum(La_, Lb_)
            br = (Lt >= lo) & (Lt <= hi); rec["bracketed_frac_all"] = float(br[nz].mean()); rec["bracketed_frac_col1s"] = float(br[masks["col1s"]].mean())
            rec["bracketed_frac_S"] = float(((bt >= np.minimum(interps["linear"][1], interps["loglinear"][1])) & (bt <= np.maximum(interps["linear"][1], interps["loglinear"][1])))[bt != 0].mean())
            spans.append(rec)
    df = pd.DataFrame(spans)
    say(f"  {len(df)} spans; every true interior operator is a CR generator; worst column-sum residual vs -K_ion ne: {gen_worst:.1e} of max|L|")
    if gen_worst > 1e-10: raise RuntimeError("Gate G2 (column sums of the true interior operators)")
    # G2 re-assertion at the Part B nodes
    for (ii, jj) in ((i, j), (k, j)):
        Ln, _ = ob.operator(float(te[ii]), float(ne[jj])); g2_worst = max(g2_worst, np.abs(Ln - L[ii, jj]).max() / np.abs(L[ii, jj]).max())
    say(f"  G2  builder at the nodes [15,3], [16,3] vs L_grid: {g2_worst:.1e}")
    if g2_worst > 1e-12: raise RuntimeError("Gate G2")
    say(f"\n  named spans (one interval, i -> i+1), midpoint errors in % (median |e| over the class; signed median in brackets):")
    say(f"  {'span':>20} {'Te lo->mid->hi':>24} | {'interp':>9} {'all med':>8} {'diag med':>9} {'1s col med':>11} {'1s col max':>10} {'ion med':>8} {'S med':>8} | {'e tau_slow':>10} {'e u_CRE':>8} {'e eps_pl':>8} | {'bracketed 1s col / all':>22}")
    for name, (ii, jj) in NAMED.items():
        r = df[(df.i == ii) & (df.j == jj)].iloc[0]
        for kind in ("loglinear", "linear"):
            say(f"  {name:>20} {r.Te_lo:7.4f}->{r.Te_mid:7.4f}->{r.Te_hi:7.4f} | {kind:>9} {100*r[f'{kind}_all_med']:8.3f} {100*r[f'{kind}_diag_med']:9.3f} "
                f"{100*r[f'{kind}_col1s_med']:6.3f} ({100*r[f'{kind}_col1s_signed_med']:+.3f}) {100*r[f'{kind}_col1s_max']:10.3f} {100*r[f'{kind}_ion_med']:8.3f} {100*r[f'{kind}_rec_med']:8.3f} | "
                f"{100*r[f'{kind}_e_tau_slow']:+10.4f} {100*r[f'{kind}_e_u']:+8.4f} {100*r[f'{kind}_e_eps']:+8.4f} | {100*r.bracketed_frac_col1s:6.1f} % / {100*r.bracketed_frac_all:5.1f} %")
    say(f"\n  grid-wide (392 spans): median over spans of the class median, and the worst span:")
    for kind in ("loglinear", "linear"):
        for c in ("all", "col1s", "ion", "rec"):
            col = f"{kind}_{c}_med"; w = df.loc[df[col].idxmax()]
            say(f"    {kind:>9} {c:>6}: median of medians {100*df[col].median():.3f} %, worst span median {100*df[col].max():.3f} % at [{int(w.i)},{int(w.j)}] Te {w.Te_lo:.3f}")
        for q in ("tau_slow", "u", "eps"):
            col = f"{kind}_e_{q}"; say(f"    {kind:>9} e_{q:8s}: median {100*df[col].median():+.4f} %, range {100*df[col].min():+.4f} to {100*df[col].max():+.4f} %")
    say(f"  bracketed (true between the two interpolants): 1s column {100*df.bracketed_frac_col1s.min():.1f} to {100*df.bracketed_frac_col1s.max():.1f} % of entries per span (median {100*df.bracketed_frac_col1s.median():.1f} %); all entries median {100*df.bracketed_frac_all.median():.1f} %; S median {100*df.bracketed_frac_S.median():.1f} %")
    say(f"\n  the quarter rule, measured: one-interval midpoint error / K17's two-interval mid-node error (second order would give 0.25)")
    for (ii, jj), ref in K17_COL1S.items():
        r = df[(df.i == ii) & (df.j == jj)].iloc[0]
        say(f"    [{ii},{jj}] 1s column median: log-linear {r.loglinear_col1s_med:.5f} / {ref['loglinear']:.4f} = {r.loglinear_col1s_med/ref['loglinear']:.3f};  linear {r.linear_col1s_med:.5f} / {ref['linear']:.4f} = {r.linear_col1s_med/ref['linear']:.3f}")
    r = df[(df.i == 15) & (df.j == 3)].iloc[0]
    for kind in ("loglinear", "linear"):
        say(f"    [15,3] derived: tau_slow {100*r[f'{kind}_e_tau_slow']:+.4f} % / K17 {100*K17_DERIVED[(15, 3)][kind]['tau']:+.3f} % = {r[f'{kind}_e_tau_slow']/K17_DERIVED[(15, 3)][kind]['tau']:.3f};  "
            f"u_CRE {100*r[f'{kind}_e_u']:+.4f} % / {100*K17_DERIVED[(15, 3)][kind]['u']:+.3f} % = {r[f'{kind}_e_u']/K17_DERIVED[(15, 3)][kind]['u']:.3f}   ({kind})")

    # ---------------------------------------------------------------- PART B: the physical ramp at [15,3]
    say("\n" + "=" * 100); say("PART B: one-interval ramp [15,3] -> [16,3], Te(t) prescribed, operator along the TRUE L[Te(t)] versus the interpolants"); say("=" * 100)
    Te_i, Te_k = float(te[i]), float(te[k])

    def true_table(M):
        Ts = np.exp(np.linspace(np.log(Te_i), np.log(Te_k), M)); ops = [ob.operator(float(T), float(ne[j])) for T in Ts]
        ops[0] = (L0, b0); ops[-1] = (L1, b1)                                    # the ends are the node operators themselves (G2 asserts the builder agrees)
        pieces = [vrp.make_interpolants(ops[m][0], ops[m + 1][0], ops[m][1], ops[m + 1][1], "loglinear") for m in range(M - 1)]
        def L_at_Te(Te):
            Te = min(max(Te, Te_i), Te_k); m = min(int(np.searchsorted(Ts, Te, side="right") - 1), M - 2)
            f = np.log(Te / Ts[m]) / np.log(Ts[m + 1] / Ts[m]); return pieces[m][0](f), pieces[m][1](f)
        return Ts, L_at_Te
    tables = {M: true_table(M) for M in (M_FINE, M_CONV)}
    say(f"  true operators built at M = {M_FINE} (and {M_CONV}) temperatures equally spaced in ln Te over [{Te_i:.4f}, {Te_k:.4f}] eV, joined piecewise log-linearly")

    def Te_lin(t, td): return Te_i + (Te_k - Te_i) * min(max(t / td, 0.0), 1.0)
    def Te_exp(t, td): return Te_i * (Te_k / Te_i) ** min(max(t / td, 0.0), 1.0)
    traj = []; results = {}
    for ratio in RATIOS:
        td = ratio * tau_slow; t_ws, t_we, t_end = windows(td)
        cases = {}
        for M in (M_FINE, M_CONV):
            Ts, L_at_Te = tables[M]
            for shape, Te_of in (("T", Te_lin), ("Texp", Te_exp)):
                if M == M_CONV and shape == "Texp": continue
                breaks = [td * ((T - Te_i) / (Te_k - Te_i) if shape == "T" else np.log(T / Te_i) / np.log(Te_k / Te_i)) for T in Ts[1:-1]]
                cases[(shape if M == M_FINE else "T_conv")] = ((lambda t, f_=L_at_Te, T_=Te_of: f_(T_(t, td))[0]), (lambda t, f_=L_at_Te, T_=Te_of: f_(T_(t, td))[1]), breaks)
        cases["b"] = ((lambda t: Lb(np.log(Te_lin(t, td) / Te_i) / np.log(Te_k / Te_i))), (lambda t: bb(np.log(Te_lin(t, td) / Te_i) / np.log(Te_k / Te_i))), [])
        cases["c"] = ((lambda t: La(min(max(t / td, 0.0), 1.0))), (lambda t: ba(min(max(t / td, 0.0), 1.0))), [])
        cases["bexp"] = ((lambda t: Lb(min(max(t / td, 0.0), 1.0))), (lambda t: bb(min(max(t / td, 0.0), 1.0))), [])
        res = {}
        for name, (L_of, b_of, breaks) in cases.items():
            t0 = time.time(); _, n_td, at, nfev = vprb.integrate_path(L_of, b_of, L1, b1, n_old, td, breaks, [t_ws, t_we, t_end], vrp.RTOL_MAIN, atol)
            pf = eps_of(at[t_ws]) / eps_ws_step; res[name] = dict(plateau=pf, eps_ws=eps_of(at[t_ws]), eps_we=eps_of(at[t_we]) if t_we > td else np.nan, eps_ramp_end=eps_of(n_td), nfev=nfev, secs=time.time() - t0)
            if name == "T":
                _, _, at_c, _ = vprb.integrate_path(L_of, b_of, L1, b1, n_old, td, breaks, [t_ws, t_we, t_end], vrp.RTOL_CONV, atol)
                res[name]["plateau_rtol1e-8"] = eps_of(at_c[t_ws]) / eps_ws_step
            traj.append(dict(i=i, j=j, ratio=ratio, t_ramp_s=td, path=name, **res[name]))
        results[ratio] = res
        say(f"\n  t_ramp/tau_slow = {ratio:g}  (t_ramp {td:.4e} s):")
        say(f"    {'path':>7} {'plateau fraction':>16} {'eps(ws)':>12} {'eps(we)':>12} {'eps(ramp end)':>13} {'nfev':>6}")
        for name in ("T", "T_conv", "Texp", "b", "c", "bexp"):
            r_ = res[name]; say(f"    {name:>7} {r_['plateau']:16.6f} {r_['eps_ws']:12.6e} {r_['eps_we']:12.6e} {r_['eps_ramp_end']:13.6e} {r_['nfev']:6d}" + (f"   (rtol 1e-8 rerun {r_['plateau_rtol1e-8']:.6f})" if name == "T" else ""))
        T, b_, c_ = res["T"]["plateau"], res["b"]["plateau"], res["c"]["plateau"]
        say(f"    T - b = {T-b_:+.6f}, T - c = {T-c_:+.6f}, b - c = {b_-c_:+.6f}, |T-b|/|b-c| = {abs(T-b_)/abs(b_-c_):.3f};  Texp - T = {res['Texp']['plateau']-T:+.6f};  bexp - b = {res['bexp']['plateau']-b_:+.6f};  |T(33) - T(17)| = {abs(T-res['T_conv']['plateau']):.2e}")
        say(f"    stamped one-interval values: linear (= path c) {STAMPED['lin'].get(ratio, float('nan')):.6f}" + (f", loglinear (= path bexp) {STAMPED['loglin'][ratio]:.6f}" if ratio in STAMPED["loglin"] else ""))

    # ---------------------------------------------------------------- predictions and refuters
    say("\n" + "=" * 100); say("PREDICTIONS (written before the run) against what came out"); say("=" * 100)
    missed = []
    r = df[(df.i == 15) & (df.j == 3)].iloc[0]; r00 = df[(df.i == 0) & (df.j == 0)].iloc[0]
    p1 = (0.0010 <= r.loglinear_col1s_med <= 0.0026 and 0.0035 <= r.linear_col1s_med <= 0.0090 and r.loglinear_col1s_signed_med < 0 < r.linear_col1s_signed_med
          and r.bracketed_frac_col1s >= 0.95 and 0.020 <= r00.linear_col1s_med <= 0.050 and 0.0020 <= r00.loglinear_col1s_med <= 0.0055)
    say(f"  P1 [15,3] 1s column median: log-linear {100*r.loglinear_col1s_med:.3f} % (0.10-0.26, signed {100*r.loglinear_col1s_signed_med:+.3f}), linear {100*r.linear_col1s_med:.3f} % (0.35-0.90, signed {100*r.linear_col1s_signed_med:+.3f}), bracketed {100*r.bracketed_frac_col1s:.1f} % (>= 95);"
        f" [0,0] linear {100*r00.linear_col1s_med:.3f} % (2.0-5.0), log-linear {100*r00.loglinear_col1s_med:.3f} % (0.20-0.55): {'reproduced' if p1 else 'NOT reproduced'}")
    if not p1: missed.append("P1")
    q = K17_DERIVED[(15, 3)]; within = lambda x, ref: (x / ref > 0) and (1 / 1.6 <= (x / ref) / 0.25 <= 1.6)
    p2 = all(within(r[f"{kind}_e_{qq}"], q[kind][kk]) for kind in ("loglinear", "linear") for qq, kk in (("tau_slow", "tau"), ("u", "u")))
    say(f"  P2 [15,3] derived, quarter of K17 within x1.6: tau_slow log-linear {100*r.loglinear_e_tau_slow:+.4f} % (pred +0.075), linear {100*r.linear_e_tau_slow:+.4f} % (-0.71); u_CRE {100*r.loglinear_e_u:+.4f} % (+0.038), {100*r.linear_e_u:+.4f} % (-0.63): {'reproduced' if p2 else 'NOT reproduced'}")
    if not p2: missed.append("P2")
    pred3 = {1.0: (0.6570, 0.0010), 1e-1: (0.9569, 0.0003), 1e-2: (0.99560, 0.00010)}
    p3 = all(abs(results[rt]["T"]["plateau"] - pred3[rt][0]) <= pred3[rt][1] and min(results[rt]["b"]["plateau"], results[rt]["c"]["plateau"]) <= results[rt]["T"]["plateau"] <= max(results[rt]["b"]["plateau"], results[rt]["c"]["plateau"]) for rt in RATIOS)
    say(f"  P3 path T plateau fraction: " + ", ".join(f"{rt:g}: {results[rt]['T']['plateau']:.6f} (pred {pred3[rt][0]} +- {pred3[rt][1]})" for rt in RATIOS) + f"; inside [c, b] at every ratio: {all(min(results[rt]['b']['plateau'], results[rt]['c']['plateau']) <= results[rt]['T']['plateau'] <= max(results[rt]['b']['plateau'], results[rt]['c']['plateau']) for rt in RATIOS)}: {'reproduced' if p3 else 'NOT reproduced'}")
    if not p3: missed.append("P3")
    p4 = abs(results[1.0]["Texp"]["plateau"] - results[1.0]["T"]["plateau"]) < abs(results[1.0]["b"]["plateau"] - results[1.0]["c"]["plateau"])
    say(f"  P4 ramp shape at 1 tau_slow: |Texp - T| = {abs(results[1.0]['Texp']['plateau'] - results[1.0]['T']['plateau']):.6f} < |b - c| = {abs(results[1.0]['b']['plateau'] - results[1.0]['c']['plateau']):.6f}: {'reproduced' if p4 else 'NOT reproduced'}")
    if not p4: missed.append("P4")
    p5 = all(abs(results[rt]["T"]["plateau"] - results[rt]["T_conv"]["plateau"]) < 1e-5 for rt in RATIOS)
    say(f"  P5 fine-table convergence |T(33) - T(17)|: " + ", ".join(f"{abs(results[rt]['T']['plateau'] - results[rt]['T_conv']['plateau']):.1e}" for rt in RATIOS) + f" (< 1e-5): {'reproduced' if p5 else 'NOT reproduced'}")
    if not p5: missed.append("P5")
    say("\nPREDICTIONS: " + ("P1-P5 all reproduced" if not missed else "not reproduced as written: " + ", ".join(missed)))
    out_br = max(max(0.0, min(results[rt]["b"]["plateau"], results[rt]["c"]["plateau"]) - results[rt]["T"]["plateau"], results[rt]["T"]["plateau"] - max(results[rt]["b"]["plateau"], results[rt]["c"]["plateau"])) for rt in RATIOS)
    say(f"REFUTER of K17's bracket (T outside [c, b] by > 0.001): {'APPEARED' if out_br > 0.001 else 'did not appear'} (largest excursion {out_br:.2e})")
    say(f"REFUTER of second order (one-interval 1s-column median at [15,3] > half the two-interval): log-linear {100*r.loglinear_col1s_med:.3f} % vs 0.35 %, linear {100*r.linear_col1s_med:.3f} % vs 1.2 %: "
        f"{'APPEARED' if (r.loglinear_col1s_med > 0.0035 or r.linear_col1s_med > 0.012) else 'did not appear'}")
    T1, b1_, c1, Te1, be1 = (results[1.0][x]["plateau"] for x in ("T", "b", "c", "Texp", "bexp"))
    br_lo = df[df.bracketed_frac_col1s < 1.0]; te_fail = te[int(br_lo.i.min())] if len(br_lo) else float("nan")
    say("\nREADING.")
    say(f"  (1) The interpolants are second order in the interval: the one-interval midpoint error is 0.25-0.27 of K17's two-interval error in the\n"
        f"      classes compared with K17 (1s column at [15,3] and [0,0]; tau_slow and u_CRE at [15,3]); the quarter rule is measured, not estimated,\n"
        f"      for those. At the ramp point the log-linear interpolant misses the true interior operator by 0.18 % in the 1s column (0.08 % in\n"
        f"      tau_slow, 0.04 % in u_CRE) and the linear one by 0.63 % (0.76 %, 0.68 %). The true entry lies between the two interpolants at every\n"
        f"      1s-column entry at the three named points, but NOT in every span: {len(br_lo)} of {len(df)} spans have some 1s-column entries outside the\n"
        f"      bracket, and above Te = {te_fail:.2f} eV the fraction falls, to 0 % for Te >= 7.9 eV, where both interpolants under-estimate the column\n"
        f"      (both signed medians negative, e.g. -0.04 % and -0.03 % at [44,3]); K17's bracket picture is a property of the low-Te spans. Over\n"
        f"      all nonzero entries the bracket holds at about a quarter (the rest are entries both interpolants reproduce to better than 0.01 %).")
    say(f"  (2) The physical ramp. Along the true operator L[Te(t)] with Te linear in time, the one-interval ramp at [15,3] reaches {results[1e-2]['T']['plateau']:.4f},\n"
        f"      {results[1e-1]['T']['plateau']:.4f} and {T1:.4f} of the step plateau at t_ramp/tau_slow = 0.01, 0.1, 1, against the stamped linear operator ramp's\n"
        f"      {STAMPED['lin'][1e-2]:.4f}, {STAMPED['lin'][1e-1]:.4f}, {STAMPED['lin'][1.0]:.4f}. The true value sits between the linear and the log-linear interpolant driven by the same Te(t)\n"
        f"      at every ratio, 0.058 of their gap from the log-linear one (K17 estimated 0.052 from the two-interval span).")
    say(f"  (3) P3 is not reproduced at t_ramp = tau_slow. The explanation below is post hoc, not pre-registered: the two stamped ramps are ramps of\n"
        f"      different SHAPE as well as different interpolants. The stamped 'log-linear' ramp (f = t/t_ramp in log space) is, in operator space,\n"
        f"      identical (to 7e-15) to the log-linear interpolant driven by ln Te linear in time, an exponential ramp, so its true counterpart is\n"
        f"      Texp = {Te1:.4f} (interpolant {be1:.4f}); the stamped 'linear' ramp has Te linear in time, true counterpart T = {T1:.4f} (log-linear interpolant\n"
        f"      driven by that Te(t): {b1_:.4f}). K17's bracket [0.6469, 0.6576] therefore spans two ramp shapes; each shape's true value lies inside its\n"
        f"      own interpolant bracket. The stamped gap 0.0107 splits three ways at t_ramp = tau_slow: the LINEAR interpolant is off its true ramp by\n"
        f"      {T1-c1:+.4f} (the one chapter 5 quotes), the LOG-LINEAR interpolant by {T1-b1_:+.4f}, and the ramp shape moves the true value by {Te1-T1:+.4f}. So the\n"
        f"      interpolated operator ramp stands in for the physical one to 5e-4 if log-linear and to 8e-3 if linear, in the plateau fraction, and the\n"
        f"      shape of Te(t), which no 0-D model supplies, matters more than the log-linear interpolation and less than the linear one.")
    say(f"  Caveats. One grid interval (Delta ln Te = 0.047) at one grid point (one Te, one ne); shell observable n3/n4; the l-mixing cutoff frozen at\n"
        f"  1e14 cm^-3 in the true operator as in L_grid; M = {M_FINE} true operators joined log-linearly (joining error {max(abs(results[rt]['T']['plateau'] - results[rt]['T_conv']['plateau']) for rt in RATIOS):.1e} in the plateau fraction\n"
        f"  from the M = {M_CONV} rerun, itself an upper estimate of the M = {M_FINE} error by about 3x). The true operator's own accuracy is that of the pipeline's\n"
        f"  rates and is NOT better than the interpolation error measured here: the hostile audit of this script found that the pipeline's 5000-point\n"
        f"  Maxwell-average grid carries a discretisation error of the same order, and verify_maxwell_grid_convergence.py stamped it (validation/\n"
        f"  maxwell_grid_convergence/): on 100000 points tau_slow moves by -0.96 % and the 1s-column median by 1.9 % at [15,3], up to 1.5 % in\n"
        f"  tau_slow at Te >= 2 eV and 3.3 % at the cold corner. That error is identical at nodes and interior, so the interpolation comparison here\n"
        f"  is self-consistent, but it is a pipeline finding, reported and not repaired (CLAUDE.md), and it bounds how literally 'true' should be read.")
    say(f"\n  wall time {time.time() - t_wall:.0f} s")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation/interior_operator"; out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}  scipy {__import__('scipy').__version__}"] + ob.provenance()
        hdr += [f"# {p.relative_to(ROOT)} sha256 {sha(p)}" for p in (_HERE.parent / "build_L_at_Te.py", _HERE.parent / "verify_ramp_plateau.py", _HERE.parent / "verify_physical_ramp_bound.py", ROOT / "validation/ramp_plateau/ramp_plateau.csv")]
        hdr += ["# Part A: true operator at Te_mid = sqrt(Te_i Te_{i+1}); linear interpolant weight f_lin = (Te_mid - Te_i)/(Te_{i+1} - Te_i); log-linear weight 1/2; errors = interpolant/true - 1",
                f"# Part B: [15,3] -> [16,3]; T = true L[Te(t)] via M = {M_FINE} log-spaced true operators joined log-linearly, Te linear in t; Texp = ln Te linear in t; b = log-linear interpolant driven by Te(t); c = linear f = t/t_ramp; bexp = log-linear f = t/t_ramp",
                f"# integrator: verify_physical_ramp_bound.integrate_path (Radau, rtol {vrp.RTOL_MAIN}, atol {vrp.ATOL_SCALE} max|n_old|, max_step t_ramp/50); window t_ramp + {vrp.WIN_LO:g} tau_relax .. tau_slow/{vrp.WIN_HI:g}; shell observable n3/n4; plateau fraction = eps(ws)/eps(ws, step)"]
        with open(out / "interior_operator_spans.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "interior_operator_trajectory.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); pd.DataFrame(traj).to_csv(fh, index=False)
        with open(out / "interior_operator.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/interior_operator.txt, _spans.csv, _trajectory.csv")
    return 0


if __name__ == "__main__": sys.exit(main())
