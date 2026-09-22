#!/usr/bin/env python
"""
verify_maxwell_grid_convergence.py
==================================
Is the 5000-point energy grid of the CCC Maxwell average converged for the
quantities the thesis quotes?

WHY THIS EXISTS
---------------
compute_K_CCC.py averages every CCC excitation cross section over a Maxwellian
on a uniform 5000-point energy grid from dE + 1e-4 eV to the top of the data
(N_GRID = 5000, "validated: <2% error"). Chapter 2 sec:maxwellian says the grid
density "is measured, not defaulted" because a 500-point grid gives ~17 % error
at 1 eV. That tests the coarse direction only. The hostile audit of
verify_interior_operator.py rebuilt the average on 25000 points and found the
1s column of L moves by 1.6 % (median) and tau_slow by -0.84 % at Te = 2.07 eV,
ne = 1.9e13: a grid effect larger than the interpolation error that script was
measuring. That number came from an audit scratch script and cannot enter the
thesis; this script stamps it.

MECHANISM (stated before the run)
---------------------------------
The raw cross section starts at E_raw.min >= dE (by 0.00 to 0.06 eV above the
threshold from the quantum numbers); np.interp returns 0 below E_raw.min, so the
integrand has a step there, and a uniform grid of spacing (E_max - dE)/N =
0.19 eV at N = 5000 places that step to within one spacing. The error is a
near-threshold, low-Te effect, largest for weak transitions whose data begin
well above threshold, and it is the same at grid nodes and between them.

METHOD
------
build_L_at_Te.OperatorBuilder is loaded (its gate reproduces L_grid at N = 5000
to 0.0); its pre-interpolated CCC grids are rebuilt from the raw cross-section
CSV at N = 5000 (control), 25000 and 100000 with exactly the same linspace /
np.interp / trapezoid rule, and L, S are assembled at every one of the 400
nodes for each N. Reported, relative to N = 5000:
  entrywise |L_N/L_5000 - 1| over the nonzero entries, by class (all, 1s column,
  diagonal), median and max, at [15,3], [23,5], [0,0] and grid-wide;
  tau_slow, u_CRE, S = f_3 - f_4 at u_CRE (verify_nmax_downward_scan.quantities
  and fdiff), and eps_plateau of the one-interval heating step i -> i+1
  (verify_physical_ramp_bound.derived), at the named points and over the 448
  defended pairs (Te >= 2 eV window-passing set is not re-derived; all heating
  spans with Te_i >= 2 eV are used, 392 - 15*8 = 272 of them, and the full 392);
  the per-transition rate change |K_N/K_5000 - 1| at the 50 Te nodes: how many
  of the 1320 transitions move by > 2 % and > 10 %, and how large those rates
  are relative to the largest rate out of the same initial state;
  convergence: the 25000 -> 100000 change against the 5000 -> 25000 change
  (a ratio near 1/4 would say first order in the spacing, near 1/16 second).

GATES
-----
G0  the builder's gate (every stored table, L_grid, S_grid reproduced at all
    nodes to 1e-12).
G1  the N = 5000 rebuild inside this script reproduces the builder's own CCC
    tables to 0.0 (so the rebuild path is the pipeline's path).

PREDICTIONS (from the audit numbers, written before the run)
------------------------------------------------------------
P1  at [15,3] (Te 2.02 eV, ne 1.93e13), N = 25000 vs 5000: tau_slow -0.5 to
    -1.2 %, u_CRE -0.5 to -1.1 %, 1s-column median 1.0 to 2.5 %.
P2  at the 50 Te nodes, 25000 vs 5000: between 250 and 500 of 1320 transitions
    move by > 2 % and between 15 and 60 by > 10 % at Te ~ 2 eV; every
    transition moving > 10 % is weaker than 1e-3 of the largest rate out of
    its initial state.
P3  the effect grows toward 1 eV: |Delta tau_slow| at [0,0] larger than at
    [15,3], and smaller at [49,7] than at [15,3].
P4  25000 -> 100000 changes tau_slow at [15,3] by less than one quarter of the
    5000 -> 25000 change (the rule is first order in the spacing).
REFUTER of chapter 2's "grid density is measured": any of tau_slow, u_CRE,
S = f_3 - f_4 or eps_plateau moving by more than 2 % between N = 5000 and
100000 at a heating span with Te_i >= 2 eV. A shift below 2 % everywhere leaves
the sentence standing with a number attached; a shift above it refutes it.

OUTPUTS (with --write): validation/maxwell_grid_convergence/
  maxwell_grid_convergence.txt, _points.csv (400 x N), _transitions.csv
"""
from __future__ import annotations
import argparse, hashlib, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
from build_L_at_Te import OperatorBuilder, CCC_N_GRID  # noqa: E402
import verify_nmax_downward_scan as ns  # noqa: E402
import verify_physical_ramp_bound as vprb  # noqa: E402
ROOT = find_repo_root(_HERE)
NS = (5000, 25000, 100000); NAMED = {"ramp point [15,3]": (15, 3), "benchmark [23,5]": (23, 5), "cold corner [0,0]": (0, 0), "hot dense [49,7]": (49, 7)}


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); t0 = time.time()
    ctx = CRContext.load(); ctx.validate(); L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; nT, nN, nS, _ = L.shape
    S = np.load(ROOT / "data/processed/cr_matrix/S_grid.npy"); g = ctx.ground_index; nv = ctx.n_values.astype(int)
    E = np.array([k for k in range(nS) if k != g]); N3 = np.where(nv == 3)[0]; N4 = np.where(nv == 4)[0]
    i3, i4 = list(N3), list(N4)
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 100); say("CONVERGENCE OF THE CCC MAXWELL-AVERAGE ENERGY GRID FOR THE THESIS QUANTITIES"); say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}"); say(ctx.describe()); say("=" * 100)
    if CCC_N_GRID != NS[0]: raise RuntimeError(f"build_L_at_Te.CCC_N_GRID = {CCC_N_GRID}, this script assumes the pipeline's 5000")
    ob = OperatorBuilder.load(ROOT); worst = ob.gate(verbose=False)
    say(f"\n  G0  build_L_at_Te gate: worst relative difference {max(worst.values()):.1e}")
    if max(worst.values()) > 1e-12: raise RuntimeError("Gate G0")

    # ---- raw cross sections, grouped exactly as compute_K_CCC.py groups them
    xs = pd.read_csv(ob.paths["ccc_xs"]); exc = xs[xs.n_f > xs.n_i]; grp = exc.groupby(["n_i", "l_i", "n_f", "l_f"], sort=True)
    raw = []
    for key in ob.ccc_keys:
        gg = grp.get_group(key).sort_values("E_eV"); raw.append((gg.E_eV.values, gg.sigma_a0sq.values))
    gap = np.array([E_raw.min() - dE for (E_raw, _), dE in zip(raw, ob.ccc_dE)])
    say(f"  raw data begin {gap.min():.4f} to {gap.max():.4f} eV above the quantum-number threshold; grid spacing at N = 5000 is {np.median([(E_raw.max() - dE) / 5000 for (E_raw, _), dE in zip(raw, ob.ccc_dE)]):.3f} eV (median over transitions)")

    import build_L_at_Te as bl
    def rates_at_N(N, chunk=60):
        """K_exc (1320, nT) with the Maxwell average on an N-point grid, built in chunks of transitions (N = 100000 would need 1 GB per array otherwise)."""
        K = np.empty((len(raw), nT))
        for c0 in range(0, len(raw), chunk):
            idx = range(c0, min(c0 + chunk, len(raw))); Eg = np.empty((len(idx), N)); Sg = np.empty_like(Eg)
            for r, k in enumerate(idx):
                E_raw, sig_raw = raw[k]; dE = ob.ccc_dE[k]
                Eg[r] = np.linspace(dE + 1e-4, E_raw.max(), N); Sg[r] = np.interp(Eg[r], E_raw, sig_raw, left=0.0, right=0.0)
            for t, Te in enumerate(te): K[idx, t] = bl.ccc_maxwell_average(Eg, Sg, float(Te))
        return K
    Kfull = {N: rates_at_N(N) for N in NS}
    if np.abs(Kfull[5000] - np.load(ob.paths["ccc_exc"])).max() != 0.0: raise RuntimeError("Gate G1: the N = 5000 chunked rebuild differs from K_CCC_exc_table.npy")
    say("  G1  N = 5000 rebuild (chunked, same linspace / interp / trapezoid) equals K_CCC_exc_table.npy exactly")

    # ---- L, S at every node for each N: the builder's Maxwell average is redirected to the precomputed table for that N
    te_index = {float(Te): t for t, Te in enumerate(te)}; original = bl.ccc_maxwell_average
    Lg = {}; Sgr = {}
    for N in NS:
        bl.ccc_maxwell_average = lambda Eg, Sg, Te, _K=Kfull[N]: _K[:, te_index[float(Te)]].copy()
        ob._cache.clear(); LN = np.empty_like(L); SN = np.empty_like(S)
        for t, Te in enumerate(te):
            for j, nej in enumerate(ne): LN[t, j], SN[t, j] = ob.operator(float(Te), float(nej))
        Lg[N] = LN; Sgr[N] = SN
    bl.ccc_maxwell_average = original; ob._cache.clear()
    d5 = np.abs(Lg[5000] - L).max() / np.abs(L).max()
    say(f"  N = 5000 assembled L vs L_grid at all nodes: {d5:.1e}")
    if d5 > 1e-12: raise RuntimeError("N = 5000 path does not reproduce L_grid")

    # ---- results
    eye = np.eye(nS, dtype=bool); off = ~eye
    def quantities(Lij, Sij): a3, a4, c3, c4, u, tau = ns.quantities(Lij, Sij, g, i3, i4); return u, tau, ns.fdiff(a3, a4, c3, c4, u)
    rows = []
    for N in NS:
        for i in range(nT):
            for j in range(nN):
                u, tau, Sf = quantities(Lg[N][i, j], Sgr[N][i, j])
                eps = vprb.derived(Lg[N][i, j], Sgr[N][i, j], Lg[N][i + 1, j], Sgr[N][i + 1, j], g, E, N3, N4)[3] if i < nT - 1 else np.nan
                nz = Lg[5000][i, j] != 0; rel = np.abs(Lg[N][i, j][nz] / Lg[5000][i, j][nz] - 1)
                col = nz & off & (np.arange(nS)[None, :] == g); relc = np.abs(Lg[N][i, j][col] / Lg[5000][i, j][col] - 1)
                rows.append(dict(N=N, i=i, j=j, Te_eV=te[i], ne_cm3=ne[j], tau_slow=tau, u_CRE=u, S=Sf, eps_plateau_step=eps,
                                 L_all_med=float(np.median(rel)), L_all_max=float(rel.max()), L_col1s_med=float(np.median(relc)), L_col1s_max=float(relc.max())))
    df = pd.DataFrame(rows); base = df[df.N == 5000].set_index(["i", "j"])
    for q in ("tau_slow", "u_CRE", "S", "eps_plateau_step"):
        df[f"d_{q}"] = df.apply(lambda r: r[q] / base.loc[(r.i, r.j), q] - 1.0, axis=1)
    say("\n" + "=" * 100); say("RESULT 1: relative change of L entries and of the thesis quantities against N = 5000, named nodes"); say("=" * 100)
    say(f"  {'point':>18} {'N':>7} | {'L all med':>9} {'L all max':>9} {'1s col med':>10} {'1s col max':>10} | {'d tau_slow':>10} {'d u_CRE':>9} {'d S':>9} {'d eps_pl':>9}")
    for name, (i, j) in NAMED.items():
        for N in NS[1:]:
            r = df[(df.N == N) & (df.i == i) & (df.j == j)].iloc[0]
            say(f"  {name:>18} {N:>7} | {100*r.L_all_med:9.3f} {100*r.L_all_max:9.2f} {100*r.L_col1s_med:10.3f} {100*r.L_col1s_max:10.2f} | {100*r.d_tau_slow:+10.3f} {100*r.d_u_CRE:+9.3f} {100*r.d_S:+9.3f} {100*r.d_eps_plateau_step:+9.3f}   (%)")
    say("\n" + "=" * 100); say("RESULT 2: over the grid (all 400 nodes; heating spans with Te_i >= 2 eV for eps_plateau), N = 25000 and 100000 against 5000"); say("=" * 100)
    for N in NS[1:]:
        d = df[df.N == N]; d2 = d[d.Te_eV >= 2.0]
        say(f"  N = {N}:")
        for q in ("tau_slow", "u_CRE", "S", "eps_plateau_step"):
            v = d[f"d_{q}"].dropna(); v2 = d2[f"d_{q}"].dropna(); w = d.loc[v.abs().idxmax()]
            say(f"    {q:17s} all nodes: median {100*v.median():+.3f} %, |max| {100*v.abs().max():.3f} % at [{int(w.i)},{int(w.j)}] (Te {w.Te_eV:.2f});  Te >= 2 eV: median {100*v2.median():+.3f} %, |max| {100*v2.abs().max():.3f} %")
        say(f"    1s column median over nodes: {100*d.L_col1s_med.median():.3f} % (max over nodes {100*d.L_col1s_med.max():.3f} %); all-entry median {100*d.L_all_med.median():.4f} %")
    say("\n" + "=" * 100); say("RESULT 3: per-transition rate change |K_N/K_5000 - 1| over the 1320 CCC excitation blocks at the 50 Te nodes"); say("=" * 100)
    ni_of = np.array([k[0] for k in ob.ccc_keys]); li_of = np.array([k[1] for k in ob.ccc_keys])
    init_keys = sorted(set((k[0], k[1]) for k in ob.ccc_keys)); init = np.array([init_keys.index((k[0], k[1])) for k in ob.ccc_keys])   # 45 initial (n, l), including (9, l)
    Kmax_init = np.array([Kfull[5000][init == s_].max(axis=0) for s_ in range(len(init_keys))])   # largest 5000-point rate out of each initial state, per Te
    weak = Kfull[5000] / Kmax_init[init]                                                           # (1320, nT) rate relative to the strongest from the same initial state
    trows = []
    for N in NS[1:]:
        rel = np.abs(Kfull[N] / Kfull[5000] - 1.0)                                            # (1320, nT)
        for t in (0, int(np.argmin(np.abs(te - 2.0236))), 23, nT - 1):
            n2 = int((rel[:, t] > 0.02).sum()); n10 = int((rel[:, t] > 0.10).sum()); wk = weak[rel[:, t] > 0.10, t]
            say(f"  N = {N:>6}, Te = {te[t]:5.2f} eV: > 2 %: {n2:4d} of 1320;  > 10 %: {n10:3d};  those > 10 % are at most {wk.max() if len(wk) else 0:.1e} of the strongest rate from their initial state;  median change {100*np.median(rel[:, t]):.3f} %, max {100*rel[:, t].max():.1f} %")
            for r in np.where(rel[:, t] > 0.10)[0]: trows.append(dict(N=N, Te_eV=te[t], n_i=ni_of[r], l_i=li_of[r], n_f=ob.ccc_keys[r][2], l_f=ob.ccc_keys[r][3], rel_change=rel[r, t], K_5000=Kfull[5000][r, t], weak=weak[r, t], gap_eV=gap[r]))
        tr = pd.DataFrame([x for x in trows if x["N"] == N])
        if len(tr): say(f"    transitions moving > 10 % (any of the four Te): gap E_raw.min - dE from {tr.gap_eV.min():.3f} to {tr.gap_eV.max():.3f} eV (all: {gap.min():.3f} to {gap.max():.3f});  the largest 5000-point rate among them {tr.K_5000.max():.2e} cm^3/s")
    say("\n" + "=" * 100); say("RESULT 4: convergence, 25000 -> 100000 against 5000 -> 25000"); say("=" * 100)
    for name, (i, j) in NAMED.items():
        r25 = df[(df.N == 25000) & (df.i == i) & (df.j == j)].iloc[0]; r100 = df[(df.N == 100000) & (df.i == i) & (df.j == j)].iloc[0]
        say(f"  {name:>18}: tau_slow {100*r25.d_tau_slow:+.3f} % -> {100*r100.d_tau_slow:+.3f} % (further step {100*(r100.d_tau_slow - r25.d_tau_slow):+.4f} %, ratio {(r100.d_tau_slow - r25.d_tau_slow)/r25.d_tau_slow if r25.d_tau_slow else float('nan'):.3f});"
            f"  u_CRE {100*r25.d_u_CRE:+.3f} % -> {100*r100.d_u_CRE:+.3f} %;  1s col median {100*r25.L_col1s_med:.3f} % -> {100*r100.L_col1s_med:.3f} %")

    # ---- predictions and refuter
    say("\n" + "=" * 100); say("PREDICTIONS (written before the run) against what came out"); say("=" * 100)
    missed = []
    r = df[(df.N == 25000) & (df.i == 15) & (df.j == 3)].iloc[0]
    p1 = (-0.012 <= r.d_tau_slow <= -0.005) and (-0.011 <= r.d_u_CRE <= -0.005) and (0.010 <= r.L_col1s_med <= 0.025)
    say(f"  P1 [15,3] 25000 vs 5000: tau_slow {100*r.d_tau_slow:+.3f} % (-0.5 to -1.2), u_CRE {100*r.d_u_CRE:+.3f} % (-0.5 to -1.1), 1s col median {100*r.L_col1s_med:.3f} % (1.0 to 2.5): {'reproduced' if p1 else 'NOT reproduced'}")
    if not p1: missed.append("P1")
    t2 = int(np.argmin(np.abs(te - 2.0236))); rel25 = np.abs(Kfull[25000] / Kfull[5000] - 1.0); n2 = int((rel25[:, t2] > 0.02).sum()); n10 = int((rel25[:, t2] > 0.10).sum())
    wk = weak[rel25[:, t2] > 0.10, t2]
    p2 = 250 <= n2 <= 500 and 15 <= n10 <= 60 and (len(wk) == 0 or wk.max() < 1e-3)
    say(f"  P2 at Te = {te[t2]:.2f} eV: {n2} transitions > 2 % (250 to 500), {n10} > 10 % (15 to 60), the > 10 % ones at most {wk.max() if len(wk) else 0:.1e} of the strongest rate from their initial state (< 1e-3): {'reproduced' if p2 else 'NOT reproduced'}")
    if not p2: missed.append("P2")
    d25 = df[df.N == 25000].set_index(["i", "j"])
    p3 = abs(d25.loc[(0, 0), "d_tau_slow"]) > abs(d25.loc[(15, 3), "d_tau_slow"]) > abs(d25.loc[(49, 7), "d_tau_slow"])
    say(f"  P3 |d tau_slow| grows toward 1 eV: [0,0] {100*abs(d25.loc[(0, 0), 'd_tau_slow']):.3f} % > [15,3] {100*abs(d25.loc[(15, 3), 'd_tau_slow']):.3f} % > [49,7] {100*abs(d25.loc[(49, 7), 'd_tau_slow']):.3f} %: {'reproduced' if p3 else 'NOT reproduced'}")
    if not p3: missed.append("P3")
    r100 = df[(df.N == 100000) & (df.i == 15) & (df.j == 3)].iloc[0]
    p4 = abs(r100.d_tau_slow - r.d_tau_slow) < 0.25 * abs(r.d_tau_slow)
    say(f"  P4 [15,3] further step 25000 -> 100000 in tau_slow {100*(r100.d_tau_slow - r.d_tau_slow):+.4f} % against {100*r.d_tau_slow:+.3f} % (ratio {abs(r100.d_tau_slow - r.d_tau_slow)/abs(r.d_tau_slow):.3f} < 0.25): {'reproduced' if p4 else 'NOT reproduced'}")
    if not p4: missed.append("P4")
    say("\nPREDICTIONS: " + ("P1-P4 all reproduced" if not missed else "not reproduced as written: " + ", ".join(missed)))
    d100 = df[(df.N == 100000) & (df.Te_eV >= 2.0)]
    worst_q = {q: 100 * d100[f"d_{q}"].abs().max() for q in ("tau_slow", "u_CRE", "S", "eps_plateau_step")}
    ref = max(worst_q.values()) > 2.0
    say(f"REFUTER (a thesis quantity moving > 2 % between N = 5000 and 100000 at Te >= 2 eV): {'APPEARED' if ref else 'did not appear'}  [" + ", ".join(f"{k} {v:.3f} %" for k, v in worst_q.items()) + "]")
    say("\nREADING. Chapter 2 tested the coarse direction (500 points, ~17 % at 1 eV) and called the 5000-point grid measured. The fine direction\n"
        "  is tested here; the numbers above say how much of each thesis quantity is grid, at the nodes and therefore everywhere. The\n"
        "  mechanism is the mis-placed threshold step (raw data begin above the quantum-number threshold, np.interp returns zero below\n"
        "  them, the trapezoid resolves the step to one spacing), so it is the weak, high-threshold-gap transitions and the low-Te end\n"
        "  that move. Nothing here is repaired: N_GRID stays 5000 in compute_K_CCC.py and every stored table is unchanged (CLAUDE.md).")
    say(f"\n  wall time {time.time() - t0:.0f} s")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation/maxwell_grid_convergence"; out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}"] + ob.provenance()
        hdr += [f"# {p.relative_to(ROOT)} sha256 {sha(p)}" for p in (_HERE.parent / "build_L_at_Te.py", _HERE.parent / "verify_nmax_downward_scan.py", _HERE.parent / "verify_physical_ramp_bound.py")]
        hdr += ["# d_<q> = q(N)/q(5000) - 1 at the same node; eps_plateau_step is the one-interval heating step i -> i+1 at fixed j (verify_physical_ramp_bound.derived)",
                "# L_*_med/max: relative change of nonzero L entries (all; 1s column off the diagonal)"]
        with open(out / "maxwell_grid_convergence_points.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "maxwell_grid_convergence_transitions.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); pd.DataFrame(trows).to_csv(fh, index=False)
        with open(out / "maxwell_grid_convergence.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/maxwell_grid_convergence.txt, _points.csv, _transitions.csv")
    return 0


if __name__ == "__main__": sys.exit(main())
