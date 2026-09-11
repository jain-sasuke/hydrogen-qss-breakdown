#!/usr/bin/env python
"""
verify_weighted_census.py
=========================
The plateau error and the ELM census, recomputed on the A-weighted
Halpha/Hbeta emissivity ratio, side by side with the shell ratio n_3/n_4 that
every artifact in validation/ currently uses.

WHY THIS EXISTS
---------------
Section sec:emissivity_generalisation proves the two-channel machinery for any
non-negative linear functional of the fast-state vector, and
verify_emissivity_generalisation.py checks the logistic bound on the line
ratio. But every NUMBER downstream (eps_plateau, the census of 45 of 448, the
worst case 0.1748, Sbar) is still computed on unweighted shell sums:
verify_divertor_map.py, verify_reservoir_gain.py, make_ch5_figures.py.
Chapter 4 quotes a line-to-shell error ratio of 0.978 to 0.9999 from a
markdown note (thesis_ready.md A6) that has no committed script behind it.

This script re-executes the divertor-map construction verbatim, computes the
shell and the A-weighted observables from the SAME solves, guards that the
shell half reproduces validation/divertor_map/divertor_map.csv row by row,
and then reports what changes when the observable is the line ratio.

It changes no thesis number. It reports.

PREDICTIONS, WRITTEN BEFORE THE FIRST RUN
-----------------------------------------
P1  The reservoir gain G = Dln u / Dln Te does not involve the observable, so
    it is identical under both observables by construction. The factorisation
    identity eps = |expm1(Sbar G Dln Te)| is likewise an algebraic identity
    (lnx and dlnTe cancel symbolically), so its residual is a log/expm1
    round trip and has ZERO severity as a test of the observable.
P2  A constant factor per line (photon-count against energy weighting)
    cancels from every ratio exactly. Its residual can only catch a
    per-channel misapplication of h nu. Also near-zero severity.
    What actually protects the weights: load_radiative_weights' (n,l)->(n,l)
    row-uniqueness selector and idx cross-check, the label assertion below,
    and an independent label-based rebuild (skeptic pass, 11 Sep 2026)
    that reproduced ratio_plateau to 1.2e-13 over all 784 rows.
P3  eps_line / eps_shell lies in 0.978 to 0.9999 over the 680 window rows,
    with the worst case at the lowest density (chapter 4, from A6). The
    refuting observation is a ratio outside that range by more than the
    fourth decimal, or a worst case that is not at ne = 1e12.
    OUTCOME (11 Sep 2026): REFUTED. The range is 0.9482 to 0.9999, worst
    5.2% at heat [48,0] (the step 9.54 -> 10.0 eV; the operator, n_new and
    the (A,C) coefficients are evaluated at the POST-step point [49,0],
    Te = 10.0 eV), ne = 1e12. 0.978 is reproduced only by restricting to
    Te < 2 eV (min at heat [14,0]), or by substituting the Hbeta denominator
    alone (n3/Hbeta against n3/n4: 0.9810 to 1.0000), which is what
    thesis_ready.md A6 literally describes. The extremum sits on the grid
    boundary with both trends still monotone, so it is bounded by the grid,
    not by physics. The departure scales as 1/(l-mixing rate): 10.0% at
    half the PSM20 rate, 2.6% at double. The sign (ratio < 1, all 784 rows)
    survives that factor of ten.
P4  The 100 us census (window_ok, Te >= 2 eV, lower bound > 0.10) stays
    within a few pairs of 45, and the worst case 0.1748 moves by less than
    the 2.2 percent that P3 allows. The refuting observation is a count that
    moves by more than five, or a change in which grid point is worst.

OBSERVABLE
----------
Halpha = A(3s-2p) n_3s + A(3p-2s) n_3p + A(3d-2p) n_3d
Hbeta  = A(4s-2p) n_4s + A(4p-2s) n_4p + A(4d-2p) n_4d      (4f carries 0)
A-values from data/processed/Radiative/radiative_rates.csv through
Balmer_transient_ratio.load_radiative_weights, which raises unless exactly one
resolved row matches each transition and cross-checks the state indices.

Read-only. Writes only under validation/weighted_census/ with --write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)
sys.path.insert(0, str(ROOT / "src" / "rates"))

# The census definition, copied from verify_divertor_map.py so that the shell
# half of this script is a re-execution and not a re-derivation. If these
# drift from that script the guard below fails on the CSV comparison.
FRAC = 0.05
WIN_LO, WIN_HI = 30.0, 30.0
THRESHOLD = 0.10
TE_FLOOR = 2.0
NE_FLOOR = 1e14
DRIVES_US = (100.0, 250.0, 506.0, 750.0)   # tab:duration_sensitivity
BENCH_TE_EV, BENCH_NE_CM3, BENCH_IJ = 2.947, 1.389e14, (23, 5)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv_with_comments(path: Path) -> list[dict]:
    with path.open() as fh:
        body = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(body))


def elm_lower_bound(ep: np.ndarray, tQ: np.ndarray, td: float) -> np.ndarray:
    return ep * (tQ / td) * (1.0 - np.exp(-td / tQ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    import Balmer_transient_ratio as btr                  # noqa: E402

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = int(ctx.ground_index)
    nv = np.asarray(ctx.n_values)
    labels = ctx.labels
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)

    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    if not sp.exists():
        raise FileNotFoundError(f"missing source vector {sp}")
    S = np.load(sp)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")

    dmap_p = ROOT / "validation/divertor_map/divertor_map.csv"
    if not dmap_p.exists():
        raise FileNotFoundError(f"missing {dmap_p}; the shell half of this "
                                f"script must be checked against it")
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
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    ib, jb = ctx.nearest_point(BENCH_TE_EV, BENCH_NE_CM3)
    if (ib, jb) != BENCH_IJ:
        raise RuntimeError(f"benchmark moved to [{ib},{jb}]; grids are not "
                           f"the ones CLAUDE.md's values were measured on")

    # ---- weight vectors, both conventions, over the full state vector ------
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
        # every weighted state must lie in the shell the line names
        for line, m in (("Halpha", 3), ("Hbeta", 4)):
            bad = [labels[i] for i in np.where(w[line] > 0)[0] if nv[i] != m]
            if bad:
                raise RuntimeError(f"{line} weights touch states outside "
                                   f"n={m}: {bad}")
        return w

    wp = weight_vectors(False)
    we = weight_vectors(True)

    out_lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    say("=" * 78)
    say("WEIGHTED CENSUS: A-weighted Halpha/Hbeta against the n=3/n=4 shell ratio")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}")
    say(f"interpreter {sys.executable}   numpy {np.__version__}")
    say(f"L_grid sha256      {sha256(lp)}")
    say(f"S_grid sha256      {sha256(sp)}")
    say(f"state_index sha256 {sha256(ctx.state_index_path)}")
    say(f"divertor_map.csv   {dmap_header[0]}")
    say(f"A-values from      {btr.DATA_RAD.relative_to(ROOT)}")
    for line in ("Halpha", "Hbeta"):
        nz = np.where(wp[line] > 0)[0]
        say(f"  {line:7s} " + ", ".join(f"{labels[i]}={wp[line][i]:.4e}" for i in nz)
            + f"   (zero on {[labels[i] for i in np.where((nv == (3 if line == 'Halpha' else 4)) & (wp[line] == 0))[0]]})")
    say(f"construction: frac {FRAC}, window {WIN_LO}*tau_relax < tau_slow/{WIN_HI}, "
        f"threshold {THRESHOLD}, scope Te >= {TE_FLOOR} eV")
    say("=" * 78)

    # ---- the divertor-map loop, re-executed, both observables --------------
    rows = []
    worst_energy_vs_photon = 0.0
    worst_energy_vs_photon_plateau = 0.0
    worst_identity = 0.0
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(len(te)):
            k = int(np.argmin(np.abs(te - te[i] * (1 + sgn * FRAC))))
            if k == i:
                continue
            dlnTe = float(np.log(te[k] / te[i]))
            for j in range(len(ne)):
                lam = np.linalg.eigvals(L[k, j])
                lam = lam[np.argsort(lam.real)[::-1]]
                if lam[0].real >= 0 or lam[1].real >= 0:
                    raise RuntimeError(f"unstable operator at [{k},{j}]")
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                win = (WIN_LO * tR) < (tQ / WIN_HI)

                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(L[k, j], -S[k, j])
                LEE = L[k, j][np.ix_(E, E)]
                LEg = L[k, j][np.ix_(E, [g])].ravel()
                n0 = np.linalg.solve(LEE, -S[k, j][E])          # recombination-fed
                n1 = np.linalg.solve(LEE, -LEg * n_old[g])      # ground-fed, pre-step n_g
                x_new = n_new[g] / n_old[g]
                sup = (np.abs(n0 + x_new * n1 - n_new[E]).max()
                       / np.abs(n_new[E]).max())
                if sup > 1e-8:
                    raise RuntimeError(f"superposition fails at [{i},{j}] {dlab}: {sup:.3e}")
                lnx = float(np.log(x_new))
                G = lnx / dlnTe

                # ---- shell observable, exactly as verify_divertor_map.py ----
                a3_0, a4_0 = n0[n3E].sum(), n0[n4E].sum()
                a3_1, a4_1 = n1[n3E].sum(), n1[n4E].sum()
                f3_s = a3_1 / (a3_0 + a3_1)
                f4_s = a4_1 / (a4_0 + a4_1)
                Rq_s = n_new[N3].sum() / n_new[N4].sum()
                Rpe_s = (a3_0 + a3_1) / (a4_0 + a4_1)
                ep_s = abs(Rpe_s / Rq_s - 1.0)
                es_s = abs(n_old[N3].sum() / n_old[N4].sum() / Rq_s - 1.0)
                Sbar_s = np.log(Rpe_s / Rq_s) / lnx

                # ---- line observable: same solves, A-weighted contractions --
                def line_quantities(w):
                    wa, wb = w["Halpha"][E], w["Hbeta"][E]
                    Ca, Cb = wa @ n0, wb @ n0            # recombination-fed emissivity
                    Aa, Ab = wa @ n1, wb @ n1            # ground-fed emissivity
                    if min(Ca, Cb, Aa, Ab) <= 0:
                        raise RuntimeError(f"non-positive emissivity at [{i},{j}] {dlab}")
                    fa = Aa / (Aa + Ca)
                    fb = Ab / (Ab + Cb)
                    Rq = (w["Halpha"] @ n_new) / (w["Hbeta"] @ n_new)
                    Rpe = (Aa + Ca) / (Ab + Cb)
                    Rold = (w["Halpha"] @ n_old) / (w["Hbeta"] @ n_old)
                    ep = abs(Rpe / Rq - 1.0)
                    es = abs(Rold / Rq - 1.0)
                    Sbar = np.log(Rpe / Rq) / lnx
                    return fa, fb, ep, es, Sbar

                fa, fb, ep_l, es_l, Sbar_l = line_quantities(wp)
                _, _, ep_e, es_e, _ = line_quantities(we)
                # one side at a time: Halpha over the n=4 shell, and n=3 shell over Hbeta
                w_num = {"Halpha": wp["Halpha"], "Hbeta": np.where(nv == 4, 1.0, 0.0)}
                w_den = {"Halpha": np.where(nv == 3, 1.0, 0.0), "Hbeta": wp["Hbeta"]}
                _, _, ep_num, _, _ = line_quantities(w_num)
                _, _, ep_den, _, _ = line_quantities(w_den)
                worst_energy_vs_photon_plateau = max(worst_energy_vs_photon_plateau,
                                                     abs(ep_e - ep_l) / max(ep_l, 1e-300))
                worst_energy_vs_photon = max(worst_energy_vs_photon,
                                             abs(es_e - es_l) / max(es_l, 1e-300))
                # factorisation identity on the line observable (P1)
                pred = abs(np.expm1(Sbar_l * G * dlnTe))
                if ep_l > 0:
                    worst_identity = max(worst_identity, abs(pred - ep_l) / ep_l)

                r = dict(direction=dlab, i=i, j=j, Te=float(te[i]), ne=float(ne[j]),
                         tau_slow=tQ, tau_relax=tR, M=tQ / tR, window_ok=bool(win),
                         dlnTe=dlnTe, lnx=lnx, G=G,
                         eps_step_shell=es_s, eps_plateau_shell=ep_s,
                         f3_shell=f3_s, f4_shell=f4_s, Sbar_shell=Sbar_s,
                         eps_step_line=es_l, eps_plateau_line=ep_l,
                         f3_line=fa, f4_line=fb, Sbar_line=Sbar_l,
                         ratio_plateau=ep_l / ep_s if ep_s > 0 else np.nan,
                         ratio_numerator_only=ep_num / ep_s if ep_s > 0 else np.nan,
                         ratio_denominator_only=ep_den / ep_s if ep_s > 0 else np.nan,
                         ratio_step=es_l / es_s if es_s > 0 else np.nan)
                for td in DRIVES_US:
                    r[f"lo_shell_{td:g}us"] = float(elm_lower_bound(ep_s, tQ, td * 1e-6))
                    r[f"lo_line_{td:g}us"] = float(elm_lower_bound(ep_l, tQ, td * 1e-6))
                rows.append(r)

    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    say(f"\nevaluated {len(rows)} (point, direction) pairs")

    # ---- guard: shell half reproduces divertor_map.csv row by row ---------
    if len(dmap) != len(rows):
        raise RuntimeError(f"divertor_map.csv has {len(dmap)} rows, recomputed {len(rows)}")
    worst = {"eps_plateau": 0.0, "tau_QSS": 0.0, "f3": 0.0, "lo_ELM_crash": 0.0}
    for r, d in zip(rows, dmap):
        if (r["direction"], r["i"], r["j"]) != (d["direction"], int(d["i"]), int(d["j"])):
            raise RuntimeError(f"row order differs from divertor_map.csv at {r['direction']} [{r['i']},{r['j']}]")
        if r["window_ok"] != (d["window_ok"] == "True"):
            raise RuntimeError(f"window_ok differs at {r['direction']} [{r['i']},{r['j']}]")
        for mine, theirs in (("eps_plateau_shell", "eps_plateau"), ("tau_slow", "tau_QSS"),
                             ("f3_shell", "f3"), ("lo_shell_100us", "lo_ELM_crash")):
            v, u = r[mine], float(d[theirs])
            worst[theirs] = max(worst[theirs], abs(v - u) / max(abs(u), 1e-300))
    for kk, v in worst.items():
        if v > 1e-9:
            raise RuntimeError(f"shell recomputation does not reproduce divertor_map.csv: "
                               f"{kk} differs by {v:.3e}")
    say("guard: shell half reproduces divertor_map.csv, worst relative "
        + ", ".join(f"{kk} {v:.1e}" for kk, v in worst.items()))

    say("\n" + "=" * 78)
    say("ALGEBRAIC IDENTITIES (round-trip only; severity zero as tests of the observable)")
    say("=" * 78)
    say(f"P2 energy against photon weighting, worst relative change in "
        f"eps_plateau: {worst_energy_vs_photon_plateau:.3e}   "
        f"(eps_step: {worst_energy_vs_photon:.3e}, inflated near its zero crossing)")
    say(f"P1 eps = |expm1(Sbar G dlnTe)| on the LINE observable, worst relative "
        f"departure: {worst_identity:.3e}  (log/expm1 round trip; cannot fail)")
    say("   G is the same number in both columns by construction; the observable "
        "enters only through Sbar. A within-shell A-value swap (3S<->3D) flips the")
    say("   sign of the corner departure and passes both of these; only the weight")
    say("   loader's row-uniqueness and the label assertion protect against it.")

    ok = A["window_ok"]
    warm = ok & (A["Te"] >= TE_FLOOR)
    dense = warm & (A["ne"] >= NE_FLOOR)
    say("\n" + "=" * 78)
    say("P3  eps_plateau(line) / eps_plateau(shell)")
    say("=" * 78)
    for name, m in (("all 784", np.ones_like(ok)), ("window_ok 680", ok),
                    ("Te>=2 448", warm), ("dense 108", dense)):
        q = A["ratio_plateau"][m]
        kmin = np.where(m)[0][np.argmin(q)]
        kmax = np.where(m)[0][np.argmax(q)]
        say(f"  {name:14s} n={m.sum():3d}  ratio {q.min():.6f} to {q.max():.6f}  "
            f"median {np.median(q):.6f}")
        say(f"  {'':14s} min at {A['direction'][kmin]} [{A['i'][kmin]},{A['j'][kmin]}] "
            f"pre-step Te={A['Te'][kmin]:.3g} ne={A['ne'][kmin]:.3g}; "
            f"max at {A['direction'][kmax]} [{A['i'][kmax]},{A['j'][kmax]}] "
            f"pre-step Te={A['Te'][kmax]:.3g} ne={A['ne'][kmax]:.3g}")
    say("  (Te printed is the PRE-step value, as in divertor_map.csv; the operator "
        "and (A,C) that set eps are those of the post-step point.)")
    say("  ratio_plateau < 1 at every row: the shell census is conservative.")
    q = A["ratio_step"][ok]
    say(f"  eps_step ratio, window_ok: {q.min():.6f} to {q.max():.6f}   "
        f"NOT an observable discrepancy: eps_step passes through zero on this")
    say(f"  grid (sec:eps_step_locus), so this is a ratio of two small numbers at "
        f"the zero crossing; {int((A['ratio_step'][ok] > 1).sum())} of {ok.sum()} rows exceed 1.")
    say(f"  chapter 4 quotes 0.978 to 0.9999 over 'the grid' (A6, no script). "
        f"Compare the window_ok line.")
    kc = int(np.where(ok)[0][np.argmin(A["ratio_plateau"][ok])])
    say(f"  one side at a time at the corner {A['direction'][kc]} [{A['i'][kc]},{A['j'][kc]}]: "
        f"Halpha numerator only {A['ratio_numerator_only'][kc]:.4f}, Hbeta denominator only "
        f"{A['ratio_denominator_only'][kc]:.4f}, both {A['ratio_plateau'][kc]:.4f}")
    say(f"  denominator-only ratio over window_ok rows: {A['ratio_denominator_only'][ok].min():.4f} to "
        f"{A['ratio_denominator_only'][ok].max():.4f}  (thesis_ready A6's 'Balmer denominator' reading)")

    # ---- sensitivity to the proton l-mixing rate ------------------------------
    # Scale every within-shell l-changing rate (same n, different state) by s,
    # adjusting the diagonal so column sums are preserved (conservation), and
    # recompute the line/shell eps ratio at the corner and the benchmark.
    same_shell = (nv[:, None] == nv[None, :]) & ~np.eye(ctx.n_states, dtype=bool)
    def eps_pair_scaled(sfac, i, k, j):
        Lp = L[k, j].copy(); Lo = L[i, j].copy()
        for Lm in (Lp, Lo):
            off = np.where(same_shell, Lm, 0.0)
            Lm += (sfac - 1.0) * off
            Lm[np.diag_indices_from(Lm)] -= (sfac - 1.0) * off.sum(axis=0)
        n_old = np.linalg.solve(Lo, -S[i, j]); n_new = np.linalg.solve(Lp, -S[k, j])
        LEE = Lp[np.ix_(E, E)]; LEg = Lp[np.ix_(E, [g])].ravel()
        n0 = np.linalg.solve(LEE, -S[k, j][E]); n1 = np.linalg.solve(LEE, -LEg * n_old[g])
        out = {}
        for nm, w in (("shell", {"Halpha": np.where(nv == 3, 1.0, 0.0), "Hbeta": np.where(nv == 4, 1.0, 0.0)}),
                      ("line", wp)):
            wa, wb = w["Halpha"][E], w["Hbeta"][E]
            Rq = (w["Halpha"] @ n_new) / (w["Hbeta"] @ n_new)
            Rpe = (wa @ (n0 + n1)) / (wb @ (n0 + n1))
            out[nm] = abs(Rpe / Rq - 1.0)
        return out["line"] / out["shell"]
    say("\n" + "=" * 78)
    say("SENSITIVITY TO THE PROTON l-MIXING RATE (within-shell rates scaled by s, conservation kept)")
    say("=" * 78)
    lmix_rows = []
    for sfac in (0.5, 1.0, 2.0, 5.0):
        vals = {}
        for name, (i_, j_) in (("corner", (int(A["i"][kc]), int(A["j"][kc]))), ("benchmark", (ib, jb))):
            k_ = int(np.argmin(np.abs(te - te[i_] * (1 + FRAC))))
            vals[name] = eps_pair_scaled(sfac, i_, k_, j_)
        say(f"  s = {sfac:<4g} line/shell eps ratio: corner {vals['corner']:.4f} "
            f"({(1-vals['corner'])*100:.1f}% departure), benchmark {vals['benchmark']:.4f}")
        lmix_rows.append(dict(lmix_scale=sfac, ratio_corner=vals["corner"], ratio_benchmark=vals["benchmark"]))
    say("  at s = 1 the corner value must equal the P3 minimum above (same arithmetic, rebuilt L).")

    say("\n" + "=" * 78)
    say("BENCHMARK [23,5] heating")
    say("=" * 78)
    kb = [n for n, r in enumerate(rows) if (r["direction"], r["i"], r["j"]) == ("heat", ib, jb)][0]
    rb = rows[kb]
    say(f"  eps_plateau  shell {rb['eps_plateau_shell']:.6f}   line {rb['eps_plateau_line']:.6f}   "
        f"ratio {rb['ratio_plateau']:.6f}   (thesis: 0.063612)")
    say(f"  eps_step     shell {rb['eps_step_shell']:.6f}   line {rb['eps_step_line']:.6f}")
    say(f"  f3 - f4      shell {rb['f3_shell']-rb['f4_shell']:+.6f}   line {rb['f3_line']-rb['f4_line']:+.6f}")
    say(f"  Sbar         shell {rb['Sbar_shell']:+.6f}   line {rb['Sbar_line']:+.6f}   G {rb['G']:+.6f}")

    say("\n" + "=" * 78)
    say("P4  THE CENSUS: lower bound > threshold, window_ok and Te >= 2 eV")
    say("=" * 78)
    say(f"  {'tau_d':>8}  {'shell count':>11} {'shell worst':>11} {'at':>16}  "
        f"{'line count':>10} {'line worst':>10} {'at':>16}  {'enter':>5} {'leave':>5}")
    census_summary = []
    for td in DRIVES_US:
        ls_ = A[f"lo_shell_{td:g}us"]
        ll_ = A[f"lo_line_{td:g}us"]
        cs = warm & (ls_ > THRESHOLD)
        cl = warm & (ll_ > THRESHOLD)
        ws = np.where(warm)[0][np.argmax(ls_[warm])]
        wl = np.where(warm)[0][np.argmax(ll_[warm])]
        enter = np.where(cl & ~cs)[0]
        leave = np.where(cs & ~cl)[0]
        say(f"  {td:>6g}us  {cs.sum():>4d}/{warm.sum():<6d} {ls_[warm].max():>11.4f} "
            f"{A['direction'][ws]+' ['+str(A['i'][ws])+','+str(A['j'][ws])+']':>16}  "
            f"{cl.sum():>4d}/{warm.sum():<5d} {ll_[warm].max():>10.4f} "
            f"{A['direction'][wl]+' ['+str(A['i'][wl])+','+str(A['j'][wl])+']':>16}  "
            f"{len(enter):>5d} {len(leave):>5d}")
        for n in enter:
            say(f"           enters: {A['direction'][n]} [{A['i'][n]},{A['j'][n]}] "
                f"shell {ls_[n]:.4f} line {ll_[n]:.4f}")
        for n in leave:
            say(f"           leaves: {A['direction'][n]} [{A['i'][n]},{A['j'][n]}] "
                f"shell {ls_[n]:.4f} line {ll_[n]:.4f}")
        census_summary.append(dict(tau_d_us=td, scope="Te>=2", n_scope=int(warm.sum()),
                                   count_shell=int(cs.sum()), worst_shell=float(ls_[warm].max()),
                                   count_line=int(cl.sum()), worst_line=float(ll_[warm].max()),
                                   n_enter=len(enter), n_leave=len(leave)))
    say("  thesis (tab:duration_sensitivity): 45/0.1748, 33/0.1664, 24/0.1533, 20/0.1435")
    say("\n  same at 100 us for the other two scopes:")
    for name, m in (("window_ok 680", ok), ("dense 108", dense)):
        ls_, ll_ = A["lo_shell_100us"], A["lo_line_100us"]
        cs, cl = m & (ls_ > THRESHOLD), m & (ll_ > THRESHOLD)
        say(f"  {name:14s} shell {cs.sum()}/{m.sum()} worst {ls_[m].max():.4f}   "
            f"line {cl.sum()}/{m.sum()} worst {ll_[m].max():.4f}   "
            f"enter {int((cl & ~cs).sum())} leave {int((cs & ~cl).sum())}")
        census_summary.append(dict(tau_d_us=100.0, scope=name.split()[0], n_scope=int(m.sum()),
                                   count_shell=int(cs.sum()), worst_shell=float(ls_[m].max()),
                                   count_line=int(cl.sum()), worst_line=float(ll_[m].max()),
                                   n_enter=int((cl & ~cs).sum()), n_leave=int((cs & ~cl).sum())))
    say("  thesis (tab:trapping_census): 202/680 worst 0.3868; 0/108 worst 0.0717")

    say("\n" + "=" * 78)
    say("Sbar RANGES (k = 1 only; reservoir_gain.csv also carries k = 2, 4)")
    say("=" * 78)
    for name, m in (("all 784", np.ones_like(ok)), ("window_ok 680", ok), ("Te>=2 448", warm)):
        ss, sl = A["Sbar_shell"][m], A["Sbar_line"][m]
        say(f"  {name:14s} |Sbar| shell {np.abs(ss).min():.4f} to {np.abs(ss).max():.4f}   "
            f"line {np.abs(sl).min():.4f} to {np.abs(sl).max():.4f}   "
            f"sign: shell all negative {bool((ss < 0).all())}, line all negative {bool((sl < 0).all())}")
    say(f"  |G| over 784: {np.abs(A['G']).min():.4f} to {np.abs(A['G']).max():.4f}, "
        f"all negative {bool((A['G'] < 0).all())}   (thesis, 2288 rows: 2.64 to 14.52)")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "weighted_census"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
               f"# L_grid sha256 {sha256(lp)}", f"# S_grid sha256 {sha256(sp)}",
               f"# state_index sha256 {sha256(ctx.state_index_path)}",
               f"# radiative_rates sha256 {sha256(btr.DATA_RAD)}",
               f"# fractional step {FRAC}, window {WIN_LO}/{WIN_HI}, threshold {THRESHOLD}, "
               f"Te floor {TE_FLOOR}, ne floor {NE_FLOOR:g}"]
        with (out / "weighted_census.csv").open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with (out / "weighted_census_summary.csv").open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(census_summary[0].keys()))
            w.writeheader(); w.writerows(census_summary)
        with (out / "weighted_census_lmix.csv").open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(lmix_rows[0].keys()))
            w.writeheader(); w.writerows(lmix_rows)
        (out / "weighted_census.txt").write_text("\n".join(out_lines) + "\n")
        say(f"\nwrote {out.relative_to(ROOT)}/weighted_census.csv ({len(rows)} rows), "
            f"weighted_census_summary.csv, weighted_census.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
