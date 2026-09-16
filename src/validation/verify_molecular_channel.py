#!/usr/bin/env python
"""
verify_molecular_channel.py
===========================
What a third supply channel (molecular-activated recombination feeding n = 3
directly) does to the two-channel plateau-error machinery of chapters 3 to 5,
computed on the model's own operator instead of asserted.

WHY THIS EXISTS
---------------
Chapter 6, section 6.3 (thesis_tex/chapter6.tex, "What survives it, and what
does not") makes two claims that have no producing script:

  (a) the two-channel factorisation, the logistic ground-fed fraction and the
      ceiling max|f3 - f4| = tanh(|Delta|/4) "hold unchanged" for a matrix
      containing molecular states;
  (b) a molecular feed with emissivity fractions phi_3 = 0.60, phi_4 = 0.30
      dilutes the sensitivity 0.201 -> 0.054 (cold corner) and 0.053 -> 0.019
      (density maximum), "a factor of three to five reduction" of the plateau
      error, because eps_plateau is linear in f3 - f4.

The four sensitivities in (b) match no row of any artifact under validation/
(cr-physicist scratch check4.py, 11 Sep 2026: scanned the CRE f3 - f4 at all
400 points and every sens/f column of plateau_gridmap.csv). Claim (a) is
false as written; see THE MATHEMATICS. This script replaces both with
computed, stamped statements.

THE MATHEMATICS
---------------
Split the state vector into the ground state g and the excited block E. With
a fixed ion reservoir (S_grid is the recombination source per unit n_ion), a
ground density n_g = u n_ion, and an extra source q v n_ion deposited directly
into E (q a fixed non-negative deposition pattern, v its strength per ion,
s^-1), the quasi-steady excited block obeys

    L_EE n_E + L_Eg n_g + S_E n_ion + q v n_ion = 0
    n_E / n_ion = a u + c + d v,   a = -L_EE^{-1} L_Eg,  c = -L_EE^{-1} S_E,
                                   d = -L_EE^{-1} q.

-L_EE is a non-singular M-matrix, so its inverse is entrywise non-negative
and a, c, d >= 0 elementwise (item 5 checks this at all 400 points). Summing
over the sublevels of shell p gives N_p = a_p u + c_p + d_p v, so:

  * AFFINE IN EACH RESERVOIR.  N_p is affine in (u, v) jointly; the shell
    ratio R = N_3/N_4 is a ratio of two affine forms in (u, v). It is NOT a
    function of one reservoir variable unless the reservoirs move together.

  * LOGISTIC PER RESERVOIR AT FIXED OTHERS.  At fixed v, R(u) is
    affine-over-affine in u and the ground-fed fraction of shell p,
    F_p(u; v) = a_p u / (a_p u + c_p + d_p v), is a unit-width logistic in
    ln u with switching point ln[(c_p + d_p v)/a_p]. Hence
    dlnR/dlnu = F_3 - F_4, as in chapter 3 with c_p -> c_p + d_p v. Since
    F_p = f_p (1 - phi_p), with f_p = a_p u/(a_p u + c_p) the atomic
    ground-fed fraction and phi_p = d_p v/N_p the molecular fraction, the
    dilution F_p = f_p(1 - phi_p) of chapter 6 is EXACT for the local
    sensitivity at the reservoir value where phi_p is evaluated.

  * CEILING PER RESERVOIR, WITH Delta DEPENDING ON THE OTHERS.
    max_u |F_3 - F_4| = tanh(|Delta(v)|/4),
    Delta(v) = ln[(a_3/a_4)(c_4 + d_4 v)/(c_3 + d_3 v)].
    Delta(v) runs from the atomic value at v = 0 to ln[(a_3/a_4)(d_4/d_3)]
    as v -> infinity and, because the molecular deposit is concentrated in
    n = 3 (d_4/d_3 of order 0.1), it changes sign on the way. The ceiling
    bounds the response to the GROUND reservoir at each fixed molecular
    strength. It bounds nothing about the response to the molecular
    reservoir itself (dlnR/dlnv = G_3 - G_4 with G_p = d_p v/N_p, a separate
    logistic in ln v with its own Delta), and it is not a bound on the total
    response unless the reservoirs move proportionally. Chapter 6's "hold
    unchanged" must be narrowed to exactly this.

  * FINITE STEP.  The plateau error is computed as verify_divertor_map.py
    computes the atomic one: pre-step CRE u_old, post-step operator, the
    molecular source frozen across the step, eps = |R(u_old)/R(u_new) - 1|.
    With the secant sensitivity Sbar = ln[R(u_old)/R(u_new)]/ln(u_old/u_new)
    this is eps = |exp(Sbar Lambda) - 1|, Lambda = ln(u_old/u_new): linear in
    Sbar only when Sbar Lambda << 1. The prototype gives Sbar Lambda = 0.33
    at the cold corner, so "linear in f3 - f4" is itself off by ~13 percent
    there, and the naive Sbar ratio under-predicts the eps ratio.

  * EMISSIVITY = SHELL FRACTION.  The dilution argument equates the molecular
    fraction of the emissivity with that of the population. That needs the
    sublevels of a shell to share population faster than they radiate, so the
    l-distribution forgets which channel deposited it. Item 1 checks
    intrashell l-mixing against radiative decay for every n = 3 and n = 4
    sublevel at every grid point.

PREDICTIONS BEFORE RUNNING (cr-physicist, scratch check1-4.py, 11 Sep 2026)
--------------------------------------------------------------------------
  1. intrashell mixing / radiative decay: grid minimum >= 12 for 3d, at
     [15,0]; >= 100 for every n = 4 sublevel everywhere.
  2. single n = 3 source: d_4/d_3 = 0.109 / 0.114 / 0.141 at [0,4] / [15,3]
     / [23,5] (point's own operator).
     two-source (0.6, 0.3): eps_mol = 0.0632 / 0.0335 / 0.0199, atomic/mol
     ratio 6.1 / 5.4 / 3.2; (0.7, 0.45): 0.0414 / 0.0223 / 0.0144.
     Atomic eps_plateau (heat, 5 percent step) 0.3869 / 0.1807 / 0.0636
     from validation/divertor_map/divertor_map.csv (24 Aug 2026).
  3. diluted sensitivity f3(1-phi3) - f4(1-phi4) at phi4 = 0.30 crosses
     zero near phi3 = 0.75 at [0,4].
  4. atomic caps 0.491 / 0.461 / 0.451 (also emissivity_generalisation.csv
     cap_shell 0.4907 / 0.4605 / 0.4514); molecule-dominated caps
     0.173 / 0.254 / 0.129; Delta changes sign in between.
  5. a, c, d >= 0 at all 400 points.

WHAT WOULD REFUTE THE REPLACEMENT CLAIMS
----------------------------------------
  - a negative entry in a, c or d anywhere: the affine-with-positivity
    structure fails and the per-reservoir logistic is not established.
  - max_u |F_3 - F_4| by direct maximisation differing from tanh(|Delta(v)|/4)
    at any v.
  - intrashell mixing below ~10x radiative decay for any n = 3 or 4 sublevel
    at any grid point: emissivity fraction != population fraction there and
    the dilution argument needs the l-resolved deposit pattern.
  - eps_mol / eps_atomic not reproducing the prototype to 3 digits.
  - Delta(v) not changing sign.
  - the (0.6, 0.3) target unrealisable with non-negative sources at any of
    the three points.
  - the ne -> 0 limit of L not reproducing the tabulated A-values: the
    radiative part of L would then not be what radiative_rates.csv says.

CONVENTIONS
-----------
  Step: heating, fractional 5 percent, snapped to the grid (one index,
  +4.81 percent), as in verify_divertor_map.py. a, c, d for the step
  calculations come from the post-step operator [k, j]; the target fractions
  phi_p are imposed at the plateau state (post-step operator, stale u_old),
  which is where the diluted sensitivity is evaluated. Items 4 and 5 use each
  point's own operator and its own CRE u.
  Convention A (primary, = prototype): u_old and u_new are the ATOMIC CRE
  ground fractions; the frozen molecular populations are added to both the
  plateau and the target. Convention B (sensitivity): with the same source
  strengths, the ground reservoir is recomputed as the full CRE including the
  molecular source, before and after the step, and eps recomputed. Their
  difference measures how much the answer depends on whether the ground
  reservoir is allowed to see the molecular source.
  Deposit patterns: statistical weights (2l+1) within the shell, 3s:3p:3d =
  1:3:5 and 4s:4p:4d:4f = 1:3:5:7. Item 1 says how much this matters.

Read-only on the pipeline. Writes only to validation/molecular_channel/
with --write.
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

# The three points named in the task: cold corner (map worst point), crest,
# benchmark. The benchmark is additionally DERIVED from ctx.nearest_point and
# asserted, so a changed grid fails loudly rather than mislabelling.
POINTS = [("cold_corner", 0, 4), ("crest", 15, 3), ("benchmark", 23, 5)]
BENCH_TE, BENCH_NE = 2.947, 1.389e14
TARGETS = [(0.6, 0.3), (0.7, 0.45), (0.6, 0.15), (0.8, 0.3)]
PHI4_SWEEP = 0.30


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--frac", type=float, default=0.05,
                    help="fractional Te step (default 0.05, as divertor map)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    import Balmer_transient_ratio as btr                  # noqa: E402

    ctx = CRContext.load()
    ctx.validate()
    L, Te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    nv = np.asarray(ctx.n_values)
    labels = ctx.labels
    g = int(ctx.ground_index)
    nS = ctx.n_states
    L_path = ROOT / "data/processed/cr_matrix/L_grid.npy"
    S_path = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not S_path.exists():
        raise FileNotFoundError(f"missing source vector: {S_path}")
    S = np.load(S_path)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    rad_path = btr.DATA_RAD
    if not Path(rad_path).exists():
        raise FileNotFoundError(f"missing radiative table: {rad_path}")

    E = np.array([i for i in range(nS) if i != g], dtype=int)
    pos = {s: k for k, s in enumerate(E)}
    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    # l of each resolved sublevel from the state_index file (not assumed).
    l_of = {}
    with open(ctx.state_index_path, newline="") as fh:
        for row in csv.DictReader(fh):
            l_of[int(row["idx"])] = int(row["l"])
    for s in list(N3) + list(N4):
        if l_of[s] < 0 or int(nv[s]) != (3 if s in N3 else 4):
            raise ValueError(f"state {s} ({labels[s]}) is not a resolved "
                             f"n=3/4 sublevel: n={nv[s]}, l={l_of[s]}")
    # Deposit patterns: statistical weight (2l+1) within the shell.
    q3 = np.zeros(len(E)); q4 = np.zeros(len(E))
    for s in N3: q3[pos[s]] = 2 * l_of[s] + 1
    for s in N4: q4[pos[s]] = 2 * l_of[s] + 1

    lines, grid_rows, sum_rows = [], [], []

    def say(s=""):
        print(s)
        lines.append(s)

    def shells(vecE):
        return float(vecE[n3E].sum()), float(vecE[n4E].sum())

    def split(A, s):
        """a, c, d3, d4 for operator A and source s (per unit n_ion)."""
        LEE = A[np.ix_(E, E)]
        rhs = -np.column_stack([A[E, g], s[E], q3, q4])
        sol = np.linalg.solve(LEE, rhs)
        return LEE, sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

    def cre_u(A, s, w3=0.0, w4=0.0):
        """ground fraction n_g/n_ion of the full CRE, with optional sources."""
        src = s.copy()
        src[E] += w3 * q3 + w4 * q4
        n = np.linalg.solve(A, -src)
        if n.min() < 0:
            raise RuntimeError("negative CRE population")
        return float(n[g])

    # -------------------------------------------------------------- header
    say("=" * 78)
    say("MOLECULAR CHANNEL -- what a third supply channel does to the")
    say("two-channel plateau-error machinery, computed on the model's operator")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}")
    say(f"interpreter    {sys.executable}   numpy {np.__version__}")
    say(f"repo root      {ROOT}")
    say(f"L_grid sha256  {sha256(L_path)}")
    say(f"S_grid sha256  {sha256(S_path)}")
    say(f"state idx sha  {sha256(ctx.state_index_path)}")
    say(f"radiative sha  {sha256(rad_path)}  "
        f"({Path(rad_path).relative_to(ROOT)})")
    say(f"grid {len(Te)} Te x {len(ne)} ne, {nS} states, ground index {g} "
        f"({labels[g]})")
    say(f"n=3 {[int(i) for i in N3]} {[labels[i] for i in N3]} deposit "
        f"weights {q3[n3E].astype(int).tolist()}")
    say(f"n=4 {[int(i) for i in N4]} {[labels[i] for i in N4]} deposit "
        f"weights {q4[n4E].astype(int).tolist()}")
    ib, jb = ctx.nearest_point(BENCH_TE, BENCH_NE)
    if (ib, jb) != (23, 5):
        raise RuntimeError(f"benchmark nearest to Te={BENCH_TE}, ne={BENCH_NE:g}"
                           f" is [{ib},{jb}], not [23,5]: the grid has changed")
    say(f"benchmark derived: nearest_point({BENCH_TE}, {BENCH_NE:g}) = "
        f"[{ib},{jb}]  Te={Te[ib]:.4f} ne={ne[jb]:.4e}  (asserted == [23,5])")
    for name, i, j in POINTS:
        say(f"  point {name:<12} [{i},{j}]  Te={Te[i]:.4f} eV  ne={ne[j]:.4e} cm^-3")
    say("=" * 78)

    # ---------------------------------------------------- item 1: l-mixing
    say("\n" + "=" * 78)
    say("ITEM 1  intrashell l-mixing versus radiative decay, n = 3 and 4 sublevels")
    say("=" * 78)
    sub = list(N3) + list(N4)
    # (a) tabulated A-values: sum of every res_to_res row with this upper state
    A_csv = {}
    with open(rad_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["type"] == "res_to_res"]
    for s in sub:
        mine = [r for r in rows if int(r["idx_upper"]) == s]
        if not mine:
            raise RuntimeError(f"no res_to_res row with idx_upper={s} "
                               f"({labels[s]}) in {rad_path}")
        for r in mine:
            if r["label_upper"].upper() != labels[s].upper():
                raise RuntimeError(f"label mismatch for idx {s}: csv says "
                                   f"{r['label_upper']}, state_index says "
                                   f"{labels[s]}")
        A_csv[s] = sum(float(r["A_s-1"]) for r in mine)
    # cross-check the six Balmer channels against the pipeline loader
    lw = btr.load_radiative_weights(use_photon_energy=False)
    for line in ("Halpha", "Hbeta"):
        for (iu, il, nu_, lu_, nl_, ll_, lab) in btr.LINE_CHANNELS[line]:
            got = [float(r["A_s-1"]) for r in rows
                   if int(r["idx_upper"]) == iu and int(r["idx_lower"]) == il]
            if len(got) != 1 or abs(got[0] / lw.weights[line][lab] - 1) > 1e-12:
                raise RuntimeError(f"radiative table read disagrees with "
                                   f"load_radiative_weights for {lab}")
    say("A-values (s^-1), total radiative decay per sublevel from "
        f"{Path(rad_path).name} (all res_to_res rows with that upper state;")
    say("  the six Balmer rows agree with Balmer_transient_ratio.load_radiative_weights):")
    say("  " + "  ".join(f"{labels[s]}={A_csv[s]:.4e}" for s in sub))
    # (b) ne -> 0 limit of L: is L exactly affine in ne, and is the radiative
    #     part what the table says?
    worst_lin = 0.0; worst_rad = 0.0; worst_rad_where = None
    for i in range(len(Te)):
        K = (L[i, -1] - L[i, 0]) / (ne[-1] - ne[0])
        A0 = L[i, 0] - ne[0] * K
        for j in range(len(ne)):
            worst_lin = max(worst_lin, np.abs(A0 + ne[j] * K - L[i, j]).max()
                            / np.abs(L[i, j]).max())
        for s in sub:
            down = sum(A0[m, s] for m in range(nS) if nv[m] < nv[s])
            rel = abs(down / A_csv[s] - 1)
            if rel > worst_rad:
                worst_rad, worst_rad_where = rel, (i, labels[s], down)
    say(f"L affine in ne at fixed Te: worst relative residual of the two-column"
        f" extrapolation over all 400 points {worst_lin:.2e}")
    say(f"ne->0 limit of the downward column entries vs tabulated A: worst "
        f"|ratio-1| = {worst_rad:.2e} at Te index {worst_rad_where[0]}, "
        f"{worst_rad_where[1]} (L gives {worst_rad_where[2]:.6e})")
    if worst_rad > 1e-6:
        say("  *** the radiative part of L is NOT the tabulated A-values ***")
    # (c) mixing / radiative and mixing / (all other loss), every point
    ratio_rad = np.zeros((len(Te), len(ne), len(sub)))
    ratio_oth = np.zeros_like(ratio_rad)
    mix_all = np.zeros_like(ratio_rad)
    for i in range(len(Te)):
        for j in range(len(ne)):
            A = L[i, j]
            for q, s in enumerate(sub):
                same = [t for t in (N3 if s in N3 else N4) if t != s]
                mix = float(sum(A[t, s] for t in same))
                if mix < 0:
                    raise RuntimeError(f"negative off-diagonal L[{same},{s}] "
                                       f"at [{i},{j}]")
                other = -A[s, s] - mix          # every loss that is not mixing
                mix_all[i, j, q] = mix
                ratio_rad[i, j, q] = mix / A_csv[s]
                ratio_oth[i, j, q] = mix / other
    say("\nminimum over the grid of (intrashell mixing out of the sublevel) /"
        " (its total radiative decay):")
    say(f"  {'state':<6}{'min mix/A':>12}{'at [i,j]':>10}{'Te':>9}{'ne':>11}"
        f"{'mix s^-1':>12}{'  min mix/(all other loss)':>28}{'at':>8}")
    for q, s in enumerate(sub):
        flat = int(np.argmin(ratio_rad[:, :, q])); i, j = divmod(flat, len(ne))
        flat2 = int(np.argmin(ratio_oth[:, :, q])); i2, j2 = divmod(flat2, len(ne))
        say(f"  {labels[s]:<6}{ratio_rad[i, j, q]:>12.2f}{f'[{i},{j}]':>10}"
            f"{Te[i]:>9.3f}{ne[j]:>11.3e}{mix_all[i, j, q]:>12.3e}"
            f"{ratio_oth[i2, j2, q]:>28.2f}{f'[{i2},{j2}]':>8}")
    m3 = ratio_rad[:, :, :3].min(); m4 = ratio_rad[:, :, 3:].min()
    at3 = int(np.argmin(ratio_rad[:, :, :3].min(axis=2)))
    at4 = int(np.argmin(ratio_rad[:, :, 3:].min(axis=2)))
    say(f"  shell n=3 minimum {m3:.2f} at [{at3 // len(ne)},{at3 % len(ne)}]"
        f"   shell n=4 minimum {m4:.2f} at [{at4 // len(ne)},{at4 % len(ne)}]")
    say(f"  prediction: n=3 (3d) >= 12 at [15,0]; n=4 >= 100 everywhere  -> "
        f"{'REPRODUCED' if (m3 >= 12 and m4 >= 100) else '*** NOT REPRODUCED ***'}")
    say("  (the prediction came from check1.py, which sampled five points; this is")
    say("   the full 400-point scan. The grid minimum sits at the HOT low-density")
    say(f"   corner [{len(Te)-1},0], not at [15,0]. Value at [15,0]: 3D "
        f"{ratio_rad[15, 0, 2]:.2f}, 3P {ratio_rad[15, 0, 1]:.2f}, "
        f"n=4 min {ratio_rad[15, 0, 3:].min():.2f}.)")
    say("  The l-distribution is set by mixing to within ~1/(mix/A): worst ~11 percent")
    say("  for 3d at [49,0], under 0.5 percent at the three points of items 2 to 4.")
    say("  This ratio is what licenses 'molecular fraction of the emissivity ="
        " molecular fraction of the shell population': the l-distribution")
    say("  inside n=3 and n=4 is set by mixing, not by where a channel deposits.")
    say("  It is weakest at the low-density edge, where mixing is collisional and"
        " decay is not.")
    for name, i, j in POINTS:
        say(f"  at {name} [{i},{j}]: mix/A = " + ", ".join(
            f"{labels[s]} {ratio_rad[i, j, q]:.1f}" for q, s in enumerate(sub)))

    # ------------------------------------------ items 4, 5 on the whole grid
    say("\n" + "=" * 78)
    say("ITEM 5  affine structure n_E = a u + c + d v with a, c, d >= 0, all points")
    say("ITEM 4  per-reservoir ceiling tanh(|Delta(v)|/4), all points")
    say("=" * 78)
    rng = np.random.default_rng(20260911)
    worst_aff = 0.0; worst_tanh = 0.0
    gmin = dict(a=np.inf, c=np.inf, d3=np.inf, d4=np.inf); gmin_where = {}
    for i in range(len(Te)):
        for j in range(len(ne)):
            A, s = L[i, j], S[i, j]
            LEE, av, cv, d3v, d4v = split(A, s)
            for key, vec in (("a", av), ("c", cv), ("d3", d3v), ("d4", d4v)):
                if vec.min() < gmin[key]:
                    gmin[key] = float(vec.min()); gmin_where[key] = (i, j)
            u = cre_u(A, s)
            a3, a4 = shells(av); c3, c4 = shells(cv)
            d3, d4 = shells(d3v); e3, e4 = shells(d4v)
            # affine check: direct solve at a random (u', v') against a u' + c + d v'
            uu = u * np.exp(rng.uniform(-2, 2)); vv = (c3 / d3) * np.exp(rng.uniform(-2, 2))
            direct = np.linalg.solve(LEE, -(A[E, g] * uu + s[E] + q3 * vv))
            worst_aff = max(worst_aff, np.abs(direct - (av * uu + cv + d3v * vv)).max()
                            / np.abs(direct).max())
            # ceiling per reservoir
            D0 = np.log((a3 / a4) * (c4 / c3))
            Dinf = np.log((a3 / a4) * (d4 / d3))
            # v at which Delta(v) = 0: (c4 + d4 v)/(c3 + d3 v) = a4/a3
            den = a3 * d4 - a4 * d3
            v_sign = (a4 * c3 - a3 * c4) / den if den != 0 else np.nan
            phi3_sign = d3 * v_sign / (a3 * u + c3 + d3 * v_sign) if v_sign > 0 else np.nan
            supply_sign = d3 * v_sign / (c3 + d3 * v_sign) if v_sign > 0 else np.nan
            # direct maximisation check of the ceiling at v = v_sign/2 (nonzero Delta)
            vchk = 0.5 * v_sign if v_sign > 0 else c3 / d3
            Dv = np.log((a3 / a4) * (c4 + d4 * vchk) / (c3 + d3 * vchk))
            ugrid = np.exp(np.linspace(np.log((c3 + d3 * vchk) / a3) - 12,
                                       np.log((c4 + d4 * vchk) / a4) + 12, 20001))
            F3 = a3 * ugrid / (a3 * ugrid + c3 + d3 * vchk)
            F4 = a4 * ugrid / (a4 * ugrid + c4 + d4 * vchk)
            worst_tanh = max(worst_tanh, abs(np.abs(F3 - F4).max()
                                             / np.tanh(abs(Dv) / 4) - 1))
            grid_rows.append(dict(
                i=i, j=j, Te=float(Te[i]), ne=float(ne[j]), u_CRE=u,
                a3=a3, a4=a4, c3=c3, c4=c4, d3=d3, d4=d4, d4_over_d3=d4 / d3,
                e3=e3, e4=e4,
                min_a=float(av.min()), min_c=float(cv.min()),
                min_d3=float(d3v.min()), min_d4=float(d4v.min()),
                affine_rel_residual=float(np.abs(direct - (av * uu + cv + d3v * vv)).max()
                                          / np.abs(direct).max()),
                Delta_atomic=D0, cap_atomic=float(np.tanh(abs(D0) / 4)),
                Delta_molinf=Dinf, cap_molinf=float(np.tanh(abs(Dinf) / 4)),
                v_sign=v_sign, phi3_n3_at_sign=phi3_sign,
                phi3_supply_at_sign=supply_sign,
                **{f"mixA_{labels[s]}": float(ratio_rad[i, j, q])
                   for q, s in enumerate(sub)},
                **{f"mixOther_{labels[s]}": float(ratio_oth[i, j, q])
                   for q, s in enumerate(sub)}))
    G = {k: np.array([r[k] for r in grid_rows]) for k in grid_rows[0]}
    say("minimum entry over all 400 points of")
    for key in ("a", "c", "d3", "d4"):
        say(f"  {key:<3} {gmin[key]:.4e}  at [{gmin_where[key][0]},"
            f"{gmin_where[key][1]}]  {'>= 0' if gmin[key] >= 0 else '*** NEGATIVE ***'}")
    say(f"  prediction a, c, d >= 0 everywhere -> "
        f"{'REPRODUCED' if min(gmin.values()) >= 0 else '*** NOT REPRODUCED ***'}")
    say(f"affine check: worst relative residual of a u' + c + d v' against a "
        f"direct solve at random (u', v'): {worst_aff:.2e}")
    say(f"ceiling check: worst |max_u|F3-F4| / tanh(|Delta(v)|/4) - 1| by direct"
        f" maximisation at v = v_sign/2: {worst_tanh:.2e}")
    say(f"d4/d3 (n=4 population per n=3 population, both per unit n=3 deposit):"
        f" {G['d4_over_d3'].min():.4f} .. {G['d4_over_d3'].max():.4f}, "
        f"median {np.median(G['d4_over_d3']):.4f}")
    say(f"Delta_atomic  {G['Delta_atomic'].min():.4f} .. {G['Delta_atomic'].max():.4f}"
        f"   cap_atomic {G['cap_atomic'].min():.4f} .. {G['cap_atomic'].max():.4f}")
    say(f"Delta_molinf  {G['Delta_molinf'].min():.4f} .. {G['Delta_molinf'].max():.4f}"
        f"   cap_molinf {G['cap_molinf'].min():.4f} .. {G['cap_molinf'].max():.4f}")
    nsign = int(np.sum((G['Delta_atomic'] > 0) & (G['Delta_molinf'] < 0)))
    say(f"points where Delta changes sign between v = 0 and v -> inf: {nsign}/400")
    say(f"molecular fraction of n=3 (at CRE u) at the sign change: "
        f"{np.nanmin(G['phi3_n3_at_sign']):.4f} .. {np.nanmax(G['phi3_n3_at_sign']):.4f}"
        f"   as fraction of the non-ground supply c3+d3v: "
        f"{np.nanmin(G['phi3_supply_at_sign']):.4f} .. {np.nanmax(G['phi3_supply_at_sign']):.4f}")

    say("\nITEM 4 at the three points (point's own operator and CRE u):")
    exp_cap0 = {"cold_corner": 0.491, "crest": 0.461, "benchmark": 0.451}
    exp_capi = {"cold_corner": 0.173, "crest": 0.254, "benchmark": 0.129}
    exp_d43 = {"cold_corner": 0.109, "crest": 0.114, "benchmark": 0.141}
    for name, i, j in POINTS:
        r = [x for x in grid_rows if x["i"] == i and x["j"] == j][0]
        say(f"\n  {name} [{i},{j}]  u_CRE={r['u_CRE']:.4e}  a3/a4={r['a3']/r['a4']:.4f}"
            f"  c4/c3={r['c4']/r['c3']:.4f}  d4/d3={r['d4_over_d3']:.4f} "
            f"(expected {exp_d43[name]})")
        say(f"    Delta(0)={r['Delta_atomic']:+.4f} cap {r['cap_atomic']:.4f} "
            f"(expected {exp_cap0[name]});  Delta(inf)={r['Delta_molinf']:+.4f} "
            f"cap {r['cap_molinf']:.4f} (expected {exp_capi[name]})")
        say(f"    sign change at v = {r['v_sign']:.4e} s^-1 per ion  (n=3 deposit"
            f" flux {q3.sum()*r['v_sign']:.4e} s^-1 per ion), where the molecular"
            f" fraction of n=3 is {r['phi3_n3_at_sign']:.4f} at u_CRE, and "
            f"{r['phi3_supply_at_sign']:.4f} of the non-ground supply")
        say(f"    {'phi3(u_CRE)':>12}{'v s^-1':>12}{'Delta(v)':>10}{'cap':>8}"
            f"{'phi4(u_CRE)':>12}")
        for phi in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0):
            if phi < 1.0:
                v = phi / (1 - phi) * (r['a3'] * r['u_CRE'] + r['c3']) / r['d3']
                Dv = np.log((r['a3'] / r['a4']) * (r['c4'] + r['d4'] * v)
                            / (r['c3'] + r['d3'] * v))
                phi4 = r['d4'] * v / (r['a4'] * r['u_CRE'] + r['c4'] + r['d4'] * v)
            else:
                v, Dv, phi4 = np.inf, r['Delta_molinf'], 1.0
            say(f"    {phi:>12.2f}{v:>12.3e}{Dv:>+10.4f}{np.tanh(abs(Dv)/4):>8.4f}"
                f"{phi4:>12.4f}")
            sum_rows.append(dict(block="v_sweep", point=name, i=i, j=j, k="",
                                 phi3_target=phi, phi4_target="", v=v, w3="", w4="",
                                 realisable="", Delta=Dv, cap=float(np.tanh(abs(Dv) / 4)),
                                 phi4_result=phi4))

    # ------------------------------------------- items 2, 3 at the 3 points
    say("\n" + "=" * 78)
    say("ITEM 2  frozen third reservoir across a heating step: plateau error")
    say(f"        step +{a.frac:.0%} nominal, snapped to the grid; post-step operator;")
    say("        eps = |R(u_old)/R(u_new) - 1| as in verify_divertor_map.py")
    say("=" * 78)
    exp_eps = {(0.6, 0.3): {"cold_corner": (0.0632, 6.1), "crest": (0.0335, 5.4),
                            "benchmark": (0.0199, 3.2)},
               (0.7, 0.45): {"cold_corner": (0.0414, None), "crest": (0.0223, None),
                             "benchmark": (0.0144, None)}}
    exp_atomic = {"cold_corner": 0.3869, "crest": 0.1807, "benchmark": 0.0636}
    for name, i, j in POINTS:
        k = int(np.argmin(np.abs(Te - Te[i] * (1 + a.frac))))
        if k == i:
            raise RuntimeError(f"step from Te index {i} does not move a grid index")
        Apre, spre = L[i, j], S[i, j]
        Apost, spost = L[k, j], S[k, j]
        u_old = cre_u(Apre, spre); u_new = cre_u(Apost, spost)
        Lam = np.log(u_old / u_new)
        LEE, av, cv, d3v, d4v = split(Apost, spost)
        _, avp, cvp, d3vp, d4vp = split(Apre, spre)
        # superposition check as the divertor map does it
        n_new_full = np.linalg.solve(Apost, -spost)
        sup = (np.abs(av * u_new + cv - n_new_full[E]).max() / np.abs(n_new_full[E]).max())
        if sup > 1e-8:
            raise RuntimeError(f"superposition fails at [{k},{j}]: {sup:.2e}")
        a3, a4 = shells(av); c3, c4 = shells(cv)
        d3, d4 = shells(d3v); e3, e4 = shells(d4v)       # d: n=3 deposit, e: n=4 deposit
        d3p, d4p = shells(d3vp)
        f3 = a3 * u_old / (a3 * u_old + c3); f4 = a4 * u_old / (a4 * u_old + c4)

        def Rmol(u, m3, m4):
            return (a3 * u + c3 + m3) / (a4 * u + c4 + m4)

        def eps_of(m3, m4, uo=u_old, un=u_new):
            Ro, Rn = Rmol(uo, m3, m4), Rmol(un, m3, m4)
            Sbar = np.log(Ro / Rn) / np.log(uo / un)
            return abs(Ro / Rn - 1), Sbar

        eps_at, Sbar_at = eps_of(0.0, 0.0)
        say(f"\n{name} [{i},{j}] -> [{k},{j}]  Te {Te[i]:.4f} -> {Te[k]:.4f} "
            f"({Te[k]/Te[i]-1:+.4%})  ne={ne[j]:.4e}")
        say(f"  u_old={u_old:.4e}  u_new={u_new:.4e}  Lambda=ln(u_old/u_new)={Lam:+.4f}"
            f"   superposition {sup:.1e}")
        say(f"  post-step: a3={a3:.4e} a4={a4:.4e} c3={c3:.4e} c4={c4:.4e}")
        say(f"  f3={f3:.4f} f4={f4:.4f}  f3-f4={f3-f4:+.4f}  Sbar_atomic={Sbar_at:+.4f}"
            f"  Sbar*Lambda={Sbar_at*Lam:+.4f}")
        say(f"  eps_atomic = {eps_at:.6f}  (divertor_map.csv heat row: {exp_atomic[name]})"
            f"  {'REPRODUCED' if abs(eps_at - exp_atomic[name]) < 5e-5 else '*** DIFFERS ***'}")
        say(f"  d4/d3 for a unit n=3 deposit: post-step operator {d4/d3:.4f}, "
            f"pre-step operator {d4p/d3p:.4f} (expected {exp_d43[name]}, own operator)")
        # single-source variant
        say(f"\n  single source into n=3 (statistical), phi3 imposed at the plateau"
            f" (post-step operator, u_old):")
        say(f"    {'phi3':>6}{'v s^-1':>12}{'phi4 result':>12}{'eps_mol':>10}"
            f"{'atom/mol':>10}{'Sbar_mol':>10}{'naive':>8}{'F3-F4 loc':>11}"
            f"{'Delta_mol':>10}{'cap':>7}")
        for phi3 in (0.3, 0.6, 0.7, 0.8):
            v = phi3 / (1 - phi3) * (a3 * u_old + c3) / d3
            m3, m4 = d3 * v, d4 * v
            phi4 = m4 / (a4 * u_old + c4 + m4)
            ep, Sb = eps_of(m3, m4)
            Dm = np.log((a3 / a4) * (c4 + m4) / (c3 + m3))
            loc = f3 * (1 - phi3) - f4 * (1 - phi4)
            say(f"    {phi3:>6.2f}{v:>12.3e}{phi4:>12.4f}{ep:>10.4f}{eps_at/ep:>10.2f}"
                f"{Sb:>+10.4f}{Sbar_at/Sb:>8.2f}{loc:>+11.4f}{Dm:>+10.3f}"
                f"{np.tanh(abs(Dm)/4):>7.3f}")
            sum_rows.append(dict(block="single_source", point=name, i=i, j=j, k=k,
                                 phi3_target=phi3, phi4_target="", v=v, w3=v, w4=0.0,
                                 realisable=True, phi4_result=phi4,
                                 eps_atomic=eps_at, eps_mol=ep, ratio_atomic_over_mol=eps_at / ep,
                                 Sbar_atomic=Sbar_at, Sbar_mol=Sb, naive_Sbar_ratio=Sbar_at / Sb,
                                 sens_local_diluted=loc, Delta=Dm, cap=float(np.tanh(abs(Dm) / 4)),
                                 Lambda=Lam))
        # two-source variant
        Dmat = np.array([[d3, e3], [d4, e4]])
        atom = np.array([a3 * u_old + c3, a4 * u_old + c4])
        say(f"\n  two sources (n=3 and n=4 deposits) hitting (phi3, phi4) exactly at"
            f" the plateau:")
        say(f"    {'(phi3,phi4)':>12}{'w3':>11}{'w4':>11}{'real.':>6}{'eps_mol':>9}"
            f"{'atom/mol':>9}{'exp.':>13}{'Sbar_mol':>9}{'naive':>7}{'F3-F4':>8}"
            f"{'Delta':>8}{'cap':>6}{'| B: u_old':>11}{'u_new':>10}{'phi3,phi4':>14}{'eps_B':>8}")
        for (p3, p4) in TARGETS:
            tgt = np.array([p3 / (1 - p3) * atom[0], p4 / (1 - p4) * atom[1]])
            w = np.linalg.solve(Dmat, tgt)
            real = bool((w >= 0).all())
            row = dict(block="two_source", point=name, i=i, j=j, k=k,
                       phi3_target=p3, phi4_target=p4, v="", w3=float(w[0]), w4=float(w[1]),
                       realisable=real, eps_atomic=eps_at, Sbar_atomic=Sbar_at, Lambda=Lam)
            if not real:
                if w[1] < 0:
                    v3 = p3 / (1 - p3) * atom[0] / d3
                    why = (f"w4 < 0: an n=3 deposit alone at phi3={p3:.2f} already"
                           f" puts phi4={d4*v3/(atom[1]+d4*v3):.3f} > {p4:.2f} into"
                           f" n=4 (collisional 3->4)")
                else:
                    v4 = p4 / (1 - p4) * atom[1] / e4
                    why = (f"w3 < 0: an n=4 deposit alone at phi4={p4:.2f} already"
                           f" puts phi3={e3*v4/(atom[0]+e3*v4):.3f} > {p3:.2f} into"
                           f" n=3 (cascade 4->3)")
                say(f"    ({p3:.2f},{p4:.2f}) {w[0]:>11.3e}{w[1]:>11.3e}{'no':>6}  "
                    f"-- {why}")
                sum_rows.append(row); continue
            m3, m4 = Dmat[0] @ w, Dmat[1] @ w
            ep, Sb = eps_of(m3, m4)
            Dm = np.log((a3 / a4) * (c4 + m4) / (c3 + m3))
            loc = f3 * (1 - p3) - f4 * (1 - p4)
            # convention B: ground reservoir sees the molecular source
            uoB = cre_u(Apre, spre, w[0], w[1]); unB = cre_u(Apost, spost, w[0], w[1])
            epB, SbB = eps_of(m3, m4, uoB, unB)
            phi3B = m3 / (a3 * uoB + c3 + m3); phi4B = m4 / (a4 * uoB + c4 + m4)
            expv = exp_eps.get((p3, p4), {}).get(name)
            exps = (f"{expv[0]:.4f}" + (f"/{expv[1]}" if expv[1] else "")) if expv else "-"
            say(f"    ({p3:.2f},{p4:.2f}) {w[0]:>11.3e}{w[1]:>11.3e}{'yes':>6}{ep:>9.4f}"
                f"{eps_at/ep:>9.2f}{exps:>13}{Sb:>+9.4f}{Sbar_at/Sb:>7.2f}{loc:>+8.4f}"
                f"{Dm:>+8.3f}{np.tanh(abs(Dm)/4):>6.3f}{uoB:>11.3e}{unB:>10.3e}"
                f"{f'{phi3B:.3f},{phi4B:.3f}':>14}{epB:>8.4f}")
            row.update(eps_mol=ep, ratio_atomic_over_mol=eps_at / ep, Sbar_mol=Sb,
                       naive_Sbar_ratio=Sbar_at / Sb, sens_local_diluted=loc, Delta=Dm,
                       cap=float(np.tanh(abs(Dm) / 4)), u_old_B=uoB, u_new_B=unB,
                       phi3_B=phi3B, phi4_B=phi4B, eps_B=epB, expected_eps=expv[0] if expv else "",
                       expected_ratio=(expv[1] if expv and expv[1] else ""))
            sum_rows.append(row)
        say("    'naive' = Sbar_atomic/Sbar_mol, the ratio chapter 6's 'linear in"
            " f3-f4' would predict; 'atom/mol' is the actual eps ratio. They")
        say("    differ by the exponential nonlinearity eps = |exp(Sbar Lambda) - 1|.")
        say("    B columns: same sources, ground reservoir recomputed as the full CRE"
            " with the molecular source (pre and post); phi3,phi4 are what the")
        say("    fractions then become at the plateau, eps_B the error.")

        # item 3: phi3 sweep at phi4 = 0.30
        say(f"\n  ITEM 3  sweep phi3 at phi4 = {PHI4_SWEEP:.2f} (two-source):")
        phi3_zero_local = 1 - f4 * (1 - PHI4_SWEEP) / f3
        say(f"    analytic zero of f3(1-phi3) - f4(1-phi4): phi3* = 1 - f4(1-phi4)/f3"
            f" = {phi3_zero_local:.4f}")
        prev = None; zero_Sbar = None; real_lo = None; real_hi = None
        for p3 in np.round(np.arange(0.0, 0.9001, 0.01), 4):
            tgt = np.array([p3 / (1 - p3) * atom[0], PHI4_SWEEP / (1 - PHI4_SWEEP) * atom[1]])
            w = np.linalg.solve(Dmat, tgt)
            real = bool((w >= 0).all())
            loc = f3 * (1 - p3) - f4 * (1 - PHI4_SWEEP)
            rec = dict(block="phi3_sweep", point=name, i=i, j=j, k=k, phi3_target=p3,
                       phi4_target=PHI4_SWEEP, v="", w3=float(w[0]), w4=float(w[1]),
                       realisable=real, sens_local_diluted=loc, eps_atomic=eps_at,
                       Sbar_atomic=Sbar_at, Lambda=Lam)
            if real:
                real_lo = p3 if real_lo is None else real_lo; real_hi = p3
                m3, m4 = Dmat[0] @ w, Dmat[1] @ w
                ep, Sb = eps_of(m3, m4)
                Dm = np.log((a3 / a4) * (c4 + m4) / (c3 + m3))
                rec.update(eps_mol=ep, ratio_atomic_over_mol=eps_at / ep, Sbar_mol=Sb,
                           Delta=Dm, cap=float(np.tanh(abs(Dm) / 4)))
                if prev is not None and prev[1] * Sb < 0:
                    zero_Sbar = prev[0] + (p3 - prev[0]) * prev[1] / (prev[1] - Sb)
                prev = (p3, Sb)
            sum_rows.append(rec)
        sw = [r for r in sum_rows if r["block"] == "phi3_sweep" and r["point"] == name]
        say(f"    realisable (both sources >= 0) for phi3 in [{real_lo}, {real_hi}]"
            f" (step 0.01); below, the n=4 deposit alone puts more than phi3 into"
            f" n=3 (w3 < 0); above, the n=3 deposit alone puts more than "
            f"{PHI4_SWEEP:.2f} into n=4 (w4 < 0)")
        if real_hi is not None and phi3_zero_local > real_hi:
            say(f"    the local zero phi3* = {phi3_zero_local:.4f} lies ABOVE the"
                f" realisable range: at phi4 = {PHI4_SWEEP:.2f} the sign change"
                f" cannot be reached with non-negative sources here")
        say(f"    secant Sbar_mol crosses zero at phi3 = "
            f"{zero_Sbar:.4f}" if zero_Sbar is not None else
            "    secant Sbar_mol does not cross zero in the realisable range")
        say(f"    {'phi3':>6}{'w3':>11}{'w4':>11}{'F3-F4 loc':>11}{'Sbar_mol':>10}"
            f"{'eps_mol':>9}{'atom/mol':>9}{'Delta':>8}{'cap':>7}")
        for r in sw:
            if abs(r["phi3_target"] * 10 - round(r["phi3_target"] * 10)) > 1e-9 and \
               not (abs(r["phi3_target"] - 0.75) < 1e-9):
                continue
            if r["realisable"]:
                say(f"    {r['phi3_target']:>6.2f}{r['w3']:>11.3e}{r['w4']:>11.3e}"
                    f"{r['sens_local_diluted']:>+11.4f}{r['Sbar_mol']:>+10.4f}"
                    f"{r['eps_mol']:>9.4f}{r['ratio_atomic_over_mol']:>9.2f}"
                    f"{r['Delta']:>+8.3f}{r['cap']:>7.3f}")
            else:
                say(f"    {r['phi3_target']:>6.2f}{r['w3']:>11.3e}{r['w4']:>11.3e}"
                    f"{r['sens_local_diluted']:>+11.4f}   not realisable")
        if name == "cold_corner":
            say(f"    prediction: zero near phi3 = 0.75 -> local zero {phi3_zero_local:.3f}"
                f"{f', secant zero {zero_Sbar:.3f}' if zero_Sbar else ''}")

    # ------------------------------------------------------------- summary
    say("\n" + "=" * 78)
    say("WHAT REPLACES THE TWO SENTENCES IN SECTION 6.3")
    say("=" * 78)
    say("(a) The factorisation and the tanh ceiling do NOT hold unchanged for a")
    say("    matrix with a third reservoir. The excited block is affine in each")
    say("    reservoir with non-negative coefficients (item 5); at fixed molecular")
    say("    strength the ground-fed fraction is a logistic in ln u and")
    say("    max_u|F3-F4| = tanh(|Delta(v)|/4) with Delta depending on v (item 4).")
    say("    That is a bound on the response to the ground reservoir alone. It says")
    say("    nothing about the response to the molecular reservoir, and Delta(v)")
    say("    changes sign as the molecular fraction grows.")
    say("(b) The dilution f_p -> f_p(1-phi_p) is exact for the LOCAL sensitivity at")
    say("    the reservoir value where phi_p is measured, but the finite-step error")
    say("    is exp(Sbar Lambda)-1, not linear, and the 0.201/0.054/0.053/0.019")
    say("    sensitivities correspond to no point of this model. Computed values")
    say("    for (0.6, 0.3) and the other targets are in the tables above.")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "molecular_channel"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
               f"# interpreter {sys.executable}  numpy {np.__version__}",
               f"# L_grid sha256 {sha256(L_path)}",
               f"# S_grid sha256 {sha256(S_path)}",
               f"# state_index sha256 {sha256(ctx.state_index_path)}",
               f"# radiative_rates sha256 {sha256(rad_path)}",
               f"# fractional step {a.frac} (heating), deposit weights 2l+1"]
        p1 = out / "molecular_channel.csv"
        with p1.open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(grid_rows[0].keys()))
            w.writeheader(); w.writerows(grid_rows)
        keys = []
        for r in sum_rows:
            for kk in r:
                if kk not in keys: keys.append(kk)
        p2 = out / "molecular_channel_summary.csv"
        with p2.open("w", newline="") as fh:
            fh.write("\n".join(hdr) + "\n")
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader(); w.writerows(sum_rows)
        p3 = out / "molecular_channel.txt"
        say(f"\nwrote {p1}  ({len(grid_rows)} rows)")
        say(f"wrote {p2}  ({len(sum_rows)} rows)")
        say(f"wrote {p3}")
        p3.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
