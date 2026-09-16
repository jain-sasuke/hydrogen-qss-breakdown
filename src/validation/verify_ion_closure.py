#!/usr/bin/env python
"""
verify_ion_closure.py
=====================
Stamp the 44-state ion-closure test that chapter 6 section 6.6 and chapter 4
(tab:closure) quote from outputs/findings_10_four_agent_review.md section 3.1.

WHY THIS EXISTS
---------------
The 43-state operator L holds the ion density fixed: recombination enters as
the constant source S*n_i and atoms that ionise leave the accounting. The slow
eigenvalue of L is therefore the CR ionisation rate of ground-state hydrogen
against a prescribed reservoir, not the equilibration rate of a closed system.
findings_10 section 3.1 closed the manifold by appending an ion state and
reported the change in tau_slow = 1/|lambda_0| at four points, over the grid,
and through the ELM census. Those numbers were reproduced by hand but no script
under src/validation/ computes them and no file under validation/ holds them
(claim_hierarchy.md:176-179; backlog "Found beyond the review" item 6). This
script is that file.

THE CONSTRUCTION
----------------
S_grid.npy holds the recombination feed per unit ion density: at every grid
point n = -L^{-1} S is the CRE population for n_i = 1 cm^-3, and S/n_e grows
with n_e (radiative plus three-body), so S has units s^-1 per ion. Appending
the ion as state 44,

    L_c = [[ L,           S      ],
           [ -colsum(L),  -sum(S) ]]

every column of L_c sums to zero: each atom that ionises out of column j (the
only loss in L not balanced by a gain in another atomic row, so -colsum(L)_j is
exactly the ionisation rate out of state j) lands in the ion row, and each
recombining ion lands in the atomic rows. Total particles, atoms plus ions, are
conserved. n_e stays fixed; this closes the atomic manifold, not the electron
balance, and adds no transport.

L_c has an exact zero eigenvalue (the conserved total). Its slow non-zero
eigenvalue is the closed-system relaxation rate. Because a zero eigenvalue
sitting next to a small negative one is where eigen-solvers lose digits, the
closed eigenvalue is computed two ways and the two are compared:
  (a) eigvals(L_c), discarding the eigenvalue nearest zero;
  (b) eigvals(L - S 1^T), the 43x43 reduction obtained by eliminating the ion
      through n_ion = -sum(n_atoms) on the zero-total subspace, which carries
      exactly the non-zero spectrum of L_c and has no zero eigenvalue.
  (c) the exact secular equation for the ground-dominated slow mode,
      lambda = A_gg - A_gE (A_EE - lambda I)^{-1} A_Eg, iterated from 0, applied
      to both L (open) and L - S 1^T (closed). A_EE has no small eigenvalue,
      so this resolves a lambda of order 1e-2 s^-1 inside an operator whose
      entries reach 1e11 s^-1 to near machine precision RELATIVE to lambda,
      where a dense eigen-solver is accurate only relative to the norm.
The open eigenvalue is likewise computed from eigvals(L) and from (c). At the
five table points every value is also referred to a 40-digit mpmath eig of the
same matrices, which is the arbiter when the float methods disagree. The
secular value is the one written to the CSV as tau_open_s / tau_closed_s;
the eig values are kept alongside.

THE CENSUS
----------
The chapter's census is verify_divertor_map.py's LOWER-bound count at
tau_d = 100 us: eps_bar = eps_plateau*(tQ/td)*(1-exp(-td/tQ)) > threshold over
the window_ok rows of divertor_map.csv, both step directions. Here tQ is
replaced by the closed tau_slow of the SAME post-step operator L[k,j] and
nothing else changes: eps_plateau and window_ok are taken from the CSV (window_ok
is decided from the open operator, as the thesis does; the closure leaves
tau_relax alone, so the window test would move only where tau_slow itself does).
The post-step index k is recomputed from the fractional step recorded in the
CSV header and the CSV's own tau_QSS is checked against eig(L[k,j]) row by row,
so a stale CSV or a wrong k fails loudly.

THE FALSIFIER
-------------
Chapter 6 says a closure factor of tau_slow/tau_d ~ 2300 at the worst-case
operator was named as the falsifier. This script reports (i) that ratio;
(ii) the lower bound the worst point would have at that factor, which is
eps_plateau*(1-1/e), still above threshold; (iii) the factor actually needed to
push the worst point below threshold; (iv) the smallest uniform factor on
tau_slow that changes the census count by one; and (v) the census under a
uniform factor 2300. These are reported, not adjudicated: what "2300 removes the
persistence" means is left to the reader with the numbers in front of them.

PREDICTIONS, WRITTEN BEFORE RUNNING
-----------------------------------
From findings_10 section 3.1 and chapter6.tex lines 933-960:
  [0,0]   open 67.23 s      closed 1.6226 s     factor 41.4
  [1,4]   open 0.23324 s    closed 0.015173 s   factor 15.4
  [15,3]  open 2.027 ms     closed 1.999 ms     factor 1.01
  [23,5]  open 22.728 us    closed 22.708 us    factor 1.00
  [0,4]   open 0.421012 s
  grid factor: min 1.00, median 1.00, max 41.4
  census at 100 us: 202 -> 200 of 680; worst 0.38682 -> 0.38563;
  Te >= 2 eV: 45 -> 45
  tau_slow[1,4]/tau_d = 2332 (the "2300")
From derivation_07:296: lambda_0(open) at [0,0] = 2.286 x K_ion(1S) n_e, and
  K_ion(1S) n_e = -colsum(L[0,0])[ground].
What would refute the chapter: any table entry off in the third significant
figure, a factor below 1 anywhere (closure can only add a decay channel to the
atoms, so tau_slow cannot lengthen), a census count other than 200, or a column
sum of L_c that does not vanish.

RESULT NOTE, added after the first run (11 Sep 2026, labelled as post hoc)
-------------------------------------------------------------------------
numpy eig(L) at [0,0] returns tau = 67.2333 s; the secular equation and the
40-digit eig both return 67.2404 s. The recorded "67.23 s" is therefore the
float-eig value and is correct to three digits, not four; the same applies to
"0.421012 s" at [0,4] (true 0.421031 s). Closed values and all factors agree
across methods at the quoted precision. This is a numerics finding about the
eigen-solver, not about the physics; it is reported here and not repaired in
any document.

Read-only with respect to the pipeline. Writes only validation/ion_closure/.
"""

from __future__ import annotations

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
TAU_D = 1e-4                       # the ELM_crash drive of verify_divertor_map
COLSUM_TOL = 1e-12                 # relative to the column's largest entry

# Recorded values, recomputed here and compared, never used in a computation.
REC_TABLE = {
    (0, 0): (67.23, 1.6226, 41.4),
    (1, 4): (0.23324, 0.015173, 15.4),
    (15, 3): (2.027e-3, 1.999e-3, 1.01),
    (23, 5): (22.728e-6, 22.708e-6, 1.00),
}
REC_L04_OPEN = 0.421012
REC_CENSUS = dict(n_open=202, n_closed=200, n_win=680,
                  worst_open=0.38682, worst_closed=0.38563,
                  n_open_te2=45, n_closed_te2=45)
REC_FALSIFIER = 2300.0
REC_LAMBDA_RATIO_00 = 2.286


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def secular(A: np.ndarray, g: int) -> tuple[float, float]:
    """
    Slow eigenvalue of A from lambda = A_gg - A_gE (A_EE - lambda I)^-1 A_Eg,
    iterated from lambda = 0. Exact for the eigenvalue whose eigenvector has a
    non-zero ground component. The map contracts at rate
    |A_gE (A_EE - lambda)^-2 A_Eg| << 1 here, so it converges linearly to the
    round-off floor set by the cancellation between A_gg and the correction;
    iteration stops when the relative change is below 1e-14 or has stopped
    shrinking for three steps. Returns (lambda, last relative change) so the
    achieved precision is reported, not assumed.
    """
    n = A.shape[0]
    E = [i for i in range(n) if i != g]
    AEE, AEg, AgE, Agg = A[np.ix_(E, E)], A[E, g], A[g, E], A[g, g]
    lam, best, stall = 0.0, np.inf, 0
    for _ in range(500):
        new = Agg - AgE @ np.linalg.solve(AEE - lam * np.eye(n - 1), AEg)
        rel = abs(new - lam) / abs(new)
        lam = new
        if rel <= 1e-14:
            break
        if rel < best:
            best, stall = rel, 0
        else:
            stall += 1
            if stall >= 3:
                break
    else:
        raise RuntimeError("secular iteration did not settle in 500 steps")
    if lam >= 0:
        raise RuntimeError("secular slow eigenvalue is non-negative")
    return float(lam), float(rel)


def slow_open(L: np.ndarray, g: int) -> tuple[float, float, float]:
    """lambda_0 of the open operator: (eig(L), secular, secular rel. change)."""
    lam = np.linalg.eigvals(L)
    l0 = lam[np.argmax(lam.real)]
    if l0.real >= 0:
        raise RuntimeError("open operator has a non-negative eigenvalue")
    if abs(l0.imag) > 1e-8 * abs(l0.real):
        raise RuntimeError(f"open slow eigenvalue is complex: {l0}")
    ls, rel = secular(L, g)
    return float(l0.real), ls, rel


def slow_closed(Lc: np.ndarray, L: np.ndarray, S: np.ndarray, g: int
                ) -> tuple[float, float, float, float, float]:
    """
    Slow non-zero eigenvalue of L_c: (eig(L_c) second from top, eig(L - S 1^T),
    secular on L - S 1^T, its rel. change), plus |eigenvalue nearest zero| of L_c.
    """
    lam = np.linalg.eigvals(Lc)
    order = np.argsort(lam.real)[::-1]
    zero, l0 = lam[order[0]], lam[order[1]]
    Lred = L - np.outer(S, np.ones(L.shape[0]))
    lam_r = np.linalg.eigvals(Lred)
    l0_red = lam_r[np.argmax(lam_r.real)]
    if l0.real >= 0 or l0_red.real >= 0:
        raise RuntimeError("closed operator has a second non-negative eigenvalue")
    if abs(l0.imag) > 1e-8 * abs(l0.real):
        raise RuntimeError(f"closed slow eigenvalue is complex: {l0}")
    ls, rel = secular(Lred, g)
    return float(l0.real), float(l0_red.real), ls, rel, float(abs(zero))


def mp_slow(A: np.ndarray, which: int = 0) -> float:
    """Real part of the (which)-th largest eigenvalue of A at 40 digits."""
    import mpmath as mp
    mp.mp.dps = 40
    ev, _ = mp.eig(mp.matrix(A.tolist()))
    re = sorted([mp.re(e) for e in ev], reverse=True)
    return float(re[which])


def rounds_to(value: float, recorded: float) -> bool:
    """True if value, rounded to the significant figures of recorded, equals it."""
    txt = f"{recorded:.15g}"
    mant = txt.split("e")[0].replace("-", "").replace(".", "").lstrip("0")
    sig = len(mant)
    return float(f"{value:.{sig}g}") == float(f"{recorded:.{sig}g}")


def lower_bound(ep: float, tq: float, td: float) -> float:
    return ep * (tq / td) * (1.0 - np.exp(-td / tq))


def tau_for_threshold(ep: float, thr: float, td: float) -> float:
    """tau at which eps_plateau*(tau/td)*(1-exp(-td/tau)) equals thr, or nan."""
    if ep <= thr:
        return float("nan")            # never above threshold at any tau
    lo, hi = 1e-12, 1e6
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if lower_bound(ep, mid, td) > thr:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def main() -> int:
    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    nT, nN, nS = L.shape[:3]

    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    dp = ROOT / "validation/divertor_map/divertor_map.csv"
    for p in (sp, dp):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
    S = np.load(sp)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")

    out = ROOT / "validation" / "ion_closure"
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    shas = {"L_grid.npy": sha256(lp), "S_grid.npy": sha256(sp),
            "state_index.csv": sha256(ctx.state_index_path),
            "divertor_map.csv": sha256(dp)}
    say("=" * 78)
    say("ION CLOSURE: tau_slow with the ion as a 44th state, open versus closed")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}")
    say(f"interpreter {sys.executable}   numpy {np.__version__}")
    say(f"repo root {ROOT}")
    for k, v in shas.items():
        say(f"sha256 {k:<18} {v}")
    say(f"grid {nT} Te x {nN} ne x {nS} states; ground index {g} "
        f"({ctx.labels[g]})")
    say("=" * 78)

    # ---- 1. what S is, and the sign structure the construction relies on ----
    say("\n1. PRECONDITIONS")
    cs = L.sum(axis=2)                       # column sums, shape (nT, nN, nS)
    say(f"  column sums of L: max {cs.max():.4e}, min {cs.min():.4e} s^-1 "
        f"(all must be <= 0: the unbalanced loss is ionisation)")
    if cs.max() > 0:
        raise RuntimeError("a column of L sums to a positive number; the "
                           "ion row would then be a source, not a sink")
    say(f"  S >= 0 everywhere: {bool((S >= 0).all())}; "
        f"sum(S)/ne at [0,0] {S[0,0].sum()/ne[0]:.3e}, at [0,{nN-1}] "
        f"{S[0,nN-1].sum()/ne[nN-1]:.3e} cm^3 s^-1  (grows with ne: "
        f"radiative + three-body; S is the rate per ion, units s^-1)")
    if not (S >= 0).all():
        raise RuntimeError("S has a negative entry")
    n00 = np.linalg.solve(L[0, 0], -S[0, 0])
    say(f"  CRE with n_i = 1 at [0,0]: n(1S) = {n00[g]:.4f} cm^-3 "
        f"(neutral/ion ratio at 1 eV, 1e12; S is per unit ion density)")

    # ---- 2. build, check, diagonalise at all points -----------------------
    say("\n2. BUILD L_c AT ALL POINTS, CHECK CONSERVATION, DIAGONALISE")
    rows = []
    worst_colsum = 0.0
    worst_zero = 0.0
    worst_open_x = 0.0
    worst_open_at = (-1, -1)
    worst_closed_x = 0.0
    worst_sec = 0.0
    ones = np.ones(nS)
    for i in range(nT):
        for j in range(nN):
            Lij, Sij = L[i, j], S[i, j]
            Lc = np.zeros((nS + 1, nS + 1))
            Lc[:nS, :nS] = Lij
            Lc[:nS, nS] = Sij
            Lc[nS, :nS] = -Lij.sum(axis=0)
            Lc[nS, nS] = -Sij.sum()
            col_res = np.abs(Lc.sum(axis=0)) / np.abs(Lc).max(axis=0)
            worst_colsum = max(worst_colsum, col_res.max())
            if col_res.max() > COLSUM_TOL:
                raise RuntimeError(
                    f"column sums of L_c do not vanish at [{i},{j}]: "
                    f"relative residual {col_res.max():.3e}")
            l0_eig, l0, rel_o = slow_open(Lij, g)
            l0c_eig, l0c_red, l0c, rel_c, zero = slow_closed(Lc, Lij, Sij, g)
            worst_sec = max(worst_sec, rel_o, rel_c)
            tau_o, tau_c = -1.0 / l0, -1.0 / l0c
            xo = abs(l0_eig / l0 - 1.0)
            xc = max(abs(l0c_eig / l0c - 1.0), abs(l0c_red / l0c - 1.0))
            if xo > worst_open_x:
                worst_open_x, worst_open_at = xo, (i, j)
            worst_closed_x = max(worst_closed_x, xc)
            worst_zero = max(worst_zero, zero / abs(l0c))
            if tau_c > tau_o * (1 + 1e-9):
                raise RuntimeError(
                    f"closed tau_slow exceeds open at [{i},{j}]: the closure "
                    f"can only add a decay channel to the atoms")
            rows.append(dict(
                i=i, j=j, Te_eV=float(te[i]), ne_cm3=float(ne[j]),
                lambda0_open=l0, lambda0_open_eig=l0_eig,
                lambda0_closed=l0c, lambda0_closed_eig44=l0c_eig,
                lambda0_closed_eig_reduced=l0c_red,
                secular_rel_change_open=rel_o, secular_rel_change_closed=rel_c,
                zero_eig_over_lambda0_closed=zero / abs(l0c),
                tau_open_s=tau_o, tau_closed_s=tau_c, factor=tau_o / tau_c,
                colsum_resid_rel=float(col_res.max()),
                K_ion_ground_ne=float(-cs[i, j, g])))
    F = np.array([r["factor"] for r in rows])
    say(f"  400 points built. worst relative column-sum residual of L_c: "
        f"{worst_colsum:.2e}  (tolerance {COLSUM_TOL:.0e})")
    say(f"  worst |zero eigenvalue| / |lambda_0 closed|: {worst_zero:.2e}")
    say(f"  open   lambda_0: eig(L) vs secular equation, worst rel. diff "
        f"{worst_open_x:.2e} at [{worst_open_at[0]},{worst_open_at[1]}]")
    say(f"  closed lambda_0: eig(L_c) and eig(L - S 1^T) vs secular, worst rel. "
        f"diff {worst_closed_x:.2e}")
    say(f"  secular iteration: worst final relative change over all 800 solves "
        f"{worst_sec:.2e} (its round-off floor; the CSV carries it per point)")
    say("  (tau_open_s / tau_closed_s below and in the CSV are the secular values)")
    say(f"  closed tau_slow <= open tau_slow at every point: True (asserted)")

    # ---- 3. the table -------------------------------------------------------
    say("\n3. THE TABLE (chapter6.tex 933-941, chapter4.tex tab:closure)")
    say(f"  {'point':>8} {'Te':>7} {'ne':>10} {'tau open':>12} {'tau closed':>12} "
        f"{'factor':>7}  {'recorded (open, closed, factor)':>34}  verdict")
    by = {(r["i"], r["j"]): r for r in rows}
    summary = []
    try:
        import mpmath  # noqa: F401
        have_mp = True
    except ImportError:
        have_mp = False
        say("  mpmath not importable: the 40-digit referee column is skipped")
    for (i, j), (ro, rc, rf) in REC_TABLE.items():
        r = by[(i, j)]
        d_o = abs(r["tau_open_s"] / ro - 1)
        d_c = abs(r["tau_closed_s"] / rc - 1)
        d_f = abs(r["factor"] / rf - 1)
        ok_o, ok_c, ok_f = (rounds_to(r["tau_open_s"], ro),
                            rounds_to(r["tau_closed_s"], rc),
                            rounds_to(r["factor"], rf))
        verdict = "OK" if (ok_o and ok_c and ok_f) else "!!"
        say(f"  [{i:>2},{j}] {te[i]:>7.4f} {ne[j]:>10.3e} "
            f"{r['tau_open_s']:>12.5e} {r['tau_closed_s']:>12.5e} "
            f"{r['factor']:>7.2f}  ({ro:.5g}, {rc:.5g}, {rf:.3g})"
            f"{'':>6}  {verdict} rounds to record: open {ok_o} closed {ok_c} "
            f"factor {ok_f}")
        Lij, Sij = L[i, j], S[i, j]
        line = (f"         eig(L) tau {-1/r['lambda0_open_eig']:.6e}   "
                f"secular {r['tau_open_s']:.6e}")
        if have_mp:
            mo = -1.0 / mp_slow(Lij)
            mc = -1.0 / mp_slow(Lij - np.outer(Sij, np.ones(nS)))
            line += (f"   mp40 open {mo:.6e}  mp40 closed {mc:.6e}  "
                     f"(secular/mp40 - 1: open {r['tau_open_s']/mo-1:+.1e}, "
                     f"closed {r['tau_closed_s']/mc-1:+.1e})")
            summary.append(dict(item=f"tau_open_mp40[{i},{j}]", value=mo,
                                recorded=ro, rel_diff=abs(mo / ro - 1)))
            summary.append(dict(item=f"tau_closed_mp40[{i},{j}]", value=mc,
                                recorded=rc, rel_diff=abs(mc / rc - 1)))
        say(line)
        summary.append(dict(item=f"tau_open[{i},{j}]", value=r["tau_open_s"],
                            recorded=ro, rel_diff=d_o))
        summary.append(dict(item=f"tau_open_numpy_eig[{i},{j}]",
                            value=-1 / r["lambda0_open_eig"], recorded=ro,
                            rel_diff=abs(-1 / r["lambda0_open_eig"] / ro - 1)))
        summary.append(dict(item=f"tau_closed[{i},{j}]", value=r["tau_closed_s"],
                            recorded=rc, rel_diff=d_c))
        summary.append(dict(item=f"factor[{i},{j}]", value=r["factor"],
                            recorded=rf, rel_diff=d_f))
    r04 = by[(0, 4)]
    say(f"  L[0,4] open tau_slow: secular {r04['tau_open_s']:.6f} s, eig(L) "
        f"{-1/r04['lambda0_open_eig']:.6f} s  (recorded {REC_L04_OPEN}; secular "
        f"rounds to record: {rounds_to(r04['tau_open_s'], REC_L04_OPEN)}, eig "
        f"rounds to record: {rounds_to(-1/r04['lambda0_open_eig'], REC_L04_OPEN)}); "
        f"closed {r04['tau_closed_s']:.6f} s, factor {r04['factor']:.2f}")
    if have_mp:
        mo4 = -1.0 / mp_slow(L[0, 4])
        say(f"         mp40 open {mo4:.6f} s")
        summary.append(dict(item="tau_open_mp40[0,4]", value=mo4,
                            recorded=REC_L04_OPEN,
                            rel_diff=abs(mo4 / REC_L04_OPEN - 1)))
    summary.append(dict(item="tau_open[0,4]", value=r04["tau_open_s"],
                        recorded=REC_L04_OPEN,
                        rel_diff=abs(r04["tau_open_s"] / REC_L04_OPEN - 1)))
    summary.append(dict(item="tau_open_numpy_eig[0,4]",
                        value=-1 / r04["lambda0_open_eig"], recorded=REC_L04_OPEN,
                        rel_diff=abs(-1 / r04["lambda0_open_eig"] / REC_L04_OPEN - 1)))
    r00 = by[(0, 0)]
    ratio = -r00["lambda0_open"] / r00["K_ion_ground_ne"]
    say(f"  derivation_07 cross-check at [0,0]: |lambda_0 open| / (K_ion(1S) ne) "
        f"= {ratio:.4f}  (recorded {REC_LAMBDA_RATIO_00}); K_ion(1S) ne = "
        f"-colsum(L)[1S] = {r00['K_ion_ground_ne']:.4e} s^-1")
    summary.append(dict(item="lambda0_open_over_Kion_ne[0,0]", value=ratio,
                        recorded=REC_LAMBDA_RATIO_00,
                        rel_diff=abs(ratio / REC_LAMBDA_RATIO_00 - 1)))

    summary_extra: list[dict] = []
    say("\n  GRID-WIDE closure factor tau_open/tau_closed:")
    say(f"    min {F.min():.4f}  median {np.median(F):.4f}  max {F.max():.2f} "
        f"(recorded 1.00 / 1.00 / 41.4)")
    q = int(np.argmax(F))
    say(f"    maximum at [{rows[q]['i']},{rows[q]['j']}]; "
        f"{int((F > 1.01).sum())} of 400 points have factor > 1.01, "
        f"{int((F > 2).sum())} have factor > 2, {int((F > 10).sum())} > 10")
    tau_o_all = np.array([r["tau_open_s"] for r in rows])
    big = F > 2
    say(f"    where factor > 2, open tau_slow is at least "
        f"{tau_o_all[big].min():.3e} s = {tau_o_all[big].min()/TAU_D:.0f} tau_d")
    for mult in (1, 3, 10):
        m_ = tau_o_all < mult * TAU_D
        qm = int(np.argmax(np.where(m_, F, -1)))
        say(f"    where open tau_slow < {mult:>2} tau_d ({int(m_.sum()):>3} points), "
            f"the largest factor is {F[m_].max():.4f} at [{rows[qm]['i']},"
            f"{rows[qm]['j']}]")
        summary_extra.append(dict(item=f"max_factor_where_tau_lt_{mult}tau_d",
                                  value=float(F[m_].max()), recorded=np.nan,
                                  rel_diff=np.nan))
    for k_, v_ in (("factor_min", F.min()), ("factor_median", np.median(F)),
                   ("factor_max", F.max())):
        summary.append(dict(item=k_, value=float(v_),
                            recorded={"factor_min": 1.00, "factor_median": 1.00,
                                      "factor_max": 41.4}[k_], rel_diff=np.nan))

    summary += summary_extra

    # ---- 4. the census ----------------------------------------------------
    say("\n4. THE ELM CENSUS WITH CLOSED tau_slow (divertor_map.csv, 100 us)")
    hdr, data = [], []
    with dp.open() as fh:
        for line in fh:
            (hdr if line.startswith("#") else data).append(line.rstrip("\n"))
    meta = "\n".join(hdr)
    if shas["L_grid.npy"] not in meta or shas["S_grid.npy"] not in meta:
        raise RuntimeError("divertor_map.csv was generated from a different "
                           "L_grid/S_grid than the ones on disk; regenerate it "
                           "with verify_divertor_map.py before using it here")
    fl = [h for h in hdr if "fractional step" in h][0]
    frac = float(fl.split("fractional step")[1].split(",")[0])
    thr = float(fl.split("threshold")[1])
    say(f"  CSV header: fractional step {frac}, threshold {thr}; "
        f"L_grid/S_grid sha match the files on disk")
    rd = list(csv.DictReader(data))
    say(f"  {len(rd)} (point, direction) rows")
    tau_c_grid = np.full((nT, nN), np.nan)
    for r in rows:
        tau_c_grid[r["i"], r["j"]] = r["tau_closed_s"]
    cen = []
    for r in rd:
        i, j = int(r["i"]), int(r["j"])
        sgn = +1 if r["direction"] == "heat" else -1
        k = int(np.argmin(np.abs(te - te[i] * (1 + sgn * frac))))
        if k == i:
            raise RuntimeError(f"step from row {i} did not move a grid index")
        tq_csv = float(r["tau_QSS"])
        tq_here = -1.0 / by[(k, j)]["lambda0_open_eig"]     # CSV used numpy eig
        if abs(tq_here / tq_csv - 1) > 1e-9:
            raise RuntimeError(
                f"CSV tau_QSS {tq_csv:.6e} at ({r['direction']},{i},{j}) does "
                f"not match eig(L[{k},{j}]) = {tq_here:.6e}; the post-step "
                f"index mapping is wrong or the CSV is stale")
        ep = float(r["eps_plateau"])
        lo_open = lower_bound(ep, tq_csv, TAU_D)
        if abs(lo_open / float(r["lo_ELM_crash"]) - 1) > 1e-9:
            raise RuntimeError("cannot reproduce the CSV's own lo_ELM_crash")
        cen.append(dict(direction=r["direction"], i=i, j=j, k_post=k,
                        Te_pre=float(r["Te"]), ne=float(r["ne"]),
                        window_ok=r["window_ok"] == "True",
                        eps_plateau=ep, tau_open=tq_csv,
                        tau_open_secular=by[(k, j)]["tau_open_s"],
                        tau_closed=tau_c_grid[k, j],
                        factor=tq_csv / tau_c_grid[k, j],
                        lo_open=lo_open,
                        lo_open_secular=lower_bound(
                            ep, by[(k, j)]["tau_open_s"], TAU_D),
                        lo_closed=lower_bound(ep, tau_c_grid[k, j], TAU_D)))
    W = np.array([c["window_ok"] for c in cen])
    LO = np.array([c["lo_open"] for c in cen])
    LC = np.array([c["lo_closed"] for c in cen])
    TE = np.array([c["Te_pre"] for c in cen])
    n_win = int(W.sum())
    n_o, n_c = int((W & (LO > thr)).sum()), int((W & (LC > thr)).sum())
    wo, wc = LO[W].max(), LC[W].max()
    m2 = W & (TE >= 2.0)
    n_o2, n_c2 = int((m2 & (LO > thr)).sum()), int((m2 & (LC > thr)).sum())
    say(f"  window_ok rows: {n_win}  (recorded {REC_CENSUS['n_win']})")
    say(f"  breakdown count, lower bound > {thr:.0%} at 100 us: open {n_o} -> "
        f"closed {n_c}  (recorded {REC_CENSUS['n_open']} -> "
        f"{REC_CENSUS['n_closed']})")
    say(f"  worst case: open {wo:.5f} -> closed {wc:.5f}  (recorded "
        f"{REC_CENSUS['worst_open']} -> {REC_CENSUS['worst_closed']}); "
        f"ratio {wo/wc:.5f}, one part in {1/(wo/wc-1):.0f}")
    LS = np.array([c["lo_open_secular"] for c in cen])
    say(f"  same census with the secular open tau_slow instead of the CSV's "
        f"numpy-eig value: {int((W & (LS > thr)).sum())}, worst {LS[W].max():.5f}")
    say(f"  Te_pre >= 2 eV: open {n_o2} -> closed {n_c2}  (recorded "
        f"{REC_CENSUS['n_open_te2']} -> {REC_CENSUS['n_closed_te2']}); "
        f"{int(m2.sum())} window_ok rows there")
    say("  window_ok is the open operator's (30 tau_relax < tau_slow/30), as in")
    say("  verify_divertor_map.py; the closure leaves tau_relax alone and")
    say("  shortens tau_slow, so re-deciding it with closed tau_slow could only")
    say("  remove rows. Re-deciding it here:")
    # tau_relax from the CSV belongs to L[k,j]; recompute the window with closed tau_slow
    n_win_c = 0
    for c, r in zip(cen, rd):
        tr = float(r["tau_relax"])
        n_win_c += int((30.0 * tr) < (c["tau_closed"] / 30.0))
    say(f"    window_ok rows with closed tau_slow: {n_win_c} (open: {n_win})")
    flipped = [c for c in cen if c["window_ok"] and c["lo_open"] > thr
               and not c["lo_closed"] > thr]
    say(f"  rows that leave the census under closure: {len(flipped)}")
    for c in flipped:
        say(f"    {c['direction']:>4} pre [{c['i']},{c['j']}] post L[{c['k_post']},"
            f"{c['j']}]  Te_pre {c['Te_pre']:.4f} ne {c['ne']:.3e}  "
            f"eps_plateau {c['eps_plateau']:.5f}  tau {c['tau_open']:.4e} -> "
            f"{c['tau_closed']:.4e} (x{c['factor']:.2f})  lower bound "
            f"{c['lo_open']:.5f} -> {c['lo_closed']:.5f}")
    qw = int(np.argmax(np.where(W, LO, -1)))
    say(f"  worst row: {cen[qw]['direction']} pre [{cen[qw]['i']},{cen[qw]['j']}] "
        f"post L[{cen[qw]['k_post']},{cen[qw]['j']}], eps_plateau "
        f"{cen[qw]['eps_plateau']:.6f}, tau {cen[qw]['tau_open']:.5e} -> "
        f"{cen[qw]['tau_closed']:.5e}")
    for k_, v_, rec in (("census_n_window_ok", n_win, REC_CENSUS["n_win"]),
                        ("census_open", n_o, REC_CENSUS["n_open"]),
                        ("census_closed", n_c, REC_CENSUS["n_closed"]),
                        ("worst_open", wo, REC_CENSUS["worst_open"]),
                        ("worst_closed", wc, REC_CENSUS["worst_closed"]),
                        ("census_open_Te_ge_2", n_o2, REC_CENSUS["n_open_te2"]),
                        ("census_closed_Te_ge_2", n_c2, REC_CENSUS["n_closed_te2"]),
                        ("census_window_ok_closed", n_win_c, np.nan)):
        summary.append(dict(item=k_, value=float(v_), recorded=rec,
                            rel_diff=abs(v_ / rec - 1) if rec == rec else np.nan))

    # ---- 5. the falsifier -------------------------------------------------
    say("\n5. THE '2300' FALSIFIER (chapter6.tex 905-912, 958-961)")
    cw = cen[qw]
    ratio_2300 = cw["tau_open"] / TAU_D
    say(f"  tau_slow(open) / tau_d at the worst-case operator L[{cw['k_post']},"
        f"{cw['j']}]: {cw['tau_open']:.5f} s / {TAU_D:.0e} s = {ratio_2300:.0f}"
        f"  (chapter: ~{REC_FALSIFIER:.0f}); measured closure factor there "
        f"{cw['factor']:.2f}")
    lo_at_td = lower_bound(cw["eps_plateau"], TAU_D, TAU_D)
    say(f"  if tau_slow were brought down to tau_d there, the lower bound would "
        f"be eps_plateau*(1-1/e) = {lo_at_td:.4f}, "
        f"{'still above' if lo_at_td > thr else 'below'} the {thr:.0%} threshold")
    tau_star = tau_for_threshold(cw["eps_plateau"], thr, TAU_D)
    f_worst = cw["tau_open"] / tau_star
    say(f"  factor needed to push the worst point itself below threshold: "
        f"tau_slow -> {tau_star:.3e} s, factor {f_worst:.0f} "
        f"({f_worst/ratio_2300:.1f} x the 2300)")
    need = []
    for c in cen:
        if c["window_ok"] and c["lo_open"] > thr:
            ts = tau_for_threshold(c["eps_plateau"], thr, TAU_D)
            need.append((c["tau_open"] / ts, c))
    need.sort(key=lambda t: t[0])
    f_min, c_min = need[0]
    say(f"  smallest uniform factor on tau_slow that changes the census count "
        f"by one: {f_min:.3f}  (row {c_min['direction']} pre [{c_min['i']},"
        f"{c_min['j']}], lower bound {c_min['lo_open']:.5f}, measured factor "
        f"there {c_min['factor']:.3f})")
    f_all = need[-1][0]
    say(f"  uniform factor that would empty the census: {f_all:.0f}")
    n_meas = sum(1 for f_, c in need if c["factor"] >= f_)
    say(f"  rows whose MEASURED closure factor reaches the factor they would "
        f"need: {n_meas}  (these are the rows that left the census)")
    say(f"  point-by-point, needed/measured factor at the {len(need)} open "
        f"breakdown rows: median needed {np.median([f_ for f_, _ in need]):.1f}, "
        f"median measured {np.median([c['factor'] for _, c in need]):.3f}")
    for Fu in (REC_FALSIFIER, f_min, f_worst):
        LU = np.array([lower_bound(c["eps_plateau"], c["tau_open"] / Fu, TAU_D)
                       for c in cen])
        nU = int((W & (LU > thr)).sum())
        nU2 = int((m2 & (LU > thr)).sum())
        say(f"  census if EVERY tau_slow were divided by {Fu:.4g}: {nU} of "
            f"{n_win}, worst {LU[W].max():.4f}; Te >= 2 eV: {nU2}")
        summary.append(dict(item=f"census_if_uniform_factor_{Fu:.4g}",
                            value=float(nU), recorded=np.nan, rel_diff=np.nan))
    say("  Reading: the chapter's 2300 is tau_slow/tau_d at L[1,4]. Dividing")
    say("  tau_slow by 2300 there lands the slow clock on the ELM duration but")
    say("  does not by itself take the worst point below the 10% threshold;")
    say(f"  that needs {f_worst:.0f}. The census count first moves at a factor")
    say(f"  {f_min:.2f}, and the measured factors reach that only at "
        f"{n_meas} rows.")
    summary += [dict(item="tau_open_over_tau_d_at_worst_operator",
                     value=ratio_2300, recorded=REC_FALSIFIER,
                     rel_diff=abs(ratio_2300 / REC_FALSIFIER - 1)),
                dict(item="lower_bound_worst_if_tau_equals_tau_d",
                     value=lo_at_td, recorded=np.nan, rel_diff=np.nan),
                dict(item="factor_to_take_worst_below_threshold",
                     value=f_worst, recorded=np.nan, rel_diff=np.nan),
                dict(item="min_uniform_factor_changing_census",
                     value=f_min, recorded=np.nan, rel_diff=np.nan),
                dict(item="uniform_factor_emptying_census",
                     value=f_all, recorded=np.nan, rel_diff=np.nan)]

    # ---- 6. write -----------------------------------------------------------
    prov = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
            f"# interpreter {sys.executable}  numpy {np.__version__}"]
    prov += [f"# sha256 {k} {v}" for k, v in shas.items()]
    prov += [f"# tau_d {TAU_D}  threshold {thr}  fractional step {frac}"]
    with (out / "ion_closure.csv").open("w", newline="") as fh:
        fh.write("\n".join(prov) + "\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (out / "ion_closure_census.csv").open("w", newline="") as fh:
        fh.write("\n".join(prov) + "\n")
        w = csv.DictWriter(fh, fieldnames=list(cen[0].keys()))
        w.writeheader()
        w.writerows(cen)
    with (out / "ion_closure_summary.csv").open("w", newline="") as fh:
        fh.write("\n".join(prov) + "\n")
        w = csv.DictWriter(fh, fieldnames=["item", "value", "recorded",
                                           "rel_diff"])
        w.writeheader()
        w.writerows(summary)
    say(f"\nwrote {out/'ion_closure.csv'} ({len(rows)} rows)")
    say(f"wrote {out/'ion_closure_census.csv'} ({len(cen)} rows)")
    say(f"wrote {out/'ion_closure_summary.csv'} ({len(summary)} rows)")
    (out / "ion_closure.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'ion_closure.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
