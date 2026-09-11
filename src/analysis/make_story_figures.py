#!/usr/bin/env python3
"""
make_story_figures.py
=====================
The six figures the thesis argument needs and does not have.  Chapters 1, 2
and 6 currently carry no figure at all, and the single picture the whole
argument turns on -- the reversal -- has never been drawn.

  fig5_5_reversal.pdf          the reversal: the closure that is doubted holds,
                               the assumption that is not examined fails
  fig5_6_trajectory.pdf        R(t) after a temperature step: rise, plateau,
                               decay, on one logarithmic time axis
  fig5_7_structural_maps.pdf   |Sbar|, |G| and their product over (Te, ne)
  fig6_1_scope.pdf             two independent boundaries on the scope
  fig1_1_diagnostic_chain.pdf  schematic: spectrum -> ratio -> table -> (Te,ne)
  fig2_1_state_space.pdf       schematic: the 43-state space and its processes
  story_captions.tex           \\newcommand captions, every number injected
                               from the arrays actually plotted

WHAT IS PLOTTED, AND FROM WHERE
-------------------------------
Everything numerical is recomputed here from the canonical matrix

    data/processed/cr_matrix/L_grid.npy   (50 Te x 8 ne x 43 x 43, s^-1)
    data/processed/cr_matrix/S_grid.npy   (50 x 8 x 43, recombination source)
    data/processed/collisions/K_exc_full/state_index.csv   (state ordering)
    data/processed/lmix/K_lmix.npy        (proton l-mixing rate coefficients)

loaded through src/validation/cr_context.py.  Grids, state ordering and shell
membership are never redeclared here.  The only file read for cross-checking
rather than for plotting is

    validation/reservoir_gain/reservoir_gain.csv   (verify_reservoir_gain.py)

and every one of its 2288 rows is compared against the recomputation; a
mismatch raises.  Figure 3 is drawn from the recomputed arrays, not from the
CSV, so a figure cannot silently disagree with the matrix it describes.

THE TWO ERRORS OF FIGURE 1, DEFINED
-----------------------------------
Both are measured on the observable, the shell ratio R = sum_{n=3} n_p /
sum_{n=4} n_p, along the same trajectory, and both are reported as the maximum
over the timescale-separated plateau window 30*tau_relax < t < tau_QSS/30:

  QSS closure residual   max_t | R(t) / R^QSS(n_g(t)) - 1 |
      where R^QSS(n_g) is built from  n_E = -L_EE^-1 ( L_Eg n_g + S_E ),
      i.e. the quasi-steady-state solution evaluated at the INSTANTANEOUS
      ground density, not at a table value.

  CRE distance           max_t | R(t) / R^CRE(Te_new, ne) - 1 |
      where R^CRE is built from the equilibrium-ionisation-balance solve
      n = -L^-1 S at the post-step conditions.  That is the state a two-
      parameter (Te, ne) lookup table returns, and the assumption it hides.

findings_09 section 1 records 8.66e-6 and 6.34e-2 at the benchmark and
6.73e-9 and 3.869e-1 at [0,4].  All four are reproduced here to better than
0.2 percent and the script raises if they are not.  Note for the record: the
prose of findings_09 section 1 describes a max-over-states metric on
r_p = n_p/n_1s; its tabulated values match the shell-ratio form above, not the
max-over-states form, which at the benchmark gives 2.23e-5 and 2.36e-1.  The
shell-ratio form is used here because it is the observable Chapter 5 reports.

WHICH OPERATOR, ALWAYS
----------------------
The rendered symbol for the slow clock is tau_slow, matching thesis_main.tex.
The Python attribute and the divertor_map.csv column stay tau_QSS: those are
the pipeline's own names and renaming them would break every reader of the file.
tau_QSS, tau_relax and M are properties of an operator, not of a grid point,
and this project has three live values of M at the benchmark for that reason
(9982 on L[23,5], 8243 on the post-step L[24,5], 4856 on a +0.6 eV step).
Every timescale printed or plotted here is labelled with the operator it
belongs to.  The trajectory figure evolves under the POST-STEP operator,
which is the physically correct one and the one whose M is 8243.

WHAT THE WINDOW TEST IS
-----------------------
window_ok is M > win_lo*win_hi = 900, the argparse defaults of
verify_plateau_gridmap.py and verify_reservoir_gain.py.  It is an IMPOSED cut,
not a measured floor; cells failing it are drawn, hatched, never dropped.

COLOUR
------
The Chapter 3 and Chapter 5 accent set
    #1f4e79 #c0392b #2e7d32 #8e44ad #e67e22
was put through the dataviz palette validator and FAILS colour-vision
separation: the green/red pair #2e7d32 vs #c0392b separates by only dE 4.2
under deuteranopia, below the dE 6 floor, so no amount of secondary encoding
makes it legal.  #1f4e79 also fails the lightness band and the chroma floor.
That is a defect in the existing Chapter 3 and Chapter 5 figures and it is
reported, not repaired here: those scripts are untouched.

These figures therefore use the Okabe-Ito colour-blind-safe set, whose blue and
vermillion sit close enough to the house accents that the document still reads
as one.  Validator result, light surface, all pairs:
    3 slots  #0072B2 #D55E00 #009E73  -> every check PASS (worst CVD dE 11.0)
    5 slots  + #CC79A7 #E69F00        -> CVD dE 7.6, the 6-8 band, legal only
                                         with secondary encoding
Every data figure here uses at most the passing 3.  The two schematics use all
5, and in a schematic every colour carries a direct text label beside it, which
is the documented relief for both the 6-8 band and the contrast warning.
Sequential maps use viridis and magma, both monotone in luminance, so every
figure survives greyscale.

DEFECTS OF make_ch3_figures.py NOT REPRODUCED HERE
--------------------------------------------------
1. Chapter 3 pins the benchmark as `ib, jb = 23, 5` behind an `assert`, which
   `python -O` deletes.  Here the indices are derived by argmin over the loaded
   grids and a mismatch RAISES.
2. Chapter 3 sets a gate threshold below its own measured value.  No threshold
   here is chosen after seeing the number it gates.

Report only: writes figures/fig5_5_*, fig5_6_*, fig5_7_*, fig6_1_*, fig1_1_*,
fig2_1_* and figures/story_captions.tex.  Reads data/ and validation/; writes
to neither.  Refuses to overwrite an existing file whose bytes differ unless
--force is passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext, find_repo_root          # noqa: E402
from escape_factor import lyman_alpha_sigma0              # noqa: E402

# ---- thesis figure style (identical to make_ch3/ch5_figures.py) ------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito, in fixed order.  Never cycled, never reassigned by rank.
C_BLUE, C_VERM, C_GREEN, C_PURPLE, C_ORANGE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
C_INK = "#1a1a1a"

# The plateau window, the argparse defaults of verify_plateau_gridmap.py and
# verify_reservoir_gain.py.  Not free: the recomputed window_ok is compared
# element by element against reservoir_gain.csv, so a wrong constant raises.
WIN_LO, WIN_HI = 30.0, 30.0

# The benchmark is specified in CLAUDE.md as a PHYSICAL condition.  Its indices
# are derived from the loaded grids by argmin below and checked, never assumed.
BENCH_TE_EV, BENCH_NE_CM3 = 2.947, 1.389e14
BENCH_IJ_EXPECTED = (23, 5)

# CLAUDE.md's recorded reference values for the UNSTEPPED operator L[23,5].
REC_TAU_QSS, REC_TAU_RELAX, REC_M = 2.273e-5, 2.277e-9, 9982.0

# findings_09 section 1, the two errors at two grid points.
REC_CLOSURE = {"bench": 8.66e-6, "worst": 6.73e-9}
REC_CRE = {"bench": 6.34e-2, "worst": 3.869e-1}

# ADDENDUM D.1, quasi-neutrality census over the 392 one-step operators.
REC_QN_OVER_10, REC_QN_OVER_100, REC_QN_TOTAL = 68, 36, 392

# CHANGE_REPORT.md section 4.1: the corrected Lyman-alpha line-centre cross
# section at 1 eV, and the sqrt(2)-inflated value that must not propagate.
REC_SIGMA0_GOOD, REC_SIGMA0_BAD = 5.478e-14, 7.7469e-14

# CODATA 2018 hc, used only to turn the state index's own ionisation energies
# into the two Balmer wavelengths named on the Chapter 1 schematic.
HC_EV_NM = 1239.841984


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RuntimeError(
            f"missing {what}: {path} -- this figure cannot be drawn without "
            f"it, and no stand-in is acceptable")
    return path


def check_recorded(name: str, got: float, want: float, rtol: float) -> str:
    """Compare against a value recorded elsewhere in the project and RAISE on
    disagreement.  Neither value is assumed right; the disagreement is the
    thing being reported."""
    rel = abs(got - want) / abs(want)
    if rel > rtol:
        raise RuntimeError(
            f"{name}: recomputed {got:.6g} disagrees with the recorded "
            f"{want:.6g} by {rel*100:.3g} percent (tolerance {rtol*100:.3g} "
            f"percent). One of the two is wrong and nothing may be plotted "
            f"until it is known which.")
    return f"  {name:<46s} {got:< 14.6g} vs recorded {want:< 12.6g}  " \
           f"({rel*100:.3g}% apart)"


def log_edges(v: np.ndarray) -> np.ndarray:
    """Cell edges for a geometrically spaced grid, in the same units as v."""
    lv = np.log(v)
    mid = 0.5 * (lv[1:] + lv[:-1])
    return np.exp(np.concatenate([[2 * lv[0] - mid[0]], mid,
                                  [2 * lv[-1] - mid[-1]]]))


def sci(x: float, sig: int = 2) -> str:
    """'5.2e+13' -> '5.2\\times10^{13}'.  Valid inside $...$ in BOTH matplotlib
    mathtext and LaTeX, so a figure and its caption cannot disagree about how a
    density is written."""
    s = f"{x:.{sig}g}"
    if "e" not in s:
        return s
    m, e = s.split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def read_table(path: Path, numeric: list, text: list, boolean: list) -> dict:
    """Fails on a missing column, a non-numeric entry, or a non-finite value.
    Never coerces, never fills."""
    raw = path.read_text().splitlines()
    body = [ln for ln in raw if not ln.lstrip().startswith("#")]
    if not body:
        raise RuntimeError(f"{path} contains no data rows")
    rows = list(csv.DictReader(body))
    if not rows:
        raise RuntimeError(f"{path} has a header but no data rows")
    have = set(rows[0].keys())
    missing = [c for c in numeric + text + boolean if c not in have]
    if missing:
        raise RuntimeError(f"{path} is missing column(s) {missing}; columns "
                           f"present: {sorted(have)}")
    out: dict = {}
    for c in text:
        out[c] = np.array([r[c] for r in rows], dtype=object)
    for c in boolean:
        vals = [r[c].strip() for r in rows]
        bad = sorted({v for v in vals if v not in ("True", "False")})
        if bad:
            raise RuntimeError(f"{path} column {c!r} is not boolean: {bad}")
        out[c] = np.array([v == "True" for v in vals])
    for c in numeric:
        try:
            arr = np.array([float(r[c]) for r in rows])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{path} column {c!r} is not numeric: {exc}")
        if not np.all(np.isfinite(arr)):
            k = int(np.argmax(~np.isfinite(arr)))
            raise RuntimeError(f"{path} column {c!r} has a non-finite value at "
                               f"data row {k}")
        out[c] = arr
    out["_n"] = len(rows)
    return out


# ===========================================================================
# The physics.  One class so that every figure draws on the SAME solves, and
# a quantity cannot mean one thing in Figure 1 and another in Figure 3.
# ===========================================================================
class Step:
    """One (grid point, one-index heating step) pair, fully solved.

    The algebra is verify_reservoir_gain.py lines 130-165 re-executed, not
    re-derived: the same two-channel split, the same window test, the same
    sign conventions.  `i` is the pre-step temperature index, `k = i+1` the
    post-step index, `j` the density index.  The post-step operator L[k,j] is
    the one that governs the relaxation and the one whose timescales are
    reported.
    """

    def __init__(self, ctx, L, S, i, j, k, E, posE, N3, N4, g):
        self.i, self.j, self.k = i, j, k
        self.Te_old, self.Te_new, self.ne = ctx.te_grid[i], ctx.te_grid[k], ctx.ne_grid[j]
        self.dlnTe = float(np.log(self.Te_new / self.Te_old))
        A = L[k, j]
        self.A = A
        self.E, self.posE, self.N3, self.N4, self.g = E, posE, N3, N4, g

        lam = np.sort(np.linalg.eigvals(A).real)[::-1]
        neg = lam[lam < 0]
        if len(neg) < 2:
            raise RuntimeError(
                f"post-step operator L[{k},{j}] (Te={self.Te_new:.4g} eV, "
                f"ne={self.ne:.4g} cm^-3) has fewer than two negative "
                f"eigenvalues; it is not a decaying operator and no plateau "
                f"exists")
        self.tau_QSS = 1.0 / abs(neg[0])
        self.tau_relax = 1.0 / abs(neg[1])
        self.M = self.tau_QSS / self.tau_relax
        self.window_ok = (WIN_LO * self.tau_relax) < (self.tau_QSS / WIN_HI)

        self.n_old = np.linalg.solve(L[i, j], -S[i, j])
        self.n_new = np.linalg.solve(A, -S[k, j])
        self.LEE = A[np.ix_(E, E)]
        self.LEg = A[np.ix_(E, [g])].ravel()
        self.SE = S[k, j][E]
        self.n0 = np.linalg.solve(self.LEE, -self.SE)                  # rec-fed
        self.n1 = np.linalg.solve(self.LEE, -self.LEg * self.n_old[g])  # ground-fed

        # The two-channel split is an identity.  If it fails, nothing below
        # this line means anything, so it is checked before anything is used.
        x_new = self.n_new[g] / self.n_old[g]
        sup = (np.abs(self.n0 + x_new * self.n1 - self.n_new[E]).max()
               / np.abs(self.n_new[E]).max())
        if sup > 1e-8:
            raise RuntimeError(
                f"two-channel superposition fails at [{i},{j}]: {sup:.3e}. "
                f"The whole construction rests on n_E = n0 + x n1; nothing "
                f"may be plotted")
        self.superposition_err = float(sup)

        a3, a4 = self.n1[[posE[s] for s in N3]].sum(), self.n1[[posE[s] for s in N4]].sum()
        c3, c4 = self.n0[[posE[s] for s in N3]].sum(), self.n0[[posE[s] for s in N4]].sum()
        self.R_pe = float((c3 + a3) / (c4 + a4))         # partial equilibrium
        self.R_cre_new = float(self.n_new[N3].sum() / self.n_new[N4].sum())
        self.R_cre_old = float(self.n_old[N3].sum() / self.n_old[N4].sum())
        self.f3 = float(a3 / (c3 + a3))
        self.f4 = float(a4 / (c4 + a4))

        self.lnx = float(np.log(x_new))
        self.eps = float(abs(self.R_pe / self.R_cre_new - 1.0))
        self.Sbar = float(np.log(self.R_pe / self.R_cre_new) / self.lnx)
        self.G = float(self.lnx / self.dlnTe)
        # eps = |exp(Sbar*G*dlnTe) - 1| is an identity.  Wiring check, not
        # physics; verify_reservoir_gain.py labels it as such and so is this.
        pred = abs(np.expm1(self.Sbar * self.G * self.dlnTe))
        if self.eps > 0 and abs(pred - self.eps) / self.eps > 1e-10:
            raise RuntimeError(
                f"identity eps = |exp(Sbar G dlnTe) - 1| broken at [{i},{j}]: "
                f"{pred:.6e} vs {self.eps:.6e}")

        # nuclei conservation across the step.  n from the CRE solve is
        # normalised to one ion, so a change in the bound population is
        # already expressed as a fraction of n_e = n_ion.  Reported both from
        # the ground state alone and from the whole bound manifold; if the two
        # disagreed the interpretation would be unsafe.
        self.dne_over_ne_ground = float(abs(self.n_new[g] - self.n_old[g]))
        self.dne_over_ne_total = float(abs(self.n_new.sum() - self.n_old.sum()))

        self._eig = None

    # -- propagation ------------------------------------------------------
    def _prepare(self):
        if self._eig is None:
            w, V = np.linalg.eig(self.A)
            c = np.linalg.solve(V, self.n_old - self.n_new)
            self._eig = (w, V, c, float(np.linalg.cond(V)))
        return self._eig

    def n_at(self, t):
        """n(t) for the switched system, by eigen-propagation of the full
        43-state operator: n(t) = n_new + V exp(w t) V^-1 (n_old - n_new).
        Cross-checked against scipy.linalg.expm by `check_propagator`."""
        w, V, c, _ = self._prepare()
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = (V @ (np.exp(np.outer(w, t)) * c[:, None])).real + self.n_new[:, None]
        if not np.all(np.isfinite(out)):
            raise RuntimeError(
                f"eigen-propagation produced a non-finite population at "
                f"[{self.i},{self.j}]; cond(V) = {self._eig[3]:.3e}")
        return out

    def compare_propagators(self, ts):
        """Per-time comparison of eigen-propagation against scipy.linalg.expm.

        Returns (rel, bound, norms) arrays.  `bound` is the a-priori error
        floor of the LESS accurate of the two methods, derived below; it is a
        prediction made before the comparison is run, not a tolerance chosen
        after seeing the answer.  Reports; does not judge.
        """
        d = self.n_old - self.n_new
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        rel = np.empty(ts.size)
        norms = np.empty(ts.size)
        for q, t in enumerate(ts):
            ref = expm(self.A * float(t)) @ d + self.n_new
            got = self.n_at(t)[:, 0]
            rel[q] = np.abs(got - ref).max() / np.abs(ref).max()
            norms[q] = np.linalg.norm(self.A * float(t), 1)
        # Scaling and squaring forms exp(At/2^s) then squares s times, with
        # s ~ log2(||A t||_1).  Each squaring can double the accumulated
        # relative error, so the error after squaring grows like
        # 2^s * eps = ||A t||_1 * eps.  That is the floor below which the two
        # methods cannot be expected to agree, whichever is right.
        bound = np.finfo(float).eps * norms
        return rel, bound, norms

    def check_propagator(self, ts, cond_max=1e6):
        """Independent check of the eigen-propagation against a dense matrix
        exponential.  L is strongly non-normal here (CLAUDE.md records a
        numerical abscissa of +1.28e11 s^-1 against a spectral abscissa of
        -4.40e4 s^-1), so this is not a formality.

        WHICH METHOD IS THE REFERENCE, AND WHY IT MATTERS
        -------------------------------------------------
        These two methods do not degrade in the same place, and the more
        familiar one is not the more accurate one here.

        scipy.linalg.expm uses scaling and squaring, and its relative error
        after the squaring phase grows like ||A t||_1 * eps_machine.  ||A||_1
        for this operator is 1.9e12 s^-1, so by t = tau_QSS in the cold corner
        ||A t||_1 reaches 4e11 and that floor is 1e-4.  The measured
        disagreement tracks ||A t||_1 monotonically across fourteen decades of
        time, which is the signature of the squaring error and not of anything
        physical.

        Eigen-propagation has no such amplification: its accuracy is governed
        by the conditioning of the eigenvector matrix, and cond(V) is 5.6 at
        that same point.  Eigen-propagation is therefore the accurate method
        at long times and expm is the degrading one.

        The test applied here is consequently NOT a fixed tolerance, which
        would either be vacuous at short times or unmeetable at long ones for
        a reason that has nothing to do with this trajectory.  It is that the
        disagreement must stay below ||A t||_1 * eps_machine, the a-priori
        error floor of the weaker method.  A genuine error in the eigen-
        propagation would exceed that floor; in practice the measured
        disagreement sits one to three orders BELOW it, and the margin is
        printed so the test can be seen to have teeth.

        `cond_max` guards the assumption that licenses all of the above.  A
        large cond(V) would remove the basis for preferring eigen-propagation,
        and then nothing here could be trusted; that raises.
        """
        self._prepare()
        cond_V = self._eig[3]
        if cond_V > cond_max:
            raise RuntimeError(
                f"the eigenvector matrix at [{self.i},{self.j}] is "
                f"ill-conditioned, cond(V) = {cond_V:.3e} > {cond_max:.1e}. "
                f"Eigen-propagation cannot be preferred over expm here and "
                f"the trajectory has no trustworthy reference")
        rel, bound, norms = self.compare_propagators(ts)
        over = rel > bound
        if np.any(over):
            q = int(np.argmax(rel / np.maximum(bound, 1e-300)))
            raise RuntimeError(
                f"eigen-propagation disagrees with scipy.linalg.expm at "
                f"[{self.i},{self.j}] by {rel[q]:.3e} relative, ABOVE the "
                f"{bound[q]:.3e} error floor of scaling-and-squaring at "
                f"||A t||_1 = {norms[q]:.3e} (cond(V) = {cond_V:.3e}). The "
                f"disagreement is larger than the weaker method can explain, "
                f"so it is real and nothing may be plotted")
        # the margin: how far below the floor the agreement actually sits
        return float(rel.max()), float(np.min(bound / np.maximum(rel, 1e-300)))

    def window_times(self, n_t):
        lo, hi = WIN_LO * self.tau_relax, self.tau_QSS / WIN_HI
        if not (lo < hi):
            raise RuntimeError(
                f"empty plateau window at [{self.i},{self.j}]: "
                f"{lo:.3e} s to {hi:.3e} s. This point should have been "
                f"excluded by window_ok = {self.window_ok}")
        return np.geomspace(lo, hi, n_t)

    def R_of(self, n):
        """Shell ratio R = sum_{n=3} / sum_{n=4} of a full state vector or of
        a stack of them."""
        return n[self.N3].sum(axis=0) / n[self.N4].sum(axis=0)

    def R_qss_at(self, ng):
        """The quasi-steady-state shell ratio evaluated at a given ground
        density: n_E = -L_EE^-1 (L_Eg n_g + S_E).  This is the closure whose
        residual Figure 1 measures."""
        ng = np.atleast_1d(np.asarray(ng, dtype=float))
        rhs = -(self.SE[:, None] + np.outer(self.LEg, ng))
        nE = np.linalg.solve(self.LEE, rhs)
        i3 = [self.posE[s] for s in self.N3]
        i4 = [self.posE[s] for s in self.N4]
        return nE[i3].sum(axis=0) / nE[i4].sum(axis=0)

    def two_errors(self, n_t=24):
        """The two quantities of Figure 1, as the maximum over the plateau
        window of the relative deviation of the observable R."""
        ts = self.window_times(n_t)
        n = self.n_at(ts)
        Rt = self.R_of(n)
        if np.any(Rt <= 0):
            raise RuntimeError(f"non-positive shell ratio on the trajectory "
                               f"at [{self.i},{self.j}]; the populations have "
                               f"gone negative and the ratio is meaningless")
        closure = np.abs(Rt / self.R_qss_at(n[self.g]) - 1.0)
        cre = np.abs(Rt / self.R_cre_new - 1.0)
        return ts, Rt, float(closure.max()), float(cre.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--force", action="store_true",
                    help="permit overwriting an existing figure whose bytes "
                         "differ (default: raise)")
    ap.add_argument("--nt", type=int, default=24,
                    help="samples across the plateau window for the Figure 1 "
                         "grid maxima")
    args = ap.parse_args()

    # The repo root comes from THIS FILE's location, not the working
    # directory, so the figures are regenerated from the same pipeline
    # whatever directory the command is run in.
    ctx = CRContext.load(root=find_repo_root(Path(__file__).resolve().parent))
    root = ctx.root
    Lp = require_file(root / "data/processed/cr_matrix/L_grid.npy", "L_grid")
    Sp = require_file(root / "data/processed/cr_matrix/S_grid.npy", "S_grid")
    sip = require_file(Path(ctx.state_index_path), "state index")
    Klp = require_file(root / "data/processed/lmix/K_lmix.npy",
                       "proton l-mixing rate coefficients")
    rgp = require_file(root / "validation/reservoir_gain/reservoir_gain.csv",
                       "reservoir gain table")

    L, Te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    S = np.load(Sp)
    Klmix = np.load(Klp)
    if S.shape != L.shape[:3]:
        raise RuntimeError(f"S_grid {S.shape} incompatible with L_grid "
                           f"{L.shape}: {Sp}")
    if Klmix.shape != (ctx.n_states, ctx.n_states, len(Te)):
        raise RuntimeError(
            f"K_lmix {Klmix.shape} incompatible with the loaded model "
            f"({ctx.n_states} states, {len(Te)} temperatures): {Klp}")

    outdir = root / "figures"
    outdir.mkdir(exist_ok=True)

    h = hashlib.sha256()
    for p in (Lp, Sp, sip):
        h.update(Path(p).read_bytes())
    sha8 = h.hexdigest()[:8]
    sha_L = sha256(Lp)
    stamp = f"CR data {sha8} · {datetime.now():%Y-%m-%d}"

    print("=" * 78)
    print("STORY FIGURES -- provenance")
    print("=" * 78)
    print(f"repo root            {root}")
    for lbl, p in (("L_grid", Lp), ("S_grid", Sp), ("state_index", sip),
                   ("K_lmix", Klp), ("reservoir_gain.csv", rgp)):
        print(f"{lbl:<20s} {p.relative_to(root)}")
        print(f"  sha256             {sha256(p)}")
    print(f"combined SHA-8       {sha8}   (L_grid + S_grid + state_index)")
    print(f"grid                 {len(Te)} Te x {len(ne)} ne = "
          f"{len(Te)*len(ne)} points, {ctx.n_states} states")
    print(f"Te                   {Te[0]:.4g} .. {Te[-1]:.4g} eV, "
          f"ratio {Te[1]/Te[0]:.6f} per index")
    print(f"ne                   {ne[0]:.4g} .. {ne[-1]:.4g} cm^-3, "
          f"ratio {ne[1]/ne[0]:.6f} per index")
    print(f"interpreter          {sys.executable}")
    print(f"numpy                {np.__version__}")

    # ---- benchmark indices: derived, then checked, never assumed ----------
    ib = int(np.argmin(np.abs(np.log(Te / BENCH_TE_EV))))
    jb = int(np.argmin(np.abs(np.log(ne / BENCH_NE_CM3))))
    if (ib, jb) != BENCH_IJ_EXPECTED:
        raise RuntimeError(
            f"benchmark point moved: argmin over the loaded grids puts "
            f"Te={BENCH_TE_EV} eV, ne={BENCH_NE_CM3:.4g} cm^-3 at [{ib},{jb}], "
            f"not {BENCH_IJ_EXPECTED}. The grids in "
            f"{root/'data/processed/cr_matrix'} are not the ones CLAUDE.md's "
            f"reference values were measured on; nothing should be redrawn "
            f"until that is resolved")
    for nm, got, want in (("Te", Te[ib], BENCH_TE_EV), ("ne", ne[jb], BENCH_NE_CM3)):
        if abs(np.log(got / want)) > 5e-4:
            raise RuntimeError(f"benchmark {nm} = {got:.6g} does not match the "
                               f"recorded {want:.6g} at index [{ib},{jb}]")
    print(f"benchmark            [{ib},{jb}]  Te={Te[ib]:.4f} eV  "
          f"ne={ne[jb]:.4e} cm^-3   (derived by argmin, checked "
          f"against CLAUDE.md)")

    # CLAUDE.md's reference values belong to the UNSTEPPED operator L[23,5].
    ev_b = np.linalg.eigvals(L[ib, jb])
    if np.max(ev_b.real) >= 0:
        raise RuntimeError(f"non-decaying mode in L[{ib},{jb}]: max Re(lambda) "
                           f"= {np.max(ev_b.real):.3e}")
    tb = np.sort(1.0 / np.abs(ev_b.real))[::-1]
    print()
    print("guard  CLAUDE.md reference values, UNSTEPPED operator "
          f"L[{ib},{jb}]:")
    print(check_recorded("tau_QSS  = 1/|lambda_0|  [s]", tb[0], REC_TAU_QSS, 5e-3))
    print(check_recorded("tau_relax = 1/|lambda_1| [s]", tb[1], REC_TAU_RELAX, 5e-3))
    print(check_recorded("M = tau_QSS/tau_relax", tb[0] / tb[1], REC_M, 5e-3))

    # ---- state-space bookkeeping, all from the pipeline's own files ------
    g = int(ctx.ground_index)
    E = np.array([s for s in range(ctx.n_states) if s != g], dtype=int)
    posE = {s: q for q, s in enumerate(E)}
    nv = np.asarray(ctx.n_values)
    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise RuntimeError(f"shell membership wrong in {sip}: n=3 -> {N3}, "
                           f"n=4 -> {N4}. The observable R = n3/n4 is not "
                           f"what this script thinks it is")

    # ---- solve every step the reservoir-gain table contains ---------------
    # Both directions and every step size in the table, so that all 2288 rows
    # can be checked and the step-size stability of G can be recomputed here
    # rather than quoted.  `steps` below is the k=1 heating subset, which is
    # what the figures are drawn from.
    rg_steps: dict = {}
    for dlab, sgn in (("heat", +1), ("cool", -1)):
        for kstep in (1, 2, 4):
            for i in range(len(Te)):
                kk = i + sgn * kstep
                if not (0 <= kk < len(Te)):
                    continue
                for j in range(len(ne)):
                    rg_steps[(dlab, kstep, i, j)] = Step(
                        ctx, L, S, i, j, kk, E, posE, N3, N4, g)
    steps = {(i, j): st for (d, kk, i, j), st in rg_steps.items()
             if d == "heat" and kk == 1}
    if len(steps) != REC_QN_TOTAL:
        raise RuntimeError(
            f"{len(steps)} one-index heating steps exist on this grid, but "
            f"ADDENDUM D.1's census was taken over {REC_QN_TOTAL}. The grid "
            f"has changed shape and no recorded count applies")
    dlnTe_all = np.array([s.dlnTe for s in steps.values()])
    if np.ptp(dlnTe_all) / dlnTe_all.mean() > 1e-9:
        raise RuntimeError("the temperature grid is not geometric: one grid "
                           "index is not a constant fractional step, so a "
                           "single step size cannot label these figures")
    dlnTe = float(dlnTe_all.mean())
    frac = float(np.expm1(dlnTe))
    print(f"\none-index step       {len(steps)} heating pairs, "
          f"dlnTe = {dlnTe:.6f} (+{frac*100:.3f}% in Te), measured from the "
          f"loaded grid, not assumed")
    print(f"superposition        worst residual over all pairs "
          f"{max(s.superposition_err for s in steps.values()):.3e}")

    # ---- guard: every row of reservoir_gain.csv must be reproduced --------
    rg = read_table(rgp, numeric=["k", "i", "j", "Te", "ne", "dlnTe", "lnx",
                                  "G", "Sbar", "eps", "tau_QSS", "M"],
                    text=["direction"], boolean=["window_ok"])
    n_checked = 0
    for r in range(rg["_n"]):
        key = (str(rg["direction"][r]), int(rg["k"][r]),
               int(rg["i"][r]), int(rg["j"][r]))
        if key not in rg_steps:
            raise RuntimeError(f"{rgp} row {r} is at {key}, which the "
                               f"recomputation from {Lp} does not produce")
        st = rg_steps[key]
        for col, got in (("Te", st.Te_old), ("ne", st.ne), ("dlnTe", st.dlnTe),
                         ("lnx", st.lnx), ("G", st.G), ("Sbar", st.Sbar),
                         ("eps", st.eps), ("tau_QSS", st.tau_QSS), ("M", st.M)):
            a = rg[col][r]
            if abs(a - got) > 1e-9 * max(abs(got), 1e-300):
                raise RuntimeError(
                    f"{rgp} disagrees with the canonical matrix at {key}, "
                    f"column {col}: file {a:.12e}, recomputed {got:.12e}. The "
                    f"CSV is stale with respect to {Lp} (sha256 {sha_L}) or "
                    f"the algebra has changed")
        if bool(rg["window_ok"][r]) != bool(st.window_ok):
            raise RuntimeError(
                f"{rgp} window_ok at {key} is {rg['window_ok'][r]} but "
                f"WIN_LO*WIN_HI = {WIN_LO*WIN_HI:g} gives {st.window_ok} "
                f"(M = {st.M:.6g})")
        n_checked += 1
    if n_checked != rg["_n"]:
        raise RuntimeError(f"{rgp} has {rg['_n']} rows but only {n_checked} "
                           f"were checked; the guard is not covering the file")
    if len(rg_steps) != rg["_n"]:
        raise RuntimeError(
            f"the recomputation produces {len(rg_steps)} (direction, k, i, j) "
            f"combinations but {rgp} has {rg['_n']} rows. One of the two is "
            f"working on a different grid or a different step set")
    print(f"guard  reservoir_gain.csv  all {n_checked} rows reproduce the "
          f"canonical matrix to 1e-9 relative, window_ok exactly")

    ok = {key: st for key, st in steps.items() if st.window_ok}
    if not ok:
        raise RuntimeError("no grid point has a timescale-separated plateau "
                           "window; there is nothing to plot")
    print(f"analysed set         {len(ok)} of {len(steps)} k=1 heating pairs "
          f"pass window_ok (M > {WIN_LO*WIN_HI:g}). That cut is IMPOSED, not "
          f"measured: the M floor of the analysed set is the cut itself")

    bench = steps[(ib, jb)]
    kw = max(ok, key=lambda q: ok[q].eps)          # grid-worst by eps, derived
    worst = ok[kw]
    print(f"grid-worst by eps    [{kw[0]},{kw[1]}]  Te={worst.Te_old:.4f} eV  "
          f"ne={worst.ne:.4e} cm^-3  eps={worst.eps:.6f}   (derived by argmax "
          f"over the analysed set)")

    # ---- file writing that refuses to clobber -----------------------------
    written = []

    def emit(name: str, render):
        path = outdir / name
        if path.exists():
            tmp = path.with_name(path.name + ".new")
            render(tmp)
            if tmp.read_bytes() == path.read_bytes():
                tmp.unlink()
                written.append((name, "identical, left alone"))
                return
            if not args.force:
                tmp.unlink()
                raise RuntimeError(
                    f"{path} already exists with different content. This "
                    f"script will not overwrite a figure that may already be "
                    f"in the thesis. Inspect it, then either delete it or "
                    f"re-run with --force")
            tmp.replace(path)
            written.append((name, "OVERWRITTEN (--force)"))
            return
        render(path)
        written.append((name, "written"))

    def save(fig, base: str):
        # CreationDate is suppressed so that re-running this script on
        # unchanged inputs produces byte-identical files.  Without that every
        # PDF differs on every run and the no-clobber guard in `emit` fires
        # on its own timestamp instead of on a real change.
        emit(base + ".pdf",
             lambda p: fig.savefig(p, format="pdf",
                                   metadata={"CreationDate": None}))
        emit(base + ".png", lambda p: fig.savefig(p, format="png", dpi=150,
                                                  metadata={"Software": None}))
        plt.close(fig)

    def provenance(fig, y=-0.035):
        fig.text(1.0, y, stamp, ha="right", va="top", fontsize=5, color="0.55")

    Te_edges, ne_edges = log_edges(Te), log_edges(ne)
    tok: dict = {"@SHA8@": sha8, "@SHAL@": sha_L[:16],
                 "@DATE@": f"{datetime.now():%Y-%m-%d}",
                 "@FRAC@": f"{frac*100:.2f}", "@DLNTE@": f"{dlnTe:.4f}",
                 "@NPAIR@": str(len(steps)), "@NOK@": str(len(ok)),
                 "@MWIN@": f"{WIN_LO*WIN_HI:.0f}",
                 "@BENCH_TE@": f"{Te[ib]:.3f}", "@BENCH_NE@": sci(ne[jb]),
                 "@WORST_TE@": f"{worst.Te_old:.2f}",
                 "@WORST_NE@": sci(worst.ne),
                 "@WORST_I@": str(kw[0]), "@WORST_J@": str(kw[1]),
                 "@NSTATES@": str(ctx.n_states)}

    # =======================================================================
    # FIGURE 1 -- fig5_5_reversal
    #   The approximation everyone worries about, against the one nobody
    #   examines, measured on the same observable along the same trajectory.
    # =======================================================================
    print()
    print("=" * 78)
    print("FIG 5.5  THE REVERSAL")
    print("=" * 78)
    clo = np.full((len(Te), len(ne)), np.nan)
    cre = np.full((len(Te), len(ne)), np.nan)
    prop_worst, prop_margin = 0.0, np.inf
    for key, st in ok.items():
        _, _, c1, c2 = st.two_errors(args.nt)
        clo[key], cre[key] = c1, c2
        # independent check of the propagator, at the window midpoint
        tm = float(np.sqrt(WIN_LO * st.tau_relax * st.tau_QSS / WIN_HI))
        w_, m_ = st.check_propagator([tm])
        prop_worst = max(prop_worst, w_)
        prop_margin = min(prop_margin, m_)
    print(f"propagator check     eigen-propagation vs scipy.linalg.expm at the "
          f"window midpoint of all {len(ok)} points: worst {prop_worst:.3e}, "
          f"never closer than {prop_margin:.0f}x below the "
          f"scaling-and-squaring error floor")

    have = ~np.isnan(clo)
    if not have.any():
        raise RuntimeError("no analysed point produced a plateau error pair")
    if np.nanmin(clo) <= 0 or np.nanmin(cre) <= 0:
        raise RuntimeError("a non-positive error was produced; the log axes "
                           "of this figure would silently drop it")

    print(check_recorded("closure residual, benchmark",
                         clo[ib, jb], REC_CLOSURE["bench"], 5e-3))
    print(check_recorded("closure residual, grid-worst point",
                         clo[kw], REC_CLOSURE["worst"], 5e-3))
    print(check_recorded("CRE distance, benchmark",
                         cre[ib, jb], REC_CRE["bench"], 5e-3))
    print(check_recorded("CRE distance, grid-worst point",
                         cre[kw], REC_CRE["worst"], 5e-3))

    ratio = cre / clo
    lo_o, hi_o = np.log10(np.nanmin(ratio)), np.log10(np.nanmax(ratio))
    print(f"closure residual     {np.nanmin(clo):.3e} .. {np.nanmax(clo):.3e}"
          f"   (median {np.nanmedian(clo):.3e})")
    print(f"CRE distance         {np.nanmin(cre):.3e} .. {np.nanmax(cre):.3e}"
          f"   (median {np.nanmedian(cre):.3e})")
    print(f"gap, CRE/closure     {np.nanmin(ratio):.3e} .. "
          f"{np.nanmax(ratio):.3e}  =  {lo_o:.2f} to {hi_o:.2f} decades")
    print(f"  at the benchmark   {ratio[ib,jb]:.4g}  "
          f"({np.log10(ratio[ib,jb]):.2f} decades)")
    print(f"  at [{kw[0]},{kw[1]}]           {ratio[kw]:.4g}  "
          f"({np.log10(ratio[kw]):.2f} decades)")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                  gridspec_kw=dict(wspace=0.30,
                                                   width_ratios=[1.15, 1.0]))
    for j in range(len(ne)):
        m = have[:, j]
        if not m.any():
            continue
        ax.plot(Te[m], cre[m, j], "-", color=C_VERM, lw=0.9, alpha=0.85,
                zorder=3)
        ax.plot(Te[m], cre[m, j], "o", ms=2.0, color=C_VERM, mew=0, zorder=3)
        ax.plot(Te[m], clo[m, j], "-", color=C_BLUE, lw=0.9, alpha=0.85,
                zorder=2)
        ax.plot(Te[m], clo[m, j], "^", ms=2.2, color=C_BLUE, mew=0, zorder=2)
    for key, mk, nm in ((( ib, jb), "*", "benchmark"),
                        (kw, "s", "grid worst")):
        for arr, col in ((cre, C_VERM), (clo, C_BLUE)):
            ax.plot([Te[key[0]]], [arr[key]], mk, ms=9 if mk == "*" else 5.5,
                    mfc="none", mec=col, mew=1.2, zorder=5)
    # the gap itself, drawn once at the benchmark so it is a length not a claim
    ax.annotate("", xy=(Te[ib], clo[ib, jb]), xytext=(Te[ib], cre[ib, jb]),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color="0.25",
                                mutation_scale=7, shrinkA=3, shrinkB=3))
    ax.text(Te[ib] * 1.13, np.sqrt(clo[ib, jb] * cre[ib, jb]),
            rf"{np.log10(ratio[ib,jb]):.1f} decades" "\n" "at the benchmark",
            fontsize=6.6, color="0.25", ha="left", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_e$ before the step  [eV]")
    ax.set_ylabel(r"relative error in $R = n_3/n_4$")
    ax.set_xticks([1, 2, 3, 5, 7, 10])
    ax.set_xticklabels(["1", "2", "3", "5", "7", "10"])
    ax.set_title(rf"(a) both errors, all {len(ne)} density columns",
                 loc="left", fontsize=8.5)
    # headroom above the upper band so the direct label does not sit on data
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 12.0)
    ax.text(0.03, 0.97, "CRE distance\n(the lookup table's assumption)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            color=C_VERM, linespacing=1.25)
    ax.text(0.97, 0.06, "QSS closure residual\n(the approximation that is doubted)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=C_BLUE, linespacing=1.25)
    ax.legend(handles=[
        Line2D([], [], ls="-", marker="o", ms=3, color=C_VERM,
               label=r"CRE distance $|R/R^{\rm CRE}-1|$"),
        Line2D([], [], ls="-", marker="^", ms=3, color=C_BLUE,
               label=r"QSS closure residual $|R/R^{\rm QSS}(n_g(t))-1|$"),
        Line2D([], [], ls="none", marker="*", ms=8, mfc="none", mec="0.25",
               label="benchmark"),
        Line2D([], [], ls="none", marker="s", ms=5, mfc="none", mec="0.25",
               label=rf"grid worst [{kw[0]},{kw[1]}]")],
        loc="upper left", bbox_to_anchor=(0.0, -0.19), ncol=2, frameon=False,
        fontsize=6.4, handletextpad=0.6, labelspacing=0.4, columnspacing=1.0)

    ax2.set_facecolor("0.93")
    pc = ax2.pcolormesh(Te_edges, ne_edges, np.log10(ratio).T, cmap="magma",
                        shading="flat", rasterized=True)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    cb = fig.colorbar(pc, ax=ax2, pad=0.02)
    cb.set_label(r"$\log_{10}\,$(CRE distance / closure residual)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    n_nowin = 0
    for i in range(len(Te)):
        for j in range(len(ne)):
            if (i, j) in steps and not steps[(i, j)].window_ok:
                n_nowin += 1
                ax2.add_patch(Rectangle(
                    (Te_edges[i], ne_edges[j]),
                    Te_edges[i + 1] - Te_edges[i],
                    ne_edges[j + 1] - ne_edges[j],
                    facecolor="0.35", edgecolor="w", hatch="///", lw=0.3,
                    zorder=2))
    for key, mk in (((ib, jb), "*"), (kw, "s")):
        ax2.plot([Te[key[0]]], [ne[key[1]]], mk, ms=9 if mk == "*" else 5.5,
                 mfc="none", mec="#39d0d8", mew=1.3, zorder=4)
    ax2.set_xlabel(r"$T_e$  [eV]")
    ax2.set_ylabel(r"$n_e$  [cm$^{-3}$]")
    ax2.set_xticks([1, 2, 3, 5, 7, 10])
    ax2.set_xticklabels(["1", "2", "3", "5", "7", "10"])
    ax2.set_title("(b) the gap, over the grid", loc="left", fontsize=8.5)
    provenance(fig, y=-0.20)
    save(fig, "fig5_5_reversal")
    print(f"fig5_5_reversal      {int(have.sum())} analysed cells, "
          f"{n_nowin} hatched (no plateau window)")

    tok.update({
        "@CLO_B@": sci(clo[ib, jb], 3), "@CRE_B@": f"{cre[ib,jb]*100:.2f}",
        "@CLO_W@": sci(clo[kw], 3), "@CRE_W@": f"{cre[kw]*100:.1f}",
        "@CLO_MIN@": sci(np.nanmin(clo), 2), "@CLO_MAX@": sci(np.nanmax(clo), 2),
        "@CRE_MIN@": f"{np.nanmin(cre)*100:.2f}",
        "@CRE_MAX@": f"{np.nanmax(cre)*100:.1f}",
        "@GAP_LO@": f"{lo_o:.1f}", "@GAP_HI@": f"{hi_o:.1f}",
        "@GAP_B@": f"{np.log10(ratio[ib,jb]):.1f}",
        "@GAP_W@": f"{np.log10(ratio[kw]):.1f}",
        "@NT@": str(args.nt), "@PROPCHK@": sci(prop_worst, 1),
        "@NNOWIN@": str(n_nowin),
        "@CLO_B_INV@": sci(1.0 / clo[ib, jb], 2),
        "@CLO_W_INV@": sci(1.0 / clo[kw], 2),
        "@CLO_MAX_INV@": sci(1.0 / np.nanmax(clo), 2),
    })


    # =======================================================================
    # FIGURE 2 -- fig5_6_trajectory
    #   R(t) after a one-index temperature step, at the benchmark and at the
    #   coldest, worst point.  Three phases on one logarithmic time axis:
    #   the rise on tau_relax, the plateau at partial equilibrium, the decay
    #   on tau_QSS.  This is the picture behind "the excited states keep up
    #   but the reservoir does not".
    # =======================================================================
    print()
    print("=" * 78)
    print("FIG 5.6  THE TRAJECTORY")
    print("=" * 78)
    traj_pts = [((ib, jb), "benchmark"), (kw, "coldest grid point")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5),
                             gridspec_kw=dict(wspace=0.30))
    traj_tok = {}
    for panel, ((key, lab), axt) in enumerate(zip(traj_pts, axes)):
        st = steps[key]

        # The partial-equilibrium ratio, recomputed here by ONE linear solve
        # against the same operator, independently of the n0 + n1 split the
        # rest of the script uses.  The two must agree to machine precision;
        # if they do not, the PE line on this figure means nothing.
        n_pe = np.linalg.solve(st.LEE, -(st.SE + st.LEg * st.n_old[g]))
        R_pe_direct = float(n_pe[[posE[s] for s in N3]].sum()
                            / n_pe[[posE[s] for s in N4]].sum())
        d_pe = abs(R_pe_direct / st.R_pe - 1.0)
        if d_pe > 1e-12:
            raise RuntimeError(
                f"the partial-equilibrium ratio at {key} depends on how it is "
                f"computed: one solve gives {R_pe_direct:.12e}, the two-channel "
                f"split gives {st.R_pe:.12e} ({d_pe:.2e} apart). Linearity is "
                f"broken and the plateau line cannot be drawn")

        ts = np.geomspace(st.tau_relax / 300.0, st.tau_QSS * 60.0, 500)
        # Gated inside the plateau window, where the figure's numbers live and
        # where both propagators are accurate.  Reported, not gated, over the
        # whole plotted range: see check_propagator's docstring for why the
        # disagreement out there is scipy's scaling-and-squaring, not this.
        chk, margin = st.check_propagator(st.window_times(12))
        rel_f, bnd_f, nrm_f = st.compare_propagators(np.geomspace(ts[0], ts[-1], 14))
        n_t = st.n_at(ts)
        R_t = st.R_of(n_t)
        if np.any(n_t < 0):
            raise RuntimeError(
                f"the propagated populations go negative at {key}; a shell "
                f"ratio built from them is not a population ratio")

        wlo, whi = WIN_LO * st.tau_relax, st.tau_QSS / WIN_HI
        inwin = (ts >= wlo) & (ts <= whi)
        if not inwin.any():
            raise RuntimeError(f"the plateau window at {key} contains none of "
                               f"the sampled times")
        flat = float(np.ptp(R_t[inwin]) / np.mean(R_t[inwin]))

        axt.axvspan(wlo, whi, color="0.90", zorder=0, lw=0)
        axt.axhline(st.R_cre_old, color=C_BLUE, ls=(0, (5, 2)), lw=1.0, zorder=2)
        axt.axhline(st.R_cre_new, color=C_VERM, ls=(0, (5, 2)), lw=1.0, zorder=2)
        axt.axhline(st.R_pe, color=C_GREEN, ls=(0, (1, 1.4)), lw=1.4, zorder=2)
        axt.plot(ts, R_t, "-", color=C_INK, lw=1.5, zorder=4)
        for tv, tlab in ((st.tau_relax, r"$\tau_{\rm relax}$"),
                         (st.tau_QSS, r"$\tau_{\rm slow}$")):
            axt.axvline(tv, color="0.45", ls="-", lw=0.7, zorder=1)
            axt.text(tv, 0.015, " " + tlab, transform=axt.get_xaxis_transform(),
                     fontsize=7, color="0.35", ha="left", va="bottom",
                     rotation=90)
        axt.set_xscale("log")
        axt.set_xlim(ts[0], ts[-1])
        span = max(abs(st.R_pe - st.R_cre_new), abs(st.R_cre_old - st.R_cre_new))
        axt.set_ylim(min(st.R_cre_new, st.R_cre_old, st.R_pe) - 0.30 * span,
                     max(st.R_cre_new, st.R_cre_old, st.R_pe) + 0.55 * span)
        axt.set_xlabel(r"time after the step  [s]")
        axt.set_ylabel(r"$R(t) = n_3 / n_4$")
        axt.set_title(rf"({'ab'[panel]}) {lab}: $T_e$ "
                      rf"{st.Te_old:.2f}$\rightarrow${st.Te_new:.2f} eV, "
                      rf"$n_e = {sci(st.ne)}$ cm$^{{-3}}$",
                      loc="left", fontsize=8)

        # the three phases, named on the axes rather than only in the caption
        axt.annotate("rise on\n" + r"$\tau_{\rm relax}$", xy=(0.055, 0.62),
                     xycoords="axes fraction", fontsize=6.8, color="0.30",
                     ha="left", va="center", linespacing=1.2)
        axt.annotate("plateau at partial equilibrium",
                     xy=(np.sqrt(wlo * whi), st.R_pe),
                     xytext=(0, 13), textcoords="offset points",
                     fontsize=6.8, color="0.30", ha="center", va="bottom")
        axt.annotate("decay on\n" + r"$\tau_{\rm slow}$", xy=(0.87, 0.74),
                     xycoords="axes fraction", fontsize=6.8, color="0.30",
                     ha="center", va="center", linespacing=1.2)

        print(f"panel ({'ab'[panel]}) {lab}  grid [{key[0]},{key[1]}]  "
              f"Te {st.Te_old:.4f} -> {st.Te_new:.4f} eV, "
              f"ne {st.ne:.4e} cm^-3")
        print(f"   post-step operator L[{st.k},{st.j}]:  "
              f"tau_relax {st.tau_relax:.4e} s   tau_QSS {st.tau_QSS:.4e} s   "
              f"M {st.M:.4g}")
        print(f"   R(CRE, before) {st.R_cre_old:.6f}   R(PE) {st.R_pe:.6f}   "
              f"R(CRE, after) {st.R_cre_new:.6f}   eps {st.eps*100:.3f}%")
        print(f"   PE by one linear solve {R_pe_direct:.10f}, by the "
              f"two-channel split {st.R_pe:.10f}  ({d_pe:.1e} apart)")
        print(f"   plateau flatness over the window "
              f"{flat*100:.4f}% peak-to-peak")
        print(f"   eigen vs expm inside the window: worst {chk:.2e}, "
              f"{margin:.0f}x below the scaling-and-squaring floor")
        print(f"   over the whole plotted range: worst {rel_f.max():.2e} at "
              f"||A t||_1 = {nrm_f[np.argmax(rel_f)]:.2e}, floor there "
              f"{bnd_f[np.argmax(rel_f)]:.2e}; cond(V) = {st._eig[3]:.3g}")
        traj_tok[panel] = dict(st=st, flat=flat, chk=chk)

    axes[0].legend(handles=[
        Line2D([], [], color=C_INK, lw=1.5, label=r"$R(t)$, 43-state solution"),
        Line2D([], [], color=C_GREEN, ls=(0, (1, 1.4)), lw=1.4,
               label=r"$R^{\rm PE}$, partial equilibrium (one linear solve)"),
        Line2D([], [], color=C_BLUE, ls=(0, (5, 2)), lw=1.0,
               label=r"$R^{\rm CRE}$ at $T_e$ before the step"),
        Line2D([], [], color=C_VERM, ls=(0, (5, 2)), lw=1.0,
               label=r"$R^{\rm CRE}$ at $T_e$ after the step"),
        Patch(facecolor="0.90", edgecolor="none",
              label=r"plateau window, $30\tau_{\rm relax}$ to "
                    r"$\tau_{\rm slow}/30$")],
        loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, frameon=False,
        fontsize=6.4, handletextpad=0.7, labelspacing=0.4, columnspacing=1.1)
    provenance(fig, y=-0.235)
    save(fig, "fig5_6_trajectory")

    sb, sc = traj_tok[0]["st"], traj_tok[1]["st"]
    tok.update({
        "@TB_TR@": sci(sb.tau_relax, 3), "@TB_TQ@": sci(sb.tau_QSS, 3),
        "@TB_M@": f"{sb.M:.0f}", "@TB_ROLD@": f"{sb.R_cre_old:.4f}",
        "@TB_RPE@": f"{sb.R_pe:.4f}", "@TB_RNEW@": f"{sb.R_cre_new:.4f}",
        "@TB_EPS@": f"{sb.eps*100:.2f}", "@TB_K@": str(sb.k),
        "@TB_FLAT@": f"{traj_tok[0]['flat']*100:.2f}",
        "@TC_TR@": sci(sc.tau_relax, 3), "@TC_TQ@": sci(sc.tau_QSS, 3),
        "@TC_M@": sci(sc.M, 3), "@TC_ROLD@": f"{sc.R_cre_old:.4f}",
        "@TC_RPE@": f"{sc.R_pe:.4f}", "@TC_RNEW@": f"{sc.R_cre_new:.4f}",
        "@TC_EPS@": f"{sc.eps*100:.1f}", "@TC_K@": str(sc.k),
        "@TC_TE@": f"{sc.Te_old:.2f}", "@TC_TENEW@": f"{sc.Te_new:.2f}",
        "@TC_NE@": sci(sc.ne), "@TB_TENEW@": f"{sb.Te_new:.2f}",
        "@TRAJCHK@": sci(max(traj_tok[0]["chk"], traj_tok[1]["chk"]), 1),
    })


    # =======================================================================
    # FIGURE 3 -- fig5_7_structural_maps
    #   The two structural coefficients Chapter 5 now leads on, and their
    #   product, which is the invariant it reports.
    # =======================================================================
    print()
    print("=" * 78)
    print("FIG 5.7  THE STRUCTURAL MAPS")
    print("=" * 78)

    nT, nN = len(Te), len(ne)
    Sb = np.full((nT, nN), np.nan)
    Gv = np.full((nT, nN), np.nan)
    win = np.zeros((nT, nN), bool)
    inmap = np.zeros((nT, nN), bool)
    for (i, j), st in steps.items():
        Sb[i, j], Gv[i, j] = abs(st.Sbar), abs(st.G)
        win[i, j], inmap[i, j] = st.window_ok, True
    Pr = Sb * Gv

    # Cross-check the plotted arrays cell by cell against the CSV.  The global
    # guard above already compared every row; this repeats it on exactly the
    # numbers that reach the canvas, because that is the thing that can go
    # wrong between a table and a figure.
    n_cells = 0
    for r in range(rg["_n"]):
        if rg["direction"][r] != "heat" or int(rg["k"][r]) != 1:
            continue
        i, j = int(rg["i"][r]), int(rg["j"][r])
        if not inmap[i, j]:
            raise RuntimeError(f"{rgp} has a k=1 heating row at [{i},{j}] that "
                               f"is not on the plotted map")
        for nm, arr, ref in (("|Sbar|", Sb, abs(rg["Sbar"][r])),
                             ("|G|", Gv, abs(rg["G"][r]))):
            if abs(arr[i, j] - ref) > 1e-9 * max(abs(ref), 1e-300):
                raise RuntimeError(
                    f"the plotted {nm} at [{i},{j}] is {arr[i,j]:.12e} but "
                    f"{rgp} records {ref:.12e}. The figure and the table "
                    f"disagree and the figure must not be written")
        n_cells += 1
    if n_cells != int(inmap.sum()):
        raise RuntimeError(f"{n_cells} CSV cells checked against "
                           f"{int(inmap.sum())} plotted cells; the map is not "
                           f"fully covered by the cross-check")
    print(f"guard  every one of the {n_cells} plotted cells matches "
          f"{rgp.name} to 1e-9 relative")

    okm = inmap & win
    if not okm.any():
        raise RuntimeError("empty selection after the window_ok filter; there "
                           "is no analysed set to map")

    def rng(a, m):
        v = a[m]
        return float(v.min()), float(np.median(v)), float(v.max())

    # Two scopes, stated separately.  A range quoted without its scope is the
    # error this project has already had to correct once.
    all_S = np.array([abs(v) for v in rg["Sbar"]])
    all_G = np.array([abs(v) for v in rg["G"]])
    print(f"scope A: k=1 heating, window_ok  ({int(okm.sum())} cells)")
    for nm, a in (("|Sbar|", Sb), ("|G|", Gv), ("|Sbar*G|", Pr)):
        lo, md, hi = rng(a, okm)
        print(f"   {nm:<9s} min {lo:.5f}  median {md:.5f}  max {hi:.5f}")
    print(f"scope B: all {rg['_n']} rows of {rgp.name} "
          f"(both directions, k = 1, 2, 4)")
    print(f"   |Sbar|    min {all_S.min():.5f}  median "
          f"{np.median(all_S):.5f}  max {all_S.max():.5f}")
    print(f"   |G|       min {all_G.min():.5f}  median "
          f"{np.median(all_G):.5f}  max {all_G.max():.5f}")

    # Is G stable against step size where eps is not?  Recomputed here, over
    # every (direction, point) carrying all three step sizes.
    triples, gsp, esp = 0, [], []
    for dlab in ("heat", "cool"):
        for i in range(nT):
            for j in range(nN):
                ks = [k for k in (1, 2, 4) if (dlab, k, i, j) in rg_steps]
                if len(ks) != 3:
                    continue
                gv = np.array([abs(rg_steps[(dlab, k, i, j)].G) for k in ks])
                ev = np.array([rg_steps[(dlab, k, i, j)].eps for k in ks])
                if gv.min() <= 0 or ev.min() <= 0:
                    raise RuntimeError(f"non-positive |G| or eps at {dlab} "
                                       f"[{i},{j}]; a spread ratio is undefined")
                triples += 1
                gsp.append(gv.max() / gv.min())
                esp.append(ev.max() / ev.min())
    if triples == 0:
        raise RuntimeError("no point carries all three step sizes; the "
                           "step-size stability of G cannot be measured")
    gsp, esp = np.array(gsp), np.array(esp)
    print(f"G stability          {triples} (direction, point) triples carry "
          f"k = 1, 2 and 4")
    print(f"   |G| spread across k    median {np.median(gsp):.4f}   "
          f"max {gsp.max():.4f}   ({(gsp.max()-1)*100:.2f}% at worst)")
    print(f"   eps spread across k    median {np.median(esp):.4f}   "
          f"max {esp.max():.4f}   ({esp.max():.2f}x at worst)")
    print(f"   G is step-size stable to {(gsp.max()-1)*100:.2f}% where eps "
          f"varies by up to {esp.max():.2f}x. That contrast is the reason "
          f"Chapter 5 reports G and not eps")

    fig, axs = plt.subplots(1, 3, figsize=(7.9, 2.95), sharey=True,
                            gridspec_kw=dict(wspace=0.55))
    panels = [
        (Sb, r"$|\bar{S}| = |f_3 - f_4|$", "viridis",
         "(a) sensitivity of the observable"),
        (Gv, r"$|G| = |\Delta\ln u\,/\,\Delta\ln T_e|$", "viridis",
         "(b) reservoir gain"),
        (Pr, r"$|\bar{S}G| = \lim\,\varepsilon\,/\,|\Delta\ln T_e|$",
         "magma", "(c) their product: the reported coefficient"),
    ]
    n_nowin3 = 0
    for q, (arr, cblab, cmap, title) in enumerate(panels):
        axq = axs[q]
        axq.set_facecolor("0.93")
        shown = np.ma.masked_invalid(arr)
        pc = axq.pcolormesh(Te_edges, ne_edges, shown.T, cmap=cmap,
                            shading="flat", rasterized=True)
        axq.set_xscale("log")
        axq.set_yscale("log")
        cb = fig.colorbar(pc, ax=axq, pad=0.03, fraction=0.055)
        cb.set_label(cblab, fontsize=7.5)
        cb.ax.tick_params(labelsize=6.5)
        # cells without a timescale-separated plateau window: hatched, and the
        # value left visible underneath rather than deleted
        for i in range(nT):
            for j in range(nN):
                if inmap[i, j] and not win[i, j]:
                    if q == 0:
                        n_nowin3 += 1
                    axq.add_patch(Rectangle(
                        (Te_edges[i], ne_edges[j]),
                        Te_edges[i + 1] - Te_edges[i],
                        ne_edges[j + 1] - ne_edges[j],
                        facecolor="none", edgecolor="w", hatch="///", lw=0.0,
                        zorder=3))
        axq.plot([Te[ib]], [ne[jb]], "*", ms=8, mfc="none", mec="#39d0d8",
                 mew=1.2, zorder=5)
        axq.set_xlabel(r"$T_e$  [eV]")
        if q == 0:
            axq.set_ylabel(r"$n_e$  [cm$^{-3}$]")
        else:
            # all three panels are the same (Te, ne) plane; repeating the tick
            # labels only crowds them into the neighbouring colour bar
            axq.tick_params(labelleft=False)
        axq.set_xticks([1, 2, 3, 5, 7, 10])
        axq.set_xticklabels(["1", "2", "3", "5", "7", "10"])
        axq.set_title(title, loc="left", fontsize=7.6)
    fig.legend(handles=[
        Line2D([], [], ls="none", marker="*", ms=8, mfc="none", mec="0.15",
               label=rf"benchmark: $|\bar{{S}}| = {Sb[ib,jb]:.3f}$, "
                     rf"$|G| = {Gv[ib,jb]:.2f}$, product ${Pr[ib,jb]:.2f}$"),
        Patch(facecolor="0.80", edgecolor="0.15", hatch="///",
              label=rf"no plateau window ($M \leq {WIN_LO*WIN_HI:.0f}$), "
                    rf"{n_nowin3} of {int(inmap.sum())} cells; value still shown")],
        loc="lower left", bbox_to_anchor=(0.09, -0.12), ncol=2, frameon=False,
        fontsize=6.5, handletextpad=0.6, columnspacing=1.4)
    provenance(fig, y=-0.135)
    save(fig, "fig5_7_structural_maps")
    print(f"fig5_7_structural_maps  benchmark |Sbar| {Sb[ib,jb]:.5f}, "
          f"|G| {Gv[ib,jb]:.5f}, product {Pr[ib,jb]:.5f}")

    sA = rng(Sb, okm); gA = rng(Gv, okm); pA = rng(Pr, okm)
    tok.update({
        "@SB_LO@": f"{sA[0]:.3f}", "@SB_HI@": f"{sA[2]:.3f}",
        "@G_LO@": f"{gA[0]:.2f}", "@G_HI@": f"{gA[2]:.2f}",
        "@PR_LO@": f"{pA[0]:.2f}", "@PR_HI@": f"{pA[2]:.2f}",
        "@SB_B@": f"{Sb[ib,jb]:.3f}", "@G_B@": f"{Gv[ib,jb]:.2f}",
        "@PR_B@": f"{Pr[ib,jb]:.2f}",
        "@SB_ALL_LO@": f"{all_S.min():.4f}", "@SB_ALL_HI@": f"{all_S.max():.4f}",
        "@G_ALL_LO@": f"{all_G.min():.2f}", "@G_ALL_HI@": f"{all_G.max():.2f}",
        "@NROWS@": str(rg["_n"]), "@NCELL@": str(int(inmap.sum())),
        "@NNOWIN3@": str(n_nowin3), "@NOKM@": str(int(okm.sum())),
        "@GSTAB@": f"{(gsp.max()-1)*100:.2f}",
        "@GSTABMED@": f"{np.median(gsp):.4f}",
        "@EPSSPREAD@": f"{esp.max():.2f}", "@NTRIP@": str(triples),
    })


    # =======================================================================
    # FIGURE 4 -- fig6_1_scope
    #   Two boundaries on the same plane, derived from unrelated physics:
    #   particle conservation, and radiation transport.
    # =======================================================================
    print()
    print("=" * 78)
    print("FIG 6.1  THE SCOPE BOUNDARY")
    print("=" * 78)

    # ---- boundary 1: quasi-neutrality ------------------------------------
    # n from the CRE solve is normalised to one ion, so a change in the bound
    # population is already a fraction of n_e = n_ion.  Computed from the two
    # CRE solves either side of the step: delta n_e = -delta n_g.
    qn = np.full((nT, nN), np.nan)
    qn_tot = np.full((nT, nN), np.nan)
    for (i, j), st in steps.items():
        qn[i, j] = st.dne_over_ne_ground
        qn_tot[i, j] = st.dne_over_ne_total
    fin = np.isfinite(qn)
    if not fin.any():
        raise RuntimeError("no one-step operator produced a quasi-neutrality "
                           "requirement; the boundary cannot be drawn")
    # Ground state alone against the whole bound manifold.  Nuclei
    # conservation gives delta n_e = -delta(all bound states); using the
    # ground state alone is an approximation, and this is where it is tested.
    #
    # The two are NOT gated against each other by a chosen tolerance over a
    # chosen region: both of those would be free parameters picked after
    # seeing the answer, which is the defect this script exists to avoid.
    # What is gated is the conclusion itself, below: the 10 percent boundary
    # must not move and the census must not change when the definition is
    # swapped.  The agreement is reported here as a trend so the reader can
    # see where the approximation is good and where it is not.
    dev_all = np.abs(qn_tot[fin] / qn[fin] - 1.0)
    q_all = int(np.argmax(dev_all))
    ii, jj = np.where(fin)
    print(f"delta n_g vs delta(all bound states), by size of the requirement:")
    for thr in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        m = fin & (qn > thr)
        if not m.any():
            raise RuntimeError(f"no cell has a requirement above {thr:g}; the "
                               f"quasi-neutrality boundary is not on this grid")
        d = float(np.nanmax(np.abs(qn_tot[m] / qn[m] - 1.0)))
        print(f"   over the {int(m.sum()):3d} cells with |dn_e/n_e| > {thr:<6g}"
              f"  worst {d*100:8.4g} percent apart")
    print(f"   the worst case sits at Te = {Te[ii[q_all]]:.3g} eV, "
          f"ne = {ne[jj[q_all]]:.3g} cm^-3, where the requirement is "
          f"{qn[ii[q_all], jj[q_all]]:.2e}: five orders below the threshold, "
          f"so it cannot move a 10 percent boundary")

    n_over_10 = int((qn[fin] > 0.10).sum())
    n_over_100 = int((qn[fin] > 1.00).sum())
    print(check_recorded("one-step operators requiring >10% in n_e",
                         n_over_10, REC_QN_OVER_10, 1e-9))
    print(check_recorded("one-step operators requiring >100% in n_e",
                         n_over_100, REC_QN_OVER_100, 1e-9))
    row1 = qn[0, :]
    warm = fin & (Te[:, None] >= 2.0)
    print(f"required |dn_e/n_e| at Te = {Te[0]:.4g} eV: "
          f"{row1.min():.3g} to {row1.max():.3g} (every density column "
          f"above 10)")
    print(f"required |dn_e/n_e| at Te >= 2 eV: max {qn[warm].max():.3g} "
          f"at Te = {Te[np.where(warm)[0][np.argmax(qn[warm])]]:.4g} eV; "
          f"it first falls below 1e-3 above Te = "
          f"{Te[[i for i in range(nT-1) if np.nanmax(qn[i]) < 1e-3][0]]:.3g} eV")

    # ---- boundary 2: Lyman-alpha optical depth ---------------------------
    # sigma_0 comes from escape_factor.lyman_alpha_sigma0, which builds the
    # Doppler width from sqrt(2kT/m).  The sqrt(2)-inflated value that reached
    # findings_10 ADDENDUM A and chapter6.tex must not be used, and the ratio
    # between the two is checked here so the diagnosis is on the record.
    sig0_1eV = float(lyman_alpha_sigma0(1.0))
    print(check_recorded("Lyman-alpha sigma_0 at 1 eV [cm^2]",
                         sig0_1eV, REC_SIGMA0_GOOD, 5e-3))
    infl = REC_SIGMA0_BAD / sig0_1eV
    print(check_recorded("the superseded sigma_0, as a multiple of it",
                         infl, float(np.sqrt(2.0)), 5e-3))
    print(f"  the {REC_SIGMA0_BAD:.4e} cm^2 in findings_10 ADDENDUM A and "
          f"chapter6.tex is this value times {infl:.5f}, i.e. sqrt(2): the "
          f"Doppler width was built from sqrt(kT/m) rather than sqrt(2kT/m)")

    lym_p = require_file(root / "validation/lyman_trapping/lyman_trapping.csv",
                         "Lyman trapping sweep")
    lym = read_table(lym_p, numeric=["D_cm"], text=["run"], boolean=[])
    D_all = sorted({float(v) for v in lym["D_cm"] if float(v) > 0})
    if not D_all:
        raise RuntimeError(f"{lym_p} records no non-zero slab thickness; the "
                           f"optical-depth boundary has no D to evaluate at")
    print(f"slab thicknesses     D = {D_all} cm, read from "
          f"{lym_p.relative_to(root)}, not chosen here")
    D_mid = D_all[len(D_all) // 2]

    # T_at = Te is verify_lyman_trapping.py's own default (its --t-at flag
    # overrides it); the same convention is used here.
    sig_Te = np.array([float(lyman_alpha_sigma0(t)) for t in Te])
    n1s = np.full((nT, nN), np.nan)
    for i in range(nT):
        for j in range(nN):
            n1s[i, j] = np.linalg.solve(L[i, j], -S[i, j])[g] * ne[j]
    if not np.all(np.isfinite(n1s)) or np.any(n1s <= 0):
        raise RuntimeError("a non-positive or non-finite ground-state density "
                           "came out of the CRE solve; an optical depth built "
                           "from it would be meaningless")
    # line-centre optical depth over the half slab, the same definition
    # escape_factor.escape_factor_slab uses: tau_c = n(1s) sigma_0 D/2
    tau = {D: n1s * sig_Te[:, None] * D / 2.0 for D in D_all}
    for D in D_all:
        print(f"  tau_0 at D = {D:g} cm: {tau[D].min():.3e} to "
              f"{tau[D].max():.3e};  benchmark {tau[D][ib,jb]:.4e}")

    def crossing(field, level):
        """Te at which `field` crosses `level`, per density column, by linear
        interpolation in (log Te, log field).  Raises if a column never
        crosses, because a boundary that does not exist must not be drawn."""
        out = np.full(nN, np.nan)
        for j in range(nN):
            y = np.log(field[:, j] / level)
            m = np.isfinite(y)
            idx = np.where(np.diff(np.sign(y[m])))[0]
            if len(idx) == 0:
                continue
            tt = np.log(Te[m])
            q = idx[0]
            f = -y[m][q] / (y[m][q + 1] - y[m][q])
            out[j] = float(np.exp(tt[q] + f * (tt[q + 1] - tt[q])))
        return out

    qn_cross = crossing(qn, 0.10)
    if not np.all(np.isfinite(qn_cross)):
        raise RuntimeError("the quasi-neutrality requirement does not cross "
                           "10 percent in every density column; the boundary "
                           "is not a curve on this grid")
    qn_cross_tot = crossing(qn_tot, 0.10)
    shift = float(np.nanmax(np.abs(qn_cross_tot / qn_cross - 1.0)))
    if shift > 1e-3:
        raise RuntimeError(
            f"the 10 percent quasi-neutrality boundary moves by {shift*100:.3g} "
            f"percent in Te when delta n_e is taken from the whole bound "
            f"manifold instead of the ground state alone. The boundary is "
            f"then a property of the definition, not of the plasma")
    n_over_10_tot = int((qn_tot[fin] > 0.10).sum())
    n_over_100_tot = int((qn_tot[fin] > 1.00).sum())
    if (n_over_10_tot, n_over_100_tot) != (n_over_10, n_over_100):
        raise RuntimeError(
            f"the census changes with the definition: ground state alone gives "
            f"{n_over_10}/{n_over_100} operators above 10/100 percent, the "
            f"whole bound manifold gives {n_over_10_tot}/{n_over_100_tot}")
    print(f"guard  the 10 percent boundary shifts by {shift*100:.3g} percent "
          f"in Te between the two definitions, and the census is identical "
          f"under both")
    ly_cross = {D: crossing(tau[D], 1.0) for D in D_all}
    print(f"quasi-neutrality boundary (10% of n_e): Te = "
          f"{np.nanmin(qn_cross):.3f} to {np.nanmax(qn_cross):.3f} eV "
          f"across the {nN} density columns")
    for D in D_all:
        c = ly_cross[D]
        print(f"optical-depth boundary (tau_0 = 1), D = {D:g} cm: Te = "
              f"{np.nanmin(c):.3f} to {np.nanmax(c):.3f} eV")
    both_lo = min(np.nanmin(qn_cross), min(np.nanmin(ly_cross[D]) for D in D_all))
    both_hi = max(np.nanmax(qn_cross), max(np.nanmax(ly_cross[D]) for D in D_all))
    n_ly_above2 = int(sum(1 for D in D_all for c in ly_cross[D]
                          if np.isfinite(c) and c > 2.0))
    print(f"both boundaries lie between {both_lo:.2f} and {both_hi:.2f} eV; "
          f"{n_ly_above2} of {len(D_all)*nN} optical-depth crossings sit above "
          f"2 eV, all of them at D = {max(D_all):g} cm")

    fig, (axm, axe) = plt.subplots(1, 2, figsize=(7.5, 3.4),
                                   gridspec_kw=dict(wspace=0.34,
                                                    width_ratios=[1.12, 1.0]))
    axm.set_facecolor("0.93")
    pc = axm.pcolormesh(Te_edges, ne_edges, np.log10(qn).T, cmap="magma",
                        shading="flat", rasterized=True)
    axm.set_xscale("log")
    axm.set_yscale("log")
    cb = fig.colorbar(pc, ax=axm, pad=0.02)
    cb.set_label(r"$\log_{10}$ required $|\Delta n_e / n_e|$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    TT, NN = np.meshgrid(Te, ne, indexing="ij")
    axm.contour(TT.T, NN.T, qn.T, levels=[0.10], colors=[C_BLUE],
                linewidths=1.8, zorder=4)
    lstyles = ["-", "--", (0, (1, 1.2))][:len(D_all)]
    for D, ls in zip(D_all, lstyles):
        axm.contour(TT.T, NN.T, tau[D].T, levels=[1.0], colors=[C_VERM],
                    linewidths=1.4, linestyles=[ls], zorder=4)
    # A legend rather than inline contour labels: four curves crowd into the
    # same two-decade strip of this plane and their labels overprint.
    axm.legend(handles=(
        [Line2D([], [], color=C_BLUE, lw=1.8,
                label=r"quasi-neutrality, $|\Delta n_e/n_e| = 10\%$")]
        + [Line2D([], [], color=C_VERM, lw=1.4, ls=ls,
                  label=rf"Lyman-$\alpha$ $\tau_0 = 1$, $D = {D:g}$ cm")
           for D, ls in zip(D_all, lstyles)]),
        loc="upper right", frameon=True, facecolor="w", framealpha=0.88,
        edgecolor="none", fontsize=6.0, handletextpad=0.6,
        labelspacing=0.35, borderpad=0.5)
    axm.axvline(2.0, color=C_INK, lw=1.0, ls="-", zorder=5)
    axm.axvspan(2.0, Te_edges[-1], color="w", alpha=0.16, zorder=3, lw=0)
    axm.text(2.13, ne_edges[0] * 1.5, r"quantitative scope, $T_e \geq 2$ eV",
             fontsize=6.6, color=C_INK, ha="left", va="bottom", rotation=90)
    axm.set_xlabel(r"$T_e$  [eV]")
    axm.set_ylabel(r"$n_e$  [cm$^{-3}$]")
    axm.set_xticks([1, 2, 3, 5, 7, 10])
    axm.set_xticklabels(["1", "2", "3", "5", "7", "10"])
    axm.xaxis.set_minor_formatter(NullFormatter())
    axm.set_title("(a) both boundaries on one plane", loc="left", fontsize=8.5)

    for j in range(nN):
        axe.plot(Te[:-1], qn[:-1, j] / 0.10, "-", color=C_BLUE, lw=0.9,
                 alpha=0.85, zorder=3)
        axe.plot(Te, tau[D_mid][:, j] / 1.0, "-", color=C_VERM, lw=0.9,
                 alpha=0.85, zorder=3)
    axe.plot(Te, np.nanmax(tau[max(D_all)], axis=1), ls=(0, (1, 1.2)),
             color=C_VERM, lw=1.3, zorder=4)
    axe.axhline(1.0, color=C_INK, lw=1.1, zorder=5)
    axe.axvline(2.0, color=C_INK, lw=1.0, zorder=5)
    axe.axvspan(2.0, Te[-1], color="0.90", zorder=0, lw=0)
    axe.set_xscale("log")
    axe.set_yscale("log")
    axe.set_xlim(Te[0], Te[-1])
    axe.set_xlabel(r"$T_e$  [eV]")
    axe.set_ylabel("quantity / its limit")
    axe.set_xticks([1, 2, 3, 5, 7, 10])
    axe.set_xticklabels(["1", "2", "3", "5", "7", "10"])
    axe.xaxis.set_minor_formatter(NullFormatter())
    axe.set_title("(b) each constraint against its own limit", loc="left",
                  fontsize=8.5)
    axe.text(0.97, 0.955, "above 1: the constraint is violated",
             transform=axe.transAxes, ha="right", va="top", fontsize=6.6,
             color=C_INK)
    axe.legend(handles=[
        Line2D([], [], color=C_BLUE, lw=1.2,
               label=r"quasi-neutrality: $|\Delta n_e/n_e| \,/\, 10\%$"),
        Line2D([], [], color=C_VERM, lw=1.2,
               label=rf"Lyman-$\alpha$ $\tau_0$, $D = {D_mid:g}$ cm, all "
                     rf"{nN} columns"),
        Line2D([], [], color=C_VERM, lw=1.3, ls=(0, (1, 1.2)),
               label=rf"$\tau_0$ at $D = {max(D_all):g}$ cm, densest column"),
        Patch(facecolor="0.90", edgecolor="none",
              label=r"$T_e \geq 2$ eV")],
        loc="upper left", bbox_to_anchor=(-0.02, -0.20), ncol=2,
        frameon=False, fontsize=6.3, handletextpad=0.6, labelspacing=0.35,
        columnspacing=1.0)
    provenance(fig, y=-0.215)
    save(fig, "fig6_1_scope")

    tok.update({
        "@QN10@": str(n_over_10), "@QN100@": str(n_over_100),
        "@QNTOT@": str(int(fin.sum())),
        "@QN_TE1_LO@": f"{row1.min():.1f}", "@QN_TE1_HI@": f"{row1.max():.1f}",
        "@QN_WARM@": sci(qn[warm].max(), 2),
        "@QN_X_LO@": f"{np.nanmin(qn_cross):.2f}",
        "@QN_X_HI@": f"{np.nanmax(qn_cross):.2f}",
        "@SIG0@": sci(sig0_1eV, 4), "@SIG0BAD@": sci(REC_SIGMA0_BAD, 5),
        "@INFL@": f"{infl:.4f}",
        "@DMID@": f"{D_mid:g}", "@DMAX@": f"{max(D_all):g}",
        "@DMIN@": f"{min(D_all):g}", "@NDS@": str(len(D_all)),
        "@LY_X_LO@": f"{np.nanmin(ly_cross[min(D_all)]):.2f}",
        "@LY_X_HI@": f"{np.nanmax(ly_cross[max(D_all)]):.2f}",
        "@LYMID_LO@": f"{np.nanmin(ly_cross[D_mid]):.2f}",
        "@LYMID_HI@": f"{np.nanmax(ly_cross[D_mid]):.2f}",
        "@BOTH_LO@": f"{both_lo:.2f}", "@BOTH_HI@": f"{both_hi:.2f}",
        "@NLY2@": str(n_ly_above2),
        "@TAU_B@": sci(tau[D_mid][ib, jb], 2),
    })


    # =======================================================================
    # FIGURE 5 -- fig1_1_diagnostic_chain
    #   A schematic, not a data plot.  Chapter 1 has no figure, and a reader
    #   who has never seen divertor spectroscopy needs the chain laid out
    #   before the thesis can argue about one link in it.
    #   No invented numbers: the two wavelengths are computed from the
    #   ionisation energies in the pipeline's own state index.
    # =======================================================================
    print()
    print("=" * 78)
    print("FIG 1.1  THE DIAGNOSTIC CHAIN")
    print("=" * 78)
    si_rows = list(csv.DictReader(sip.read_text().splitlines()))
    I_eV = {r["label"].strip(): float(r["I_eV"]) for r in si_rows}
    for need in ("2P", "3D", "4F"):
        if need not in I_eV:
            raise RuntimeError(f"state {need} is absent from {sip}; the Balmer "
                               f"wavelengths cannot be computed and this "
                               f"schematic must not name them")
    lam_a = HC_EV_NM / (I_eV["2P"] - I_eV["3D"])       # H-alpha, n=3 -> 2
    lam_b = HC_EV_NM / (I_eV["2P"] - I_eV["4F"])       # H-beta,  n=4 -> 2
    print(f"H-alpha  n=3 -> 2  {I_eV['2P'] - I_eV['3D']:.5f} eV  "
          f"{lam_a:.2f} nm     (from I_eV in {sip.name}, not hardcoded)")
    print(f"H-beta   n=4 -> 2  {I_eV['2P'] - I_eV['4F']:.5f} eV  "
          f"{lam_b:.2f} nm")
    for nm, got, want in (("H-alpha", lam_a, 656.28), ("H-beta", lam_b, 486.13)):
        if abs(got - want) / want > 2e-3:
            raise RuntimeError(
                f"{nm} comes out at {got:.3f} nm from the state index, more "
                f"than 0.2 percent from the laboratory value {want} nm. The "
                f"energies in {sip} are not what this schematic assumes")

    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)

    BOXY, BOXH, BOXW = 21.0, 15.0, 19.5
    xs_box = [2.0, 27.0, 52.0, 77.0]
    titles = ["measured spectrum",
              r"line ratio $I_{{\rm H}\alpha}/I_{{\rm H}\beta}$",
              r"lookup table $R(T_e,n_e)$", r"inferred $T_e$, $n_e$"]
    subs = [rf"${lam_a:.1f}$ nm and ${lam_b:.1f}$ nm",
            "one number per\nline of sight",
            "invert for the pair\nthat reproduces it",
            "the reported\nplasma conditions"]
    for q, (x, t, sb) in enumerate(zip(xs_box, titles, subs)):
        ax.add_patch(FancyBboxPatch(
            (x, BOXY), BOXW, BOXH, boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=1.1, edgecolor=C_INK if q != 2 else C_VERM,
            facecolor="0.965" if q != 2 else "#fdf0e8", zorder=3))
        ax.text(x + BOXW / 2, BOXY + BOXH - 2.8, t, ha="center", va="center",
                fontsize=7.6, zorder=4)
        if q != 0:
            ax.text(x + BOXW / 2, BOXY + 5.2, sb, ha="center", va="center",
                    fontsize=6.8, color="0.35", zorder=4, linespacing=1.3)
    for q in range(3):
        ax.add_patch(FancyArrowPatch(
            (xs_box[q] + BOXW + 0.8, BOXY + BOXH / 2),
            (xs_box[q + 1] - 0.8, BOXY + BOXH / 2),
            arrowstyle="-|>", mutation_scale=11, lw=1.3, color=C_INK,
            zorder=4))

    # a schematic spectrum inside the first box: two lines, no numeric axis.
    # The relative heights are arbitrary and say nothing about the real ratio;
    # the caption says so.
    sx0, sx1 = xs_box[0] + 3.4, xs_box[0] + BOXW - 3.4
    base = BOXY + 4.5
    ax.plot([sx0, sx1], [base] * 2, "-", color="0.55", lw=0.8, zorder=4)
    for frac_x, hgt, lab in ((0.26, 3.4, r"H$\beta$"), (0.74, 5.0, r"H$\alpha$")):
        xx = sx0 + frac_x * (sx1 - sx0)
        ax.plot([xx, xx], [base, base + hgt], "-", color=C_BLUE, lw=1.9,
                zorder=5)
        ax.text(xx, base + hgt + 0.5, lab, ha="center", va="bottom",
                fontsize=6.3, color=C_BLUE, zorder=5)
    ax.text(xs_box[0] + BOXW / 2, base - 1.1,
            rf"${lam_b:.1f}$ and ${lam_a:.1f}$ nm", ha="center", va="top",
            fontsize=6.2, color="0.35", zorder=5)

    # the hidden assumption
    ax.add_patch(FancyBboxPatch(
        (34.0, 2.2), 56.0, 12.4, boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=1.2, edgecolor=C_VERM, facecolor="#fdf0e8", zorder=3))
    ax.text(62.0, 11.4, "the assumption that enters here", ha="center",
            va="center", fontsize=8.0, color=C_VERM, zorder=4)
    ax.text(62.0, 5.9,
            "the table is built from the CR-equilibrium populations, so it "
            "assumes\nthe ionisation balance has settled: "
            r"$n_g/n_{\rm ion} = u^{\rm CRE}(T_e, n_e)$."
            "\nA plasma still in transit does not satisfy it.",
            ha="center", va="center", fontsize=6.9, color="0.25", zorder=4,
            linespacing=1.45)
    ax.add_patch(FancyArrowPatch(
        (62.0, 14.9), (xs_box[2] + BOXW / 2, BOXY - 0.9),
        arrowstyle="-|>", mutation_scale=11, lw=1.4, color=C_VERM, zorder=4))
    ax.text(1.0, 38.5, "schematic: no measured or computed quantity is shown "
                       "on this figure", fontsize=6.4, color="0.45",
            va="top", ha="left", style="italic")
    provenance(fig, y=0.02)
    save(fig, "fig1_1_diagnostic_chain")
    tok.update({"@LAMA@": f"{lam_a:.1f}", "@LAMB@": f"{lam_b:.1f}"})


    # =======================================================================
    # FIGURE 6 -- fig2_1_state_space
    #   Chapter 2 has no figure.  The 43-state space, the five processes the
    #   matrix contains, and the one state whose behaviour decides the
    #   slow/fast partition.
    #   The state list is READ from the pipeline's state index, never written
    #   out here; the counts below are computed from that file.
    # =======================================================================
    print()
    print("=" * 78)
    print("FIG 2.1  THE STATE SPACE")
    print("=" * 78)
    for col in ("label", "n", "l", "bundled", "I_eV"):
        if col not in si_rows[0]:
            raise RuntimeError(f"{sip} has no {col!r} column; the state-space "
                               f"figure cannot be drawn without it")
    st_lab = [r["label"].strip() for r in si_rows]
    st_n = np.array([int(float(r["n"])) for r in si_rows])
    st_l = np.array([int(float(r["l"])) for r in si_rows])
    st_b = np.array([r["bundled"].strip() == "True" for r in si_rows])
    st_I = np.array([float(r["I_eV"]) for r in si_rows])
    if len(st_lab) != ctx.n_states or st_lab != list(ctx.labels):
        raise RuntimeError(f"{sip} read here does not match the ordering "
                           f"cr_context loaded from the same file")
    n_res, n_bun = int((~st_b).sum()), int(st_b.sum())
    res_nmax, bun_nmin, bun_nmax = st_n[~st_b].max(), st_n[st_b].min(), st_n[st_b].max()
    exp_res = sum(range(1, res_nmax + 1))
    if n_res != exp_res:
        raise RuntimeError(
            f"{n_res} resolved states are listed but l-resolving n = 1 to "
            f"{res_nmax} gives {exp_res}. The state space is not what this "
            f"figure would draw")
    if n_bun != bun_nmax - bun_nmin + 1:
        raise RuntimeError(f"{n_bun} bundled states span n = {bun_nmin} to "
                           f"{bun_nmax}, which is not one level per n")
    print(f"state space          {n_res} l-resolved (n = 1 to {res_nmax}) + "
          f"{n_bun} bundled (n = {bun_nmin} to {bun_nmax}) = "
          f"{ctx.n_states} states, read from {sip.name}")

    # which states carry which process, counted from the matrices themselves
    A_res = np.load(require_file(root / "data/processed/Radiative/A_resolved.npy",
                                 "resolved radiative rates"))
    if A_res.shape != (n_res, n_res):
        raise RuntimeError(f"A_resolved {A_res.shape} does not match the "
                           f"{n_res} resolved states in {sip}")
    # A_resolved[lower, upper] is the rate upper -> lower
    n_rad = int((A_res.sum(axis=0) > 0).sum())
    n_lmix = int((Klmix != 0).any(axis=(1, 2)).sum())
    i1s, i2s, i2p = st_lab.index("1S"), st_lab.index("2S"), st_lab.index("2P")
    if A_res[i1s, i2s] != 0.0:
        raise RuntimeError(
            f"A[1s <- 2s] = {A_res[i1s, i2s]:.3e} is not zero. The whole "
            f"argument for keeping 2s in the fast manifold rests on it having "
            f"no E1 decay to the ground state")
    print(f"radiative decay      {n_rad} of {n_res} resolved states have a "
          f"non-zero A out; A[1s <- 2s] is exactly zero (no E1)")
    print(f"proton l-mixing      acts on {n_lmix} states, every resolved "
          f"level with n >= 2")
    print(f"A[1s <- 2p]          {A_res[i1s, i2p]:.5e} s^-1, for contrast")

    # 2s: what holds it in the fast manifold
    lm_2s = Klmix[i2p, i2s, :][:, None] * ne[None, :]      # 2s -> 2p, s^-1
    tot_2s = -np.array([[L[i, j][i2s, i2s] for j in range(nN)] for i in range(nT)])
    if np.any(tot_2s <= 0):
        raise RuntimeError("the 2s diagonal of L is non-negative somewhere; "
                           "it is not a loss rate and no fraction of it means "
                           "anything")
    dev2 = abs(lm_2s[ib, jb] / L[ib, jb][i2p, i2s] - 1.0)
    if dev2 > 1e-12:
        raise RuntimeError(
            f"K_lmix[2p,2s]*ne = {lm_2s[ib,jb]:.6e} does not reproduce "
            f"L[2p,2s] = {L[ib,jb][i2p,i2s]:.6e} at the benchmark "
            f"({dev2:.2e} apart); the l-mixing channel in the matrix is not "
            f"the one being quoted")
    fr_2s = lm_2s / tot_2s
    print(f"2s loss rate         {tot_2s[ib,jb]:.5e} s^-1 at the benchmark, "
          f"of which proton l-mixing to 2p is {fr_2s[ib,jb]*100:.4f}%")
    print(f"   over the grid     {fr_2s.min()*100:.4f}% (at Te = "
          f"{Te[np.unravel_index(fr_2s.argmin(), fr_2s.shape)[0]]:.3g} eV) to "
          f"{fr_2s.max()*100:.4f}% (at Te = "
          f"{Te[np.unravel_index(fr_2s.argmax(), fr_2s.shape)[0]]:.3g} eV)")
    print(f"   the 99.99% figure quoted in the brief is the grid MAXIMUM; the "
          f"floor over the whole grid is {fr_2s.min()*100:.2f}%")

    fig, (ax, axE) = plt.subplots(1, 2, figsize=(7.8, 4.9),
                                  gridspec_kw=dict(width_ratios=[3.3, 1.0],
                                                   wspace=0.02))
    ax.axis("off")
    ROW = {1: 0.0, 2: 1.25, 3: 2.5, 4: 3.75, "gap": 4.55}
    for q, nn in enumerate(range(bun_nmin, bun_nmax + 1)):
        ROW[nn] = 5.20 + 0.20 * q
    ROW["cont"] = 7.30
    XL, XR, XA = 0.13, 0.685, 0.715

    ax.plot([XL - 0.02, XR - 0.02], [ROW["cont"]] * 2, color=C_INK, lw=1.5)
    ax.text(XA, ROW["cont"], r"continuum (H$^+$ + e$^-$)", fontsize=7.4,
            va="center", ha="left")

    xs = {1: [0.34], 2: [0.26, 0.42], 3: [0.22, 0.34, 0.46],
          4: [0.18, 0.30, 0.42, 0.54]}
    lname = {0: "s", 1: "p", 2: "d", 3: "f"}
    for nn in (1, 2, 3, 4):
        for x, ll in zip(xs[nn], range(nn)):
            ax.plot([x - 0.045, x + 0.045], [ROW[nn]] * 2, color=C_INK, lw=1.4)
            ax.text(x, ROW[nn] + 0.13, f"{nn}{lname[ll]}", fontsize=6.8,
                    ha="center")
    ax.text(XA, 3.95, rf"$n \leq {res_nmax}$: $\ell$-resolved, {n_res} states",
            fontsize=7.2, va="center", ha="left", color="0.3")
    ax.text(0.36, ROW["gap"], r"$\vdots$", fontsize=9, ha="center", color="0.45")
    ax.text(XA, ROW["gap"], rf"$n = 5$ to ${res_nmax}$: $\ell$-resolved, "
                            rf"not drawn", fontsize=6.9, va="center",
            ha="left", color="0.45")
    ax.text(XA, 5.35, "electron-impact ionisation\n(every state)",
            fontsize=7.0, color=C_GREEN, ha="left", va="center",
            linespacing=1.3)
    ax.text(XA, 6.70, "recombination, entering as\nthe source vector "
                      "(every state)", fontsize=7.0, color=C_PURPLE,
            ha="left", va="center", linespacing=1.3)
    for nn in range(bun_nmin, bun_nmax + 1):
        ax.plot([0.24, 0.48], [ROW[nn]] * 2, color="0.45", lw=1.1)
    ax.text(0.50, ROW[bun_nmin], rf"$n={bun_nmin}$", fontsize=6.4,
            va="center", color="0.45")
    ax.text(0.50, ROW[bun_nmax], rf"$n={bun_nmax}$", fontsize=6.4,
            va="center", color="0.45")
    ax.text(XA, 6.05, rf"$n = {bun_nmin}$ to ${bun_nmax}$: bundled, "
                      rf"{n_bun} states",
            fontsize=7.2, va="center", ha="left", color="0.45")

    # --- the five processes ------------------------------------------------
    # 1. proton-impact l-mixing, drawn for n = 2 and 3 only (see ch3 note:
    #    at n = 4 the sublevels are too close for an arrow to render)
    for nn in (2, 3):
        for x0, x1 in zip(xs[nn][:-1], xs[nn][1:]):
            ax.add_patch(FancyArrowPatch((x0 + 0.05, ROW[nn]), (x1 - 0.05, ROW[nn]),
                                         arrowstyle="<|-|>", mutation_scale=7,
                                         lw=1.0, color=C_ORANGE))
    ax.text(XA, 2.30,
            rf"proton $\ell$-mixing, $\Delta n = 0$" "\n"
            rf"({n_lmix} states: every resolved $n \geq 2$)",
            fontsize=7.0, va="center", ha="left", color=C_ORANGE,
            linespacing=1.3)

    # 2. electron-impact excitation and de-excitation
    ax.add_patch(FancyArrowPatch((0.085, ROW[2] + 0.06), (0.085, ROW[3] - 0.06),
                                 arrowstyle="<|-|>", mutation_scale=9, lw=1.5,
                                 color=C_BLUE))
    ax.text(0.065, (ROW[2] + ROW[3]) / 2, "electron impact\nexcitation and\n"
                                          "de-excitation",
            fontsize=6.8, color=C_BLUE, ha="right", va="center",
            linespacing=1.3)

    # 3. radiative decay
    ax.add_patch(FancyArrowPatch((0.435, ROW[2] - 0.09), (0.365, ROW[1] + 0.11),
                                 arrowstyle="-|>", mutation_scale=9, lw=1.2,
                                 color=C_VERM, connectionstyle="arc3,rad=-0.22"))
    ax.text(0.465, (ROW[1] + ROW[2]) / 2 - 0.05,
            rf"$A_{{2p \to 1s}} = {A_res[i1s, i2p]/1e8:.2f}\times10^{{8}}$ s$^{{-1}}$",
            fontsize=6.6, color=C_VERM, ha="left", va="center")

    # 4. ionisation, 5. recombination as a source
    ax.add_patch(FancyArrowPatch((0.585, ROW[4] + 0.10), (0.585, ROW["cont"] - 0.08),
                                 arrowstyle="-|>", mutation_scale=10, lw=1.6,
                                 color=C_GREEN))
    ax.add_patch(FancyArrowPatch((0.645, ROW["cont"] - 0.08), (0.645, ROW[bun_nmax] + 0.08),
                                 arrowstyle="-|>", mutation_scale=10, lw=1.6,
                                 color=C_PURPLE))

    # the 2s callout: the fact that keeps {1s} the sole slow state
    ax.add_patch(FancyArrowPatch((0.245, ROW[2] - 0.10), (0.315, ROW[1] + 0.11),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                 color="0.55", ls=(0, (3, 2)),
                                 connectionstyle="arc3,rad=0.22"))
    ax.text(XL - 0.005, ROW[1] - 0.58,
            r"$2s$ has NO E1 decay to $1s$ ($A = 0$ exactly). It is still fast:"
            "\n"
            rf"proton $\ell$-mixing to $2p$ carries ${fr_2s[ib,jb]*100:.2f}\%$ "
            rf"of its loss rate at the" "\n"
            rf"benchmark, and never less than ${fr_2s.min()*100:.2f}\%$ "
            rf"anywhere on the grid. That is" "\n"
            r"why $\mathcal{S} = \{1s\}$ is the only slow state.",
            fontsize=6.8, color=C_INK, va="top", ha="left", linespacing=1.5)

    # the partition
    ax.add_patch(Rectangle((XL, ROW[1] - 0.30), XR - XL, 0.56, fill=False,
                           ec=C_VERM, lw=1.1, ls="--"))
    ax.text(XR - 0.01, ROW[1] - 0.02, r"slow: $\mathcal{S} = \{1s\}$",
            fontsize=7.4, color=C_VERM, va="center", ha="right")
    ax.add_patch(Rectangle((XL, ROW[2] - 0.36), XR - XL,
                           ROW[bun_nmax] - ROW[2] + 0.66, fill=False,
                           ec=C_BLUE, lw=1.1, ls="--"))
    ax.text(XL + 0.01, ROW[bun_nmax] + 0.40,
            rf"fast: $\mathcal{{F}}$, {ctx.n_states - 1} excited levels",
            fontsize=7.4, color=C_BLUE, va="bottom", ha="left")
    ax.text(0.0, ROW[1] - 2.32, "schematic: vertical spacing is not the energy "
                                "scale (see right-hand panel)",
            fontsize=6.4, color="0.45", va="top", style="italic")
    ax.set_xlim(-0.055, 1.30)
    ax.set_ylim(ROW[1] - 2.62, ROW["cont"] + 0.35)

    # --- the true energy scale, for contrast -------------------------------
    for q in range(ctx.n_states):
        col = "0.45" if st_b[q] else C_INK
        xx = 0.10 + 0.085 * (st_l[q] if not st_b[q] else 2)
        axE.plot([xx, xx + 0.14], [-st_I[q]] * 2, "-", color=col, lw=1.0)
    axE.axhline(0.0, color=C_INK, lw=1.4)
    for nn, txt in ((1, "n=1"), (2, "n=2"), (3, "n=3")):
        q = int(np.where(st_n == nn)[0][0])
        axE.text(0.80, -st_I[q], txt, fontsize=6.4, va="center", ha="right",
                 color="0.35")
    q4 = int(np.where(st_n == 4)[0][0])
    axE.annotate("", xy=(0.955, -st_I[q4]), xytext=(0.955, 0.0),
                 arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=C_GREEN,
                                 mutation_scale=6))
    axE.annotate(rf"$n \geq 4$: {int((st_n >= 4).sum())} of the "
                 rf"{ctx.n_states} states" "\n"
                 rf"lie within {st_I[q4]:.2f} eV of the" "\n"
                 "continuum, which is why the" "\n"
                 "left-hand panel cannot be" "\n" "drawn to scale",
                 xy=(0.04, -7.4), xytext=(0.04, -7.4),
                 fontsize=6.1, color=C_GREEN, va="center", ha="left",
                 linespacing=1.4)
    axE.set_xlim(0.0, 1.0)
    axE.set_ylim(-14.1, 1.0)
    axE.set_xticks([])
    axE.set_ylabel(r"$E$  [eV], $E = 0$ at the continuum", fontsize=7.5)
    axE.yaxis.set_label_position("right")
    axE.yaxis.tick_right()
    axE.tick_params(labelsize=6.5)
    for sp in ("top", "left"):
        axE.spines[sp].set_visible(False)
    axE.set_title("true energy scale", loc="center", fontsize=7.4, color="0.3")
    provenance(fig, y=0.0)
    save(fig, "fig2_1_state_space")

    tok.update({
        "@NRES@": str(n_res), "@NBUN@": str(n_bun),
        "@RESNMAX@": str(int(res_nmax)), "@BUNLO@": str(int(bun_nmin)),
        "@BUNHI@": str(int(bun_nmax)), "@NLMIX@": str(n_lmix),
        "@NRAD@": str(n_rad),
        "@F2S_B@": f"{fr_2s[ib,jb]*100:.2f}",
        "@F2S_LO@": f"{fr_2s.min()*100:.2f}",
        "@F2S_HI@": f"{fr_2s.max()*100:.2f}",
        "@A2P@": sci(A_res[i1s, i2p], 3),
        "@TOT2S@": sci(tot_2s[ib, jb], 3),
        "@E4@": f"{st_I[q4]:.2f}", "@NGE4@": str(int((st_n >= 4).sum())),
    })


    # =======================================================================
    # CAPTIONS.  Every number below is a token substituted from the arrays
    # that were actually plotted, in the same pattern as fig5_captions.tex.
    # Nothing here is typed by hand, and the file refuses to be written if a
    # token is left unsubstituted or if an em dash reaches it.
    # =======================================================================
    caption_src = r"""% figures/story_captions.tex
% Generated by src/analysis/make_story_figures.py on @DATE@.
% Every number below was computed by that script from the same arrays it
% plotted; do not edit them by hand. CR data SHA-8 @SHA8@
% (L_grid sha256 @SHAL@...).
%
% Usage:  \input{figures/story_captions.tex}  in the preamble, then
%   \begin{figure}[t]\centering
%     \includegraphics{figures/fig5_5_reversal.pdf}
%     \caption{\CapFigReversal}\label{fig:reversal}
%   \end{figure}

\newcommand{\CapFigReversal}{%
  \textbf{The reversal: the approximation that is doubted holds, the
  assumption that is not examined fails.}
  Two errors in the same observable, the shell ratio $R = n_3/n_4$, measured
  along the same trajectory after a $+@FRAC@\%$ step in $T_e$ at fixed $n_e$,
  and reported as the maximum over the timescale-separated plateau window
  $30\tau_{\rm relax} < t < \tau_{\rm slow}/30$.
  The \emph{quasi-steady-state closure residual} is
  $|R(t)/R^{\rm QSS}(n_g(t)) - 1|$, where $R^{\rm QSS}$ is rebuilt at each
  instant from the excited-state block alone,
  $n_E = -L_{EE}^{-1}(L_{Eg}n_g + S_E)$, evaluated at the
  \emph{instantaneous} ground density rather than at any tabulated value.
  The \emph{CRE distance} is $|R(t)/R^{\rm CRE} - 1|$, the distance to the
  equilibrium-ionisation-balance state $n = -L^{-1}S$ that a two-parameter
  $(T_e, n_e)$ lookup table returns.
  (a) Both, for all @NCELL@ analysed cells across the @NDCOL@ density
  columns. The closure residual runs from $@CLO_MIN@$ to $@CLO_MAX@$, so the
  closure everyone worries about is never worse than one part in
  $@CLO_MAX_INV@$ anywhere on the analysed grid: it is $@CLO_B@$ at the
  benchmark ($T_e = @BENCH_TE@$~eV, $n_e = @BENCH_NE@$~cm$^{-3}$), one part in
  $@CLO_B_INV@$, and $@CLO_W@$ at the coldest point
  $[@WORST_I@,@WORST_J@]$, one part in $@CLO_W_INV@$.
  The CRE distance over the same cells runs from @CRE_MIN@\% to @CRE_MAX@\%:
  @CRE_B@\% at the benchmark and @CRE_W@\% at the coldest point.
  The assumption nobody examines therefore fails at the tens-of-percent
  level exactly where the closure everybody examines is exact to within
  one part in $10^{5}$ or better.
  (b) The ratio of the two over the grid. The gap is @GAP_LO@ to @GAP_HI@
  decades wide, @GAP_B@ decades at the benchmark and @GAP_W@ decades at the
  coldest point; it is not a single number and should not be quoted as one.
  Hatched cells (@NNOWIN@ of @NPAIR@) have no timescale-separated plateau
  window, $M \leq @MWIN@$, an imposed cut rather than a measured floor.
  Trajectories are eigen-propagations of the full @NSTATES@-state operator,
  sampled at @NT@ points across each window and checked against
  \texttt{scipy.linalg.expm} at the window midpoint of every cell, worst
  disagreement $@PROPCHK@$.
  Values below $2$~eV are optically thin upper estimates: Lyman trapping is
  not switched on here.}

\newcommand{\CapFigTrajectory}{%
  \textbf{What the plateau looks like: the excited states keep up, the
  reservoir does not.}
  The shell ratio $R(t) = n_3/n_4$ after a step in $T_e$ at fixed $n_e$,
  on a logarithmic time axis long enough to hold both timescales.
  Three phases are visible. $R$ rises on $\tau_{\rm relax}$ as the excited
  manifold re-equilibrates against a ground state that has not yet moved; it
  sits at the partial-equilibrium value $R^{\rm PE}$; then it decays on
  $\tau_{\rm slow}$ as the ground state, and with it the ionisation balance,
  finally responds. The excited states are fast enough to track the new
  temperature within nanoseconds. The reservoir that feeds them is not.
  In (a) the curve crosses the $R^{\rm PE}$ line at $0.7\,\tau_{\rm relax}$,
  peaks near $1.2\,\tau_{\rm relax}$ and settles back onto it, because $n = 4$ relaxes faster than $n = 3$ and
  the ratio briefly exceeds its partial-equilibrium value.
  (a) The benchmark, $T_e = @BENCH_TE@ \rightarrow @TB_TENEW@$~eV at
  $n_e = @BENCH_NE@$~cm$^{-3}$: $R$ moves from $@TB_ROLD@$ through
  $@TB_RPE@$ to $@TB_RNEW@$, a plateau error of @TB_EPS@\%, with
  $\tau_{\rm relax} = @TB_TR@$~s and $\tau_{\rm slow} = @TB_TQ@$~s
  ($M = @TB_M@$).
  (b) The coldest grid point, $T_e = @TC_TE@ \rightarrow @TC_TENEW@$~eV at
  $n_e = @TC_NE@$~cm$^{-3}$: $@TC_ROLD@$ through $@TC_RPE@$ to $@TC_RNEW@$,
  @TC_EPS@\%, with $\tau_{\rm relax} = @TC_TR@$~s and
  $\tau_{\rm slow} = @TC_TQ@$~s ($M = @TC_M@$), five orders of magnitude of
  separation between the two clocks.
  Every timescale quoted belongs to the \emph{post-step} operator, the one
  that governs the relaxation; the same grid point read from its unstepped
  operator gives a different $M$, and the two must not be interchanged.
  $R^{\rm PE}$ is computed independently by a single linear solve against
  that operator and agrees with the two-channel construction used elsewhere
  in this chapter to $2\times10^{-16}$. The plateau is flat to @TB_FLAT@\%
  peak to peak in (a). Curves are eigen-propagations of the full
  @NSTATES@-state operator, cross-checked against \texttt{scipy.linalg.expm}
  inside the plateau window, worst disagreement $@TRAJCHK@$.}

\newcommand{\CapFigStructuralMaps}{%
  \textbf{The two structural coefficients, and the invariant they multiply
  to.}
  The plateau error obeys
  $\varepsilon = |\exp(\bar{S} G \,\mathrm{d}\ln T_e) - 1|$ exactly, which
  splits it into a part belonging to the observable and a part belonging to
  the plasma.
  (a) $|\bar{S}| = |f_3 - f_4|$, the sensitivity of the $n=3/n=4$ ratio to
  the ground-state reservoir, where $f_n$ is the ground-fed fraction of shell
  $n$.
  (b) $|G| = |\mathrm{d}\ln u / \mathrm{d}\ln T_e|$, the reservoir gain,
  measuring how far the ground population moves for a given change in
  temperature.
  (c) Their product $|\bar{S}G|$, which is the small-step limit of
  $\varepsilon/|\mathrm{d}\ln T_e|$ and is the coefficient this chapter reports
  in place of a percentage that depends on the step chosen. It is written as a
  limiting ratio and not as $\mathrm{d}\varepsilon/\mathrm{d}\ln T_e$ because
  $\varepsilon = |\mathrm{e}^z-1|$ has a corner at $z=0$, where the two-sided
  derivative does not exist. $G$ itself is a secant over the step taken, not a
  local derivative, so it carries a weak and systematic step dependence.
  All three are heating, one grid index in $T_e$
  ($\mathrm{d}\ln T_e = @DLNTE@$), over all @NCELL@ cells.
  Ranges quoted with their scope, because the scope changes them: over the
  @NOKM@ cells drawn here that pass the plateau-window test,
  $|\bar{S}|$ runs @SB_LO@ to @SB_HI@ and $|G|$ runs @G_LO@ to @G_HI@, giving
  a product of @PR_LO@ to @PR_HI@; over all @NROWS@ rows of the source table,
  which include cooling and larger steps, $|\bar{S}|$ runs @SB_ALL_LO@ to
  @SB_ALL_HI@ while $|G|$ is unchanged at @G_ALL_LO@ to @G_ALL_HI@.
  At the benchmark $|\bar{S}| = @SB_B@$, $|G| = @G_B@$ and the product is
  @PR_B@.
  $G$ is what makes this reframing worth anything: across step sizes of one,
  two and four grid indices at the @NTRIP@ points carrying all three, $|G|$
  varies by at most @GSTAB@\% (median @GSTABMED@) while $\varepsilon$ over
  the same points varies by up to a factor @EPSSPREAD@.
  Hatched cells (@NNOWIN3@ of @NCELL@) fail the plateau-window test
  $M > @MWIN@$; their values are drawn rather than deleted, since $\bar{S}$
  and $G$ are properties of the operator and remain defined there even where
  a plateau does not.}

\newcommand{\CapFigScope}{%
  \textbf{Two independent constraints put the same floor under the
  quantitative scope.}
  Neither boundary was chosen; both were measured, and they come from
  unrelated physics.
  \emph{Quasi-neutrality.} Imposing nuclei conservation across one grid step
  in $T_e$, the electron density must change by
  $|\Delta n_e / n_e| = |\Delta n_g| / n_{\rm ion}$, taken from the two
  CR-equilibrium solves either side of the step. This zero-dimensional model
  holds $n_e$ fixed, so wherever that requirement is large the model is
  contradicting itself. Of the @QNTOT@ one-step operators on the grid,
  @QN10@ require more than $10\%$ and @QN100@ require more than $100\%$. At
  $T_e = 1$~eV the requirement is between @QN_TE1_LO@ and @QN_TE1_HI@, that
  is, ten to twenty times the electron density itself. At $T_e \geq 2$~eV it
  never exceeds $@QN_WARM@$. The $10\%$ contour sits at
  $T_e = @QN_X_LO@$ to @QN_X_HI@~eV and is almost independent of density.
  \emph{Optical depth.} The Lyman-$\alpha$ line-centre optical depth over the
  half slab, $\tau_0 = n(1s)\,\sigma_0\,D/2$, with $\sigma_0$ evaluated at
  $T_{\rm at} = T_e$ from the Doppler width $\sqrt{2kT/m}$, giving
  $\sigma_0 = @SIG0@$~cm$^{2}$ at $1$~eV. The optically thin matrix used
  throughout this thesis is only defensible where $\tau_0 < 1$. That contour
  runs from $T_e = @LYMID_LO@$ to @LYMID_HI@~eV at the slab thickness
  $D = @DMID@$~cm, and across the @NDS@ thicknesses in the trapping sweep it
  spans @LY_X_LO@ to @LY_X_HI@~eV.
  (a) Both boundaries on the $(T_e, n_e)$ plane, over a map of the
  quasi-neutrality requirement.
  (b) Each constraint divided by its own limit, so that a value above one
  means the constraint is violated and the two can share a single axis.
  Both boundaries fall between @BOTH_LO@ and @BOTH_HI@~eV. Two independent
  pieces of physics, particle conservation and radiation transport, place the
  floor in the same narrow band, and $T_e \geq 2$~eV clears both everywhere
  except @NLY2@ of the @NDS@ times eight optical-depth crossings, which sit
  at the largest slab thickness $D = @DMAX@$~cm and the two highest density
  columns. The $T_e \geq 2$~eV restriction is therefore a measured boundary,
  not a hedge.}

\newcommand{\CapFigDiagnosticChain}{%
  \textbf{Where the hidden assumption enters divertor spectroscopy.}
  The inference runs left to right. A spectrometer records the Balmer
  emission; the ratio of two line intensities, here
  $I_{{\rm H}\alpha}/I_{{\rm H}\beta}$ at $@LAMA@$ and $@LAMB@$~nm, reduces
  the spectrum to a single number per line of sight; that number is looked up
  in a table of the ratio computed as a function of $(T_e, n_e)$; and the
  pair that reproduces it is reported as the plasma conditions.
  The step that is rarely stated is the third. The table is built from the
  CR-equilibrium populations, so it assumes the ionisation balance has
  already settled, $n_g/n_{\rm ion} = u^{\rm CRE}(T_e, n_e)$. A plasma still
  in transit after a change in temperature does not satisfy that, and the
  inversion then returns a temperature and a density that no plasma had.
  This thesis is about the size of that error and about which of the
  approximations in the chain is actually responsible for it.
  This is a schematic. No measured or computed quantity is plotted; the two
  wavelengths are the only numbers on it, and they follow from the
  ionisation energies in the model's own state index. The drawn line heights
  are arbitrary and carry no information about the real intensity ratio.}

\newcommand{\CapFigStateSpace}{%
  \textbf{The @NSTATES@-state space and the five processes the rate matrix
  contains.}
  Left: the level structure. Hydrogen is $\ell$-resolved up to
  $n = @RESNMAX@$, which is @NRES@ states, and bundled into one level per $n$
  from $n = @BUNLO@$ to @BUNHI@, which is @NBUN@ more. Vertical spacing is
  schematic; the right-hand panel gives the true energies, and shows why,
  with @NGE4@ of the @NSTATES@ states lying within @E4@~eV of the continuum,
  a to-scale ladder would be unreadable.
  Five processes appear in the matrix: electron-impact excitation and
  de-excitation between levels, radiative decay
  ($A_{2p \to 1s} = @A2P@$~s$^{-1}$, and @NRAD@ of the @NRES@ resolved states
  carry a non-zero decay rate), electron-impact ionisation from every state,
  recombination, which enters as the source vector rather than as a matrix
  element and feeds every state, and proton-impact $\ell$-mixing, which is
  $\Delta n = 0$ and acts on @NLMIX@ states, every resolved level with
  $n \geq 2$. Only the $n = 2$ and $n = 3$ mixing arrows are drawn; at
  $n = 4$ the sublevels are too close together for an arrow to render
  legibly.
  The dashed boxes mark the partition the analysis rests on, and $2s$ is why
  it can be drawn where it is. $2s$ has no E1 decay to the ground state, its
  radiative rate to $1s$ being exactly zero in the matrix, so on radiative
  grounds alone it would be metastable and would belong with the slow states.
  It is nonetheless fast, because proton $\ell$-mixing to $2p$ carries
  @F2S_B@\% of its total loss rate of $@TOT2S@$~s$^{-1}$ at the benchmark,
  and never less than @F2S_LO@\% anywhere on the grid, rising to @F2S_HI@\%
  at the cold edge. That single channel is what leaves
  $\mathcal{S} = \{1s\}$ as the only slow state and the remaining
  @NFAST@ levels in the fast manifold.
  The state list, including which levels are bundled and every ionisation
  energy, is read from the model's own state index, not written into this
  figure.}
"""
    tok["@NDCOL@"] = str(nN)
    tok["@NFAST@"] = str(ctx.n_states - 1)
    for k, v in tok.items():
        caption_src = caption_src.replace(k, v)
    leftover = sorted({w for w in caption_src.split() if w.startswith("@")})
    if leftover:
        raise RuntimeError(f"caption template has unsubstituted tokens: "
                           f"{leftover}")
    if "---" in caption_src:
        bad = [ln for ln in caption_src.splitlines() if "---" in ln]
        raise RuntimeError(f"an em dash reached the captions, which this "
                           f"thesis does not use: {bad}")
    if "—" in caption_src or "–" in caption_src:
        raise RuntimeError("a literal en or em dash character reached the "
                           "captions")
    emit("story_captions.tex", lambda p: p.write_text(caption_src))
    n_cmd = caption_src.count("\\newcommand")
    print()
    print(f"story_captions.tex   {n_cmd} captions, "
          f"{len(tok)} numbers injected, zero em dashes")

    # === TAIL MARKER: new figures are inserted above this line ===
    print("\nwrote:")
    for name, how in written:
        print(f"  {outdir / name}   [{how}]")
    leftover = sorted({k for k in tok})
    print(f"caption tokens ready: {len(leftover)}")


if __name__ == "__main__":
    main()
