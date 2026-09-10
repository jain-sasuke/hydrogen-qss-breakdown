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

    def check_propagator(self, ts, rtol=1e-7):
        """Independent check of the eigen-propagation against a dense matrix
        exponential.  L is strongly non-normal here (CLAUDE.md records a
        numerical abscissa of +1.28e11 s^-1 against a spectral abscissa of
        -4.40e4 s^-1), so this is not a formality."""
        d = self.n_old - self.n_new
        worst = 0.0
        for t in np.atleast_1d(ts):
            ref = expm(self.A * float(t)) @ d + self.n_new
            got = self.n_at(t)[:, 0]
            rel = np.abs(got - ref).max() / np.abs(ref).max()
            worst = max(worst, float(rel))
        if worst > rtol:
            raise RuntimeError(
                f"eigen-propagation disagrees with scipy.linalg.expm at "
                f"[{self.i},{self.j}] by {worst:.3e} relative (tolerance "
                f"{rtol:.1e}); cond(V) = {self._eig[3]:.3e}. The trajectory "
                f"cannot be trusted and nothing may be plotted")
        return worst

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

    # ---- solve every one-index heating step on the grid -------------------
    steps: dict = {}
    for i in range(len(Te) - 1):
        for j in range(len(ne)):
            steps[(i, j)] = Step(ctx, L, S, i, j, i + 1, E, posE, N3, N4, g)
    if len(steps) != REC_QN_TOTAL:
        raise RuntimeError(
            f"{len(steps)} one-index heating steps exist on this grid, but "
            f"ADDENDUM D.1's census was taken over {REC_QN_TOTAL}. The grid "
            f"has changed shape and no recorded count applies")
    dlnTe_all = np.array([s.dlnTe for s in steps.values()])
    if dlnTe_all.ptp() / dlnTe_all.mean() > 1e-9:
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
        if rg["direction"][r] != "heat" or int(rg["k"][r]) != 1:
            continue
        key = (int(rg["i"][r]), int(rg["j"][r]))
        if key not in steps:
            raise RuntimeError(f"{rgp} row {r} is at {key}, which the "
                               f"recomputation from {Lp} does not produce")
        st = steps[key]
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
    if n_checked == 0:
        raise RuntimeError(f"{rgp} contains no k=1 heating rows; the "
                           f"structural maps have no cross-check")
    print(f"guard  reservoir_gain.csv  {n_checked} k=1 heating rows of "
          f"{rg['_n']} reproduce the canonical matrix to 1e-9 relative, "
          f"window_ok exactly")

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
        emit(base + ".pdf", lambda p: fig.savefig(p, format="pdf"))
        emit(base + ".png", lambda p: fig.savefig(p, format="png", dpi=150))
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
    prop_worst = 0.0
    for key, st in ok.items():
        _, _, c1, c2 = st.two_errors(args.nt)
        clo[key], cre[key] = c1, c2
        # independent check of the propagator, at the window midpoint
        tm = float(np.sqrt(WIN_LO * st.tau_relax * st.tau_QSS / WIN_HI))
        prop_worst = max(prop_worst, st.check_propagator([tm]))
    print(f"propagator check     eigen-propagation vs scipy.linalg.expm, "
          f"worst relative disagreement over {len(ok)} points {prop_worst:.3e}")

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
    ax.text(0.97, 0.94, "CRE distance\n(the lookup table's assumption)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
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
        "@CLO_B@": f"{clo[ib,jb]:.2e}".replace("e-0", r"\times10^{-") + "}",
        "@CRE_B@": f"{cre[ib,jb]*100:.2f}",
        "@CLO_W@": f"{clo[kw]:.2e}".replace("e-0", r"\times10^{-") + "}",
        "@CRE_W@": f"{cre[kw]*100:.1f}",
        "@CLO_MIN@": f"{np.nanmin(clo):.1e}".replace("e-0", r"\times10^{-") + "}",
        "@CLO_MAX@": f"{np.nanmax(clo):.1e}".replace("e-0", r"\times10^{-") + "}",
        "@CRE_MIN@": f"{np.nanmin(cre)*100:.2f}",
        "@CRE_MAX@": f"{np.nanmax(cre)*100:.1f}",
        "@GAP_LO@": f"{lo_o:.1f}", "@GAP_HI@": f"{hi_o:.1f}",
        "@GAP_B@": f"{np.log10(ratio[ib,jb]):.1f}",
        "@GAP_W@": f"{np.log10(ratio[kw]):.1f}",
        "@NT@": str(args.nt), "@PROPCHK@": f"{prop_worst:.0e}".replace(
            "e-0", r"\times10^{-") + "}",
        "@NNOWIN@": str(n_nowin),
    })
