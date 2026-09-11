#!/usr/bin/env python3
"""
verify_partial_fe.py
=====================
Stamps, as a reproducible artifact, a skeptic-pass result that (until now)
existed only in scratch scripts: the negative partial correlation between
ln M and ln eps_plateau reported in Chapter 5 Section 5.9
(`validation/partial_correlation/partial_correlation_sweep.csv`,
`src/analysis/make_ch5_figures.py:partial()`) is not an artifact of the
particular polynomial control basis (quadratic-with-cross-term) that the
chapter happens to use.

THE QUESTION
------------
Chapter 5's headline residualises ln M and ln eps_plateau on a quadratic
polynomial in (ln Te, ln ne) and reports the residual (partial) correlation
as negative, reversing the raw (uncontrolled) positive correlation. A basis
choice is always suspect: maybe the sign flip is a feature of that specific
six-term basis and would disappear (or reverse again) under a different,
non-parametric way of removing the (Te, ne) trend.

This script residualises on eight different control designs, from a bare
linear model up to a fully saturated two-way (Te-row x ne-column) fixed-
effects model that removes ANY additive function of Te and ANY additive
function of ne, parametric or not -- the least assumption-laden control
short of removing the interaction term too (which would remove essentially
all remaining variation, since the grid has only 50x8 cells).

PREDICTION -- written before running any of the code below
------------------------------------------------------------
  1. The two-way fixed-effects partial correlation is AT LEAST AS NEGATIVE as
     the quadratic-control partial, in every direction (heat, cool, pooled).
     Rationale: if the quadratic surface under-fits the true (Te, ne) trend
     surface, some of the real trend leaks into the quadratic residuals and
     attenuates the partial toward zero; the fully saturated FE model cannot
     leak in this way, so it should show the same sign at least as strongly.
  2. The Te-row-only fixed-effects partial correlation is POSITIVE (it
     removes the Te trend, which Part 4 of verify_m_rank_test.py showed is
     positive and dominant, but leaves the ne dimension, and the reverse
     question there was uniformly co-monotonic).
  3. The within-Te-row permutation test on eps_plateau gives p < 0.001 for
     both the quadratic and the two-way FE control, in every direction.

REFUTING OBSERVATION, stated in advance
------------------------------------------
  A positive two-way FE partial correlation in ANY direction, OR a
  permutation p-value above 0.01 in any direction/control combination.
  Either would mean the quadratic-basis sign is not robust to using a
  non-parametric control, and Chapter 5's headline would need to be
  qualified as basis-dependent rather than reported flat.

DATA AND PROVENANCE
--------------------
Loads `validation/divertor_map/divertor_map.csv` (784 rows: 50 Te x 8 ne x
2 step directions). The CSV's own `i`, `j` index columns are checked against
`Te_grid_L.npy` / `ne_grid_L.npy` (via `cr_context.py` -- the grid is never
redefined here) to 1e-9 relative before anything downstream trusts them.

VARIABLES
---------
  x = ln M                    (M = tau_QSS / tau_relax)
  y = ln eps_plateau
  a = ln Te   -- PRE-step Te (the CSV's own `Te` column, Te_grid[i]; this is
                what the chapter's own sweep and `verify_m_rank_test.py` use)
  b = ln ne   -- (the CSV's own `ne` column, ne_grid[j])

A second variant of `a` is also reported: the POST-step Te, Te_grid[k] where
k is the grid index the +/-5% fractional step lands on (same construction as
`verify_divertor_map.py`: k = argmin|Te_grid - Te_grid[i]*(1+frac)| for heat,
(1-frac) for cool). The fractional step is READ from divertor_map.csv's own
header line ("fractional step ..."), not hardcoded, and is asserted to be
0.05 as the task expects.

Controls 6 (two-way FE) and 7 (Te-row FE only) are index-based (dummies on
the integer grid row/column i, j), not value-based, so they are IDENTICAL
under the pre-step and post-step `a` variant -- reported as such, not
silently computed once and mislabelled.

CONTROLS
--------
  1  linear                 [1, a, b]
  2  quadratic (chapter's)  [1, a, b, a^2, a*b, b^2]
  3  cubic                  all monomials in (a,b) to total degree 3 (10 terms)
  4  additive quadratic     [1, a, a^2, b, b^2]
  5  additive quartic       [1, a, a^2, a^3, a^4, b, b^2, b^3, b^4]
  6  two-way fixed effects  intercept + dummies for every Te-row i present
                             (less one, dropped for identifiability) +
                             dummies for every ne-column j present (less one)
  7  Te-row FE only         intercept + dummies for every Te-row i present
                             (less one)
  8  quadratic + direction  control 2 plus a heat/cool dummy (POOLED only)

Every design is solved twice: once via `numpy.linalg.lstsq` on a centred-and-
scaled design (intercept column left raw; every other column has its mean
subtracted and is divided by its own std, which does not change the column
space and therefore not the residuals, only the conditioning), and once via
QR (`numpy.linalg.qr`, reduced mode, triangular solve). The two coefficient
vectors are asserted to agree to 1e-8 absolute before either is trusted; the
design's condition number (after centring/scaling) is reported alongside.

STABILITY (Te>=2 eV scope only, controls 2 and 6, each direction)
--------------------------------------------------------------------
  - leave-one-ne-column-out (8 runs: drop each of the 8 density columns)
  - leave-one-Te-row-out (one run per distinct Te-row index present)
  - permutation: shuffle y (ln eps_plateau) WITHIN each Te row (i.e. among
    the ne columns that share a Te row), holding x, a, b, i, j fixed; 2000
    draws, numpy.random.default_rng(20260911); report null mean, null sd,
    and the fraction of draws with partial <= the observed (unpermuted)
    value. Uses the QR projector so each draw is a single matrix-vector
    product, not a fresh least-squares solve.

Every leave-one-out value is written to the detail CSV.

Report only. Writes to validation/partial_fe/ only with --write. Does not
modify any existing script or artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402

SEED = 20260911
N_PERM = 2000


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_header_lines(path: Path) -> list[str]:
    lines = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                lines.append(line.rstrip("\n"))
            else:
                break
    return lines


# --------------------------------------------------------------------------
# Design matrices
# --------------------------------------------------------------------------
CONTINUOUS_CONTROLS = [
    "linear", "quadratic", "cubic", "add_quadratic", "add_quartic",
]
FE_CONTROLS = ["twoway_fe", "te_fe"]
DIRECTION_ONLY_CONTROLS = ["quad_direction"]
ALL_CONTROLS = CONTINUOUS_CONTROLS + FE_CONTROLS + DIRECTION_ONLY_CONTROLS


def dummies(levels_col: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    """One dummy per distinct level present, dropping the first (lowest)
    level for identifiability. Levels are read from the DATA actually
    present in this scope/direction/leave-one-out subset, never from the
    full grid -- a dropped level simply does not get a column, which is the
    correct behaviour for a fixed-effects design fit on a subset."""
    levels = sorted(set(int(v) for v in levels_col))
    cols, names = [], []
    for lev in levels[1:]:
        cols.append((levels_col == lev).astype(float))
        names.append(f"lvl{lev}")
    return cols, names


def build_design(control: str, a: np.ndarray, b: np.ndarray,
                  i_idx: np.ndarray | None = None,
                  j_idx: np.ndarray | None = None,
                  dir_is_heat: np.ndarray | None = None
                  ) -> tuple[np.ndarray, list[str]]:
    n = len(a)
    ones = np.ones(n)
    if control == "linear":
        cols, names = [ones, a, b], ["1", "a", "b"]
    elif control == "quadratic":
        cols = [ones, a, b, a * a, a * b, b * b]
        names = ["1", "a", "b", "a2", "ab", "b2"]
    elif control == "cubic":
        cols = [ones, a, b, a * a, a * b, b * b,
                 a ** 3, a * a * b, a * b * b, b ** 3]
        names = ["1", "a", "b", "a2", "ab", "b2", "a3", "a2b", "ab2", "b3"]
    elif control == "add_quadratic":
        cols = [ones, a, a * a, b, b * b]
        names = ["1", "a", "a2", "b", "b2"]
    elif control == "add_quartic":
        cols = [ones, a, a ** 2, a ** 3, a ** 4, b, b ** 2, b ** 3, b ** 4]
        names = ["1", "a", "a2", "a3", "a4", "b", "b2", "b3", "b4"]
    elif control == "twoway_fe":
        if i_idx is None or j_idx is None:
            raise ValueError("twoway_fe needs i_idx and j_idx")
        ic, iname = dummies(i_idx)
        jc, jname = dummies(j_idx)
        cols = [ones] + ic + jc
        names = ["1"] + [f"i_{n_}" for n_ in iname] + [f"j_{n_}" for n_ in jname]
    elif control == "te_fe":
        if i_idx is None:
            raise ValueError("te_fe needs i_idx")
        ic, iname = dummies(i_idx)
        cols = [ones] + ic
        names = ["1"] + [f"i_{n_}" for n_ in iname]
    elif control == "quad_direction":
        if dir_is_heat is None:
            raise ValueError("quad_direction needs dir_is_heat")
        cols = [ones, a, b, a * a, a * b, b * b, dir_is_heat.astype(float)]
        names = ["1", "a", "b", "a2", "ab", "b2", "is_heat"]
    elif control == "quartic_full":
        # every monomial a^p * b^q with p+q <= 4 (15 terms including "1"):
        # the coordinator's "full quartic with cross terms" control.
        cols, names = [ones], ["1"]
        for deg in range(1, 5):
            for p in range(deg, -1, -1):
                q = deg - p
                cols.append((a ** p) * (b ** q))
                names.append(f"a{p}b{q}")
    else:
        raise ValueError(f"unknown control {control!r}")
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise RuntimeError(
            f"design for control={control!r} is column-rank-deficient: "
            f"shape {X.shape}, rank {np.linalg.matrix_rank(X)} -- cannot "
            f"identify all coefficients on this subset")
    return X, names


def center_scale(X: np.ndarray) -> np.ndarray:
    """Column 0 (intercept, all-ones) is left raw. Every other column is
    centred and scaled to unit std. This does not change span{X} (a centred
    column plus its mean times the intercept column reconstructs the
    original column, and the intercept is already in the span), so it does
    not change the least-squares residuals -- only the conditioning of the
    normal equations / QR factorisation."""
    Xcs = X.astype(float).copy()
    for c in range(1, X.shape[1]):
        col = Xcs[:, c]
        mu, sd = col.mean(), col.std(ddof=0)
        if sd == 0.0:
            raise RuntimeError(
                f"column {c} of the design is constant after centring "
                f"(no variance) -- degenerate design on this subset")
        Xcs[:, c] = (col - mu) / sd
    return Xcs


def solve_lstsq_and_qr(Xcs: np.ndarray, v: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Returns (coeffs_lstsq, coeffs_qr, max_abs_diff, Q). Q is the reduced
    QR factor, reused by the caller to build the projector for permutations
    without re-factorising."""
    coeffs_ls, *_ = np.linalg.lstsq(Xcs, v, rcond=None)
    Q, R = np.linalg.qr(Xcs, mode="reduced")
    coeffs_qr = np.linalg.solve(R, Q.T @ v)
    diff = float(np.max(np.abs(coeffs_ls - coeffs_qr)))
    return coeffs_ls, coeffs_qr, diff, Q


def partial_correlation(X: np.ndarray, x: np.ndarray, y: np.ndarray
                         ) -> dict:
    Xcs = center_scale(X)
    cond = float(np.linalg.cond(Xcs))
    cx_ls, cx_qr, dx, Qx = solve_lstsq_and_qr(Xcs, x)
    cy_ls, cy_qr, dy, Qy = solve_lstsq_and_qr(Xcs, y)
    max_diff = max(dx, dy)
    if max_diff >= 1e-8:
        raise AssertionError(
            f"lstsq vs QR solutions disagree by {max_diff:.3e} (>= 1e-8) "
            f"for a design of shape {X.shape}, cond={cond:.3e}")
    rx_ls = x - Xcs @ cx_ls
    ry_ls = y - Xcs @ cy_ls
    rx_qr = x - Xcs @ cx_qr
    ry_qr = y - Xcs @ cy_qr
    r_ls = float(np.corrcoef(rx_ls, ry_ls)[0, 1])
    r_qr = float(np.corrcoef(rx_qr, ry_qr)[0, 1])
    if abs(r_ls - r_qr) >= 1e-8:
        raise AssertionError(
            f"partial correlation from lstsq ({r_ls:.10f}) and QR "
            f"({r_qr:.10f}) disagree by {abs(r_ls - r_qr):.3e} (>= 1e-8)")
    return dict(partial=r_ls, cond=cond, max_coef_diff=max_diff,
                n=len(x), p=X.shape[1], resid_x=rx_ls, resid_y=ry_ls, Q=Qx)


def compute_r2(X: np.ndarray, v: np.ndarray) -> dict:
    """R^2 of a centred-and-scaled least-squares fit of v on X: fraction of
    variance in v explained by X's column space (here, the saturated
    two-way fixed effects). Solved and cross-checked the same way as
    partial_correlation: lstsq vs QR, asserted to agree to 1e-8."""
    Xcs = center_scale(X)
    c_ls, c_qr, diff, _ = solve_lstsq_and_qr(Xcs, v)
    if diff >= 1e-8:
        raise AssertionError(
            f"lstsq vs QR disagree by {diff:.3e} (>= 1e-8) in an R^2 fit, "
            f"design shape {X.shape}")
    resid = v - Xcs @ c_ls
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    return dict(r2=1.0 - ss_res / ss_tot, ss_res=ss_res, ss_tot=ss_tot,
                n=len(v), p=X.shape[1])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                     help="path to divertor_map.csv (default: pipeline location)")
    ap.add_argument("--write", action="store_true",
                     help="save outputs to validation/partial_fe/")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ctx = CRContext.load()
    root = ctx.root
    csv_path = Path(args.csv).resolve() if args.csv else (
        root / "validation" / "divertor_map" / "divertor_map.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"divertor map not found: {csv_path}")

    header_lines = read_header_lines(csv_path)
    sha_lines = [h for h in header_lines if "sha256" in h]
    if len(sha_lines) != 3:
        raise ValueError(
            f"expected 3 sha256 lines in {csv_path}'s header, found "
            f"{len(sha_lines)}: {header_lines}")

    frac_match = None
    for h in header_lines:
        m = re.search(r"fractional step\s+([0-9.]+)", h)
        if m:
            frac_match = float(m.group(1))
    if frac_match is None:
        raise ValueError(
            f"could not find a 'fractional step' value in {csv_path}'s "
            f"header -- refusing to hardcode 0.05")
    frac = frac_match
    if abs(frac - 0.05) > 1e-12:
        raise ValueError(
            f"divertor_map.csv's own header records fractional step "
            f"{frac}, not the 0.05 this task specifies -- STOP, do not "
            f"silently substitute 0.05")

    df = pd.read_csv(csv_path, comment="#")
    required = {"direction", "i", "j", "Te", "ne", "tau_QSS", "tau_relax",
                "M", "window_ok", "eps_plateau"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    if len(df) != 784:
        raise ValueError(f"expected 784 rows, found {len(df)} in {csv_path}")

    # ---- label i, j against the pipeline's own grids; never redefine them --
    Te_grid, ne_grid = ctx.te_grid, ctx.ne_grid
    i_idx_full, j_idx_full = df["i"].to_numpy(int), df["j"].to_numpy(int)
    te_check = Te_grid[i_idx_full]
    ne_check = ne_grid[j_idx_full]
    if not np.allclose(te_check, df["Te"].to_numpy(), rtol=1e-9):
        bad = int(np.argmax(np.abs(te_check - df["Te"].to_numpy())))
        raise ValueError(
            f"Te column disagrees with Te_grid_L.npy indexed by column i "
            f"(worst row {bad}: CSV Te={df['Te'].iloc[bad]!r}, "
            f"Te_grid[i]={te_check[bad]!r})")
    if not np.allclose(ne_check, df["ne"].to_numpy(), rtol=1e-9):
        bad = int(np.argmax(np.abs(ne_check - df["ne"].to_numpy())))
        raise ValueError(
            f"ne column disagrees with ne_grid_L.npy indexed by column j "
            f"(worst row {bad}: CSV ne={df['ne'].iloc[bad]!r}, "
            f"ne_grid[j]={ne_check[bad]!r})")

    # ---- post-step Te index k, exactly the verify_divertor_map.py rule ----
    k_idx_full = np.empty(len(df), dtype=int)
    for row_i in range(len(df)):
        sgn = +1 if df["direction"].iloc[row_i] == "heat" else -1
        te0 = df["Te"].iloc[row_i]
        k_idx_full[row_i] = int(np.argmin(np.abs(Te_grid - te0 * (1 + sgn * frac))))
    df["k"] = k_idx_full
    df["Te_post"] = Te_grid[k_idx_full]

    df["x"] = np.log(df["M"].to_numpy())
    df["y"] = np.log(df["eps_plateau"].to_numpy())
    df["a_pre"] = np.log(df["Te"].to_numpy())
    df["a_post"] = np.log(df["Te_post"].to_numpy())
    df["b"] = np.log(df["ne"].to_numpy())
    df["dir_is_heat"] = (df["direction"] == "heat")

    out_dir = args.out or (root / "validation" / "partial_fe")
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 78)
    say("PARTIAL CORRELATION UNDER FIXED-EFFECTS AND POLYNOMIAL CONTROLS")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"python interpreter: {sys.executable}")
    say(f"repo root: {root}")
    say(f"loaded: {csv_path}  ({len(df)} rows)")
    for h in header_lines:
        say(f"  (source header) {h}")
    say(f"Te grid: {len(Te_grid)} pts from cr_context (Te_grid_L.npy), "
        f"{Te_grid.min():.4f}-{Te_grid.max():.4f} eV")
    say(f"ne grid: {len(ne_grid)} pts from cr_context (ne_grid_L.npy), "
        f"{ne_grid.min():.3e}-{ne_grid.max():.3e} cm^-3")
    say("CSV i/j columns verified against Te_grid[i], ne_grid[j] to 1e-9 relative: OK")
    say(f"fractional step parsed from divertor_map.csv header: {frac} "
        f"(asserted == 0.05: OK)")
    say("")
    say("PREDICTION (from the docstring, written before computing):")
    say("  1. two-way FE partial <= quadratic partial (more negative or equal)")
    say("     in every direction.")
    say("  2. Te-row-only FE partial is POSITIVE in every direction.")
    say("  3. permutation p < 0.001 for quadratic and two-way FE, Te>=2 scope,")
    say("     every direction.")
    say("  REFUTING: a positive two-way FE partial anywhere, or any")
    say("  permutation p > 0.01.")
    say("=" * 78)

    # ---- min/max M, window_ok(680) vs all(784) -----------------------------
    say("\n" + "=" * 78)
    say("RAW M RANGE (sanity check against Chapter 5's grid-wide figures)")
    say("=" * 78)
    ok_all = df["window_ok"].to_numpy(bool)
    say(f"M over window_ok rows (n={int(ok_all.sum())}/680 expected): "
        f"{df['M'][ok_all].min():.6e} .. {df['M'][ok_all].max():.6e}")
    say(f"M over ALL rows (n={len(df)}/784 expected): "
        f"{df['M'].min():.6e} .. {df['M'].max():.6e}")
    if int(ok_all.sum()) != 680:
        say(f"  *** window_ok count is {int(ok_all.sum())}, not the "
            f"expected 680 -- reported, not corrected ***")

    # ---- raw Pearson / Spearman of (x, y), same scopes ---------------------
    say("\n" + "=" * 78)
    say("RAW (unresidualised) PEARSON / SPEARMAN OF (ln M, ln eps_plateau)")
    say("=" * 78)
    raw_rows = []

    def scope_frame(base: pd.DataFrame, scope: str) -> pd.DataFrame:
        if scope == "window_ok":
            return base[base["window_ok"]]
        if scope == "window_ok_Te2":
            return base[base["window_ok"] & (base["Te"] >= 2.0)]
        raise ValueError(scope)

    frames = {
        "heat": df[df.direction == "heat"],
        "cool": df[df.direction == "cool"],
        "pooled": df,
    }
    for dname, dsub in frames.items():
        for scope in ("window_ok", "window_ok_Te2"):
            ssub = scope_frame(dsub, scope)
            n = len(ssub)
            pr, pp = stats.pearsonr(ssub["x"], ssub["y"])
            sr, sp = stats.spearmanr(ssub["x"], ssub["y"])
            raw_rows.append(dict(direction=dname, scope=scope, n=n,
                                  pearson=pr, pearson_p=pp,
                                  spearman=sr, spearman_p=sp))
            say(f"[{dname:>6}] scope={scope:<14} n={n:<4d}  "
                f"pearson={pr:+.4f} (p={pp:.2e})  spearman={sr:+.4f} (p={sp:.2e})")

    # ---- main table: controls x scopes x directions x a-variant ------------
    say("\n" + "=" * 78)
    say("MAIN TABLE -- partial correlation, all controls / scopes / directions")
    say("(a_variant: pre = ln(pre-step Te, the CSV's Te column); "
        "post = ln(Te_grid[k]), the +/-5% post-step grid point)")
    say("Controls 6 (twoway_fe) and 7 (te_fe) are INDEX-based (dummies on i, "
        "j) and are therefore IDENTICAL under pre/post a_variant -- computed "
        "and listed under both tags for a uniform table, not recomputed.")
    say("=" * 78)

    main_rows = []
    failures = []

    def run_control(control, ssub, dname, scope, avariant):
        a = ssub[f"a_{avariant}"].to_numpy()
        b = ssub["b"].to_numpy()
        i_idx = ssub["i"].to_numpy(int)
        j_idx = ssub["j"].to_numpy(int)
        dir_is_heat = ssub["dir_is_heat"].to_numpy(bool)
        x = ssub["x"].to_numpy()
        y = ssub["y"].to_numpy()
        X, names = build_design(control, a, b, i_idx, j_idx, dir_is_heat)
        res = partial_correlation(X, x, y)
        row = dict(direction=dname, scope=scope, a_variant=avariant,
                   control=control, n=res["n"], n_params=res["p"],
                   partial=res["partial"], design_cond=res["cond"],
                   max_lstsq_qr_coef_diff=res["max_coef_diff"])
        return row

    for dname, dsub in frames.items():
        controls_here = list(CONTINUOUS_CONTROLS) + list(FE_CONTROLS)
        if dname == "pooled":
            controls_here = controls_here + DIRECTION_ONLY_CONTROLS
        for scope in ("window_ok", "window_ok_Te2"):
            ssub = scope_frame(dsub, scope)
            for avariant in ("pre", "post"):
                for control in controls_here:
                    try:
                        row = run_control(control, ssub, dname, scope, avariant)
                    except (AssertionError, RuntimeError, ValueError) as e:
                        failures.append(dict(
                            stage="main_table", direction=dname, scope=scope,
                            a_variant=avariant, control=control, error=str(e)))
                        say(f"*** FAILED [{dname}/{scope}/{avariant}/{control}]: {e}")
                        continue
                    main_rows.append(row)

    for r in main_rows:
        say(f"[{r['direction']:>6}] scope={r['scope']:<14} a={r['a_variant']:<4} "
            f"control={r['control']:<14} n={r['n']:<4d} p={r['n_params']:<3d} "
            f"partial={r['partial']:+.6f}  cond={r['design_cond']:.3e}  "
            f"lstsq-QR diff={r['max_lstsq_qr_coef_diff']:.2e}")

    # ---- reference reproduction --------------------------------------------
    say("\n" + "=" * 78)
    say("REFERENCE REPRODUCTION")
    say("=" * 78)

    def find_main(direction, scope, control, avariant="pre"):
        for r in main_rows:
            if (r["direction"] == direction and r["scope"] == scope
                    and r["control"] == control and r["a_variant"] == avariant):
                return r
        return None

    refs = [
        ("quadratic, pooled, window_ok_Te2", "pooled", "window_ok_Te2", "quadratic", -0.531),
        ("linear, pooled, window_ok_Te2", "pooled", "window_ok_Te2", "linear", 0.417),
        ("twoway_fe, pooled, window_ok_Te2", "pooled", "window_ok_Te2", "twoway_fe", -0.617),
        ("twoway_fe, heat, window_ok_Te2", "heat", "window_ok_Te2", "twoway_fe", -0.939),
        ("twoway_fe, cool, window_ok_Te2", "cool", "window_ok_Te2", "twoway_fe", -0.770),
    ]
    ref_rows = []
    for label, direction, scope, control, target in refs:
        r = find_main(direction, scope, control)
        if r is None:
            say(f"{label:<45} NOT COMPUTED (see failures)")
            ref_rows.append(dict(label=label, direction=direction, scope=scope,
                                  control=control, target=target,
                                  observed=float("nan"), abs_diff=float("nan")))
            continue
        diff = abs(r["partial"] - target)
        say(f"{label:<45} target={target:+.4f}  observed={r['partial']:+.6f}  "
            f"|diff|={diff:.4f}  n={r['n']}")
        ref_rows.append(dict(label=label, direction=direction, scope=scope,
                              control=control, target=target,
                              observed=r["partial"], abs_diff=diff))
    for direction in ("heat", "cool", "pooled"):
        r = find_main(direction, "window_ok_Te2", "te_fe")
        if r is None:
            say(f"te_fe, {direction}, window_ok_Te2 NOT COMPUTED (see failures)")
            ref_rows.append(dict(label=f"te_fe, {direction}, window_ok_Te2",
                                  direction=direction, scope="window_ok_Te2",
                                  control="te_fe", target=float("nan"),
                                  observed=float("nan"), abs_diff=float("nan")))
            continue
        in_band = 0.38 <= r["partial"] <= 0.59
        say(f"te_fe, {direction}, window_ok_Te2 target band [0.38,0.59]  "
            f"observed={r['partial']:+.6f}  {'IN BAND' if in_band else '*** OUT OF BAND ***'}")
        ref_rows.append(dict(label=f"te_fe, {direction}, window_ok_Te2",
                              direction=direction, scope="window_ok_Te2",
                              control="te_fe", target=float("nan"),
                              observed=r["partial"], abs_diff=float("nan"),
                              in_band_0p38_0p59=in_band))

    # ---- coordinator request: two-way FE R^2, and full quartic+cross control
    say("\n" + "=" * 78)
    say("TWO-WAY FE R^2 -- fraction of variance in ln M / ln eps_plateau")
    say("explained by the saturated Te-row x ne-column fixed effects")
    say("(Te>=2 eV & window_ok, a_variant=pre; FE is index-based so pre/post")
    say(" are identical by construction, as noted in the main table)")
    say("=" * 78)
    r2_rows = []
    for dname, dsub in frames.items():
        ssub = scope_frame(dsub, "window_ok_Te2")
        a = ssub["a_pre"].to_numpy()
        b = ssub["b"].to_numpy()
        i_idx = ssub["i"].to_numpy(int)
        j_idx = ssub["j"].to_numpy(int)
        X, _ = build_design("twoway_fe", a, b, i_idx, j_idx)
        x = ssub["x"].to_numpy()
        y = ssub["y"].to_numpy()
        try:
            r2x = compute_r2(X, x)
            r2y = compute_r2(X, y)
        except AssertionError as e:
            failures.append(dict(stage="twoway_fe_r2", direction=dname, error=str(e)))
            say(f"*** FAILED [{dname}/twoway_fe R^2]: {e}")
            continue
        r2_rows.append(dict(direction=dname, scope="window_ok_Te2",
                             control="twoway_fe", n=r2x["n"], n_params=r2x["p"],
                             r2_lnM=r2x["r2"], r2_ln_eps_plateau=r2y["r2"]))
        say(f"[{dname:>6}] n={r2x['n']:<4d} p={r2x['p']:<3d}  "
            f"R^2(ln M)={r2x['r2']:.4f}   R^2(ln eps_plateau)={r2y['r2']:.4f}")
    r2_targets = {"heat": (0.998, 0.992), "pooled": (0.987, 0.983)}
    say("  cross-check against the coordinator's independent recomputation:")
    for dname, (tM, tE) in r2_targets.items():
        row = next((r for r in r2_rows if r["direction"] == dname), None)
        if row is None:
            say(f"    [{dname}] NOT COMPUTED (see failures)")
            continue
        say(f"    [{dname}] target R^2(ln M)={tM:.3f} observed={row['r2_lnM']:.4f} "
            f"|diff|={abs(row['r2_lnM']-tM):.4f};  target R^2(eps)={tE:.3f} "
            f"observed={row['r2_ln_eps_plateau']:.4f} "
            f"|diff|={abs(row['r2_ln_eps_plateau']-tE):.4f}")

    say("\n" + "=" * 78)
    say("FULL QUARTIC WITH CROSS TERMS -- all a^p b^q, p+q<=4 (15 terms)")
    say("(Te>=2 eV & window_ok, a_variant=pre, the chapter's sweep variant)")
    say("=" * 78)
    quartic_full_rows = []
    for dname, dsub in frames.items():
        ssub = scope_frame(dsub, "window_ok_Te2")
        try:
            row = run_control("quartic_full", ssub, dname, "window_ok_Te2", "pre")
        except (AssertionError, RuntimeError, ValueError) as e:
            failures.append(dict(stage="quartic_full", direction=dname, error=str(e)))
            say(f"*** FAILED [{dname}/quartic_full]: {e}")
            continue
        quartic_full_rows.append(row)
        say(f"[{dname:>6}] n={row['n']:<4d} p={row['n_params']:<3d}  "
            f"partial={row['partial']:+.6f}  cond={row['design_cond']:.3e}  "
            f"lstsq-QR diff={row['max_lstsq_qr_coef_diff']:.2e}")
    qf_targets = {"pooled": -0.524, "heat": -0.281, "cool": -0.335}
    say("  cross-check against the coordinator's independent recomputation:")
    for dname, target in qf_targets.items():
        row = next((r for r in quartic_full_rows if r["direction"] == dname), None)
        if row is None:
            say(f"    [{dname}] NOT COMPUTED (see failures)")
            continue
        diff = abs(row["partial"] - target)
        say(f"    [{dname}] target={target:+.3f} observed={row['partial']:+.6f} "
            f"|diff|={diff:.4f}")

    # ---- stability: leave-one-out, Te>=2 scope, quadratic & twoway_fe ------
    say("\n" + "=" * 78)
    say("STABILITY -- Te>=2 eV & window_ok scope, controls: quadratic, twoway_fe")
    say("a_variant = pre (the chapter's sweep variant) throughout")
    say("=" * 78)

    loo_rows = []
    for dname, dsub in frames.items():
        ssub = scope_frame(dsub, "window_ok_Te2")
        for control in ("quadratic", "twoway_fe"):
            try:
                full_row = run_control(control, ssub, dname, "window_ok_Te2", "pre")
            except (AssertionError, RuntimeError, ValueError) as e:
                failures.append(dict(stage="stability_baseline", direction=dname,
                                      control=control, error=str(e)))
                say(f"*** FAILED baseline [{dname}/{control}]: {e}")
                continue
            observed = full_row["partial"]
            say(f"\n[{dname:>6}] control={control}  observed partial={observed:+.6f} "
                f"(n={full_row['n']})")

            # leave-one-ne-column-out
            j_levels = sorted(ssub["j"].unique())
            say(f"  leave-one-ne-column-out ({len(j_levels)} runs):")
            for j0 in j_levels:
                sub2 = ssub[ssub["j"] != j0]
                try:
                    r2 = run_control(control, sub2, dname, "window_ok_Te2", "pre")
                    val, err = r2["partial"], ""
                except (AssertionError, RuntimeError, ValueError) as e:
                    val, err = float("nan"), str(e)
                    failures.append(dict(stage="loo_density", direction=dname,
                                          control=control, dropped_j=int(j0), error=err))
                loo_rows.append(dict(test="loo_density_column", direction=dname,
                                      control=control, dropped_level=int(j0),
                                      n=len(sub2), partial=val, baseline_partial=observed,
                                      error=err))
                say(f"    drop j={j0} (ne={ne_grid[j0]:.3e}): n={len(sub2)}  "
                    f"partial={val:+.6f}" + (f"  ERROR: {err}" if err else ""))

            # leave-one-Te-row-out
            i_levels = sorted(ssub["i"].unique())
            say(f"  leave-one-Te-row-out ({len(i_levels)} runs):")
            for i0 in i_levels:
                sub2 = ssub[ssub["i"] != i0]
                try:
                    r2 = run_control(control, sub2, dname, "window_ok_Te2", "pre")
                    val, err = r2["partial"], ""
                except (AssertionError, RuntimeError, ValueError) as e:
                    val, err = float("nan"), str(e)
                    failures.append(dict(stage="loo_te_row", direction=dname,
                                          control=control, dropped_i=int(i0), error=err))
                loo_rows.append(dict(test="loo_te_row", direction=dname,
                                      control=control, dropped_level=int(i0),
                                      n=len(sub2), partial=val, baseline_partial=observed,
                                      error=err))
            vals = np.array([r["partial"] for r in loo_rows
                              if r["test"] == "loo_te_row" and r["direction"] == dname
                              and r["control"] == control and not np.isnan(r["partial"])])
            say(f"    {len(i_levels)} rows dropped one at a time: "
                f"partial range {vals.min():+.6f} .. {vals.max():+.6f}, "
                f"median {np.median(vals):+.6f}")

    # ---- permutation test ---------------------------------------------------
    say("\n" + "=" * 78)
    say(f"PERMUTATION TEST -- shuffle y within each Te row, {N_PERM} draws, "
        f"seed={SEED}, Te>=2 eV & window_ok scope")
    say("=" * 78)
    perm_summary = []
    for dname, dsub in frames.items():
        ssub = scope_frame(dsub, "window_ok_Te2")
        i_idx = ssub["i"].to_numpy(int)
        groups = {i0: np.where(i_idx == i0)[0] for i0 in np.unique(i_idx)}
        for control in ("quadratic", "twoway_fe"):
            a = ssub["a_pre"].to_numpy()
            b = ssub["b"].to_numpy()
            j_idx = ssub["j"].to_numpy(int)
            dir_is_heat = ssub["dir_is_heat"].to_numpy(bool)
            x = ssub["x"].to_numpy()
            y = ssub["y"].to_numpy()
            X, _ = build_design(control, a, b, i_idx, j_idx, dir_is_heat)
            Xcs = center_scale(X)
            cond = float(np.linalg.cond(Xcs))
            cx_ls, cx_qr, dx, Qx = solve_lstsq_and_qr(Xcs, x)
            if dx >= 1e-8:
                raise AssertionError(
                    f"permutation baseline lstsq/QR disagree by {dx:.3e}")
            rx = x - Xcs @ cx_ls
            cy_ls, cy_qr, dy, Qy = solve_lstsq_and_qr(Xcs, y)
            if dy >= 1e-8:
                raise AssertionError(
                    f"permutation baseline lstsq/QR disagree by {dy:.3e}")
            ry = y - Xcs @ cy_ls
            observed = float(np.corrcoef(rx, ry)[0, 1])

            rng = np.random.default_rng(SEED)
            null_vals = np.empty(N_PERM)
            y_perm = y.copy()
            for draw in range(N_PERM):
                for i0, idxs in groups.items():
                    y_perm[idxs] = rng.permutation(y[idxs])
                ry_perm = y_perm - Qy @ (Qy.T @ y_perm)
                null_vals[draw] = np.corrcoef(rx, ry_perm)[0, 1]

            null_mean, null_sd = float(null_vals.mean()), float(null_vals.std(ddof=1))
            frac_le = float((null_vals <= observed).mean())
            say(f"[{dname:>6}] control={control:<10} observed={observed:+.6f}  "
                f"null mean={null_mean:+.6f}  null sd={null_sd:.6f}  "
                f"frac(null<=observed)={frac_le:.5f}  "
                f"{'p<0.001 HELD' if frac_le < 0.001 else ('p<0.01 (weaker than predicted)' if frac_le < 0.01 else '*** p>0.01: REFUTATION CRITERION MET ***')}")
            perm_summary.append(dict(
                direction=dname, control=control, n=len(x), n_draws=N_PERM,
                seed=SEED, observed_partial=observed, null_mean=null_mean,
                null_sd=null_sd, frac_null_le_observed=frac_le))

    # ---- verdict --------------------------------------------------------------
    say("\n" + "=" * 78)
    say("VERDICT ON THE THREE PREDICTIONS")
    say("=" * 78)
    p1_holds = True
    for dname in ("heat", "cool", "pooled"):
        rq = find_main(dname, "window_ok_Te2", "quadratic")
        rf = find_main(dname, "window_ok_Te2", "twoway_fe")
        if rq is None or rf is None:
            p1_holds = False
            say(f"  [{dname}] cannot check (missing computation)")
            continue
        ok = rf["partial"] <= rq["partial"] + 1e-9
        p1_holds &= ok
        say(f"  [{dname}] quadratic={rq['partial']:+.6f}  twoway_fe={rf['partial']:+.6f}  "
            f"twoway_fe <= quadratic: {'OK' if ok else '*** VIOLATED ***'}")
    p2_holds = True
    for dname in ("heat", "cool", "pooled"):
        rf = find_main(dname, "window_ok_Te2", "te_fe")
        if rf is None:
            p2_holds = False
            continue
        ok = rf["partial"] > 0
        p2_holds &= ok
        say(f"  [{dname}] te_fe={rf['partial']:+.6f}  positive: {'OK' if ok else '*** VIOLATED ***'}")
    p3_holds = all(r["frac_null_le_observed"] < 0.001 for r in perm_summary)
    say(f"  permutation p<0.001 in every direction/control: "
        f"{'OK' if p3_holds else '*** VIOLATED (see table above) ***'}")
    refuted = (not p1_holds) or any(
        find_main(d, "window_ok_Te2", "twoway_fe") is not None
        and find_main(d, "window_ok_Te2", "twoway_fe")["partial"] > 0
        for d in ("heat", "cool", "pooled")
    ) or any(r["frac_null_le_observed"] > 0.01 for r in perm_summary)
    say(f"\nOVERALL: {'REFUTED' if refuted else 'NOT REFUTED'} by the stated criterion "
        f"(positive two-way FE partial anywhere, or any permutation p>0.01)")

    if failures:
        say("\n" + "=" * 78)
        say(f"*** {len(failures)} STAGE(S) FAILED / RAISED -- reported, not silently skipped ***")
        say("=" * 78)
        for f in failures:
            say(f"  {f}")

    # ---- write outputs -------------------------------------------------------
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)
        header = [
            f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}",
            f"# python interpreter: {sys.executable}",
        ] + sha_lines + [
            f"# source csv: {csv_path.relative_to(root)}  sha256 {sha256(csv_path)}",
            f"# fractional step (parsed from source header): {frac}",
            f"# permutation seed {SEED}, draws {N_PERM}",
        ]

        detail_path = out_dir / "partial_fe.csv"
        detail_frames = []
        for r in main_rows:
            r2 = dict(r)
            r2["test"] = "main_table"
            detail_frames.append(r2)
        for r in loo_rows:
            detail_frames.append(r)
        det_df = pd.DataFrame(detail_frames)
        with open(detail_path, "w") as fh:
            fh.write("\n".join(header) + "\n")
        det_df.to_csv(detail_path, mode="a", index=False)

        summary_path = out_dir / "partial_fe_summary.csv"
        summary_frames = []
        for r in ref_rows:
            r2 = dict(r)
            r2["test"] = "reference_reproduction"
            summary_frames.append(r2)
        for r in raw_rows:
            r2 = dict(r)
            r2["test"] = "raw_pearson_spearman"
            summary_frames.append(r2)
        for r in perm_summary:
            r2 = dict(r)
            r2["test"] = "permutation"
            summary_frames.append(r2)
        for r in r2_rows:
            r2d = dict(r)
            r2d["test"] = "twoway_fe_r2"
            summary_frames.append(r2d)
        for r in quartic_full_rows:
            r2d = dict(r)
            r2d["test"] = "quartic_full"
            summary_frames.append(r2d)
        summary_frames.append(dict(
            test="M_range", direction="pooled", scope="window_ok",
            n=int(ok_all.sum()), observed=None,
            M_min=float(df['M'][ok_all].min()), M_max=float(df['M'][ok_all].max())))
        summary_frames.append(dict(
            test="M_range", direction="pooled", scope="all",
            n=len(df), observed=None,
            M_min=float(df['M'].min()), M_max=float(df['M'].max())))
        sum_df = pd.DataFrame(summary_frames)
        with open(summary_path, "w") as fh:
            fh.write("\n".join(header) + "\n")
        sum_df.to_csv(summary_path, mode="a", index=False)

        txt_path = out_dir / "partial_fe.txt"
        with open(txt_path, "w") as fh:
            fh.write("\n".join(header) + "\n")
            fh.write("\n".join(lines) + "\n")

        print(f"\nwrote {detail_path}")
        print(f"wrote {summary_path}")
        print(f"wrote {txt_path}")
    else:
        print("\n(--write not given; nothing saved)")


if __name__ == "__main__":
    main()
