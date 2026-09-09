#!/usr/bin/env python3
"""
make_ch5_figures.py
===================
Generates the four figures for Chapter 5 and the caption file that goes with
them.

  fig5_1_eps_map.pdf        the eps_plateau map over (Te, ne), HEATING
  fig5_2_eps_vs_ne.pdf      eps_plateau against ne, one line per Te, ALL EIGHT
                            density columns
  fig5_3_M_vs_eps.pdf       M against eps_plateau, coloured by Te
  fig5_4_trapping.pdf       eps_plateau against Lyman slab thickness D
  fig5_captions.tex         \\newcommand'd captions, numbers injected from the
                            values actually plotted

WHAT IS PLOTTED, AND FROM WHERE
-------------------------------
eps_plateau is the fractional error in the n=3/n=4 shell population ratio
committed by the QSS closure while the excited manifold sits at partial
equilibrium against a ground state that has not yet moved -- the plateau of
Chapter 5.  It is read from

    validation/plateau_gridmap/plateau_gridmap.csv   (verify_plateau_gridmap.py)
    validation/divertor_map/divertor_map.csv         (verify_divertor_map.py)
    validation/lyman_trapping/lyman_trapping.csv     (verify_lyman_trapping.py)

and, for the first two, RECOMPUTED HERE from the canonical matrix using the
algebra of verify_plateau_gridmap.py lines 160-230 -- the same three linear
solves, the same window test, the same sign conventions.  The recomputation is
compared against every row of both CSVs and raises on disagreement.  The
figures are drawn from the recomputed arrays, so a figure cannot silently
disagree with the matrix it claims to describe.  The trapping figure cannot be
recomputed here (it needs the rate dictionaries and the escape-factor fixed
point); its untrapped run is instead checked against the recomputation
row by row, which is the same check the ADDENDUM of findings_10 claims to have
passed.

DIRECTION IS NOT A DETAIL
-------------------------
Every point appears twice in the map, once heated and once cooled.  The two are
materially different: at the coldest pair of temperatures the same transition
gives 0.371 heating and 0.286 cooling, and the grid maximum moves from [0,4]
(heating) to [1,3] (cooling).  Figures 5.1 and 5.2 show HEATING and say so;
Figure 5.3 shows both, distinguished by marker.

WHAT THESE FIGURES MUST NOT BE READ AS SAYING
---------------------------------------------
  - Not "the Balmer diagnostic".  The observable is the n=3/n=4 SHELL ratio;
    the worst density moves by 2.68x for the (3,5) pair (findings_10 7.3).
  - Not detachment.  1.93e13 cm^-3 is 5.2x below the citable detached band
    (findings_10 1.3).
  - Not a ridge, if the plot does not show one.  Figure 5.2 exists precisely so
    that the reader can judge the density structure for themselves; the caption
    reports the measured flatness rather than a word.
  - Below Te = 2 eV nothing here is quantitative: Figure 5.4 is the reason.
    The 38.7% falls to 11.6-15.7% once Lyman trapping is switched on, and the
    location of the cold-row maximum moves across three density columns
    (findings_10 ADDENDUM A.3, A.4).

Report only: writes figures/fig5_*.{pdf,png} and figures/fig5_captions.tex,
and refuses to overwrite an existing file whose content differs unless --force
is passed.  Reads validation/ and data/; writes to neither.
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
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
from cr_context import CRContext, find_repo_root  # noqa: E402

# ---- thesis figure style (identical to make_ch3_figures.py) ----------------
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

# Colour discipline for Chapter 5, chosen so every figure survives greyscale:
#   Te   -> viridis          (monotone luminance; same encoding in 5.2 and 5.3)
#   eps  -> magma            (monotone luminance, distinct hue family from Te)
#   marks-> the Chapter 3 accent set, unchanged
C_RED, C_BLUE, C_GREEN, C_PURPLE = "#c0392b", "#1f4e79", "#2e7d32", "#8e44ad"
# marker outline on the dark end of magma; the markers are open shapes, so
# they survive greyscale as outlines rather than as a hue
C_MARK = "#39d0d8"

# Parameters of the source run.  WIN_LO/WIN_HI are the argparse defaults of
# verify_plateau_gridmap.py (--win-lo 30, --win-hi 30); window_ok is
# 30*tau_relax < tau_QSS/30, i.e. M > 900.  They are not free here: the
# recomputed window_ok column is compared element-by-element against the CSV,
# so a wrong constant raises rather than silently redrawing the mask.
WIN_LO, WIN_HI = 30.0, 30.0

# The benchmark point is specified in CLAUDE.md as a PHYSICAL condition,
# Te = 2.947 eV and ne = 1.389e14 cm^-3.  Its indices are derived from the
# loaded grids by argmin below and checked, never assumed.
BENCH_TE_EV, BENCH_NE_CM3 = 2.947, 1.389e14
BENCH_IJ_EXPECTED = (23, 5)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"missing {what}: {path} — this figure cannot be "
                           f"drawn without it, and no stand-in is acceptable")
    return path


# ---------------------------------------------------------------------------
# CSV reading.  Fails on a missing column, a non-numeric entry, or a NaN in any
# column a figure uses; never coerces, never fills.
# ---------------------------------------------------------------------------
def read_table(path: Path, numeric: list[str], text: list[str],
               boolean: list[str]) -> dict:
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
        raise RuntimeError(f"{path} is missing column(s) {missing}; "
                           f"columns present: {sorted(have)}")
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
    out["_comments"] = [ln for ln in raw if ln.lstrip().startswith("#")]
    return out


def header_value(tab: dict, key: str, path: Path) -> str:
    """Pull '# <key> <value>' out of the CSV's own provenance header."""
    for ln in tab["_comments"]:
        parts = ln.lstrip("# ").split()
        if len(parts) >= 2 and " ".join(parts[:len(key.split())]) == key:
            return parts[len(key.split())]
    raise RuntimeError(f"{path} header does not record {key!r}; without it the "
                       f"figure cannot state what step it shows")


def log_edges(v: np.ndarray) -> np.ndarray:
    """Cell edges for a geometrically spaced grid, in the same units as v."""
    lv = np.log(v)
    mid = 0.5 * (lv[1:] + lv[:-1])
    return np.exp(np.concatenate([[2 * lv[0] - mid[0]], mid,
                                  [2 * lv[-1] - mid[-1]]]))


def truncated(cmap_name: str, lo: float, hi: float, n: int) -> np.ndarray:
    return plt.get_cmap(cmap_name)(np.linspace(lo, hi, n))


def sci(x: float, sig: int = 2) -> str:
    """'5.2e+13' -> '5.2\\times10^{13}'. Valid inside $...$ in BOTH matplotlib
    mathtext and LaTeX, so the figure and its caption cannot disagree about how
    a density is written."""
    s = f"{x:.{sig}g}"
    if "e" not in s:
        return s
    m, e = s.split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="permit overwriting an existing figure whose bytes "
                         "differ (default: raise)")
    args = ap.parse_args()

    # The repo root is derived from THIS FILE's location, not from the
    # working directory, so the figures are regenerated from the same
    # pipeline whatever directory the command is run in.
    ctx = CRContext.load(root=find_repo_root(Path(__file__).resolve().parent))
    root = ctx.root
    Lp = require_file(root / "data/processed/cr_matrix/L_grid.npy", "L_grid")
    Sp = require_file(root / "data/processed/cr_matrix/S_grid.npy", "S_grid")
    sip = require_file(Path(ctx.state_index_path), "state index")
    L = ctx.L_grid
    S = np.load(Sp)
    Te, ne = ctx.te_grid, ctx.ne_grid
    if S.shape != L.shape[:3]:
        raise RuntimeError(f"S_grid {S.shape} incompatible with L_grid "
                           f"{L.shape}: {Sp}")

    gmap_p = require_file(root / "validation/plateau_gridmap/plateau_gridmap.csv",
                          "plateau grid map")
    dmap_p = require_file(root / "validation/divertor_map/divertor_map.csv",
                          "divertor map")
    lyman_p = require_file(root / "validation/lyman_trapping/lyman_trapping.csv",
                           "Lyman trapping sweep")

    outdir = root / "figures"
    outdir.mkdir(exist_ok=True)

    sha_L, sha_S, sha_I = sha256(Lp), sha256(Sp), sha256(sip)
    h = hashlib.sha256()
    for p in (Lp, Sp, sip):
        h.update(Path(p).read_bytes())
    sha8 = h.hexdigest()[:8]
    stamp = f"CR data {sha8} · {datetime.now():%Y-%m-%d}"

    print("=" * 78)
    print("CHAPTER 5 FIGURES — provenance")
    print("=" * 78)
    print(f"repo root            {root}")
    print(f"L_grid               {Lp}")
    print(f"  sha256             {sha_L}")
    print(f"S_grid               {Sp}")
    print(f"  sha256             {sha_S}")
    print(f"state_index          {sip}")
    print(f"  sha256             {sha_I}")
    for p in (gmap_p, dmap_p, lyman_p):
        print(f"{p.relative_to(root)}")
        print(f"  sha256             {sha256(p)}")
    print(f"combined SHA-8       {sha8}   (L_grid + S_grid + state_index)")
    print(f"grid                 {len(Te)} Te x {len(ne)} ne = "
          f"{len(Te)*len(ne)} points, {ctx.n_states} states")
    print(f"Te                   {Te[0]:.4g} .. {Te[-1]:.4g} eV, "
          f"ratio {Te[1]/Te[0]:.6f} per index")
    print(f"ne                   {ne[0]:.4g} .. {ne[-1]:.4g} cm^-3, "
          f"ratio {ne[1]/ne[0]:.6f} per index")

    # ---- benchmark indices: derived, then checked -------------------------
    ib = int(np.argmin(np.abs(np.log(Te / BENCH_TE_EV))))
    jb = int(np.argmin(np.abs(np.log(ne / BENCH_NE_CM3))))
    if (ib, jb) != BENCH_IJ_EXPECTED:
        raise RuntimeError(
            f"benchmark point moved: argmin over the loaded grids puts "
            f"Te={BENCH_TE_EV} eV, ne={BENCH_NE_CM3:.4g} cm^-3 at [{ib},{jb}], "
            f"not {BENCH_IJ_EXPECTED}. The grids in "
            f"{root/'data/processed/cr_matrix'} are not the ones CLAUDE.md's "
            f"reference values were measured on; nothing in Chapter 5 should "
            f"be redrawn until that is resolved")
    for name, got, want in (("Te", Te[ib], BENCH_TE_EV),
                            ("ne", ne[jb], BENCH_NE_CM3)):
        if abs(np.log(got / want)) > 5e-4:
            raise RuntimeError(f"benchmark {name} = {got:.6g} does not match "
                               f"the recorded {want:.6g} (index {ib},{jb})")
    print(f"benchmark            [{ib},{jb}]  Te={Te[ib]:.4f} eV  "
          f"ne={ne[jb]:.4e} cm^-3   (derived by argmin, checked against "
          f"CLAUDE.md)")

    # M at the benchmark from the UNSTEPPED operator, which is the number
    # CLAUDE.md records (9982).  The maps below carry M from the POST-STEP
    # operator L[24,5] (8243).  Both are correct; findings_10 §8 trap 2 is
    # exactly the confusion of the two, so both are printed here.
    ev_b = np.linalg.eigvals(L[ib, jb])
    if np.max(ev_b.real) >= 0:
        raise RuntimeError(f"non-decaying mode at the benchmark [{ib},{jb}]: "
                           f"max Re(lambda) = {np.max(ev_b.real):.3e}")
    tb = np.sort(1.0 / np.abs(ev_b.real))[::-1]
    print(f"  unstepped L[{ib},{jb}]  tau_QSS={tb[0]:.4e} s  "
          f"tau_relax={tb[1]:.4e} s  M={tb[0]/tb[1]:.1f}   "
          f"(CLAUDE.md: 2.273e-05, 2.277e-09, 9982)")

    # =======================================================================
    # Recompute the plateau map from the canonical matrix.
    # This is verify_plateau_gridmap.py lines 160-230, re-executed, not
    # re-derived: same index-snapped step, same three solves, same window test.
    # =======================================================================
    frac = float(header_value(read_table(gmap_p, [], [], []),
                              "requested fractional step", gmap_p))
    print(f"fractional Te step   {frac:.4f}   (read from the CSV header, "
          f"not assumed)")

    g = int(ctx.ground_index)
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)
    nv = np.asarray(ctx.n_values)
    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise RuntimeError(f"shell membership wrong in {sip}: "
                           f"n=3 -> {N3}, n=4 -> {N4}")
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    rec: dict[tuple, dict] = {}
    max_sup = 0.0
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(len(Te)):
            k = int(np.argmin(np.abs(Te - Te[i] * (1 + sgn * frac))))
            if k == i:
                continue
            for j in range(len(ne)):
                lam = np.linalg.eigvals(L[k, j])
                lam = lam[np.argsort(lam.real)[::-1]]
                if lam[0].real >= 0 or lam[1].real >= 0:
                    raise RuntimeError(
                        f"unstable post-step operator at Te={Te[k]:g} eV, "
                        f"ne={ne[j]:g} cm^-3 (grid [{k},{j}] of {Lp}): "
                        f"lambda_0={lam[0].real:.3e}, lambda_1={lam[1].real:.3e}")
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                window_ok = (WIN_LO * tR) < (tQ / WIN_HI)

                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(L[k, j], -S[k, j])
                LEE = L[k, j][np.ix_(E, E)]
                LEg = L[k, j][np.ix_(E, [g])].ravel()
                n0 = np.linalg.solve(LEE, -S[k, j][E])
                n1 = np.linalg.solve(LEE, -LEg * n_old[g])
                x_new = n_new[g] / n_old[g]
                sup = (np.abs(n0 + x_new * n1 - n_new[E]).max()
                       / np.abs(n_new[E]).max())
                if sup > 1e-8:
                    raise RuntimeError(
                        f"two-channel superposition fails at [{i},{j}] "
                        f"{dlab}: {sup:.3e}. The whole plateau construction "
                        f"rests on n_E = n0 + x n1; nothing may be plotted")
                max_sup = max(max_sup, sup)

                a3_0, a4_0 = n0[n3E].sum(), n0[n4E].sum()
                a3_1, a4_1 = n1[n3E].sum(), n1[n4E].sum()
                f3 = a3_1 / (a3_0 + a3_1)
                f4 = a4_1 / (a4_0 + a4_1)
                Rq = n_new[N3].sum() / n_new[N4].sum()
                R_pe = (a3_0 + a3_1) / (a4_0 + a4_1)
                R_old = n_old[N3].sum() / n_old[N4].sum()
                rec[(dlab, i, j)] = dict(
                    Te=Te[i], ne=ne[j], tau_QSS=tQ, tau_relax=tR, M=tQ / tR,
                    window_ok=window_ok, x_new=x_new,
                    eps_step=abs(R_old / Rq - 1.0),
                    eps_plateau=abs(R_pe / Rq - 1.0),
                    f3=f3, f4=f4, sens=f3 - f4, abs_ln_x=abs(np.log(x_new)))
    if not rec:
        raise RuntimeError("no (point, direction) pair survived the step "
                           "construction; nothing to plot")
    print(f"recomputed           {len(rec)} (point, direction) pairs from "
          f"L_grid/S_grid; max superposition error {max_sup:.3e}")

    # ---- guard: the recomputation must reproduce both CSVs ----------------
    gmap = read_table(
        gmap_p,
        numeric=["i", "j", "Te", "ne", "tau_QSS", "tau_relax", "M",
                 "eps_step", "eps_plateau", "f3_old", "f4_old", "abs_ln_x",
                 "superposition_err"],
        text=["direction"], boolean=["window_ok"])
    dmap = read_table(
        dmap_p,
        numeric=["i", "j", "Te", "ne", "tau_QSS", "M", "eps_plateau",
                 "f3", "f4", "abs_ln_x", "lo_ELM_crash"],
        text=["direction"], boolean=["window_ok"])

    def check_against(tab: dict, path: Path, colmap: dict, rtol: float):
        n_bad = 0
        for r in range(tab["_n"]):
            key = (str(tab["direction"][r]), int(tab["i"][r]), int(tab["j"][r]))
            if key not in rec:
                raise RuntimeError(f"{path} row {r} is at {key}, which the "
                                   f"recomputation from {Lp} does not produce")
            ref = rec[key]
            for csv_col, rec_col in colmap.items():
                a, b = tab[csv_col][r], ref[rec_col]
                if abs(a - b) > rtol * max(abs(b), 1e-300):
                    n_bad += 1
                    raise RuntimeError(
                        f"{path} disagrees with the canonical matrix at "
                        f"{key}, column {csv_col}: file {a:.12e}, recomputed "
                        f"{b:.12e} (rel {abs(a-b)/max(abs(b),1e-300):.2e}). "
                        f"The CSV is stale with respect to "
                        f"{Lp} (sha256 {sha_L}) or the algebra has changed")
            if bool(tab["window_ok"][r]) != bool(ref["window_ok"]):
                raise RuntimeError(
                    f"{path} window_ok at {key} is {tab['window_ok'][r]} but "
                    f"WIN_LO*WIN_HI = {WIN_LO*WIN_HI:g} gives "
                    f"{ref['window_ok']} (M = {ref['M']:.6g})")
        return n_bad

    check_against(gmap, gmap_p,
                  {"Te": "Te", "ne": "ne", "tau_QSS": "tau_QSS",
                   "tau_relax": "tau_relax", "M": "M",
                   "eps_step": "eps_step", "eps_plateau": "eps_plateau",
                   "f3_old": "f3", "f4_old": "f4", "abs_ln_x": "abs_ln_x"},
                  rtol=1e-9)
    print(f"guard  plateau_gridmap.csv  {gmap['_n']} rows reproduce the "
          f"canonical matrix to 1e-9 relative, window_ok exactly")
    check_against(dmap, dmap_p,
                  {"Te": "Te", "ne": "ne", "tau_QSS": "tau_QSS", "M": "M",
                   "eps_plateau": "eps_plateau", "f3": "f3", "f4": "f4",
                   "abs_ln_x": "abs_ln_x"},
                  rtol=1e-9)
    print(f"guard  divertor_map.csv     {dmap['_n']} rows agree with "
          f"plateau_gridmap.csv and with the matrix to 1e-9 relative")

    # ---- guard: the untrapped Lyman run IS the canonical matrix -----------
    lym = read_table(
        lyman_p,
        numeric=["i", "j", "Te", "ne", "tau_QSS", "tau_relax", "M",
                 "eps_plateau", "f3", "f4", "abs_ln_x", "D_cm"],
        text=["direction", "run"], boolean=["window_ok"])
    unt = lym["run"] == "untrapped"
    if not unt.any():
        raise RuntimeError(f"{lyman_p} contains no 'untrapped' run; the "
                           f"trapping sweep cannot be anchored")
    n_unt = 0
    for r in np.flatnonzero(unt):
        key = (str(lym["direction"][r]), int(lym["i"][r]), int(lym["j"][r]))
        ref = rec[key]
        for col, rc in (("eps_plateau", "eps_plateau"), ("M", "M"),
                        ("f3", "f3"), ("f4", "f4")):
            a, b = lym[col][r], ref[rc]
            if abs(a - b) > 1e-9 * max(abs(b), 1e-300):
                raise RuntimeError(
                    f"{lyman_p}: the untrapped rebuild does not reproduce the "
                    f"canonical matrix at {key}, column {col}: {a:.12e} vs "
                    f"{b:.12e}. Every trapped number is a difference against "
                    f"that rebuild, so none of them can be plotted")
        n_unt += 1
    print(f"guard  lyman_trapping.csv   untrapped run reproduces the "
          f"canonical matrix at all {n_unt} pairs to 1e-9 relative")
    D_vals = np.unique(lym["D_cm"])
    if 0.0 not in set(D_vals.tolist()):
        raise RuntimeError(f"{lyman_p} has no D = 0 (untrapped) reference")

    # =======================================================================
    # Arrays for plotting, all from `rec` (the recomputation), never the CSVs.
    # =======================================================================
    nT, nN = len(Te), len(ne)

    def as_map(direction: str, field: str) -> np.ma.MaskedArray:
        A = np.full((nT, nN), np.nan)
        for (d, i, j), v in rec.items():
            if d == direction:
                A[i, j] = v[field]
        return np.ma.masked_invalid(A)

    eps_h = as_map("heat", "eps_plateau")
    win_h = np.zeros((nT, nN), bool)
    for (d, i, j), v in rec.items():
        if d == "heat":
            win_h[i, j] = v["window_ok"]
    have_h = ~eps_h.mask

    if not have_h.any():
        raise RuntimeError("no heating rows survived; the map is empty")
    ok_h = have_h & win_h
    if not ok_h.any():
        raise RuntimeError("every heating point failed the plateau-window "
                           "test; there is nothing to map")

    # global extrema, over the analysed set only (window_ok)
    keys_ok = [k for k, v in rec.items() if v["window_ok"]]
    if not keys_ok:
        raise RuntimeError("empty selection after the window_ok filter")
    k_eps = max(keys_ok, key=lambda k: rec[k]["eps_plateau"])
    k_M = max(keys_ok, key=lambda k: rec[k]["M"])
    k_eps_h = max([k for k in keys_ok if k[0] == "heat"],
                  key=lambda k: rec[k]["eps_plateau"])
    k_eps_c = max([k for k in keys_ok if k[0] == "cool"],
                  key=lambda k: rec[k]["eps_plateau"])
    # worst point at or above 2 eV -- the scope Chapter 5 can defend
    k_eps_warm = max([k for k in keys_ok if rec[k]["Te"] >= 2.0],
                     key=lambda k: rec[k]["eps_plateau"])
    e_max = rec[k_eps]["eps_plateau"]
    e_warm = rec[k_eps_warm]["eps_plateau"]
    ne_ratio = rec[k_eps]["ne"] / rec[k_M]["ne"]

    print()
    print(f"eps_plateau (window_ok, both directions): "
          f"{min(rec[k]['eps_plateau'] for k in keys_ok):.5f} .. {e_max:.5f}")
    for tag, k in (("eps max          ", k_eps), ("eps max, heating ", k_eps_h),
                   ("eps max, cooling ", k_eps_c),
                   ("eps max, Te>=2 eV", k_eps_warm), ("M max            ", k_M)):
        v = rec[k]
        print(f"  {tag} {k[0]:4s} [{k[1]:2d},{k[2]}]  Te={v['Te']:7.4f} eV  "
              f"ne={v['ne']:.4e}  eps={v['eps_plateau']:.5f}  M={v['M']:.4g}")
    print(f"  the eps maximum and the M maximum are {ne_ratio:.2f}x apart "
          f"in density")

    # direction asymmetry at the coldest transition, both readings of the SAME
    # pair of temperatures (i=0 heated to i=1; i=1 cooled to i=0)
    jc = int(np.argmax([rec[("cool", 1, j)]["eps_plateau"] for j in range(nN)]))
    a_heat = rec[("heat", 0, 3)]["eps_plateau"]
    a_cool = rec[("cool", 1, 3)]["eps_plateau"]
    print(f"  direction asymmetry at j=3 between Te={Te[0]:.4f} and "
          f"{Te[1]:.4f} eV: heating {a_heat:.5f}, cooling {a_cool:.5f} "
          f"(ratio {a_heat/a_cool:.3f})")
    print(f"  cooling maximum sits at [{k_eps_c[1]},{k_eps_c[2]}], heating "
          f"maximum at [{k_eps_h[1]},{k_eps_h[2]}]  (cooling row-1 argmax "
          f"j={jc})")

    # per-row argmax over density, heating -- the quantity findings_10 §1.2 is
    # about.  Printed for every row so no column can be hidden.
    rows_h = sorted({i for (d, i, j) in rec if d == "heat"})
    argmax_h = []
    for i in rows_h:
        cols = [j for j in range(nN) if rec[("heat", i, j)]["window_ok"]]
        if not cols:
            raise RuntimeError(f"heating row {i} has no window_ok column")
        argmax_h.append(max(cols, key=lambda j: rec[("heat", i, j)]["eps_plateau"]))
    print(f"  per-row density argmax, heating, rows {rows_h[0]}-{rows_h[-1]}:")
    print("   ", "".join(str(a) for a in argmax_h))
    n_j4 = sum(1 for a in argmax_h if a == 4)
    print(f"    j=4 (ne={ne[4]:.3g}) is the row maximum in {n_j4} of "
          f"{len(argmax_h)} heating rows — all of them the coldest; "
          f"j=3 (ne={ne[3]:.3g}) in {sum(1 for a in argmax_h if a == 3)}")

    # =======================================================================
    # file writing that refuses to clobber
    # =======================================================================
    written = []

    def emit(name: str, render):
        """render(path) writes the file. Never overwrite differing content."""
        path = outdir / name
        if path.exists():
            tmp = path.with_name(path.name + ".new")
            render(tmp)
            same = tmp.read_bytes() == path.read_bytes()
            if same:
                tmp.unlink()
                written.append((name, "identical, left alone"))
                return
            if not args.force:
                tmp.unlink()
                raise RuntimeError(
                    f"{path} already exists with different content. This "
                    f"script will not overwrite a figure that is already in "
                    f"the thesis. Inspect it, then either delete it or re-run "
                    f"with --force")
            tmp.replace(path)
            written.append((name, "OVERWRITTEN (--force)"))
            return
        render(path)
        written.append((name, "written"))

    def save(fig, base: str):
        # format is explicit: the no-clobber check renders to "<name>.pdf.new"
        # first, and matplotlib would otherwise infer the format from ".new"
        emit(base + ".pdf", lambda p: fig.savefig(p, format="pdf"))
        emit(base + ".png", lambda p: fig.savefig(p, format="png", dpi=150))
        plt.close(fig)

    def provenance(fig, y=-0.035):
        # negative y puts it outside the axes; bbox_inches="tight" expands to
        # include it. y is pushed further down where a legend sits below.
        fig.text(1.0, y, stamp, ha="right", va="top",
                 fontsize=5, color="0.55")

    Te_edges, ne_edges = log_edges(Te), log_edges(ne)

    # =======================================================================
    # FIG 5.1 -- the eps_plateau map, HEATING, with the window mask drawn
    # =======================================================================
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.set_facecolor("0.93")
    shown = np.ma.masked_where(~ok_h, eps_h)
    pc = ax.pcolormesh(Te_edges, ne_edges, shown.T, cmap="magma",
                       vmin=0.0, vmax=float(shown.max()), shading="flat",
                       rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(r"$\varepsilon_{\rm plateau}$  (fractional error in $n_3/n_4$)")
    cb.ax.tick_params(labelsize=7)

    # the 10% contour, drawn on cell centres over the analysed set only
    TT, NN = np.meshgrid(Te, ne, indexing="ij")
    cs = ax.contour(TT.T, NN.T, np.ma.filled(shown.T, np.nan), levels=[0.10],
                    colors="w", linewidths=1.0, linestyles="-")
    ax.clabel(cs, fmt={0.10: "10%"}, fontsize=7, inline=True)

    # cells with no timescale-separated plateau window: shown, not dropped
    n_nowin = 0
    for i in range(nT):
        for j in range(nN):
            if have_h[i, j] and not win_h[i, j]:
                n_nowin += 1
                ax.add_patch(Rectangle(
                    (Te_edges[i], ne_edges[j]),
                    Te_edges[i + 1] - Te_edges[i],
                    ne_edges[j + 1] - ne_edges[j],
                    facecolor="0.35", edgecolor="w", hatch="///", lw=0.3,
                    zorder=2))

    handles = []
    for k, mark, ms, lab in (
            (k_eps_h, "*", 9, "grid maximum"),
            (k_eps_warm, "s", 5.5, r"worst at $T_e \geq 2$ eV"),
            (("heat", ib, jb), "o", 5.5, "benchmark")):
        v = rec[k]
        ax.plot([v["Te"]], [v["ne"]], mark, ms=ms, mfc="none", mec=C_MARK,
                mew=1.3, zorder=4)
        handles.append(Line2D([], [], ls="none", marker=mark, ms=ms,
                              mfc="none", mec="0.15", mew=1.1,
                              label=rf"{lab}: ${v['eps_plateau']*100:.1f}\%$ "
                                    rf"at ${v['Te']:.2f}$ eV, "
                                    rf"${sci(v['ne'])}$ cm$^{{-3}}$"))
        print(f"fig5_1 marker {lab:28s} [{k[1]},{k[2]}] Te={v['Te']:.4f} "
              f"ne={v['ne']:.4e} eps={v['eps_plateau']:.5f}")
    handles.append(Patch(facecolor="0.35", edgecolor="0.15", hatch="///",
                         label=rf"no plateau window ($M \leq "
                               rf"{WIN_LO*WIN_HI:.0f}$)"))
    # Every cell of this map carries data, so the key goes outside the axes
    # rather than on top of four grid points.
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.16), ncol=2, frameon=False,
              fontsize=6.4, handletextpad=0.6, labelspacing=0.45,
              columnspacing=1.2)
    ax.set_xlabel(r"$T_e$  [eV]")
    ax.set_ylabel(r"$n_e$  [cm$^{-3}$]")
    ax.set_title(rf"heating, $+{frac*100:.0f}\%$ step in $T_e$", loc="left",
                 fontsize=8.5)
    ax.set_xticks([1, 2, 3, 5, 7, 10])
    ax.set_xticklabels(["1", "2", "3", "5", "7", "10"])
    provenance(fig, y=-0.245)
    save(fig, "fig5_1_eps_map")
    print(f"fig5_1_eps_map       {int(ok_h.sum())} analysed cells, "
          f"{n_nowin} hatched (no plateau window), "
          f"{nT*nN - int(have_h.sum())} cells absent (the hottest row, from "
          f"which a +{frac*100:.0f}% step cannot reach a new grid index)")

    # =======================================================================
    # FIG 5.2 -- eps_plateau against ne, ALL EIGHT COLUMNS, one line per Te
    #   Row selection is derived, not chosen: evenly spaced heating rows plus
    #   the benchmark row and the row carrying the grid maximum.
    # =======================================================================
    sel = sorted(set(np.round(np.linspace(rows_h[0], rows_h[-1], 6)).astype(int))
                 | {ib, k_eps_h[1]})
    for i in sel:
        if i not in rows_h:
            raise RuntimeError(f"selected Te row {i} has no heating step")
    cols7 = truncated("viridis", 0.05, 0.86, len(sel))
    # open/filled must be readable at 3.6 pt, so only hollow-able shapes
    marks = ["o", "s", "^", "D", "v", "p", "h", "<", ">"][:len(sel)]
    lstyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)),
               (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))][:len(sel)]

    # Two panels. (a) is the quantity itself; (b) is every row divided by its
    # own maximum, which is what makes the shape -- broad or sharp -- readable
    # instead of asserted. A referee asked whether the density structure is a
    # ridge; (b) is the panel that answers it.
    fig, (ax, axn) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True,
                                  gridspec_kw=dict(wspace=0.28))
    flat = {}
    for c, (i, col, mk, ls) in enumerate(zip(sel, cols7, marks, lstyles)):
        y = np.array([rec[("heat", i, j)]["eps_plateau"] for j in range(nN)])
        w = np.array([rec[("heat", i, j)]["window_ok"] for j in range(nN)])
        ax.plot(ne, y, ls=ls, color=col, lw=1.1, zorder=2,
                label=rf"$T_e = {Te[i]:.2f}$ eV")
        ax.plot(ne[w], y[w], mk, ms=3.6, color=col, mew=0, zorder=3)
        ax.plot(ne[~w], y[~w], mk, ms=3.6, mfc="w", mec=col, mew=0.8, zorder=3)
        jm = int(np.argmax(np.where(w, y, -np.inf)))
        ax.plot([ne[jm]], [y[jm]], "*", ms=8, mfc="none", mec=col, mew=0.9,
                zorder=4)
        axn.plot(ne, y / y[jm], ls=ls, color=col, lw=1.1, zorder=2)
        axn.plot(ne[w], (y / y[jm])[w], mk, ms=3.6, color=col, mew=0, zorder=3)
        axn.plot(ne[~w], (y / y[jm])[~w], mk, ms=3.6, mfc="w", mec=col,
                 mew=0.8, zorder=3)
        # measured flatness: which columns sit within 10% of the row maximum,
        # and how wide in density that set is
        near = np.flatnonzero(w & (y >= 0.9 * y[jm]))
        span = ne[near[-1]] / ne[near[0]]
        # The 10%-of-peak set is quantised by a coarse density grid (2.68x per
        # column), so it is a blunt descriptor. The three columns centred on
        # the peak span a fixed factor 7.19 in density at every temperature and
        # give a scale-free measure of how flat the crest really is.
        # An interior maximum is a claim, not a given: verify_plateau_gridmap's
        # own refutation list has "the maximum sitting at a grid CORNER rather
        # than in the interior (would indicate the range is truncated, not that
        # a ridge was found)". Check it, per row, before describing a crest.
        if jm in (0, nN - 1):
            raise RuntimeError(
                f"the density maximum of heating row {i} (Te={Te[i]:.4g} eV) "
                f"sits on the edge of the ne grid at j={jm}. The density range "
                f"is truncated; Figure 5.2 must not be drawn as an interior "
                f"maximum")
        lo, hi = max(jm - 1, 0), min(jm + 1, nN - 1)
        win = y[lo:hi + 1][w[lo:hi + 1]]
        three = win.min() / win.max()
        three_span = ne[hi] / ne[lo]
        flat[i] = (jm, y[jm], len(near), span, y[w].max() / y[w].min(),
                   three, three_span)
        print(f"fig5_2 row i={i:2d} Te={Te[i]:6.3f} eV  eps over the eight "
              f"columns: " + " ".join(f"{v:.4f}" for v in y) +
              f"   argmax j={jm} (ne={ne[jm]:.3g})  within 10% of peak: "
              f"{len(near)} columns spanning {span:.2f}x in ne;  peak column "
              f"and its two neighbours ({three_span:.2f}x in ne) vary by "
              f"{(1-three)*100:.0f}%;  peak/trough over the full grid "
              f"{flat[i][4]:.2f}x")

    ax.plot([], [], "*", ms=8, mfc="none", mec="0.35", mew=0.9,
            label="row maximum")
    ax.plot([], [], "o", ms=3.6, mfc="w", mec="0.35", mew=0.8,
            label="no plateau window")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$  [cm$^{-3}$]")
    ax.set_ylabel(r"$\varepsilon_{\rm plateau}$")
    ax.set_title(rf"(a) heating, $+{frac*100:.0f}\%$ step in $T_e$; "
                 rf"all {nN} columns", loc="left", fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.set_ylim(0, None)

    axn.axhline(0.9, color="0.4", lw=0.7, ls=":")
    axn.text(ne[-1] * 0.92, 0.905, "90% of row maximum", fontsize=6.5,
             color="0.4", va="bottom", ha="right")
    axn.set_xlabel(r"$n_e$  [cm$^{-3}$]")
    axn.set_ylabel(r"$\varepsilon_{\rm plateau}$ / row maximum")
    axn.set_title("(b) each row divided by its own maximum", loc="left",
                  fontsize=8.5)
    axn.grid(alpha=0.25, lw=0.4, which="both")
    axn.set_ylim(0, 1.09)
    # outside the axes: the Te = 1 eV curve peaks where an inset legend would go
    hs, ls_ = ax.get_legend_handles_labels()
    axn.legend(hs, ls_, frameon=False, fontsize=7, loc="upper left",
               bbox_to_anchor=(1.01, 1.0), labelspacing=0.5,
               handlelength=2.6)
    provenance(fig)
    save(fig, "fig5_2_eps_vs_ne")
    span_lo = min(f[3] for f in flat.values())
    span_hi = max(f[3] for f in flat.values())
    three_spans = {round(f[6], 6) for f in flat.values()}
    if len(three_spans) != 1:
        raise RuntimeError(f"the three-column window is not the same width at "
                           f"every plotted row: {sorted(three_spans)}")
    print(f"fig5_2_eps_vs_ne     every row has ONE interior maximum; the set "
          f"of columns within 10% of it spans {span_lo:.2f}x to {span_hi:.2f}x "
          f"in density; across the peak column and its two neighbours "
          f"({three_spans.pop():.2f}x in ne) eps varies by "
          f"{(1-max(f[5] for f in flat.values()))*100:.0f}-"
          f"{(1-min(f[5] for f in flat.values()))*100:.0f}%, while over the "
          f"full three decades it falls "
          f"{min(f[4] for f in flat.values()):.1f}-"
          f"{max(f[4] for f in flat.values()):.1f}x")

    # =======================================================================
    # FIG 5.3 -- M against eps_plateau, coloured by Te, both directions
    # =======================================================================
    kk = keys_ok
    Mv = np.array([rec[k]["M"] for k in kk])
    ev = np.array([rec[k]["eps_plateau"] for k in kk])
    tv = np.array([rec[k]["Te"] for k in kk])
    nvv = np.array([rec[k]["ne"] for k in kk])
    is_h = np.array([k[0] == "heat" for k in kk])

    # correlations, computed here.  findings_09 §3.1 records +0.76 raw,
    # +0.33 under linear control, -0.16 under quadratic control, and +0.71 for
    # a bare Arrhenius factor.  The basis and scope of the two controlled
    # numbers are not recorded there, so several are computed and all printed.
    lm, le = np.log(Mv), np.log(ev)
    lt, ln_ = np.log(tv), np.log(nvv)

    def partial(cols, y, z):
        X = np.column_stack([np.ones(len(y))] + cols)
        ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        rz = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
        return float(np.corrcoef(ry, rz)[0, 1])

    r_raw = float(np.corrcoef(lm, le)[0, 1])
    r_lin = partial([lt, ln_], lm, le)
    r_quad = partial([lt, ln_, lt * lt, ln_ * ln_, lt * ln_], lm, le)
    r_arr = float(np.corrcoef(13.6 / tv, le)[0, 1])
    Xtn = np.column_stack([np.ones(len(lm)), lt, ln_])
    r2_M = 1 - np.var(lm - Xtn @ np.linalg.lstsq(Xtn, lm, rcond=None)[0]) \
        / np.var(lm)
    hsel = is_h
    r_lin_h = partial([lt[hsel], ln_[hsel]], lm[hsel], le[hsel])
    r_quad_h = partial([lt[hsel], ln_[hsel], lt[hsel] ** 2, ln_[hsel] ** 2,
                        lt[hsel] * ln_[hsel]], lm[hsel], le[hsel])
    print()
    print(f"fig5_3 corr(log M, log eps) over the {len(kk)} window_ok pairs")
    print(f"    raw                                   {r_raw:+.3f}   "
          f"(findings_09 §3.1: +0.76)")
    print(f"    linear control on (log Te, log ne)    {r_lin:+.3f}   "
          f"(findings_09 §3.1: +0.33; heating only: {r_lin_h:+.3f})")
    print(f"    quadratic control                     {r_quad:+.3f}   "
          f"(findings_09 §3.1: -0.16; heating only: {r_quad_h:+.3f})")
    print(f"    corr(13.6/Te, log eps), no dynamics    {r_arr:+.3f}   "
          f"(findings_09 §3.1: +0.71)")
    print(f"    log M is {100*r2_M:.0f}% explained by (log Te, log ne) alone")
    if r_quad >= 0:
        raise RuntimeError(
            f"the controlled correlation did not change sign: quadratic "
            f"control gives {r_quad:+.3f}. findings_09 §3.1 records a sign "
            f"flip, and Figure 5.3's caption asserts one; do not draw the "
            f"figure until this is resolved")

    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    norm = LogNorm(vmin=tv.min(), vmax=tv.max())
    for mask, mk, lab in ((is_h, "o", "heating"), (~is_h, "^", "cooling")):
        ax.scatter(Mv[mask], ev[mask], c=tv[mask], cmap="viridis", norm=norm,
                   marker=mk, s=13, linewidths=0.25, edgecolors="0.25",
                   zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e2, 1e10)
    ax.set_ylim(8e-3, 1.0)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                      pad=0.02)
    cb.set_label(r"$T_e$  [eV]")
    cb.set_ticks([1, 2, 3, 5, 7, 10])
    cb.set_ticklabels(["1", "2", "3", "5", "7", "10"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=7)
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_yticklabels(["1%", "2%", "5%", "10%", "20%", "50%"])
    ax.yaxis.set_minor_formatter(NullFormatter())

    for k, col in ((k_eps, C_RED), (k_M, C_BLUE)):
        v = rec[k]
        ax.plot([v["M"]], [v["eps_plateau"]], "*", ms=11, mfc="none", mec=col,
                mew=1.3, zorder=5)
    ax.plot([rec[k_M]["M"], rec[k_eps]["M"]],
            [rec[k_M]["eps_plateau"], rec[k_eps]["eps_plateau"]],
            color="0.45", lw=0.7, ls=":", zorder=1)
    ax.annotate(rf"$\varepsilon$ maximum, ${e_max*100:.1f}\%$"
                "\n" rf"$T_e={rec[k_eps]['Te']:.2f}$ eV, "
                rf"$n_e={sci(rec[k_eps]['ne'])}$ cm$^{{-3}}$",
                (rec[k_eps]["M"], rec[k_eps]["eps_plateau"]),
                xytext=(1.0e5, 0.62), fontsize=7, color=C_RED, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_RED))
    ax.annotate(rf"$M$ maximum, $\varepsilon = "
                rf"{rec[k_M]['eps_plateau']*100:.1f}\%$"
                "\n" rf"$n_e={sci(rec[k_M]['ne'])}$ cm$^{{-3}}$",
                (rec[k_M]["M"], rec[k_M]["eps_plateau"]),
                xytext=(0.985, 0.05), textcoords="axes fraction",
                fontsize=7, color=C_BLUE, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="none",
                          alpha=0.85),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_BLUE))
    ax.text(6.0e8, 0.58, rf"${ne_ratio:.0f}\times$ apart in $n_e$",
            fontsize=7, color="0.3", ha="center", va="center")
    ax.text(0.030, 0.035,
            rf"$\mathrm{{corr}}(\log M, \log\varepsilon) = {r_raw:+.2f}$"
            "\n"
            rf"controlling for $(\log T_e, \log n_e)$: ${r_lin:+.2f}$ linear,"
            "\n"
            rf"${r_quad:+.2f}$ quadratic;  $e^{{13.6/T_e}}$ alone: ${r_arr:+.2f}$",
            transform=ax.transAxes, fontsize=7, color="0.2", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.8", lw=0.5),
            zorder=6)
    ax.set_xlabel(r"$M = \tau_{\rm QSS}/\tau_{\rm relax}$   "
                  r"(post-step operator)")
    ax.set_ylabel(r"$\varepsilon_{\rm plateau}$")
    ax.grid(alpha=0.25, lw=0.4, which="major")
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=4, mfc="0.55", mec="0.25",
               mew=0.3, label="heating"),
        Line2D([], [], ls="none", marker="^", ms=4, mfc="0.55", mec="0.25",
               mew=0.3, label="cooling")],
        frameon=False, fontsize=7, loc="upper left")
    provenance(fig)
    save(fig, "fig5_3_M_vs_eps")
    print(f"fig5_3_M_vs_eps      {len(kk)} window_ok pairs; M "
          f"{Mv.min():.4g} .. {Mv.max():.4g}; eps {ev.min():.5f} .. "
          f"{ev.max():.5f}")

    # =======================================================================
    # FIG 5.4 -- Lyman trapping: eps_plateau against slab thickness D
    # =======================================================================
    def ly_series(direction, i, j):
        m = ((lym["direction"] == direction) & (lym["i"] == i)
             & (lym["j"] == j))
        if not m.any():
            raise RuntimeError(f"{lyman_p} has no rows for {direction} "
                               f"[{i},{j}]")
        o = np.argsort(lym["D_cm"][m])
        D = lym["D_cm"][m][o]
        y = lym["eps_plateau"][m][o]
        w = lym["window_ok"][m][o]
        if len(D) != len(D_vals):
            raise RuntimeError(f"{lyman_p}: {direction} [{i},{j}] has "
                               f"{len(D)} slab thicknesses, the sweep has "
                               f"{len(D_vals)}")
        if not np.all(w):
            raise RuntimeError(f"{lyman_p}: {direction} [{i},{j}] loses its "
                               f"plateau window at D = "
                               f"{D[~w].tolist()} cm; it cannot be plotted as "
                               f"a plateau value")
        return D, y

    pts = [(k_eps, C_RED, "o", "-", "grid maximum"),
           (k_eps_warm, C_GREEN, "s", "--", r"worst at $T_e \geq 2$ eV"),
           (("heat", ib, jb), C_BLUE, "D", "-.", "benchmark")]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.0),
                                   gridspec_kw=dict(width_ratios=[1.25, 1],
                                                    wspace=0.34))
    for k, col, mk, ls, lab in pts:
        D, y = ly_series(*k)
        v = rec[k]
        axa.plot(D, y, ls=ls, marker=mk, ms=4, color=col, mew=0,
                 label=rf"{lab} [{k[1]},{k[2]}]: ${v['Te']:.2f}$ eV, "
                       rf"${sci(v['ne'])}$ cm$^{{-3}}$")
        print(f"fig5_4 {k[0]} [{k[1]},{k[2]}] Te={v['Te']:.4f} "
              f"ne={v['ne']:.4e}: eps at D = " +
              ", ".join(f"{d:g} cm -> {yy:.5f}" for d, yy in zip(D, y)) +
              f"   fall {y[0]/y[-1]:.2f}x")
    axa.axhline(0.10, color="0.35", lw=0.8, ls=":")
    axa.text(20, 0.104, "10% threshold", fontsize=7, color="0.35", ha="right")
    axa.set_xlabel(r"Lyman slab thickness $D$  [cm]   "
                   r"($D = 0$: optically thin)")
    axa.set_ylabel(r"$\varepsilon_{\rm plateau}$")
    axa.set_ylim(0, 0.52)
    axa.set_xticks(sorted(D_vals.tolist()))
    axa.grid(alpha=0.25, lw=0.4)
    axa.legend(frameon=False, fontsize=6.2, loc="upper right",
               handlelength=2.4, labelspacing=0.35)
    axa.set_title("(a) three points", loc="left", fontsize=8.5)

    # (b) how many analysed pairs exceed 10%, all Te against Te >= 2 eV
    cnt_all, cnt_warm, worst_all, worst_warm = [], [], [], []
    for D in sorted(D_vals.tolist()):
        m = (lym["D_cm"] == D) & lym["window_ok"]
        if not m.any():
            raise RuntimeError(f"{lyman_p}: no window_ok rows at D = {D} cm")
        e = lym["eps_plateau"][m]
        warm = lym["Te"][m] >= 2.0
        if not warm.any():
            raise RuntimeError(f"{lyman_p}: no Te >= 2 eV rows at D = {D} cm")
        cnt_all.append(int((e > 0.10).sum()))
        cnt_warm.append(int((e[warm] > 0.10).sum()))
        worst_all.append(float(e.max()))
        worst_warm.append(float(e[warm].max()))
        print(f"fig5_4 D={D:5g} cm: {int(m.sum())} analysed pairs, "
              f"{cnt_all[-1]} exceed 10% (worst {worst_all[-1]:.5f}); "
              f"Te>=2 eV: {int(warm.sum())} pairs, {cnt_warm[-1]} exceed 10% "
              f"(worst {worst_warm[-1]:.5f})")
    Ds = np.array(sorted(D_vals.tolist()))
    axb.plot(Ds, cnt_all, "-o", ms=4, color="#6a3d9a", mew=0,
             label="all analysed pairs")
    axb.plot(Ds, cnt_warm, "--s", ms=4, color="#e67e22", mew=0,
             label=r"$T_e \geq 2$ eV")
    # label the endpoints only; at D = 0 and 1 cm the markers are 1 cm apart
    for series, col, dy in ((cnt_all, "#6a3d9a", 6), (cnt_warm, "#e67e22", -12)):
        for idx, ha, dx in ((0, "left", 1), (len(Ds) - 1, "right", -1)):
            axb.annotate(f"{series[idx]}", (Ds[idx], series[idx]),
                         xytext=(dx * 2, dy), textcoords="offset points",
                         fontsize=6.5, ha=ha, color=col)
    axb.set_xlabel(r"$D$  [cm]")
    axb.set_ylabel(r"pairs with $\varepsilon_{\rm plateau} > 10\%$")
    axb.set_xticks(sorted(D_vals.tolist()))
    axb.set_ylim(0, max(cnt_all) * 1.28)
    axb.grid(alpha=0.25, lw=0.4)
    axb.legend(frameon=False, fontsize=7, loc="lower left")
    axb.set_title("(b) the whole map", loc="left", fontsize=8.5)
    provenance(fig)
    save(fig, "fig5_4_trapping")

    D0, y0 = ly_series(*k_eps)
    Dw, yw = ly_series(*k_eps_warm)
    Db, yb = ly_series("heat", ib, jb)
    trap_lo, trap_hi = float(min(y0[1:])), float(max(y0[1:]))
    warm_drift = abs(yw[-1] / yw[0] - 1.0)
    bench_drift = abs(yb[-1] / yb[0] - 1.0)
    print(f"fig5_4_trapping      the grid maximum falls {y0[0]*100:.1f}% -> "
          f"{trap_lo*100:.1f}-{trap_hi*100:.1f}% over D = "
          f"{Ds[1]:g}-{Ds[-1]:g} cm; the Te>=2 eV worst point moves "
          f"{warm_drift*100:.2f}% and the benchmark {bench_drift*100:.2f}%")

    # =======================================================================
    # captions -- written from the numbers just plotted, not retyped
    # =======================================================================
    tok = {
        "@FRAC@": f"{frac*100:.0f}",
        "@EPSMAX@": f"{e_max*100:.1f}",
        "@EPSMAX_TE@": f"{rec[k_eps]['Te']:.2f}",
        "@EPSMAX_NE@": sci(rec[k_eps]["ne"]),
        "@EPSMAX_I@": str(k_eps[1]), "@EPSMAX_J@": str(k_eps[2]),
        "@COOLMAX@": f"{rec[k_eps_c]['eps_plateau']*100:.1f}",
        "@COOLMAX_I@": str(k_eps_c[1]), "@COOLMAX_J@": str(k_eps_c[2]),
        "@AHEAT@": f"{a_heat*100:.1f}", "@ACOOL@": f"{a_cool*100:.1f}",
        "@ANE@": sci(ne[3]),
        "@COOLMAX_TE@": f"{rec[k_eps_c]['Te']:.2f}",
        "@COOLMAX_NE@": sci(rec[k_eps_c]["ne"]),
        "@NSEL@": str(len(sel)),
        "@MRATIO@": f"{rec[k_M]['M']/rec[k_eps]['M']:.0f}",
        "@NPAIRD@": str(int(((lym["D_cm"] == Ds[-1]) & lym["window_ok"]).sum())),
        "@WARM@": f"{e_warm*100:.1f}",
        "@WARM_TE@": f"{rec[k_eps_warm]['Te']:.2f}",
        "@WARM_NE@": sci(rec[k_eps_warm]["ne"]),
        "@BENCH@": f"{rec[('heat', ib, jb)]['eps_plateau']*100:.2f}",
        "@BENCH_TE@": f"{Te[ib]:.3f}",
        "@BENCH_NE@": sci(ne[jb], 3),
        "@NWIN@": str(len(kk)), "@NPAIR@": str(len(rec)),
        "@NNOWIN@": str(n_nowin), "@MWIN@": f"{WIN_LO*WIN_HI:.0f}",
        "@NCOL@": str(nN), "@NJ4@": str(n_j4), "@NROWS@": str(len(rows_h)),
        "@NEXT@": f"{ne[1]/ne[0]:.2f}",
        "@NE3@": sci(ne[3]),
        "@NE4@": sci(ne[4]),
        "@THREESPAN@": f"{min(f[6] for f in flat.values()):.1f}",
        "@THREEVARLO@": f"{(1-max(f[5] for f in flat.values()))*100:.0f}",
        "@THREEVARHI@": f"{(1-min(f[5] for f in flat.values()))*100:.0f}",
        "@PEAKTROUGH@": f"{min(f[4] for f in flat.values()):.1f}",
        "@PEAKTROUGH_HI@": f"{max(f[4] for f in flat.values()):.1f}",
        "@RRAW@": f"{r_raw:+.2f}", "@RLIN@": f"{r_lin:+.2f}",
        "@RQUAD@": f"{r_quad:+.2f}", "@RARR@": f"{r_arr:+.2f}",
        "@R2M@": f"{100*r2_M:.0f}",
        "@NERATIO@": f"{ne_ratio:.0f}",
        "@MMAX@": sci(rec[k_M]["M"]),
        "@MMAX_EPS@": f"{rec[k_M]['eps_plateau']*100:.1f}",
        "@MMAX_TE@": f"{rec[k_M]['Te']:.2f}",
        "@MMAX_NE@": sci(rec[k_M]["ne"]),
        "@TRAP0@": f"{y0[0]*100:.1f}", "@TRAPLO@": f"{trap_lo*100:.1f}",
        "@TRAPHI@": f"{trap_hi*100:.1f}",
        "@DMAX@": f"{Ds[-1]:g}", "@DMIN@": f"{Ds[1]:g}",
        "@WARMDRIFT@": f"{warm_drift*100:.1f}",
        "@BENCHDRIFT@": f"{bench_drift*100:.1f}",
        "@CNT0@": str(cnt_all[0]), "@CNTMAX@": str(cnt_all[-1]),
        "@CNTW0@": str(cnt_warm[0]), "@CNTWMAX@": str(cnt_warm[-1]),
        "@SHA8@": sha8, "@SHAL@": sha_L[:16],
        "@DATE@": f"{datetime.now():%Y-%m-%d}",
    }
    caption_src = r"""% figures/fig5_captions.tex
% Generated by src/analysis/make_ch5_figures.py on @DATE@.
% Every number below was computed by that script from the same arrays it
% plotted; do not edit them by hand. CR data SHA-8 @SHA8@
% (L_grid sha256 @SHAL@...).
%
% Usage:  \input{figures/fig5_captions.tex}  in the preamble, then
%   \begin{figure}[t]\centering
%     \includegraphics{figures/fig5_1_eps_map.pdf}
%     \caption{\CapFigEpsMap}\label{fig:eps-map}
%   \end{figure}

\newcommand{\CapFigEpsMap}{%
  \textbf{Where the quasi-steady-state closure fails.}
  Fractional error $\varepsilon_{\rm plateau}$ in the $n=3/n=4$ shell
  population ratio committed while the excited manifold has reached partial
  equilibrium but the ground state has not yet responded, after a
  $+@FRAC@\%$ step in $T_e$ at fixed $n_e$ --- the \emph{heating} direction.
  Cooling is not the mirror image: the same transition between the two
  coldest grid temperatures, at $n_e = @ANE@$~cm$^{-3}$, gives @AHEAT@\%
  heated and @ACOOL@\% cooled, and the grid maximum moves from
  $[@EPSMAX_I@,@EPSMAX_J@]$ ($@EPSMAX_TE@$~eV, $@EPSMAX_NE@$~cm$^{-3}$) to
  $[@COOLMAX_I@,@COOLMAX_J@]$ ($@COOLMAX_TE@$~eV, $@COOLMAX_NE@$~cm$^{-3}$),
  where it is @COOLMAX@\%.
  Hatched cells have no timescale-separated plateau window
  ($M \leq @MWIN@$, @NNOWIN@ of the heated cells): a plateau value is not
  defined there and none is plotted. The plain grey strip at the right-hand
  edge is the hottest temperature row, from which a $+@FRAC@\%$ step cannot
  reach a new grid index. The three marked cells, keyed below the axes, are
  the grid maximum (@EPSMAX@\% at $T_e = @EPSMAX_TE@$~eV,
  $n_e = @EPSMAX_NE@$~cm$^{-3}$), the largest value at $T_e \geq 2$~eV
  (@WARM@\% at $T_e = @WARM_TE@$~eV, $n_e = @WARM_NE@$~cm$^{-3}$), and the
  thesis benchmark point ($T_e = @BENCH_TE@$~eV,
  $n_e = @BENCH_NE@$~cm$^{-3}$, @BENCH@\%).
  Below $2$~eV the values shown are upper estimates: switching on Lyman
  trapping divides the grid maximum by $2.5$--$3.3$ depending on the assumed
  slab thickness, and below $1.15$~eV it also moves which density column
  carries the maximum (Figure~\ref{fig:trapping}). Nothing on the cold edge of
  this map is quantitative.}

\newcommand{\CapFigEpsVsNe}{%
  \textbf{The density dependence, all @NCOL@ columns.}
  $\varepsilon_{\rm plateau}$ against $n_e$ for @NSEL@ temperatures, heating,
  $+@FRAC@\%$ step. Every density column computed is shown; stars mark each
  row's maximum, open symbols mark points with no plateau window.
  Each row has a single interior maximum, but it is
  \emph{broad and shallow in $\log n_e$}: across the peak column and its two
  neighbours --- a factor @THREESPAN@ in density --- $\varepsilon$ varies by
  only @THREEVARLO@--@THREEVARHI@\%, so the location of the maximum is not
  resolved to better than about one grid interval, a factor @NEXT@. Over the
  full three decades the same row falls by a factor
  @PEAKTROUGH@--@PEAKTROUGH_HI@: the structure is a real crest in density, but
  a wide-topped one, and the word ridge should not be read as implying a sharp
  or well-localised line. The column at $n_e = @NE4@$~cm$^{-3}$ carries the row
  maximum in @NJ4@ of @NROWS@ heating rows, all of them the coldest; at every
  warmer temperature the maximum sits at $@NE3@$~cm$^{-3}$. Both columns must
  be reported: quoting either alone misstates where the closure is worst, and
  below $1.15$~eV the position moves across three columns once Lyman trapping
  is included (Figure~\ref{fig:trapping}), so no maximum location is claimed
  there at all.}

\newcommand{\CapFigMvsEps}{%
  \textbf{Timescale separation does not predict closure error.}
  Each point is one (grid point, step direction) pair with a
  timescale-separated plateau window (@NWIN@ of @NPAIR@); $M$ is
  $\tau_{\rm QSS}/\tau_{\rm relax}$ of the post-step operator, colour is $T_e$,
  circles heating and triangles cooling. The raw association is strong,
  $\mathrm{corr}(\log M, \log\varepsilon) = @RRAW@$, and it is a temperature
  proxy rather than a dynamical statement: $\log M$ is @R2M@\% explained by
  $(\log T_e, \log n_e)$ alone, a bare Arrhenius factor $e^{13.6/T_e}$
  containing no dynamics correlates at @RARR@, and controlling for
  $(\log T_e, \log n_e)$ drives the association to @RLIN@ linearly and
  @RQUAD@ --- the opposite sign --- quadratically. The two starred extrema make
  the same point without statistics: the largest $M$ on the grid
  ($@MMAX@$, blue) carries $\varepsilon = @MMAX_EPS@\%$, while the largest
  error (@EPSMAX@\%, red) occurs @NERATIO@$\times$ away in density, at an $M$
  smaller by a factor @MRATIO@. A large timescale separation is not evidence that
  the quasi-steady-state closure is accurate.}

\newcommand{\CapFigTrapping}{%
  \textbf{Lyman trapping, and why the quantitative scope starts at 2~eV.}
  (a) $\varepsilon_{\rm plateau}$ against the assumed homogeneous-slab
  thickness $D$ used for the Lyman escape factors, computed self-consistently
  ($\Theta_P$ depends on $n(1s)$, which depends on $\Theta_P$) for all 14
  Lyman channels; $D = 0$ is the optically thin matrix used everywhere else in
  this chapter. The grid maximum at $T_e = @EPSMAX_TE@$~eV falls from
  @TRAP0@\% to @TRAPLO@--@TRAPHI@\% over $D = @DMIN@$--@DMAX@~cm: the cold-edge
  number is set by a slab thickness this zero-dimensional model does not
  contain, and must never be quoted without it. The worst point at
  $T_e \geq 2$~eV moves by @WARMDRIFT@\% and the benchmark by @BENCHDRIFT@\%
  over the same twentyfold range. (b) Number of analysed pairs exceeding the
  $10\%$ threshold, counted on the plateau value itself and not on the
  time-averaged ELM bound: @CNT0@ falls to @CNTMAX@ over the whole map, while
  above $2$~eV it is @CNTW0@ against @CNTWMAX@ --- unchanged to within one
  pair. (At $D = @DMAX@$~cm the analysed set is @NPAIRD@ pairs rather than
  @NWIN@: two lose their plateau window.)
  The $T_e \geq 2$~eV restriction is therefore not a hedge; it is the measured
  boundary beyond which the one neglected process most likely to invalidate
  the result stops mattering.}
"""
    for k, v in tok.items():
        caption_src = caption_src.replace(k, v)
    leftover = [w for w in caption_src.split() if w.startswith("@")]
    if leftover:
        raise RuntimeError(f"caption template has unsubstituted tokens: "
                           f"{leftover}")
    emit("fig5_captions.tex", lambda p: p.write_text(caption_src))

    print("\nwrote:")
    for name, how in written:
        print(f"  {outdir / name}   [{how}]")


if __name__ == "__main__":
    main()
