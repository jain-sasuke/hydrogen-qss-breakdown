#!/usr/bin/env python
"""
verify_lyman_optical_depth.py
=============================
Lyman-alpha line-centre optical depth of the CRE ground-state population,
per grid point, with the slab-thickness, isotope and neutral-temperature
sensitivities, the tau_0 = 1 boundary, and the breakdown census recount that
chapters 5 and 6 quote.

WHY THIS EXISTS
---------------
Chapter 5 (thesis_tex/chapter5.tex, sec:optical_depth) quotes Lyman-alpha
optical depths per centimetre of 114 at Te = 1.00 eV, ne = 5.18e13 cm^-3,
0.036 at 2.02 eV and 0.0023 at 2.95 eV, and attributes them to
verify_lyman_trapping.py. That script never prints an optical depth. The
numbers are a hand calculation whose Doppler width was built from
sqrt(kT/m) rather than sqrt(2kT/m); they are sqrt(2) too large. Chapter 6's
table (80.3, 10.6, 0.224, 0.0255, 0.00163) uses the width the repo's own code
uses. Chapter 6's prose two paragraphs later nevertheless repeats 0.036 and
0.0023, and calls the ne = 5.18e13 row "the benchmark point", which is at
ne = 1.389e14. Nothing in the repository emits the per-centimetre optical
depth as a file with provenance. This script does.

PHYSICS
-------
Line-centre absorption cross-section for a Doppler profile (Rybicki and
Lightman eq. 10.6 with the normalised profile phi(0) = 1/(sqrt(pi) dnu_D);
Mihalas 9-2; ADAS214 eq. 3.14.5):

    sigma_0 = (pi e^2 / m_e c) f_12 / (sqrt(pi) dnu_D)
    dnu_D   = (nu_0 / c) sqrt(2 k T_n / m)          [1/e half-width]

kappa_0 = sigma_0 n(1s) is the absorption coefficient per centimetre.
n(1s) is the CRE ground-state density: n = -L^{-1} S per unit n_ion, times
n_ion = n_e (quasineutral pure hydrogen), which is the expression
src/analysis/make_story_figures.py uses for the same quantity.

Two optical-depth definitions are carried side by side, because the census
depends on which one is meant:
    tau_half = kappa_0 D/2     the half-slab depth escape_factor.py requires
                               (a photon born at the slab centre)
    tau_full = kappa_0 D       the full path length

Sensitivities: deuterium mass (dnu_D falls by sqrt(m_D/m_H), sigma_0 and
kappa_0 rise by the same factor), and a fixed neutral temperature
T_n = 3 eV (Franck-Condon atoms) instead of the project's T_n = Te.
The superseded sqrt(kT/m) width is also carried, labelled WRONG, so that the
thesis's present numbers can be reproduced and identified.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not modify any matrix, tolerance or data file. It reads the
canonical L_grid, S_grid and state_index through cr_context, reads
validation/divertor_map/divertor_map.csv (census) and
validation/lyman_trapping/lyman_trapping.txt (self-consistent escape factors)
as artifacts, checks their recorded provenance against the live data, and
stops if either is stale.

PREDICTIONS (written before the first run)
------------------------------------------
P1  sigma_0(T) recomputed here from the constants in escape_factor.py agrees
    with escape_factor.lyman_alpha_sigma0 to 1e-6 relative (asserted).
P2  Hydrogen kappa_0 at [0,4], [3,4], [10,4], [15,4], [23,4] reproduces
    chapter 6's table 80.3, 10.6, 0.224, 0.0255, 0.00163 to three digits.
P3  The sqrt(kT/m) (WRONG) column reproduces chapter 5's 114, 0.036, 0.0023
    at [0,4], [15,4], [23,4]. The deuterium column at [0,4] is 113.6, so a
    deuterium reading would also "explain" 114; the two are distinguished
    only by the [15,4] and [23,4] values, which are the same for both.
P4  T_n = 3 eV at [0,4] gives 46.4 per cm (80.3 / sqrt(3)).
P5  The tau_half = 1 boundary at D = 5 cm, hydrogen, runs from 1.125 to
    1.984 eV across the eight density columns, matching
    make_story_figures.py; it is unchanged by which sigma_0 the thesis
    quotes, because that error is sqrt(2) in a quantity that varies by
    five orders of magnitude over 1-3 eV.
P6  Of the 202 breakdown pairs (window_ok, lo_ELM_crash > 0.10) in
    divertor_map.csv, 157 have pre-step Te < 2 eV;
    (a) half-slab hydrogen: 85 have tau(5 cm) > 1 and 15 have > 100;
    (b) full-path hydrogen: 100 and 23;
    (c) full-path with the WRONG width: 105 and 26, which is what
        chapter 5 (~1551) and chapter 6 (~192) currently print.
    So the thesis's 105/26 reproduce only under (c).
P7  At [0,4], D = 5 cm: tau_half = 200.7 and the one-shot population escape
    factor from the thin n(1s) is 1.178e-3; the self-consistent value in
    lyman_trapping.txt is 1.44e-2; the self-consistent minimum over the grid
    is 7.8e-4 at D = 20 cm.
P8  The benchmark [23,5] escape factor at D = 1 cm in lyman_trapping.txt is
    0.99862, not the 0.9999 chapter 6 (~286) prints.

REFUTING OBSERVATION
--------------------
Any of P1-P8 failing. In particular: if the hydrogen column does not
reproduce chapter 6's table, the thesis's "correct" numbers are also
unexplained; if 105/26 reproduce under (a) or (b), the diagnosis of the
census as a width error is wrong.

OUTPUTS (with --write)
----------------------
validation/lyman_optical_depth/lyman_optical_depth.csv           per grid point
validation/lyman_optical_depth/lyman_optical_depth_summary.csv   every quoted number
validation/lyman_optical_depth/lyman_optical_depth.txt           this run's log
Each carries a '#' header with the script name, date, interpreter and the
sha256 of L_grid.npy, S_grid.npy and state_index.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── repo wiring: the root comes from this file's location, not the cwd ───────
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from escape_factor import (                               # noqa: E402
    escape_factor_quadrature,
    lyman_alpha_sigma0,
)

SCRIPT = _HERE.name

# ── physical constants, Gaussian CGS ─────────────────────────────────────────
# The first five are the values escape_factor.lyman_alpha_sigma0 uses; they
# are repeated here so that sigma_0 can be rebuilt independently of that
# function and compared with it (P1). m_H there is the proton mass.
E_ESU = 4.80326e-10        # statcoulomb
M_E = 9.10938e-28          # g
C_CGS = 2.99792e10         # cm/s
M_H = 1.67262e-24          # g   (proton mass, as in escape_factor.py)
EV_TO_ERG = 1.60218e-12
F_12 = 0.4162              # Ly-alpha absorption oscillator strength
LAMBDA_0 = 1.21567e-5      # cm
# Deuteron mass, CODATA 2018: 3.3435837768e-27 kg. Used with the same
# nucleus-mass convention as M_H above. m_D/m_H = 1.99900, sqrt = 1.41386.
M_D = 3.34358e-24          # g
T_N_FIXED = 3.0            # eV, Franck-Condon neutral temperature sensitivity

# ── the numbers the thesis prints, with their location ───────────────────────
# (index_i, index_j): value per cm.  Chapter 6 table ~155-159, ne = 5.18e13.
CH6_TABLE = {(0, 4): 80.3, (3, 4): 10.6, (10, 4): 0.224,
             (15, 4): 0.0255, (23, 4): 0.00163}
# Chapter 5 ~2148-2151 (and repeated in chapter 6 prose ~176-177).
CH5_PROSE = {(0, 4): 114.0, (15, 4): 0.036, (23, 4): 0.0023}
CH5_CENSUS = dict(n_pairs=202, n_cold=157, n_tau_gt_1=105, n_tau_gt_100=26)
CH6_BENCH_D1_THETA = 0.9999        # chapter 6 ~286
CH6_BENCH_D20_THETA = 0.973        # chapter 6 ~287
CH6_ONE_SHOT = dict(tau_half=201.0, theta=1.2e-3)   # chapter 6 ~166-168
BENCH_TE, BENCH_NE = 2.947, 1.389e14                # CLAUDE.md benchmark


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_array16(a: np.ndarray) -> str:
    """verify_lyman_trapping.py hashes array BYTES, not the file; 16 hex."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def digits_agree(value: float, recorded: float) -> int:
    """How many significant digits of `recorded` `value` reproduces.
    `recorded` is a rounded thesis number, so the test is: does `value`
    round to `recorded` at the precision `recorded` is written to?"""
    s = f"{recorded:.10g}"
    sig = len(s.replace(".", "").replace("-", "").lstrip("0"))
    for d in range(sig, 0, -1):
        if float(f"{value:.{d}g}") == float(f"{recorded:.{d}g}"):
            return d
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 1. sigma_0: independent rebuild and comparison with the module
# ══════════════════════════════════════════════════════════════════════════════

def sigma0_independent(T_n_eV: float, mass_g: float = M_H,
                       width_factor: float = 2.0) -> float:
    """
    sigma_0 = (pi e^2 / m_e c) f_12 / (sqrt(pi) dnu_D),
    dnu_D = (nu_0/c) sqrt(width_factor k T / m).
    width_factor = 2 is the 1/e Doppler half-width (correct);
    width_factor = 1 is the sqrt(kT/m) width behind the thesis's 114.
    """
    nu0 = C_CGS / LAMBDA_0
    v = np.sqrt(width_factor * T_n_eV * EV_TO_ERG / mass_g)
    dnu_D = (nu0 / C_CGS) * v
    prefactor = np.pi * E_ESU ** 2 / (M_E * C_CGS)
    return prefactor * F_12 / (np.sqrt(np.pi) * dnu_D)


# ══════════════════════════════════════════════════════════════════════════════
# 2. the tau_0 = 1 boundary, same interpolation as make_story_figures.crossing
# ══════════════════════════════════════════════════════════════════════════════

def crossing(field: np.ndarray, level: float, Te: np.ndarray) -> np.ndarray:
    """Te at which `field` crosses `level`, per density column, by linear
    interpolation in (log Te, log field); NaN where a column never crosses.
    Copied in form from make_story_figures.py so the two agree by
    construction rather than by tolerance."""
    nN = field.shape[1]
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. readers for the two artifacts this script consumes
# ══════════════════════════════════════════════════════════════════════════════

def read_divertor_map(path: Path, sha_L: str, sha_S: str, sha_si: str):
    """divertor_map.csv, with its '#' provenance header checked against the
    live data. A stale map would be counted against a different n(1s)."""
    if not path.is_file():
        raise FileNotFoundError(f"census source missing: {path}")
    header, rows = [], []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                header.append(line.strip())
            else:
                rows.append(line)
    recorded = " ".join(header)
    for lbl, sha in (("L_grid", sha_L), ("S_grid", sha_S),
                     ("state_index", sha_si)):
        if sha not in recorded:
            raise RuntimeError(
                f"{path} was generated from a different {lbl} (its header "
                f"does not carry sha256 {sha[:16]}...). The census would be "
                f"counted against a different CRE solution. STOPPING.")
    table = list(csv.DictReader(rows))
    if not table:
        raise RuntimeError(f"{path} has no data rows")
    return header, table


def read_lyman_trapping_log(path: Path, sha_L16: str):
    """
    Parse the SELF-CONSISTENT TRAPPED RUN blocks of lyman_trapping.txt.
    Returns {D: dict(theta_min=..., theta_max=..., rows=[(Te, ne, n1s, theta)])}.
    lyman_trapping.csv does not carry Theta (checked), so the log is the only
    persisted record of the self-consistent escape factors.
    """
    if not path.is_file():
        raise FileNotFoundError(f"self-consistent escape factors missing: {path}")
    text = path.read_text()
    m = re.search(r"L_grid \(canon\)\s+\([^)]*\)\s+([0-9a-f]{16})", text)
    if not m:
        raise RuntimeError(f"{path} carries no 'L_grid (canon)' hash line; "
                           f"cannot establish which matrix it was run on")
    if m.group(1) != sha_L16:
        raise RuntimeError(
            f"{path} was run on L_grid {m.group(1)}, the live L_grid hashes "
            f"to {sha_L16} (sha256 of the array bytes, first 16 hex). "
            f"The self-consistent escape factors belong to a different "
            f"matrix. STOPPING.")
    blocks = {}
    pat = re.compile(r"SELF-CONSISTENT TRAPPED RUN: D = ([0-9.]+) cm")
    starts = [(mm.start(), float(mm.group(1))) for mm in pat.finditer(text)]
    if not starts:
        raise RuntimeError(f"{path} has no SELF-CONSISTENT TRAPPED RUN block")
    for k, (pos, D) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(text)
        chunk = text[pos:end]
        mm = re.search(r"Theta_P\(Ly-alpha\): min ([0-9.eE+-]+)\s+max ([0-9.eE+-]+)",
                       chunk)
        if not mm:
            raise RuntimeError(f"{path}: no Theta_P min/max line for D={D}")
        rows = []
        for line in chunk.splitlines():
            parts = line.split()
            if len(parts) == 7:
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                rows.append(tuple(vals[:4]))          # Te, ne, n1s, Theta
        if not rows:
            raise RuntimeError(f"{path}: no per-point rows for D={D}")
        blocks[D] = dict(theta_min=float(mm.group(1)),
                         theta_max=float(mm.group(2)), rows=rows)
    return blocks


def log_theta_at(blocks, D: float, Te: float, ne: float) -> float:
    """Theta_Lya from the log at the row whose (Te, ne) match to the log's
    own print precision (Te 3 decimals, ne 4 significant)."""
    if D not in blocks:
        raise KeyError(f"lyman_trapping.txt has no D = {D} cm run; "
                       f"available: {sorted(blocks)}")
    for te_r, ne_r, _, th in blocks[D]["rows"]:
        if abs(te_r - Te) < 6e-4 and rel(ne_r, ne) < 6e-4:
            return th
    raise KeyError(f"lyman_trapping.txt D={D}: no row at Te={Te:.3f}, "
                   f"ne={ne:.3e}; the log only prints rows (0,10,23)x(0,3,4,5)")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--slab", type=float, nargs="+", default=[1.0, 5.0, 20.0],
                    help="slab thicknesses D [cm]")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    D_list = [float(d) for d in a.slab]
    if 5.0 not in D_list:
        raise ValueError("the census and the thesis's quoted table are at "
                         "D = 5 cm; --slab must include 5")

    out_dir = Path(a.out) if a.out else ROOT / "validation" / "lyman_optical_depth"
    log_fh = None
    if a.write:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_fh = (out_dir / "lyman_optical_depth.txt").open("w")
        sys.stdout = _Tee(sys.__stdout__, log_fh)

    # ── load through cr_context; never redefine a grid ──────────────────────
    ctx = CRContext.load(root=ROOT)
    Te, ne, L = ctx.te_grid, ctx.ne_grid, ctx.L_grid
    g = ctx.ground_index
    nT, nN = len(Te), len(ne)
    L_path = ROOT / "data/processed/cr_matrix/L_grid.npy"
    S_path = ROOT / "data/processed/cr_matrix/S_grid.npy"
    si_path = Path(ctx.state_index_path)
    if not S_path.is_file():
        raise FileNotFoundError(f"S_grid missing: {S_path}")
    S = np.load(S_path)
    if S.shape != L.shape[:3]:
        raise RuntimeError(f"S_grid {S.shape} incompatible with L_grid "
                           f"{L.shape}; one of them is stale")
    if ctx.labels[g] != "1S":
        raise RuntimeError(f"ground state label is {ctx.labels[g]!r}, not '1S'; "
                           f"state ordering is not what n(1s) assumes")

    sha_L, sha_S, sha_si = (sha256_file(L_path), sha256_file(S_path),
                            sha256_file(si_path))
    sha_L16 = sha256_array16(L)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    prov_lines = [
        f"# generated {stamp} by {SCRIPT}",
        f"# interpreter {sys.executable}  numpy {np.__version__}",
        f"# L_grid.npy      sha256 {sha_L}",
        f"# S_grid.npy      sha256 {sha_S}",
        f"# state_index.csv sha256 {sha_si}  ({si_path.relative_to(ROOT)})",
    ]
    for line in prov_lines:          # first lines of the log: '#' provenance
        print(line)
    print("=" * 78)
    print("LYMAN-ALPHA OPTICAL DEPTH -- provenance")
    print("=" * 78)
    print(ctx.describe())
    print(f"  S_grid         : {S.shape}")
    print(f"  L_grid bytes   : sha256[:16] {sha_L16} "
          f"(the hash convention lyman_trapping.txt uses)")
    print()

    summary = []   # (item, quantity, value, recorded, source, status)

    def record(item, quantity, value, recorded=None, source="", tol_digits=3):
        if recorded is None:
            summary.append((item, quantity, value, "", source, ""))
            return f"{quantity} = {value:.6g}"
        d = digits_agree(value, recorded)
        status = "REPRODUCED" if d >= min(tol_digits, len(
            f"{recorded:.10g}".replace(".", "").lstrip("0"))) else "NOT REPRODUCED"
        summary.append((item, quantity, value, recorded, source,
                        f"{status} ({d} digits)"))
        return (f"{quantity} = {value:.6g}   recorded {recorded:g} "
                f"[{source}]  -> {status}, {d} digits, rel {rel(value, recorded):.2e}")

    # ── P1: sigma_0 module vs independent ───────────────────────────────────
    print("=" * 78)
    print("P1  sigma_0: escape_factor.lyman_alpha_sigma0 vs independent rebuild")
    print("=" * 78)
    worst = 0.0
    for T in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
        s_mod, s_ind = float(lyman_alpha_sigma0(T)), sigma0_independent(T)
        r = rel(s_ind, s_mod)
        worst = max(worst, r)
        print(f"  T = {T:5.2f} eV: module {s_mod:.6e}  independent {s_ind:.6e}"
              f"  rel {r:.2e}")
    if worst > 1e-6:
        raise RuntimeError(f"sigma_0 rebuild disagrees with the module by "
                           f"{worst:.2e} > 1e-6; the constants or the formula "
                           f"differ. STOPPING.")
    print(f"  [PASS] agree to {worst:.1e} (asserted < 1e-6)")
    s1 = float(lyman_alpha_sigma0(1.0))
    print("  " + record("P1", "sigma_0(1 eV, H) [cm^2]", s1, 5.47e-14,
                        "chapter 6 ~146; escape_factor.validate Test 4"))
    print(f"  sqrt(kT/m) (WRONG) width at 1 eV: "
          f"{sigma0_independent(1.0, width_factor=1.0):.4e} cm^2 = "
          f"{sigma0_independent(1.0, width_factor=1.0)/s1:.5f} x correct")
    print(f"  deuterium at 1 eV: {sigma0_independent(1.0, M_D):.4e} cm^2 = "
          f"{sigma0_independent(1.0, M_D)/s1:.5f} x hydrogen "
          f"(sqrt(m_D/m_H) = {np.sqrt(M_D/M_H):.5f}; sqrt 2 = {np.sqrt(2):.5f})")
    print(f"  T_n = {T_N_FIXED:g} eV: {sigma0_independent(T_N_FIXED):.4e} cm^2 = "
          f"{sigma0_independent(T_N_FIXED)/s1:.5f} x the 1 eV value")
    print()

    # ── item 1: n(1s) and kappa_0 on the whole grid ──────────────────────────
    print("=" * 78)
    print("1   n(1s) = n_e [-L^{-1} S]_1s and kappa_0 = sigma_0 n(1s) on the grid")
    print("=" * 78)
    n1s_per_ion = np.empty((nT, nN))
    for i in range(nT):
        for j in range(nN):
            n1s_per_ion[i, j] = np.linalg.solve(L[i, j], -S[i, j])[g]
    if not np.all(np.isfinite(n1s_per_ion)) or np.any(n1s_per_ion <= 0):
        raise RuntimeError("non-positive or non-finite n(1s) from the CRE "
                           "solve; an optical depth built from it is meaningless")
    n1s = n1s_per_ion * ne[None, :]                      # n_ion = n_e

    sig_H = np.array([sigma0_independent(t) for t in Te])            # T_n = Te
    sig_D = np.array([sigma0_independent(t, M_D) for t in Te])
    sig_T3 = np.full(nT, sigma0_independent(T_N_FIXED))
    sig_W = np.array([sigma0_independent(t, width_factor=1.0) for t in Te])
    kap = dict(H=n1s * sig_H[:, None], D=n1s * sig_D[:, None],
               Tn3=n1s * sig_T3[:, None], WRONG=n1s * sig_W[:, None])
    print(f"  kappa_0 (H, T_n = Te) range: {kap['H'].min():.3e} to "
          f"{kap['H'].max():.3e} per cm; max at "
          f"[{','.join(str(int(k)) for k in np.unravel_index(kap['H'].argmax(), kap['H'].shape))}]")
    print(f"  n(1s)/n_ion range: {n1s_per_ion.min():.3e} to "
          f"{n1s_per_ion.max():.3e}")
    print()

    # ── item 2: the quoted table ────────────────────────────────────────────
    print("=" * 78)
    print("2   kappa_0 per cm at the thesis's table points and the benchmark")
    print("=" * 78)
    ib, jb = ctx.nearest_point(BENCH_TE, BENCH_NE)
    assert (ib, jb) == (23, 5), (
        f"benchmark Te={BENCH_TE}, ne={BENCH_NE:.3e} maps to [{ib},{jb}], "
        f"not [23,5]; the grid is not the one CLAUDE.md describes")
    print(f"  benchmark (Te {BENCH_TE}, ne {BENCH_NE:.3e}) -> [{ib},{jb}] "
          f"= Te {Te[ib]:.4f}, ne {ne[jb]:.4e}   [asserted (23,5)]")
    print()
    print(f"  {'pt':>7s} {'Te':>7s} {'ne':>10s} {'n(1s)':>11s} "
          f"{'H (T_n=Te)':>12s} {'deuterium':>12s} {'T_n=3eV':>12s} "
          f"{'sqrt(kT/m) WRONG':>17s}")
    for (i, j) in [(0, 4), (3, 4), (10, 4), (15, 4), (23, 4), (23, 5)]:
        print(f"  [{i:2d},{j}] {Te[i]:7.3f} {ne[j]:10.3e} {n1s[i, j]:11.4e} "
              f"{kap['H'][i, j]:12.5g} {kap['D'][i, j]:12.5g} "
              f"{kap['Tn3'][i, j]:12.5g} {kap['WRONG'][i, j]:17.5g}")
    print()
    print("  P2  hydrogen vs chapter 6 table (~155-159):")
    for (i, j), v in CH6_TABLE.items():
        print("    " + record("P2", f"kappa_0 H [{i},{j}] per cm",
                              kap['H'][i, j], v, "chapter6.tex table"))
    print("  P3  sqrt(kT/m) WRONG width vs chapter 5 prose (~2148-2151):")
    for (i, j), v in CH5_PROSE.items():
        print("    " + record("P3", f"kappa_0 WRONG-width [{i},{j}] per cm",
                              kap['WRONG'][i, j], v, "chapter5.tex sec:optical_depth",
                              tol_digits=2))
    print("  P3' deuterium at [0,4] (also reproduces 114 to 3 digits?):")
    print("    " + record("P3", "kappa_0 deuterium [0,4] per cm",
                          kap['D'][0, 4], 113.6, "task expectation"))
    print("    " + record("P3", "kappa_0 deuterium [0,4] vs ch5 114",
                          kap['D'][0, 4], 114.0, "chapter5.tex", tol_digits=3))
    print("  P4  T_n = 3 eV at [0,4]:")
    print("    " + record("P4", "kappa_0 H, T_n=3 eV [0,4] per cm",
                          kap['Tn3'][0, 4], 46.4, "task expectation"))
    print("  benchmark [23,5] (chapter 6 ~177 calls the [23,4] value 'the "
          "benchmark point'; it is not):")
    print("    " + record("2", "kappa_0 H [23,5] per cm", kap['H'][23, 5]))
    print("    " + record("2", "kappa_0 WRONG-width [23,5] per cm",
                          kap['WRONG'][23, 5]))
    print(f"    ratio of [23,4] to [23,5] hydrogen: "
          f"{kap['H'][23, 4]/kap['H'][23, 5]:.4f}; neither the table's 0.00163 "
          f"nor the prose's 0.0023 is the benchmark value")
    print()

    # ── item 3: tau_0 = 1 boundary per density column ───────────────────────
    print("=" * 78)
    print("3   tau_0 = 1 boundary per density column (log-log interpolation in Te)")
    print("=" * 78)
    print(f"  columns: " + " ".join(f"{n:9.2e}" for n in ne))
    bounds = {}
    for D in D_list:
        for defn, fac in (("half-slab", 0.5), ("full-path", 1.0)):
            for iso in ("H", "D"):
                c = crossing(kap[iso] * D * fac, 1.0, Te)
                bounds[(D, defn, iso)] = c
                fin = np.isfinite(c)
                print(f"  D={D:4.1f} {defn:9s} {iso}: "
                      + " ".join(f"{v:9.3f}" for v in c)
                      + f"   -> {np.nanmin(c):.3f} to {np.nanmax(c):.3f} eV"
                      + f"  ({fin.sum()} crossings, {(c[fin] > 2.0).sum()} above 2 eV)")
        c = crossing(kap["WRONG"] * D * 0.5, 1.0, Te)
        bounds[(D, "half-slab", "WRONG")] = c
        print(f"  D={D:4.1f} half-slab WRONG: "
              + " ".join(f"{v:9.3f}" for v in c)
              + f"   -> {np.nanmin(c):.3f} to {np.nanmax(c):.3f} eV")
    c5 = bounds[(5.0, "half-slab", "H")]
    print("  P5  D = 5 cm half-slab hydrogen:")
    print("    " + record("P5", "tau_half=1 boundary D=5 H, min Te [eV]",
                          float(np.nanmin(c5)), 1.125, "make_story_figures.py; task"))
    print("    " + record("P5", "tau_half=1 boundary D=5 H, max Te [eV]",
                          float(np.nanmax(c5)), 1.984, "make_story_figures.py; task"))
    cw = bounds[(5.0, "half-slab", "WRONG")]
    shift = np.nanmax(np.abs(cw / c5 - 1.0))
    print(f"    the sqrt(2) width error moves the D = 5 boundary by at most "
          f"{100*shift:.2f}% in Te ({np.nanmin(cw):.3f} to {np.nanmax(cw):.3f} eV)")
    summary.append(("P5", "max |dTe/Te| of D=5 boundary, WRONG vs H", shift, "",
                    "computed", ""))
    n_above2 = sum(int((bounds[(D, 'half-slab', 'H')][np.isfinite(
        bounds[(D, 'half-slab', 'H')])] > 2.0).sum()) for D in D_list)
    print(f"    half-slab hydrogen crossings above 2 eV over all D: {n_above2} "
          f"(make_story_figures reports these all lie at D = {max(D_list):g} cm)")
    print()

    # ── item 4: census recount ──────────────────────────────────────────────
    print("=" * 78)
    print("4   breakdown census against tau(5 cm), PRE-step (i, j) of each pair")
    print("=" * 78)
    dm_path = ROOT / "validation/divertor_map/divertor_map.csv"
    dm_header, dm = read_divertor_map(dm_path, sha_L, sha_S, sha_si)
    print(f"  source {dm_path.relative_to(ROOT)}; its header carries the live "
          f"L_grid, S_grid and state_index hashes [checked]")
    print("  convention: the csv's (i, j, Te, ne) are the pre-step point "
          "(verify_divertor_map.py:196, Te=Te[i]); the thesis counts pairs by "
          "that point, and so does this recount. The post-step point is 5% "
          "hotter and is not stored.")
    ok = [r for r in dm if r["window_ok"] == "True"]
    bd = [r for r in ok if float(r["lo_ELM_crash"]) > 0.10]
    # the csv's Te, ne must be the live grid at (i, j) or the indices are stale
    for r in bd:
        i, j = int(r["i"]), int(r["j"])
        if rel(float(r["Te"]), Te[i]) > 1e-9 or rel(float(r["ne"]), ne[j]) > 1e-9:
            raise RuntimeError(f"divertor_map.csv row (i={i}, j={j}) has "
                               f"Te={r['Te']}, ne={r['ne']} but the live grid has "
                               f"{Te[i]}, {ne[j]}; the map's indices are stale")
    n_cold = sum(1 for r in bd if float(r["Te"]) < 2.0)
    print("  " + record("P6", "window_ok rows", len(ok), 680, "chapter 5 table"))
    print("  " + record("P6", "breakdown pairs (lo_ELM_crash > 0.10)", len(bd),
                        CH5_CENSUS["n_pairs"], "chapter 5 ~1549"))
    print("  " + record("P6", "of which pre-step Te < 2 eV", n_cold,
                        CH5_CENSUS["n_cold"], "chapter 5 ~1550"))
    variants = [
        ("(a) half-slab, hydrogen",         "H", 0.5, (85, 15)),
        ("(b) full-path, hydrogen",         "H", 1.0, (100, 23)),
        ("(c) full-path, sqrt(kT/m) WRONG", "WRONG", 1.0, (105, 26)),
        ("(d) half-slab, sqrt(kT/m) WRONG", "WRONG", 0.5, None),
        ("(e) half-slab, deuterium",        "D", 0.5, None),
        ("(f) full-path, deuterium",        "D", 1.0, None),
        ("(g) half-slab, hydrogen, T_n=3 eV", "Tn3", 0.5, None),
    ]
    census = {}
    for name, iso, fac, expect in variants:
        taus = np.array([kap[iso][int(r["i"]), int(r["j"])] * 5.0 * fac for r in bd])
        c1, c100 = int((taus > 1).sum()), int((taus > 100).sum())
        census[name] = (c1, c100)
        line = f"  {name:<36s} tau(5 cm) > 1: {c1:4d}   > 100: {c100:4d}"
        if expect:
            e1, e100 = expect
            st1 = "REPRODUCED" if c1 == e1 else "NOT REPRODUCED"
            st2 = "REPRODUCED" if c100 == e100 else "NOT REPRODUCED"
            line += f"   expected {e1}/{e100}: {st1}/{st2}"
            summary.append(("P6", f"census {name} n(tau>1)", c1, e1, "task", st1))
            summary.append(("P6", f"census {name} n(tau>100)", c100, e100, "task", st2))
        else:
            summary.append(("4", f"census {name} n(tau>1)", c1, "", "computed", ""))
            summary.append(("4", f"census {name} n(tau>100)", c100, "", "computed", ""))
        print(line)
    thesis = (CH5_CENSUS["n_tau_gt_1"], CH5_CENSUS["n_tau_gt_100"])
    matches = [n for n, v in census.items() if v == thesis]
    print(f"  the thesis's {thesis[0]}/{thesis[1]} (chapter 5 ~1551, chapter 6 "
          f"~192) is reproduced by: {matches if matches else 'NONE'}")
    summary.append(("P6", "variants reproducing thesis 105/26", len(matches), 1,
                    "chapter 5/6", "; ".join(matches) if matches else "NONE"))
    print()

    # ── item 5: escape factors ──────────────────────────────────────────────
    print("=" * 78)
    print("5   escape factors: one-shot (thin n(1s)) vs self-consistent (log)")
    print("=" * 78)
    lt_csv = ROOT / "validation/lyman_trapping/lyman_trapping.csv"
    lt_txt = ROOT / "validation/lyman_trapping/lyman_trapping.txt"
    if lt_csv.is_file():
        with lt_csv.open() as fh:
            cols = fh.readline().strip().split(",")
        has_theta = any("theta" in c.lower() for c in cols)
        print(f"  {lt_csv.relative_to(ROOT)} columns carry Theta? {has_theta}"
              + ("" if has_theta else "  -> reading the run log instead"))
    blocks = read_lyman_trapping_log(lt_txt, sha_L16)
    print(f"  {lt_txt.relative_to(ROOT)}: L_grid (canon) hash matches live "
          f"array [checked]; D runs found: {sorted(blocks)}")

    tau_half_04 = kap["H"][0, 4] * 5.0 / 2.0
    th_one = float(escape_factor_quadrature(tau_half_04))
    th_sc = log_theta_at(blocks, 5.0, Te[0], ne[4])
    print("  P7  [0,4], D = 5 cm:")
    print("    " + record("P7", "tau_half [0,4] D=5 H", tau_half_04, 200.7, "task"))
    print("    " + record("P7", "tau_half [0,4] D=5 H vs ch6 '201'", tau_half_04,
                          CH6_ONE_SHOT["tau_half"], "chapter 6 ~166"))
    print("    " + record("P7", "one-shot Theta_P [0,4] D=5", th_one, 1.178e-3,
                          "task", tol_digits=4))
    print("    " + record("P7", "one-shot Theta_P vs ch6 '1.2e-3'", th_one,
                          CH6_ONE_SHOT["theta"], "chapter 6 ~167", tol_digits=2))
    print("    " + record("P7", "self-consistent Theta_Lya [0,4] D=5 (log)", th_sc,
                          1.44e-2, "task; lyman_trapping.txt"))
    print(f"    one-shot: A_eff(2p->1s) reduced by {-np.log10(th_one):.2f} orders; "
          f"self-consistent: {-np.log10(th_sc):.2f} orders "
          f"(trapping depletes n(1s) by {n1s[0,4]/[r for r in blocks[5.0]['rows'] if abs(r[0]-Te[0])<6e-4 and rel(r[1],ne[4])<6e-4][0][2]:.2f}x, "
          f"which is why the self-consistent factor is {th_sc/th_one:.1f}x larger)")
    summary.append(("P7", "one-shot orders of A reduction [0,4] D=5",
                    -np.log10(th_one), "", "computed", ""))
    summary.append(("P7", "self-consistent orders of A reduction [0,4] D=5",
                    -np.log10(th_sc), "", "lyman_trapping.txt", ""))
    Dmin = min(blocks, key=lambda d: blocks[d]["theta_min"])
    print("    " + record("P7", f"self-consistent grid minimum Theta_Lya (D={Dmin:g})",
                          blocks[Dmin]["theta_min"], 7.8e-4,
                          "task; lyman_trapping.txt", tol_digits=2))
    print(f"    grid minimum over D: "
          + ", ".join(f"D={d:g}: {blocks[d]['theta_min']:.4e} "
                      f"({-np.log10(blocks[d]['theta_min']):.2f} orders)"
                      for d in sorted(blocks)))
    if Dmin != max(blocks):
        print(f"    NOTE: minimum is at D = {Dmin:g}, not the thickest slab")
    # one-shot at [0,7] -- a DIFFERENT point (Te = 1 eV, ne = 1e15)
    print(f"  [0,7] (Te {Te[0]:.3f}, ne {ne[7]:.3e}) one-shot -- a different "
          f"grid point from [0,4], quoted for the chapter-3 '3e-5' comparison:")
    for D in (5.0, 10.0):
        t = kap["H"][0, 7] * D / 2.0
        th = float(escape_factor_quadrature(t))
        exp = {5.0: 7e-5, 10.0: 3.7e-5}[D]
        print("    " + record("5", f"one-shot Theta_P [0,7] D={D:g} (tau_half {t:.1f})",
                              th, exp, "task '~'", tol_digits=1))
    print()

    # ── item 6: benchmark escape factor from the log ────────────────────────
    print("=" * 78)
    print("6   benchmark [23,5] self-consistent Theta_Lya from lyman_trapping.txt")
    print("=" * 78)
    for D, recd, tol in ((1.0, CH6_BENCH_D1_THETA, 4), (20.0, CH6_BENCH_D20_THETA, 3)):
        th = log_theta_at(blocks, D, Te[23], ne[5])
        print("  " + record("P8", f"self-consistent Theta_Lya [23,5] D={D:g}", th,
                            recd, f"chapter 6 ~286-287", tol_digits=tol))
    th1 = log_theta_at(blocks, 1.0, Te[23], ne[5])
    print("  " + record("P8", "Theta_Lya [23,5] D=1 vs artifact 0.99862", th1,
                        0.99862, "task; lyman_trapping.txt", tol_digits=5))
    print(f"  1 - Theta at D = 1 cm: {1-th1:.2e} (chapter 6's 0.9999 implies "
          f"1e-4; the log says {1-th1:.1e}, a factor {(1-th1)/1e-4:.0f} more absorption)")
    print()

    # ── write ───────────────────────────────────────────────────────────────
    if a.write:
        grid_csv = out_dir / "lyman_optical_depth.csv"
        with grid_csv.open("w", newline="") as fh:
            for line in prov_lines:
                fh.write(line + "\n")
            fh.write("# n1s = n_e * [-L^{-1} S]_1s (n_ion = n_e); kappa0 = sigma0 * n1s "
                     "[per cm]; tau_half = kappa0*D/2; tau_full = kappa0*D\n")
            fh.write(f"# H: m = {M_H:.5e} g (proton), T_n = Te; D: m = {M_D:.5e} g "
                     f"(deuteron), T_n = Te; Tn3: hydrogen, T_n = {T_N_FIXED:g} eV; "
                     f"WRONG: hydrogen with the superseded sqrt(kT/m) width\n")
            cols = ["i", "j", "Te_eV", "ne_cm3", "n1s_per_ion", "n1s_cm3",
                    "sigma0_H_cm2", "kappa0_H", "kappa0_D", "kappa0_Tn3", "kappa0_WRONG"]
            for D in D_list:
                tag = f"{D:g}cm"
                cols += [f"tau_half_H_{tag}", f"tau_full_H_{tag}",
                         f"tau_half_D_{tag}", f"tau_half_Tn3_{tag}",
                         f"tau_full_WRONG_{tag}"]
            w = csv.writer(fh)
            w.writerow(cols)
            for i in range(nT):
                for j in range(nN):
                    row = [i, j, f"{Te[i]:.10g}", f"{ne[j]:.10g}",
                           f"{n1s_per_ion[i, j]:.10e}", f"{n1s[i, j]:.10e}",
                           f"{sig_H[i]:.10e}", f"{kap['H'][i, j]:.10e}",
                           f"{kap['D'][i, j]:.10e}", f"{kap['Tn3'][i, j]:.10e}",
                           f"{kap['WRONG'][i, j]:.10e}"]
                    for D in D_list:
                        row += [f"{kap['H'][i, j]*D/2:.10e}", f"{kap['H'][i, j]*D:.10e}",
                                f"{kap['D'][i, j]*D/2:.10e}", f"{kap['Tn3'][i, j]*D/2:.10e}",
                                f"{kap['WRONG'][i, j]*D:.10e}"]
                    w.writerow(row)
        sum_csv = out_dir / "lyman_optical_depth_summary.csv"
        with sum_csv.open("w", newline="") as fh:
            for line in prov_lines:
                fh.write(line + "\n")
            fh.write("# census rows come from validation/divertor_map/divertor_map.csv, "
                     "self-consistent Theta from validation/lyman_trapping/"
                     "lyman_trapping.txt; both checked against the live hashes\n")
            w = csv.writer(fh)
            w.writerow(["item", "quantity", "value", "recorded", "source", "status"])
            for item, q, v, recd, src, st in summary:
                w.writerow([item, q, f"{v:.10g}" if isinstance(v, float) else v,
                            recd, src, st])
            for (D, defn, iso), c in bounds.items():
                w.writerow(["3", f"tau=1 boundary D={D:g} {defn} {iso} per column [eV]",
                            " ".join(f"{v:.4f}" for v in c), "", "computed", ""])
        print(f"  wrote {grid_csv}")
        print(f"  wrote {sum_csv}")
        print(f"  wrote {out_dir / 'lyman_optical_depth.txt'}")

    # ── verdict ─────────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("VERDICT on the predictions")
    print("=" * 78)
    bad = [s for s in summary if s[5].startswith("NOT")]
    for s in summary:
        if s[3] != "":
            print(f"  {s[0]:<3s} {s[1]:<52s} {s[2]!s:>14s} vs {s[3]!s:<10s} {s[5]}")
    print(f"  {len(bad)} of {sum(1 for s in summary if s[3] != '')} recorded "
          f"comparisons NOT reproduced")

    if log_fh is not None:
        sys.stdout = sys.__stdout__
        log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
