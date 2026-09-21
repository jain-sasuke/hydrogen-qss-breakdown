#!/usr/bin/env python
"""
verify_tn_boundary.py
=====================
Neutral-temperature sensitivity of the Lyman-alpha tau_half = 1 boundary
(chapter 5 sec:optical_depth, chapter 6 sec:opacity), as a stamped run.

WHY THIS EXISTS
---------------
The Lyman-alpha line-centre cross-section is built from a Doppler width
sqrt(2 k T_n / m), and the project sets the neutral temperature T_n equal to
the electron temperature. sigma_0 therefore scales as T_n^(-1/2): colder
absorbing atoms (wall-reflected, charge-exchange cooled) have a narrower line
and a LARGER cross-section, so the plasma is more opaque and the tau_half = 1
boundary moves to higher Te. A Round 5 reviewer objected that the thesis
tested only the hotter case (T_n = 3 eV Franck-Condon atoms,
verify_lyman_optical_depth.py column 'Tn3'; trapped census 170/680 -> 173/680
at D = 5 cm, chapter 6 ~409) and never the colder one that moves the boundary
the wrong way for the scope sentence "of the 24 optical-depth crossings, 2 lie
above 2 eV, both at D = 20 cm" (chapter 5, sec:optical_depth).
verify_lyman_optical_depth.py hard-codes T_N_FIXED = 3.0 and has no T_n
option. This script imports that script's functions (sigma0_independent,
crossing, read_divertor_map, digits_agree, the constants) rather than
re-deriving them, adds the colder conventions, and reads
validation/lyman_optical_depth/ as its gate.

METHOD
------
n(1s) = n_e [-L^{-1} S]_1s at every grid point (the same one-line expression
as verify_lyman_optical_depth.py:440, which does not factor it into an
importable function; gate G1 checks it against that script's stamped csv to
1e-9). kappa_0 = sigma_0(T_n) n(1s) per cm and tau_half = kappa_0 D/2 for
D = 1, 5, 20 cm under four neutral-temperature conventions:
    Te     T_n = Te            (the project's)
    3eV    T_n = 3 eV          (the existing Franck-Condon test, T_N_FIXED)
    1eV    T_n = 1 eV          (colder atoms)
    0.5eV  T_n = 0.5 eV        (wall-reflected atoms)
For each density column j and each D the Te at which tau_half crosses 1 is
found by verify_lyman_optical_depth.crossing: linear interpolation of ln tau
against ln Te between the adjacent grid rows that bracket the sign change
('none' if the column never crosses). The shift of that crossing relative to
the T_n = Te convention is reported in eV and per cent, together with a
local-slope estimate of the same shift,
    d ln Te = 0.5 ln(Te0/T_n) / |s + 0.5|,
where s is d ln tau/d ln Te of the T_n = Te depth between the bracketing rows
and Te0 the T_n = Te crossing; +0.5 because a fixed T_n removes the Te^(-1/2)
of sigma_0. Crossings above 2 eV are counted per convention and per D.
The breakdown census at D = 5 cm is recounted the cheap way, as
verify_lyman_optical_depth.py item 4 does: the 202 breakdown pairs of
validation/divertor_map/divertor_map.csv (window_ok, lo_ELM_crash > 0.10),
counted by the pre-step point's tau_half(5 cm) > 1 and > 100 under each
convention. The thesis's 170/680 -> 173/680 is a DIFFERENT census: the
breakdown count of the self-consistently TRAPPED map rebuilt with
verify_lyman_trapping.py --t-at 3, which requires rebuilding L at 400 points
to a Theta <-> n(1s) fixed point. That rebuild is not run here (instructed),
and no artifact under validation/ records the T_at = 3 eV run; the script says
so rather than substituting the thin recount for it.

GATES (the run stops if any fails)
----------------------------------
G1  T_n = Te kappa_0 at [0,4], [3,4], [10,4], [15,4], [23,4] reproduces
    chapter 6's 80.3, 10.6, 0.224, 0.0255, 0.00163 per cm to 3 digits, and
    n1s_cm3, sigma0_H_cm2, kappa0_H and every tau_half_H column of
    validation/lyman_optical_depth/lyman_optical_depth.csv at all 400 rows to
    1e-9 relative; the T_n = Te tau_half = 1 boundaries reproduce the
    'tau=1 boundary D=... half-slab H per column' rows of the stamped summary
    csv to 1e-4 eV (they are printed to 4 decimals). The csv headers must
    carry the live L_grid, S_grid and state_index sha256.
G2  sigma0_independent(T) equals escape_factor.lyman_alpha_sigma0(T) to 1e-6
    relative at every grid Te and at 0.5, 1, 3 eV.
G3  the T_n = 3 eV column reproduces the stamped kappa0_Tn3 and
    tau_half_Tn3 columns at all 400 rows to 1e-9, and the census
    '(g) half-slab, hydrogen, T_n=3 eV' 78 / 12 of the summary csv.

PREDICTIONS (written before the first run)
------------------------------------------
P1  Identity of the code path: kappa_0(T_n)/kappa_0(T_n = Te) = sqrt(Te/T_n)
    at every grid point to 1e-12. At Te = 2 eV exactly the factors are 2.000
    (0.5 eV), 1.414 (1 eV), 0.8165 (3 eV); at the nearest grid row [15,:],
    Te = 2.0236 eV, they are 2.012, 1.4225, 0.8213.
P2  As stated by the task: a factor 2 in kappa_0 moves the tau_half = 1
    crossing upward by about 0.10 to 0.15 eV at D = 5 cm, and the crossings
    stay below 2 eV at D = 1 and 5 cm for every convention.
P2' This script's hand estimate from the stamped T_n = Te boundaries and the
    chapter-6 table (written before the run, disagreeing with P2's second
    clause): between the 1.60 and 2.02 eV rows at j = 4 the depth falls
    8.8-fold, s = -ln 8.8 / ln(2.0236/1.5999) = -9.2, so a fixed T_n gives
    d ln Te = 0.5 ln(Te0/T_n)/8.7. The D = 5 cm, j = 7 crossing sits at
    1.984 eV under T_n = Te: T_n = 1 eV (factor 1.41) moves it by +0.04 in
    ln Te to about 2.06 eV (the stamped deuterium and WRONG boundaries, a
    uniform sqrt 2, give 2.067 and are the closest proxy); T_n = 0.5 eV
    (factor 1.99) by +0.08 to about 2.15 eV. j = 6 (1.8025 eV) stays below
    2 eV under every convention (about 1.93 eV at 0.5 eV); at D = 1 cm the
    hottest crossing (1.667 eV) reaches about 1.78 eV at 0.5 eV. So P2's
    second clause is expected to FAIL at D = 5 cm, j = 7, for T_n = 1 and
    0.5 eV, by 0.06 to 0.17 eV. Shifts run 0.03 to 0.26 eV, largest at
    D = 20, j = 7, T_n = 0.5 eV (2.356 -> about 2.61 eV); none exceeds 0.5 eV.
P3  At D = 20 cm, crossings above 2 eV: T_n = Te 2 of 8 (j = 6, 7: the two
    the thesis reports); T_n = 3 eV 2 of 8 (about 2.07 and 2.32 eV); T_n = 1 eV
    2 of 8 with j = 5 marginal at about 1.98 eV (deuterium proxy 1.989);
    T_n = 0.5 eV 3 of 8 (j = 5 at about 2.06 eV, 6, 7). The task expected
    3 to 4 of 8 for 0.5 eV.
P4  Over the 24 crossings (3 D x 8 columns), above 2 eV: Te 2, 3 eV 2,
    1 eV 3, 0.5 eV 4. Every column crosses exactly once at every D and
    convention (tau_half monotone in Te), so there are 24 crossings each.
P5  Thin census of the 202 breakdown pairs at D = 5 cm, tau_half > 1 / > 100:
    Te 85/15 (stamped), 3 eV 78/12 (stamped, G3), 1 eV between 85/15 and the
    deuterium proxy 93/19, 0.5 eV above 93/19 (about 95-100 / 19-22).

REFUTING OBSERVATION (for the chapter 5 scope sentence)
------------------------------------------------------
Any convention moving a D = 5 cm crossing above 2 eV, or a boundary shift
exceeding 0.5 eV anywhere. P2' predicts the first clause fires for T_n = 1 and
0.5 eV at j = 7. If it does, "of the 24 optical-depth crossings, 2 lie above
2 eV, both at D = 20 cm" holds only under T_n = Te and must carry that
condition; whether the 2 eV working boundary survives is then answered by how
far above 2 eV the colder crossings reach.

OUTPUTS (with --write)
----------------------
validation/tn_boundary/tn_boundary.csv            per grid point: n1s, kappa_0
                                                  and tau_half per convention, D
validation/tn_boundary/tn_boundary_crossings.csv  per (D, convention, j)
validation/tn_boundary/tn_boundary_summary.csv    every quoted number, gates
validation/tn_boundary/tn_boundary.txt            this run's log
Each carries a '#' header with the script name, date, interpreter and the
sha256 of L_grid.npy, S_grid.npy, state_index.csv and the two stamped inputs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── repo wiring: import the parent script's functions, not copies of them ────
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import verify_lyman_optical_depth as vlod                 # noqa: E402
from cr_context import CRContext                          # noqa: E402
from escape_factor import lyman_alpha_sigma0              # noqa: E402  (path set by vlod)

ROOT = vlod.ROOT
SCRIPT = _HERE.name
sigma0_independent = vlod.sigma0_independent
crossing = vlod.crossing
read_hashed_csv = vlod.read_divertor_map     # generic: checks '#' header hashes, returns rows
digits_agree = vlod.digits_agree
rel = vlod.rel

# name, T_n [eV] (None = Te)
CONVENTIONS = [("Te", None), ("3eV", vlod.T_N_FIXED), ("1eV", 1.0), ("0.5eV", 0.5)]
D_LIST = [1.0, 5.0, 20.0]
CH6_TABLE = vlod.CH6_TABLE                    # {(i, j): per cm}
STAMPED_TN3_CENSUS = (78, 12)                 # lyman_optical_depth_summary.csv row (g)
THESIS_ABOVE2 = 2                             # chapter 5: 2 of 24, both at D = 20 cm
TRAPPED_CENSUS_TN3 = "170/680 -> 173/680 at D = 5 cm (chapter 6 ~409-410)"


def tn_label(name: str, Tn) -> str:
    return "T_n = Te" if Tn is None else f"T_n = {Tn:g} eV"


def bracket_slope(field_col: np.ndarray, Te: np.ndarray) -> tuple[float, int]:
    """d ln field / d ln Te between the two grid rows bracketing the first
    sign change of ln field (the rows crossing() interpolates between)."""
    y = np.log(field_col)
    idx = np.where(np.diff(np.sign(y)))[0]
    if len(idx) == 0:
        return float("nan"), -1
    q = int(idx[0])
    s = (y[q + 1] - y[q]) / (np.log(Te[q + 1]) - np.log(Te[q]))
    return float(s), q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else ROOT / "validation" / "tn_boundary"

    log: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    summary: list[tuple] = []   # (item, quantity, value, recorded, source, status)

    def record(item, quantity, value, recorded=None, source="", tol_digits=3):
        if recorded is None:
            summary.append((item, quantity, value, "", source, ""))
            return f"{quantity} = {value:.6g}"
        d = digits_agree(value, recorded)
        need = min(tol_digits, len(f"{recorded:.10g}".replace(".", "").lstrip("0")))
        status = "REPRODUCED" if d >= need else "NOT REPRODUCED"
        summary.append((item, quantity, value, recorded, source, f"{status} ({d} digits)"))
        return (f"{quantity} = {value:.6g}   recorded {recorded:g} [{source}]"
                f"  -> {status}, {d} digits, rel {rel(value, recorded):.2e}")

    # ── load through cr_context; never redefine a grid ──────────────────────
    ctx = CRContext.load(root=ROOT)
    Te, ne, L = ctx.te_grid, ctx.ne_grid, ctx.L_grid
    g = ctx.ground_index
    nT, nN = len(Te), len(ne)
    L_path = ROOT / "data/processed/cr_matrix/L_grid.npy"
    S_path = ROOT / "data/processed/cr_matrix/S_grid.npy"
    si_path = Path(ctx.state_index_path)
    lod_csv = ROOT / "validation/lyman_optical_depth/lyman_optical_depth.csv"
    lod_sum = ROOT / "validation/lyman_optical_depth/lyman_optical_depth_summary.csv"
    dm_csv = ROOT / "validation/divertor_map/divertor_map.csv"
    for p in (S_path, lod_csv, lod_sum, dm_csv):
        if not p.is_file():
            raise FileNotFoundError(f"required input missing: {p}")
    S = np.load(S_path)
    if S.shape != L.shape[:3]:
        raise RuntimeError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    if ctx.labels[g] != "1S":
        raise RuntimeError(f"ground state label is {ctx.labels[g]!r}, not '1S'")

    sha_L, sha_S, sha_si = (vlod.sha256_file(L_path), vlod.sha256_file(S_path),
                            vlod.sha256_file(si_path))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    prov_lines = [
        f"# generated {stamp} by {SCRIPT}",
        f"# interpreter {sys.executable}  numpy {np.__version__}",
        f"# L_grid.npy      sha256 {sha_L}",
        f"# S_grid.npy      sha256 {sha_S}",
        f"# state_index.csv sha256 {sha_si}  ({si_path.relative_to(ROOT)})",
        f"# lyman_optical_depth.csv         sha256 {vlod.sha256_file(lod_csv)}",
        f"# lyman_optical_depth_summary.csv sha256 {vlod.sha256_file(lod_sum)}",
        f"# divertor_map.csv                sha256 {vlod.sha256_file(dm_csv)}",
        f"# functions imported from {vlod.SCRIPT}: sigma0_independent, crossing, "
        f"read_divertor_map, digits_agree; constants M_H, T_N_FIXED = {vlod.T_N_FIXED:g}",
    ]
    for line in prov_lines:
        say(line)
    say("=" * 78)
    say("NEUTRAL-TEMPERATURE SENSITIVITY OF THE LYMAN-ALPHA tau_half = 1 BOUNDARY")
    say("=" * 78)
    say(ctx.describe())
    say(f"  S_grid         : {S.shape}")
    say(f"  conventions    : " + ", ".join(tn_label(n, t) for n, t in CONVENTIONS))
    say(f"  slabs D [cm]   : {D_LIST}   (tau_half = kappa_0 D/2)")
    say(f"  Te grid spacing: ln(Te[i+1]/Te[i]) = {np.log(Te[1]/Te[0]):.5f} "
        f"(min) to {np.log(Te[-1]/Te[-2]):.5f} (max)")
    say()

    # ── stamped inputs, hash-checked ────────────────────────────────────────
    _, lod_rows = read_hashed_csv(lod_csv, sha_L, sha_S, sha_si)
    _, sum_rows = read_hashed_csv(lod_sum, sha_L, sha_S, sha_si)
    _, dm_rows = read_hashed_csv(dm_csv, sha_L, sha_S, sha_si)
    if len(lod_rows) != nT * nN:
        raise RuntimeError(f"{lod_csv} has {len(lod_rows)} rows, expected {nT*nN}")
    say(f"  {lod_csv.relative_to(ROOT)}: header carries live L/S/state_index hashes [checked], {len(lod_rows)} rows")
    say(f"  {lod_sum.relative_to(ROOT)}: header carries live hashes [checked], {len(sum_rows)} rows")
    say(f"  {dm_csv.relative_to(ROOT)}: header carries live hashes [checked], {len(dm_rows)} rows")
    say()

    # ── G2: sigma_0 code path vs the module ─────────────────────────────────
    say("=" * 78)
    say("G2  sigma0_independent(T) vs escape_factor.lyman_alpha_sigma0(T)")
    say("=" * 78)
    worst = 0.0
    for T in list(Te) + [0.5, 1.0, vlod.T_N_FIXED]:
        worst = max(worst, rel(sigma0_independent(T), float(lyman_alpha_sigma0(T))))
    say(f"  worst relative difference over {nT} grid Te and 0.5, 1, 3 eV: {worst:.2e}")
    if worst > 1e-6:
        raise AssertionError(f"G2 FAILED: sigma_0 rebuild differs from the module by {worst:.2e}")
    say("  G2 PASSED")
    for T in (0.5, 1.0, 2.0, vlod.T_N_FIXED):
        say(f"    sigma_0({T:g} eV) = {sigma0_independent(T):.6e} cm^2")
    summary.append(("G2", "worst rel sigma_0 rebuild vs module", worst, "", "computed", "PASSED"))
    say()

    # ── n(1s), kappa_0 per convention ───────────────────────────────────────
    n1s_per_ion = np.empty((nT, nN))
    for i in range(nT):
        for j in range(nN):
            n1s_per_ion[i, j] = np.linalg.solve(L[i, j], -S[i, j])[g]
    if not np.all(np.isfinite(n1s_per_ion)) or np.any(n1s_per_ion <= 0):
        raise RuntimeError("non-positive or non-finite n(1s) from the CRE solve")
    n1s = n1s_per_ion * ne[None, :]
    sig: dict[str, np.ndarray] = {}
    for name, Tn in CONVENTIONS:
        if Tn is None:
            sig[name] = np.array([sigma0_independent(t) for t in Te])
        else:
            sig[name] = np.full(nT, sigma0_independent(Tn))
    kap = {name: n1s * sig[name][:, None] for name in sig}

    # ── G1: T_n = Te against chapter 6 and the stamped csv ──────────────────
    say("=" * 78)
    say("G1  T_n = Te kappa_0 vs chapter 6 table and lyman_optical_depth.csv")
    say("=" * 78)
    g1_fail = False
    for (i, j), v in CH6_TABLE.items():
        line = record("G1", f"kappa_0 (T_n=Te) [{i},{j}] per cm", kap["Te"][i, j], v,
                      "chapter6.tex table")
        say("  " + line)
        if "NOT" in line:
            g1_fail = True
    lod = {(int(r["i"]), int(r["j"])): r for r in lod_rows}
    worst_cols: dict[str, float] = {}
    for i in range(nT):
        for j in range(nN):
            r = lod[(i, j)]
            checks = {"n1s_cm3": (n1s[i, j], float(r["n1s_cm3"])),
                      "sigma0_H_cm2": (sig["Te"][i], float(r["sigma0_H_cm2"])),
                      "kappa0_H": (kap["Te"][i, j], float(r["kappa0_H"])),
                      "kappa0_Tn3": (kap["3eV"][i, j], float(r["kappa0_Tn3"]))}
            for D in D_LIST:
                checks[f"tau_half_H_{D:g}cm"] = (kap["Te"][i, j] * D / 2, float(r[f"tau_half_H_{D:g}cm"]))
                checks[f"tau_half_Tn3_{D:g}cm"] = (kap["3eV"][i, j] * D / 2, float(r[f"tau_half_Tn3_{D:g}cm"]))
            for k, (mine, theirs) in checks.items():
                worst_cols[k] = max(worst_cols.get(k, 0.0), rel(mine, theirs))
    say("  all 400 rows vs the stamped csv, worst relative difference:")
    for k, v in worst_cols.items():
        say(f"    {k:22s} {v:.2e}")
        summary.append(("G1" if "Tn3" not in k else "G3", f"worst rel vs stamped csv {k}", v, "", lod_csv.name, ""))
    if max(v for k, v in worst_cols.items() if "Tn3" not in k) > 1e-9:
        g1_fail = True
    # boundaries for T_n = Te vs the stamped summary rows
    bounds: dict[tuple[float, str], np.ndarray] = {}
    for D in D_LIST:
        for name, _ in CONVENTIONS:
            bounds[(D, name)] = crossing(kap[name] * D / 2, 1.0, Te)
    stamped_b = {}
    for r in sum_rows:
        q = r["quantity"]
        if q.startswith("tau=1 boundary D=") and "half-slab H per column" in q:
            D = float(q.split("D=")[1].split()[0])
            stamped_b[D] = np.array([float(x) for x in r["value"].split()])
    if sorted(stamped_b) != sorted(D_LIST):
        raise RuntimeError(f"stamped summary has boundary rows for D = {sorted(stamped_b)}, need {D_LIST}")
    say("  T_n = Te boundaries vs the stamped summary rows (printed to 4 decimals):")
    for D in D_LIST:
        d = np.abs(bounds[(D, "Te")] - stamped_b[D]).max()
        say(f"    D = {D:4.1f} cm: max |dTe| = {d:.1e} eV")
        summary.append(("G1", f"max |dTe| T_n=Te boundary D={D:g} vs stamped", d, "", lod_sum.name, ""))
        if d > 1e-4:
            g1_fail = True
    if g1_fail:
        raise AssertionError("G1 FAILED: the T_n = Te column does not reproduce the stamped artifact; do not read on")
    say("  G1 PASSED")
    say()

    # ── G3: T_n = 3 eV column and census (g) ────────────────────────────────
    say("=" * 78)
    say("G3  T_n = 3 eV column vs the stamped Tn3 columns and census (g) 78/12")
    say("=" * 78)
    w3 = max(v for k, v in worst_cols.items() if "Tn3" in k)
    say(f"  worst relative difference, kappa0_Tn3 and tau_half_Tn3 columns, 400 rows: {w3:.2e}")
    ok = [r for r in dm_rows if r["window_ok"] == "True"]
    bd = [r for r in ok if float(r["lo_ELM_crash"]) > 0.10]
    for r in bd:
        i, j = int(r["i"]), int(r["j"])
        if rel(float(r["Te"]), Te[i]) > 1e-9 or rel(float(r["ne"]), ne[j]) > 1e-9:
            raise RuntimeError(f"divertor_map.csv row (i={i}, j={j}) does not sit on the live grid")
    say(f"  divertor_map: window_ok {len(ok)}, breakdown pairs (lo_ELM_crash > 0.10) {len(bd)}")
    if (len(ok), len(bd)) != (680, 202):
        raise AssertionError(f"G3 FAILED: census base is {len(ok)}/{len(bd)}, the stamped run had 680/202")
    census: dict[str, tuple[int, int]] = {}
    for name, _ in CONVENTIONS:
        taus = np.array([kap[name][int(r["i"]), int(r["j"])] * 5.0 / 2 for r in bd])
        census[name] = (int((taus > 1).sum()), int((taus > 100).sum()))
    c3 = census["3eV"]
    say(f"  census (g) recount, T_n = 3 eV, half-slab, D = 5 cm: tau > 1: {c3[0]}, > 100: {c3[1]}"
        f"   stamped {STAMPED_TN3_CENSUS[0]}/{STAMPED_TN3_CENSUS[1]}")
    summary.append(("G3", "census T_n=3eV n(tau_half(5cm)>1)", c3[0], STAMPED_TN3_CENSUS[0], lod_sum.name,
                    "REPRODUCED" if c3[0] == STAMPED_TN3_CENSUS[0] else "NOT REPRODUCED"))
    summary.append(("G3", "census T_n=3eV n(tau_half(5cm)>100)", c3[1], STAMPED_TN3_CENSUS[1], lod_sum.name,
                    "REPRODUCED" if c3[1] == STAMPED_TN3_CENSUS[1] else "NOT REPRODUCED"))
    if w3 > 1e-9 or c3 != STAMPED_TN3_CENSUS:
        raise AssertionError("G3 FAILED: the T_n = 3 eV convention does not reproduce the stamped artifact")
    say("  G3 PASSED")
    say()

    # ── P1: the scaling identity ────────────────────────────────────────────
    say("=" * 78)
    say("P1  kappa_0(T_n)/kappa_0(T_n = Te) = sqrt(Te/T_n)")
    say("=" * 78)
    worst_id = 0.0
    for name, Tn in CONVENTIONS[1:]:
        ratio = kap[name] / kap["Te"]
        expect = np.sqrt(Te / Tn)[:, None]
        worst_id = max(worst_id, float(np.abs(ratio / expect - 1).max()))
    say(f"  worst |ratio/sqrt(Te/T_n) - 1| over the grid and three fixed conventions: {worst_id:.1e}")
    i2 = int(np.argmin(np.abs(Te - 2.0)))
    say(f"  nearest grid row to 2 eV: [{i2},:] Te = {Te[i2]:.4f} eV")
    for name, Tn in CONVENTIONS[1:]:
        f_exact = sigma0_independent(Tn) / sigma0_independent(2.0)
        f_grid = float(kap[name][i2, 0] / kap["Te"][i2, 0])
        say(f"    {tn_label(name, Tn):14s}: factor at Te = 2 eV exactly {f_exact:.4f}; at [{i2},:] {f_grid:.4f}")
        summary.append(("P1", f"kappa_0 factor {name} vs Te at 2 eV exactly", f_exact, "", "computed", ""))
        summary.append(("P1", f"kappa_0 factor {name} vs Te at grid row {i2}", f_grid, "", "computed", ""))
    summary.append(("P1", "worst |ratio/sqrt(Te/Tn)-1|", worst_id, "", "computed",
                    "PASSED" if worst_id < 1e-12 else "FAILED"))
    say()

    # ── the boundary under each convention ──────────────────────────────────
    say("=" * 78)
    say("RESULT 1  tau_half = 1 crossing Te [eV] per density column, per convention and D")
    say("=" * 78)
    say("  columns j: " + " ".join(f"{j:>9d}" for j in range(nN)))
    say("  ne [cm^-3]: " + " ".join(f"{n:9.2e}" for n in ne))
    cross_rows: list[list] = []
    above2: dict[tuple[float, str], int] = {}
    ncross: dict[tuple[float, str], int] = {}
    max_shift = (0.0, None)
    d5_above2: list[str] = []
    for D in D_LIST:
        say(f"\n  D = {D:g} cm")
        base = bounds[(D, "Te")]
        for name, Tn in CONVENTIONS:
            c = bounds[(D, name)]
            fin = np.isfinite(c)
            # exactly-once check: number of sign changes of ln tau per column
            nsc = [int(len(np.where(np.diff(np.sign(np.log(kap[name][:, j] * D / 2))))[0])) for j in range(nN)]
            ncross[(D, name)] = int(fin.sum())
            above2[(D, name)] = int((c[fin] > 2.0).sum())
            vals = " ".join(f"{v:9.3f}" if np.isfinite(v) else f"{'none':>9s}" for v in c)
            say(f"    {tn_label(name, Tn):14s} {vals}   crossings {fin.sum()}, above 2 eV {above2[(D, name)]}")
            if any(k != 1 for k in nsc):
                say(f"      NOTE sign changes per column {nsc} (not exactly one everywhere)")
            if name != "Te":
                sh = c - base
                pct = 100 * (c / base - 1)
                say(f"    {'  shift [eV]':14s} " + " ".join(f"{v:+9.3f}" if np.isfinite(v) else f"{'none':>9s}" for v in sh))
                say(f"    {'  shift [%]':14s} " + " ".join(f"{v:+9.2f}" if np.isfinite(v) else f"{'none':>9s}" for v in pct))
                # local-slope estimate
                est = np.full(nN, np.nan)
                for j in range(nN):
                    s, q = bracket_slope(kap["Te"][:, j] * D / 2, Te)
                    if np.isfinite(s) and np.isfinite(base[j]):
                        est[j] = base[j] * np.exp(0.5 * np.log(base[j] / Tn) / abs(s + 0.5)) - base[j]
                say(f"    {'  slope est.':14s} " + " ".join(f"{v:+9.3f}" if np.isfinite(v) else f"{'none':>9s}" for v in est)
                    + "   (0.5 ln(Te0/T_n)/|s+0.5| from the bracketing rows of the T_n = Te depth)")
                for j in range(nN):
                    if np.isfinite(sh[j]) and abs(sh[j]) > max_shift[0]:
                        max_shift = (float(abs(sh[j])), (D, name, j))
                    if D == 5.0 and np.isfinite(c[j]) and c[j] > 2.0:
                        d5_above2.append(f"{tn_label(name, Tn)} j={j} ({ne[j]:.2e}) at {c[j]:.3f} eV")
            for j in range(nN):
                s, q = bracket_slope(kap[name][:, j] * D / 2, Te)
                cross_rows.append([f"{D:g}", name, "Te" if Tn is None else f"{Tn:g}", j, f"{ne[j]:.6e}",
                                   f"{c[j]:.5f}" if np.isfinite(c[j]) else "none",
                                   f"{c[j]-base[j]:+.5f}" if (name != "Te" and np.isfinite(c[j])) else "",
                                   f"{100*(c[j]/base[j]-1):+.3f}" if (name != "Te" and np.isfinite(c[j])) else "",
                                   int(np.isfinite(c[j]) and c[j] > 2.0),
                                   f"{s:.4f}" if np.isfinite(s) else "", nsc[j]])
    say()
    say("  crossings above 2 eV, per convention (D = 1 / 5 / 20 cm, total of 24):")
    tot_above2: dict[str, int] = {}
    for name, Tn in CONVENTIONS:
        per = [above2[(D, name)] for D in D_LIST]
        tot = sum(per)
        tot_above2[name] = tot
        ntot = sum(ncross[(D, name)] for D in D_LIST)
        say(f"    {tn_label(name, Tn):14s} {per[0]} / {per[1]} / {per[2]}   total {tot} of {ntot}")
        for D in D_LIST:
            summary.append(("R1", f"crossings above 2 eV {name} D={D:g}", above2[(D, name)], "", "computed", ""))
        summary.append(("R1", f"crossings above 2 eV {name} total", tot,
                        THESIS_ABOVE2 if name == "Te" else "", "chapter 5 sec:optical_depth" if name == "Te" else "computed",
                        ("REPRODUCED" if tot == THESIS_ABOVE2 else "NOT REPRODUCED") if name == "Te" else ""))
    say(f"  largest |shift| anywhere: {max_shift[0]:.3f} eV at D = {max_shift[1][0]:g} cm, "
        f"{max_shift[1][1]}, j = {max_shift[1][2]}")
    summary.append(("R1", "largest |boundary shift| [eV]", max_shift[0], "", f"D={max_shift[1][0]:g} {max_shift[1][1]} j={max_shift[1][2]}", ""))
    # hottest crossing per convention at D = 5 and overall
    for name, Tn in CONVENTIONS:
        c5 = bounds[(5.0, name)]
        cmax = max(np.nanmax(bounds[(D, name)]) for D in D_LIST)
        say(f"  {tn_label(name, Tn):14s}: D = 5 cm boundary {np.nanmin(c5):.3f} to {np.nanmax(c5):.3f} eV; hottest crossing over all D {cmax:.3f} eV")
        summary.append(("R1", f"D=5 boundary max Te {name} [eV]", float(np.nanmax(c5)), "", "computed", ""))
        summary.append(("R1", f"hottest crossing over all D {name} [eV]", float(cmax), "", "computed", ""))
    say()

    # ── census at D = 5 cm (thin recount) ───────────────────────────────────
    say("=" * 78)
    say("RESULT 2  thin recount of the 202 breakdown pairs by tau_half(5 cm), pre-step point")
    say("=" * 78)
    say(f"  source {dm_csv.relative_to(ROOT)}: window_ok {len(ok)}, lo_ELM_crash > 0.10: {len(bd)}, "
        f"pre-step Te < 2 eV: {sum(1 for r in bd if float(r['Te']) < 2.0)}")
    for name, Tn in CONVENTIONS:
        c1, c100 = census[name]
        say(f"    {tn_label(name, Tn):14s} tau_half(5 cm) > 1: {c1:4d}   > 100: {c100:4d}")
        summary.append(("R2", f"thin census {name} n(tau_half(5cm)>1)", c1, 85 if name == "Te" else "", "computed",
                        ("REPRODUCED" if c1 == 85 else "NOT REPRODUCED") if name == "Te" else ""))
        summary.append(("R2", f"thin census {name} n(tau_half(5cm)>100)", c100, 15 if name == "Te" else "", "computed",
                        ("REPRODUCED" if c100 == 15 else "NOT REPRODUCED") if name == "Te" else ""))
    say()
    say("  NOT RECOUNTED HERE: the thesis's trapped-run census " + TRAPPED_CENSUS_TN3 + ".")
    say("  That is the breakdown count of the map rebuilt with self-consistent Lyman escape")
    say("  factors (verify_lyman_trapping.py --t-at 3): 400 trapped L matrices iterated to a")
    say("  Theta <-> n(1s) fixed point. It is not run here (instructed), and no artifact under")
    say("  validation/ records a T_at = 3 eV trapped run (lyman_trapping.txt holds only T_at = Te).")
    say("  The colder-T_n analogue of 170 -> 173 therefore has no number from this script.")
    say()

    # ── write ───────────────────────────────────────────────────────────────
    if a.write:
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_csv = out_dir / "tn_boundary.csv"
        with grid_csv.open("w", newline="") as fh:
            for line in prov_lines:
                fh.write(line + "\n")
            fh.write("# n1s = n_e [-L^{-1} S]_1s (n_ion = n_e); kappa0 = sigma0(T_n) n1s [per cm]; tau_half = kappa0 D/2\n")
            fh.write("# conventions: Te (T_n = Te), 3eV, 1eV, 0.5eV (fixed T_n); hydrogen, m = proton, width sqrt(2kT_n/m)\n")
            cols = ["i", "j", "Te_eV", "ne_cm3", "n1s_cm3"] + [f"kappa0_{n}" for n, _ in CONVENTIONS]
            for D in D_LIST:
                cols += [f"tau_half_{n}_{D:g}cm" for n, _ in CONVENTIONS]
            w = csv.writer(fh)
            w.writerow(cols)
            for i in range(nT):
                for j in range(nN):
                    row = [i, j, f"{Te[i]:.10g}", f"{ne[j]:.10g}", f"{n1s[i, j]:.10e}"]
                    row += [f"{kap[n][i, j]:.10e}" for n, _ in CONVENTIONS]
                    for D in D_LIST:
                        row += [f"{kap[n][i, j]*D/2:.10e}" for n, _ in CONVENTIONS]
                    w.writerow(row)
        cr_csv = out_dir / "tn_boundary_crossings.csv"
        with cr_csv.open("w", newline="") as fh:
            for line in prov_lines:
                fh.write(line + "\n")
            fh.write("# Te_cross: ln tau_half interpolated linearly in ln Te between the bracketing rows "
                     "(verify_lyman_optical_depth.crossing); shift relative to the T_n = Te convention; "
                     "slope = d ln tau/d ln Te between the bracketing rows; n_sign_changes should be 1\n")
            w = csv.writer(fh)
            w.writerow(["D_cm", "convention", "T_n_eV", "j", "ne_cm3", "Te_cross_eV", "shift_eV",
                        "shift_pct", "above_2eV", "slope_lnTau_lnTe", "n_sign_changes"])
            w.writerows(cross_rows)
        sum_csv = out_dir / "tn_boundary_summary.csv"
        with sum_csv.open("w", newline="") as fh:
            for line in prov_lines:
                fh.write(line + "\n")
            w = csv.writer(fh)
            w.writerow(["item", "quantity", "value", "recorded", "source", "status"])
            for item, q, v, recd, src, st in summary:
                w.writerow([item, q, f"{v:.10g}" if isinstance(v, float) else v, recd, src, st])
        say(f"  wrote {grid_csv}")
        say(f"  wrote {cr_csv}")
        say(f"  wrote {sum_csv}")

    # ── verdict ─────────────────────────────────────────────────────────────
    say()
    say("=" * 78)
    say("VERDICT on the predictions")
    say("=" * 78)
    say(f"  P1  scaling identity worst deviation {worst_id:.1e} (asserted < 1e-12): "
        f"{'HOLDS' if worst_id < 1e-12 else 'FAILS'}")
    d1_max = {n: float(np.nanmax(bounds[(1.0, n)])) for n, _ in CONVENTIONS}
    d5_max = {n: float(np.nanmax(bounds[(5.0, n)])) for n, _ in CONVENTIONS}
    say(f"  P2  (task) crossings stay below 2 eV at D = 1 and 5 cm for every convention: "
        f"{'HOLDS' if not d5_above2 and all(v < 2 for v in d1_max.values()) else 'FAILS'}")
    say(f"      D = 1 cm hottest crossing per convention: " + ", ".join(f"{n} {v:.3f}" for n, v in d1_max.items()))
    say(f"      D = 5 cm hottest crossing per convention: " + ", ".join(f"{n} {v:.3f}" for n, v in d5_max.items()))
    if d5_above2:
        say("      D = 5 cm crossings above 2 eV: " + "; ".join(d5_above2))
    say(f"  P2' (hand) D = 5 cm j = 7: 1eV about 2.06 eV, 0.5eV about 2.15 eV; found "
        f"{bounds[(5.0, '1eV')][7]:.3f} and {bounds[(5.0, '0.5eV')][7]:.3f} eV; "
        f"largest shift about 0.26 eV at D = 20, j = 7, 0.5eV; found {max_shift[0]:.3f} eV at "
        f"D = {max_shift[1][0]:g}, {max_shift[1][1]}, j = {max_shift[1][2]}")
    say(f"  P3  D = 20 cm above 2 eV: " + ", ".join(f"{n} {above2[(20.0, n)]}" for n, _ in CONVENTIONS)
        + "   (predicted Te 2, 3eV 2, 1eV 2, 0.5eV 3; task expected 3 to 4 for 0.5eV)")
    say(f"  P4  total above 2 eV of 24: " + ", ".join(f"{n} {tot_above2[n]}" for n, _ in CONVENTIONS)
        + "   (predicted 2, 2, 3, 4)")
    say(f"  P5  thin census D = 5 cm: " + ", ".join(f"{n} {census[n][0]}/{census[n][1]}" for n, _ in CONVENTIONS)
        + "   (predicted 85/15, 78/12, 88-92/16-18, 95-100/19-22)")
    say()
    refute_a = bool(d5_above2)
    refute_b = max_shift[0] > 0.5
    say(f"  REFUTER (a) a convention moves a D = 5 cm crossing above 2 eV: {'FIRED' if refute_a else 'did not fire'}")
    say(f"  REFUTER (b) a boundary shift exceeds 0.5 eV anywhere: {'FIRED' if refute_b else 'did not fire'} "
        f"(largest {max_shift[0]:.3f} eV)")
    bad = [s for s in summary if str(s[5]).startswith("NOT") or str(s[5]) == "FAILED"]
    say(f"  {len(bad)} of {sum(1 for s in summary if s[3] != '')} recorded comparisons NOT reproduced")
    for s in bad:
        say(f"    {s[0]} {s[1]}: {s[2]} vs {s[3]}")

    if a.write:
        (out_dir / "tn_boundary.txt").write_text("\n".join(log) + "\n")
        print(f"  wrote {out_dir / 'tn_boundary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
