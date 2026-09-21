#!/usr/bin/env python
"""
verify_window_membership_exposure.py
====================================
Window membership at k = 10, 20, 30, 50 AND 100 recounted from the chapter's
own one-grid-interval artifacts, the one-interval eps_plateau / M values for
the Chapter 5 k-window paragraph, and the distribution of tau_slow over the
defended window-passing pairs against the 100 us ELM exposure. A stamped
recount; no eigendecomposition is re-run here.

WHY THIS EXISTS
---------------
The Round 4 review of Chapter 5 found two things.
(a) The k-window paragraph (thesis_tex/chapter5.tex ~1861-1884) quotes
    eps_plateau = 0.32902966 (benchmark) and 0.24075077 (cold corner) and
    M = 4856 from validation/plateau_slowmode/ (Aug 2026), whose step is a
    hard-coded +0.6 eV ABSOLUTE step, not the chapter's one-grid-interval
    (+4.81 %) step. The chapter's own one-interval values live in
    validation/plateau_gridmap/plateau_gridmap.csv and
    validation/divertor_map/divertor_map.csv. The paragraph also asserts that
    "at k = 100 the benchmark itself is excluded"; the stamped sweep
    validation/window_sweep/ (11 Sep, verify_window_sweep.py) recounts
    membership at k = 10/20/30/50 only, never at k = 100.
(b) Chapter 5 ~line 166 says a 100 us exposure "spends its integration time
    looking at the plateau" although tau_slow is 22.7 us at the benchmark;
    whether that sentence is defensible depends on how tau_slow is
    distributed over the defended window-passing pairs, which no artifact
    tabulates.
The rule is that every number in the thesis comes from a run; this is that
run.

METHOD
------
Inputs are read-only stamped artifacts (sha256 of each is printed and written
into every output header; the sha256 lines that the producing scripts stamped
into the CSV headers are checked against the CURRENT pipeline files L_grid,
S_grid, state_index, and the run stops if any is stale):
  validation/plateau_gridmap/plateau_gridmap.csv   784 rows  (verify_plateau_gridmap.py)
  validation/divertor_map/divertor_map.csv          784 rows  (verify_divertor_map.py, k = 30)
  validation/window_sweep/window_sweep_summary.csv  k = 10/20/30/50 reference counts (G2)
  validation/plateau_slowmode/plateau_slowmode.csv  the +0.6 eV values the chapter now quotes (context only)
  validation/divertor_map_w{10,20,50}/divertor_map.csv  stored window_ok at k != 30, if present (G3)
tau_QSS in these CSVs is the chapter's tau_slow: 1/|lambda_0| of the
post-step operator L[k_new, j] for that (direction, i, j) pair, exactly as
verify_divertor_map.py computes it (NOT 1/|lambda_0| of L[i, j] itself; at
the benchmark pair heat[23,5] the post-step operator is L[24,5]).
Window criterion, replicated verbatim from verify_divertor_map.py line 176
with win_lo = win_hi = k:
    window_ok(k)  =  (k * tau_relax) < (tau_QSS / k)          [strict <]
which is M > k^2 up to rounding (k = 30: M > 900; k = 100: M > 1e4); both
forms are evaluated and any row on which they disagree is reported.
Census logic, replicated verbatim from verify_window_sweep.py:
    defended  = Te >= 2.0 eV                     (the "warm" criterion)
    warm(k)   = window_ok(k) & defended          (census denominator)
    dense(k)  = warm(k) & ne >= 1e14 cm^-3
    lo_100us  = the stored column lo_ELM_crash  (= eps_plateau*(tQ/td)*(1-exp(-td/tQ)), td = 1e-4 s;
                recomputed here from eps_plateau and tau_QSS and compared to the stored column)
    lo_506us  = eps_plateau*(tQ/506e-6)*(1-exp(-506e-6/tQ))   (Loarte 2003 Sec. 5; not a stored column)
    census100(k) = warm(k) & (lo_100us > 0.10);  worst = max lo_100us over census100 and its (direction,i,j)
    census506(k) = warm(k) & (lo_506us > 0.10)
    also census100 at thresholds 0.05 and 0.20.
Grid labels: Te[i] and ne[j] in the CSVs are checked against cr_context's
Te_grid_L / ne_grid_L (no grid is defined in this file), and the achieved
step Te_new/Te - 1 is checked to be exactly one grid interval.
tau_slow distribution: over the defended window-passing pairs at k = 30,
counts with tau_QSS < 100 us, < 22.7 us (the chapter's benchmark tau_slow,
1/|lambda_0| of L[23,5], CLAUDE.md), < the benchmark pair's own tau_QSS,
< 506 us, > 1 ms; min / median / max; heating / cooling split; and the same
restricted to the census100 members.

GATES (the run stops if any fails)
----------------------------------
G0  the sha256 lines stamped in each input CSV header equal the sha256 of the
    current L_grid.npy, S_grid.npy, state_index.csv; Te[i], ne[j] in the
    CSVs equal cr_context's grids to 1e-12 relative; both files have 784
    rows, 392 heat + 392 cool, with the same (direction, i, j) key set.
G1  plateau_gridmap and divertor_map agree row by row (keyed on
    (direction, i, j)) on tau_QSS, M and eps_plateau to 1e-9 relative
    (tau_relax and window_ok reported as well).
G2  the recount at k = 10, 20, 30, 50 reproduces window_sweep_summary.csv
    exactly: n_window_ok 779 / 728 / 680 / 596 of 784; n_warm 547 / 496 /
    448 / 364; census @100 us = 45 at every k; worst 0.1748 at (heat, 15, 3);
    and the remaining summary columns (n_dense, census_506us = 24,
    worst_506us, thr 5 % = 141, thr 20 % = 0).
G3  (only if validation/divertor_map_w{10,20,50}/divertor_map.csv exist)
    the recounted window_ok(k) equals the stored window_ok column of the
    k-sibling file on every row, and the sibling's eps_plateau equals the
    k = 30 eps_plateau bit for bit (window_sweep CHECK 1).

PREDICTIONS (written before the run; from the coordinator's by-eye tally of
the same CSV -- they are the thing under test)
---------------------------------------------------------------------------
P1  window-passing at k = 30 by direction: heating 338 of 392, cooling 342 of
    392; 104 pairs excluded in total.
P2  k = 100: n_window_ok = 452 of 784; warm = 221 of the 552 defended pairs;
    census numerator @100 us = 45, unchanged from k = 10..50; the benchmark
    heating pair heat[23,5] has M = 8243 (nearest integer) and is EXCLUDED at
    k = 100 (M < 1e4) but INCLUDED at k = 30.
P3  one-interval values for the requote: eps_plateau(heat,23,5) = 0.063612,
    eps_plateau(heat,0,0) = 0.083004, M(heat,23,5) = 8243; eps_plateau and M
    for heat[0,4] and heat[15,3] printed for cross-reference (no prediction).
P4  tau_slow over the 448 defended window-passing pairs at k = 30: 237 have
    tau_slow < 100 us, 142 have tau_slow < 22.7 us, 84 have tau_slow > 1 ms.
    The < 506 us count, min / median / max, heating / cooling split and the
    census-member restriction carry no prediction and are reported.
P5  eps_plateau is identical across k for every row (computed before the
    window gate), as window_sweep CHECK 1 found -- re-tested on the stored
    k-sibling files under G3.

REFUTERS
--------
P2  any census numerator other than 45 at k = 100 means the census depends
    on k, contradicting Chapter 5 sec 5.8 ("k sets the denominator of every
    census and never the numerator's magnitude").
P4  fewer than 45 of the 448 (10 %) defended window-passing pairs with
    tau_slow < 100 us would make the Chapter 5 ~166 sentence defensible as
    written; the prediction (237, i.e. 53 %) would not.

OUTPUTS (with --write)
----------------------
validation/window_membership_exposure/window_membership_exposure.csv          per pair: tau_QSS, M, eps_plateau, lo_100us, lo_506us, window_ok at each k, census flags
validation/window_membership_exposure/window_membership_exposure_summary.csv  per k: the counts
validation/window_membership_exposure/window_membership_exposure.txt          this run's log
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr_context import CRContext  # noqa: E402

K_LIST = [10, 20, 30, 50, 100]
K_CANON = 30
TE_WARM_MIN = 2.0          # eV, verify_window_sweep.py --te-warm-min default
NE_DENSE_MIN = 1e14        # cm^-3, verify_window_sweep.py --ne-dense-min default
CENSUS_THR = 0.10          # verify_window_sweep.py --census-threshold default
THRESHOLDS = (0.05, 0.10, 0.20)
TD_100US = 1e-4            # DRIVES "ELM_crash" in verify_divertor_map.py
TD_506US = 506e-6          # Loarte 2003 Sec. 5, ITER pedestal
TAU_BENCH_CHAPTER = 22.7e-6  # s, the chapter's benchmark tau_slow = 1/|lambda_0| of L[23,5] (CLAUDE.md 22.73 us)
NAMED = {"benchmark": ("heat", 23, 5), "cold corner": ("heat", 0, 0),
         "cold/dense": ("heat", 0, 4), "census worst": ("heat", 15, 3)}

# G2 expectations named by the coordinator (also read back from the stamped summary CSV)
G2_EXPECT = {10: dict(n_window_ok=779, n_warm=547), 20: dict(n_window_ok=728, n_warm=496),
             30: dict(n_window_ok=680, n_warm=448), 50: dict(n_window_ok=596, n_warm=364)}
G2_CENSUS = 45
G2_WORST = 0.1748
G2_WORST_AT = ("heat", 15, 3)

# predictions
P1 = dict(heat=338, cool=342, excluded=104)
P2 = dict(n_window_ok=452, n_warm=221, n_defended=552, census=45, M_bench=8243)
P3 = dict(eps_bench=0.063612, eps_cold=0.083004, M_bench=8243)
P4 = dict(n_defended_ok=448, below_100us=237, below_22p7us=142, above_1ms=84, refuter_frac=0.10)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_stamped_csv(p: Path) -> tuple[list[str], pd.DataFrame]:
    if not p.is_file():
        raise FileNotFoundError(f"required stamped input missing: {p}")
    hdr = []
    with open(p) as fh:
        for line in fh:
            if line.startswith("#"):
                hdr.append(line.rstrip("\n"))
            else:
                break
    df = pd.read_csv(p, comment="#")
    return hdr, df


def header_sha(hdr: list[str], key: str) -> str | None:
    for line in hdr:
        if key in line:
            return line.strip().split()[-1]
    return None


def window_ok_verbatim(k: float, tau_relax: np.ndarray, tau_QSS: np.ndarray) -> np.ndarray:
    """verify_divertor_map.py line 176 with win_lo = win_hi = k."""
    return (k * tau_relax) < (tau_QSS / k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ctx = CRContext.load(); root = ctx.root
    TeL, neL = ctx.te_grid, ctx.ne_grid
    nT, nN = len(TeL), len(neL)

    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    for p in (L_path, S_path, ctx.state_index_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    sha_now = {"L_grid sha256": sha256_file(L_path), "S_grid sha256": sha256_file(S_path),
               "state_index sha256": sha256_file(ctx.state_index_path)}

    gm_path = root / "validation/plateau_gridmap/plateau_gridmap.csv"
    dm_path = root / "validation/divertor_map/divertor_map.csv"
    ws_path = root / "validation/window_sweep/window_sweep_summary.csv"
    sm_path = root / "validation/plateau_slowmode/plateau_slowmode.csv"
    sib_paths = {k: root / f"validation/divertor_map_w{k}/divertor_map.csv" for k in (10, 20, 50)}
    inputs = {"plateau_gridmap.csv": gm_path, "divertor_map.csv": dm_path,
              "window_sweep_summary.csv": ws_path, "plateau_slowmode.csv": sm_path}
    for lab, p in inputs.items():
        if not p.is_file():
            raise FileNotFoundError(f"required stamped input missing: {p}")
    sha_in = {lab: sha256_file(p) for lab, p in inputs.items()}
    sib_present = {k: p.is_file() for k, p in sib_paths.items()}
    for k, p in sib_paths.items():
        if sib_present[k]:
            sha_in[f"divertor_map_w{k}/divertor_map.csv"] = sha256_file(p)

    log: list[str] = []
    def say(s: str = "") -> None:
        print(s); log.append(s)
    say("=" * 78)
    say("WINDOW MEMBERSHIP (k = 10..100) AND EXPOSURE vs tau_slow -- recount from stamped one-interval artifacts")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"interpreter   {sys.executable}   numpy {np.__version__}   pandas {pd.__version__}")
    say(ctx.describe())
    say("  inputs (read-only, sha256):")
    for lab, p in inputs.items():
        say(f"    {p.relative_to(root)}  {sha_in[lab]}")
    for k, p in sib_paths.items():
        say(f"    {p.relative_to(root)}  {sha_in.get(f'divertor_map_w{k}/divertor_map.csv', 'ABSENT -- G3 at this k cannot run')}")
    say("  current pipeline files (sha256):")
    for lab, v in sha_now.items():
        say(f"    {lab:20s} {v}")
    say(f"  k swept        : {K_LIST}   canonical k = {K_CANON}")
    say(f"  window test    : (k*tau_relax) < (tau_QSS/k)   [verify_divertor_map.py l.176, strict <]  i.e. M > k^2")
    say(f"  defended       : Te >= {TE_WARM_MIN} eV;  dense: also ne >= {NE_DENSE_MIN:.0e};  census threshold {CENSUS_THR:.0%}")
    say("=" * 78)

    # --- load ------------------------------------------------------------------
    gm_hdr, gm = read_stamped_csv(gm_path)
    dm_hdr, dm = read_stamped_csv(dm_path)
    ws_hdr, ws = read_stamped_csv(ws_path)
    sm_hdr, sm = read_stamped_csv(sm_path)

    # --- G0: staleness, grid labels, key set -----------------------------------
    say("\nG0  stamped header sha256 vs current pipeline files; grid labels; key set")
    for lab, hdr, p in (("plateau_gridmap", gm_hdr, gm_path), ("divertor_map", dm_hdr, dm_path),
                        ("window_sweep_summary", ws_hdr, ws_path), ("plateau_slowmode", sm_hdr, sm_path)):
        for key, expect in sha_now.items():
            got = header_sha(hdr, key)
            if got is None:
                raise ValueError(f"{p}: header missing '{key}' line")
            if got != expect:
                raise AssertionError(f"G0 FAILED: {p} header {key} = {got} but the current pipeline file has {expect}; "
                                     f"the artifact is stale relative to the grids; do not read on")
        say(f"    {lab:22s} header L_grid / S_grid / state_index sha256 all equal the current files")
    for lab, df in (("plateau_gridmap", gm), ("divertor_map", dm)):
        if len(df) != 784:
            raise AssertionError(f"G0 FAILED: {lab} has {len(df)} rows, expected 784")
        vc = df.direction.value_counts()
        if not (vc.get("heat", 0) == 392 and vc.get("cool", 0) == 392):
            raise AssertionError(f"G0 FAILED: {lab} direction counts {vc.to_dict()}, expected 392 heat + 392 cool")
        if df.duplicated(["direction", "i", "j"]).any():
            raise AssertionError(f"G0 FAILED: {lab} has duplicate (direction,i,j) keys")
        if df.i.max() >= nT or df.j.max() >= nN:
            raise AssertionError(f"G0 FAILED: {lab} indices exceed the grid ({nT} x {nN})")
        rel_te = np.abs(df.Te.values / TeL[df.i.values] - 1).max()
        rel_ne = np.abs(df["ne"].values / neL[df.j.values] - 1).max()   # df.ne is DataFrame.ne (not-equal), not the column
        if max(rel_te, rel_ne) > 1e-12:
            raise AssertionError(f"G0 FAILED: {lab} Te[i]/ne[j] do not match Te_grid_L/ne_grid_L (rel {rel_te:.1e}, {rel_ne:.1e})")
        say(f"    {lab:22s} 784 rows = 392 heat + 392 cool, unique keys, Te[i] & ne[j] equal the pipeline grids (rel {rel_te:.1e}, {rel_ne:.1e})")
    gm = gm.sort_values(["direction", "i", "j"]).reset_index(drop=True)
    dm = dm.sort_values(["direction", "i", "j"]).reset_index(drop=True)
    if not (gm[["direction", "i", "j"]].values == dm[["direction", "i", "j"]].values).all():
        raise AssertionError("G0 FAILED: the two files do not share the same (direction,i,j) key set")
    heat = (gm.direction == "heat").values
    i_new = np.where(heat, gm.i.values + 1, gm.i.values - 1)
    if i_new.min() < 0 or i_new.max() >= nT:
        raise AssertionError("G0 FAILED: a pair steps off the grid")
    step_rel = np.abs((gm.Te_new.values / gm.Te.values) / (TeL[i_new] / TeL[gm.i.values]) - 1).max()
    if step_rel > 1e-12:
        raise AssertionError(f"G0 FAILED: Te_new is not one grid interval from Te (rel {step_rel:.1e})")
    frac = gm.frac_achieved.values
    say(f"    heat i range {gm.i[heat].min()}..{gm.i[heat].max()}, cool i range {gm.i[~heat].min()}..{gm.i[~heat].max()};"
        f" Te_new/Te is exactly one grid interval on every row (rel {step_rel:.1e});"
        f" achieved |step| {np.abs(frac).min():.6f} .. {np.abs(frac).max():.6f}  (heating {frac[heat].mean():+.4%}, cooling {frac[~heat].mean():+.4%})")
    say("    G0 PASSED")

    # --- G1: row-by-row agreement of the two producing scripts -----------------
    say("\nG1  plateau_gridmap vs divertor_map, row by row on the shared keys (worst relative difference)")
    g1 = {}
    for col in ("tau_QSS", "tau_relax", "M", "eps_plateau"):
        g1[col] = float(np.abs(gm[col].values / dm[col].values - 1).max())
        say(f"    {col:12s} {g1[col]:.2e}")
    n_wdiff = int((gm.window_ok.values.astype(bool) != dm.window_ok.values.astype(bool)).sum())
    say(f"    window_ok    differs on {n_wdiff} rows")
    if max(g1["tau_QSS"], g1["M"], g1["eps_plateau"]) > 1e-9:
        raise AssertionError("G1 FAILED: the two stamped artifacts disagree on tau_QSS, M or eps_plateau; do not read on")
    if n_wdiff:
        raise AssertionError("G1 FAILED: the two stamped artifacts disagree on window_ok at k = 30")
    say("    G1 PASSED  (divertor_map.csv is used below; plateau_gridmap.csv supplies Te_new)")

    # --- the working arrays (from divertor_map.csv) ----------------------------
    direction = dm.direction.values; ii = dm.i.values; jj = dm.j.values
    Te = dm.Te.values; ne = dm["ne"].values
    tQ = dm.tau_QSS.values; tR = dm.tau_relax.values; M = dm.M.values
    eps = dm.eps_plateau.values
    lo100_stored = dm.lo_ELM_crash.values
    lo100 = eps * (tQ / TD_100US) * (1 - np.exp(-TD_100US / tQ))
    lo506 = eps * (tQ / TD_506US) * (1 - np.exp(-TD_506US / tQ))
    stored_ok30 = dm.window_ok.values.astype(bool)
    defended = Te >= TE_WARM_MIN
    n_defended = int(defended.sum())
    m_rel = float(np.abs(M / (tQ / tR) - 1).max())
    lo_rel = float(np.abs(lo100 / lo100_stored - 1).max())
    say(f"\n  consistency of the stored columns: |M/(tau_QSS/tau_relax) - 1| max {m_rel:.1e};"
        f"  |lo_100us recomputed / lo_ELM_crash stored - 1| max {lo_rel:.1e}")
    if m_rel > 1e-12 or lo_rel > 1e-12:
        raise AssertionError("stored M or lo_ELM_crash is not what its own formula gives; do not read on")
    say(f"  defended pairs (Te >= {TE_WARM_MIN} eV): {n_defended} = {int((defended & heat).sum())} heating + {int((defended & ~heat).sum())} cooling")

    # --- the recount at every k --------------------------------------------------
    ok_by_k: dict[int, np.ndarray] = {}
    summ: dict[int, dict] = {}
    for k in K_LIST:
        ok = window_ok_verbatim(float(k), tR, tQ)
        ok_M = M > float(k) ** 2
        n_form_diff = int((ok != ok_M).sum())
        ok_by_k[k] = ok
        warm = ok & defended
        dense = warm & (ne >= NE_DENSE_MIN)
        c100 = warm & (lo100_stored > CENSUS_THR)
        c506 = warm & (lo506 > CENSUS_THR)
        if c100.any():
            q = np.where(c100)[0][int(np.argmax(lo100_stored[c100]))]
            worst100 = float(lo100_stored[q]); worst_at = (str(direction[q]), int(ii[q]), int(jj[q]))
        else:
            worst100, worst_at = float("nan"), None
        worst506 = float(lo506[c506].max()) if c506.any() else (float(lo506[warm].max()) if warm.any() else float("nan"))
        summ[k] = dict(K=k, n_window_ok=int(ok.sum()), n_heat_ok=int((ok & heat).sum()), n_cool_ok=int((ok & ~heat).sum()),
                       n_excluded=int((~ok).sum()), n_warm=int(warm.sum()), n_defended=n_defended, n_dense=int(dense.sum()),
                       census_100us=int(c100.sum()), worst_100us=worst100, worst_at=worst_at,
                       census_506us=int(c506.sum()), worst_506us=worst506,
                       census_thr005=int((warm & (lo100_stored > 0.05)).sum()), census_thr020=int((warm & (lo100_stored > 0.20)).sum()),
                       forms_disagree_rows=n_form_diff, bench_included=bool(ok[(direction == "heat") & (ii == 23) & (jj == 5)][0]))
    n_wdiff30 = int((ok_by_k[K_CANON] != stored_ok30).sum())
    say(f"\n  recount at k = {K_CANON} vs the stored window_ok column: differs on {n_wdiff30} rows")
    if n_wdiff30:
        raise AssertionError("the verbatim window test does not reproduce the stored window_ok at k = 30; do not read on")

    say("\n" + "-" * 78)
    say("RESULT 1: membership by k (784 pairs; census = defended & window_ok & lower bound > 10 %)")
    say("-" * 78)
    say("    k    M>k^2   window_ok  heat  cool  excluded | warm(of defended)  dense | census@100us  worst  at            | census@506us  worst  | thr5%  thr20% | (k*tR<tQ/k) vs M>k^2 differ | bench heat[23,5]")
    for k in K_LIST:
        s = summ[k]
        wa = f"{s['worst_at'][0]}[{s['worst_at'][1]},{s['worst_at'][2]}]" if s["worst_at"] else "--"
        say(f"  {k:4d}  {k*k:6d}   {s['n_window_ok']:5d}     {s['n_heat_ok']:4d}  {s['n_cool_ok']:4d}   {s['n_excluded']:4d}   |  {s['n_warm']:4d} of {s['n_defended']}      {s['n_dense']:4d}  |"
            f"   {s['census_100us']:3d}       {s['worst_100us']:.4f} {wa:13s} |   {s['census_506us']:3d}      {s['worst_506us']:.4f} |  {s['census_thr005']:3d}    {s['census_thr020']:3d}   |"
            f"        {s['forms_disagree_rows']:3d} rows          | {'included' if s['bench_included'] else 'EXCLUDED'}")

    # --- G2 ---------------------------------------------------------------------
    say("\nG2  recount at k = 10, 20, 30, 50 vs validation/window_sweep/window_sweep_summary.csv (stamped 11 Sep)")
    ws = ws.set_index("K")
    g2_ok = True
    for k in (10, 20, 30, 50):
        if k not in ws.index:
            raise AssertionError(f"G2 FAILED: window_sweep_summary.csv has no row for k = {k}")
        s = summ[k]; r = ws.loc[k]
        checks = [("n_window_ok", s["n_window_ok"], int(r.n_window_ok)), ("n_warm", s["n_warm"], int(r.n_warm)),
                  ("n_dense", s["n_dense"], int(r.n_dense)), ("census_100us", s["census_100us"], int(r.census_100us)),
                  ("census_506us", s["census_506us"], int(r.census_506us)),
                  ("census_thr005", s["census_thr005"], int(r.census_thr005)), ("census_thr020", s["census_thr020"], int(r.census_thr020))]
        bad = [f"{n}: got {a}, stamped {b}" for n, a, b in checks if a != b]
        if abs(s["worst_100us"] - float(r.worst_100us)) > 1e-6:
            bad.append(f"worst_100us: got {s['worst_100us']:.6f}, stamped {float(r.worst_100us):.6f}")
        if abs(s["worst_506us"] - float(r.worst_506us)) > 1e-6:
            bad.append(f"worst_506us: got {s['worst_506us']:.6f}, stamped {float(r.worst_506us):.6f}")
        wa = s["worst_at"]; wa_str = f"{wa[0]}[{wa[1]},{wa[2]}]" if wa else ""
        if not str(r.worst_at).startswith(wa_str) or not wa_str:
            bad.append(f"worst_at: got {wa_str}, stamped {r.worst_at}")
        # the coordinator's named expectations
        e = G2_EXPECT[k]
        if s["n_window_ok"] != e["n_window_ok"] or s["n_warm"] != e["n_warm"]:
            bad.append(f"coordinator's expectation n_window_ok {e['n_window_ok']} / n_warm {e['n_warm']} not met")
        if s["census_100us"] != G2_CENSUS or wa != G2_WORST_AT or abs(s["worst_100us"] - G2_WORST) > 5e-5:
            bad.append(f"coordinator's expectation census {G2_CENSUS}, worst {G2_WORST} at {G2_WORST_AT} not met")
        say(f"    k={k:3d}: " + ("all columns reproduced" if not bad else "MISMATCH -- " + "; ".join(bad)))
        g2_ok &= not bad
    if not g2_ok:
        raise AssertionError("G2 FAILED: the recount does not reproduce the stamped window_sweep; do not read on")
    say("    G2 REPRODUCED")

    # --- G3 / P5: stored sibling files ------------------------------------------
    say("\nG3 / P5  stored k-sibling divertor maps: recounted window_ok vs stored; eps_plateau vs k = 30 (bit for bit)")
    for k in (10, 20, 50):
        if not sib_present[k]:
            say(f"    k={k:3d}: {sib_paths[k].relative_to(root)} ABSENT -- row-by-row check at this k NOT performed")
            continue
        sh, sd = read_stamped_csv(sib_paths[k])
        for key, expect in sha_now.items():
            got = header_sha(sh, key)
            if got != expect:
                raise AssertionError(f"G3 FAILED: {sib_paths[k]} header {key} = {got} is stale vs current {expect}")
        sd = sd.sort_values(["direction", "i", "j"]).reset_index(drop=True)
        if len(sd) != 784 or not (sd[["direction", "i", "j"]].values == dm[["direction", "i", "j"]].values).all():
            raise AssertionError(f"G3 FAILED: {sib_paths[k]} key set differs from divertor_map.csv")
        d_ok = int((sd.window_ok.values.astype(bool) != ok_by_k[k]).sum())
        d_eps = float(np.abs(sd.eps_plateau.values - eps).max())
        d_tq = float(np.abs(sd.tau_QSS.values / tQ - 1).max())
        d_tr = float(np.abs(sd.tau_relax.values / tR - 1).max())
        say(f"    k={k:3d}: window_ok recount differs from stored on {d_ok} rows;  max|eps_plateau(k) - eps_plateau(30)| = {d_eps:.3e}"
            f" {'(identical)' if d_eps == 0.0 else '*** DIFFERS ***'};  tau_QSS rel {d_tq:.1e}, tau_relax rel {d_tr:.1e}")
        if d_ok:
            raise AssertionError(f"G3 FAILED: recounted window_ok at k = {k} does not equal the stored sibling column")
        if d_eps != 0.0:
            raise AssertionError(f"G3 FAILED: eps_plateau at k = {k} is not bit-identical to k = 30 (P5 refuted)")
    say("    G3 PASSED on every sibling present;  P5: eps_plateau is a column never touched by k in this recount, and the"
        " stored siblings agree bit for bit")

    # --- P1 -----------------------------------------------------------------------
    s30 = summ[K_CANON]
    say("\n" + "-" * 78)
    say("P1  window-passing at k = 30 by direction")
    say("-" * 78)
    p1 = (s30["n_heat_ok"] == P1["heat"]) and (s30["n_cool_ok"] == P1["cool"]) and (s30["n_excluded"] == P1["excluded"])
    say(f"    heating {s30['n_heat_ok']} of 392 (predicted {P1['heat']}),  cooling {s30['n_cool_ok']} of 392 (predicted {P1['cool']}),"
        f"  excluded {s30['n_excluded']} (predicted {P1['excluded']})  -> {'as predicted' if p1 else 'NOT as predicted'}")
    excl30 = ~ok_by_k[K_CANON]
    say(f"    excluded pairs by density column j: " + ", ".join(f"j={j}: {int((excl30 & (jj == j)).sum())}" for j in range(nN)))
    say(f"    excluded pairs by Te class: Te < 2 eV {int((excl30 & ~defended).sum())}, Te >= 2 eV {int((excl30 & defended).sum())}")

    # --- P2 -----------------------------------------------------------------------
    s100 = summ[100]
    say("\n" + "-" * 78)
    say("P2  k = 100 (M > 1e4)")
    say("-" * 78)
    bsel = (direction == "heat") & (ii == 23) & (jj == 5)
    M_b = float(M[bsel][0])
    p2_counts = (s100["n_window_ok"] == P2["n_window_ok"]) and (s100["n_warm"] == P2["n_warm"]) and (n_defended == P2["n_defended"])
    p2_census = s100["census_100us"] == P2["census"]
    p2_bench = (round(M_b) == P2["M_bench"]) and (not s100["bench_included"]) and s30["bench_included"]
    say(f"    n_window_ok {s100['n_window_ok']} of 784 (predicted {P2['n_window_ok']});  heating {s100['n_heat_ok']}, cooling {s100['n_cool_ok']}")
    say(f"    warm (window_ok & Te >= 2) {s100['n_warm']} of {n_defended} defended (predicted {P2['n_warm']} of {P2['n_defended']});  dense {s100['n_dense']}")
    say(f"    census @100us {s100['census_100us']} / {s100['n_warm']}  worst {s100['worst_100us']:.4f} at {s100['worst_at']}  (predicted numerator {P2['census']});"
        f"  census @506us {s100['census_506us']}")
    say(f"    benchmark heat[23,5]: M = {M_b:.4f} -> {round(M_b)} (predicted {P2['M_bench']});  k=30 (M > 900): {'included' if s30['bench_included'] else 'EXCLUDED'};"
        f"  k=100 (M > 1e4): {'included' if s100['bench_included'] else 'EXCLUDED'}")
    say(f"    -> counts {'as predicted' if p2_counts else 'NOT as predicted'};  census numerator {'as predicted' if p2_census else 'NOT as predicted'};"
        f"  benchmark membership {'as predicted' if p2_bench else 'NOT as predicted'}")
    census_by_k = {k: summ[k]["census_100us"] for k in K_LIST}
    if len(set(census_by_k.values())) == 1:
        say(f"    REFUTER (census numerator changes with k): did not appear -- {census_by_k}")
    else:
        say(f"    REFUTER FIRED: census numerator depends on k -- {census_by_k}")
    # which census members would leave at each k (membership, for the record)
    c30 = ok_by_k[K_CANON] & defended & (lo100_stored > CENSUS_THR)
    say(f"    census members at k = 30 whose M is below 1e4 (would leave the census at k = 100): {int((c30 & (M <= 1e4)).sum())};"
        f"  census-member M range {M[c30].min():.3e} .. {M[c30].max():.3e}")

    # --- P3 -----------------------------------------------------------------------
    say("\n" + "-" * 78)
    say("P3  one-interval values for the requote (from divertor_map.csv == plateau_gridmap.csv, G1)")
    say("-" * 78)
    say("    pair              Te [eV]   Te_new [eV]   ne [cm^-3]    tau_QSS [s]    tau_relax [s]   M            eps_plateau   lo_100us    lo_506us")
    vals = {}
    for name, (d, i_, j_) in NAMED.items():
        sel = (direction == d) & (ii == i_) & (jj == j_)
        if sel.sum() != 1:
            raise AssertionError(f"pair {d}[{i_},{j_}] not found exactly once")
        q = int(np.where(sel)[0][0]); vals[name] = q
        say(f"    {name:13s} {d}[{i_:2d},{j_}]  {Te[q]:.4f}    {gm.Te_new.values[q]:.4f}       {ne[q]:.4e}   {tQ[q]:.6e}   {tR[q]:.6e}   {M[q]:.4f}   {eps[q]:.8f}    {lo100_stored[q]:.6f}    {lo506[q]:.6f}")
    qb, qc = vals["benchmark"], vals["cold corner"]
    p3 = (abs(eps[qb] - P3["eps_bench"]) < 5e-7) and (abs(eps[qc] - P3["eps_cold"]) < 5e-7) and (round(M[qb]) == P3["M_bench"])
    say(f"    eps_plateau(heat,23,5) = {eps[qb]:.6f} (predicted {P3['eps_bench']}),  eps_plateau(heat,0,0) = {eps[qc]:.6f} (predicted {P3['eps_cold']}),"
        f"  M(heat,23,5) = {M[qb]:.1f} (predicted {P3['M_bench']})  -> {'as predicted' if p3 else 'NOT as predicted'}")
    say("    for comparison, what the chapter now quotes from validation/plateau_slowmode/plateau_slowmode.csv (+0.6 eV ABSOLUTE step):")
    for _, r in sm.iterrows():
        say(f"      {r.label:11s} [{int(r.i)},{int(r.j)}]  Te {r.Te:.4f} -> {r.Te_new:.4f} eV (step {100*(r.Te_new/r.Te-1):+.1f} %)  tau_QSS {r.tau_QSS:.6e}  M {r.M:.2f}  eps_plateau {r.eps_plateau:.8f}")
    say(f"    one-interval step at the benchmark is {100*frac[qb]:+.2f} % (Te {Te[qb]:.4f} -> {gm.Te_new.values[qb]:.4f} eV); at the cold corner {100*frac[qc]:+.2f} %")
    say(f"    NOTE tau_QSS of the benchmark PAIR is {tQ[qb]*1e6:.2f} us = 1/|lambda_0| of the post-step operator L[24,5];"
        f" the chapter's 22.7 us is 1/|lambda_0| of L[23,5] itself (CLAUDE.md 22.73 us). Both are reported below.")

    # --- P4 -----------------------------------------------------------------------
    say("\n" + "-" * 78)
    say("P4  tau_slow (= tau_QSS of the pair) over the defended window-passing pairs at k = 30, against the 100 us exposure")
    say("-" * 78)
    sel = ok_by_k[K_CANON] & defended
    n_sel = int(sel.sum())
    tau_b_pair = float(tQ[qb])

    def tally(mask: np.ndarray, label: str) -> dict:
        t = tQ[mask]; n = int(mask.sum())
        out = dict(n=n, below_100us=int((t < 100e-6).sum()), below_22p7us=int((t < TAU_BENCH_CHAPTER).sum()),
                   below_bench_pair=int((t < tau_b_pair).sum()), below_506us=int((t < 506e-6).sum()), above_1ms=int((t > 1e-3).sum()),
                   tmin=float(t.min()) if n else float("nan"), tmed=float(np.median(t)) if n else float("nan"), tmax=float(t.max()) if n else float("nan"))
        say(f"    {label:38s} n={n:4d}:  <100us {out['below_100us']:4d} ({100*out['below_100us']/max(n,1):5.1f} %),  <22.7us {out['below_22p7us']:4d},"
            f"  <{tau_b_pair*1e6:.2f}us(bench pair) {out['below_bench_pair']:4d},  <506us {out['below_506us']:4d},  >1ms {out['above_1ms']:4d};"
            f"  min {out['tmin']:.3e}  median {out['tmed']:.3e}  max {out['tmax']:.3e} s")
        return out

    all_t = tally(sel, "defended & window_ok(30), all")
    tally(sel & heat, "  heating")
    tally(sel & ~heat, "  cooling")
    cen_t = tally(c30, "census members @100us (k=30)")
    tally(c30 & heat, "  heating")
    tally(c30 & ~heat, "  cooling")
    p4 = (n_sel == P4["n_defended_ok"]) and (all_t["below_100us"] == P4["below_100us"]) and (all_t["below_22p7us"] == P4["below_22p7us"]) \
        and (all_t["above_1ms"] == P4["above_1ms"])
    say(f"    predicted over {P4['n_defended_ok']}: <100us {P4['below_100us']}, <22.7us {P4['below_22p7us']}, >1ms {P4['above_1ms']}"
        f"  -> {'as predicted' if p4 else 'NOT as predicted'}")
    refuter_n = int(np.ceil(P4["refuter_frac"] * n_sel))
    if all_t["below_100us"] < refuter_n:
        say(f"    REFUTER (fewer than {refuter_n} of {n_sel} = 10 % below 100 us, which would make the ~166 sentence defensible): APPEARED")
    else:
        say(f"    REFUTER (fewer than {refuter_n} of {n_sel} = 10 % below 100 us): did not appear -- {all_t['below_100us']} of {n_sel}"
            f" ({100*all_t['below_100us']/n_sel:.1f} %) have tau_slow < 100 us, i.e. the ground state relaxes within the exposure there")
    # where in the grid are the short-tau_slow pairs
    short = sel & (tQ < 100e-6)
    say(f"    the {int(short.sum())} defended window-passing pairs with tau_slow < 100 us, by density column j: "
        + ", ".join(f"j={j}: {int((short & (jj == j)).sum())}" for j in range(nN)))
    say(f"    their Te range {Te[short].min():.3f} .. {Te[short].max():.3f} eV; the remaining {int((sel & ~short).sum())} (tau_slow >= 100 us) have ne <= {ne[sel & ~short].max():.3e} cm^-3")
    # what the 100 us lower bound retains at those pairs (recovery factor tQ/td*(1-exp(-td/tQ)))
    fac = (tQ / TD_100US) * (1 - np.exp(-TD_100US / tQ))
    say(f"    recovery factor lo_100us/eps_plateau over the defended window-passing pairs: min {fac[sel].min():.4f}  median {np.median(fac[sel]):.4f}  max {fac[sel].max():.4f};"
        f"  at the benchmark pair {fac[qb]:.4f}")

    say("\nREADING: membership is recounted from the same tau_QSS / tau_relax columns that verify_divertor_map.py wrote,\n"
        "with its own inequality; k = 100 is the one new value. The one-interval eps_plateau and M for the requote are\n"
        "printed under P3 next to the +0.6 eV values the paragraph currently quotes. P4 is the tally that decides whether\n"
        "the 'spends its integration time looking at the plateau' sentence survives. Nothing here re-runs any physics.")

    if args.write:
        out = Path(args.out) if args.out else root / "validation/window_membership_exposure"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}",
               f"# interpreter {sys.executable}  numpy {np.__version__}  pandas {pd.__version__}",
               f"# L_grid sha256 {sha_now['L_grid sha256']}",
               f"# S_grid sha256 {sha_now['S_grid sha256']}",
               f"# state_index sha256 {sha_now['state_index sha256']}"]
        for lab, v in sha_in.items():
            hdr.append(f"# input {lab} sha256 {v}")
        hdr.append(f"# window test (k*tau_relax) < (tau_QSS/k) [verify_divertor_map.py l.176]; defended Te >= {TE_WARM_MIN} eV; census thr {CENSUS_THR}; td 100us = {TD_100US}, 506us = {TD_506US}")
        rows = pd.DataFrame(dict(direction=direction, i=ii, j=jj, Te=Te, Te_new=gm.Te_new.values, ne=ne, frac_achieved=frac,
                                 tau_QSS=tQ, tau_relax=tR, M=M, eps_plateau=eps, lo_100us=lo100_stored, lo_506us=lo506,
                                 defended=defended, **{f"window_ok_k{k}": ok_by_k[k] for k in K_LIST},
                                 census_100us_k30=c30, census_506us_k30=ok_by_k[K_CANON] & defended & (lo506 > CENSUS_THR)))
        srows = []
        for k in K_LIST:
            s = dict(summ[k]); wa = s.pop("worst_at")
            s["worst_at"] = f"{wa[0]}[{wa[1]},{wa[2]}]" if wa else ""
            srows.append(s)
        with open(out / "window_membership_exposure.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); rows.to_csv(fh, index=False)
        with open(out / "window_membership_exposure_summary.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(srows).to_csv(fh, index=False)
        with open(out / "window_membership_exposure.txt", "w") as fh:
            fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/window_membership_exposure{{.csv,_summary.csv,.txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
