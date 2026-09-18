#!/usr/bin/env python
"""
verify_reservoir_vs_adas.py
===========================
Does ADAS's own reservoir ratio u_ADAS = ACD96/SCD96 respond to a k = 1
temperature step the way the model's u_CRE does?  Level offset and step
response, measured separately.

WHY THIS EXISTS
---------------
The reservoir gain of Chapter 4 is built from Delta ln u across one grid step,
with u = n_g / n_ion at collisional-radiative equilibrium, u_CRE = [-L^{-1} S]_g.
diagnose_gate_d.py compared the model's ground-fed SCD and its ACD to ADAS96
one coefficient at a time and found each within a factor ~2 at most points.
The reservoir does not care about either coefficient alone. At CRE the ground
row of L n + S n_ion = 0, together with the column-sum identity
sum_q L_qp = -K_ion(p) n_e, collapses exactly to

    n_e n_g SCD_gf = n_e n_ion ACD      =>      u_CRE = ACD / SCD_gf

so ADAS carries its own reservoir ratio u_ADAS = ACD96/SCD96 and its own step
response Delta ln u_ADAS, with no use of this model at all. Whether the two
step responses agree is the sharpest external test the gain can be given from
tabulated data. It is a different question from the level offset
u_model/u_ADAS, which this script reports separately and does not fold in.

METHOD
------
  model, at every (i, j) of the 50 x 8 grid, by the construction of
  diagnose_gate_d.py (re-implemented line for line, n_ion = n_e):
      n_CRE = L^{-1}(-S n_ion);  n_E = n^(0) + n^(1) with
      n^(0) = -L_EE^{-1} S_E n_ion (recombination-fed),
      n^(1) = -L_EE^{-1} L_Eg n_g (ground-fed);
      SCD_gf = K_ion(g) + K_ion(E).n^(1) / n_g;
      ACD    = [ L_gE n^(0) + S_g n_ion ] / (n_e n_ion);
      u_model = [L^{-1}(-S)]_g   (the u_CRE of molecular_channel.csv).
  ADAS, two reads of the same tables, both reported:
    (a) the gate-D read, imported verbatim from diagnose_gate_d.adas_at:
        linear in Te AND in the coefficient value (not log), among ADAS n_e
        nodes within a factor 2 of the model n_e, with no interpolation in
        n_e (at n_e = 1e12 and 1e15 exactly one node is in band, elsewhere
        two, whose rows are then interleaved in the Te-sorted abscissae).
    (b) this script's read: bilinear in (log10 Te, log10 n_e) on the native
        adf11 column log10_K, i.e. log-log in both variables.  The ADAS96
        tables are the native node set (29 Te x 24 n_e); only 7 Te nodes
        (1, 1.5, 2, 3, 5, 7, 10 eV) span the model's 49 steps, so under (b)
        d ln u_ADAS / d ln Te is piecewise constant across each ADAS interval.
  u_ADAS = ACD96/SCD96 at every grid point under both reads.
  Steps: every k = 1 pair (i, i+1) at fixed j, 49 x 8 = 392 steps, written in
  both directions (784 rows, the layout of reservoir_gain.csv); cooling is the
  sign flip of heating, so all |.| statistics are quoted over the 392 steps.
      Delta ln u_model = ln u_model(i+1)/u_model(i),   Delta ln u_ADAS likewise,
      diff = Delta ln u_model - Delta ln u_ADAS.
  "Te >= 2 eV" means the lower endpoint Te[i] >= 2 eV (both endpoints then are).
  "interior" means 1 <= i and i+1 <= 48 (neither endpoint on the Te boundary).
  RESULT 3: the coefficient ratios Chapter 4 quotes from gate_D_diagnosis.csv
  (eta_SCD_gf = SCD_gf/SCD96 and eta_ACD = ACD_model/ACD96: count within
  [0.5, 2.0], min/max with location, median; the same over Te >= 2 eV;
  [23,5], [0,4]; median by Te row for i = 0..5) recomputed under the log-log
  read and printed beside the gate-D-read values, over the 400 grid points
  (one value per point; step rows are not double-counted).

GATES (all must pass before anything else is reported)
------------------------------------------------------
  G0  10**log10_K reproduces the *_cm3_s column of each ADAS table (< 1e-6);
      every model grid point lies inside the ADAS node range (no extrapolation);
      the two-channel split reproduces n_CRE at every point (< 1e-8, as in
      diagnose_gate_d.py).
  G1  SCD_gf, ACD, eta_SCD_groundfed and eta_ACD recomputed here, with the
      gate-D ADAS read, reproduce the stored columns of
      validation/gate_D_diagnosis/gate_D_diagnosis.csv at all 400 points.
      That file stores 7 significant figures (".6e"), so a 1e-8 relative
      criterion is below its resolution and cannot be met by any
      recomputation; the gate applied is EXACT agreement at the stored
      precision: |recomputed - stored| <= half an ulp of the stored value.
      The raw relative deviation and the count of identical ".6e" strings are
      printed so the reader can see it is rounding and nothing else.
  G2  u_model reproduces molecular_channel.csv u_CRE at all 400 points (< 1e-8).
  G3  u_model = ACD/SCD_gf at all 400 points; worst relative deviation
      reported, at full precision and also from the stored 7-digit columns.

PREDICTIONS (written before the run, from a prior read-only reconstruction)
--------------------------------------------------------------------------
  P1  u_model/u_ADAS (log-log read): median about 1.34; range about 1.05
      (near [47,6]) to 7.5 (near [2,0]); about 1.34 at [23,5], 1.38 at [0,4].
  P2  Delta ln u at [23,5] -> [24,5]: model about -0.270, ADAS about -0.248.
  P3  over the 392 steps: median |diff| about 0.032 against median
      |Delta ln u_model| about 0.26.
  P4  maximum |diff| about 1.64, at the cold edge where ADAS96 has sparse
      Te nodes.
  P5  G3 worst deviation about 6.7e-7.  (If the full-precision identity is far
      tighter and the 7-digit stored columns give ~6.7e-7, the prediction
      was measuring file rounding.)
  Tolerances used to score them: P1 medians/points +-0.05, range ends +-0.05
  and +-0.5; P2 +-0.01; P3 +-0.005 and +-0.02; P4 +-0.1 and Te < 2 eV.
  The ADAS read the prior reconstruction used was not recorded, so every
  prediction is scored under both reads, (a) and (b); the read that
  reproduces them identifies what the reconstruction did.
  RESULT 3 (added after the first run showed the gate-D read overshoots SCD96
  by up to x6.65 across the 1-1.5 eV interval; written before that block ran):
  P6  under the log-log read, eta_SCD_gf = SCD_gf/SCD96 within a factor two
      rises above the 323/400 of the gate-D read.
  P7  its minimum rises well above 0.130 (scored as > 0.3); the [2,0] value
      moves by about x6.65, from ~0.13 to ~0.86 (scored +-0.05); the median
      moves by less than the minimum does (in ln).
  P8  eta_ACD = ACD_model/ACD96 changes little (median within 5 %, still
      400/400 within a factor two), because ACD96 varies slowly in Te.

REFUTING OBSERVATION (for the model's reservoir gain)
-----------------------------------------------------
  |Delta ln u_model - Delta ln u_ADAS| > 0.1 at any interior step with
  Te >= 2 eV.  Also counted over all Te >= 2 eV steps including the top edge.
  For RESULT 3: eta_SCD_gf minimum still below 0.2 under the log-log read,
  which would mean the low-Te SCD deficit is real and not the read.

WHAT THE LEVEL OFFSET MEANS (stated separately from the gain)
-------------------------------------------------------------
  u_model/u_ADAS > 1 means the model's ground reservoir per ion sits above
  ADAS's by that factor.  It moves the operating point, i.e. where on the
  tanh(|Delta|/4) cap of Chapter 5 the plateau sits, beneath the cap; it does
  not enter Delta ln u, which is what this script tests.  This script computes
  neither Delta nor the cap.

OUTPUTS (with --write): validation/reservoir_vs_adas/reservoir_vs_adas.{csv,txt}
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)
_DGD = _HERE.parent / "diagnose_gate_d.py"
_s = importlib.util.spec_from_file_location("dgd", _DGD); dgd = importlib.util.module_from_spec(_s); _s.loader.exec_module(dgd)

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def loglog_table(path: Path, col: str):
    """Native ADAS node set and the log10_K column pivoted to (n_Te, n_ne)."""
    if not path.exists(): raise RuntimeError(f"missing {path}; no stand-in is acceptable")
    d = pd.read_csv(path)
    for c in ("Te_eV", "ne_cm3", "log10_K", col):
        if c not in d.columns: raise RuntimeError(f"{path.name} lacks column {c}: {list(d.columns)}")
    te_n, ne_n = np.sort(d.Te_eV.unique()), np.sort(d.ne_cm3.unique())
    if len(d) != len(te_n) * len(ne_n): raise RuntimeError(f"{path.name}: {len(d)} rows is not a full {len(te_n)}x{len(ne_n)} node grid")
    piv = d.pivot(index="Te_eV", columns="ne_cm3", values="log10_K").loc[te_n, ne_n].values
    if np.isnan(piv).any(): raise RuntimeError(f"{path.name}: pivot has holes")
    chk = float(np.abs(10.0 ** d.log10_K.values / d[col].values - 1).max())
    return te_n, ne_n, piv, chk

def loglog_at(te_n, ne_n, piv, Te, ne):
    """Bilinear in (log10 Te, log10 ne) on log10_K. Returns (value, index of the ADAS Te interval)."""
    X, Y, x, y = np.log10(te_n), np.log10(ne_n), np.log10(Te), np.log10(ne)
    if not (X[0] <= x <= X[-1] and Y[0] <= y <= Y[-1]): raise RuntimeError(f"({Te}, {ne}) outside the ADAS node range; refusing to extrapolate")
    a = min(int(np.searchsorted(X, x, side="right")) - 1, len(X) - 2); b = min(int(np.searchsorted(Y, y, side="right")) - 1, len(Y) - 2)
    tx, ty = (x - X[a]) / (X[a + 1] - X[a]), (y - Y[b]) / (Y[b + 1] - Y[b])
    v = (1 - tx) * (1 - ty) * piv[a, b] + tx * (1 - ty) * piv[a + 1, b] + (1 - tx) * ty * piv[a, b + 1] + tx * ty * piv[a + 1, b + 1]
    return 10.0 ** v, a

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; g = ctx.ground_index; nT, nN, nS, _ = L.shape
    E = np.array([i for i in range(nS) if i != g])
    P = lambda *p: ROOT.joinpath(*p)
    inputs = {"L_grid.npy": P("data/processed/cr_matrix/L_grid.npy"), "S_grid.npy": P("data/processed/cr_matrix/S_grid.npy"),
              "Te_grid_L.npy": P("data/processed/cr_matrix/Te_grid_L.npy"), "ne_grid_L.npy": P("data/processed/cr_matrix/ne_grid_L.npy"),
              "state_index.csv": ctx.state_index_path, "K_ion_final.npy": P("data/processed/collisions/tics/K_ion_final.npy"),
              "SCD96_interpolated.csv": dgd.ADAS_SCD, "ACD96_interpolated.csv": dgd.ADAS_ACD,
              "gate_D_diagnosis.csv": P("validation/gate_D_diagnosis/gate_D_diagnosis.csv"),
              "molecular_channel.csv": P("validation/molecular_channel/molecular_channel.csv"), "diagnose_gate_d.py": _DGD}
    for nm, p in inputs.items():
        if not p.exists(): raise RuntimeError(f"missing input {nm}: {p}")
    S = np.load(inputs["S_grid.npy"]); K = np.load(inputs["K_ion_final.npy"])
    if S.shape != (nT, nN, nS) or K.shape != (nS, nT): raise RuntimeError(f"S_grid {S.shape} / K_ion {K.shape} do not match L_grid {L.shape}")
    gd = pd.read_csv(inputs["gate_D_diagnosis.csv"]).sort_values(["i", "j"]).reset_index(drop=True)
    mol = pd.read_csv(inputs["molecular_channel.csv"], comment="#").sort_values(["i", "j"]).reset_index(drop=True)
    for nm, d in (("gate_D_diagnosis.csv", gd), ("molecular_channel.csv", mol)):
        if len(d) != nT * nN or not (d.i.values == np.repeat(np.arange(nT), nN)).all() or not (d.j.values == np.tile(np.arange(nN), nT)).all():
            raise RuntimeError(f"{nm}: expected {nT*nN} rows covering the full grid, got {len(d)}")
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("RESERVOIR RATIO u = ACD/SCD: MODEL AGAINST ADAS96 -- level offset and k = 1 step response")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}  numpy {np.__version__}"); say(ctx.describe()); say("=" * 78)

    # ---- ADAS tables: native node set, log-log read (b), and the gate-D read (a) imported verbatim ----
    te_s, ne_s, piv_s, chk_s = loglog_table(dgd.ADAS_SCD, "SCD_cm3_s"); te_a, ne_a, piv_a, chk_a = loglog_table(dgd.ADAS_ACD, "ACD_cm3_s")
    gte_s, gne_s, gv_s = dgd.load_adas(dgd.ADAS_SCD, "SCD_cm3_s"); gte_a, gne_a, gv_a = dgd.load_adas(dgd.ADAS_ACD, "ACD_cm3_s")
    say(f"\nADAS96 tables: SCD {piv_s.shape[0]} Te x {piv_s.shape[1]} ne nodes, ACD {piv_a.shape[0]} x {piv_a.shape[1]}; Te nodes {te_s.min():.3g}..{te_s.max():.3g} eV, ne nodes {ne_s.min():.3g}..{ne_s.max():.3g} cm^-3")
    if not (np.allclose(te_s, te_a) and np.allclose(ne_s, ne_a)): raise RuntimeError("SCD96 and ACD96 node sets differ")
    inwin = te_s[(te_s >= te.min() * (1 - 1e-9)) & (te_s <= te.max() * (1 + 1e-9))]
    say(f"ADAS Te nodes spanning the model's {te.min():.3g}..{te.max():.3g} eV grid: {', '.join(f'{v:.4g}' for v in inwin)}")
    say(f"  node ratios: {', '.join(f'{r:.4f}' for r in inwin[1:] / inwin[:-1])};  model step ratio {te[1]/te[0]:.4f} ({nT-1} steps)")
    say("  model steps per ADAS interval: " + ", ".join(f"[{lo:.3g},{hi:.3g}] {((te >= lo*(1-1e-9)) & (te < hi*(1-1e-9))).sum()}" for lo, hi in zip(inwin[:-1], inwin[1:])))
    say("  model ne bracketed by ADAS ne nodes: " + "; ".join(f"{ne[j]:.3g} in [{ne_s[max(min(np.searchsorted(ne_s, ne[j]*(1+1e-9))-1, len(ne_s)-2),0)]:.3g},{ne_s[max(min(np.searchsorted(ne_s, ne[j]*(1+1e-9))-1, len(ne_s)-2),0)+1]:.3g}]" for j in range(nN)))
    say("\nGate-D read (diagnose_gate_d.adas_at, imported verbatim): linear in Te and in value, ADAS ne nodes within a factor 2 of the model ne, no ne interpolation.")
    for j in range(nN):
        r = ne_s / ne[j]; m = (r >= 0.5) & (r <= 2.0); say(f"    ne[{j}] = {ne[j]:.3e}: nodes in band {', '.join(f'{v:.3g}' for v in ne_s[m])}")
    say("This script's read: bilinear in (log10 Te, log10 ne) on the native log10_K column.")
    def used_nodes(nodes, vals):
        X = np.log10(nodes); out = set()
        for v in vals:
            x = np.log10(v); q = min(int(np.searchsorted(X, x, side="right")) - 1, len(X) - 2); t = (x - X[q]) / (X[q + 1] - X[q])
            if t < 1: out.add(q)
            if t > 0: out.add(q + 1)
        return nodes[sorted(out)]
    say(f"ADAS96 nodes carrying weight in the log-log read: Te {', '.join(f'{v:.4g}' for v in used_nodes(te_s, te))} eV;  ne {', '.join(f'{v:.3g}' for v in used_nodes(ne_s, ne))} cm^-3  (of {len(te_s)} x {len(ne_s)} tabulated)")
    src = _DGD.read_text().splitlines()
    def ln(pat):
        hits = [k + 1 for k, s_ in enumerate(src) if pat in s_]
        if len(hits) != 1: raise RuntimeError(f"diagnose_gate_d.py: pattern {pat!r} found {len(hits)} times; cannot cite a line number")
        return hits[0]
    say(f"Reported, not repaired: src/validation/diagnose_gate_d.py adas_at (def at line {ln('def adas_at')}) keeps the ADAS rows with ne_tab/ne in [0.5, 2.0] (lines {ln('r = ne_tab / ne')}-{ln('m = (r >= band')}), "
        f"sorts them by Te (line {ln('o = np.argsort')}) and returns np.interp(Te, Te_rows, value_rows) (line {ln('return float(np.interp')}): linear in Te and linear in the coefficient value, no logarithm, "
        f"no interpolation in ne. That file is unchanged; its sha256 is in the header.")

    # ---- G0 ----
    say("\nGATES"); say(f"  G0  10**log10_K vs SCD_cm3_s: {chk_s:.3e};  vs ACD_cm3_s: {chk_a:.3e}")
    if max(chk_s, chk_a) > 1e-6: raise RuntimeError("Gate G0: log10_K and the value column disagree")

    # ---- model construction (diagnose_gate_d.py, line for line) and both ADAS reads at every point ----
    Q = {k: np.full((nT, nN), np.nan) for k in ("u_model", "SCD_gf", "ACD", "SCD_ADAS_ll", "ACD_ADAS_ll", "SCD_ADAS_gd", "ACD_ADAS_gd")}
    Ka = np.zeros((nT, nN), int); worst_split = 0.0
    for i in range(nT):
        for j in range(nN):
            A = L[i, j]; n_ion = ne[j]; Sv = S[i, j] * n_ion
            n_cre = np.linalg.solve(A, -Sv)
            if np.any(n_cre <= 0): raise RuntimeError(f"non-positive CRE population at [{i},{j}]")
            LEE = A[np.ix_(E, E)]
            n0 = np.linalg.solve(LEE, -Sv[E]); n1 = np.linalg.solve(LEE, -A[np.ix_(E, [g])].ravel() * n_cre[g])
            resid = np.abs(n0 + n1 - n_cre[E]).max() / np.abs(n_cre[E]).max(); worst_split = max(worst_split, resid)
            if resid > 1e-8: raise RuntimeError(f"Gate G0: two-channel split residual {resid:.3e} at [{i},{j}]")
            Q["SCD_gf"][i, j] = K[g, i] + (K[E, i] @ n1) / n_cre[g]
            Q["ACD"][i, j] = (float(A[g, E] @ n0) + float(Sv[g])) / (ne[j] * n_ion)
            Q["u_model"][i, j] = np.linalg.solve(A, -S[i, j])[g]
            Q["SCD_ADAS_ll"][i, j], Ka[i, j] = loglog_at(te_s, ne_s, piv_s, te[i], ne[j]); Q["ACD_ADAS_ll"][i, j], _ = loglog_at(te_a, ne_a, piv_a, te[i], ne[j])
            Q["SCD_ADAS_gd"][i, j] = dgd.adas_at(gte_s, gne_s, gv_s, te[i], ne[j]); Q["ACD_ADAS_gd"][i, j] = dgd.adas_at(gte_a, gne_a, gv_a, te[i], ne[j])
    if not all(np.isfinite(v).all() and (v > 0).all() for v in Q.values()): raise RuntimeError("Gate G0: a non-finite or non-positive quantity was produced")
    say(f"      two-channel split residual, worst of 400: {worst_split:.3e};  every grid point inside the ADAS node range: yes")

    # ---- G1: reproduce gate_D_diagnosis.csv at its stored precision ----
    eta_gf, eta_acd = Q["SCD_gf"] / Q["SCD_ADAS_gd"], Q["ACD"] / Q["ACD_ADAS_gd"]
    say("  G1  recomputed vs stored gate_D_diagnosis.csv (file stores 7 s.f.; gate = exact at stored precision, i.e. |diff| <= half an ulp of the stored value):")
    g1_ok = True; g1_raw = 0.0
    for nm, arr, col in (("eta_SCD_groundfed", eta_gf, "eta_SCD_groundfed"), ("eta_ACD", eta_acd, "eta_ACD"), ("SCD_groundfed", Q["SCD_gf"], "SCD_groundfed"), ("ACD_model", Q["ACD"], "ACD_model")):
        st = gd[col].values.reshape(nT, nN); rel = np.abs(arr / st - 1); g1_raw = max(g1_raw, rel.max())
        half_ulp = 0.5 * 10.0 ** (np.floor(np.log10(np.abs(st))) - 6)
        within = (np.abs(arr - st) <= half_ulp * (1 + 1e-9)).sum(); same = sum(f"{x:.6e}" == f"{y:.6e}" for x, y in zip(arr.ravel(), st.ravel()))
        say(f"      {nm:18s} max rel dev {rel.max():.3e}   within half-ulp {within}/400   identical '.6e' strings {same}/400")
        g1_ok &= within == nT * nN
    say(f"      literal 1e-8 criterion: not testable against a '.6e' file (raw max {g1_raw:.3e} is the file's rounding); gate applied as stated above")
    if not g1_ok: raise RuntimeError("Gate G1: recomputation does not reproduce gate_D_diagnosis.csv at its stored precision")
    # ---- G2 ----
    relU = np.abs(Q["u_model"].ravel() / mol.u_CRE.values - 1).max(); say(f"  G2  u_model reproduces molecular_channel.csv u_CRE: max rel dev {relU:.3e}")
    if relU > 1e-8: raise RuntimeError("Gate G2")
    # ---- G3 ----
    dev = np.abs(Q["u_model"] / (Q["ACD"] / Q["SCD_gf"]) - 1); iw, jw = np.unravel_index(dev.argmax(), dev.shape)
    dev_st = np.abs(mol.u_CRE.values / (gd.ACD_model.values / gd.SCD_groundfed.values) - 1)
    say(f"  G3  identity u_model = ACD/SCD_gf: worst rel dev {dev.max():.3e} at [{iw},{jw}]  (from the stored 7-digit columns instead: {dev_st.max():.3e})")
    if dev.max() > 1e-8: raise RuntimeError("Gate G3")
    say("\n  ALL GATES PASSED.")

    # ---- RESULT: level offset ----
    u_ll, u_gd = Q["ACD_ADAS_ll"] / Q["SCD_ADAS_ll"], Q["ACD_ADAS_gd"] / Q["SCD_ADAS_gd"]
    eta_ll_s, eta_ll_a = Q["SCD_gf"] / Q["SCD_ADAS_ll"], Q["ACD"] / Q["ACD_ADAS_ll"]
    R, Rg = Q["u_model"] / u_ll, Q["u_model"] / u_gd
    say("\n" + "=" * 78); say("RESULT 1 -- level offset u_model / u_ADAS at the 400 grid points"); say("=" * 78)
    imn, jmn = np.unravel_index(R.argmin(), R.shape); imx, jmx = np.unravel_index(R.argmax(), R.shape)
    say(f"  log-log read : median {np.median(R):.4f}   min {R.min():.4f} at [{imn},{jmn}] (Te {te[imn]:.3g}, ne {ne[jmn]:.3g})   max {R.max():.4f} at [{imx},{jmx}] (Te {te[imx]:.3g}, ne {ne[jmx]:.3g})")
    say(f"                 at benchmark [23,5]: {R[23,5]:.4f}   at cold corner [0,4]: {R[0,4]:.4f}")
    say(f"  gate-D read  : median {np.median(Rg):.4f}   min {Rg.min():.4f}   max {Rg.max():.4f}   at [23,5]: {Rg[23,5]:.4f}   at [0,4]: {Rg[0,4]:.4f}")
    say(f"  gate-D read / log-log read of u_ADAS itself: min {(u_gd/u_ll).min():.3f}, max {(u_gd/u_ll).max():.3f} (at [{np.unravel_index((u_gd/u_ll).argmax(), R.shape)[0]},{np.unravel_index((u_gd/u_ll).argmax(), R.shape)[1]}]); equal to 1e-3 at {int((np.abs(u_gd/u_ll-1)<1e-3).sum())}/400 points")
    say(f"  ADAS SCD, gate-D read / log-log read: max {(Q['SCD_ADAS_gd']/Q['SCD_ADAS_ll']).max():.3f} at [{np.unravel_index((Q['SCD_ADAS_gd']/Q['SCD_ADAS_ll']).argmax(), R.shape)[0]},{np.unravel_index((Q['SCD_ADAS_gd']/Q['SCD_ADAS_ll']).argmax(), R.shape)[1]}]  (linear-in-value interpolation across an ADAS interval where SCD varies by orders of magnitude)")
    say("\n  is the offset structured in Te or in ne?  (log-log read)")
    colmed = np.median(R, axis=0); rowmed = np.median(R, axis=1)
    say("    median by ne column: " + "  ".join(f"{ne[j]:.2e}:{colmed[j]:.3f}" for j in range(nN)))
    say("    median by Te row   :"); 
    for k in range(0, nT, 10): say("      " + "  ".join(f"[{i}]{te[i]:.3g}:{rowmed[i]:.3f}" for i in range(k, min(k + 10, nT))))
    lnR = np.log(R); tot = lnR.var(); vr = lnR.mean(axis=1).var(); vc = lnR.mean(axis=0).var()
    say(f"    spread of row medians (Te): {rowmed.min():.3f}..{rowmed.max():.3f} (ratio {rowmed.max()/rowmed.min():.2f});  of column medians (ne): {colmed.min():.3f}..{colmed.max():.3f} (ratio {colmed.max()/colmed.min():.2f})")
    say(f"    variance of ln R: total {tot:.4f}; row means (Te) {vr:.4f} = {100*vr/tot:.0f}%; column means (ne) {vc:.4f} = {100*vc/tot:.0f}%; remainder {100*(1-(vr+vc)/tot):.0f}%")
    say(f"    over Te >= 2 eV only: median {np.median(R[te >= 2.0]):.4f}, min {R[te >= 2.0].min():.4f}, max {R[te >= 2.0].max():.4f}")

    # ---- RESULT: k = 1 step response ----
    rows = []
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(nT):
            ip = i + sgn
            if not (0 <= ip < nT): continue
            lo = min(i, ip)
            for j in range(nN):
                dm = np.log(Q["u_model"][ip, j] / Q["u_model"][i, j]); da = np.log(u_ll[ip, j] / u_ll[i, j]); dg = np.log(u_gd[ip, j] / u_gd[i, j])
                rows.append(dict(direction=dlab, k=1, i=i, j=j, i_post=ip, Te=float(te[i]), ne=float(ne[j]), Te_post=float(te[ip]), dlnTe=float(np.log(te[ip] / te[i])),
                                 Te_lo_ge2=bool(te[lo] >= 2.0), interior=bool(lo >= 1 and lo + 1 <= nT - 2),
                                 adas_Te_lo=float(te_s[Ka[i, j]]), adas_Te_hi=float(te_s[Ka[i, j] + 1]), crosses_adas_node=bool(Ka[ip, j] != Ka[i, j]),
                                 u_model=float(Q["u_model"][i, j]), SCD_gf=float(Q["SCD_gf"][i, j]), ACD_model=float(Q["ACD"][i, j]),
                                 SCD_ADAS_loglog=float(Q["SCD_ADAS_ll"][i, j]), ACD_ADAS_loglog=float(Q["ACD_ADAS_ll"][i, j]), u_ADAS_loglog=float(u_ll[i, j]),
                                 SCD_ADAS_gateD=float(Q["SCD_ADAS_gd"][i, j]), ACD_ADAS_gateD=float(Q["ACD_ADAS_gd"][i, j]), u_ADAS_gateD=float(u_gd[i, j]),
                                 ratio_loglog=float(R[i, j]), ratio_gateD=float(Rg[i, j]),
                                 eta_SCD_groundfed_stored=float(gd.eta_SCD_groundfed.values.reshape(nT, nN)[i, j]), eta_SCD_groundfed_recomputed=float(eta_gf[i, j]),
                                 eta_ACD_stored=float(gd.eta_ACD.values.reshape(nT, nN)[i, j]), eta_ACD_recomputed=float(eta_acd[i, j]),
                                 eta_SCD_groundfed_loglog=float(eta_ll_s[i, j]), eta_ACD_loglog=float(eta_ll_a[i, j]),
                                 dlnu_model=float(dm), dlnu_ADAS_loglog=float(da), dlnu_ADAS_gateD=float(dg), diff_loglog=float(dm - da), diff_gateD=float(dm - dg)))
    df = pd.DataFrame(rows); H = df[df.direction == "heat"].reset_index(drop=True); C = df[df.direction == "cool"]
    if len(H) != (nT - 1) * nN or not np.allclose(np.sort(np.abs(H.diff_loglog)), np.sort(np.abs(C.diff_loglog))): raise RuntimeError("step table is not the 392-step set in both directions")
    say("\n" + "=" * 78); say(f"RESULT 2 -- k = 1 step response over the {len(H)} steps (cooling is the sign flip; |.| statistics identical)"); say("=" * 78)
    def block(sub, lab):
        ad = sub.diff_loglog.abs(); r = sub.loc[ad.idxmax()]
        say(f"  {lab} ({len(sub)} steps): median |dlnu_model| {sub.dlnu_model.abs().median():.4f};  median |diff| {ad.median():.4f};  max |diff| {ad.max():.4f} at [{int(r.i)},{int(r.j)}]->[{int(r.i_post)},{int(r.j)}] (Te {r.Te:.3g}->{r.Te_post:.3g}, ne {r['ne']:.3g}; ADAS interval [{r.adas_Te_lo:.3g},{r.adas_Te_hi:.3g}])")
        say(f"      |diff| > 0.1 at {int((ad > 0.1).sum())} steps;  > 0.05 at {int((ad > 0.05).sum())};  median |dlnu_ADAS| {sub.dlnu_ADAS_loglog.abs().median():.4f};  median |diff|/|dlnu_model| {(ad/sub.dlnu_model.abs()).median():.3f}")
        return ad
    block(H, "all"); ad2 = block(H[H.Te_lo_ge2], "Te >= 2 eV"); block(H[H.Te_lo_ge2 & H.interior], "Te >= 2 eV, interior")
    say(f"  gate-D read instead: median |diff| {H.diff_gateD.abs().median():.4f}, max {H.diff_gateD.abs().max():.4f} (all);  median {H[H.Te_lo_ge2].diff_gateD.abs().median():.4f}, max {H[H.Te_lo_ge2].diff_gateD.abs().max():.4f} (Te >= 2 eV)")
    say(f"  steps whose endpoints straddle an ADAS Te node: {int(H.crosses_adas_node.sum())}; median |diff| there {H[H.crosses_adas_node].diff_loglog.abs().median():.4f}, elsewhere {H[~H.crosses_adas_node].diff_loglog.abs().median():.4f}")
    say("  |diff| by ne column (median, max), all steps: " + "  ".join(f"{ne[j]:.1e}:({H[H.j==j].diff_loglog.abs().median():.3f},{H[H.j==j].diff_loglog.abs().max():.3f})" for j in range(nN)))
    for (i, j, lab) in ((23, 5, "benchmark [23,5]->[24,5]"), (0, 4, "cold corner [0,4]->[1,4]")):
        r = H[(H.i == i) & (H.j == j)].iloc[0]
        say(f"\n  {lab}  (Te {r.Te:.4g}->{r.Te_post:.4g} eV, ne {r['ne']:.3e}; ADAS Te interval [{r.adas_Te_lo:.3g},{r.adas_Te_hi:.3g}], crosses node: {r.crosses_adas_node})")
        say(f"      u_model {r.u_model:.4e}   u_ADAS log-log {r.u_ADAS_loglog:.4e} (ratio {r.ratio_loglog:.4f})   u_ADAS gate-D {r.u_ADAS_gateD:.4e} (ratio {r.ratio_gateD:.4f})")
        say(f"      Delta ln u: model {r.dlnu_model:+.4f}   ADAS log-log {r.dlnu_ADAS_loglog:+.4f}   diff {r.diff_loglog:+.4f}   |  ADAS gate-D {r.dlnu_ADAS_gateD:+.4f}   diff {r.diff_gateD:+.4f}")
        say(f"      SCD_gf {r.SCD_gf:.4e}  SCD96 ll {r.SCD_ADAS_loglog:.4e}  gd {r.SCD_ADAS_gateD:.4e}   |   ACD {r.ACD_model:.4e}  ACD96 ll {r.ACD_ADAS_loglog:.4e}  gd {r.ACD_ADAS_gateD:.4e}")
    say("\n  cold edge, first four steps (model vs ADAS log-log, per ne column):")
    for i in range(4):
        say(f"    [{i}]->[{i+1}] Te {te[i]:.4g}->{te[i+1]:.4g}: " + "  ".join(f"j{j}:{H[(H.i==i)&(H.j==j)].dlnu_model.iloc[0]:+.3f}/{H[(H.i==i)&(H.j==j)].dlnu_ADAS_loglog.iloc[0]:+.3f}" for j in range(nN)))

    # ---- RESULT 3: the coefficient ratios Chapter 4 quotes, both reads, one value per grid point ----
    say("\n" + "=" * 78); say("RESULT 3 -- eta_SCD_gf = SCD_gf/SCD96 and eta_ACD = ACD_model/ACD96 at the 400 grid points (one value per point), log-log read beside the gate-D read"); say("=" * 78)
    hi = te >= 2.0; allm = np.ones(nT, bool)
    def st(arr, rowmask):
        A_ = np.where(rowmask[:, None], arr, np.nan); v = A_[np.isfinite(A_)]
        mn = np.unravel_index(np.nanargmin(A_), A_.shape); mx = np.unravel_index(np.nanargmax(A_), A_.shape)
        return f"within [0.5,2]: {int(((v >= 0.5) & (v <= 2.0)).sum()):3d}/{v.size}   min {v.min():.3f} at [{mn[0]},{mn[1]}]   max {v.max():.3f} at [{mx[0]},{mx[1]}]   median {np.median(v):.3f}"
    for lab, ll, gdv in (("eta_SCD_gf", eta_ll_s, eta_gf), ("eta_ACD", eta_ll_a, eta_acd)):
        say(f"\n  {lab}")
        say(f"    log-log read, all 400          : {st(ll, allm)}"); say(f"    gate-D read,  all 400          : {st(gdv, allm)}")
        say(f"    log-log read, Te >= 2 eV ({int(hi.sum())*nN}): {st(ll, hi)}"); say(f"    gate-D read,  Te >= 2 eV ({int(hi.sum())*nN}): {st(gdv, hi)}")
        say(f"    at [23,5]: log-log {ll[23,5]:.3f}  gate-D {gdv[23,5]:.3f}     at [0,4]: log-log {ll[0,4]:.3f}  gate-D {gdv[0,4]:.3f}     at [2,0]: log-log {ll[2,0]:.3f}  gate-D {gdv[2,0]:.3f}")
        say("    median by Te row, i = 0..5 (log-log / gate-D): " + "  ".join(f"[{i}]{te[i]:.3g}: {np.median(ll[i]):.3f}/{np.median(gdv[i]):.3f}" for i in range(6)))
        rr = ll / gdv; rmx_ = np.unravel_index(rr.argmax(), rr.shape)
        say(f"    log-log/gate-D pointwise (= ADAS gate-D read / log-log read): min {rr.min():.3f}  max {rr.max():.3f} at [{rmx_[0]},{rmx_[1]}];  over Te >= 2 eV: {rr[hi].min():.3f}..{rr[hi].max():.3f}")
    n_ll_s, n_gd_s = int(((eta_ll_s >= 0.5) & (eta_ll_s <= 2.0)).sum()), int(((eta_gf >= 0.5) & (eta_gf <= 2.0)).sum())
    n_ll_a = int(((eta_ll_a >= 0.5) & (eta_ll_a <= 2.0)).sum())

    # ---- predictions and refuter ----
    b = H[(H.i == 23) & (H.j == 5)].iloc[0]
    def score(Rx, dcol, fcol):
        adx = H[fcol].abs(); rmx = H.loc[adx.idxmax()]; mn = np.unravel_index(Rx.argmin(), Rx.shape); mx = np.unravel_index(Rx.argmax(), Rx.shape)
        return [("P1 median ratio ~1.34", abs(np.median(Rx) - 1.34) < 0.05, f"{np.median(Rx):.4f}"),
                ("P1 range ~1.05..7.5", abs(Rx.min() - 1.05) < 0.05 and abs(Rx.max() - 7.5) < 0.5, f"{Rx.min():.4f}..{Rx.max():.4f} at [{mn[0]},{mn[1]}], [{mx[0]},{mx[1]}]"),
                ("P1 [23,5] ~1.34, [0,4] ~1.38", abs(Rx[23, 5] - 1.34) < 0.05 and abs(Rx[0, 4] - 1.38) < 0.05, f"{Rx[23,5]:.4f}, {Rx[0,4]:.4f}"),
                ("P2 [23,5]->[24,5] model ~-0.270, ADAS ~-0.248", abs(b.dlnu_model + 0.270) < 0.01 and abs(b[dcol] + 0.248) < 0.01, f"{b.dlnu_model:+.4f}, {b[dcol]:+.4f}"),
                ("P3 median |diff| ~0.032 vs median |dlnu_model| ~0.26", abs(adx.median() - 0.032) < 0.005 and abs(H.dlnu_model.abs().median() - 0.26) < 0.02, f"{adx.median():.4f}, {H.dlnu_model.abs().median():.4f}"),
                ("P4 max |diff| ~1.64 at the cold edge", abs(adx.max() - 1.64) < 0.1 and rmx.Te < 2.0, f"{adx.max():.4f} at [{int(rmx.i)},{int(rmx.j)}] (Te {rmx.Te:.3g})")]
    p5 = ("P5 G3 worst ~6.7e-7", abs(dev.max() / 6.7e-7 - 1) < 0.3, f"full precision {dev.max():.2e}; from the stored 7-digit columns {dev_st.max():.2e}")
    for rd, Rx, dcol, fcol in (("(b) log-log read, this script's result", R, "dlnu_ADAS_loglog", "diff_loglog"), ("(a) gate-D read", Rg, "dlnu_ADAS_gateD", "diff_gateD")):
        say(f"\nPREDICTIONS scored under {rd} (tolerances in the docstring):")
        for lab, ok, val in score(Rx, dcol, fcol): say(f"  {'reproduced    ' if ok else 'NOT reproduced'}  {lab}: {val}")
    say(f"  {'reproduced    ' if p5[1] else 'NOT reproduced'}  {p5[0]}: {p5[2]}")
    p678 = [("P6 eta_SCD_gf within factor 2 rises above 323/400 under log-log", n_ll_s > 323, f"{n_ll_s}/400 (gate-D {n_gd_s}/400)"),
            ("P7 eta_SCD_gf min > 0.3; [2,0] ~0.86 (+-0.05, ~x6.65); median moves less than the min (in ln)",
             eta_ll_s.min() > 0.3 and abs(eta_ll_s[2, 0] - 0.86) < 0.05 and abs(np.log(np.median(eta_ll_s) / np.median(eta_gf))) < abs(np.log(eta_ll_s.min() / eta_gf.min())),
             f"min {eta_ll_s.min():.3f} (gate-D {eta_gf.min():.3f}); [2,0] {eta_ll_s[2,0]:.3f} (gate-D {eta_gf[2,0]:.3f}, x{eta_ll_s[2,0]/eta_gf[2,0]:.2f}); median {np.median(eta_ll_s):.3f} (gate-D {np.median(eta_gf):.3f})"),
            ("P8 eta_ACD changes little: median within 5 %, 400/400 within factor 2", abs(np.median(eta_ll_a) / np.median(eta_acd) - 1) < 0.05 and n_ll_a == nT * nN,
             f"median {np.median(eta_ll_a):.3f} (gate-D {np.median(eta_acd):.3f}); within factor 2 {n_ll_a}/400; largest pointwise change {np.abs(np.log(eta_ll_a/eta_acd)).max():.3f} in ln")]
    say("\nPREDICTIONS for RESULT 3 (written before that block ran):")
    for lab, ok, val in p678: say(f"  {'reproduced    ' if ok else 'NOT reproduced'}  {lab}: {val}")
    ref_int = H[H.Te_lo_ge2 & H.interior]; ref_all = H[H.Te_lo_ge2]
    n_int = int((ref_int.diff_loglog.abs() > 0.1).sum()); n_all = int((ref_all.diff_loglog.abs() > 0.1).sum())
    say(f"\nREFUTER (|Delta ln u_model - Delta ln u_ADAS| > 0.1 at any interior step with Te >= 2 eV): " + (f"APPEARED at {n_int} of {len(ref_int)} steps" if n_int else f"did not appear (0 of {len(ref_int)} steps; worst {ref_int.diff_loglog.abs().max():.4f})"))
    say(f"  including the top Te edge (all Te >= 2 eV steps): {n_all} of {len(ref_all)} exceed 0.1; worst {ref_all.diff_loglog.abs().max():.4f} at [{int(ref_all.loc[ref_all.diff_loglog.abs().idxmax()].i)},{int(ref_all.loc[ref_all.diff_loglog.abs().idxmax()].j)}]")
    say(f"  under the gate-D read instead: {int((ref_int.diff_gateD.abs() > 0.1).sum())} of {len(ref_int)} interior Te >= 2 eV steps exceed 0.1; worst {ref_int.diff_gateD.abs().max():.4f}")
    mn3 = np.unravel_index(eta_ll_s.argmin(), eta_ll_s.shape)
    say(f"REFUTER for RESULT 3 (eta_SCD_gf minimum still below 0.2 under the log-log read): " + (f"APPEARED -- min {eta_ll_s.min():.3f} at [{mn3[0]},{mn3[1]}]; the low-Te deficit is real, not the read" if eta_ll_s.min() < 0.2 else f"did not appear (min {eta_ll_s.min():.3f} at [{mn3[0]},{mn3[1]}])"))
    say("\nLEVEL OFFSET, stated separately: u_model/u_ADAS > 1 means the model's ground reservoir per ion sits above ADAS's by that factor.\n"
        "It moves the operating point beneath the tanh(|Delta|/4) cap; it does not enter Delta ln u. This script computes neither Delta nor the cap.")

    if a.write:
        out = Path(a.out) if a.out else P("validation/reservoir_vs_adas"); out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}  pandas {pd.__version__}"]
        hdr += [f"# {nm} sha256 {sha(p)}" for nm, p in inputs.items()]
        hdr += ["# ADAS read: log-log bilinear on log10_K (columns *_loglog); gate-D read = diagnose_gate_d.adas_at (columns *_gateD); rows = k=1 steps, both directions, pre-step point quantities on each row (eta_*_loglog and eta_*_recomputed are the RESULT 3 per-point ratios; every grid point appears as a pre-step point in at least one direction)"]
        with open(out / "reservoir_vs_adas.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "reservoir_vs_adas.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/reservoir_vs_adas.{{csv,txt}}")
    return 0

if __name__ == "__main__": sys.exit(main())
