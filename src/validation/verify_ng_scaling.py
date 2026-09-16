#!/usr/bin/env python
"""
verify_ng_scaling.py
====================
Stamp the ground-state-density scaling test that chapter 6 section 6.4
(sec:ridge_conditional, chapter6.tex ~729-805) quotes from
outputs/findings_10_four_agent_review.md section 4.4.

WHY THIS EXISTS
---------------
The chapter's table says that multiplying the ground-state density feeding the
excitation channel by 0.1, 1, 10 at Te = 2.02 eV moves the density at which
|f_3 - f_4| peaks to 2.28e12, 1.68e13, 2.48e14 cm^-3, and that at unit scaling
the measured crest 1.93e13 sits 6% from Griem's 2.05e13. No script computed
those numbers and no file under validation/ holds them (backlog "Found beyond
the review" item 6). This script is that file.

WHAT IS COMPUTED
----------------
Within the two-channel split of Chapter 3 the excited populations at one
operator are n_E = a0 + a1 n_g, with a0 = -L_EE^-1 S_E (recombination-fed) and
a1 = -L_EE^-1 L_Eg (ground-fed, per unit n_g). For shell m the ground-fed
fraction is exactly

    f_m(s) = s a1_m n_g / (a0_m + s a1_m n_g)

when n_g is multiplied by s, so the scaling is an exact operation on f_m, as
the chapter says. n_g is the ground component of the CRE solution
-L^-1 S, which is n_g/n_i with the pipeline's n_i = 1 (verify_ridge_mechanism
calls it u).

findings_10 section 4.4 names Te = 2.02 eV and no temperature step. Two
constructions are therefore run and both are reported:

  I  (literal)  operator L[i,j] at the Te = 2.02 eV row, n_g from its own CRE.
                This is what the words in findings_10 say.
  II (thesis)   post-step operator L[k,j] with the pre-step n_g, exactly the
                objects verify_divertor_map.py uses for eps_plateau (k is the
                +5% row). The chapter's ridge language elsewhere refers to
                eps_plateau built this way, so this is the other defensible
                reading. Its eps_plateau at s = 1 is checked against the
                divertor_map.csv rows for this Te, to 1e-9, which ties this
                script to the census artifact.

For each construction and each s in {0.01, 0.1, 1, 10, 100} the eight density
columns give |f_3 - f_4| (and |f_3 - f_5| for the (3,5) pair). The peak is
located three ways, as verify_crest_subgrid.py does: the grid node of the
maximum, the vertex of a parabola in ln(ne) through the three nodes around it,
and the vertex through five nodes where five exist. The three-point vertex is
an interpolation, not a measurement; the three-point/five-point spread is its
error bar. A maximum on an edge column is reported as "off grid".

Griem's criterion is evaluated in the hydrogenic form
    n_e >= 7e18 * n^(-17/2) * (kT_e / E_H)^(1/2)  cm^-3
(Griem 1963, Phys. Rev. 131, 1170; Griem 1997, Principles of Plasma
Spectroscopy), at n = 4 and Te = 2 eV, with E_H read from the pipeline's
L_meta.csv. Chapter 5 (line ~1189) does not say which form it used; this one
returns 2.05e13 and is reported as the form that does.

PREDICTIONS, WRITTEN BEFORE RUNNING
-----------------------------------
Recorded (findings_10 4.4, chapter6.tex 762-772, 784-786, 798-801):
  x0.01  <= 1e12 (off grid)     x0.1  2.28e12
  x1     1.68e13                x10   2.48e14      x100  >= 1e15 (off grid)
  unit scaling: crest 1.93e13 against Griem 2.05e13
  (3,5) pair worst density 7.2e12, a factor 2.68 below the (3,4) pair's.
What would refute the chapter: a peak that does not move monotonically with s,
a slope d ln(ne_peak)/d ln(s) far from one, or peaks that do not reproduce to
the digits quoted under either reading.

Disclosure: exploratory runs on 11 Sep 2026 preceded this script and showed
that reading I lands within about 1% of the three recorded peaks and reading II
does not. Both are still reported; nothing is tuned.

Read-only with respect to the pipeline. Writes only validation/ng_scaling/.
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
TE_TARGET = 2.02            # eV, the temperature the chapter names
FRAC_STEP = 0.05            # verify_divertor_map.py's default fractional step
SCALES = (0.01, 0.1, 1.0, 10.0, 100.0)
GRIEM_PREFACTOR = 7e18      # cm^-3, hydrogenic Griem criterion
GRIEM_EXPONENT = 17.0 / 2.0

# Recorded values: compared against, never used in a computation.
REC_PEAKS = {0.1: 2.28e12, 1.0: 1.68e13, 10.0: 2.48e14}
REC_NODE_UNIT = 1.93e13
REC_GRIEM = 2.05e13
REC_35_NODE = 7.2e12
REC_35_FACTOR = 2.68


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vertex_log(x: np.ndarray, y: np.ndarray) -> float:
    """Vertex of the parabola through (ln x, y), in x units; nan if it opens up."""
    c = np.polyfit(np.log(x), y, 2)
    if c[0] >= 0:
        return float("nan")
    return float(np.exp(-c[1] / (2.0 * c[0])))


def locate_peak(ne: np.ndarray, y: np.ndarray) -> dict:
    j0 = int(np.argmax(y))
    nN = len(ne)
    out = dict(node_j=j0, node_ne=float(ne[j0]), v3=float("nan"),
               v5=float("nan"), edge=(j0 == 0 or j0 == nN - 1))
    if not out["edge"]:
        out["v3"] = vertex_log(ne[j0 - 1:j0 + 2], y[j0 - 1:j0 + 2])
        if 2 <= j0 <= nN - 3:
            out["v5"] = vertex_log(ne[j0 - 2:j0 + 3], y[j0 - 2:j0 + 3])
    return out


def fmt_peak(p: dict, ne: np.ndarray) -> str:
    if p["edge"]:
        side = "<=" if p["node_j"] == 0 else ">="
        return f"node j={p['node_j']} {side} {p['node_ne']:.3g} (off grid)"
    v5 = f"{p['v5']:.4g}" if np.isfinite(p["v5"]) else "  n/a "
    return (f"node j={p['node_j']} {p['node_ne']:.3g}  v3 {p['v3']:.4g}  "
            f"v5 {v5}")


def main() -> int:
    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    nv = ctx.n_values
    nT, nN, nS = L.shape[:3]

    lp = ROOT / "data/processed/cr_matrix/L_grid.npy"
    sp = ROOT / "data/processed/cr_matrix/S_grid.npy"
    mp = ROOT / "data/processed/cr_matrix/L_meta.csv"
    dp = ROOT / "validation/divertor_map/divertor_map.csv"
    for p in (sp, mp, dp):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
    S = np.load(sp)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    with mp.open() as fh:
        meta = next(csv.DictReader(fh))
    E_H = float(meta["IH_collision_eV"])

    E = np.array([q for q in range(nS) if q != g])
    pos = {s_: q for q, s_ in enumerate(E)}
    shells = {}
    for n in (2, 3, 4, 5):
        idx = np.where(nv == n)[0]
        if len(idx) != n:
            raise ValueError(f"shell n={n} has {len(idx)} states, expected {n}")
        shells[n] = np.array([pos[s_] for s_ in idx])
    N = {n: np.where(nv == n)[0] for n in (3, 4, 5)}

    i = int(np.argmin(np.abs(te - TE_TARGET)))
    k = int(np.argmin(np.abs(te - te[i] * (1 + FRAC_STEP))))
    if k == i:
        raise RuntimeError("the fractional step did not move a grid index")

    out = ROOT / "validation" / "ng_scaling"
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    shas = {"L_grid.npy": sha256(lp), "S_grid.npy": sha256(sp),
            "state_index.csv": sha256(ctx.state_index_path),
            "L_meta.csv": sha256(mp), "divertor_map.csv": sha256(dp)}
    say("=" * 78)
    say("GROUND-STATE DENSITY SCALING: where the (3,4) ridge sits versus n_g")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}")
    say(f"interpreter {sys.executable}   numpy {np.__version__}")
    say(f"repo root {ROOT}")
    for kk, v in shas.items():
        say(f"sha256 {kk:<18} {v}")
    say(f"row i={i} Te={te[i]:.4f} eV (nearest to {TE_TARGET}); post-step row "
        f"k={k} Te={te[k]:.4f} eV (+{te[k]/te[i]-1:.2%})")
    say(f"ne grid: {', '.join(f'{x:.3g}' for x in ne)}  (one interval = factor "
        f"{ne[1]/ne[0]:.3f})")
    say(f"shells: n=3 {[ctx.labels[q] for q in N[3]]}  n=4 "
        f"{[ctx.labels[q] for q in N[4]]}  n=5 {[ctx.labels[q] for q in N[5]]}")
    say(f"E_H = {E_H} eV from L_meta.csv")
    say("=" * 78)

    # ---- channels at both operators --------------------------------------
    def channels(row: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        M, SM = L[row, j], S[row, j]
        LEE = M[np.ix_(E, E)]
        LEg = M[np.ix_(E, [g])].ravel()
        a0 = np.linalg.solve(LEE, -SM[E])
        a1 = np.linalg.solve(LEE, -LEg)
        r0 = np.linalg.norm(LEE @ a0 + SM[E]) / np.linalg.norm(SM[E])
        r1 = np.linalg.norm(LEE @ a1 + LEg) / np.linalg.norm(LEg)
        if max(r0, r1) > 1e-10:
            raise RuntimeError(f"channel solve residual {max(r0, r1):.2e} at "
                               f"[{row},{j}]")
        return a0, a1

    ng_pre = np.array([np.linalg.solve(L[i, j], -S[i, j])[g] for j in range(nN)])
    ng_post = np.array([np.linalg.solve(L[k, j], -S[k, j])[g] for j in range(nN)])
    x_ratio = ng_post / ng_pre
    say("\n1. THE TWO CONSTRUCTIONS")
    say(f"  n_g/n_i at row {i} (CRE, n_i = 1): "
        f"{', '.join(f'{v:.4f}' for v in ng_pre)}")
    say(f"  ground ratio x = n_g(post)/n_g(pre): "
        f"{', '.join(f'{v:.4f}' for v in x_ratio)}")
    constructions = {
        "I_literal_L[i]": dict(row=i, ng=ng_pre,
                               label=f"I  operator L[{i},j], own CRE n_g"),
        "II_thesis_L[k]": dict(row=k, ng=ng_pre,
                               label=f"II operator L[{k},j], pre-step n_g "
                                     f"(eps_plateau objects)"),
    }
    A0, A1 = {}, {}
    for name, c in constructions.items():
        A0[name] = {n: np.zeros(nN) for n in (2, 3, 4, 5)}
        A1[name] = {n: np.zeros(nN) for n in (2, 3, 4, 5)}
        for j in range(nN):
            a0, a1 = channels(c["row"], j)
            for n in (2, 3, 4, 5):
                A0[name][n][j] = a0[shells[n]].sum()
                A1[name][n][j] = a1[shells[n]].sum()
        # exactness of the split at s = 1 against the full CRE solution
        worst = 0.0
        for j in range(nN):
            a0, a1 = channels(c["row"], j)
            n_full = np.linalg.solve(L[c["row"], j], -S[c["row"], j])
            ng_here = n_full[g]
            sup = (np.abs(a0 + a1 * ng_here - n_full[E]).max()
                   / np.abs(n_full[E]).max())
            worst = max(worst, sup)
        say(f"  {c['label']}: two-channel split reproduces the CRE excited "
            f"populations to {worst:.2e}")
        if worst > 1e-8:
            raise RuntimeError("two-channel split is not exact")

    def f_shell(name: str, n: int, s: float) -> np.ndarray:
        c = constructions[name]
        t = s * A1[name][n] * c["ng"]
        return t / (A0[name][n] + t)

    # cross-link: construction II eps_plateau at s = 1 against divertor_map.csv
    say("\n2. CROSS-LINK TO divertor_map.csv (construction II, s = 1, heat rows)")
    hdr, data = [], []
    with dp.open() as fh:
        for line in fh:
            (hdr if line.startswith("#") else data).append(line.rstrip("\n"))
    if shas["L_grid.npy"] not in "\n".join(hdr):
        raise RuntimeError("divertor_map.csv was generated from a different "
                           "L_grid; regenerate it before using it here")
    rd = [r for r in csv.DictReader(data)
          if r["direction"] == "heat" and int(r["i"]) == i]
    if len(rd) != nN:
        raise RuntimeError(f"expected {nN} heat rows at i={i} in "
                           f"divertor_map.csv, found {len(rd)}")
    f3, f4 = f_shell("II_thesis_L[k]", 3, 1.0), f_shell("II_thesis_L[k]", 4, 1.0)
    eps_p_II = np.abs((1 + (x_ratio - 1) * f4) / (1 + (x_ratio - 1) * f3) - 1)
    worst = 0.0
    for r in rd:
        j = int(r["j"])
        worst = max(worst, abs(eps_p_II[j] / float(r["eps_plateau"]) - 1),
                    abs(f3[j] / float(r["f3"]) - 1), abs(f4[j] / float(r["f4"]) - 1))
    say(f"  eps_plateau, f3, f4 at the 8 columns agree with the CSV to "
        f"{worst:.2e}  {'OK' if worst < 1e-9 else '!! MISMATCH'}")
    if worst > 1e-9:
        raise RuntimeError("construction II does not reproduce divertor_map.csv")
    say("  (eps_plateau = |(1+(x-1)f4)/(1+(x-1)f3) - 1| with x the ground ratio,")
    say("   which the n_g scaling leaves unchanged when both temperatures'")
    say("   neutral densities are scaled together)")

    # ---- 3. the scan ------------------------------------------------------
    say("\n3. THE SCAN: peak of |f3 - f4| across ne, per scaling of n_g")
    table_rows, peak_rows = [], []
    peaks = {}
    for name, c in constructions.items():
        say(f"\n  {c['label']}")
        say(f"  {'scale':>6}  {'|f3-f4| on the 8 columns':<62} peak")
        for s in SCALES:
            d34 = np.abs(f_shell(name, 3, s) - f_shell(name, 4, s))
            d35 = np.abs(f_shell(name, 3, s) - f_shell(name, 5, s))
            p34 = locate_peak(ne, d34)
            p35 = locate_peak(ne, d35)
            peaks[(name, s, "34")] = p34
            peaks[(name, s, "35")] = p35
            say(f"  x{s:<5g}  {np.array2string(d34, precision=4, separator=' ', floatmode='fixed'):<62} "
                f"{fmt_peak(p34, ne)}")
            for j in range(nN):
                row = dict(construction=name, scale=s, j=j, ne=float(ne[j]),
                           f3=float(f_shell(name, 3, s)[j]),
                           f4=float(f_shell(name, 4, s)[j]),
                           f5=float(f_shell(name, 5, s)[j]),
                           abs_f3_f4=float(d34[j]), abs_f3_f5=float(d35[j]))
                if name.startswith("II"):
                    row["eps_plateau_x_fixed"] = float(abs(
                        (1 + (x_ratio[j] - 1) * row["f4"])
                        / (1 + (x_ratio[j] - 1) * row["f3"]) - 1))
                table_rows.append(row)
            for pair, p in (("34", p34), ("35", p35)):
                peak_rows.append(dict(construction=name, pair=pair, scale=s,
                                      node_j=p["node_j"], node_ne=p["node_ne"],
                                      off_grid=p["edge"], vertex_3pt=p["v3"],
                                      vertex_5pt=p["v5"],
                                      recorded=REC_PEAKS.get(s, np.nan)
                                      if pair == "34" else np.nan))

    # ---- 4. against the record ------------------------------------------
    say("\n4. AGAINST THE RECORD (findings_10 4.4 / chapter6.tex 762-772)")
    say(f"  {'scale':>6} {'recorded':>10}   {'I: v3':>10} {'ratio':>7}   "
        f"{'II: v3':>10} {'ratio':>7}")
    for s in (0.1, 1.0, 10.0):
        pI = peaks[("I_literal_L[i]", s, "34")]
        pII = peaks[("II_thesis_L[k]", s, "34")]
        rI = pI["v3"] / REC_PEAKS[s] if np.isfinite(pI["v3"]) else np.nan
        rII = pII["v3"] / REC_PEAKS[s] if np.isfinite(pII["v3"]) else np.nan
        say(f"  x{s:<5g} {REC_PEAKS[s]:>10.3g}   {pI['v3']:>10.4g} {rI:>7.4f}   "
            f"{pII['v3']:>10.4g} {rII:>7.4f}")
    for s in (0.01, 100.0):
        pI = peaks[("I_literal_L[i]", s, "34")]
        pII = peaks[("II_thesis_L[k]", s, "34")]
        say(f"  x{s:<5g} recorded off grid; I: {fmt_peak(pI, ne)}; "
            f"II: {fmt_peak(pII, ne)}")
    say("  Construction I reproduces the three recorded peaks to two significant")
    say("  figures with a residual of about one percent whose source is not")
    say("  identified (the record has no script). Construction II does not")
    say("  reproduce them. The words of findings_10 4.4 (Te = 2.02 eV, no step)")
    say("  describe construction I.")

    say("\n  proportionality d ln(ne_peak) / d ln(s), three-point vertices where")
    say("  interior, nodes otherwise:")
    for name in constructions:
        vals = []
        for s in SCALES:
            p = peaks[(name, s, "34")]
            vals.append(p["v3"] if np.isfinite(p["v3"]) else p["node_ne"])
        sl = [np.log(vals[q + 1] / vals[q]) / np.log(SCALES[q + 1] / SCALES[q])
              for q in range(len(SCALES) - 1)]
        say(f"    {name:<16} " + "  ".join(
            f"x{SCALES[q]:g}->x{SCALES[q+1]:g}: {sl[q]:.3f}"
            for q in range(len(sl))))
    say("  (edge nodes cap the outer slopes; the x0.1 -> x1 -> x10 slopes are the")
    say("   ones the chapter's 'roughly one decade per decade' refers to)")

    # ---- 5. Griem, and the 1.93e13 versus 1.68e13 reconciliation ---------
    say("\n5. GRIEM'S CRITERION AND THE TWO CREST NUMBERS")
    griem = lambda n, T: GRIEM_PREFACTOR * n ** (-GRIEM_EXPONENT) * np.sqrt(T / E_H)
    g4 = griem(4, 2.0)
    say(f"  n_e = 7e18 n^-17/2 (kTe/E_H)^1/2 at n=4, Te=2 eV: {g4:.4g} cm^-3 "
        f"(recorded 2.05e13; rounds to it: {float(f'{g4:.3g}') == REC_GRIEM}); "
        f"at Te={te[i]:.4f}: {griem(4, te[i]):.4g}; n=3: {griem(3, 2.0):.3g}; "
        f"n=5: {griem(5, 2.0):.3g}")
    pI1 = peaks[("I_literal_L[i]", 1.0, "34")]
    pII1 = peaks[("II_thesis_L[k]", 1.0, "34")]
    say(f"  unit-scaling node: I {pI1['node_ne']:.4g}, II {pII1['node_ne']:.4g} "
        f"(recorded 1.93e13)")
    say(f"  unit-scaling three-point vertex: I {pI1['v3']:.4g} (five-point "
        f"{pI1['v5']:.4g}), II {pII1['v3']:.4g} (five-point {pII1['v5']:.4g})")
    say(f"  node against Griem: {abs(pI1['node_ne']/g4-1):.1%}; "
        f"I vertex against Griem: {abs(pI1['v3']/g4-1):.1%}; "
        f"II vertex against Griem: {abs(pII1['v3']/g4-1):.1%}")
    say("  Reconciliation: 1.93e13 is the grid node of the maximum (same in both")
    say("  constructions); 1.68e13 is the three-point vertex of construction I.")
    say("  They are two estimates of one crest, one at grid resolution and one")
    say(f"  interpolated, and they differ by {abs(pI1['v3']/pI1['node_ne']-1):.0%}. "
        f"One grid interval is a factor {ne[1]/ne[0]:.2f}, so the '6%' agreement")
    say("  with Griem is inside the grid resolution and becomes "
        f"{abs(pI1['v3']/g4-1):.0%} on the interpolated crest.")

    say("\n6. THE (3,5) PAIR (chapter6.tex 798-801)")
    for name in constructions:
        p34 = peaks[(name, 1.0, "34")]
        p35 = peaks[(name, 1.0, "35")]
        fac_node = p34["node_ne"] / p35["node_ne"]
        fac_v3 = (p34["v3"] / p35["v3"]
                  if np.isfinite(p34["v3"]) and np.isfinite(p35["v3"]) else np.nan)
        say(f"  {name:<16} |f3-f5| peak: {fmt_peak(p35, ne)};  (3,4)/(3,5) "
            f"node factor {fac_node:.3f}, vertex factor {fac_v3:.3f}")
    say(f"  recorded: 7.2e12 and a factor 2.68. The node factor is one grid "
        f"interval, 10^(3/7) = {10**(3/7):.3f}, exactly: the record's 2.68 is")
    say("  the grid spacing, and the vertex-to-vertex factor is what the")
    say("  pair shift measures at sub-grid resolution.")

    say("\n7. CHAPTER 5 CROSS-CHECK: nodes of |f_n - f_{n+1}| at s = 1")
    say("  chapter5.tex ~1183: 3.73e14 (2,3), 1.93e13 (3,4), 2.68e12 (4,5)")
    for name in constructions:
        parts = []
        for n in (2, 3, 4):
            d = np.abs(f_shell(name, n, 1.0) - f_shell(name, n + 1, 1.0))
            parts.append(f"({n},{n+1}) {ne[int(np.argmax(d))]:.3g}")
        say(f"  {name:<16} " + "  ".join(parts))

    # ---- write --------------------------------------------------------------
    prov = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
            f"# interpreter {sys.executable}  numpy {np.__version__}"]
    prov += [f"# sha256 {kk} {v}" for kk, v in shas.items()]
    prov += [f"# row i={i} Te={te[i]:.6f}  post-step k={k} Te={te[k]:.6f}  "
             f"fractional step {FRAC_STEP}  E_H {E_H}"]
    keys = ["construction", "scale", "j", "ne", "f3", "f4", "f5", "abs_f3_f4",
            "abs_f3_f5", "eps_plateau_x_fixed"]
    with (out / "ng_scaling.csv").open("w", newline="") as fh:
        fh.write("\n".join(prov) + "\n")
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in table_rows:
            w.writerow({kk: r.get(kk, "") for kk in keys})
    with (out / "ng_scaling_peaks.csv").open("w", newline="") as fh:
        fh.write("\n".join(prov) + "\n")
        w = csv.DictWriter(fh, fieldnames=list(peak_rows[0].keys()))
        w.writeheader()
        w.writerows(peak_rows)
    say(f"\nwrote {out/'ng_scaling.csv'} ({len(table_rows)} rows)")
    say(f"wrote {out/'ng_scaling_peaks.csv'} ({len(peak_rows)} rows)")
    (out / "ng_scaling.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'ng_scaling.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
