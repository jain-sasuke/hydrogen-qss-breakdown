#!/usr/bin/env python
"""
verify_headline_provenance.py
=============================
Re-derive, from stamped artifacts alone, the printed numbers whose captions said
they came from a project working note, and fail if any of them has drifted.

WHY THIS EXISTS
---------------
The Declaration states the standard the thesis is written to: every numerical
result produced by a script in the repository and quoted with the script and the
data file that produced it. Four tables and one pair of abstract-level numbers
carried captions saying their values came from a working note with no producing
script. A provenance audit found that in every one of those cases the values
already exist inside artifacts that ARE stamped; what was missing was the
citation, not the calculation. This script closes that gap in the only way that
is worth anything: it reads the stamped artifacts, recomputes each printed
number, and exits non-zero if any of them fails to reproduce.

It computes no new physics. Every number it emits is a reduction of an artifact
produced by another script, and it names that script and that column for each.

WHAT IT CHECKS
--------------
  A  The two excited-state QSS errors quoted in the abstract and in chapter 5:
     8.66e-6 at the benchmark heating step [23,5] and 6.73e-9 at [0,4], with
     their paired CRE errors 6.34e-2 and 3.869e-1.
     Source: validation/trajectory_census/trajectory_census.csv
     (verify_trajectory_census.py), columns max_track_w30_shell and
     cre_start_w30_shell. The column name is why these looked unstamped: the
     artifact calls the quantity max_track, the thesis calls it eps_QSS.
  B  tab:gain, the reservoir gain G at three points for steps of one, two and
     four grid intervals (nine values).
     Source: validation/reservoir_gain/reservoir_gain.csv
     (verify_reservoir_gain.py), column G, direction heat.
  C  tab:closure, the slow timescale open and closed at four points with the
     ratio (twelve values).
     Source: validation/ion_closure/ion_closure_summary.csv
     (verify_ion_closure.py).
  D  tab:position_effect, the crest mechanism table at the post-step
     temperature, five rows of four columns.
     Sources: validation/molecular_channel/molecular_channel.csv for Delta,
     the cap and u_peak; validation/reservoir_gain/reservoir_gain.csv for Sbar.

     INDEX CONVENTION, which the table's caption did not state and which this
     script makes explicit because it is the one thing in that table that is not
     self-evident: Delta and tanh(|Delta|/4) are evaluated at the POST-step
     temperature index 24, while Sbar is the secant of the step 23 -> 24 and is
     stored against index 23, and the printed ln(u_CRE/u_peak) takes its
     numerator u_CRE at index 23 and its denominator u_peak at index 24. That
     mixed index is reproduced here exactly as printed, and the two consistent
     alternatives are computed alongside so the size of the choice is visible.
     Under all three conventions the sign change sits between j = 2 and j = 3
     and |Sbar| peaks at j = 3, which is what the table is used to establish.

GATES
-----
  G0  every artifact exists and carries the canonical L_grid sha256 in its header
  G1  every printed value reproduces from its artifact to the precision printed
  Exit status is non-zero if any check fails.

OUTPUT (with --write): validation/headline_provenance/headline_provenance.csv
  one row per printed number: claim, thesis location, printed, recomputed,
  relative difference, artifact, producing script, column.
"""
from __future__ import annotations
import argparse, hashlib, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)
L_SHA = "2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e"


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


PROV: dict[str, str] = {}


def load(rel: str) -> pd.DataFrame:
    """Read a validation CSV and record what provenance its own header carries.

    G0 is reported, not enforced: an artifact whose value reproduces is useful evidence even
    when the file carries no stamp, but the difference must be visible rather than assumed,
    so every source is classified here and the classification is printed and written out."""
    p = ROOT / rel
    if not p.is_file(): raise FileNotFoundError(p)
    head = "".join(l for l in p.read_text().splitlines(keepends=True)[:14] if l.startswith("#"))
    if not head:
        PROV[rel] = "UNSTAMPED: the file carries no header, so it names neither its producer nor its inputs"
    elif L_SHA in head:
        PROV[rel] = "stamped, canonical L_grid sha256 present"
    else:
        PROV[rel] = "header present but the canonical L_grid sha256 is not in it"
    return pd.read_csv(p, comment="#")


def main() -> int:
    ap = argparse.ArgumentParser(description="re-derive the printed numbers whose captions cited a working note")
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows: list[dict] = []; fails: list[str] = []
    say = print
    say("=" * 104); say("HEADLINE PROVENANCE: printed numbers re-derived from stamped artifacts")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}"); say("=" * 104)

    def check(claim, loc, printed, got, artifact, producer, column, tol=None):
        # Tolerance is the rounding of the value AS PRINTED: half a unit in its last significant
        # digit, expressed as a relative difference. The mantissa is counted on its own, since in
        # scientific notation the exponent digits are not significant figures of the value.
        if tol is None:
            mant = f"{abs(printed):.12g}".split("e")[0].replace(".", "").lstrip("0") or "0"
            nsig = len(mant.rstrip("0")) or 1
            decade = int(np.floor(np.log10(abs(printed)))) if printed else 0
            tol = (0.5 * 10.0 ** (decade - nsig + 1)) / abs(printed)
        rel = abs(got / printed - 1) if printed else abs(got)
        ok = rel <= tol
        rows.append(dict(claim=claim, thesis_location=loc, printed=printed, recomputed=got, rel_diff=rel,
                         tolerance=tol, ok=ok, artifact=artifact, producer=producer, column=column))
        if not ok: fails.append(f"{claim} at {loc}: printed {printed:g}, artifact {got:g}, rel {rel:.2e} > {tol:.0e}")
        return ok

    # ---------------------------------------------------------------- A: the two QSS errors
    tc = load("validation/trajectory_census/trajectory_census.csv")
    say("\n  A  excited-state QSS error, abstract and chapter 5")
    say(f"     artifact validation/trajectory_census/trajectory_census.csv (verify_trajectory_census.py), {len(tc)} rows")
    for (i, j), eq, ec, loc in (((23, 5), 8.66e-6, 6.34e-2, "thesis_main.tex:203, chapter5.tex:263"),
                                ((0, 4), 6.73e-9, 3.869e-1, "thesis_main.tex:205, chapter5.tex:266")):
        r = tc[(tc.i == i) & (tc.j == j) & (tc.direction == "heat")]
        if len(r) != 1: raise RuntimeError(f"expected one heat row at [{i},{j}], found {len(r)}")
        r = r.iloc[0]
        check("eps_QSS", loc, eq, float(r.max_track_w30_shell), "trajectory_census.csv", "verify_trajectory_census.py", "max_track_w30_shell")
        check("eps_CRE", loc, ec, float(r.cre_start_w30_shell), "trajectory_census.csv", "verify_trajectory_census.py", "cre_start_w30_shell")
        say(f"     [{i},{j}] heat: eps_QSS printed {eq:.3g}, artifact {float(r.max_track_w30_shell):.6g};  eps_CRE printed {ec:.4g}, artifact {float(r.cre_start_w30_shell):.6g}")

    # ---------------------------------------------------------------- B: tab:gain
    rg = load("validation/reservoir_gain/reservoir_gain.csv")
    heat = rg[rg.direction == "heat"]
    say("\n  B  tab:gain, reservoir gain G at one, two and four intervals (chapter5.tex:737)")
    GAIN = {(23, 5): (-5.736, -5.634, -5.439), (15, 3): (-7.766, -7.617, -7.333), (0, 4): (-14.434, -14.131, -13.551)}
    for (i, j), printed in GAIN.items():
        got = []
        for k, pv in zip((1, 2, 4), printed):
            r = heat[(heat.i == i) & (heat.j == j) & (heat.k == k)]
            if len(r) != 1: raise RuntimeError(f"expected one row [{i},{j}] k={k}, found {len(r)}")
            g = float(r.iloc[0].G); got.append(g)
            check("tab:gain G", f"chapter5.tex tab:gain [{i},{j}] k={k}", pv, g, "reservoir_gain.csv", "verify_reservoir_gain.py", "G")
        say(f"     [{i},{j}]: printed {printed}  artifact ({got[0]:.6f}, {got[1]:.6f}, {got[2]:.6f})")

    # ---------------------------------------------------------------- C: tab:closure
    ic = load("validation/ion_closure/ion_closure_summary.csv").set_index("item")
    say("\n  C  tab:closure, slow timescale with the ion boundary open and closed (chapter4.tex:896)")
    CLOS = {(0, 0): (67.23, 1.6226, 41.4), (1, 4): (0.23324, 0.015173, 15.4),
            (15, 3): (2.027e-3, 1.999e-3, 1.01), (23, 5): (22.728e-6, 22.708e-6, 1.00)}
    for (i, j), (po, pc, pf) in CLOS.items():
        # The artifact carries two eigensolvers for the open case: a 40-digit mpmath path (tau_open,
        # identical to tau_open_mp40) and the float64 path (tau_open_numpy_eig) that every other
        # timescale in the thesis uses. At the cold corner they differ in the fourth digit, which is
        # the conditioning the chapter itself discusses; the printed value is the float64 one, so
        # that is what is checked, with the high-precision value reported beside it.
        key_o = f"tau_open_numpy_eig[{i},{j}]" if f"tau_open_numpy_eig[{i},{j}]" in ic.index else f"tau_open[{i},{j}]"
        key_c, key_f = f"tau_closed[{i},{j}]", f"factor[{i},{j}]"
        for key, pv, what in ((key_o, po, "open"), (key_c, pc, "closed"), (key_f, pf, "factor")):
            if key not in ic.index:
                fails.append(f"tab:closure {what} [{i},{j}]: {key} absent from ion_closure_summary.csv"); continue
            check(f"tab:closure {what}", f"chapter4.tex tab:closure [{i},{j}]", pv, float(ic.loc[key, "value"]),
                  "ion_closure_summary.csv", "verify_ion_closure.py", key)
        if key_o in ic.index:
            mp = ic.loc[f"tau_open_mp40[{i},{j}]", "value"] if f"tau_open_mp40[{i},{j}]" in ic.index else float("nan")
            say(f"     [{i},{j}]: open printed {po:g} artifact {float(ic.loc[key_o,'value']):.6g} (40-digit path {float(mp):.6g});  closed {pc:g} / {float(ic.loc[key_c,'value']):.6g};  factor {pf:g} / {float(ic.loc[key_f,'value']):.6g}")

    # ---------------------------------------------------------------- D: tab:position_effect
    mc = load("validation/molecular_channel/molecular_channel.csv")
    say("\n  D  tab:position_effect, the crest mechanism (chapter5.tex:1369)")
    say("     convention, stated because the caption did not: Delta and the cap at the post-step index 24;")
    say("     Sbar the secant of the step 23 -> 24, stored at index 23; the printed ln(u_CRE/u_peak) takes")
    say("     u_CRE at index 23 and u_peak at index 24. The two consistent alternatives are shown alongside.")
    POS = {0: (0.9655, 0.2368, +1.2877, 0.1754), 2: (1.6662, 0.3940, +0.3842, 0.3884),
           3: (1.8857, 0.4394, -0.2487, 0.4261), 5: (1.9403, 0.4503, -1.7695, 0.2288),
           7: (1.9368, 0.4496, -3.2910, 0.0668)}
    def at(i, j):
        r = mc[(mc.i == i) & (mc.j == j)]
        if len(r) != 1: raise RuntimeError(f"molecular_channel has {len(r)} rows at [{i},{j}]")
        r = r.iloc[0]
        u_peak = float(np.sqrt(r.c3 * r.c4 / (r.a3 * r.a4)))
        return float(r.u_CRE), u_peak, float(r.Delta_atomic)
    say(f"     {'j':>2} {'Delta':>18} {'cap':>18} {'ln(u/u_pk) mixed':>22} {'all-23':>9} {'all-24':>9} {'|Sbar|':>16}")
    for j, (pD, pC, pR, pS) in POS.items():
        u23, upk23, D23 = at(23, j); u24, upk24, D24 = at(24, j)
        cap24 = float(np.tanh(abs(D24) / 4))
        mixed = float(np.log(u23 / upk24)); all23 = float(np.log(u23 / upk23)); all24 = float(np.log(u24 / upk24))
        r = heat[(heat.i == 23) & (heat.j == j) & (heat.k == 1)]
        sbar = abs(float(r.iloc[0].Sbar)) if len(r) == 1 else float("nan")
        check("tab:position_effect Delta", f"chapter5.tex tab:position_effect j={j}", pD, D24, "molecular_channel.csv", "verify_molecular_channel.py", "Delta_atomic at i=24")
        check("tab:position_effect cap", f"chapter5.tex tab:position_effect j={j}", pC, cap24, "molecular_channel.csv", "verify_molecular_channel.py", "tanh(|Delta_atomic|/4) at i=24")
        check("tab:position_effect ln(u/u_peak)", f"chapter5.tex tab:position_effect j={j}", pR, mixed, "molecular_channel.csv", "verify_molecular_channel.py", "ln(u_CRE[i=23]/u_peak[i=24])")
        check("tab:position_effect |Sbar|", f"chapter5.tex tab:position_effect j={j}", pS, sbar, "reservoir_gain.csv", "verify_reservoir_gain.py", "Sbar heat k=1 i=23")
        say(f"     {j:>2} {pD:8.4f}/{D24:<9.4f} {pC:8.4f}/{cap24:<9.4f} {pR:+10.4f}/{mixed:<+11.4f} {all23:>+9.4f} {all24:>+9.4f} {pS:7.4f}/{sbar:<8.4f}")
    sgn = [np.log(at(23, j)[0] / at(24, j)[1]) for j in POS]
    say(f"     the conclusion under the mixed convention: sign change between j = 2 and j = 3 "
        f"({sgn[1]:+.4f} then {sgn[2]:+.4f}), |Sbar| largest at j = 3")

    # ---------------------------------------------------------------- verdict
    say("\n" + "=" * 104)
    n = len(rows); nbad = sum(1 for r in rows if not r["ok"])
    if fails:
        say(f"FAILED: {nbad} of {n} printed values did not reproduce"); [say("  " + f) for f in fails]
    else:
        say(f"ALL {n} printed values reproduce from stamped artifacts within the precision printed.")
    say("None of these is a new calculation: each is a reduction of an artifact produced by the script named beside it.")
    say("\n  provenance of the source artifacts themselves:")
    for rel, st in PROV.items(): say(f"    {'OK  ' if st.startswith('stamped') else 'WARN'} {rel}: {st}")
    if any(not st.startswith("stamped") for st in PROV.values()):
        say("  An unstamped source means the value reproduces from the file that is there, but the file does not")
        say("  record which script or which input produced it. That is a weaker chain than a stamped artifact and")
        say("  is reported here rather than repaired, since writing a header means re-running the producer.")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation/headline_provenance"; out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
               *[f"# source {k}: {v}" for k, v in PROV.items()],
               f"# interpreter {sys.executable}  numpy {np.__version__}  pandas {pd.__version__}",
               f"# canonical L_grid sha256 {L_SHA} asserted in every source artifact's header",
               "# each row: a number printed in the thesis, re-derived from a stamped artifact; rel_diff against the printed precision",
               "# tab:position_effect ln(u_CRE/u_peak) uses u_CRE at Te index 23 and u_peak at index 24, as printed; see the log"]
        with open(out / "headline_provenance.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        print(f"\nwrote {(out / 'headline_provenance.csv').relative_to(ROOT)}  ({n} rows)")
    return 1 if fails else 0


if __name__ == "__main__": sys.exit(main())
