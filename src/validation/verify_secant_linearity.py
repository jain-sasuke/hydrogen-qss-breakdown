#!/usr/bin/env python
"""
verify_secant_linearity.py
==========================
Establish that the step-dependence of the reservoir gain G is secant drift,
not measurement instability.

WHY THIS EXISTS
---------------
Chapter 5 defines the reservoir gain by

    Delta ln u = G * Delta ln Te

and `verify_reservoir_gain.py` computes it as G = ln x / dlnTe, which is a
SECANT over the step actually taken, not a derivative. An earlier draft of
eq:gain_def wrote G as d ln u / d ln Te, a local partial derivative. Those two
cannot both be right, and the table settles it: G takes three different values
(k = 1, 2, 4) at the same grid point, which no derivative can do.

That left the 5-6% spread across step size looking like scatter in a quantity
the chapter calls a property of the operating point. This script asks whether
the spread is instead the systematic, predictable drift of a secant away from
its tangent as the interval widens.

WHAT IT COMPUTES
----------------
For every grid point carrying all three step sizes, using only the stamped
`validation/reservoir_gain/reservoir_gain.csv`:

    fit    |G| = A - B*k   through the k = 1 and k = 4 entries
    predict the WITHHELD k = 2 entry
    report the relative error of that prediction

A is then an estimate of the tangent, i.e. of the derivative the earlier
draft claimed G already was.

THE CLAIM BEING TESTED
----------------------
If the spread is secant drift with slowly varying curvature, |G| is linear in
k and the withheld midpoint is predicted to well under a percent.

The refuting observation, stated in advance: if the k = 2 prediction erred by
anything approaching the 5-6% total spread, the linear-secant reading would be
wrong and the spread would have to be reported as unexplained step-dependence.
The script prints the worst case, so the claim carries a number.

This script READS a stamped artifact and does arithmetic on it. It runs no
solve and regenerates nothing. If the artifact is missing it fails loudly
rather than recomputing a substitute (CLAUDE.md rules 2 and 4).
"""
from __future__ import annotations

import csv as _csv
import statistics as _st
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()

# Deliberately NOT importing cr_context. That module is the mandated loader for
# grids, state ordering and matrices (CLAUDE.md rule 1); this script loads none
# of them -- it reduces one stamped CSV -- and importing it would pull in numpy
# for nothing. The root marker below is cr_context.REL_CR_MATRIX verbatim, so
# the two agree on what a repo root is.
_REL_CR_MATRIX = Path("data/processed/cr_matrix")


def find_repo_root(start: Path) -> Path:
    """Walk upward for the directory holding data/processed/cr_matrix."""
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / _REL_CR_MATRIX).is_dir():
            return candidate
    raise SystemExit(
        f"FATAL: no repo root containing {_REL_CR_MATRIX} above {here}."
    )


ROOT = find_repo_root(_HERE)
SRC = ROOT / "validation" / "reservoir_gain" / "reservoir_gain.csv"
OUT = ROOT / "validation" / "secant_linearity"

STEPS = (1, 2, 4)


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"FATAL: required artifact not found: {path}\n"
            "  Run verify_reservoir_gain.py first. This script does not "
            "recompute G; it reduces the stamped output."
        )
    with path.open() as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    rows = list(_csv.DictReader(lines))
    if not rows:
        raise SystemExit(f"FATAL: {path} contains no data rows.")
    need = {"direction", "k", "i", "j", "Te", "ne", "G", "window_ok"}
    missing = need - set(rows[0])
    if missing:
        raise SystemExit(f"FATAL: {path} is missing columns: {sorted(missing)}")
    return rows


def main() -> int:
    rows = _load(SRC)
    gain: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    meta: dict[tuple[int, int], tuple[float, float]] = {}
    for r in rows:
        if r["direction"] != "heat" or r["window_ok"] != "True":
            continue
        key = (int(r["i"]), int(r["j"]))
        gain[key][int(r["k"])] = abs(float(r["G"]))
        meta[key] = (float(r["Te"]), float(r["ne"]))

    full = {p: d for p, d in gain.items() if set(STEPS) <= set(d)}
    if not full:
        raise SystemExit(
            "FATAL: no grid point carries all of k = 1, 2, 4 with window_ok. "
            "The artifact does not support this reduction."
        )

    out_rows, errs = [], []
    for p, d in sorted(full.items()):
        B = (d[1] - d[4]) / 3.0          # |G| = A - B k, through k = 1 and 4
        A = d[1] + B                     # intercept: the tangent estimate
        pred = A - 2 * B                 # withheld midpoint
        err = abs(pred - d[2]) / d[2] * 100.0
        errs.append(err)
        Te, ne = meta[p]
        out_rows.append(dict(i=p[0], j=p[1], Te=f"{Te:.6g}", ne=f"{ne:.6g}",
                             G_k1=f"{d[1]:.6f}", G_k2=f"{d[2]:.6f}",
                             G_k4=f"{d[4]:.6f}", A_tangent=f"{A:.6f}",
                             B_slope=f"{B:.6f}", G_k2_pred=f"{pred:.6f}",
                             rel_err_pct=f"{err:.6f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "secant_linearity.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    errs.sort()
    med, worst = _st.median(errs), max(errs)
    n = len(errs)
    print(f"source     {SRC}")
    print(f"points     {n} heating points carrying k = 1, 2, 4")
    print("test       fit |G| = A - Bk through k = 1, 4; predict withheld k = 2")
    print(f"result     median {med:.4f}%, 90th {errs[int(0.9 * n)]:.4f}%, "
          f"worst {worst:.4f}%")
    print(f"           under 0.1%: {sum(1 for e in errs if e < 0.1)}/{n}")
    for name, key in (("benchmark", (23, 5)), ("crest", (15, 3)),
                      ("cold edge", (0, 4))):
        if key not in full:
            print(f"  {name:<10} NOT PRESENT in the artifact")
            continue
        d = full[key]
        B = (d[1] - d[4]) / 3.0
        A = d[1] + B
        print(f"  {name:<10} k=1 {d[1]:.3f}  k=2 {d[2]:.3f}  k=4 {d[4]:.3f}"
              f"   pred k=2 {A - 2 * B:.3f}   tangent A {A:.3f}")
    print(f"wrote      {OUT / 'secant_linearity.csv'}")
    if worst > 1.0:
        print("REFUTED: the withheld midpoint errs by more than 1%; the "
              "step-dependence is not simple secant drift.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
