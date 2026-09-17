#!/usr/bin/env python
"""
verify_anderson_subset_impact.py
================================
Which transitions carry the atomic-data sensitivity of the timescale
separation: the CCC -> RMPS substitution of ccc_anderson_grid_impact.py,
repeated for subsets of the 85 benchmarked transitions.

WHY THIS EXISTS
---------------
Chapter 2 (tab:atomic_sensitivity) argues that the n=5 disagreement between
CCC and RMPS does not reach the thesis because the 50 transitions into n=5
move M by under 1% while the 14 ground-state transitions move it by 12%.
Those subset rows were computed once and recorded in
data/processed/collisions/ccc_anderson.md (section 7.5); no script in the
repository produced them, and the table cited a CSV that does not contain
them. This is the producing script.

METHOD (identical to ccc_anderson_grid_impact.py, section [C])
-------------------------------------------------------------
L(Te, ne) = R(Te) + ne C(Te), recovered by least squares over the density
axis and asserted exact. For each transition in a subset, the excitation and
de-excitation elements of C are scaled by the same factor K_RMPS / K_CCC,
read per temperature from ccc_vs_anderson2002_thesis_Te_grid.csv, so detailed
balance survives; the diagonal of each source column is compensated so that
column sums, and with them the loss to the continuum, are unchanged.
tau_slow = 1/|lambda_0| and tau_relax = 1/|lambda_1| from the two smallest
|Re lambda| of the re-solved operator at every grid point; no eigenvalue
filter.

PREDICTIONS (written before the run, from ccc_anderson.md section 7.5)
---------------------------------------------------------------------
P0  the unperturbed operator reproduces tau_slow = 22.73 us,
    tau_relax = 2.277 ns and M = 9982 at the benchmark point  (asserted)
P1  all 85 transitions:   mean|d tau_slow| 11.42 %,  dM(benchmark) -11.29 %
P2  n_upper = 5 only:      4.72 %,   -0.94 %
P3  n_upper <= 4 only:     7.25 %,  -10.82 %
P4  ground state only:    10.45 %,  -12.36 %
P5  n_upper <= 3 only:     4.54 %,   -6.97 %
P6  n_upper = 4 only:      2.97 %,   -4.37 %

REFUTING OBSERVATION
--------------------
P2 giving |dM| of several percent, or P4 giving much less than P1, would mean
the section's argument -- the slow mode is ground-state depletion, so the
1s -> nl rates set it almost alone -- is unsupported.

OUTPUTS (with --write)
----------------------
validation/anderson_subset_impact/anderson_subset_impact.csv   one row per subset
validation/anderson_subset_impact/anderson_subset_impact.txt   this run's log
Both carry a '#' header with the script name, date, interpreter and the
sha256 of L_grid.npy, Te_grid_L.npy, ne_grid_L.npy, state_index.csv and
ccc_vs_anderson2002_thesis_Te_grid.csv.
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

BENCH_TE, BENCH_NE = 2.947, 1.389e14
REC_TAU_SLOW_US, REC_TAU_RELAX_NS, REC_M = 22.73, 2.277, 9982.0
L_CHAR = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}

# (name, predicate on (n_lo, n_up), recorded mean|d tau_slow| %, recorded dM %)
SUBSETS = [
    ("all benchmarked (n_upper <= 5)", lambda nlo, nup: True,          11.42, -11.29),
    ("n_upper = 5 only",               lambda nlo, nup: nup == 5,       4.72,  -0.94),
    ("n_upper <= 4 only",              lambda nlo, nup: nup <= 4,       7.25, -10.82),
    ("ground state 1s -> nl only",     lambda nlo, nup: nlo == 1,      10.45, -12.36),
    ("n_upper <= 3 only",              lambda nlo, nup: nup <= 3,       4.54,  -6.97),
    ("n_upper = 4 only",               lambda nlo, nup: nup == 4,       2.97,  -4.37),
    ("1s -> 5l only (ground-fed n=5)", lambda nlo, nup: nlo == 1 and nup == 5, None, None),
    ("nl -> 5l', n >= 2 (excited-fed n=5)", lambda nlo, nup: nlo >= 2 and nup == 5, None, None),
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true", help="stamp outputs under validation/")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ctx = CRContext.load()
    root = ctx.root
    L, TeL, neL = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    nT, nN, nS, _ = L.shape
    csv_path = root / "data/processed/collisions/ccc_vs_anderson2002_thesis_Te_grid.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} missing: run ccc_anderson_grid_impact.py first")

    si = pd.read_csv(ctx.state_index_path)
    for c in ("idx", "n", "l", "bundled"):
        if c not in si.columns:
            raise KeyError(f"{ctx.state_index_path} has no column {c!r}")
    nl2idx = {(int(r.n), int(r.l)): int(r.idx) for _, r in si.iterrows() if not r.bundled}

    log: list[str] = []
    def say(s: str = "") -> None:
        print(s); log.append(s)

    say("=" * 78)
    say("ANDERSON SUBSET IMPACT -- which transitions move the timescale separation")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"interpreter   {sys.executable}")
    say(ctx.describe())
    say(f"  ratio source : {csv_path.relative_to(root)}")
    say("=" * 78)

    # --- L = R + ne C, exact ------------------------------------------------
    A = np.vstack([np.ones_like(neL), neL]).T
    coef = np.linalg.lstsq(A, L.transpose(1, 0, 2, 3).reshape(nN, -1), rcond=None)[0]
    R = coef[0].reshape(nT, nS, nS)
    C = coef[1].reshape(nT, nS, nS)
    rel = np.abs(R[:, None] + neL[None, :, None, None] * C[:, None] - L).max() / np.abs(L).max()
    say(f"\nL = R + ne*C exact to {rel:.2e} (relative)")
    if rel > 1e-10:
        raise ValueError("L is not linear in ne; the substitution is invalid")

    def timescales(Lg):
        ev = np.sort(np.abs(np.linalg.eigvals(Lg.reshape(-1, nS, nS)).real), axis=1)
        return (1.0 / ev[:, 0]).reshape(nT, nN), (1.0 / ev[:, 1]).reshape(nT, nN)

    ti, ni = ctx.nearest_point(BENCH_TE, BENCH_NE)
    tQ0, tR0 = timescales(L)
    M0 = tQ0 / tR0
    say(f"benchmark nearest_point({BENCH_TE}, {BENCH_NE:.3e}) = [{ti},{ni}]  "
        f"Te={TeL[ti]:.4f} ne={neL[ni]:.4e}")
    say(f"P0  tau_slow {tQ0[ti,ni]*1e6:.3f} us (recorded {REC_TAU_SLOW_US}), "
        f"tau_relax {tR0[ti,ni]*1e9:.4f} ns ({REC_TAU_RELAX_NS}), M {M0[ti,ni]:.1f} ({REC_M:.0f})")
    if not (abs(tQ0[ti, ni]*1e6 - REC_TAU_SLOW_US) < 0.05
            and abs(tR0[ti, ni]*1e9 - REC_TAU_RELAX_NS) < 0.005):
        raise AssertionError("P0 FAILED: the unperturbed operator does not reproduce "
                             "the recorded benchmark timescales; do not read on")
    say("    P0 REPRODUCED")

    # --- per-transition ratio K_RMPS / K_CCC on the working grid ------------
    g = pd.read_csv(csv_path)
    trans = {}
    for lab, s in g.groupby("label"):
        s = s.sort_values("Te_eV")
        if not np.allclose(s.Te_eV.values, TeL):
            raise ValueError(f"{lab}: CSV temperatures do not match Te_grid_L")
        lo, up = lab.split("->")
        trans[lab] = dict(n_lo=int(lo[:-1]), l_lo=L_CHAR[lo[-1]],
                          n_up=int(up[:-1]), l_up=L_CHAR[up[-1]],
                          f=(s.K_And2002 / s.K_CCC_stored).values)
    say(f"\n{len(trans)} transitions with a ratio on all {nT} temperatures")

    def substitute(pred):
        Cp = C.copy(); n = 0
        for lab, t in trans.items():
            if not pred(t["n_lo"], t["n_up"]):
                continue
            i = nl2idx.get((t["n_lo"], t["l_lo"])); j = nl2idx.get((t["n_up"], t["l_up"]))
            if i is None or j is None:
                raise KeyError(f"{lab}: state not l-resolved in state_index")
            for a, b in ((j, i), (i, j)):          # excitation, de-excitation
                d = C[:, a, b] * (t["f"] - 1.0)
                Cp[:, a, b] += d
                Cp[:, b, b] -= d                    # column sum unchanged
            n += 1
        drift = np.abs((Cp - C).sum(axis=1)).max() / np.abs(C).max()
        if drift > 1e-12:
            raise AssertionError(f"column-sum drift {drift:.2e}")
        tQ, tR = timescales(R[:, None] + neL[None, :, None, None] * Cp[:, None])
        dQ = 100 * (tQ / tQ0 - 1); dM = 100 * ((tQ / tR) / M0 - 1)
        return n, dQ, dM

    say("\n" + "-" * 78)
    say(f"  {'subset':38s} {'n':>3} {'mean|dtau|':>11} {'max|dtau|':>10} "
        f"{'dM@bench':>9} {'mean|dM|':>9}   recorded")
    rows = []; n_fail = 0
    for name, pred, rec_t, rec_m in SUBSETS:
        n, dQ, dM = substitute(pred)
        mt, xt, bm_, mm = np.abs(dQ).mean(), np.abs(dQ).max(), dM[ti, ni], np.abs(dM).mean()
        if rec_t is None:
            verdict = "(not previously recorded)"
        else:
            ok = abs(mt - rec_t) <= 0.006 and abs(bm_ - rec_m) <= 0.006
            verdict = f"{rec_t:.2f} / {rec_m:+.2f}  {'REPRODUCED' if ok else '*** NOT REPRODUCED ***'}"
            n_fail += (not ok)
        say(f"  {name:38s} {n:3d} {mt:10.2f}% {xt:9.2f}% {bm_:+8.2f}% {mm:8.2f}%   {verdict}")
        rows.append(dict(subset=name, n_transitions=n, mean_abs_pct_d_tau_slow=mt,
                         max_abs_pct_d_tau_slow=xt, pct_d_M_benchmark=bm_,
                         mean_abs_pct_d_M=mm, recorded_mean_abs_pct_d_tau_slow=rec_t,
                         recorded_pct_d_M_benchmark=rec_m))
    say("-" * 78)
    say("\nNOTE  n_upper = 5 is a net of two opposing parts: the five 1s -> 5l "
        "transitions alone and the forty-five excited-fed ones pull dM in "
        "opposite directions (last two rows).")
    say(f"\nVERDICT: {n_fail} of 6 recorded rows not reproduced")

    if args.write:
        out = Path(args.out) if args.out else root / "validation/anderson_subset_impact"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}",
               f"# interpreter {sys.executable}  numpy {np.__version__}",
               f"# L_grid.npy      sha256 {sha256_file(root / 'data/processed/cr_matrix/L_grid.npy')}",
               f"# Te_grid_L.npy   sha256 {sha256_file(root / 'data/processed/cr_matrix/Te_grid_L.npy')}",
               f"# ne_grid_L.npy   sha256 {sha256_file(root / 'data/processed/cr_matrix/ne_grid_L.npy')}",
               f"# state_index.csv sha256 {sha256_file(ctx.state_index_path)}",
               f"# ratio csv       sha256 {sha256_file(csv_path)}  ({csv_path.relative_to(root)})",
               f"# benchmark point [{ti},{ni}]  Te={TeL[ti]:.4f} eV  ne={neL[ni]:.4e} cm^-3"]
        with open(out / "anderson_subset_impact.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n")
            pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "anderson_subset_impact.txt", "w") as fh:
            fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/anderson_subset_impact.{{csv,txt}}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
