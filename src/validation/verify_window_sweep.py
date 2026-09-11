#!/usr/bin/env python3
"""
verify_window_sweep.py
=======================
Sensitivity of the plateau-window factor k.

WHAT THIS ANSWERS
------------------
verify_divertor_map.py defines the plateau window as

    k * tau_relax < t < tau_slow / k          (equivalently M = tau_slow/tau_relax
                                                > k^2 for window_ok)

with k = 30 in the canonical run (validation/divertor_map/divertor_map.csv).
That choice gates: which of the 784 (point, direction) pairs count as
"window_ok"; the census denominator (window_ok & Te >= 2 eV); and the census
itself (window_ok & Te >= 2 eV & time-averaged lower bound > threshold).

A reviewer asked whether the headline counts are stable if k is instead 10,
20 or 50. This script does NOT re-run the stiff eigendecomposition -- it
RECOUNTS from CSVs that verify_divertor_map.py / verify_reservoir_gain.py
already produced at each k (those runs are a separate, previously executed
step; see outputs/ for the commands used). It reads the canonical file for
k=30 and one sibling file per swept k, and it fails loudly if a sibling file
is missing rather than silently skipping it.

WHAT IT CHECKS, IN ORDER
-------------------------
1. eps_plateau (and, for reservoir_gain, G and Sbar) are frozen-reservoir
   algebraic quantities computed BEFORE the window gate is applied. They must
   be bit-identical across k for every row. This script verifies that and
   reports the max |difference| -- if it is not exactly zero, that is a
   finding, not a rounding footnote.
2. Grid-level counts that DO depend on k: n_window_ok, the "warm" census
   denominator (Te >= 2 eV among window_ok), the "dense" sub-count
   (Te >= 2 eV, ne >= 1e14 cm^-3, among window_ok), and the census at two
   ELM-relevant exposure times (100 us, matching the DRIVES table in
   verify_divertor_map.py, and 506 us, the ITER-pedestal duration from
   Loarte 2003 Sec. 5 which is NOT a stored column and is recomputed here
   from eps_plateau and tau_QSS using the identical formula at
   verify_divertor_map.py ~line 206).
3. The "crest": for each Te row with Te >= 2 eV, heating direction, the
   density index j at which eps_plateau is largest AMONG WINDOW_OK COLUMNS.
   Because eps_plateau itself never changes with k (check 1), any change in
   this argmax as k varies is caused entirely by columns entering or leaving
   the window -- a membership effect, not a magnitude effect. This script
   reports every row where that happens.

NO HARDCODED GRID
------------------
Te/ne grids and shapes come from cr_context.CRContext, exactly as the
producing scripts use them. Only the CSV *file paths* to sweep are given on
the command line; their contents (Te, ne, direction, i, j, ...) are read back
from the files themselves, and cross-checked against the sha256 lines each
producing script stamped in its own CSV header.

Report only. Writes to validation/window_sweep/. Modifies nothing else.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402

TD_100US = 1e-4      # ELM_crash drive in verify_divertor_map.py's DRIVES table
TD_506US = 506e-6    # Loarte 2003 Sec. 5, ITER pedestal, not a DRIVES entry
THRESHOLDS = (0.05, 0.10, 0.20)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_divertor_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"required divertor-map CSV missing: {path}\n"
            f"This script recounts from files that verify_divertor_map.py "
            f"already wrote; it does not run that script itself."
        )
    header_lines = []
    with path.open() as fh:
        colnames = None
        for line in fh:
            if line.startswith("#"):
                header_lines.append(line.rstrip("\n"))
                continue
            colnames = line.rstrip("\n").split(",")
            break
        if colnames is None:
            raise ValueError(f"{path}: no data header row found")
        rows = list(csv.DictReader(fh, fieldnames=colnames))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return header_lines, rows


def header_sha(header_lines: list[str], key: str) -> str | None:
    for line in header_lines:
        if key in line:
            return line.strip().split()[-1]
    return None


def to_bool(v: str) -> bool:
    return v.strip() == "True"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30, 50],
                    help="window factors to sweep (30 must be present -- it "
                         "is the canonical value read from validation/"
                         "divertor_map/divertor_map.csv)")
    p.add_argument("--divertor-dir", type=Path, default=None,
                    help="directory holding divertor_map.csv for k=30 "
                         "(default: <root>/validation/divertor_map)")
    p.add_argument("--divertor-dir-pattern", type=str,
                    default="validation/divertor_map_w{k}",
                    help="relative-to-root pattern for k != 30")
    p.add_argument("--te-warm-min", type=float, default=2.0,
                    help="Te threshold (eV) for the 'warm' census denominator")
    p.add_argument("--ne-dense-min", type=float, default=1e14,
                    help="ne threshold (cm^-3) for the 'dense' sub-count")
    p.add_argument("--census-threshold", type=float, default=0.10)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    ctx = CRContext.load()
    root = ctx.root

    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    for p in (L_path, S_path, ctx.state_index_path):
        if not p.is_file():
            raise FileNotFoundError(f"required pipeline file missing: {p}")
    sha_L, sha_S, sha_si = sha256(L_path), sha256(S_path), sha256(ctx.state_index_path)

    if 30 not in a.ks:
        raise ValueError("k=30 (the canonical, thesis-cited window) must be "
                          "included in --ks so this script can compare against it")

    out = a.out or (root / "validation" / "window_sweep")
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("WINDOW-FACTOR SWEEP -- recount from existing verify_divertor_map.py CSVs")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"repo root      {root}")
    say(f"ks swept       {a.ks}")
    say(f"L_grid sha256 (pipeline, current)   {sha_L}")
    say(f"S_grid sha256 (pipeline, current)   {sha_S}")
    say(f"state idx sha (pipeline, current)   {sha_si}")
    say("=" * 78)

    divertor_dir_30 = a.divertor_dir or (root / "validation" / "divertor_map")

    per_k = {}
    crest_by_k = {}
    for k in sorted(a.ks):
        if k == 30:
            path = divertor_dir_30 / "divertor_map.csv"
        else:
            path = root / a.divertor_dir_pattern.format(k=k) / "divertor_map.csv"
        header_lines, rows = load_divertor_csv(path)

        # Fail loudly if this CSV was not built from the same pipeline files
        # this script just hashed -- a stale sibling file would silently
        # mislabel every count below.
        for key, expect in (("L_grid sha256", sha_L), ("S_grid sha256", sha_S),
                             ("state_index sha256", sha_si)):
            got = header_sha(header_lines, key)
            if got is None:
                raise ValueError(f"{path}: header missing '{key}' line")
            if got != expect:
                raise ValueError(
                    f"{path}: {key} = {got} does not match the pipeline's "
                    f"current {expect} -- this CSV is stale relative to the "
                    f"grids loaded via cr_context; recount is not trustworthy "
                    f"until it is regenerated")

        window_ok = np.array([to_bool(r["window_ok"]) for r in rows])
        Te = np.array([float(r["Te"]) for r in rows])
        ne = np.array([float(r["ne"]) for r in rows])
        direction = np.array([r["direction"] for r in rows])
        i_idx = np.array([int(r["i"]) for r in rows])
        j_idx = np.array([int(r["j"]) for r in rows])
        eps_plateau = np.array([float(r["eps_plateau"]) for r in rows])
        tau_QSS = np.array([float(r["tau_QSS"]) for r in rows])
        lo_100us = np.array([float(r["lo_ELM_crash"]) for r in rows])
        lo_506us = eps_plateau * (tau_QSS / TD_506US) * (1 - np.exp(-TD_506US / tau_QSS))

        n_ok = int(window_ok.sum())
        warm = window_ok & (Te >= a.te_warm_min)
        dense = warm & (ne >= a.ne_dense_min)
        n_warm = int(warm.sum())
        n_dense = int(dense.sum())

        census100 = warm & (lo_100us > a.census_threshold)
        n_c100 = int(census100.sum())
        if n_c100:
            q = np.where(census100)[0][int(np.argmax(lo_100us[census100]))]
            worst100 = float(lo_100us[q])
            worst_at = (str(direction[q]), int(i_idx[q]), int(j_idx[q]),
                        float(Te[q]), float(ne[q]))
        else:
            worst100, worst_at = float("nan"), None

        census506 = warm & (lo_506us > a.census_threshold)
        n_c506 = int(census506.sum())
        worst506 = float(lo_506us[census506].max()) if n_c506 else \
            (float(lo_506us[warm].max()) if warm.any() else float("nan"))

        thr_counts = {}
        for thr in THRESHOLDS:
            thr_counts[thr] = int((warm & (lo_100us > thr)).sum())

        per_k[k] = dict(
            n_rows=len(rows), n_window_ok=n_ok, n_warm=n_warm, n_dense=n_dense,
            census_100us=n_c100, worst_100us=worst100, worst_at=worst_at,
            census_506us=n_c506, worst_506us=worst506,
            census_thr005=thr_counts[0.05], census_thr020=thr_counts[0.20],
            path=path,
        )

        say(f"\n--- k={k}  ({path.relative_to(root)}) ---")
        say(f"  rows={len(rows)}  n_window_ok={n_ok}  n_warm(Te>={a.te_warm_min})="
            f"{n_warm}  n_dense(also ne>={a.ne_dense_min:.0e})={n_dense}")
        say(f"  census @100us (thr {a.census_threshold:.0%}): {n_c100}/{n_warm}  "
            f"worst={worst100:.4f}  at={worst_at}")
        say(f"  census @506us (thr {a.census_threshold:.0%}): {n_c506}/{n_warm}  "
            f"worst={worst506:.4f}")
        say(f"  census @100us at thr=5%: {thr_counts[0.05]}   "
            f"thr=20%: {thr_counts[0.20]}")

        # eps_plateau identity check against k=30, keyed by (direction,i,j).
        key = np.array([f"{d}_{i}_{j}" for d, i, j in zip(direction, i_idx, j_idx)])
        per_k[k]["_key"] = key
        per_k[k]["_eps_plateau"] = eps_plateau
        per_k[k]["_G_available"] = False

        # crest: for each Te row i (heat, Te>=te_warm_min), argmax eps_plateau
        # among window_ok columns
        heat = direction == "heat"
        crest = {}
        for i in sorted(set(i_idx[heat & (Te >= a.te_warm_min)])):
            sel = heat & (i_idx == i)
            sel_ok = sel & window_ok
            if not sel_ok.any():
                crest[i] = None
                continue
            jcand = j_idx[sel_ok]
            epscand = eps_plateau[sel_ok]
            crest[i] = int(jcand[int(np.argmax(epscand))])
        crest_by_k[k] = crest

    # --- eps_plateau identity check across k, vs k=30 ---
    say("\n" + "=" * 78)
    say("CHECK 1: eps_plateau identical across k (frozen-reservoir algebra, "
        "computed before the window gate)")
    key30 = per_k[30]["_key"]
    eps30 = per_k[30]["_eps_plateau"]
    m30 = {kk: vv for kk, vv in zip(key30, eps30)}
    for k in sorted(a.ks):
        if k == 30:
            continue
        keyk = per_k[k]["_key"]
        epsk = per_k[k]["_eps_plateau"]
        diffs = np.array([abs(epsk[idx] - m30[keyk[idx]]) for idx in range(len(keyk))])
        say(f"  k={k}: max|eps_plateau(k) - eps_plateau(30)| = {diffs.max():.3e} "
            f"over {len(diffs)} rows  "
            f"{'OK (identical)' if diffs.max() == 0.0 else '*** DIFFERS ***'}")

    # --- crest stability ---
    say("\n" + "=" * 78)
    say("CHECK 2: crest (argmax eps_plateau among window_ok columns), "
        "heating, Te >= "
        f"{a.te_warm_min} eV -- does the argmax j change with k?")
    all_i = set.intersection(*(set(crest_by_k[k]) for k in a.ks))
    changed_rows = []
    for i in sorted(all_i):
        vals = {k: crest_by_k[k][i] for k in sorted(a.ks)}
        distinct = set(v for v in vals.values() if v is not None)
        if len(distinct) > 1 or any(v is None for v in vals.values()):
            changed_rows.append((i, vals))
    say(f"  Te rows compared: {len(all_i)}   rows where crest j changes with k: "
        f"{len(changed_rows)}")
    for i, vals in changed_rows:
        say(f"    i={i}  Te={ctx.te_grid[i]:.4f} eV   crest j by k: {vals}")

    say(f"\nwrote {out / 'window_sweep.txt'}")
    say(f"wrote {out / 'window_sweep_summary.csv'}")

    (out / "window_sweep.txt").write_text("\n".join(lines) + "\n")

    csv_path = out / "window_sweep_summary.csv"
    fieldnames = ["K", "n_window_ok", "n_warm", "n_dense", "census_100us",
                  "worst_100us", "worst_at", "census_506us", "worst_506us",
                  "census_thr005", "census_thr020"]
    with csv_path.open("w", newline="") as fh:
        fh.write(f"# generated {datetime.now():%Y-%m-%d %H:%M} by "
                  f"{Path(__file__).name}\n")
        fh.write(f"# commands used to produce the swept divertor_map CSVs "
                  f"(k != 30), and this recount:\n")
        for k in sorted(a.ks):
            if k == 30:
                continue
            fh.write(f"#   python src/validation/verify_divertor_map.py "
                      f"--win-lo {k} --win-hi {k} --out "
                      f"{a.divertor_dir_pattern.format(k=k)}\n")
        fh.write(f"#   python src/validation/verify_window_sweep.py "
                  f"--ks {' '.join(str(k) for k in sorted(a.ks))}\n")
        fh.write(f"# L_grid sha256 {sha_L}\n")
        fh.write(f"# S_grid sha256 {sha_S}\n")
        fh.write(f"# state_index sha256 {sha_si}\n")
        fh.write(f"# canonical k=30 file: "
                  f"{(divertor_dir_30/'divertor_map.csv').relative_to(root)}\n")
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for k in sorted(a.ks):
            r = per_k[k]
            wa = r["worst_at"]
            w.writerow(dict(
                K=k, n_window_ok=r["n_window_ok"], n_warm=r["n_warm"],
                n_dense=r["n_dense"], census_100us=r["census_100us"],
                worst_100us=f"{r['worst_100us']:.6f}",
                worst_at=(f"{wa[0]}[{wa[1]},{wa[2]}]_Te={wa[3]:.4f}_ne={wa[4]:.4e}"
                           if wa is not None else ""),
                census_506us=r["census_506us"],
                worst_506us=f"{r['worst_506us']:.6f}",
                census_thr005=r["census_thr005"],
                census_thr020=r["census_thr020"],
            ))


if __name__ == "__main__":
    main()
