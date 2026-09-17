#!/usr/bin/env python
"""
verify_ccc_accounting.py
========================
The CCC dataset, counted from the raw directory to the 819 matrix pairs, with
every integer asserted.

WHY THIS EXISTS
---------------
Chapter 2 (sec on the CCC data) records two provenance discrepancies as "noted
here rather than reconciled": the acquisition report's 3117 files against 3115
on disk, and the Week-2 count of 1740 parsed blocks against the current 2190.
This script does the accounting from the files themselves.

CHAIN
-----
  raw files in data/raw/ccc/e-H_XSEC_LS   (gitignored; must be present)
    = STATE.STATE files, split by direction (filenames read FINAL.INITIAL)
      + files carrying bundled-shell labels
  STATE.STATE with dn != 0 = excitation + de-excitation      (the Week-2 1740)
  K_CCC_metadata.csv excitation blocks by n_final              (the current 1320)
  collapse over l of the n = 9 and n = 10 final shells         (618 CCC pairs)
  + Vriens-Smeets pairs from K_exc_meta.csv 'source'           (819 total)
  = C(36,2) - same-shell pairs + 36*7 + C(7,2) for 36 resolved + 7 bundled states

PREDICTIONS (written before the run)
-----------------------------------
P1  files on disk == lines in file_list.txt == 3115
P2  STATE.STATE dn != 0: 870 excitation + 870 de-excitation = 1740
P3  metadata excitation blocks = 1320 = 870 + 450, the 450 all with n_final = 10
P4  by n_final: 546 (<= 8), 324 (= 9), 450 (= 10); 324 = 36 x 9, 450 = 45 x 10
P5  CCC pairs in K_exc_meta = 618 = 546 + 36 + 36; VS pairs = 201 = 36*5 + 21; total 819
P6  819 = C(36,2) - 84 + 36*7 + C(7,2)
The 3117 is the size of the database as REQUESTED (data/Processed/collisions/
REPORT.md, line ~708), not a count of files received; which two files are
absent cannot be settled without the provider's manifest and is reported open.

REFUTING OBSERVATION: any assertion above failing.

OUTPUTS (with --write): validation/ccc_accounting/ccc_accounting.{csv,txt}
"""
from __future__ import annotations
import argparse, hashlib, os, re, sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr_context import CRContext  # noqa: E402

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); root = ctx.root
    raw = root / "data/raw/ccc/e-H_XSEC_LS"; fl = root / "data/raw/ccc/file_list.txt"
    meta = root / "data/processed/collisions/ccc/K_CCC_metadata.csv"
    kmeta = root / "data/processed/collisions/K_exc_full/K_exc_meta.csv"
    for p in (raw, fl, meta, kmeta):
        if not p.exists(): raise FileNotFoundError(f"{p} missing; the accounting cannot be done without it")
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("CCC DATASET ACCOUNTING -- raw files to matrix pairs")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}"); say(f"interpreter {sys.executable}")
    say(ctx.describe()); say("=" * 78)
    rows = []
    def check(name, got, exp):
        ok = got == exp; rows.append(dict(step=name, value=got, expected=exp, ok=ok))
        say(f"  {name:58s} {got:>6}  (expected {exp})  {'OK' if ok else '*** FAIL ***'}")
        if not ok: raise AssertionError(f"{name}: {got} != {exp}")
    files = [f for f in os.listdir(raw) if not f.startswith(".")]
    check("P1 files on disk in e-H_XSEC_LS", len(files), 3115)
    check("P1 lines in file_list.txt", sum(1 for _ in open(fl)), 3115)
    pat = re.compile(r"^(\d+)[A-Z]\.(\d+)[A-Z]$")
    ss = [(int(m.group(1)), int(m.group(2))) for f in files if (m := pat.match(f))]
    exc = sum(nf > ni for nf, ni in ss); dex = sum(nf < ni for nf, ni in ss); same = sum(nf == ni for nf, ni in ss)
    say(f"  STATE.STATE files {len(ss)}: excitation {exc}, de-excitation {dex}, dn=0 {same}; other labels {len(files)-len(ss)}")
    check("P2 STATE.STATE excitation (FINAL > INITIAL)", exc, 870)
    check("P2 STATE.STATE de-excitation", dex, 870)
    check("P2 Week-2 count = exc + de-exc", exc + dex, 1740)
    m = pd.read_csv(meta)
    check("P3 metadata excitation blocks", len(m), 1320)
    byf = m.groupby("n_f").size()
    check("P4 blocks with n_final <= 8", int(byf[byf.index <= 8].sum()), 546)
    check("P4 blocks with n_final = 9", int(byf.get(9, 0)), 324)
    check("P4 blocks with n_final = 10", int(byf.get(10, 0)), 450)
    check("P3 1320 - 1740/2 = blocks the extended parse added", 1320 - 870, 450)
    check("P4 distinct initial states feeding n=9", int(m[m.n_f == 9].groupby(["n_i", "l_i"]).ngroups), 36)
    check("P4 distinct initial states feeding n=10", int(m[m.n_f == 10].groupby(["n_i", "l_i"]).ngroups), 45)
    km = pd.read_csv(kmeta)
    src = km.source.value_counts().to_dict(); say(f"  K_exc_meta sources: {src}")
    ccc_total = int(sum(v for k, v in src.items() if k.startswith("CCC")))
    check("P5 CCC resolved pairs (source CCC)", int(src.get("CCC", 0)), 546)
    check("P5 CCC pairs collapsed to n=9 (source CCC_n9)", int(src.get("CCC_n9", 0)), 36)
    check("P5 CCC pairs collapsed to n=10 (source CCC_n10)", int(src.get("CCC_n10", 0)), 36)
    check("P5 CCC pairs in the merged table, all sources", ccc_total, 618)
    check("P5 Vriens-Smeets pairs (source VS)", int(src.get("VS", 0)), 201)
    check("P5 total excitation pairs", int(len(km)), 819)
    si = pd.read_csv(ctx.state_index_path); nres = int((~si.bundled).sum()); nbun = int(si.bundled.sum())
    same_shell = sum(len(list(combinations(g, 2))) for _, g in si[~si.bundled].groupby("n").idx)
    check("P6 resolved states", nres, 36); check("P6 bundled states", nbun, 7)
    check("P6 C(36,2) - same-shell + 36*7 + C(7,2)", len(list(combinations(range(nres), 2))) - same_shell + nres * nbun + len(list(combinations(range(nbun), 2))), 819)
    say("\nOPEN: the acquisition report records 3117 files as the size of the database requested; 3115 arrived.\n"
        "      Which two are absent needs the provider's manifest and is not settled here.")
    say("\nVERDICT: every integer in the chain closes; the only unreconciled number is the 3117, which is not a count of files received.")
    if a.write:
        out = Path(a.out) if a.out else root / "validation/ccc_accounting"; out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}", f"# interpreter {sys.executable}",
               f"# K_CCC_metadata.csv sha256 {sha(meta)}", f"# K_exc_meta.csv sha256 {sha(kmeta)}", f"# file_list.txt sha256 {sha(fl)}",
               f"# state_index.csv sha256 {sha(ctx.state_index_path)}", f"# raw directory {raw.relative_to(root)} ({len(files)} files, gitignored)"]
        with open(out / "ccc_accounting.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "ccc_accounting.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/ccc_accounting.{{csv,txt}}")
    return 0

if __name__ == "__main__": sys.exit(main())
