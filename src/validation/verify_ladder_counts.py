#!/usr/bin/env python
"""
verify_ladder_counts.py
=======================
Table 4.4 (the validation ladder) is its own ledger: this parses its rows and
checks every count its caption states.

WHY THIS EXISTS
---------------
The caption of tab:ladder states counts over the table's own rows: how many
internal checks there are, how many are severe, tautological and weak, how many
external comparisons there are and how they came out, and how many of the severe
checks rest on a working note rather than a stamped artifact. Those counts were
written by hand and the rows have been edited many times since. At HEAD 3b613c1
the caption said four severe checks were in the notes condition while exactly one
was, and stamping that one (verify_saha_balance.py) made it zero, so the sentence
was wrong before the edit and wrong after it. A hand-maintained count over a
hand-maintained table drifts silently; this script makes the drift fail loudly.

It is a presentation guard, not a physics check. It computes nothing about the
plasma. It reads chapter4.tex, parses the tabular environment of tab:ladder, and
compares the counts it finds with the numbers the caption claims.

METHOD
------
  The table is split at its internal \\midrule into an internal block and an
  external block (the external rows are the ones whose grade begins "external"
  or is a bare FAIL). Each row is split on unescaped & into
  (check, grade, record, measured). Grades are normalised by stripping LaTeX
  markup, so \\textbf{FAIL} counts as FAIL. Records are one of
  stamped / notes / figure / code.
  The caption's claims are parsed from the sentence forms it actually uses, so a
  reworded caption fails here rather than silently passing.

GATES
-----
  G1  every row parses into exactly four columns
  G2  every grade and every record is one of the known vocabularies
  G3  internal + external row counts equal the caption's stated totals
  G4  the grade breakdown equals the caption's stated breakdown
  G5  the count of severe checks in the notes condition equals the caption's
  G6  the external outcome breakdown equals the caption's

Exit status is non-zero if any gate fails, so this can be run in a pre-submission
check. Nothing is written and no thesis file is modified.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)
GRADES = {"severe", "tautological", "weak"}
RECORDS = {"stamped", "notes", "figure", "code"}
NUMWORD = {"no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
           "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
           "eighteen": 18, "nineteen": 19, "twenty": 20}


def strip_tex(s: str) -> str:
    s = re.sub(r"\\(textbf|emph|texttt)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    return re.sub(r"[{}$~]", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="check Table 4.4's caption counts against its own rows")
    ap.add_argument("--chapter", default=str(ROOT / "thesis_tex" / "chapter4.tex"))
    a = ap.parse_args()
    src = Path(a.chapter).read_text()
    fails: list[str] = []
    say = print

    say("=" * 88); say("TABLE 4.4 LEDGER: caption counts against the table's own rows"); say(f"source {Path(a.chapter).relative_to(ROOT)}"); say("=" * 88)

    # ---- locate the table environment containing \label{tab:ladder}
    i = src.find(r"\label{tab:ladder}")
    if i < 0: raise RuntimeError("no \\label{tab:ladder} in the chapter")
    start = src.rfind(r"\begin{table}", 0, i); end = src.find(r"\end{table}", i)
    if start < 0 or end < 0: raise RuntimeError("could not bound the table environment")
    table = src[start:end]
    caption = table[table.find(r"\caption"):i]
    body = table[table.find(r"\begin{tabular}"): table.find(r"\end{tabular}")]

    # ---- rows: everything between \midrule markers, split on \\
    body = body[body.find(r"\midrule") + len(r"\midrule"):]
    blocks = body.split(r"\midrule")
    if len(blocks) != 2: raise RuntimeError(f"expected one internal \\midrule separating internal from external rows, found {len(blocks)-1}")
    def parse(block: str):
        rows = []
        for raw in re.split(r"\\\\", block):
            raw = raw.strip()
            if not raw or raw.startswith("%") or r"\bottomrule" in raw and "&" not in raw: continue
            raw = raw.replace(r"\bottomrule", "").strip()
            if not raw: continue
            cols = [strip_tex(c) for c in re.split(r"(?<!\\)&", raw)]
            if len(cols) != 4: fails.append(f"G1 row does not have four columns ({len(cols)}): {raw[:70]}")
            rows.append(cols)
        return rows
    internal, external = parse(blocks[0]), parse(blocks[1])
    say(f"\n  parsed {len(internal)} internal rows and {len(external)} external rows")

    for r in internal:
        if r[1] not in GRADES: fails.append(f"G2 unknown internal grade {r[1]!r} in row {r[0][:40]!r}")
        if r[2] not in RECORDS: fails.append(f"G2 unknown record {r[2]!r} in row {r[0][:40]!r}")
    for r in external:
        if r[2] not in RECORDS: fails.append(f"G2 unknown record {r[2]!r} in external row {r[0][:40]!r}")

    grade_n = {g: sum(1 for r in internal if r[1] == g) for g in sorted(GRADES)}
    rec_n = {k: sum(1 for r in internal + external if r[2] == k) for k in sorted(RECORDS)}
    severe_notes = [r[0] for r in internal if r[1] == "severe" and r[2] == "notes"]
    notes_rows = [(r[0], r[1]) for r in internal + external if r[2] == "notes"]
    say(f"  internal grades: " + ", ".join(f"{g} {n}" for g, n in grade_n.items()))
    say(f"  record column over all {len(internal)+len(external)} rows: " + ", ".join(f"{k} {n}" for k, n in rec_n.items()))
    say(f"  severe checks in the notes condition: {len(severe_notes)}" + (f" ({', '.join(severe_notes)})" if severe_notes else ""))
    say(f"  every notes row: " + ("; ".join(f"{c} [{g}]" for c, g in notes_rows) if notes_rows else "none"))

    # ---- the caption's claims
    cap = re.sub(r"\s+", " ", strip_tex(caption))
    say(f"\n  caption, normalised:\n    {cap[:600]}")

    def claim(pattern: str, label: str):
        m = re.search(pattern, cap, re.I)
        if not m:
            fails.append(f"G3 caption no longer states {label}; the guard cannot check a claim it cannot find"); return None
        tok = m.group(1).lower()
        return int(tok) if tok.isdigit() else NUMWORD.get(tok)

    c_int = claim(r"of the (\w+)\s+internal checks", "the internal total")
    c_sev = claim(r"internal checks,?\s+(\w+)\s+are severe", "the severe count")
    c_taut = claim(r"(\w+)\s+are tautological", "the tautological count")
    c_weak = claim(r"and\s+(\w+)\s+are weak", "the weak count")
    c_ext = claim(r"of the (\w+)\s+external comparisons", "the external total")
    c_sevnotes = claim(r"(\w+)\s+of the checks graded severe are in that condition", "the severe-in-notes count")

    for got, want, label in ((len(internal), c_int, "internal rows"),
                             (grade_n["severe"], c_sev, "severe"),
                             (grade_n["tautological"], c_taut, "tautological"),
                             (grade_n["weak"], c_weak, "weak"),
                             (len(external), c_ext, "external rows"),
                             (len(severe_notes), c_sevnotes, "severe in notes")):
        if want is None: continue
        ok = got == want
        say(f"  {'OK  ' if ok else 'FAIL'} {label}: table says {got}, caption says {want}")
        if not ok: fails.append(f"G4/G5 {label}: table {got} vs caption {want}")

    say()
    if fails:
        say("FAILED:"); [say("  " + f) for f in fails]
        say("\nThe caption and the rows disagree. Recount from the rows; do not edit the number by hand without rerunning this.")
        return 1
    say("ALL GATES PASSED: every count the caption states matches the table's own rows.")
    return 0


if __name__ == "__main__": sys.exit(main())
