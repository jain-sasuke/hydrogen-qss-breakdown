"""
audit_writers.py
================
Protocol D.4: one writer per output path.

`qss_analysis.py` and `validate_gates.py` both wrote `M_grid.npy`,
`tau_QSS_grid.npy` and `tau_relax_grid.npy`, and both once carried the
`eigs < -1.0` filter. Whichever ran last owned the files. The protocol records
a decision as REQUIRED and this script checks whether it was implemented --
mechanically, over every .py in the repo, rather than from memory.

It also checks a second thing worth having: every script whose docstring claims
"Report only" is verified to contain no write call. A claim of read-only that
nobody checks is worth nothing.

WHAT THIS CAN AND CANNOT DO
---------------------------
It is a STATIC screen, not a proof. It finds write calls by AST and resolves
the destination when the argument is a literal, an f-string, or a simple
`root / "a/b/c"` expression. Paths assembled through variables, loops or helper
functions are reported as UNRESOLVED with their source line, because reporting
them as absent would be worse than reporting them as unknown.

So: a clean report means no multi-writer path was FOUND. It does not prove none
exists. Read the UNRESOLVED list.

Report only. Writes nothing.

    python src/validation/audit_writers.py
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

# call patterns that put bytes on disk, and which argument holds the path
WRITE_CALLS = {
    "save": 0, "savez": 0, "savez_compressed": 0, "savetxt": 0,      # numpy
    "to_csv": 0, "to_parquet": 0, "to_excel": 0, "to_hdf": 0,        # pandas
    "savefig": 0,                                                    # matplotlib
    "write_text": None, "write_bytes": None,                         # Path methods
    "dump": 1,                                                       # json/pickle
    "copy": 1, "copy2": 1, "copyfile": 1, "move": 1,                 # shutil
    "mkdir": None, "makedirs": None,                                 # dirs, noted
    "open": 0,                                                       # only if mode is w/a
}
BENIGN = {"mkdir", "makedirs"}

D4_PATHS = ("M_grid.npy", "tau_QSS_grid.npy", "tau_relax_grid.npy")


def literal_path(node: ast.AST) -> str | None:
    """Best-effort static resolution of a path expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):                       # f-string
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            else:
                out.append("{...}")
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = literal_path(node.left)
        right = literal_path(node.right)
        if right is None:
            return None
        return f"{left or '<expr>'}/{right}"
    if isinstance(node, ast.Call):                            # Path("...")
        for a in node.args:
            r = literal_path(a)
            if r:
                return r
    if isinstance(node, ast.Attribute):
        return None
    return None


def scan(path: Path) -> tuple[list[tuple[str, str, int]], list[tuple[str, int]], bool]:
    """Return (resolved writes, unresolved writes, claims_report_only)."""
    src = path.read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], [], False
    doc = (ast.get_docstring(tree) or "").lower()
    # "Report only" alone is not a read-only claim: make_ch3_figures.py says
    # "Report only: writes to figures/, modifies nothing else", which discloses
    # its writes. Only an UNQUALIFIED claim counts -- either an explicit
    # "writes nothing", or "report only" with no mention of writing at all.
    claims = ("writes nothing" in doc
              or "nothing was written" in doc
              or ("report only" in doc and "write" not in doc))

    resolved: list[tuple[str, str, int]] = []
    unresolved: list[tuple[str, int]] = []
    lines = src.splitlines()

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None)
        if name not in WRITE_CALLS or name in BENIGN:
            continue

        if name == "open":                        # only a write mode counts
            mode = ""
            if len(n.args) > 1:
                mode = literal_path(n.args[1]) or ""
            for kw in n.keywords:
                if kw.arg == "mode":
                    mode = literal_path(kw.value) or ""
            if not any(c in mode for c in "wax"):
                continue

        idx = WRITE_CALLS[name]
        target = None
        if idx is None:                           # p.write_text(...) -> p is fn.value
            target = literal_path(fn.value) if isinstance(fn, ast.Attribute) else None
        elif len(n.args) > idx:
            target = literal_path(n.args[idx])

        line = n.lineno
        if target:
            resolved.append((target, name, line))
        else:
            snippet = lines[line - 1].strip() if line - 1 < len(lines) else ""
            unresolved.append((f"{name}(): {snippet[:80]}", line))
    return resolved, unresolved, claims


def main() -> None:
    here = Path(__file__).resolve().parent
    root = next((p for p in [here, *here.parents]
                 if (p / "data/processed/cr_matrix/L_grid.npy").exists()), None)
    if root is None:
        print("could not locate the repo root; run from inside the repo")
        sys.exit(1)

    files = sorted((root / "src").rglob("*.py"))
    files = [f for f in files if "__pycache__" not in f.parts]
    print("=" * 78)
    print(f"AUDIT OF OUTPUT WRITERS   ({len(files)} files under {root / 'src'})")
    print("=" * 78)

    by_target: dict[str, list[str]] = defaultdict(list)
    by_basename: dict[str, set[str]] = defaultdict(set)
    readers: dict[str, list[str]] = defaultdict(list)
    unresolved_all: list[tuple[str, str, int]] = []
    liars: list[str] = []

    for f in files:
        rel = str(f.relative_to(root))
        res, unres, claims = scan(f)
        for target, _call, _ln in res:
            by_target[target].append(rel)
            by_basename[Path(target).name].add(target)
        for snippet, ln in unres:
            unresolved_all.append((rel, snippet, ln))
        if claims and (res or unres):
            liars.append(rel)

    print()
    print("-" * 78)
    print("1. PATHS WITH MORE THAN ONE WRITER   (protocol D.4)")
    print("-" * 78)
    multi = {t: sorted(set(w)) for t, w in by_target.items()
             if len(set(w)) > 1 and "{...}" not in t}
    vacuous = {t: sorted(set(w)) for t, w in by_target.items()
               if len(set(w)) > 1 and "{...}" in t}
    if not multi:
        print("  none found among statically resolved paths")
    for t, ws in sorted(multi.items()):
        flag = "  <<< D.4" if Path(t).name in D4_PATHS else ""
        print(f"  {t}{flag}")
        for w in ws:
            print(f"      {w}")
    if vacuous:
        print()
        print("  (excluded: these 'paths' contain unresolved f-string segments,")
        print("   so a shared name proves nothing. See section 4.)")
        for t in sorted(vacuous):
            print(f"    {t}  <- {len(vacuous[t])} scripts")

    print()
    print("-" * 78)
    print("2. THE THREE PATHS D.4 NAMES EXPLICITLY")
    print("-" * 78)
    for t in D4_PATHS:
        ws = sorted({w for tgt, wl in by_target.items()
                     if Path(tgt).name == t for w in wl})
        if not ws:
            print(f"  {t:22s} no resolved writer found "
                  f"(may be written through an unresolved expression)")
        elif len(ws) == 1:
            print(f"  {t:22s} single writer: {ws[0]}   OK")
        else:
            print(f"  {t:22s} {len(ws)} WRITERS -- D.4 NOT IMPLEMENTED")
            for w in ws:
                print(f"      {w}")

    print()
    print("-" * 78)
    print("2b. SAME FILENAME, DIFFERENT DIRECTORIES")
    print("    Not a D.4 violation, but D.5 was burned by exactly this:")
    print("    S_grid was read as the ionisation array when it is the")
    print("    recombination source.")
    print("-" * 78)
    collisions = {b: sorted(ts) for b, ts in by_basename.items() if len(ts) > 1}
    if not collisions:
        print("  none")
    for b, ts in sorted(collisions.items()):
        print(f"  {b}")
        for t in ts:
            print(f"      {t}")
            for w in sorted(set(by_target.get(t, []))):
                print(f"          written by {w}")

    print()
    print("-" * 78)
    print("5. WHO READS THE MULTI-WRITER PATHS?")
    print("    A path with two writers only matters if something consumes it.")
    print("-" * 78)
    watch = set(D4_PATHS) | {Path(t).name for t in multi}
    for f in files:
        try:
            txt = f.read_text(errors="replace")
        except OSError:
            continue
        for w in watch:
            if w in txt:
                rel_f = str(f.relative_to(root))
                is_writer = rel_f in {
                    x for t, wl in by_target.items()
                    if Path(t).name == w for x in wl}
                # this auditor names every watched path in its own source, so
                # exclude it or it reports itself as a consumer of everything
                if not is_writer and f.name != Path(__file__).name:
                    readers[w].append(rel_f)
    if not readers:
        print("  no consumers found -- the ambiguity is inert")
    for w in sorted(readers):
        print(f"  {w}")
        for r in sorted(set(readers[w])):
            print(f"      read/mentioned by {r}")

    print()
    print("-" * 78)
    print("3. SCRIPTS CLAIMING 'Report only' THAT CONTAIN WRITE CALLS")
    print("-" * 78)
    if not liars:
        print("  none -- every read-only claim checks out")
    for f in liars:
        print(f"  {f}")
        for rel, snip, ln in unresolved_all:
            if rel == f:
                print(f"      line {ln}: {snip}")

    print()
    print("-" * 78)
    print("4. UNRESOLVED WRITE DESTINATIONS   (read these; the screen is static)")
    print("-" * 78)
    if not unresolved_all:
        print("  none")
    cur = None
    for rel, snip, ln in unresolved_all:
        if rel != cur:
            print(f"  {rel}")
            cur = rel
        print(f"      line {ln:5d}  {snip}")

    print()
    print("=" * 78)
    n_multi = len(multi)
    print(f"SUMMARY: {len(by_target)} resolved output paths, "
          f"{n_multi} with multiple writers, "
          f"{len(unresolved_all)} unresolved write calls, "
          f"{len(liars)} false read-only claims")
    print("=" * 78)
    print("  A clean section 1 means no multi-writer path was FOUND. It does not")
    print("  prove none exists -- section 4 is the part that needs human eyes.")
    print()
    print("  Report only. Nothing was written.")


if __name__ == "__main__":
    main()