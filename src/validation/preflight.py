"""
preflight.py
============
Run this FIRST, from anywhere in the repo. It checks that every verification
script in this set can actually run on this machine, and says exactly what is
wrong if not, rather than letting a script die halfway through a 400-point loop.

It checks, in order:

  1. Python and NumPy versions, and whether np.trapezoid exists (NumPy >= 2.0).
     np.trapz was REMOVED in NumPy 2.x and np.trapezoid does not exist before
     it, so a script hardcoding either breaks on one side of that boundary.
  2. That cr_context imports and CRContext.load() succeeds from each script's
     own directory assumption.
  3. That every attribute the scripts read off ctx actually exists.
  4. That the data files exist, with their SHA-256, against the expected
     baseline.
  5. That escape_factor imports (needed only by verify_ch3_groupB).
  6. That each script file is present, parses, and declares no undefined
     all-caps constants.

Report only. Writes nothing, imports no script's main().

    python preflight.py
    python src/validation/preflight.py      # either location works
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import platform
import sys
from pathlib import Path

EXPECT_L_SHA = "2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e"
EXPECT_S_SHA = "7822f536590c76bae8355bddef023fc53e0f3595d2f4687313792e8f277fb80a"

SCRIPTS = [
    ("verify_ch3_claims.py", "validation"),
    ("verify_ch3_groupB.py", "validation"),
    ("verify_eps_gridmap.py", "validation"),
    ("verify_grid_coverage.py", "validation"),
    ("verify_ridge_mechanism.py", "validation"),
    ("make_ch3_figures.py", "analysis"),
]

CTX_ATTRS = ["L_grid", "te_grid", "ne_grid", "root",
             "n_values", "ground_index", "n_states", "state_index_path"]

PASS, FAIL, NOTE = [], [], []


def ok(msg):
    PASS.append(msg); print(f"  [ OK ] {msg}")


def bad(msg, fix=""):
    FAIL.append((msg, fix)); print(f"  [FAIL] {msg}")
    if fix:
        print(f"         fix: {fix}")


def note(msg, fix=""):
    """Advisory: the environment is sound, this file just is not in place."""
    NOTE.append((msg, fix)); print(f"  [NOTE] {msg}")
    if fix:
        print(f"         fix: {fix}")


def collect_bindings(tree: ast.AST) -> set[str]:
    """Every name this module binds.

    An earlier version of this function handled only ast.Assign, so an
    annotated assignment such as `FAILURES: list[str] = []` (AnnAssign) was
    reported as an undefined constant. That was a false positive in this
    checker, not a defect in the script it flagged. The forms below are the
    ones these scripts actually use; anything missed here produces a false
    alarm, which is why the check is advisory rather than blocking.
    """
    bound: set[str] = set()

    def add_target(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                add_target(t)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            add_target(n.target)
        elif isinstance(n, ast.NamedExpr):
            add_target(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            add_target(n.target)
        elif isinstance(n, (ast.comprehension,)):
            add_target(n.target)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            add_target(n.optional_vars)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(n.name)
            a = n.args
            bound.update(x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs))
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
        elif isinstance(n, ast.Lambda):
            a = n.args
            bound.update(x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs))
        elif isinstance(n, ast.ClassDef):
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return bound


def find_repo_root(start: Path) -> Path | None:
    """Walk up looking for the data directory. Do not guess a path."""
    for p in [start, *start.parents]:
        if (p / "data/processed/cr_matrix/L_grid.npy").exists():
            return p
    return None


def main() -> None:
    print("=" * 78)
    print("1. INTERPRETER AND NUMPY")
    print("=" * 78)
    print(f"  python     {sys.version.split()[0]}  ({platform.platform()})")
    print(f"  executable {sys.executable}")
    if sys.version_info < (3, 9):
        bad(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old",
            "these scripts use `X | None` annotations; need 3.9+ with "
            "`from __future__ import annotations`, 3.10+ to be safe")
    else:
        ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    try:
        import numpy as np
    except ImportError:
        bad("numpy not importable", "conda activate cr")
        summarise(); return
    print(f"  numpy      {np.__version__}")
    has_tz, has_tp = hasattr(np, "trapezoid"), hasattr(np, "trapz")
    print(f"  np.trapezoid {has_tz}   np.trapz {has_tp}")
    if not (has_tz or has_tp):
        bad("neither np.trapezoid nor np.trapz exists")
    else:
        ok("an integration routine is available")

    print()
    print("=" * 78)
    print("2. REPO ROOT AND DATA FILES")
    print("=" * 78)
    here = Path(__file__).resolve().parent
    root = find_repo_root(here) or find_repo_root(Path.cwd())
    if root is None:
        bad("could not locate data/processed/cr_matrix/L_grid.npy by walking up "
            f"from {here} or {Path.cwd()}",
            "run from inside the repo, or check the pipeline has produced it")
        summarise(); return
    ok(f"repo root {root}")

    for rel, expect in (("data/processed/cr_matrix/L_grid.npy", EXPECT_L_SHA),
                        ("data/processed/cr_matrix/S_grid.npy", EXPECT_S_SHA)):
        f = root / rel
        if not f.exists():
            bad(f"missing {rel}")
            continue
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if sha == expect:
            ok(f"{rel}  sha256 matches baseline")
        else:
            bad(f"{rel} sha256 {sha[:16]}... does NOT match the baseline "
                f"{expect[:16]}...",
                "every number in Chapter 3 was verified against the baseline "
                "matrix. If the matrix changed deliberately, re-run every "
                "verification script before quoting anything.")

    print()
    print("=" * 78)
    print("3. cr_context AND CRContext")
    print("=" * 78)
    cands = [root / "src/validation", root / "src/analysis", here, Path.cwd()]
    ctx_path = next((p for p in cands if (p / "cr_context.py").exists()), None)
    if ctx_path is None:
        bad("cr_context.py not found in " + ", ".join(str(c) for c in cands))
        summarise(); return
    ok(f"cr_context.py at {ctx_path}")
    sys.path.insert(0, str(ctx_path))
    try:
        from cr_context import CRContext
        ctx = CRContext.load()
        ok("CRContext.load() succeeded")
    except Exception as exc:                      # noqa: BLE001
        bad(f"CRContext.load() raised {type(exc).__name__}: {exc}")
        summarise(); return

    missing = [a for a in CTX_ATTRS if not hasattr(ctx, a)]
    if missing:
        bad(f"CRContext is missing attributes the scripts read: {missing}",
            "these scripts were written against a CRContext exposing "
            f"{CTX_ATTRS}")
    else:
        ok(f"all {len(CTX_ATTRS)} ctx attributes present")
        try:
            print(f"         grid {len(ctx.te_grid)} Te x {len(ctx.ne_grid)} ne"
                  f" = {len(ctx.te_grid)*len(ctx.ne_grid)} points;"
                  f" {ctx.n_states} states; ground index {ctx.ground_index}")
        except Exception as exc:                  # noqa: BLE001
            bad(f"ctx attributes exist but are not usable: {exc}")

    print()
    print("=" * 78)
    print("4. OPTIONAL MODULE: escape_factor (verify_ch3_groupB only)")
    print("=" * 78)
    ef = next((p for p in (root / "src/analysis", root / "src/validation")
               if (p / "escape_factor.py").exists()), None)
    if ef is None:
        bad("escape_factor.py not found",
            "only verify_ch3_groupB needs it; the others are unaffected")
    else:
        sys.path.insert(0, str(ef))
        try:
            import escape_factor                                  # noqa: F401
            from escape_factor import escape_factor_slab, lyman_alpha_sigma0
            ok(f"escape_factor imports from {ef}")
        except Exception as exc:                  # noqa: BLE001
            bad(f"escape_factor present but not importable: {exc}")

    print()
    print("=" * 78)
    print("5. THE SCRIPTS THEMSELVES")
    print("=" * 78)
    for name, subdir in SCRIPTS:
        p = root / "src" / subdir / name
        if not p.exists():
            alt = next((q for q in (root / "src").rglob(name)), None)
            if alt:
                bad(f"{name} expected in src/{subdir}/ but found at "
                    f"{alt.relative_to(root)}",
                    f"move it to src/{subdir}/ -- its sys.path assumes that "
                    f"location")
            else:
                note(f"{name} not present under src/",
                     f"copy it to src/{subdir}/ -- the environment is fine, "
                     f"the file just has not been placed yet")
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as exc:
            bad(f"{name} does not parse: line {exc.lineno}: {exc.msg}")
            continue
        # undefined ALL-CAPS module constants: the NQUAD class of bug
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        bound = collect_bindings(tree)
        undef = sorted(x for x in names - bound
                       if x.isupper() and len(x) > 2 and not hasattr(builtins, x))
        if undef:
            note(f"{name}: ALL-CAPS names apparently unassigned: {undef}",
                 "advisory only -- verify by eye before acting; this checker "
                 "has produced false positives on unusual binding forms")
        else:
            ok(f"{name} parses, no undefined constants "
               f"({p.stat().st_size // 1024} kB)")

    summarise()


def summarise() -> None:
    print()
    print("=" * 78)
    print(f"SUMMARY: {len(PASS)} passed, {len(FAIL)} blocking, "
          f"{len(NOTE)} advisory")
    print("=" * 78)
    if NOTE:
        print("  Advisory (environment is sound; these do not block a run):")
        for msg, _ in NOTE:
            print(f"    - {msg}")
        print()
    if FAIL:
        print("  BLOCKING:")
        for msg, fix in FAIL:
            print(f"    - {msg}")
        print("\n  Do not run the verification scripts until these are cleared.")
    else:
        print("  All checks passed. Run order:")
        print("    1  src/validation/verify_ch3_claims.py")
        print("    2  src/validation/verify_grid_coverage.py")
        print("    3  src/validation/verify_ch3_groupB.py")
        print("    4  src/validation/verify_eps_gridmap.py")
        print("    5  src/validation/verify_ridge_mechanism.py")
        print("    6  src/analysis/make_ch3_figures.py     (writes figures/)")
        print("\n  1-5 are report-only. Only 6 writes anything.")
    print()
    print("  Report only. Nothing was written.")


if __name__ == "__main__":
    main()